#!/usr/bin/env bash
# Integration tests for scripts/worktree.sh.
#
#   make test-worktree
#
# Deliberately NOT part of `make test`: this drives git end to end (clones, branches,
# pushes to a throwaway local remote) and is slower than the unit tests.
#
# Safety: every test runs against a fresh clone in a temp directory with its own bare
# remote. Nothing here touches the real repository, its branches, or GitHub.
#
# `make` is shimmed onto PATH so `make setup` returns instantly instead of building a
# 1.4 GB venv. Set FAKE_MAKE_FAIL=1 to make the shim fail, which exercises rollback.
# The venv-isolation property this stubs out is a property of uv + editable installs,
# not of this script; it is verified separately and documented in AGENTS.md.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
PASS=0; FAIL=0
trap 'rm -rf "$TMP"' EXIT

# --- shim `make` so `make setup` is instant -----------------------------------------
mkdir -p "$TMP/bin"
cat > "$TMP/bin/make" <<'SHIM'
#!/usr/bin/env bash
[ -n "${FAKE_MAKE_FAIL:-}" ] && { echo "fake make: failing on purpose" >&2; exit 1; }
exit 0
SHIM
chmod +x "$TMP/bin/make"
export PATH="$TMP/bin:$PATH"
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com

# --- tiny assertion helpers ----------------------------------------------------------
ok()   { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; [ -n "${2:-}" ] && printf '       %s\n' "$2"; }
is()   { [ "$2" = "$3" ] && ok "$1" || bad "$1" "expected '$3', got '$2'"; }
yes_() { [ "$2" = 0 ] && ok "$1" || bad "$1" "command failed (exit $2)"; }
no_()  { [ "$2" != 0 ] && ok "$1" || bad "$1" "command unexpectedly succeeded"; }

# --- fixture: fresh remote + clone per test ------------------------------------------
N=0
fixture() {
  N=$((N+1)); FX="$TMP/fx$N"
  mkdir -p "$FX"; ( cd "$FX"
    git init -q --bare -b main remote.git   # -b: do not depend on init.defaultBranch (F4)
    git clone -q "$REPO" primary 2>/dev/null
    cd primary
    git remote set-url origin "$FX/remote.git"
    mkdir -p scripts docs/project
    cp "$REPO/scripts/worktree.sh" scripts/
    cp "$REPO/docs/project/STATUS.md" docs/project/
    chmod +x scripts/worktree.sh
    git add -A >/dev/null; git commit -qm "fixture baseline"
    git push -q -u origin main
  ) >/dev/null 2>&1
  P="$FX/primary"
}
wt()        { ( cd "$P" && scripts/worktree.sh "$@" ); }
claims()    { sed -n '/^## In progress/,/^## Paused/p' "$P/docs/project/STATUS.md" | grep -c "$1" || true; }
on_remote() { ( cd "$P" && git show origin/main:docs/project/STATUS.md ) | grep -c "$1" || true; }

echo "worktree.sh integration tests"

# ------------------------------------------------------------------ new: happy path --
echo; echo "new — happy path"
fixture
out=$(wt new claude/alpha 2>&1); rc=$?
yes_ "new succeeds"                        "$rc"
is   "worktree directory created"          "$([ -d "$FX/temporallens-claude-alpha" ] && echo y || echo n)" "y"
is   "branch exists"                       "$( (cd "$P" && git show-ref --verify --quiet refs/heads/claude/alpha) && echo y || echo n)" "y"
is   "branch is cut from main"             "$( (cd "$P" && git merge-base --is-ancestor claude/alpha main) && echo y || echo n)" "y"
is   "claim row present locally"           "$(claims 'claude/alpha')" "1"
is   "claim is PUSHED (visible on remote)" "$(on_remote 'claude/alpha')" "1"

# --------------------------------------------------------------- new: input guards --
echo; echo "new — guards"
fixture
out=$(wt new "bad//name" 2>&1); rc=$?
no_ "rejects malformed branch name" "$rc"
is  "no claim leaked on rejection"  "$(claims 'bad')" "0"

out=$(wt new nonslash 2>&1); rc=$?
no_ "rejects branch without <owner>/<task>" "$rc"

wt new claude/dup >/dev/null 2>&1
out=$(wt new claude/dup 2>&1); rc=$?
no_ "refuses duplicate branch" "$rc"

# ---------------------------------------------------- new: exact-match claim rows ----
echo; echo "new — claim rows are matched exactly, not by substring"
fixture
wt new claude/foo-bar >/dev/null 2>&1
wt new claude/foo     >/dev/null 2>&1
is "both claims coexist" "$(claims 'claude/foo')" "2"
( cd "$FX/temporallens-claude-foo" && git commit -q --allow-empty -m w )
( cd "$P" && git merge -q --no-edit claude/foo )
wt done claude/foo >/dev/null 2>&1
is "removing claude/foo leaves claude/foo-bar" "$(claims 'claude/foo-bar')" "1"
is "claude/foo itself is gone"                 "$(claims 'claude/foo`')" "0"

# ------------------------------------------------------------- new: rollback paths --
echo; echo "new — rollback leaves nothing behind"
fixture
out=$(FAKE_MAKE_FAIL=1 wt new claude/rollback 2>&1); rc=$?
no_ "aborts when setup fails"          "$rc"
is  "worktree removed"                 "$([ -d "$FX/temporallens-claude-rollback" ] && echo y || echo n)" "n"
is  "branch deleted"                   "$( (cd "$P" && git show-ref --verify --quiet refs/heads/claude/rollback) && echo y || echo n)" "n"
is  "claim released locally"           "$(claims 'claude/rollback')" "0"
is  "claim released on remote"         "$(on_remote 'claude/rollback')" "0"

echo; echo "new — unreachable remote is fatal, not a warning"
fixture
( cd "$P" && git remote set-url origin "$FX/nope.git" )
out=$(wt new claude/netfail 2>&1); rc=$?
no_ "aborts when the remote is unreachable" "$rc"
( cd "$P" && git remote set-url origin "$FX/remote.git" )
is "main not left ahead of origin (no invisible claim)" \
   "$( (cd "$P" && git rev-list --count origin/main..main) )" "0"

# Distinct from the above: the remote is REACHABLE (fetch succeeds) but REJECTS the
# push. Without this case a non-fatal push would go undetected, because the
# unreachable-remote test aborts at fetch and never reaches the push.
echo; echo "new — a rejected push is fatal and reverts the claim"
fixture
cat > "$FX/remote.git/hooks/pre-receive" <<'HOOK'
#!/bin/sh
echo "remote: pushes rejected (test)" >&2
exit 1
HOOK
chmod +x "$FX/remote.git/hooks/pre-receive"
out=$(wt new claude/pushfail 2>&1); rc=$?
no_ "aborts when the claim push is rejected" "$rc"
is  "claim commit reverted, main == origin/main" \
    "$( (cd "$P" && git rev-list --count origin/main..main) )" "0"
is  "no local claim row left behind" "$(claims 'claude/pushfail')" "0"
is  "no worktree created"            "$([ -d "$FX/temporallens-claude-pushfail" ] && echo y || echo n)" "n"
rm -f "$FX/remote.git/hooks/pre-receive"

# ------------------------------------------------------- new: primary-checkout state -
echo; echo "new — primary checkout preconditions"
fixture
( cd "$P" && git switch -qc some/other )
out=$(wt new claude/beta 2>&1); rc=$?
no_ "refuses when primary is not on main" "$rc"
( cd "$P" && git switch -q main && echo dirt > dirty.txt )
out=$(wt new claude/beta 2>&1); rc=$?
no_ "refuses when primary is dirty" "$rc"
rm -f "$P/dirty.txt"

fixture
( cd "$FX" && git clone -q remote.git other >/dev/null 2>&1
  cd other && git commit -q --allow-empty -m "someone else's work" && git push -q origin main )
out=$(wt new claude/behind 2>&1); rc=$?
no_ "refuses when main is behind origin" "$rc"

# ------------------------------------------- new: branch-name charset (F2) -----------
echo; echo "new — rejects names that would corrupt the STATUS table or collide"
fixture
# 'claude-foo/bar' would map to the same directory as 'claude/foo-bar'; banning the
# hyphen in the OWNER makes branch <-> directory a bijection.
for badname in 'claude/foo|bar' 'claude/foo`bar' 'claude/foo/bar' 'Claude/Upper' 'claude/' '/task' 'claude-foo/bar'; do
  out=$(wt new "$badname" 2>&1); rc=$?
  no_ "rejects '$badname'" "$rc"
done
is "no claim rows leaked by any rejection" "$(claims 'claude')" "0"
out=$(wt new codex/loader_v2 2>&1); rc=$?
yes_ "accepts a legitimate name with underscore/digits" "$rc"
out=$(wt new claude/foo-bar 2>&1); rc=$?
yes_ "accepts a hyphenated TASK (only the owner is restricted)" "$rc"

# ------------------------------------ new: local commit failure is atomic (F1) -------
echo; echo "new — a failing pre-commit hook leaves STATUS untouched"
fixture
cat > "$P/.git/hooks/pre-commit" <<'HOOK'
#!/bin/sh
echo "pre-commit: rejecting (test)" >&2
exit 1
HOOK
chmod +x "$P/.git/hooks/pre-commit"
before=$( (cd "$P" && git rev-parse HEAD) )
out=$(wt new claude/hookfail 2>&1); rc=$?
no_ "aborts when the claim commit is rejected" "$rc"
is  "HEAD unchanged"                "$( (cd "$P" && git rev-parse HEAD) )" "$before"
is  "STATUS not left modified"      "$( (cd "$P" && git status --porcelain docs/project/STATUS.md | wc -l | tr -d ' ') )" "0"
is  "no claim row left behind"      "$(claims 'claude/hookfail')" "0"
is  "no worktree created"           "$([ -d "$FX/temporallens-claude-hookfail" ] && echo y || echo n)" "n"
rm -f "$P/.git/hooks/pre-commit"

# ----------------------------------------------------------------------- lock --------
echo; echo "lock"
fixture
mkdir -p "$P/.git/worktree-helper.lock"
out=$(wt new claude/locked 2>&1); rc=$?
no_ "refuses while another run holds the lock" "$rc"
rmdir "$P/.git/worktree-helper.lock"

# ----------------------------------------------------------------------- done --------
echo; echo "done — refuses to release unfinished work"
fixture
wt new claude/wip >/dev/null 2>&1
( cd "$FX/temporallens-claude-wip" && git commit -q --allow-empty -m "unmerged work" )
out=$(wt done claude/wip 2>&1); rc=$?
no_ "refuses when the branch is unmerged"  "$rc"
is  "worktree kept"                        "$([ -d "$FX/temporallens-claude-wip" ] && echo y || echo n)" "y"
is  "branch kept"                          "$( (cd "$P" && git show-ref --verify --quiet refs/heads/claude/wip) && echo y || echo n)" "y"
is  "claim still held"                     "$(claims 'claude/wip')" "1"

echo "dirt" > "$FX/temporallens-claude-wip/scratch.txt"
out=$(wt done claude/wip 2>&1); rc=$?
no_ "refuses when the worktree is dirty" "$rc"
rm -f "$FX/temporallens-claude-wip/scratch.txt"

echo; echo "done — happy path"
fixture
wt new claude/finish >/dev/null 2>&1
( cd "$FX/temporallens-claude-finish" && git commit -q --allow-empty -m "work" )
( cd "$P" && git merge -q --no-edit claude/finish )
out=$(wt done claude/finish 2>&1); rc=$?
yes_ "done succeeds after merge"    "$rc"
is   "worktree removed"             "$([ -d "$FX/temporallens-claude-finish" ] && echo y || echo n)" "n"
is   "merged branch deleted"        "$( (cd "$P" && git show-ref --verify --quiet refs/heads/claude/finish) && echo y || echo n)" "n"
is   "claim released locally"       "$(claims 'claude/finish')" "0"
is   "claim released on remote"     "$(on_remote 'claude/finish')" "0"
is   "placeholder row restored"     "$(claims '_(none')" "1"

out=$(wt done claude/never 2>&1); rc=$?
no_ "refuses for a nonexistent worktree" "$rc"

# --------------------------------------------------------------------- summary -------
echo
if [ "$FAIL" -eq 0 ]; then
  printf '\033[32m%d passed, 0 failed\033[0m\n' "$PASS"; exit 0
else
  printf '\033[31m%d passed, %d FAILED\033[0m\n' "$PASS" "$FAIL"; exit 1
fi
