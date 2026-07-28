#!/usr/bin/env bash
# Manage per-agent git worktrees for parallel work.
#
#   scripts/worktree.sh new  <owner>/<task>   # claim on main, then worktree + branch + venv
#   scripts/worktree.sh list                  # show worktrees and their branches
#   scripts/worktree.sh done <owner>/<task>   # release claim, remove worktree, delete branch
#
# Example:  scripts/worktree.sh new claude/f1-baseline
#
# What this enforces, each of which fails silently if done by hand:
#
#  1. The STATUS claim is committed AND PUSHED to `main` before the branch exists.
#     A claim on a feature branch — or an unpushed one — is invisible to every other
#     agent, which defeats claiming entirely. Remote failures are therefore fatal.
#  2. The branch is cut from an up-to-date `main`, explicitly. `git worktree add -b`
#     with no start point silently branches from the CALLER's HEAD.
#  3. Every worktree gets its own venv (`temporallens` is installed editable, so a
#     shared venv imports and tests the WRONG source tree).
#  4. Every mutating command is all-or-nothing: `new` rolls back a partial creation,
#     `done` verifies everything before touching anything.
#  5. Only one agent mutates the primary checkout at a time (lock).
#
# Set WORKTREE_ALLOW_OFFLINE=1 only for a repository with no remote; it downgrades
# the mandatory fetch/push to local-only and forfeits cross-agent coordination.
set -euo pipefail

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
die() { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*"; }

BASE=main
OFFLINE=${WORKTREE_ALLOW_OFFLINE:-0}
git rev-parse --show-toplevel >/dev/null 2>&1 || die "not inside the temporallens repository"

PRIMARY=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
PARENT=$(dirname "$PRIMARY")
dir_for() { echo "$PARENT/temporallens-${1//\//-}"; }
gp() { git -C "$PRIMARY" "$@"; }

# ---- lock: one mutating transaction against the primary checkout at a time -------
LOCK="$PRIMARY/.git/worktree-helper.lock"
lock() {
  mkdir "$LOCK" 2>/dev/null || die "another worktree.sh is running (lock: $LOCK). If stale: rmdir '$LOCK'"
  # shellcheck disable=SC2064
  trap "rmdir '$LOCK' 2>/dev/null || true" EXIT
}

# ---- STATUS "In progress" table --------------------------------------------------
# Rows are matched on the EXACT branch cell, never a substring: `claude/foo` must not
# match `claude/foo-bar`.
status_row() { # $1=add|remove  $2=<owner>/<task>
  ACTION=$1 BRANCH=$2 python3 - "$PRIMARY/docs/project/STATUS.md" <<'PY'
import os, re, sys
path, action, branch = sys.argv[1], os.environ["ACTION"], os.environ["BRANCH"]
owner, task = branch.split("/", 1)
row = f"| {task} | {owner} | `{branch}` |"
lines = open(path).read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("## In progress"))
end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
sep = next(i for i in range(start, end) if re.match(r"^\|[-| ]+\|$", lines[i]))

def branch_of(line):                      # exact 3rd cell, backticks stripped
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells[2].strip("`") if len(cells) >= 3 else None

body = [i for i in range(sep + 1, end) if lines[i].lstrip().startswith("|")]
if action == "add":
    if any(branch_of(lines[i]) == branch for i in body):
        sys.exit(f"claim for {branch} is already present")
    for i in reversed(body):
        if "_(none" in lines[i]:
            del lines[i]
    lines.insert(sep + 1, row)
else:
    hits = [i for i in body if branch_of(lines[i]) == branch]
    if not hits:
        sys.exit(f"no claim row for {branch}")
    for i in reversed(hits):
        del lines[i]
    if not [i for i in range(sep + 1, end) if lines[i].lstrip().startswith("|")]:
        lines.insert(sep + 1, "| _(none — claim work here before starting)_ | — | — |")
open(path, "w").write("\n".join(lines) + "\n")
PY
}

# ---- preconditions ---------------------------------------------------------------
has_remote() { gp remote get-url origin >/dev/null 2>&1; }

require_clean_primary_on_base() {
  [ -z "$(gp status --porcelain)" ] || die "primary checkout $PRIMARY has uncommitted changes"
  local head; head=$(gp rev-parse --abbrev-ref HEAD)
  [ "$head" = "$BASE" ] || die "primary checkout must be on '$BASE' (it is on $head)"
  return 0
}

require_base_current() {
  if ! has_remote; then
    [ "$OFFLINE" = 1 ] || die "no 'origin' remote. Claims cannot be shared; set WORKTREE_ALLOW_OFFLINE=1 to proceed anyway"
    return 0
  fi
  note "fetching to confirm $BASE is current"
  if ! gp fetch --quiet origin "$BASE"; then
    [ "$OFFLINE" = 1 ] || die "fetch failed. A claim made now could collide with another agent's; fix connectivity or set WORKTREE_ALLOW_OFFLINE=1"
    return 0
  fi
  local behind; behind=$(gp rev-list --count "$BASE..origin/$BASE")
  [ "$behind" -eq 0 ] || die "$BASE is $behind commit(s) behind origin/$BASE — run: git -C $PRIMARY pull"
  return 0
}

# Commit a STATUS change on main and push it. Push failure is FATAL and undoes the
# commit, so `main` is never left with an unpushed (invisible) claim.
commit_status() { # $1=message
  # All-or-nothing: a failing hook, signing error, or bad git config must not leave
  # STATUS modified or staged in the primary checkout.
  local before; before=$(gp rev-parse HEAD)
  if ! gp add docs/project/STATUS.md || ! gp commit --quiet -m "$1"; then
    gp reset --hard --quiet "$before"
    die "could not commit '$1' (hook, signing, or git config?) — STATUS restored, nothing changed"
  fi
  if has_remote && [ "$OFFLINE" != 1 ]; then
    gp push --quiet origin "$BASE" || {
      gp reset --hard --quiet "$before"
      die "push of '$1' failed — reverted. The claim must be visible to other agents; fix connectivity and retry"
    }
  fi
  return 0
}

cmd=${1:-}; branch=${2:-}

case "$cmd" in
  new)
    [ -n "$branch" ] || usage
    # Deliberately narrower than git allows. `git check-ref-format` accepts names that
    # break things here: a pipe or backtick corrupts the STATUS markdown table (extra
    # cells), and extra slashes collide in the directory name.
    #
    # The OWNER may not contain '-', the task may. That single restriction makes
    # <owner>/<task> <-> temporallens-<owner>-<task> a bijection: the first hyphen after
    # the prefix always ends the owner. Without it, claude/foo-bar and claude-foo/bar
    # both map to temporallens-claude-foo-bar.
    printf '%s' "$branch" | grep -Eq '^[a-z0-9][a-z0-9._]*/[a-z0-9][a-z0-9._-]*$' \
      || die "branch must be <owner>/<task>: owner [a-z0-9._] (no hyphen), task [a-z0-9._-], exactly one slash (got '$branch')"
    git check-ref-format --branch "$branch" >/dev/null 2>&1 \
      || die "'$branch' is not a valid git branch name"
    dir=$(dir_for "$branch")
    [ -e "$dir" ] && die "$dir already exists"
    git show-ref --verify --quiet "refs/heads/$branch" && die "branch $branch already exists"

    lock
    require_clean_primary_on_base
    require_base_current

    note "claiming '$branch' in STATUS on $BASE"
    status_row add "$branch"
    commit_status "Claim $branch in STATUS"

    # From here the claim exists remotely: every later failure must release it.
    rollback() {
      echo "==> rolling back" >&2
      git worktree remove --force "$dir" 2>/dev/null || true
      git branch -D "$branch" 2>/dev/null || true
      status_row remove "$branch" 2>/dev/null || true
      gp add docs/project/STATUS.md 2>/dev/null || true
      gp commit --quiet -m "Release $branch claim (creation failed)" 2>/dev/null || true
      if has_remote && [ "$OFFLINE" != 1 ]; then
        gp push --quiet origin "$BASE" 2>/dev/null \
          || echo "    WARNING: could not push the release; run: git -C $PRIMARY push" >&2
      fi
    }

    note "creating worktree $dir from $BASE"
    git worktree add --quiet "$dir" -b "$branch" "$BASE" || { rollback; die "git worktree add failed"; }

    note "bootstrapping its own venv (required — see header)"
    make -C "$dir" setup || { rollback; die "make setup failed in $dir"; }

    echo
    echo "Worktree ready: $dir"
    echo "Claim is committed and pushed on $BASE — other agents can see it."
    echo "Next:  cd $dir"
    ;;

  list) git worktree list ;;

  done)
    [ -n "$branch" ] || usage
    dir=$(dir_for "$branch")

    lock
    # PREFLIGHT — verify everything before mutating anything. An unmerged branch must
    # keep BOTH its worktree and its claim, or unfinished work becomes invisible.
    [ -d "$dir" ] || die "no worktree at $dir"
    [ -z "$(git -C "$dir" status --porcelain)" ] || {
      git -C "$dir" status --short >&2
      die "$dir has uncommitted changes — commit or discard them first"; }
    require_clean_primary_on_base
    require_base_current
    gp merge-base --is-ancestor "$branch" "$BASE" \
      || die "'$branch' is not merged into $BASE. Merge it first (see AGENTS.md step 6); worktree and claim are untouched"

    note "releasing the claim in STATUS on $BASE"
    status_row remove "$branch"
    commit_status "Release $branch claim in STATUS"

    note "removing worktree $dir"
    git worktree remove "$dir"
    git branch -d "$branch" >/dev/null && note "deleted merged branch $branch"
    ;;

  *) usage ;;
esac
