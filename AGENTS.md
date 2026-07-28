# AGENTS.md — operating rules for agents in this repository

Canonical instructions for any agent (Codex, Claude Code, or a human using either) working in
TemporalLens. Codex reads this file natively; Claude Code reads it via the `@AGENTS.md` import in
`CLAUDE.md`. Keep the shared rules here, not duplicated per-agent.

> **Start every session at [`docs/project/STATUS.md`](docs/project/STATUS.md)** — the single source
> of truth for repo state, latest changes, and the next task to pick up. Read it before starting and
> update it in the same change that moves the work (see Definition of Done).

## Workspace root and session handshake — check this first

The canonical primary checkout on this machine is:

```
/Users/denisamcdonald/Projects/TemporalLens/temporallens
```

A valid agent workspace may be either that primary checkout or a linked Git worktree belonging to
this repository. Worktree, collaborator, and cloud-clone paths will differ, so repository identity
comes from Git rather than from matching the literal path above.

- **Not** the parent `…/TemporalLens/` (that is not a git repo; it holds a separate
  `planning-documents/` repo as a sibling).
- **Not** any ChatGPT-project mirror under `~/.codex/.chatgpt-projects/…` (read-only, empty sources).

Before planning or editing, every agent must:

1. Run `git rev-parse --show-toplevel`, `git status --short --branch`, and
   `git rev-parse --short HEAD`.
2. Read `docs/project/STATUS.md`.
3. Report the root, branch/HEAD, current changes, and next task.

The reported root must be the primary checkout or a linked worktree shown by `git worktree list`.
If it is not, stop and re-root before writing anything.

## Repository map

```
configs/experiment/{foundation,generation,language}/  # one YAML per ablation, 1:1 with the spec
configs/experiment/robustness_targets.yaml            # shared registry of robustness targets (D2)
docs/experiments/                                     # AUTHORITY on what each ablation measures
docs/project/{STATUS,DECISIONS}.md                    # live state + decision provenance
src/temporallens/                                     # package (data, models, training, eval, …)
scripts/                                              # verify_environment, evaluate (+ pending)
tests/                                                # environment + unit tests
```

## Commands

Working now:

```bash
make setup      # uv sync --python 3.11 --extra dev  (idempotent)
make verify     # env + MPS check
make test       # pytest
make lint       # ruff + mypy
make format     # black + ruff --fix
make test-worktree  # integration tests for scripts/worktree.sh (slower; run when it changes)
```

Interface-only — the scripts they call are not written yet (`train_encoder.py`, `train_adapter.py`,
`make_report.py`; `evaluate.py` is a stub): `debug`, `train-baseline`, `debug-adapter`, `smoke-1b`,
`train-adapter-cloud`, `eval-noise`, `eval-robustness`, `report`. Do not present these as functional.

## Authority and vocabulary

- `docs/experiments/README.md` is the authority on protocol; if a config and the spec disagree, the
  spec is right until someone fixes it. Every ablation maps to exactly one config file — preserve
  that 1:1 mapping.
- The vocabulary in `docs/experiments/README.md` §2 is **binding**: encoder / projector / adapter /
  head / soft prefix / frozen / readout / input path. Use those exact terms in code, configs,
  commits, and prose. This discipline is why the spec exists.

## Hard constraints

- **Never fabricate NinaPro data.** It is downloaded under its own terms into `data/raw/` (gitignored).
- **Leakage rules** in `docs/experiments/generative-arm.md` are non-negotiable; a result traceable to
  a violation is treated as no result.
- **Never commit** data, checkpoints, secrets, or run outputs — `.gitignore` already excludes them.
- `frozen` means `requires_grad=False` and `eval()` mode. The encoder and language model are frozen;
  the projector and head are not.
- **Checkpoint contract:** every training checkpoint saves `{model_state, model_config}` so any
  consumer can rebuild the model from the checkpoint alone (no external `model_type`).

## Working agreement — STATUS and DECISIONS

These two files are the project's memory. Every agent uses them the same way, every session.

### Scope — which file gets what

| | `docs/project/STATUS.md` | `docs/project/DECISIONS.md` |
|---|---|---|
| Holds | **Work** — what is done, in flight, paused, queued | **Choices** — what was decided, by whom, when |
| Mutability | **Mutable.** Items enter, move, and leave | **Append-only.** Rows are never deleted; status changes to Superseded/Rejected |
| Answers | "What do I pick up next?" | "Why is it this way, and who decided?" |

**The test:** does this *change what we build*, or *record that we are building it*? Changing what
we build (an option was chosen, a scope call was made, an approach was rejected) → DECISIONS, plus a
STATUS item if it creates work. Recording progress on something already agreed → STATUS only.

Never record the same thing as fact in both. STATUS may *reference* a decision ID (D1, D2, …); it
must not restate its rationale — that lives in DECISIONS and `docs/experiments/`.

**Never hand-copy machine-derivable state into these files.** Working-tree status, staged/unstaged
inventories, branch lists, file listings, installed versions — anything a command answers in a
second — is stale the moment it is written and contradicts its own source. Record *decisions* and
*progress*; let `git status`, `git log`, and `git worktree list` report state. Concretely:
**Latest changes** carries shipped work in prose and **no commit hashes** — `git log --oneline` is
the live record, and a copied hash is bookkeeping that goes stale. Anything in flight goes to
**In progress** or **Paused / mid-flight**, never into the changelog.

### Who may update them — two tiers

**Tier A — the agent updates the file itself, then states plainly in the conversation what it
changed.** No approval needed. This is factual bookkeeping:

- Recording work that is finished **and** passed `make verify && make test && make lint`.
- Claiming or releasing an item in **In progress** (task · owner · branch).
- Moving a queued item to in-flight; adding a raised-but-not-started item to **Next up**.
- Adding a line to **Latest changes**; correcting something that is simply wrong (a stale hash).

**Tier B — the agent proposes in the conversation and waits for the owner's approval before
writing.** This is anything that changes *intent*:

- Any new, reversed, or re-scoped decision (any DECISIONS row that is not a status bump).
- Re-prioritizing **Next up**, or dropping/deferring a queued item.
- Marking something Done when verification was partial, tests were skipped, or the result is
  ambiguous.
- Anything the agent is unsure how to classify. **When in doubt, Tier B.**

**Always, both tiers:** say explicitly in the conversation which file changed and how — e.g.
"STATUS: moved F0→F1 to In progress (claude/f1-baseline); DECISIONS: no change." The owner must
never have to diff a file to learn what an agent recorded.

### Session-end sweep — leave nothing lingering

Before ending a session or handing off, the agent performs this sweep and reports the result:

1. **Mid-flight work** → **Paused / mid-flight** in STATUS: what is done, what remains, the branch
   the code is on, and the next concrete step. Enough for a different agent to resume cold.
2. **Ideas raised but not acted on** → **Next up** if it is work; **Pending** in DECISIONS if it is
   an unmade choice. An idea mentioned only in conversation is considered lost.
3. **Nothing stays only in the conversation.** Chat is not storage.

### Keeping both agents current

Git gives no real-time shared state: each worktree holds its own copy of these files, so one agent's
edit is invisible to the other until it is committed and pulled. The discipline that closes the gap:

- **Read STATUS at session start** (step 2 of the handshake above) — never rely on a remembered state.
- **`git pull` before claiming**, and **commit + push a claim immediately**. An unpushed claim does
  not exist, and the collision it was meant to prevent will happen.
- **One self-contained line per entry.** Line granularity is what lets concurrent edits merge.
- `.gitattributes` sets `merge=union` on both files, so two agents appending different lines merge
  automatically instead of conflicting. Union merge keeps *both* sides — after any merge, re-read
  the section and remove duplicates or reorder if needed.

## Concurrency protocol — read before parallel work

Codex and Claude share this repository. To avoid clobbering each other:

- **One agent per working copy.** Never let two agents edit the same working tree at once.
- **The primary checkout always stays on `main`.** This is the invariant everything else rests on:
  `main` is the coordination channel, so there must always be a working copy sitting on it, ready to
  accept a claim or release commit. Never check a task branch out in the primary checkout — if you
  want a branch, you want a worktree; they come together.
- **Claim work in `docs/project/STATUS.md`** ("In progress": task · owner · branch) *before* starting.
- Small, focused commits. Reference the decision or task they implement.

### Two paths, chosen by whether anyone else is working

| | Sequential (you are the only agent) | Concurrent (another agent is active) |
|---|---|---|
| Where | Primary checkout, directly on `main` | A worktree on `<owner>/<task>` |
| Branch | none | yes |
| Claim/release | not needed | required, on `main`, before the branch exists |
| Setup | already done | `scripts/worktree.sh new …` |

There is no third mode. "Task branch checked out in the primary checkout" is the one combination to
avoid: it breaks the invariant above, and `scripts/worktree.sh done` will refuse to release the claim
because the primary checkout is no longer on `main`.

### What may be committed to `main`

`main` is both the integration branch **and the coordination channel**. Two kinds of commit are
allowed directly on it:

1. **Coordination commits** — STATUS claim/release rows, and only those. These *must* go on `main`:
   a claim committed on a feature branch is invisible to every other agent, which defeats claiming
   entirely. Commit and push immediately; an unpushed claim does not exist.
2. **Sequential implementation work**, when only one agent is active.

**Concurrent implementation work never goes on `main`** — it goes on a `<owner>/<task>` branch in its
own worktree, and reaches `main` by merge. `scripts/worktree.sh` handles the coordination commits for
you; do not hand-edit STATUS claims while a worktree flow is in progress.

### The concurrent workflow, start to finish

```bash
# 1-4. pull-check main → claim in STATUS → push the claim → branch from main → venv
scripts/worktree.sh new claude/f1-baseline

# 5. work
cd ../temporallens-claude-f1-baseline
#    ... implement; make verify && make test && make lint; update STATUS/DECISIONS ...

# 5a. catch up with main BEFORE merging — other agents' claims have landed there.
git merge main            # resolve here, in your worktree, not on main

# 6. integrate (from the primary checkout, which is on main)
git -C ../temporallens merge claude/f1-baseline

# 7. remove worktree, delete the merged branch, release the claim on main
scripts/worktree.sh done claude/f1-baseline
```

Step **5a is not optional**: `main` moves while you work, and merging it into your branch first means
conflicts surface in your worktree instead of on the shared branch. Step **6 must precede step 7** —
`done` checks `git branch --merged main` and will keep an unmerged branch rather than delete work.

**Who owns the claim row.** Exactly one writer, at exactly two moments:

| | Adds/removes the "In progress" row |
|---|---|
| `scripts/worktree.sh new` | adds it, on `main`, before the branch exists |
| `scripts/worktree.sh done` | removes it, on `main`, after the merge succeeds |
| **A task branch** | **never touches the In progress table** |

A branch that edits the In progress table creates a second writer and a merge duplicate. Branches
*do* update the rest of STATUS (Latest changes, Done, Paused) per Definition of Done — just not that
table. With this rule the union merge has nothing to duplicate; if you ever do see two identical
rows, someone broke it — delete the extra and check what edited it.

### Creating a worktree

Use the helper — it performs the venv step that correctness depends on:

```bash
scripts/worktree.sh new claude/f1-baseline   # worktree + branch + its own venv (~3 s)
scripts/worktree.sh list
scripts/worktree.sh done claude/f1-baseline  # remove worktree + merged branch
```

Conventions it enforces:

| | Convention |
|---|---|
| Branch | `<owner>/<task>` — `claude/f1-baseline`, `codex/loader`, `denisa/…`. Owner: `[a-z0-9._]`, **no hyphen**; task: `[a-z0-9._-]`. The owner restriction keeps branch ↔ directory a bijection (otherwise `claude/foo-bar` and `claude-foo/bar` collide). |
| Directory | `../temporallens-<owner>-<task>` (sibling of the repo, never nested inside it) |
| `main` | Stays checked out in the primary checkout; it is the integration branch and the coordination channel (see above for what may be committed to it). |
| Base | Every task branch is cut from an up-to-date `main`, explicitly — never from the caller's HEAD. |

**Each worktree gets its own venv — this is correctness, not tidiness.** `temporallens` is installed
editable, so a venv resolves `import temporallens` to the source tree it was created from. Using
another worktree's venv silently imports and tests the *wrong files* while every command appears to
succeed. Verified: with the main venv used from inside a worktree, an edit made in the worktree is
invisible — the import still resolves to the primary checkout. `make setup` in each worktree costs
~3 s because uv hardlinks from its cache on the same volume.

### Which agent drives it

**Every agent and person uses `scripts/worktree.sh`. One mechanism, no exceptions.**

- **Codex:** do **not** use the app's native worktree feature for this repository. Per Codex's own
  documentation it is a separate lifecycle — worktrees are created under `$CODEX_HOME/worktrees`
  rather than beside the repo, they start in **detached HEAD** rather than on a named branch, and
  dependency setup requires configuring a separate local-environment setup script. None of that
  matches the conventions above, and a second lifecycle would put worktrees, branch names, and venv
  bootstrapping in two different places.
- **Claude, and humans:** the same script.

The script is what enforces the three things that fail *silently* when done by hand: the claim
reaching `main` before the branch exists, the branch being cut from an up-to-date `main`, and the
per-worktree venv. A second mechanism that skips any of them reintroduces the bug it was written to
prevent.

One rulebook, one tool. A second mechanism would inevitably diverge on branch naming, directory
layout, or the venv step — which is the divergence this section exists to prevent.

### Teardown

`scripts/worktree.sh done <owner>/<task>` — after the merge (step 6 above). It removes the worktree,
deletes the branch **only if merged**, and releases the STATUS claim on `main`. It refuses to touch a
worktree with uncommitted changes. Both `new` and `done` require the primary checkout to be clean and
on `main`; that is the invariant, not an inconvenience.

## Definition of done

A change is done when:

1. `make verify && make test && make lint` all pass.
2. **If `scripts/worktree.sh`, `tests/test_worktree.sh`, or the concurrency workflow changed,
   `make test-worktree` must also pass.** It is excluded from `make test` for speed, not because it
   is optional — that script commits and pushes to `main`, so it is the least forgiving code here.
3. If configs changed, `docs/experiments/` is updated and the 1:1 config↔spec mapping holds.
4. **`docs/project/STATUS.md` is updated** per the working agreement above — Latest changes, In
   progress, and Next up reflect reality; any new decision is logged in `DECISIONS.md`. This step is
   not optional: STATUS is the one file everyone reads, so a change that doesn't update it is not
   done.
5. The change was reported in the conversation (which files, what changed).
6. Nothing ignored (data, checkpoints, secrets) is staged.
