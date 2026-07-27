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
**Latest changes** takes commits only (a line without a hash does not belong there), and anything
in flight goes to **In progress** or **Paused / mid-flight**, never into the changelog.

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
- For parallel work, give each agent an isolated checkout over the shared history:
  `git worktree add ../temporallens-<task> -b <branch>`.
- **Each worktree needs its own venv.** `temporallens` is installed editable, so a venv resolves
  `import temporallens` to the source tree it was created from. Sharing one venv across worktrees
  silently tests the *wrong* files. Run `make setup` inside every new worktree (fast — uv hardlinks
  from its cache).
- **Claim work in `docs/project/STATUS.md`** ("In progress": task · owner · branch) *before* starting.
- Small, focused commits. Reference the decision or task they implement.

## Definition of done

A change is done when:

1. `make verify && make test && make lint` all pass.
2. If configs changed, `docs/experiments/` is updated and the 1:1 config↔spec mapping holds.
3. **`docs/project/STATUS.md` is updated** per the working agreement above — Latest changes, In
   progress, and Next up reflect reality; any new decision is logged in `DECISIONS.md`. This step is
   not optional: STATUS is the one file everyone reads, so a change that doesn't update it is not
   done.
4. The change was reported in the conversation (which files, what changed).
5. Nothing ignored (data, checkpoints, secrets) is staged.
