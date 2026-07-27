# AGENTS.md — operating rules for agents in this repository

Canonical instructions for any agent (Codex, Claude Code, or a human using either) working in
TemporalLens. Codex reads this file natively; Claude Code reads it via the `@AGENTS.md` import in
`CLAUDE.md`. Keep the shared rules here, not duplicated per-agent.

## Workspace root — check this first

The one correct root is this repository:

```
/Users/denisamcdonald/Projects/TemporalLens/temporallens
```

- **Not** the parent `…/TemporalLens/` (that is not a git repo; it holds a separate
  `planning-documents/` repo as a sibling).
- **Not** any ChatGPT-project mirror under `~/.codex/.chatgpt-projects/…` (read-only, empty sources).

Verify before working: `git rev-parse --show-toplevel` must print `…/temporallens`. If an agent is
rooted elsewhere, either re-root it or use absolute paths under this repo — never write relative to
the wrong root.

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

## Concurrency protocol — read before parallel work

Codex and Claude share this repository. To avoid clobbering each other:

- **One agent per working copy.** Never let two agents edit the same working tree at once.
- For parallel work, give each agent an isolated checkout over the shared history:
  `git worktree add ../temporallens-<task> -b <branch>`.
- **Claim work in `docs/project/STATUS.md`** ("In progress": task · owner · branch) *before* starting.
- Small, focused commits. Reference the decision or task they implement.

## Definition of done

A change is done when:

1. `make verify && make test && make lint` all pass.
2. If configs changed, `docs/experiments/` is updated and the 1:1 config↔spec mapping holds.
3. `docs/project/STATUS.md` reflects the new state; any new decision is logged in `DECISIONS.md`.
4. Nothing ignored (data, checkpoints, secrets) is staged.
