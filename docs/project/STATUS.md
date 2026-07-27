# STATUS — start here

**This is the single source of truth for the state of the repository.** Every agent and person
starts here to see what is done, what changed most recently, and what to pick up next — and updates
it in the same change that moves the work.

**Scope: this file holds _work_ and is mutable** — items enter, move, and leave. *Choices* (what was
decided, by whom, when) are append-only in [DECISIONS.md](DECISIONS.md); technical rationale lives in
[../experiments/README.md](../experiments/README.md). Reference a decision by ID here; never restate
its rationale. Update rules, autonomy tiers, and the session-end sweep: see **AGENTS.md → Working
agreement**.

> 👉 **Next to pick up:** verify or re-root Claude desktop on the repo (remaining P0), then
> freeze the split (P0). See [Next up](#next-up-priority-order).

_Last updated: 2026-07-27 · main @ 505c92f_

## Latest changes

**Commits only** — one line per commit, newest first. No entry without a hash: uncommitted state is
what `git status` is for, and hand-copied inventories are stale the moment they are written.
In-flight work belongs in [In progress](#in-progress) or [Paused / mid-flight](#paused--mid-flight).
Full history: `git log --oneline`.

- `505c92f` — Tier-1 governance added (AGENTS, CLAUDE, STATUS, DECISIONS).
- `8087c5d` — robustness evaluation decoupled from the target model (D2).
- `c1d3868` — initial commit: environment, experiment specifications, and configs.

## Phase

Week 0–1 (Setup → Milestone 0). Local environment complete and verified; the research pipeline
(loader, models, training, evaluation) is **not yet implemented**. Milestone 0 (F0→F1) has not
started.

## Done

- Reproducible local env: uv + Python 3.11, PyTorch MPS, `make setup/verify/test/lint` all green,
  `uv.lock` tracked.
- Experiment specification authored: `docs/experiments/` (shared protocol + generative + language
  arms); every ablation maps 1:1 to a config under `configs/experiment/{foundation,generation,language}/`.
- Decisions D1–D5 recorded (see DECISIONS.md).
- Repo on GitHub over SSH: `git@github.com:denisaqori/temporallens.git`, `main` pushed.
- Robustness evaluation decoupled from the target model (D2): shared `robustness_targets.yaml`,
  `scripts/evaluate.py` stub, `eval-robustness` target.
- Governance Tier-1: `AGENTS.md`, `CLAUDE.md`, this file, `DECISIONS.md`.

## In progress

Claim here **before** starting; commit and push the claim immediately (an unpushed claim does not
exist). Release the row when the work lands.

| Task | Owner | Branch |
|---|---|---|
| _(none — claim work here before starting)_ | — | — |

## Paused / mid-flight

Work that is started but not finished. Each entry must be resumable cold by a different agent:
what is done, what remains, which branch, and the next concrete step.

| Item | Owner | Branch | Done so far → next step |
|---|---|---|---|
| _(none)_ | — | — | — |

## Next up (priority order)

1. **P0 — Finish agent re-rooting.** This Codex desktop task now uses the primary checkout, the
   invalid ChatGPT-project mirror task is being retired as a work surface, and the Claude CLI was
   reported correctly launched from the repository. Claude desktop remains unverified: run the
   session handshake there and re-root its project folder to this repository if needed.
2. **P0 — Freeze the split** (train/val/test subjects and repetitions) and the headline statistic.
   Everything downstream depends on it.
3. **P1 — F0→F1 vertical slice**: NinaPro DB2 Exercise B loader + `prepare_dataset.py`, 1-D CNN
   encoder + head, training loop, `make debug` (F0) passing end to end, then the F1 baseline.
4. **P1 — Honor the checkpoint contract** in the F1 trainer (`{model_state, model_config}`).
5. **P2 — Reconcile** README/planning-docs wording with the authoritative spec where they drift.

## Known open items (not yet scheduled)

- Robustness configs are target-agnostic in schema but **not executable** until the eval logic and
  the arm models exist (`evaluate.py` is a stub).
- `head.pooling: last_token` assumes right-padding; the L-series trainer must pin the tokenizer or
  pool the true last non-pad index (silent failure otherwise).
- Makefile `debug/train-*/report` targets reference scripts that are not written yet.
- GitHub Issues/Project sync deferred to Tier-2 (`gh` not installed).
