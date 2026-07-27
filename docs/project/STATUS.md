# STATUS

Live project state. The single human-readable answer to "what is done, what is in progress, what is
next." Update it in the same change that moves the work. Decision *provenance* lives in
[DECISIONS.md](DECISIONS.md); technical rationale for protocol decisions lives in
[../experiments/README.md](../experiments/README.md).

_Last updated: 2026-07-27 · main @ 8087c5d_

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

| Task | Owner | Branch |
|---|---|---|
| _(claim work here before starting — see AGENTS.md concurrency protocol)_ | — | — |

## Next up (priority order)

1. **P0 — Re-root both agents** on `…/temporallens` (Codex: open the real repo; Claude: point the
   desktop project at the repo folder). Parity + safety.
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
