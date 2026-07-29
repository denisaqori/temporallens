# STATUS — start here

**This is the single source of truth for the state of the repository.** Every agent and person
starts here to see what is done, what changed most recently, and what to pick up next — and updates
it in the same change that moves the work.

**Scope: this file holds _work_ and is mutable** — items enter, move, and leave. *Choices* (what was
decided, by whom, when) are append-only in [DECISIONS.md](DECISIONS.md); technical rationale lives in
[../experiments/README.md](../experiments/README.md). Reference a decision by ID here; never restate
its rationale. Update rules, autonomy tiers, and the session-end sweep: see **AGENTS.md → Working
agreement**.

> 👉 **Next to pick up:** freeze the split (P0) — subjects, repetitions, and the headline statistic.
> See [Next up](#next-up-priority-order).

_Last updated: 2026-07-29_

## Latest changes

**A curated highlight list of shipped work, newest first — deliberately no commit hashes.** `git
log --oneline` is the complete, live record; duplicating hashes here only creates bookkeeping that
goes stale. This list answers "what actually changed lately?" in prose, at a glance.

Entries are added when work lands. Coordination commits (`Claim …` / `Release … in STATUS`) are
omitted — they are workflow bookkeeping, not changes to the project, and listing them would bury
the real entries. Uncommitted state belongs to `git status`; in-flight work belongs in
[In progress](#in-progress) or [Paused / mid-flight](#paused--mid-flight), never here.

- Evaluation protocol reconciled against the group's published method (D9–D11): epoch selection is
  the smoothed validation peak, not a raw best epoch or a fixed budget (§5.2); the refit is
  cross-checked against the eight fold-model test scores (§5.3); confusion matrices are reported for
  cross-validation as well as test; class imbalance is handled by weighted loss, never resampling,
  and the test set is never balanced (§3.6). The three foundation configs declared no `loss` at all
  and now do.
- Metric set settled (D8): per-class precision and recall added beside the confusion matrix on the
  six classification configs; spec §3.4 now states that cross-validation and testing run the same
  metric set, and that fold confusion matrices are averaged element-wise rather than summed.
- Checkpoint naming split in two (D7): `refit.pt` is the one artifact downstream consumers read;
  per-fold checkpoints moved to `folds/fold{k}/best.pt`. Swept through the robustness registry, the
  language and generation configs, all three spec documents, and `evaluate.py`; spec §5.1 defines
  the distinction.
- Validation scheme settled (D6): 8-fold CV over the training subjects, refit on all 32 downstream.
- Worktree mechanism: `scripts/worktree.sh` + `tests/test_worktree.sh` (`make test-worktree`);
  claim-on-`main` coordination, atomic create/teardown.
- Split-freeze findings recorded: no validation set, repetitions undefined, inferential unit
  unspecified.
- STATUS/DECISIONS working agreement; union-merge on project memory.
- Governance Tier-1 added: `AGENTS.md`, `CLAUDE.md`, STATUS, DECISIONS.
- Robustness evaluation decoupled from the target model (D2).
- Initial commit: environment, experiment specifications, and configs.

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
- Concurrency mechanism: `scripts/worktree.sh` (single tool for every agent), the
  primary-stays-on-`main` invariant, and `tests/test_worktree.sh` covering its mutating paths.
  Hardened across four review rounds with Codex; see DECISIONS.
- Agent re-rooting complete: every work surface (Codex desktop, Claude terminal CLI, Claude desktop)
  resolves to this repository; the ChatGPT-project mirror is retired as a work surface.

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

1. **P0 — Freeze the split** (train/val/test subjects and repetitions) and the headline statistic.
   Everything downstream depends on it: numbers computed on different splits are not comparable, and
   the whole language arm rests on comparing L2 against the F1 reference row. Freeze once, record as
   *data* (a committed manifest the loader asserts against), never revise.

   **Settled — the validation scheme (D6, D7).** 8-fold CV over the 32 training subjects, 4
   validation subjects per fold; the 8×8 fold×test-subject matrix is extended analysis only; the
   downstream model is refit on all 32 at the median per-fold best epoch, and its test score is the
   F1 reference row. Checkpoints named accordingly (`refit.pt` vs `folds/fold{k}/best.pt`), already
   swept through configs and specs.

   **Still blocking the manifest** — four unmade choices, all tracked in DECISIONS → Pending:
   - **Repetition policy.** DB2 has 6 repetitions per movement per subject; the string "repetition"
     appears nowhere in `docs/experiments/` or `configs/`. Sharpest for D5: which repetitions supply
     the k calibration examples vs. the evaluation windows, or the adaptation baseline leaks.
   - **Inferential unit.** Per-window vs. per-subject changes every confidence interval, and D6's
     8×8 matrix adds a second axis — fold variance and subject variance must stay separate.
   - **ECE binning.** Bin count and equal-width vs. equal-mass are unspecified; unfixed, the same
     model yields different numbers and runs are not comparable.
   - **Overconfidence-error definition.** §3.4's prose and the literature term name two different
     metrics; the language arm's headline claim rests on this one.

   **Deliverable:** a committed split manifest (e.g. `configs/splits/subject_independent_v1.yaml`)
   listing exact subject IDs per role and per fold, plus a hash the loader asserts, so the split
   cannot drift.
2. **P1 — F0→F1 vertical slice**: NinaPro DB2 Exercise B loader + `prepare_dataset.py`, 1-D CNN
   encoder + head, training loop, `make debug` (F0) passing end to end, then the F1 baseline.
3. **P1 — Honor the checkpoint contract** in the F1 trainer (`{model_state, model_config}`).
4. **P2 — Reconcile** README/planning-docs wording with the authoritative spec where they drift.

## Known open items (not yet scheduled)

- Robustness configs are target-agnostic in schema but **not executable** until the eval logic and
  the arm models exist (`evaluate.py` is a stub).
- `head.pooling: last_token` assumes right-padding; the L-series trainer must pin the tokenizer or
  pool the true last non-pad index (silent failure otherwise).
- Makefile `debug/train-*/report` targets reference scripts that are not written yet.
- GitHub Issues/Project sync deferred to Tier-2 (`gh` not installed).
