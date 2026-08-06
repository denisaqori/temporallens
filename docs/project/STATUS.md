# STATUS — start here

**This is the single source of truth for the state of the repository.** Every agent and person
starts here to see what is done, what changed most recently, and what to pick up next — and updates
it in the same change that moves the work.

**Scope: this file holds _work_ and is mutable** — items enter, move, and leave. *Choices* (what was
decided, by whom, when) are append-only in [DECISIONS.md](DECISIONS.md); technical rationale lives in
[../experiments/README.md](../experiments/README.md). Reference a decision by ID here; never restate
its rationale. Update rules, autonomy tiers, and the session-end sweep: see **AGENTS.md → Working
agreement**.

> 👉 **Next to pick up:** D18–D21 fully specify F1 epoch selection, so the F0→F1 vertical slice can
> proceed. The generative arm has additional fail-closed decisions in [Next up](#next-up-priority-order).

_Last updated: 2026-08-05_

## Latest changes

**A curated highlight list of shipped work, newest first — deliberately no commit hashes.** `git
log --oneline` is the complete, live record; duplicating hashes here only creates bookkeeping that
goes stale. This list answers "what actually changed lately?" in prose, at a glance.

Entries are added when work lands. Coordination commits (`Claim …` / `Release … in STATUS`) are
omitted — they are workflow bookkeeping, not changes to the project, and listing them would bury
the real entries. Uncommitted state belongs to `git status`; in-flight work belongs in
[In progress](#in-progress) or [Paused / mid-flight](#paused--mid-flight), never here.

- Pre-F0/F1 metric hygiene: the minimized VAE objective is named `negative_elbo` (D22), correcting
  prose that had a decreasing ELBO proving optimization when the bound is maximized. Metric
  registration now reads first-column keys from the two reference tables rather than any backticked
  identifier, so a key like `seed` no longer passes. G2 and G4 gained fail-closed blockers, and four
  gaps the audit exposed — VAE objective components, the G2 quality/acceptance contract, G3/G4
  repeated-run aggregation, and G4 headline ECE aggregation — are recorded as Pending.
- Every `evaluation.metrics` key is mentioned in a spec. The 14 generation-arm keys the audit found
  undocumented — the VAE terms, G2 gate metrics, and *k*-indexed curves — are tabulated in
  generative-arm.md, and a lexical test rejects a config metric that no spec mentions. This is
  name-level coverage, not an executable definition. `make lint` also runs `black --check` now,
  which is why the drift it would have caught went unnoticed.
- Protocol-audit hardening landed: the frozen split is independently pinned and strictly validated,
  reportable configs consume it without duplicated subject/repetition lists, debug splits are
  explicitly non-reportable, packaged installs can resolve the manifest, and blocked G-series
  choices fail closed. F1 epoch selection is executable under D18–D21, and G3 robustness compares
  both matched adaptation strategies.
- Subject-independent split frozen as versioned data (D16): fixed test/training roles, eight
  validation folds, corrected label columns, and subject-calibration repetitions now load through
  one verified manifest.
- G3 calibration protocol frozen (D14, D15): *k* is complete real gesture trials per subject, with
  nested shared schedules and matched real-adaptation versus real-plus-synthetic strategies.
- Evaluation protocol reconciled against the group's published method (D9–D11): epoch selection is
  the smoothed validation peak, not a raw best epoch or a fixed budget (§5.2); the refit is
  cross-checked against the eight fold-model test scores (§5.3); confusion matrices are reported for
  cross-validation as well as test; class imbalance is handled by weighted loss, never resampling,
  and the test set is never balanced (§3.6). The three foundation configs declared no `loss` at all
  and now do.
- Data-loading and repetition rules taken from the DB2 descriptor (D12, D13): `restimulus` /
  `rerepetition` as the binding label columns with an assert on load, no baseline subtraction, all
  of a subject's repetitions on one side of the split, and disjoint calibration/evaluation
  repetitions chosen to span the drift the descriptor measured. Spec §3.1, §3.2, generative-arm G3.
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
- Protocol and workflow choices recorded in DECISIONS.md.
- Repo on GitHub over SSH: `git@github.com:denisaqori/temporallens.git`, `main` pushed.
- Robustness evaluation decoupled from the target model (D2): shared `robustness_targets.yaml`,
  `scripts/evaluate.py` stub, `eval-robustness` target.
- Governance Tier-1: `AGENTS.md`, `CLAUDE.md`, this file, `DECISIONS.md`.
- Concurrency mechanism: `scripts/worktree.sh` (single tool for every agent), the
  primary-stays-on-`main` invariant, and `tests/test_worktree.sh` covering its mutating paths.
  Hardened across four review rounds with Codex; see DECISIONS.
- Agent re-rooting complete: every work surface (Codex desktop, Claude terminal CLI, Claude desktop)
  resolves to this repository; the ChatGPT-project mirror is retired as a work surface.
- G3 calibration-budget and strategy choices frozen (D14, D15); specifications and configs
  reconciled. The remaining reporting and training-policy choices stay open below.
- Split manifest and verified loader landed (D16).

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

1. **P0 — Owner protocol decisions.** F0→F1 can proceed under D18–D21; the remaining choices block
   headline inference or their named G-series configs. All are tracked in DECISIONS → Pending and
   must be resolved explicitly rather than receiving runner defaults.

   **Still open** — tracked in DECISIONS → Pending:
   - **Inferential unit.** Per-window vs. per-subject changes every confidence interval, and D6's
     8×8 matrix adds a second axis — fold variance and subject variance must stay separate.
     Deferred by Denisa on 2026-07-30 pending further discussion.
   - **G1 subject-embedding contract.** Fix its estimator/pooling, pseudo-calibration schedule,
     target exclusion, population embedding, and class-information control.
   - **G2 gate population/final-test policy.** Fix the development folds/aggregation and one-shot
     final diagnostic, including coverage of the conditional distributions used across *k*.
   - **G3 headline extraction.** The unit is fixed, but target accuracy, curve interpolation or
     monotonic treatment, unreachable targets, uncertainty, and subject aggregation are not.
   - **G3 calibration-subgroup aggregation.** Seen/unseen gesture reporting is required, but its
     pooled-window versus class-macro definition remains open; empty groups report `NA`.
   - **G3 adaptation training budget and mixture.** Fix optimizer steps, batch size,
     real/synthetic batch composition and weighting, real-window exposure, and epoch semantics.
   - **G3 adaptation objective.** Fix optimizer, loss, class weights, learning rate,
     regularization, and absent-class handling.
   - **G3 replay control.** Add a population-conditioned balanced-replay control or narrow the
     causal claim to avoid attributing generic replay gains to subject conditioning.
   - **G3 schedule reproducibility.** Fix the PRNG and schedule-index base; selected trials are
     persisted with each run.
   - **G3 calibration-rest ownership.** Rest does not increment *k*, but the runner needs a fixed
     rule for whether a selected gesture trial permits one adjacent rest interval. Inspect a real
     MAT file before deciding; never silently expose all held-out-subject rest.
   - **ECE binning.** Bin count and equal-width vs. equal-mass are unspecified; unfixed, the same
     model yields different numbers and runs are not comparable.
   - **Overconfidence-error definition.** §3.4's prose and the literature term name two different
     metrics; the language arm's headline claim rests on this one.
   - **G0/G1 VAE objective components.** The minimized objective is correctly named
     `negative_elbo`; the reconstruction likelihood/reduction, units, and raw-versus-weighted KL
     reporting remain open.
   - **G2 metric/gate contract.** Fix grouped discriminator evaluation, AUC orientation and
     dependence-aware CI, CI/per-class gate behavior, nearest-neighbour semantics, and a genuine
     diversity or coverage diagnostic.
   - **G3/G4 within-subject repeated-run aggregation.** Fix aggregation across the persisted
     schedule/seed/draw runs without duplicating the population reference or treating reused
     evaluation windows as independent observations; record adaptation-seed and synthetic-draw IDs.
   - **G4 headline ECE aggregation.** Fix subject weighting and whether pooled-prediction ECE is
     headline, supplemental, or omitted.

2. **P1 — F0→F1 vertical slice**: NinaPro DB2 Exercise B loader + `prepare_dataset.py`, 1-D CNN
   encoder + head, training loop, `make debug` (F0) passing end to end, then the F1 baseline under
   D18–D21. The F1 trainer must honor the checkpoint contract
   (`{model_state, model_config}`) as an acceptance criterion, not a later task.
3. **P2 — Reconcile** remaining planning-docs wording with the authoritative spec where it drifts.

## Known open items (not yet scheduled)

- Robustness configs are target-agnostic in schema but **not executable** until the eval logic and
  the arm models exist (`evaluate.py` is a stub).
- G3 robustness expansion across held-out subject × schedule × *k* is declared for both adapted
  strategies, but the runner must resolve `latest` to an immutable run ID before evaluating them.
- Evaluation outputs currently mix primitive metrics, grouped curves, metadata, uncertainty, and
  derived estimands. Define and validate a result schema that separates those roles alongside the
  F0/F1 evaluator, then migrate the G-series configs before their runners are implemented.
- The new G0/G1 objective-component and G3/G4 repeated-run Pending choices are not yet wired into
  runner-enforced config schemas; add the appropriate execution/reportability gates when
  implementing their runners.
- `head.pooling: last_token` assumes right-padding; the L-series trainer must pin the tokenizer or
  pool the true last non-pad index (silent failure otherwise).
- Makefile `debug/train-*/report` targets reference scripts that are not written yet.
- GitHub Issues/Project sync deferred to Tier-2 (`gh` not installed).
