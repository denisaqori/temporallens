# STATUS — start here

**This is the single source of truth for the state of the repository.** Every agent and person
starts here to see what is done, what changed most recently, and what to pick up next — and updates
it in the same change that moves the work.

**Scope: this file holds _work_ and is mutable** — items enter, move, and leave. *Choices* (what was
decided, by whom, when) are append-only in [DECISIONS.md](DECISIONS.md); technical rationale lives in
[../experiments/README.md](../experiments/README.md). Reference a decision by ID here; never restate
its rationale. Update rules, autonomy tiers, and the session-end sweep: see **AGENTS.md → Working
agreement**.

> 👉 **Next developer task:** continue the F0→F1 vertical slice. **Next owner task:** settle the
> remaining protocol choices in [Next up](#next-up-priority-order), beginning with the F1
> architecture and calibration clarifications recorded under
> [Known open items](#known-open-items-not-yet-scheduled).

_Last updated: 2026-10-01_

## Latest changes

**A curated highlight list of shipped work, newest first — deliberately no commit hashes.** `git
log --oneline` is the complete, live record; duplicating hashes here only creates bookkeeping that
goes stale. This list answers "what actually changed lately?" in prose, at a glance.

Entries are added when work lands. Coordination commits (`Claim …` / `Release … in STATUS`) are
omitted — they are workflow bookkeeping, not changes to the project, and listing them would bury
the real entries. Uncommitted state belongs to `git status`; in-flight work belongs in
[In progress](#in-progress) or [Paused / mid-flight](#paused--mid-flight), never here.

- Repository consistency pass hardened raw/processed DB2 validation, made debug window caps
  class-covering, repaired run-log serialization and worktree provenance, reconciled config/spec
  metric contracts, corrected stale implementation claims, and added metadata-only acquisition
  provenance. The F1 architecture and remaining calibration choices are recorded below for owner
  resolution; no protocol decision was silently supplied by the implementation.
- Corrective pass on D25–D27 after Codex review. Added the per-subject macro-F1, Brier and
  overconfidence keys D26 promised but no config could emit (F2 exempted, and the reason written
  into the config: its test windows come from subjects that are in training, so a per-subject
  number there measures nothing). Specified the paired bootstrap, the post-scaling bin recompute,
  distinct raw/scaled result keys, and `NA` on zero errors. Re-blocked G1 — rotation *selection*
  still inherits D14's Pending PRNG convention, and the class-composition probe was a slogan
  rather than a gate. Added a `subject_embedding_changes_sample` check, since nothing verified the
  embedding affected the output at all. Corrected "pairing cancels the subject effect" to
  "controls shared subject variation", and defined the minimum reportable difference as a CI
  criterion rather than a power calculation the no-tests-at-n=8 rule forbids.
- `tests/test_project_memory.py` enforces the bookkeeping that kept breaking by hand: no blank
  line splitting a Markdown table, every `blocked_on` entry naming a field that exists, Pending
  and Resolved disjoint, and STATUS's open list matching Pending in **both** directions.
- Three protocol decisions settled (D25–D27). Overconfidence error is split into the binned
  positive gap and a separate descriptive `mean_confidence_when_wrong`, now on all five foundation
  configs so the language arm has an F1 reference row to compare against. The subject is the
  inferential unit for every metric, with paired cross-arm intervals and a stated minimum
  reportable difference. G1's subject-embedding contract is fixed — class-centred mean residual,
  rotated support repetitions, learned population vector, class-composition probe — which unblocks
  the generative chain's root; G1 still fails closed on the VAE objective components.
- Calibration measurement and windowing boundaries settled (D23, D24), and three protocol
  assumptions checked against the shipped data for the first time. `stimulus` disagrees with
  `restimulus` on 19.9% of samples with a median onset shift of 856 ms, so D12 is now a measured
  fact rather than a quoted one; rest is 50.2% of windows at stride 100, about 17.5x the median
  gesture class, which is why `macro_f1` and not `accuracy` is the headline (D11); and
  `rerepetition` is non-zero during rest, which satisfies the inspection precondition the G3
  rest-ownership question was waiting on. ECE now has fixed bins (adaptive, M=10), temperature
  scaling with an out-of-fold T source, subject-level resampling, and `brier_score` beside it
  across all nine configs that report calibration.
- Data layer for F0→F1: DB2 Exercise B reader (corrected label columns asserted on load), a
  processed per-subject format, and a window indexer that never crosses a `(label, repetition)`
  boundary. Indexing is separate from materialising, because materialising a 28-subject fold at
  stride 100 would need about 8 GB.
- README reframed the project around calibration burden, renamed the central-question and method
  sections, and tightened contribution and evaluation prose without changing protocol.
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
- Data-loading and repetition rules taken from the DB2 descriptor (D12, D13): `restimulus` /
  `rerepetition` as the binding label columns with an assert on load, no baseline subtraction, all
  of a subject's repetitions on one side of the split, and disjoint calibration/evaluation
  repetitions chosen to span the drift the descriptor measured. Spec §3.1, §3.2, generative-arm G3.
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
- Split-freeze audit recorded the gaps later resolved by D6 and D13; the general inferential unit
  remains open outside D23's calibration-specific rule.
- STATUS/DECISIONS working agreement; union-merge on project memory.
- Governance Tier-1 added: `AGENTS.md`, `CLAUDE.md`, STATUS, DECISIONS.
- Robustness evaluation decoupled from the target model (D2).
- Initial commit: environment, experiment specifications, and configs.

## Phase

Week 0–1 (Setup → Milestone 0). Local environment complete and verified. **Milestone 0 (F0→F1) has
started**: NinaPro DB2 Exercise B is downloaded locally and the data layer (reader, processed
format, windower) is implemented and tested. Normalization, the encoder and head, the training
loop, `prepare_dataset.py` and `train_encoder.py` are still to write, so no run has executed
end to end yet.

## Done

- Reproducible local env: uv + Python 3.11, PyTorch MPS, `make setup/verify/test/lint` all green,
  `uv.lock` tracked.
- NinaPro DB2 Exercise B acquisition and verification completed for all 40 subjects; repository
  provenance/checksums live in `docs/data/`, while the archives and MAT files remain ignored.
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

1. **P0 — Owner protocol decisions.** F0 plumbing can proceed under D18–D21, but a reportable F1
   still needs the architecture and calibration clarifications listed under Known open items. The
   choices below block headline inference or their named G-series configs; all are tracked in
   DECISIONS → Pending and must be resolved explicitly rather than receiving runner defaults.

   **Still open** — tracked in DECISIONS → Pending:
   - **G2 gate population and final-test policy.** Fix the development folds/aggregation and one-shot
     final diagnostic, including coverage of the conditional distributions used across *k*.
   - **G3 headline extraction.** The unit is fixed, but target accuracy, curve interpolation or
     monotonic treatment, unreachable targets, uncertainty, and subject aggregation are not.
   - **G3 calibration-subgroup aggregation.** Seen/unseen gesture reporting is required, but its
     pooled-window versus class-macro definition remains open; empty groups report `NA`.
   - **G3 adaptation training budget and mixture.** Fix optimizer steps, batch size,
     real/synthetic batch composition and weighting, real-window exposure, and epoch semantics.
   - **G3 adaptation objective.** Fix optimizer, loss, class weights, learning rate,
     regularization, and absent-class handling.
   - **G3 population-replay control.** Add a population-conditioned balanced-replay control or narrow the
     causal claim to avoid attributing generic replay gains to subject conditioning.
   - **G3 schedule reproducibility.** Fix the PRNG and schedule-index base; selected trials are
     persisted with each run.
   - **G3 calibration-rest ownership.** Rest does not increment *k*, but the runner needs a fixed
     rule for whether a selected gesture trial permits one adjacent rest interval. Inspection is
     complete and assignment is feasible; never silently expose all held-out-subject rest.
   - **G1 embedding operational detail.** D27 fixed the estimator, schedule and population
     vector, but rotation *selection* inherits D14's Pending PRNG convention and the
     class-composition probe needs a target, subject-grouped split, score and threshold before it
     can gate anything.
   - **G0/G1 VAE objective components.** The minimized objective is correctly named
     `negative_elbo`; the reconstruction likelihood/reduction, units, and raw-versus-weighted KL
     reporting remain open.
   - **G2 quality-metric and acceptance contract.** Fix grouped discriminator evaluation, AUC orientation and
     dependence-aware CI, CI/per-class gate behavior, nearest-neighbour semantics, and a genuine
     diversity or coverage diagnostic.
   - **G3/G4 within-subject repeated-run aggregation.** Fix aggregation across the persisted
     schedule/seed/draw runs without duplicating the population reference or treating reused
     evaluation windows as independent observations; record adaptation-seed and synthetic-draw IDs.
   - **G4 headline ECE aggregation.** Fix subject weighting and whether pooled-prediction ECE is
     headline, supplemental, or omitted.

2. **P1 — F0→F1 vertical slice**: implement normalization + `prepare_dataset.py`, the 1-D CNN
   encoder + head, and the training loop; get `make debug` (F0) passing end to end, then run the F1
   baseline under D18–D21. The F1 trainer must honor the checkpoint contract
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
- D23's temperature rule is internally inconsistent: one pooled out-of-fold temperature and eight
  per-fold temperatures whose median feeds the refit cannot both be the procedure. F2 also has no
  subject-CV folds from which its configured temperature could come. Resolve both before calibrated
  F1/F2 evaluation; robustness runs consume the frozen target checkpoint's stored temperature.
- Calibration execution still needs exact Brier reduction/reporting, bootstrap level/count/seed,
  deterministic equal-mass tie handling, and distinct result keys for raw versus scaled ECE.
- Retain the simulation or analysis artifact that justified D23's M=10 choice, and recompute any
  finite-sample ECE detection threshold for each actual evaluation design rather than reusing 0.037.
- The F1 reference architecture and optimizer are not fully reconstructible from its config: layer
  stack, kernels/pooling/activation/normalization, optimizer/scheduler, initialization, and gradient
  handling must be frozen before implementing `model_config` and the trainer.
- Processed NPZ recordings load eagerly and are not memory-mappable. The F1 batching/normalization
  path must avoid duplicating the roughly multi-GiB 32-subject signal across workers; `num_workers`
  is 0 as the safe local default until a measured lazy/memory-mapped design exists.
- D10 requires the refit-minus-fold-mean result, but its output key/schema is not yet defined.
- L2–L4 do not yet encode a cross-validation/refit selection contract and request `overconfidence_error` (settled as
  D25) without an evaluation gate; settle the selection contract before language-arm execution.
- F3–F5 still need executable perturbation definitions (order relative to normalization,
  noise/channel-mask sampling and seeds, repetitions, and aggregation).
- Governance wording still conflicts: AGENTS calls DECISIONS rows append-only while DECISIONS moves
  approved proposals from Pending to Resolved, and the accepted L2/L3/L4 readout row has no decision
  ID. Reconcile the retention rule and assign stable identifiers without rewriting history.
- The `merge=union` rule assumes self-contained one-line entries, but STATUS is mutable and most
  entries span multiple lines. Concurrent deletion/reordering can retain stale fragments instead of
  conflicting. Decide whether to restore normal merges or restructure project memory before relying
  on union merge as collision protection.
- The new G0/G1 objective-component and G3/G4 repeated-run Pending choices are not yet wired into
  runner-enforced config schemas; add the appropriate execution/reportability gates when
  implementing their runners.
- `head.pooling: last_token` assumes right-padding; the L-series trainer must pin the tokenizer or
  pool the true last non-pad index (silent failure otherwise).
- The interface-only Make targets listed in AGENTS.md reference scripts that are not written yet.
- GitHub Issues/Project sync deferred to Tier-2 (`gh` not installed).
