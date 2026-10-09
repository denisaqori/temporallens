# STATUS — start here

**This is the single source of truth for the state of the repository.** Every agent and person
starts here to see what is done, what changed most recently, and what to pick up next — and updates
it in the same change that moves the work.

**Scope: this file holds _work_ and is mutable** — items enter, move, and leave. *Choices* (what was
decided, by whom, when) are append-only in [DECISIONS.md](DECISIONS.md); technical rationale lives
in [../experiments/README.md](../experiments/README.md). Reference a decision by ID here; never
restate its rationale. Update rules, autonomy tiers, and the session-end sweep: see **AGENTS.md →
Working agreement**.

> 👉 **Next developer task:** continue the F0→F1 vertical slice. **Next owner task:** settle the
> remaining protocol choices in [Next up](#next-up-priority-order), beginning with the F1
> architecture and calibration clarifications recorded under
> [Known open items](#known-open-items-not-yet-scheduled).

_Last updated: 2026-10-08_

## Latest changes

**A curated highlight list of shipped work, newest first — deliberately no commit hashes.**
`git log --oneline` is the complete, live record; duplicating hashes here only creates bookkeeping
that goes stale. This list answers "what actually changed lately?" in prose, at a glance.

Entries are added when work lands. Coordination commits (`Claim …` / `Release … in STATUS`) are
omitted — they are workflow bookkeeping, not changes to the project, and listing them would bury
the real entries. Uncommitted state belongs to `git status`; in-flight work belongs in
[In progress](#in-progress) or [Paused / mid-flight](#paused--mid-flight), never here.

- **F0 runs end to end**, and every config key now binds. `scripts/train_encoder.py` consumes each
  leaf key or refuses: a key it does not map, check or record is an error, which is what makes the
  defect behind `loss`, `device`, `model.type` and `dataset.normalize` structurally impossible
  rather than something an audit finds. It also refuses what it cannot honour exactly — F1's
  `split_manifest` config and F2's random-window split are both rejected rather than run as a single
  subject split, and `wandb_mode: online` is rejected because nothing logs remotely. Two tests keep
  the key set honest: one asserts every run-config field has a config path, the other trains and
  compares the metric keys produced against the list the loader gates on. `device` became an
  enumerated field where an unavailable pin raises instead of falling back, and the run log records
  the resolved device so `auto` stays reproducible. `cnn1d.ARCHITECTURE` holds the D32 stack as a
  frozen constant, recorded into `model_config` and validated on rebuild, so a same-name different-
  stack checkpoint is refused rather than loaded into right-shaped wrong layers. The `normalize`
  values were renamed to `train_subject_covered_sample_stats` / `train_split_covered_sample_stats`,
  because `*_global_stats` named the partition and left the fitting set unstated — the one thing D31
  is about. F0 now declares `weight_decay` instead of silently running at 0.0 while F1 declared
  1e-4, and F1's redundant `early_stopping` key is gone.
- Protocol gating closed and the architecture ratified. Eight configs gained `protocol:` blocks, so
  all seven robustness targets are blocked rather than four of them silently dispatchable; readiness
  is now a declaration a test requires instead of something inferred from a missing block. The
  perturbation config also gates itself, because a type and a level list are not an operational
  definition — F3–F5 declare the five missing fields as `null` and refuse as a whole. D23's
  temperature source turned out to be asserted two incompatible ways in the spec; §3.4 now names
  both candidates and settles neither, and F1/F2 fail closed on it. D32 freezes the `cnn1d` layer
  stack, D11 gains the absent-class rule, and D31 is marked reviewed.
- Repository-wide consistency review over every source file, config, spec, and script. It found no
  mismatch between a stated decision value and its implementation across 23 checks, and no dangling
  decision or section reference. Three defects in the F0 training path were fixed: the run config
  silently ignored its declared loss (it now carries and validates the field, so a config asking for
  anything but class-weighted cross-entropy is refused rather than quietly retrained), the
  window-boundary audit was quadratic and is now a vectorised search, and a dead parameter was
  removed. Normalization statistics are pinned to the covered-once fitting set and required inside
  the checkpoint (D31).
- Protocol and rendering repair: G3/G4 now fail closed on within-subject repeat aggregation and the
  synthetic-only *k*=0 step budget; D28's fold-complement development artifacts have explicit build,
  locator, and provenance contracts; robustness dispatch blocks pending sentinels and validates
  aliases; and the language path has tested mask-derived pooling and position helpers. Markdown
  tables and source wrapping were repaired and protected by repository-wide structural tests.
- F0 training path: the window dataset, inverse-frequency class weights (D11, absent classes at
  weight 0), the single-split training loop, and `scripts/prepare_dataset.py`, which converts raw
  `.mat` recordings to the processed format. `docs/api/README.md` records why each module exists and
  which decision it implements rather than restating signatures, and a test asserts every public
  function is documented there.
- F0 model layer: the 1-D CNN encoder and classification head behind `build_model`, the checkpoint
  contract (`{model_state, model_config}`, refused without normalization statistics), channel
  normalization, and the metric set — accuracy, confusion matrices, per-class precision/recall/F1,
  macro-F1, Brier, adaptive-bin ECE, overconfidence error, per-subject reduction, and the paired
  bootstrap interval. `frozen` is enforced as `requires_grad=False` plus a `train()` override that
  keeps a frozen encoder in eval mode.
- G3/G4 adaptation is specified and reconciled (D28–D30). The objective is additive rather than
  averaged, so the replay arms embed the whole real-only objective with its coefficient unchanged
  and add exactly one intervention. Both loss sources are normalised to per-source means because the
  real set spans roughly 97 to 3,300 active-gesture windows across the grid before any permitted
  rest; an unnormalised sum would make the loss scale a function of *k*. Regularisation is L2-SP to
  the F1 head with Adam weight decay zeroed. Adaptation is full-batch and restarts at every *k*. The
  two replay arms encode their conditioning separately, share paired RNG and fixed bank draws, and
  deduplicate both pairs of identical *k*=0 results. D28 explicitly amends D11 for adaptation, and
  D29 propagates through G4 and all three adapted robustness slices. Seen/unseen reporting is mean
  per-class recall. The numeric candidate grid and repeat-domain/aggregation contract stay open.
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
- Checkpoint naming separated by purpose (D7, narrowly amended by D28): `refit.pt` is the final
  downstream artifact; `folds/fold{k}/best.pt` is analysis-only; and
  `development/fold{k}/fixed_budget.pt` is used only for D28 adaptation selection. Spec §5.1 defines
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
started**: NinaPro DB2 Exercise B is downloaded locally, and the data layer (reader, processed
format, windower), normalization, the 1-D CNN encoder and head, the checkpoint contract, the metric
set, class weights, the window dataset, the F0 training loop and `prepare_dataset.py` are all
implemented and tested. **F0 now runs end to end on real NinaPro data** — `make prepare-debug-subjects
&& make debug` trains on subjects 1–2, evaluates on held-out subject 3, and writes a run directory
in about five seconds. It is a smoke test and says so on stdout; no measurement has been produced.
What remains for Milestone 0 is F1's harness: 8-fold cross-validation, D18–D21 selection with D19
escalation, the refit, D10's cross-check, and D23 temperature scaling.

## Done

- Reproducible local env: uv + Python 3.11, PyTorch MPS, `make setup/verify/test/lint` all green,
  `uv.lock` tracked.
- NinaPro DB2 Exercise B acquisition and verification completed for all 40 subjects; repository
  provenance/checksums live in `docs/data/`, while the archives and MAT files remain ignored.
- Experiment specification authored: `docs/experiments/` (shared protocol + generative + language
  arms); every ablation maps 1:1 to a config under
  `configs/experiment/{foundation,generation,language}/`.
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
- G3 calibration-budget and strategy choices frozen (D14, D15 as amended by D28–D30);
  specifications and configs reconciled. The numeric candidate grid and remaining reporting
  choices stay open below.
- Split manifest and verified loader landed (D16).
- F0 vertical slice complete: raw `.mat` → processed `.npz` → windowed → normalized → trained →
  evaluated → logged, on real DB2 data, driven by `make debug` from its own config.

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
   - **G2 gate population and final-test policy.** Fix the development folds/aggregation and
     one-shot final diagnostic, including coverage of the conditional distributions used across
     *k*.
   - **G3 headline extraction.** The unit is fixed, but target accuracy, curve interpolation or
     monotonic treatment, unreachable targets, uncertainty, and subject aggregation are not.
   - **G3 schedule reproducibility.** Fix the PRNG and schedule-index base; selected trials are
     persisted with each run.
   - **G3 calibration-rest ownership.** Rest does not increment *k*, but the runner needs a fixed
     rule for whether a selected gesture trial permits one adjacent rest interval. Inspection is
     complete and assignment is feasible; never silently expose all held-out-subject rest.
   - **G1 embedding operational detail.** D27 fixed the estimator, schedule and population
     vector, but rotation *selection* inherits D14's Pending PRNG convention and the
     class-composition probe needs a target, subject-grouped split, score and threshold before it
     can gate anything.
   - **G3 adaptation candidate grid.** D28 fixed the objective's form, losses, regulariser and
     full-batch rule. The numeric learning-rate / step / L2-SP grid, global-versus-per-*k* step
     scope, separate synthetic-only *k*=0 step assignment, selection metric and aggregation, and
     tie-break are undeclared. A grid declared after results are seen is not a selection protocol.
   - **G0/G1 VAE objective components.** The minimized objective is correctly named
     `negative_elbo`; the reconstruction likelihood/reduction, units, and raw-versus-weighted KL
     reporting remain open.
   - **G2 quality-metric and acceptance contract.** Fix grouped discriminator evaluation, AUC
     orientation and dependence-aware CI, CI/per-class gate behavior, nearest-neighbour semantics,
     and a genuine diversity or coverage diagnostic.
   - **G3/G4 within-subject repeated-run aggregation.** Fix the adaptation-seed and synthetic-draw
     counts/derivations, whether those axes are paired or crossed, and aggregation across the
     persisted schedule/seed/draw runs without duplicating the population reference or treating
     reused evaluation windows as independent observations.
   - **G4 headline ECE aggregation.** Fix subject weighting and whether pooled-prediction ECE is
     headline, supplemental, or omitted.

2. **P1 — F1 harness.** F0 is done and green. Remaining: 8-fold cross-validation over the 32
   training subjects, D18–D21 epoch selection with D19's single escalation, the refit on all 32, D10's
   refit-minus-fold-mean cross-check, and D23 temperature scaling. The loader currently refuses F1's
   config on purpose, so lifting that refusal is the acceptance signal for this item. The trainer must
   honor the checkpoint contract (`{model_state, model_config}`) as an acceptance criterion, not a
   later task, and a *reportable* F1 additionally needs D23's temperature source resolved — the config
   fails closed on it today.
3. **P2 — Reconcile** remaining planning-docs wording with the authoritative spec where it drifts.

## Known open items (not yet scheduled)

- Robustness configs are target-agnostic in schema but **not executable** until the eval logic and
  the arm models exist (`evaluate.py` is a stub).
- G3 robustness expansion across held-out subject × schedule × *k* is declared for all three
  adapted strategies, including adaptation-seed and replay-only synthetic-draw axes. The runner
  must resolve `latest` to an immutable run ID, deduplicate both strategy pairs' shared *k*=0 head
  artifacts while still evaluating each held-out subject, and wait for the Pending repeat-domain
  and within-subject aggregation rules before reportable evaluation.
- Evaluation outputs currently mix primitive metrics, grouped curves, metadata, uncertainty, and
  derived estimands. Define and validate a result schema that separates those roles alongside the
  F0/F1 evaluator, then migrate the G-series configs before their runners are implemented.
- **D23's temperature source is unresolved and now fails closed.** One pooled out-of-fold
  temperature and the median of eight per-fold temperatures are different estimators and cannot both
  be the procedure; the spec asserted both. §3.4 now names both candidates and settles neither, and
  F1/F2 declare `temperature_source: pending_decision` behind a blocking `protocol.status`. F2
  constrains the choice further — it is a random window split with no folds, so it needs its own rule
  under either option. Robustness runs consume the frozen target checkpoint's stored temperature.
- Calibration execution still needs exact Brier reduction/reporting, bootstrap level/count/seed,
  deterministic equal-mass tie handling, and distinct result keys for raw versus scaled ECE.
- Retain the simulation or analysis artifact that justified D23's M=10 choice, and recompute any
  finite-sample ECE detection threshold for each actual evaluation design rather than reusing 0.037.
- The F1 reference **optimizer** is still not frozen: scheduler, initialization and gradient
  handling are undeclared. D32 froze the layer stack, so what remains is the optimizer side plus the
  config-surface question above.
- Processed NPZ recordings load eagerly and are not memory-mappable. The F1 batching/normalization
  path must avoid duplicating the roughly multi-GiB 32-subject signal across workers; `num_workers`
  is 0 as the safe local default until a measured lazy/memory-mapped design exists.
- D10 requires the refit-minus-fold-mean result, but its output key/schema is not yet defined.
- L2–L4 do not yet encode a cross-validation/refit selection contract and request
  `overconfidence_error` (settled as D25) without an evaluation gate; settle the selection contract
  before language-arm execution.
- F3–F5 still need executable perturbation definitions (order relative to normalization,
  noise/channel-mask sampling and seeds, repetitions, and aggregation).
- Governance wording still conflicts: AGENTS calls DECISIONS rows append-only while DECISIONS moves
  approved proposals from Pending to Resolved, and the accepted L2/L3/L4 readout row has no decision
  ID. Reconcile the retention rule and assign stable identifiers without rewriting history.
- The `merge=union` rule assumes self-contained one-line entries, but STATUS is mutable and most
  entries span multiple lines. Concurrent deletion/reordering can retain stale fragments instead of
  conflicting. Decide whether to restore normal merges or restructure project memory before relying
  on union merge as collision protection.
- The G0/G1 objective-component Pending choice is not yet wired into a runner-enforced config
  schema; add the appropriate execution/reportability gate when implementing that runner. G3/G4's
  repeated-run contract is now fail-closed in both generation configs and the robustness registry.
- `head.pooling: last_token` means the last non-padding token: the L-series trainer must gather the
  rightmost unmasked index rather than the final position, assert the row has an unmasked token, and
  pass mask-derived `position_ids` under left-padding (silent failure otherwise).
- The consume-or-refuse rule covers the F0/F1 encoder path only. `scripts/evaluate.py` and the
  unwritten `train_adapter.py` / `make_report.py` still read the keys they happen to want, so the
  generation and language configs can carry keys nothing consumes. Extend the rule to each loader as
  it is written rather than auditing for it later.
- Debug configs deliberately carry no `protocol:` block: their results are never reportable, so a
  reportability gate on them would mean nothing. They are also absent from the robustness registry,
  which is what the declared-readiness test keys on. If a debug config is ever added to the
  registry, it needs a status like every other entry.
- `invalid_shared_k0_aliases` checks only that a `shared_k0_with` target exists, not that the graph
  is acyclic. A self-reference or an A→B→A cycle passes validation and would leave a future runner's
  *k*=0 deduplication unresolvable. No cycle exists in the registry today.
- The interface-only Make targets listed in AGENTS.md reference scripts that are not written yet.
- GitHub Issues/Project sync deferred to Tier-2 (`gh` not installed).
