# Language arm (L-series) — secondary

**The question.** Does mapping temporal signal embeddings into a *frozen* language model's embedding space improve decoding, calibration, or failure reporting beyond a well-trained encoder alone?

**The expected answer is "little, for raw accuracy."** That is not a failure mode. A clean negative, properly isolated by L3 and L4, is publishable, and more interesting than a positive nobody can attribute. This arm is built so a null result stands up as well as a positive one, which holds only while the controls stay airtight.

Read [README.md](README.md) first: §2 (vocabulary), §3 (shared protocol), §4 (the readout rule) are assumed throughout and are not repeated here.

**Prerequisite:** the CUDA F1 run must be complete (D47). Each development fold freezes its matching F1 fold encoder, `checkpoints/baseline_cnn_subject_split/development/fold{k}/fixed_budget.pt`, and every reporting refit freezes `checkpoints/baseline_cnn_subject_split/refit.pt`. L4 reads only those checkpoints' normalization statistics, since it has no encoder input path, and the local L1 smoke test freezes the preliminary MPS encoder instead. F1's reference row is the refit's held-out-subject macro-F1 and calibration, not the cross-validation mean: fold models train on fewer subjects, so measuring an adapter built on the refit encoder against a fold-mean baseline would mix the effect under test with a difference in training-set size (README §5.3).

---

## The comparison this arm exists to make

| Row | Input path | Frozen backbone | Trainable | Config |
|---|---|---|---|---|
| **Reference** | — (encoder only) | — | head | F1 |
| **L2** | soft prefix | Llama 3.2 3B, **pretrained** | projector + head | `adapter_llama3b_subject_split` |
| **L3** | soft prefix | matched-size, **random init** | projector + head | `adapter_random_transformer` |
| **L4** | engineered text | Llama 3.2 3B, **pretrained** | head only | `adapter_text_summary_only` |
| **L6** | — (embedding only) | none | head only (one of two MLPs) | `embedding_mlp_control` |

Three comparisons, each answering a different question:

- **L2 vs. L3** — size, readout, and input path all held constant; the weights are pretrained in one and random in the other. Any claim about **language pretraining** rests on this row. If no difference is detected, report no detected incremental pretraining benefit; do not infer equivalence or that capacity caused the result.
- **L2 vs. L4** — same backbone and readout, but a learned CNN latent delivered as a soft prefix is compared with 72 engineered signal statistics delivered as text. This compares the **whole learned-latent pipeline against the engineered-text pipeline** (D36). Representation and delivery both change, so it cannot isolate the projector or assert that the model receives the same information. A null is no detected pipeline advantage, not equivalence. No additional same-information control is added to v1.

- **L2 vs. L6** — whether the whole LLM route beats a credible conventional decoder of the same frozen embedding (D77). L3 cannot answer this: a frozen random transformer may damage the signal, so L2 > L3 alone does not show that the LLM adds anything over ordinary nonlinear re-heading. L6's head differs from the shared transformer readout by design, so this compares whole decoders, not readouts.

The L2/L3/L4 comparisons are void if their readout differs. See README §4. D79 names L2 vs. L3 and L2 vs. L6 on macro-F1 and scaled ECE as this arm's four primary comparisons; everything else, including L2 vs. L4 and every comparison with F1, is descriptive (see Inference and claims below).

**Repeated runs (D36/D66).** L2, L3, L4 and L6 (D77) each use three reporting indices, 0, 1 and 2, paired by index and separate from the pilot streams. Each repeats the language fitting/selection/refit procedure; the split, CUDA F1 encoder lineage and selected arm LR remain fixed. Use recorded component RNG streams for trainable initialization/data order, and L3's frozen random-backbone draw. Pairing indices alone does not assert identical tensors or updates; D70 specifies which components actually share starting values/order. Compute each seed's metrics separately, average metrics equally within each subject, then apply D26's paired comparison over eight subjects; there are not 24 independent subject-seed observations. Preserve all seed results and show seed spread separately, with no logit ensemble, best-seed selection or outcome-triggered extra seeds. Three is a budget-conscious reference, not proof of a precise estimate of stochastic variation. Initialization, sampling and hyperparameter choices can materially affect benchmarking, so these conditional language repeats must not be described as full-pipeline retraining uncertainty ([Bouthillier et al., 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)). D43/D70 fix the RNG identities and D71/D72 fix layouts/initialization/numerics; D73 fixes batch/accumulation boundaries; runtime implementation remains Pending.

**Calibration sources (D69).** Each arm/reporting seed fits one refit temperature from its own exact-language-budget OOF predictions under D33/D39; the detailed lineage and separate development diagnostic are below. Final calibration reporting keeps scaled ECE primary, raw ECE and raw/scaled Brier alongside (README §3.4). The source is now specified, not an available fitted artifact: `evaluation.ece.temperature_source` names the matching refit's `model_config.probability_calibration`, and the `evaluation.temperature_fit` declaration is ready, but the overall implementation gate blocks the still-unimplemented reporting procedure. No F1 temperature is inherited.

**Common-budget aggregation (D49; owner-approved).** Select the language budget at the peak of the **subject-equal mean cross-validation curve**, not the median of the folds' individual peak times. At a shared nominal checkpoint time, compute 18-class macro-F1 separately for each query subject, with labels 0–17 and zero for an undefined class F1; average subjects equally within folds and folds equally across D16's eight four-subject groups. This gives each development subject weight 1/32. Aggregate the curves first, then locate the unsmoothed peak on D63's fixed checkpoint grid with D61's earliest-eligible exact tie-break. Do not smooth the reported checkpoint metrics or replace the common-budget checkpoint by a fold's individual peak. F1 retains D18's ceiling-median rule; D49 does not amend it.

**Equal-epoch budget transfer and update reporting (D50).** Epochs are the controlling language budget: one epoch is one pass through that fit's eligible training windows. Transfer the selected nominal epoch budget B from the complement-28 fold trajectories to the fresh all-32 refit as the same B epochs, not the same optimizer-step count. This preserves training passes per example rather than total optimizer exposure. The refit generally makes more updates: approximately 32/28 = 1.143 times as many when subject window counts are similar, but that ratio is explanatory, not a measured update count. Report the nominal epoch target, actual consumed training-window count/epoch position and cumulative completed optimizer updates at every selection checkpoint and retained development/refit artifact, with training-window count, batch size, gradient accumulation and final-partial-batch/accumulation handling. Count completed optimizer updates, not microbatches or an estimate based on subject count. D63 fixes fractional-target mapping below; D73 fixes batch/accumulation execution below; the runner remains unimplemented. F1 is unchanged.

**Optimizer identity (D51).** Use Adam, not AdamW, for all three reportable language arms, with coupled L2 rather than decoupled weight decay. This is an explicit reference choice, not evidence that Adam outperforms AdamW here. It does not automatically inherit F1's numeric optimizer settings. D52–D54 separately fix the coefficient, groups and Adam mechanics below; D55 fixes the initial LR grid and selection policy. D71/D72 fix initialization/precision/gradient/failure rules; D73 fixes the remaining batch/accumulation rules below; the runner remains unimplemented. The CNN encoder and language backbone remain frozen; the optimizer operates only on the declared trainable projector/head, or head alone for L4.

**Coupled-L2 coefficient (D52).** Fix `training.weight_decay` at `1e-4` for L2/L3/L4. This is Adam's coupled-L2 coefficient, not an equivalent AdamW decay value or a demonstrated optimum. Keep it fixed rather than selecting its value from results. D53 specifies which parameters receive it; never regularize the frozen CNN or language backbone.

**Weight-matrix-only regularization (D53).** Apply the `1e-4` coefficient only to trainable weight matrices in the projector/head; L4 has only the head. Trainable biases and normalization parameters use `weight_decay: 0.0`. Identify normalization parameters from their owning module before inspecting tensor shape: a normalization scale can itself be multidimensional, so `ndim >= 2` alone is not the grouping rule. Frozen parameters never enter either optimizer group. Verify that every trainable parameter is assigned exactly once, by parameter identity, with no overlap; block unknown roles rather than guessing. Persist group membership, parameter names/shapes and coefficients. Implement the penalty once through Adam, with no duplicate manual L2 term. F1's existing all-trainable-parameter rule remains unchanged.

**Adam mechanics and constant learning rate (D54).** Use betas `(0.9, 0.999)`, epsilon `1e-8` and `amsgrad: false` across L2/L3/L4. Keep the learning rate constant throughout each fit: no warmup, decay schedule or validation-driven scheduler. D54 fixes the recipe; D55 separately fixes the initial LR grid and selection policy below. A constant LR avoids horizon-dependent scheduling, but the full training-prefix and validation-isolation checks still apply before checkpoint reuse. These are explicit reference settings, not a measured optimum. F1 is unchanged.

**Initial learning-rate grid and arm-level selection (D55).** L2/L3/L4 share the initial candidate grid `{3e-5, 1e-4, 3e-4}`. Select LR separately for each arm using development data only, then freeze that arm's selected LR across its reporting seeds and matching refits; do not reselect LR per reporting seed or force equal winners across arms. Matching the candidate set does not require matching the winning value, and the grid is a project reference choice, not a literature-derived optimum. The existing `training.learning_rate: 0.0001` is an unselected scaffold and must never serve as a fallback when selection is unresolved. Later runs must resolve and record the selected value and its development-selection provenance before using it. D56–D65 fix pilot size, fold IDs, separation from reporting, score aggregation, both tie conventions, pilot seed index, checkpoint evaluation, shared maximum horizon and bounded grid expansion below; D70 fixes exact component keys/sharing below; D71/D72 fix numerical execution; D73 fixes batch/accumulation boundaries; runtime implementation remains Pending. Only D65's predeclared one-expansion rule is allowed; the arm LR is frozen after that pilot selection lifecycle completes. `training.learning_rate_selection` is declared ready, while the overall implementation gate stays closed.

**LR pilot size (D56).** Screen the initial grid on the same two fixed D16 development folds across arms and LR candidates, with one pilot seed, not a repeat sweep. Those folds contain eight distinct query subjects, with the matching 28-subject complement used for each fit. This is a limited development screen, not a replacement for the later eight-fold procedure or evidence of a broadly optimal LR. Three arms × three initial LRs × two folds × one pilot seed gives 18 language fits: projector/head fits for L2/L3, head-only fits for L4. That is a fit count, not a runtime benchmark; it excludes the F1 encoder procedure, reporting fits and D65's conditional additional-candidate fits. The pilot still uses one seed; D66 separately fixes three reporting seeds. D57–D65 fix fold IDs, the separate reporting lifecycle, score aggregation, both tie conventions, pilot seed index, checkpoint evaluation, shared maximum horizon and bounded grid expansion below. D70 fixes component identities/sharing; D71/D72 fix numerical execution; D73 fixes batch/accumulation boundaries; runtime implementation remains Pending. The pilot declaration is ready; the overall implementation gate remains closed.

**LR pilot fold IDs (D57).** Use the frozen D16 manifest's zero-based folds **0 and 4**, shared across L2/L3/L4 and all LR candidates. Their query subjects are `[1, 11, 21, 31]` and `[6, 16, 26, 36]`, respectively; each fit uses the other 28 development subjects for training and the matching fold-local F1 encoder/statistics, with L4 reading statistics but having no encoder input path. Resolve membership from `dataset.split_manifest`, not a new hand-written subject split. These are predeclared folds, not a performance-selected subset or a claim of demographic representativeness. No final test subject enters the pilot. F1, the later eight-fold procedure, the 18-fit initial pilot count and all remaining gates are unchanged.

**Separate pilot and reporting lifecycle (D58).** Give the pilot dedicated RNG streams separate from reporting, and start reporting fits freshly from their reporting streams. Never promote a winning pilot checkpoint to a reporting fold, resume or warm-start reporting from pilot projector/head parameters, buffers, optimizer state or RNG progress, or count a pilot fit as a reporting repeat. L3's reporting random-backbone initialization likewise uses reporting-phase RNG, not the pilot's random-backbone artifact; D68 fixes sharing within a reporting seed's folds/refit. Keep pilot artifacts and predictions for selection provenance/audit, not as reporting-model predictions or a reporting calibrator source. Carry forward the selected arm LR and its selection provenance. This separation does not prohibit shared frozen F1 checkpoints/statistics, fixed pretrained backbones or correctly identity-bound deterministic input caches, and does not change F1 D48 or valid future-approved common-budget checkpoint reuse within reporting. The initial 18 pilot fits are additional to reporting fits. This is random-stream/run-state separation, not a new subject split or unbiased nested validation: the development subjects are reused. D62 fixes pilot seed index 0; D70 fixes phase/component keys and sharing; D71/D72 fix initialization/numerical execution; D73 fixes batch/accumulation boundaries; D66 fixes three reporting indices and D67/D69 fix the model/temperature lineage; overall execution remains blocked by the explicit implementation gate.

**LR pilot candidate scoring (D59).** For each arm and candidate LR, compute each query subject's macro-F1 at every common nominal checkpoint time using labels `0 … dataset.num_classes−1` (currently 0–17, including rest) and D21's undefined-class F1 = 0.0. Average the four subject scores equally within each pilot fold, then average the two fold curves equally, giving each of the eight query subjects weight 1/8. Score the LR by the unsmoothed peak of that aggregated curve on D63's shared grid, and rank LRs separately for each arm. Aggregate curves before locating the peak: do not average individual fold peaks, or pool windows across subjects before computing F1. The optimized pilot score is a selection statistic, not independent validation or final performance. Its peak time is not transferred as the reporting/refit budget; D58's fresh reporting procedure retains its own selection contract. D60/D61 resolve candidate-LR and peak-time ties below. D64 fixes the common maximum horizon and D65 the bounded grid expansion; D67–D73 settle the remaining lineage/execution declarations. The scoring declaration is ready, but the overall implementation gate stays closed.

**Exact candidate-LR score ties (D60).** Compare each arm's candidate scores before display rounding, using exact equality and no near-tie tolerance band. Among LRs whose score exactly equals the maximum candidate score, choose `1e-4` if it is in that tied-best set; otherwise choose the minimum LR in that set. A uniquely better candidate always wins, even if its advantage is small, and `1e-4` is never preferred when its score is below the maximum. Do not use rounded scores or approximate-equality defaults such as a nonzero relative tolerance. This is a deterministic tie convention, not a claim that `1e-4` is optimal or a statistical equivalence test. It settles ties between candidate LRs; D61 separately fixes ties between checkpoint times below. The tie-break and parent pilot/scoring/selection declarations are ready; the overall implementation gate stays closed. F1 is unchanged.

**Earliest eligible peak-time ties (D61).** In both pilot peak-time diagnostics and eight-fold language budget selection, choose the earliest eligible checkpoint whose unrounded aggregated selection-curve score exactly equals the maximum eligible score. Do not round scores or use approximate equality. D63 supplies the positive-time grid and unsmoothed curve below; time zero remains ineligible. D64 separately fixes the maximum horizon and reporting-boundary safeguard. None of these decisions transfers the pilot peak time into reporting or changes F1. The single tie policy lives at `training.selection.peak_time_tie_break`, referenced by pilot scoring so the two uses cannot drift; the selection/scoring declarations are ready, but the overall implementation gate stays closed.

**Pilot seed index 0 (D62).** Label the one pilot replicate with index `0`, under D43's zero-based convention and D58's separate pilot phase. This is an index, not root seed 0 or a directly supplied native RNG seed: `experiment.seed` remains 42, and D70 fixes its phase/component derivation keys and sharing below. A later reporting seed index 0 must still have a distinct reporting-phase identity, not reuse the pilot stream. D66/D68 separately fix three reporting indices and L3's within-seed sharing. D71/D72 fix initialization/numerical execution; D73 fixes batch/accumulation boundaries; overall execution remains implementation-blocked.

**Checkpoint grid and unsmoothed selection (D63).** Use the same nominal epoch grid across L2/L3/L4, LR candidates, seeds and pilot/reporting: every positive multiple of 0.25 epoch within the accepted horizon, plus 0.0625 (1/16) and 0.125 (1/8) epoch in the first epoch only. The two early points occur only once, not in every epoch. Evaluate every query subject's complete eligible validation window set at each checkpoint, not a changing subset. Select from the unsmoothed subject-equal mean macro-F1 curve under D61; all positive grid checkpoints are eligible. Log time zero diagnostically but never select an untrained time-zero model. Store one `training.selection.checkpoint_evaluation` policy and reference it from pilot scoring, not two independently editable grids.

**Completed-update mapping (D63).** For a fit with N eligible training windows, a nominal target t means cumulative exposure t × N, counting windows consumed across successive passes. Evaluate at the first completed optimizer update reaching that exposure; never evaluate mid-accumulation, force a flush or insert an update just to meet a checkpoint. Aggregate folds by nominal t while retaining each fit's actual exposure and update counts. If targets map to the same update, evaluate the unchanged model once and record all mapped nominal targets. Record nominal t, actual cumulative consumed training windows, actual epoch position and cumulative completed optimizer updates. Transfer selected nominal B to the all-32 refit using its own N and the same mapping, not a fold's rounded actual epoch position or update count. D73 fixes final-partial-batch, accumulation-tail, loss-normalization and terminal-horizon execution below; D72 forbids skipped updates. D73's natural epoch-end update guarantees an H=10 endpoint without training past the cap; no checkpoint-induced flush or extra endpoint is permitted. The grid does not silently add an off-grid horizon endpoint or round B to a whole epoch. Disabling temporal smoothing here changes neither signal preprocessing nor temperature scaling.

**Tradeoff and cost.** No temporal smoothing preserves early selection opportunities and avoids trailing-window boundary/weighting choices; a full one-epoch trailing mean would exclude every checkpoint before epoch 1. It does not remove selection noise. Maximizing a finite-development criterion over more checkpoints/LRs can overfit that criterion; the final test remains untouched, but the selected maximum is not an unbiased performance estimate ([Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)). The grid is a predeclared reference choice, not a measured convergence optimum. D64's ten-epoch horizon entails 42 nominal positive-time validation checkpoints per fit (40 quarter-epoch targets plus two early targets), not 42 fits. The former five-epoch proposal would have had 22; these counts are not total-runtime multipliers, and coincident update boundaries can reduce actual evaluations. Measure full-validation and checkpoint-retention costs before execution. Any proposed timing-driven grid revision requires an explicit contract update before validation-score inspection and must apply consistently across arms; no adaptive checkpoint skipping.

**Shared ten-epoch maximum (D64).** Train every pilot arm/LR candidate and every reporting selection fold through the same maximum H=10 epochs. Select each arm's LR from its complete ten-epoch pilot curves, then freeze it across reporting seeds under D55; select the reporting budget B from the approved aggregated curve, allowing any earlier positive D63 checkpoint. The all-32 refit transfers selected nominal B under D50/D63, not H=10. If an aggregated reporting selection curve chooses nominal B=10 under D61's earliest exact-tie rule, D87 keeps B=10 for that arm/seed, flags it and requires the owner's acknowledgement before the accepted manifest is finalized; D64's original stop is retired (see [Boundary selections and owner acknowledgement](#boundary-selections-and-owner-acknowledgement-d87d88)). Check D66/D67's complete reporting domain, now four pipelines (L2, L3, L4 and D77's L6) × three seeds, not one fold at a time. An individual fold peaking at ten raises no flag; a frozen pilot selection peaking there raises only D87's pre-reporting early warning, and pilot peak times never become reporting budgets. An exact plateau also attained earlier selects that earlier time and raises no flag. There is no automatic extension, five-to-ten escalation or horizon-triggered LR reranking. D65 separately defines a bounded LR-grid expansion completed during the pilot, without changing this horizon.

Ten is a declared resource cap, not a literature-derived optimum or proof of convergence. An interior observed maximum does not prove training has converged either. Using one horizon avoids ranking LRs under five epochs and later accepting reporting selections under ten; it costs more selection training than a five-epoch run but does not force longer refits. Measure throughput before execution; if the declared cap is unaffordable, request a protocol change rather than silently shortening it. F1's D18/D19 horizon and scratch-escalation rules are unchanged.

**One shared LR-grid expansion (D65).** Complete the initial pilot on {3e-5, 1e-4, 3e-4}, select each arm's winner using D59/D60, then check the selected LR against the initial minimum and maximum. If any arm selects an endpoint, expand the common grid for all three arms once, adding both 1e-5 and 1e-3: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}. Use the same D57 folds, D62 pilot index, D64 ten-epoch maximum, D63 checkpoint policy and D70 component-stream rules for the added candidates. Each added LR starts a fresh fit; never continue a model trained at another LR. Reuse completed original-candidate pilot results only when their fitting protocol and artifact identities remain identical and verified. Rerank each arm across all five candidates, then freeze its selected LR across reporting seeds. An initial interior winner does not exempt its arm from the shared expansion triggered by another arm.

Apply D60's tie-break before both endpoint checks: an endpoint merely tied with a preferred interior winner triggers neither expansion nor a D88 flag. If any selected winner after expansion is again an outer endpoint (1e-5 or 1e-3), D88 keeps that LR, flags it and adds it to the pre-reporting owner review; the former stop is retired. There is no second automatic expansion, per-arm expansion counter, or reporting/test-triggered LR search. This is a resource-bounded search convention, not proof of an interior/global optimum. The initial 18 pilot fits gain 3 arms × 2 added LRs × 2 folds × 1 pilot replicate = 12 fits when triggered, for 30 maximum; existing verified pilot results are not retrained merely to expand the candidate list. These are fit counts, not runtime estimates. D58 still prohibits promoting pilot state into reporting. D71/D72 fix initialization/precision/gradient/failure behavior, including stopping the pilot on numerical failure. D73 fixes batch/accumulation boundaries; do not invent a score or silently drop/reseed a failed candidate.

**Reporting CV and refit lineage (D67).** For each arm/reporting seed, use all eight four-subject D16 query groups and their complement-28 training sets. Each fold uses its matching frozen CUDA F1 development encoder, training-only normalization and class weights. L4 consumes statistics/weights without a CNN input path. All six recording repetitions follow their subject; do not import G3's [1, 4] subset. Initialize trainable modules freshly per fit under D71's approved initialization rules (head only for L4). Select a separate nominal language budget B for each arm/seed under D49/D61/D63/D64; this is not the F1 encoder's D18/D19 budget. After every reporting budget freezes (D64/D87), retain each fold's exact-B language parameters/buffers with frozen-parent identity, source run/fold/budget/digests and training RNG provenance. Require immutable snapshots and horizon/validation-independent training prefixes; missing/incompatible snapshots block rather than using own-best or nearest checkpoints. This avoids eight additional fits. Freshly initialize the all-32 trainable modules and optimizer on the CUDA F1 refit parent and its all-32 normalization/class weights, transferring nominal B under D50/D63. D70 fixes RNG identities below; D71/D72 fix initialization/numerical/failure rules; D73 fixes batch/accumulation boundaries; runtime implementation remains gated.

**L3 backbone lifecycle (D68).** Use one immutable frozen random transformer per reporting seed, shared across that seed's eight folds and refit, with state identity verified. Different reporting seeds use different draws; never inherit the pilot backbone. Match L2 architecture/dtype and retain architecture revision, dtype, model-state digest and reporting seed index. Fresh trainable modules do not mean resampling that frozen backbone within a seed. L2/L4 retain their fixed pretrained backbone. D70 requires immutable revisions resolved before the pilot and fixes stream identities; D71/D72 fix initialization distributions and numerical precision below, without inheriting F1 defaults.

**Refit temperature lineage (D69).** For each arm/seed, pool the 32 query subjects' OOF logits from its eight exact-B language models; each subject's model and F1 parent excluded that subject from training. Include recording repetitions 1–6; D81 excludes the repetition-0 rest segment from fitting. Fit one T using D33's ordinary NLL, equal subject weight, equal recording-repetition weight within subject and equal window weight within repetition, with D39's unchanged solver. Recording repetitions are not reporting seeds. Attach T and fitting provenance only to the matching arm/seed all-32 refit; never pool seeds, average temperatures, use pilot predictions, inherit F1 T or fit on final subjects. Retain arm/seed, language budget, language/F1 checkpoint and frozen-backbone identities, query subject/repetition identities, logits and config/manifest digests. Transfer from complement-28 models to the all-32 refit remains an approximation. Scalar temperature fitting follows [Guo et al., §4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf); that paper does not establish this project's OOF/refit transfer as a guarantee.

**Scaled development diagnostics (D69).** Never score pooled T on its own fitting logits as held-out calibration. At each frozen exact-B fold checkpoint, fit a separate T on three query subjects and score only the fourth, rotating all four without retraining or reselecting B. Inherit only D33 objective/weighting and D39 solver rules, not the pooled source/refit role. Recompute bins per subject/probability version, retain model/fitting/scoring identities and T, and preserve each subject-by-seed metric for D66's seed mean within subject before subject aggregation/inference. Equal-weight fold summaries are descriptive, not substitutes for those subject values. Four fits per fold give 32 scalar fits per arm/seed; none feeds refit T. LR/B used these subjects, so these are conditional development diagnostics, not unbiased nested CV. All refit/diagnostic temperatures wait for every reporting budget to freeze and D87's boundary flags to be recorded; never drop a boundary-selected seed.

### RNG identities and sharing (D70)

Use D43's `temporallens_rng_v1` through `utils.randomness.derive_seed`, keeping root `experiment.seed: 42`. The unsigned 64-bit derived seed initializes the appropriate native PyTorch generator/RNG context directly. Do not use `assignment_rng` or label training provenance with `randomization_provenance`'s PCG64 backend; protocol assignments remain PCG64, F1 retains D38's direct seed and the bootstrap retains D26's seed. Define the policy once at `training.randomness`; pilot and reporting declarations reference it.

Use scope `language_pilot` for pilot index 0 and `language_reporting` for reporting indices 0–2. Let `fit_index` be the D16 fold index 0–7, or 8 for the all-32 refit (a fitting-unit label, not a ninth fold); pilot fits use only 0 and 4. Epoch indices start at zero. The arm mapping is `l2: 2, l3: 3, l4: 4`; include arm only where listed below. Every key also includes its phase scope, root seed and distinct purpose.

| Purpose | Named integer indices | Sharing |
|---|---|---|
| `data_order` | `seed_index, fit_index, epoch_index` | Same permutation across arms and LR candidates for that fit/epoch |
| `head_init` | `seed_index, fit_index` | Same initial values across L2/L3/L4 and LR candidates; different fits have different draws |
| `projector_init` | `seed_index, fit_index` | Same initial values across L2/L3 and LR candidates; not used by L4 |
| `l3_backbone_init` | `seed_index` | One immutable draw per phase/seed across folds, candidates and reporting refit |
| `training_stochasticity` | `seed_index, fit_index, arm_id` | Same starting training/dropout stream across LR candidates within an arm, separate across arms |
| `loader_base_seed` | `seed_index, fit_index, epoch_index` | Dedicated worker-generator stream, separate from the sampler |
| `validation_loader` | `seed_index, fit_index, arm_id` | Separate local generator, recreated at the same starting state for every full validation pass |

Shared initialization means copying matching initial tensors into independent modules without shared storage, never sharing live parameters or trained state. Every fit/candidate has fresh Adam state, including D65's added candidates. Constructors, frozen-model loading and validation must not advance training streams. Reconstruct each keyed stream freshly for each corresponding fit/candidate; never pass one advancing mutable generator between candidates. D71/D72 specify the initialization distributions and precision below.

Create the same complete eligible window pool across arms, ordered by subject ID then recording start sample, with source recording/window-definition identities recorded. L4 transforms the values, not membership/order. No arm-specific filtering, capping, resampling or length bucketing; do not silently drop processing/tokenization failures. Input transforms are deterministic. Use separate sampler and loader generators, FIFO delivery (`in_order: true`), and nonpersistent workers reseeded from each epoch's loader stream. The derived loader seed seeds its generator; PyTorch subsequently draws worker base seeds from that generator. Seed worker Python and legacy NumPy RNGs with `torch.initial_seed() % 2**32`, without truncating the main PyTorch seed. This follows [PyTorch's worker-seeding guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html#dataloader) and [DataLoader's generator/order contract](https://docs.pytorch.org/docs/2.15/data.html#randomness-in-multi-process-data-loading).

Validation uses complete deterministically ordered query windows and its own reset local loader generator, preserving/restoring training CPU and relevant-device RNG states and leaving training buffers unchanged. Do not add checkpoint time to its key: deterministic validation needs no changing random input. Scalar temperature fitting is deterministic and gets no new stream.

Exclude LR, candidate position/grid size/search round, maximum horizon H, selected budget B and full config/logging digests from seed keys. Adding LR candidates must not alter existing candidates' keys. This is RNG identity, not fitted-artifact identity: actual LR and full data/model/config/artifact identities remain mandatory run provenance and D65 reuse checks.

Resolve model, tokenizer and architecture-config references to immutable revisions before the pilot and hold the applicable identities fixed across arms/phases; this is a resolution/verification requirement, not authorization to download a model or invent revision values now. Persist exact keys/native seeds/backend/device/library versions, canonical window-pool identity, epoch-order digests, initialization/backbone state digests and full fitting identity. Resume additionally needs the relevant RNG states and data-order cursor. This pairing can reduce incidental comparison noise but does not guarantee variance reduction or semantic equivalence of identically initialized heads. Seeds also do not guarantee cross-device/version equality ([PyTorch reproducibility limits](https://docs.pytorch.org/docs/2.15/notes/randomness.html)). D71/D72 resolve initialization/numerical/failure rules below and D73 resolves batch/accumulation boundaries; the overall implementation gate remains closed. These are declarations, not an implemented training lifecycle.

### Network layouts and initialization (D71)

The reportable L2/L3 projector and shared L2/L3/L4 head now have explicit layouts. Let D be the hidden width read from the pinned transformer configuration; H continues to denote the ten-epoch selection horizon, not a network width.

| Module | Ordered operations | Output |
|---|---|---|
| Projector, L2/L3 only | `Linear(256, 256) → GELU → Linear(256, 8D)` | Reshape the flat output into eight D-dimensional soft-prefix vectors |
| Head, all three arms | `Linear(D, 256) → GELU → Dropout(0.1) → Linear(256, 18)` | Eighteen unnormalized class logits |

Both linear layers in each module have biases. Add no normalization or output activation; the projector has no dropout or extra output scaling. These are compact reference architectures, not tuned optima. For D=3072 the projector has 6,381,824 trainable parameters including biases; this is an architecture count, not a measured runtime or evidence that the width is optimal. L4 still has no projector or encoder input path; its comparison remains a whole-pipeline comparison under D36.

Initialize each trainable linear layer's weights and biases in FP32 from `Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))`, the explicit [PyTorch Linear initialization rule](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html). This is not a special ReLU-gain initializer, Xavier initialization or zero-bias policy. Use D70's separate component streams, cloning matching initial values into independent modules; fresh fitting state does not authorize a different draw for each LR candidate. Resolve and record the initialization device before the pilot and hold it fixed for paired draws, with backend/library identity retained. No trainable normalization is added by this architecture; D53's ownership/group checks remain binding.

For L3 only, construct the matched pinned Transformers architecture from its configuration, never pretrained weights, using its native initialization in FP32 before casting the frozen parameters to BF16. Preserve special rules for normalization parameters, padding embedding rows, tied tensors and model-defined buffers; do not Gaussian-randomize every tensor. The inspected implementation uses normal weight draws controlled by `initializer_range`, normalization scales of one and appropriate zero biases/padding rows; the pinned executed source/config and resulting state digest, not prose about an unrelated library version, define the realization. Direct BF16 sampling is not interchangeable with FP32 sampling followed by casting. Tensor-wise construction may avoid a second full-model copy only if it preserves the declared initialization/draw identity. Never reinitialize L2/L4's pretrained weights. D68/D70's one frozen backbone per phase/seed and immutable sharing rules remain unchanged.

### Numerical execution and failure behavior (D72)

The same declared recipe applies to pilot candidates, reporting selection fits and refits. It is fixed before scores are inspected, not adapted to whichever arm fails or appears weaker. `training.numerical_execution` carries this declaration; actual construction, precision checks and preflight are still future runner work.

| Component | Numerical contract |
|---|---|
| Frozen F1 encoder and preprocessing | Preserve and verify the matching parent's FP32 inference contract; never cast it through an outer BF16 stack conversion |
| Frozen transformer parameters and floating-point inputs, all three arms | BF16 without quantization; retain the pinned model's native internal promotions/reductions |
| Trainable projector/head | FP32 parameter storage, actual forward computation, gradients and Adam moments |
| Logits and classification loss | FP32 |
| Temperature fitting | Existing D39 float64 execution, unchanged |

Cast the soft prefix from FP32 to BF16 at transformer entry without detaching it; cast the pooled transformer state to FP32 before the head. Frozen parameters still have `requires_grad=False` and `eval()`, but L2/L3 require autograd through the transformer to the projector: enclosing that path in `no_grad()` would break training. L4's fixed input path has no trainable projector. Use explicit dtype boundaries with no global autocast and no gradient scaling. FP32 storage alone would not ensure FP32 operations inside an autocast region ([PyTorch precision behavior](https://docs.pytorch.org/docs/2.15/amp.html)). BF16 applies to floating-point embeddings, not token/position IDs or mask construction; retain valid integer/binary control-tensor semantics. BF16 storage does not assert that every transformer operation or reduction executes in BF16.

Use no gradient clipping; log the accumulated trainable-gradient norm before each update. Require expected trainable gradients to exist and be finite; a finite zero is not automatically a failure. Check finite inputs, soft prefixes where present, pooled features, logits, loss and accumulated gradients before updating, then parameters and Adam state after updating and before saving. This does not prohibit intentional masked-attention infinities in masks. A numerical failure stops the affected phase for a recorded investigation/repair; a failed pilot candidate stops the pilot rather than receiving zero, being silently discarded or triggering a replacement draw. Never skip batches/updates, drop a reporting seed, replace NaNs or silently change LR, dtype, batch size, backend or seed. An interruption may resume from complete saved state under D70; numerical failure does not authorize repeated retries until a seed succeeds. A valid but poor finite score is not a numerical failure.

Require native CUDA BF16 support, rejecting emulation and silent FP16/CPU/4-bit fallbacks. Enable strict deterministic algorithms with errors rather than warnings; disable cuDNN benchmarking and require deterministic convolution. Language FP32 matrix operations use IEEE FP32 rather than TF32; preserve the frozen F1 parent's recorded inference settings separately, blocking incompatible consumption instead of changing F1. Resolve any required CUDA workspace configuration before CUDA initialization. Before the pilot, without validation-score inspection, verify and pin one common compatible attention implementation and actual kernel backend for all three arms, including L2/L3's required backward path. A wrapper name such as `sdpa` alone does not identify the executed kernel. Record actual hardware, software, precision/backend flags and initialization identities; no silent mid-run backend fallback. Unsupported hardware/operators block. Strict operation selection can cost performance and does not promise equality across devices/releases ([PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)); newer documentation is not proof that the pinned execution version supports a particular fused backend.

D73 below resolves the remaining batch/accumulation boundaries; D72's no-skipped-update rule is unchanged. Numerical declarations are not an implemented trainer or CUDA preflight; overall reportability stays explicitly implementation-blocked.

### Batch/accumulation boundaries and loss normalization (D73)

Use the selected physical microbatch size and selected accumulation count from D78/D86, with 4×8 as the reference layout. Their product is 32 windows, the nominal full update, not the size of every update. Set `drop_last: false`, retaining any final partial microbatch smaller than the selected microbatch size. Update after the selected accumulation count of microbatches or at the natural end of an epoch, whichever comes first; an epoch's remaining group receives one update using its actual size. Clear accumulated gradients/counters afterward, without resetting Adam moments, model state or training RNG. Do not carry gradients across epochs. These boundaries are independent of checkpoint times, selected B and announced maximum horizon, so epoch-end flushing does not violate D63's training-prefix rule.

For one actual optimizer group G, let CE_i be the unweighted hard-label cross-entropy and w[y_i] the fixed matching-fold/refit training class weight. Optimize the class-weighted mean:

```text
loss_G = sum(i in G, w[y_i] * CE_i) / sum(i in G, w[y_i])
```

Precompute the common denominator from that exact group's training labels and weights before backward passes. Each microbatch contributes its weighted loss sum divided by this same denominator, accumulating gradients without retaining every microbatch graph. The denominator must be finite and positive; failure stops the phase under D72, never skips a group. Do not average microbatch means, divide by the accumulation count again, divide by a nominal 32, or rescale LR/decay for tails. Apply Adam's coupled-L2 term once per actual optimizer update, through the optimizer, not once per microbatch. This is [PyTorch's weighted-mean cross-entropy for class-index targets](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html), applied to the combined update group. It matches that objective mathematically, not necessarily a physical batch-32 execution bit-for-bit: dropout draws, padding kernels and floating-point summation can differ.

For N eligible windows, a complete epoch has ceil(N/32) updates. An illustrative 35-window epoch has one 32-window group and one 3-window group, not an unnormalized or accumulation-diluted tail. A small tail still receives a full Adam update and can be noisier or have greater per-window influence than a full group; the paired epoch shuffle prevents the same fixed windows always owning that position. This is the declared actual-group recipe, not an exact full-epoch gradient or a convergence guarantee. Existing segmentation/window-boundary exclusions, G3 rest ownership and evaluation metric aggregation are unchanged.

At nominal checkpoint t, use D63's first natural completed update reaching tN; validation never flushes an incomplete group. A refit transfers nominal B to its own N and stops at the first such natural update reaching BN, without truncating a microbatch/group or creating a B-triggered update. Actual exposure may exceed BN by strictly less than 32 windows, and never beyond ceil(B) full passes. The hard H=10 selection maximum ends at exactly 10N windows, including the tenth epoch's natural tail update; no training past H or extra terminal update. D87's reporting-boundary flag continues to test nominal B=10, not rounded actual exposure.

Record nominal budget, actual forward/backward window exposure and epoch position, completed optimizer updates, actual windows/microbatches and class-weight denominator per update, plus tail handling. Prefetch and label lookahead are not training exposure. For interruption recovery preserve complete accumulation/RNG/cursor state, or resume from the last fully saved completed-update boundary; never silently lose an unfinished group.

D73 settled the last owner choice listed at the time; D74–D83 then reopened fold-test calibration. The owner has now approved D84–D86, settling that choice and the campaign/profiling follow-ups, and D87–D88, replacing the three post-selection stops with flags the owner must acknowledge. The configs therefore retain `protocol.status: blocked_pending_implementation` and `blocked_on: [protocol.implementation_status]`. No language network/selector/trainer, CUDA preflight, selected LR/B, fitted temperature or reportable artifact is implemented or produced by these declarations. Remove the implementation gate only when the later runner consumes and verifies the complete contract.

### Input assembly (D74)

Every transformer arm (L2, L3, L4) feeds the frozen model one sequence in a fixed order, recorded in `input_assembly`:

1. the tokenizer's begin-of-text token, once, at position 0;
2. the content: the projector's eight soft-prefix vectors for L2/L3, or the rendered feature text for L4;
3. the fixed suffix `"\nGesture:"`.

Tokenize the suffix, and L4's content, standalone without special tokens, then concatenate the ids, so the suffix ids are identical in every arm. The head reads the decoder's final normalized hidden state at the last suffix token, so all three arms classify from the same literal token. The language-model head is not used. Use no chat template (this is the base model, not an instruction-tuned one) and never truncate.

There is no padding. L2/L3 sequences all have the same length, and L4's representation is defined by one unpadded forward pass per window, so a cached L4 feature never depends on which windows shared its batch. The combined attention mask and explicit position ids still derive from the mask (README §4) as a defensive invariant. Record the tokenizer revision, template identity, suffix token ids and sequence token counts.

Begin-of-text comes first because the model never saw anything before it during pretraining; prepending the prefix to the tokenizer's output would have placed the soft prefix ahead of it. The suffix wording is a frozen reference choice, not a tuned prompt. L4's prompt is about 200–250 tokens and L2/L3's about a dozen, so the readout token sees different context lengths; that is part of D36's whole-pipeline comparison.

### Artifacts and parent references (D75/D76/D85)

All four reportable language pipelines share one immutable campaign rooted at `checkpoints/language_campaigns/{campaign_id}/` (D85). Paths below are relative to that root; the experiment directory retains D76's phase/seed/fold structure. F1's canonical paths do not move.

| Artifact | Path |
|---|---|
| Frozen campaign definition | `campaign_definition.json` |
| Accepted campaign manifest | `accepted_manifest.json` |
| Reporting refit, per arm and seed | `<experiment>/reporting/seed{s}/refit.pt` |
| Exact-B development snapshot, per arm, seed and fold | `<experiment>/reporting/seed{s}/development/fold{k}/fixed_budget.pt` |
| Fold-test calibrator (D84) | `<experiment>/reporting/seed{s}/development/fold{k}/fold_test_temperature.json` |
| Pilot candidate, audit only (D58) | `<experiment>/pilot/{candidate}/fold{k}/` |
| L3 frozen random backbone (D75) | `adapter_random_transformer/{phase}/seed{s}/backbone.pt` |

Debug and preliminary runs (L0, L1, and any local run on the preliminary encoder) never write these paths. A language checkpoint's `model_state` holds only its trainable modules. Its `model_config` references the frozen parents by path and SHA-256 (the F1 encoder checkpoint, plus the pinned transformer revision or the L3 backbone artifact), copies the parent's normalization statistics and records the D74 template. Parent locators name their base: campaign root for bundled artifacts, repository root for unchanged F1 paths, or an explicit immutable external artifact for pretrained weights. Missing parents fail without automatic download. Class weights must be recomputed from the fit's own training windows and verified against the parent F1 artifact's `class_weights_sha256`. Both common-budget development parents and the all-32 refit must persist that digest using `d11_class_weights_v1_shape_dtype_little_endian_float32`: a versioned dtype/shape-tagged encoding of the actual D11 float32 vector, class order 0–17, shape `[18]`, contiguous little-endian bytes. Source-window identity is recorded separately; equal weights do not establish equal fitting pools, and D81 changes only temperature fitting, not the class-weight window pool. Today's F0 saver supplies neither vector nor digest; shared producer/consumer implementation and missing/mismatched-digest failures remain F1 acceptance work. "Rebuild from the checkpoint alone" means no external model type or config, with digest-verified immutable parents described in the checkpoint.

L3's backbone is persisted once per phase and seed (pilot index 0, reporting indices 0–2) as an immutable BF16 state in the two-key format, roughly 6.4 GB each by parameter count. Every L3 checkpoint references it, and consumers load and verify it rather than regenerating it: regeneration from the seed key on another machine need not reproduce the state exactly, and a final or robustness evaluation would then block. Regeneration may still serve as an audit check.

**Campaign lifecycle (D85).** Freeze the campaign definition before fitting; it identifies the declared procedure, code/config/split and frozen-parent identities, without pretending selected LR/B/T are known yet. Finalized files are published atomically without replacement. Existing files may be reused only after complete identity and digest verification; mismatches fail, never overwrite. Keep interruption/resume snapshots separate from finalized products. Exact saved model/optimizer/RNG/cursor recovery may continue the same unchanged campaign; a numerical repair or changed fitting identity creates a new campaign and preserves the previous artifacts and failure record. Shared pretrained weights need not be copied into every campaign; L3 still has four persisted states, not one per fit.

Finalize the accepted manifest only after all required training, selection and calibration artifacts freeze and before any final-test inference. It binds all four pipelines and reporting indices 0–2, their selected LR/head and per-seed B, all twelve refits and 96 exact-B snapshots, L3 and F1/pretrained parent identities, each calibration role and its record digests, every D87/D88 boundary flag with the owner's recorded acknowledgement, and the preflight/layout/measured-budget approval. Consumers receive one explicit accepted-manifest path and SHA-256; never resolve by directory order or a mutable `latest` pointer. Checkpoints reference their parents and the pre-fit definition digest, while the final manifest references checkpoint digests, avoiding a digest cycle. Final-test results do not select the accepted campaign. These publication/resolution checks are requirements for the future runner, not implemented by the declaration tests.

### Physical batch layout from profiling (D78/D86)

D73's objective uses a nominal 32-window update with 4×8 as its reference physical layout. Microbatches of 4 short sequences may limit GPU utilization, so before the pilot, once D74 fixes the inputs, profile 4×8, 8×4, 16×2 and 32×1 on the intended CUDA hardware with D72's deterministic settings and pinned attention backend. Measure the complete fitting path on development inputs without inspecting validation scores, including forward, backward through the frozen transformer, finite checks, optimizer updates, validation and checkpointing. D86 selects the common feasible layout minimizing projected total campaign GPU time, not one arm's update time. The projection includes the maximum permitted pilot expansion, both L6 pilot heads, conservative preselection refit and reporting-head cost bounds, one-time cache construction/storage I/O, validation and checkpointing; report initial-grid and expanded-grid projections separately. A ten-epoch cost upper bound is not permission to train a refit beyond selected B or bypass D64.

With dropout disabled and identical initial state, logical update groups and class-weight denominators, compare every trainable tensor with the 4×8 reference: `norm(g - g_ref) <= 1e-6 + 1e-2 * norm(g_ref)`. A zero reference uses the absolute tolerance; missing or non-finite gradients fail. No passing layout blocks preflight rather than allowing tolerance relaxation after measurement. These are predeclared numerical-fidelity conventions, not measured optima. For cost use warm-up and repeated complete-path timings with the actual declared dropout enabled, record peak memory and isolate profiling RNG from pilot/reporting streams. L4's transformer cache construction remains one unpadded sequence per window; its cached head fits can batch.

Every candidate preserves D73's update groups, group denominator, natural epoch tails and completed-update counts; only dropout-draw grouping and floating-point summation order change. Freeze the selected common layout before the pilot and retain D78's owner approval of the measured campaign budget. No profiling run, speedup, selected layout or GPU-hour figure is supplied by this protocol package.

### Boundary selections and owner acknowledgement (D87/D88)

Three places used to stop the arm for an undeclared owner decision: a reporting curve selecting B=10 (D64), an L2–L4 winner still at 1e-5 or 1e-3 after D65's expansion, and an L6 selection at 1e-5 or 1e-3 (D77). D87 and D88 replace each stop with a predeclared rule: keep the selection, flag it, report it, and require the owner to acknowledge it before the campaign moves past a fixed point.

**Why the B=10 stop was replaced.** The stop could not tell an arm that is still improving from a flat but noisy curve. Once a curve flattens, its maximum lands on whichever checkpoint got the luckiest noise, and correlated neighbouring checkpoints push that maximum toward the ends. In a simulation on D63's grid, with a curve flat from epoch 3 and Gaussian AR(1) checkpoint noise (correlation 0–0.95), one curve picked epoch 10 in 3.5–8.6% of runs, and at least one of twelve independent curves did in 35–66%. That is a simulation under stated assumptions, not a measurement of these arms; curves that peak early and decline would rarely fire, and a truly rising curve nearly always would. The stop fired only after all 96 selection trajectories had run, and its response was undeclared, so the choice would have been made knowing which arm hit the boundary.

**Reporting boundary (D87).** A reporting arm/seed whose aggregated curve selects nominal B=10 keeps B=10; its refit and temperatures proceed. For every reporting curve, flagged or not, report the late-curve gain: the curve's value at epoch 10 minus its best eligible value at or before epoch 8. It is descriptive, never a threshold, and lets a reader judge whether undertraining is plausible.

**Pilot early warning (D87).** For each arm, take the aggregated two-fold pilot curve of its frozen selection (L6: its selected head and LR). If that curve's earliest exact peak is epoch 10, flag it with its late-curve gain. The pilot peak time still never becomes a reporting budget.

**Endpoint selections (D88).** After D65's single expansion, an L2–L4 winner still at 1e-5 or 1e-3 keeps that LR; there is no second expansion. An L6 selection at 1e-5 or 1e-3 keeps its head–LR pair, and switching heads to avoid the endpoint stays forbidden. Report each such selection with its complete pilot score profile and its margin over the adjacent learning rate (within the selected head for L6).

**Owner acknowledgement.** Flags are reviewed at two fixed points, and the runner surfaces each flag as soon as it is raised:

| Review | Flags | Must be acknowledged before |
|---|---|---|
| Pre-reporting | Pilot early warning (D87); endpoint selections (D88) | Any reporting fit |
| Pre-final-test | Reporting boundary (D87) | D85's accepted-manifest finalization, hence any final-test inference |

Each acknowledgement records the owner's interpretation note for that flag, which the write-up carries. It changes nothing inside the campaign: changing H, the grid, an LR, a head or a budget is a new decision and a new D85 campaign that preserves the earlier artifacts. A campaign with no flags needs no acknowledgement. The accepted manifest lists every flag and its acknowledgement.

### Inference and claims (D79)

This arm follows D79's project-wide framework (README §3.4). Its primary family has four comparisons, named before any result:

| Comparison | Endpoint |
|---|---|
| L2 vs. L3 | macro-F1; scaled ECE |
| L2 vs. L6 | macro-F1; scaled ECE |

For each, compute every subject's metric separately for each reporting seed, average the three seeds within the subject, and take the paired L2-minus-comparator difference. Scaled ECE uses each seed's own refit temperature and bins computed within that subject. Report all eight subject differences, their mean, a pointwise paired-t 95% interval, how many subjects favor each arm, and the exhaustive two-sided sign-flip p-value for the mean with its Holm adjustment over the four comparisons. The interval's nominal coverage assumes independent normally distributed subject differences; otherwise it is approximate, particularly at n=8, and the sign-flip symmetry assumption below does not establish its coverage. These numbers are reported, not used as a pass/fail gate; the write-up describes the size and consistency of each difference. Never pool windows or average logits.

Know what those numbers can show at n = 8. The sign-flip test has 256 sign patterns, so its smallest two-sided p-value is 2/256 ≈ 0.0078, reached only when all eight differences share a sign. Holm's first step over four comparisons needs p ≤ 0.0125, so a Holm-adjusted p below 0.05 requires at least one comparison where all eight subjects agree in direction. One exact zero difference doubles the smallest attainable p. The test assumes independent subjects with sign-symmetric differences under no effect; enumerating every pattern removes sampling error, not that assumption. A null is inconclusive evidence, never equivalence, and a detected difference is not automatically practically important.

Everything else is secondary and descriptive: L2 vs. L4, L3 vs. L6, every comparison with F1, raw ECE, Brier, overconfidence, per-class results, the D80 subgroups and robustness. F1 is a single run while each language arm averages three seeds, so F1's own training variability is not represented in those descriptive contrasts.

### Calibration population and diagnostics (D80/D81)

Every temperature fit, both the refit T and the development diagnostics, uses recording repetitions 1–6 only (D81). Each subject also has one terminal repetition-0 rest segment (31, 13 and 28 windows in subjects 1–3, all rest). Treated as a seventh equally weighted repetition, those few windows would carry about a seventh of the subject's fitting weight, so they are excluded from fitting. They stay in language selection scores and evaluation metrics wherever already eligible, under the frozen temperature; D81 does not expand any evaluation population.

Rest is 50–59% of eligible windows in subjects 1–3, so overall ECE and Brier can hide whether an effect concerns rest or active gestures; window fraction is not a decomposition of metric contributions. D80 therefore adds descriptive rest-only and gesture-only raw/scaled ECE and Brier, grouping windows by true label. They use the same frozen temperature and the same 18-class probabilities and predictions; never remove or renormalize the rest class or fit a subgroup temperature, and recompute bins within each subgroup and subject. Pooled ECE is not a weighted average of the two subgroups, because opposing errors can cancel inside pooled bins. These true-label-conditioned diagnostics localize error patterns, not certify subgroup calibration: conditioning on the outcome changes the question, so a subgroup gap can coexist with global calibration.

### Fold-test analysis (D83/D84)

D8's extended analysis and D83's refit-versus-fold diagnostic score each arm/seed's exact-B development fold models on the eight final subjects. D84 gives each model its own temperature, fitted on its four development query subjects using recording repetitions 1–6 only. Use ordinary NLL, equal subjects, equal repetitions within subject and equal windows within repetition, and D39's unchanged numerical recipe and failure checks; do not use training class weights. Fit after all reporting budgets freeze and D87's flags are recorded, then freeze before any final-test inference. Inheriting `evaluation.temperature_fit` means its objective, weighting and repetition filter, not its pooled-OOF prediction source.

Persist each fold-test T as a separate immutable `fold_test_temperature.json`, bound to the exact-B checkpoint SHA-256, parent/backbone/campaign identities, arm/seed/fold/B, fitting subjects/repetitions, logits/weights/config/split digests, T and solver diagnostics. Never rewrite the retained model to add calibration metadata after its digest is used elsewhere. Report raw and scaled fold-test metrics; never fit on final subjects, substitute the refit's pooled-OOF T or F1 T, average Ts, or reuse the three-query-fit/one-query-score development diagnostic calibrators. The four query subjects influenced model selection, so their fitting diagnostics are not unbiased validation. D83 remains a debugging diagnostic: no required direction, no validation of temperature transfer, no causal four-extra-subject interpretation and no test-driven retuning.

**Compute and interpretation (D66–D69, D77, D84).** Four pipelines × three seeds × (eight fold fits + one refit) gives 108 reporting fits: 96 full-horizon selection trajectories and twelve selected-B refits. Add L2–L4's 18 or 30 pilot fits and L6's 20 for 146 or 158 language model-training fits, excluding F1 and scalar-temperature fitting. Calibration adds twelve refit temperatures, 384 three-fit/one-score development diagnostic temperatures and 96 fold-test temperatures, 492 scalar fits overall, with no extra model training from D84. Durations differ greatly (L6 and L4's cached head training are cheap next to L2/L3's passes through the 3B model); fit counts are not runtime estimates, and D78/D86's profiling supplies the measured budget. Report all eight seed-averaged subject differences and keep seed spread separate. Uncertainty is conditional on the fixed split/F1 lineage, frozen pilot decisions and realized reporting models, not complete end-to-end retraining/tuning variability.

D49–D88 settle the language owner contract: selection/exposure, optimizer/pilot/seed/refit/RNG/numerical rules, input assembly, no-LLM control, calibration roles, immutable campaign lifecycle, profiling acceptance/ranking and boundary-selection flags. Runtime implementation, preflight resolution and fitted artifacts remain outstanding. `training.epochs: 10` is the selection maximum, not the refit budget, and `learning_rate: 0.0001` remains an unselected scaffold. The [combined handoff](../project/F1_LANGUAGE_IMPLEMENTATION_HANDOFF.md) records approval and acceptance checks; the explicit implementation gate remains closed until they pass.

---

## L0 — Mock-LLM plumbing check

**Config:** `adapter_mock_debug.yaml` · **Runs:** locally, seconds · **Mode:** `debug`

**What it measures.** Nothing scientific. It verifies that the shapes line up end to end: encoder → projector → *N* soft-prefix vectors → concatenation with prompt embeddings → backbone forward → pooled hidden state → head → 18 logits → cross-entropy → backward.

**Why it exists.** Every dimension bug in this arm is cheaper to find against a 128-d mock than against a 3B model on a rented GPU. The mock backbone is a small random module with a configurable `embedding_dim`; no Llama weights, no tokenizer, no network.

**Easily missed.**
- `mock_llm.embedding_dim` must equal `adapter.output_dim` (both 128 here). This config is the one place `output_dim` is a literal rather than `auto_from_llm_config`, because there is no model config to read.
- The encoder is **not** frozen here (`encoder.freeze: false`, `checkpoint: null`) — this is a plumbing test, not a measurement. Do not copy that setting into L1–L4.
- Passing L0 proves shapes, not correctness. A projector that emits constant vectors passes L0.

---

## L1 — Real Llama 1B smoke test (local)

**Config:** `adapter_llama1b_local.yaml` · **Runs:** locally on MPS · **Mode:** `debug`

**What it measures.** That the *real* embedding-injection path works: a genuine Llama tokenizer and embedding table, real `inputs_embeds` assembly, on Apple Silicon.

**Why it exists.** To prove the mechanism before renting a GPU. Cloud time is the project's only recurring cost, and discovering an injection bug there is the expensive way to find it.

**Easily missed.**
- **This is not a result.** The explicit debug split uses subjects 1–3 for training and subject 4 for holdout, capped at `max_windows_per_subject: 100` for one epoch. Never report a number from this config.
- It runs before the CUDA F1 exists, so, like any local debug run of a downstream arm, it freezes the preliminary MPS encoder (D47). Its `encoder.checkpoint` names the preliminary refit the F1 harness writes, `checkpoints/preliminary/baseline_cnn_subject_split/refit.pt`, never the canonical parent; a regression test derives that path from the harness and fails if the config drifts back.
- Llama weights are **gated**. `hf auth login` and accepting Meta's licence must happen first; this is an account action, not a code action.
- Exercise D74's input assembly (begin-of-text, prefix, `"\nGesture:"` suffix, readout at the last suffix token), so the injection path proved here is the one the reportable arms use.
- `torch_dtype: float16` on MPS: some ops silently fall back or lose precision. If loss is NaN, test in float32 before assuming the adapter is wrong.
- `llm.device: mps` and `training.device: mps` are hard-coded here on purpose — this config is local-only by design. The 3B config uses `cuda` for the same reason.
- 1B and 3B have **different hidden sizes**. This is why `adapter.output_dim` is `auto_from_llm_config`; a value hard-coded to pass here fails on the 3B run.

---

## L2 — Frozen pretrained LLM, subject-independent (the method)

**Config:** `adapter_llama3b_subject_split.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether soft-prefix conditioning of a frozen, *pretrained* language model improves held-out-subject decoding, calibration, or overconfidence beyond the encoder alone.

**Why it exists.** This is the project's secondary hypothesis, and also the piece that demonstrates how to wire a modality into a frozen LLM's embedding space. That skill transfers whether or not the result comes out positive.

**How to read it.** Never alone. L2 is meaningful only as the triple (L2, L3, L4) plus the F1 reference. Reporting L2 against F1 without L3 invites the immediate objection that any gain came from parameter count.

**Easily missed.**
- The encoder is **frozen** and loaded from F1. If it trains here, this stops being an adapter study and becomes a bigger-model study.
- `adapter.output_dim: auto_from_llm_config` — read `hidden_size` from the model config at runtime. Hard-coding it is the likeliest cause of "worked locally, failed on the pod."
- **`head.pooling: last_token` means the last non-padding token**, not literally the final position. `hidden_states[:, -1]` is correct only under left-padding; `attention_mask.sum(-1) - 1` only under right-padding and only without interior zeros. Use the rightmost index where the mask is 1 — `L - 1 - mask.to(torch.int64).flip(-1).argmax(-1)` — which holds either way, then assert every row has an unmasked token and that the mask at the chosen index is 1. An all-pad row sends `argmax` to 0 and silently reselects the final position. This hides because the longest sequence in a batch is right under either padding, so only short rows are corrupted while training runs.
- For every tokenizer-backed soft-prefix path (L1–L3), assemble begin-of-text, the *N* soft-prefix vectors and the suffix in D74's order, build `combined_attention_mask` to match, and derive both the pooling index and explicit `position_ids` from that same combined mask before passing all three with `inputs_embeds`. D74 sequences carry no padding, but keep the mask-derived indexing as a defensive invariant: with padded text, default sequential positions would offset the text by the pad count.
- D73's nominal full update is `batch_size × gradient_accumulation_steps` = 32, with 4 × 8 as the reference layout; epoch-end tails are smaller. D78 selects the physical split once, by profiling before the pilot. Changing either value any other way, for example to fit memory, changes the declared fitting procedure and is not an automatic fallback.
- `wandb_mode: online` here. Requires `wandb login` on the pod; the local JSON logger is still the source of truth.

---

## L3 — Matched-size random-init transformer (rigor baseline) — NEW

**Config:** `adapter_random_transformer.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether the effect in L2 is attributable to **language pretraining** or merely to routing signals through a large fixed transformer.

**Why it exists.** A strong reviewer will raise it immediately, and no claim about "language" holds up without it. A frozen random transformer is still a high-dimensional non-linear feature transform, and those help on their own.

**Easily missed.**
- **"Matched" is a real constraint,** not a label. Same architecture, hidden size, layer count, and head count as `baseline.match_model` — instantiate from the *config* of `meta-llama/Llama-3.2-3B` with random weights, never a differently-shaped stand-in. If sizes differ, this controls for nothing.
- It is **frozen**, as L2 is. A trainable random transformer would be a different experiment.
- Its random backbone is **persisted** once per phase and seed and loaded by digest (D75); never regenerate it to rebuild a checkpoint.
- This row is *why* the readout is discriminative (README §4): a random model has no usable LM head, so a generative readout cannot be run here at all.
- Seed matters more than usual — the "features" are a draw from an initialization distribution. Use D36's predeclared paired repeat policy for all three arms; no outcome-triggered reruns.
- **A null here is the good outcome for rigor.** If L2 ≈ L3, say so plainly: language pretraining did not help, and that is the result.

---

## L4 — Text-summary-only (rigor baseline) — NEW

**Config:** `adapter_text_summary_only.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether the complete learned-latent soft-prefix pipeline outperforms an engineered-text summary through the same frozen model and readout.

**Why it exists.** A strong, inexpensive engineered-text route is a practical alternative to the learned path. Matching its performance would remove evidence for an incremental pipeline benefit, but would not establish equivalence or identify the projector's causal contribution. Make this baseline strong; a weak summary would exaggerate L2's practical advantage.

### Feature set (decision D4)

Six features per channel × 12 channels = **72 scalars**, computed on the **normalized** signal using the same `train_subject_covered_sample_stats` statistics as the encoder path. D82 fixes the exact definitions, because each reasonable implementation choice produces different prompt tokens:

- Normalize exactly as the encoder path does (float32 `apply_channel_stats`), then compute every feature in float64.
- `standard_deviation` is the population SD (ddof 0).
- `waveform_length` sums |xᵢ₊₁ − xᵢ| over the 399 consecutive differences of the 400-sample window.
- `zero_crossings` counts the i with xᵢ·xᵢ₊₁ < 0. The rule is strict, so an exact zero never completes a crossing, and there is no amplitude threshold. On the normalized signal these are crossings of the training-population channel mean, not of the window's own mean.

| Feature | Contributes |
|---|---|
| `mean`, `standard_deviation`, `minimum`, `maximum` | Amplitude distribution |
| `waveform_length` — Σ\|xᵢ₊₁ − xᵢ\| | Temporal: signal path length |
| `zero_crossings` | Temporal: coarse frequency content |

**Why `rms` was removed.** With the population SD, `rms = sqrt(mean² + sd²)` exactly, for any window. With the sample SD the identity becomes rms² = mean² + sd²·(N−1)/N, so the earlier claim that it held for both was wrong; at the fixed N = 400 RMS is still a deterministic function of `mean` and `standard_deviation`. Given those two it carries **zero** additional information. Including it wastes tokens and misrepresents the baseline as richer than it is.

**Why two temporal features were added.** This is the more serious of the two problems. `mean`, `sd`, `min`, `max` (and `rms`) are all **permutation-invariant**: shuffle the samples within a window and none of them move. A baseline built only from those receives *no temporal information at all*, while L2's encoder path runs a 1-D CNN over the ordered signal. Any L2 − L4 gap would then be measuring **temporal structure vs. none** as much as **soft prefix vs. text**, which plants a confound inside the very control meant to rule confounds out. Waveform length and zero crossings carry some temporal information, but they do not preserve the full ordered waveform or make the summary equivalent to a learned CNN latent. The D36 attribution limit still applies.

### Prompt formatting (decision D4)

1. **Fixed-point, 2 decimals.** Post-normalization values sit around ±5, so `%.2f` keeps ample resolution at ~5 characters. **No scientific notation** (`1.2e-05` fragments badly under BPE) and no full float repr (17 digits of token noise). D82: render with Python's `.2f` on the float64 value, print `-0.00` as `0.00`, and print `zero_crossings` as an integer.
2. **One labelled line per channel**, so position is never inferred: `ch01 mean=-0.12 sd=0.98 min=-3.41 max=3.02 wl=41.70 zc=88`. A bare list of 72 comma-separated numbers forces the model to count positions, which it does badly. Handicapping the prompt defeats the baseline's purpose. D82: channels `ch01`–`ch12` in dataset order, lines joined by a newline with no trailing newline, followed by D74's suffix.
3. **The template is a hyperparameter.** Freeze it, version it, and record it verbatim in `run.json`. If it drifts between runs, L4 is not comparable to itself.

Budget ≈ 200–250 prompt tokens per window.

**Easily missed.**
- **No projector and no encoder** — deliberately. `adapter:` and `encoder:` are absent, and the head is the only trainable module. The absence *is* the experiment, not an oversight.
- The backbone must be the **same** frozen pretrained model as L2. Comparing L4 on 1B against L2 on 3B measures model size.
- Features are computed on the normalized signal, not raw mV. Otherwise L2 and L4 see different preprocessing, and the comparison is already compromised.

---

## L5 — Generative readout — DEFERRED (D3)

Not in v1. The frozen LLM's own LM head emits a label token instead of a head reading the hidden state. It would add natural-language output for the structured-report demo, but no inferential power the L2/L3/L4 triple does not already provide, and it costs real work (label tokenization, multi-token labels, calibrated 18-class probabilities from token logits).

**Revisit if** L2 beats L3 under the shared readout, or if the structured-report demo is needed for outreach. If it is ever run, it is a secondary ablation **on L2 only** — never a control's readout.

---

## L6 — Embedding-only MLP control (no LLM) — NEW

**Config:** `embedding_mlp_control.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** How well a conventional nonlinear decoder reads the same frozen F1 embedding that L2's projector receives, with no transformer at all (D77).

**Why it exists.** L2 can only re-read F1's 256-number embedding; no transformer adds information. L2 vs. L3 shows whether pretrained weights beat random ones inside that architecture, but a frozen random transformer may damage the signal, so L2 > L3 can occur without the LLM adding anything over ordinary re-heading. L6 is the row that answers the arm's stated question, *beyond a well-trained encoder alone*, against a decoder that had a fair chance to be good.

**How it is built.**

- Same fold parents, window pool, class-weighted loss, D73/D78 batch rules, D51–D54 Adam recipe, three reporting seeds, D49/D61/D63/D64 selection, D67 lineage and D69 temperatures as L2; the configs share those blocks verbatim.
- Two predeclared heads, both with biased linear layers, GELU and dropout 0.1 after each hidden activation, FP32 throughout:

| Head | Layers | Trainable parameters |
|---|---|---|
| small | 256 → 256 → 18 | 70,418 |
| wide | 256 → 4096 → 1536 → 18 | 7,373,330, comparable to L2's projector plus head (about 7.17 million) |

- Its own pilot on D57's folds 0 and 4 with pilot index 0, over the five-point grid {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} from the start: 2 heads × 5 LRs × 2 folds = 20 fits. Select one head–LR pair by D59's peak-of-mean score. Ties go to the higher exact score, then the smaller head, then D60's LR preference within that head. Freeze the pair across reporting seeds.
- A selected LR of 1e-5 or 1e-3 is kept, flagged and reported under D88, and reporting fits wait for the owner's acknowledgement; choosing the other head to avoid the endpoint is forbidden. L6 neither triggers nor waits for D65's shared expansion, so a cheap control cannot force extra 3B-model fits.
- Arm ID 6. Initialization and training stochasticity use head-specific streams (`l6_head_init`, `l6_training_stochasticity`), shared across LRs within a head; the window order is shared with the other arms. Its reporting curves are inside D64's guard over four pipelines × three seeds.

**Easily missed.**

- A wider search makes the comparison fairer to the control; it does not guarantee a conservative one or better held-out performance.
- The wide head is not tiny. It is only cheap next to a pass through the 3B model, mostly because its frozen F1 embeddings can be cached per fold.
- L6 is not an L-arm readout control. Its head differs from the shared transformer readout by design, so L2 vs. L6 compares whole decoders.

---

## Robustness (D2)

The three F3–F5 perturbations are also evaluated against the L2, L3, and L4 models, not only against the F1 encoder. Whether soft-prefix conditioning preserves or degrades robustness under subject shift is a genuine result, and possibly this arm's most interesting one — the language arm's value, if any, is more likely to appear in calibration and failure behaviour than in clean accuracy.

The registry ([`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml)) lists `l2_frozen_llm`, `l3_random_transformer`, `l4_text_summary` and `l6_embedding_mlp_control`. Resolve one explicit digest-verified accepted campaign manifest, then every reporting seed's refit with its own temperature, and average metrics within each subject (D66/D76/D85). Campaign/seed-aware resolution is not implemented, so each target's `artifact_resolution: pending_implementation` keeps it blocked even after its source protocol becomes ready. L4's perturbation acts on the raw window before normalization and feature rendering. See README §6 for the target registry and checkpoint contract.

---

## Reporting checklist

Before any L-series number leaves this repository:

- [ ] The four D79 primaries reported first, with all eight subject differences, mean, paired-t interval, agreement count and sign-flip/Holm p-values, and no significance label
- [ ] L3, L4 and L6 reported alongside L2 — never L2 versus F1 alone
- [ ] F1's reference macro-F1 and calibration reported alongside, read as descriptive pipeline contrasts that include a tuned, larger readout and a single F1 run against three-seed means
- [ ] Identical readout confirmed across L2, L3, L4
- [ ] Per-subject macro-F1 and calibration shown for every arm, with seed spread kept separate from subject spread
- [ ] Raw and scaled ECE and Brier, overconfidence and the D80 rest/gesture diagnostics reported, not only macro-F1
- [ ] Selected LR (and L6 head), per-seed language budget B and refit temperatures reported with their development provenance
- [ ] Every reporting curve's late-curve gain, and every D87/D88 boundary flag with the owner's acknowledgement note, reported
- [ ] A null result stated plainly as inconclusive, never as equivalence
