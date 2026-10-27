# Generative arm (G-series) — the headline

**The question.** How many real calibration gesture trials does a new, unseen subject need to reach a target decoding macro-F1 — and can synthetic data from a conditional generator replace some of them?

**The deliverable is a curve, not a number.** Macro-F1 against *k*, the count of real calibration gesture trials supplied in total by the held-out subject, plotted for each calibration strategy. The single headline sentence falls out of the curve: *augmentation reaches the target macro-F1 with approximately N fewer real gesture trials per subject.*

Read [README.md](README.md) first: §2 (vocabulary), §3 (shared protocol) are assumed here. Note especially the two senses of "calibration" in §2 — this document uses **subject calibration** (the *k* real gesture trials) unless it says *probability calibration*.

**Prerequisite:** F1 must be complete. The generator operates in the frozen encoder's latent space, so it cannot be built before the encoder exists.

---

## The leakage rules — read before anything else

The generative arm is the part of this project where a positive result is easiest to fake without noticing. These rules are **non-negotiable**, and a result traceable to a violation is treated as **no result**.

1. **The generator never trains on held-out-subject data.** Not their test windows, not their calibration windows, not their statistics. It is trained on training subjects only.
2. **At k = 0**, generation is class-conditioned only, using a population/default subject embedding. There is no subject-specific information available, by definition.
3. **At k > 0**, any subject-specific conditioning is derived **exclusively** from the windows wholly contained in those *k* permitted real gesture trials — never from any other window belonging to that subject.
4. **Normalization statistics** follow README §3.3: computed on training subjects, applied unchanged. A subject-specific normalization fitted on the full held-out recording is a leak, and it is invisible in the logs.
5. **The evaluation set is fixed** across all *k* and all strategies. If the test windows shift as *k* grows, the curve measures the test set rather than the method.

**Easily missed:** the most common leak is not in the generator at all — it is fitting *any* statistic (normalization, subject embedding, class prior) on data the protocol has not yet "paid" for. No strategy may see more than the selected *k* trials: all three adapted strategies use exactly that same real set, while `population_no_adaptation` deliberately uses none. Their overlapping windows are derived examples, not additional calibration units.

---

## G0 — Generator plumbing check

**Config:** `gen_vae_debug.yaml` · **Runs:** locally, minutes · **Mode:** `debug`

**What it measures.** That the conditional VAE trains at all: encoder latents load, the class-conditioning and subject-conditioning paths accept their inputs, the negative-ELBO objective is finite and updates parameters, and samples come back at the right shape.

**Why it exists.** Same reason as L0 — find shape and conditioning bugs on 3 subjects locally before spending cloud time.

**Easily missed.** A finite loss and parameter update prove plumbing, not usefulness. During KL warmup, the changing weight means objective values are not directly comparable across epochs. Check that reconstructions are not the dataset mean, and that varying the class label actually changes the sample. Both failures can accompany a beautiful loss curve.

**Subject-conditioning diagnostic (D37).** Keep population-only training for this small smoke test, but evaluate the decoder with two fixed diagnostic subject vectors: the zero vector and the first unit basis vector, at the declared subject-embedding dimension. Hold the class label, latent-noise draw and decoder evaluation mode fixed while changing only that vector; record the two vectors and outputs and check that they differ. These vectors are synthetic test inputs, not NinaPro subjects or reportable data. This checks structural sensitivity of the conditioning path, not learned personalization: G1 must still test that training learned to use calibration-derived subject information, with its class-composition probe and other gates.

---

## G1 — Conditional VAE training

**Config:** `gen_vae_train.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** A conditional VAE over the **frozen encoder's latent space**, conditioned on gesture class and a subject/calibration embedding.

**Why latent space rather than raw signal.** Small, fast, and stable to train; and the downstream classifier consumes encoder latents anyway, so synthesizing at that level is sufficient for the personalization question. Raw-waveform synthesis is a larger project and is not required for the headline claim.

**Easily missed.**
- **Posterior collapse.** The classic conditional-VAE failure: the decoder learns to ignore the latent and generate from the conditioning alone. Symptom: KL term → 0. Monitor the KL term separately from the total negative-ELBO objective, and warm up its weight rather than applying it at full strength from step 0.
- **The conditioning must actually condition.** Verify that sampling with a fixed latent and varying the class label produces different outputs. If not, the "conditional" generator is unconditional and every downstream result is meaningless.
- **Training subjects only** (leakage rule 1). This constrains the subject-embedding design: it must be *computable* for an unseen subject from the windows contained in their *k* permitted gesture trials, so it cannot be a learned per-subject lookup table indexed by subject ID. An embedding that only exists for training subjects cannot be produced for a new user at all.

### The subject-embedding contract (D27)

`calibration_derived` fixes what information the embedding may use, not how to compute it. D27 fixes the rest.

| Field                     | Decision                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Estimator                 | **Class-centred mean latent residual.** Subtract the population per-class mean latent from each support window, average the residuals, project to 16-d |
| Centring reference        | Population per-class means fitted on **training subjects only**                                                                                        |
| *k* schedule              | Draw *k* **uniformly from the deployment grid** `{0,1,2,5,10,17,20,34}`, per subject per step, support built by D14's nested prefix                    |
| Support rotation          | Rotate the support repetition pair among **{1,4}, {2,5}, {3,6}**; targets are that draw's complement                                                   |
| Population embedding      | A single **learned vector**, used at *k*=0 and for an empty support set                                                                                |
| Class-information control | A **probe** that must sit at chance when predicting the support set's class composition from the embedding                                             |

**Why centring rather than a plain mean.** Encoder latents cluster by gesture, so the mean of a support set largely encodes *which gestures the subject happened to demonstrate* — and at *k* < 17 the support covers only some classes. The residual against the population's gesture-*g* mean is "how this person's gesture-*g* latent differs from the average person's", which is subject identity with class content removed **by construction**. The probe is the check that it worked, not the mechanism. A learned set encoder was rejected: at *k*=1 the support is ~97 windows of a single gesture, and a parameter-heavy encoder trained on 32 subjects would become the bottleneck rather than the data.

**Why rotate the support repetitions.** Fixing support to {1,4} would mirror inference exactly, but it permanently spends two of six repetitions as conditioning-only data. Rotating lets every repetition serve as a reconstruction target in two of three rotations, which matters more at this stage. Rotation is **augmentation, not mismatch**: it stops the estimator memorising position-specific artefacts of reps 1 and 4, and an estimator robust across positions handles {1,4} at inference. The pairs all span early-to-late so the support *shape* still matches inference and still straddles the drift the DB2 descriptor measured.

**Easily missed — rotation is training-subjects-only.** Held-out subjects keep D13's fixed {1,4} calibration and {2,3,5,6} evaluation. Rotating *their* repetitions would hand the method evaluation data it never paid for, which is leakage rule 3.

**Target exclusion survives rotation.** Support and targets stay complementary *within each draw*, so a target latent can never appear in the set conditioning its own reconstruction. Exclusion is per-draw rather than global, and needs no separate mechanism.

**Pseudo-calibration randomness (D43).** Both the random *k* draws and support-rotation choices inherit README §3.5's seed factory, explicit PCG64 assignment backend, zero-based generated draw indices, and source-pool/assignment/RNG provenance. Persist a lossless pseudo-calibration draw ledger; it must identify the subject, training step/draw, chosen *k*, rotation and selected support trials once their exact component keys and rotation-selection rule are settled. Record each per-step draw before the fitting update that consumes it, or precompute and persist the ledger before training. The factory does not decide those still-Pending sampling details or the class-composition probe, and the config continues to block execution on them.

**Carry this forward.** At small *k* the embedding may carry almost no information — *k*=1 is one gesture from one session, so its residual is noisy and gesture-specific. That is a possible result, not a defect, but it means a small-*k* gain could come from class-balanced rehearsal rather than subject conditioning. D29's paired population-replay control separates those explanations; the causal claim depends on preserving that matched contrast.

G1 still fails closed on the VAE objective components (`training.objective_components`): D22 fixed the objective's sign and name, not its reconstruction likelihood, reductions, units or KL-warmup contract.

---

## G2 — Synthetic quality by discrimination (the gate)

**Config:** `gen_synthetic_quality.yaml` · **Runs:** cloud or local · **Mode:** `evaluate`

**What it measures.** A **classifier two-sample test**: train a discriminator to tell real encoder latents from generated ones. AUC near chance means the approved discriminator did not detect separability; it establishes distributional similarity only under the approved grouping, orientation, uncertainty, and sanity-control contract.

**Why it exists — it is a gate, not a metric.** It runs *before* G3 is interpreted. Without it, a personalization "gain" could come from a degenerate generator: one emitting near-copies of a few training examples, or class-typical blobs that happen to be linearly separable. Either produces an artifact rather than a result, which is why G2 has to pass before the headline claim means anything.

**Easily missed.**
- **Both failure directions matter.** Strong separation means the generator is unrealistic; its numeric AUC direction depends on the approved label/score orientation. AUC ≈ 0.5 from a *weak* discriminator means nothing at all — it must be strong enough to separate obviously-bad synthetic data, so calibrate it against a deliberately poor generator (e.g. Gaussian noise at matched moments) and confirm strong separation there.
- **Fidelity is not diversity.** A generator that reproduces ten real samples perfectly scores near chance and is useless. Nearest-neighbour distance from each synthetic sample to its closest real training sample can expose memorization, which is both a quality failure *and* a privacy consideration, but it cannot establish diversity by itself. G2 needs a separate coverage or synthetic–synthetic diversity diagnostic.
- **The iterative development gate cannot use the final test subjects.** The old wording, "test on held-out real data," left two different held-out roles collapsed: inner subject validation for model development and the eight final test subjects. G2 is fail-closed until the development population/aggregation and the one-shot final diagnostic failure policy are approved. A final diagnostic may never trigger tuning on those same subjects.
- **Evaluate per class.** A generator can be excellent for rest and useless for rare movements, and a pooled AUC hides that.
- Report AUC with a confidence interval. A single number near 0.5 with a wide interval is not evidence of anything. Discriminator splitting and interval resampling must respect subject/trial dependence rather than treating overlapping windows as independent.

**Gate condition:** if the approved development gate fails, fix the generator before the protocol is frozen. If the eventual one-shot final diagnostic fails, G3 is not reported; its data cannot be recycled into another development iteration. The exact two-stage population policy and the quality-metric/acceptance contract remain Pending.

**Shared development artifacts (D35).** The development gate consumes the encoder/generator pairs already specified for D28 at `development/fold{fold_index}/fixed_budget.pt`, with their provenance verified against the same manifest and frozen source configs. This is one shared artifact set, currently planned rather than built. Both components exclude the current outer four subjects; using an all-32 refit for either would expose the gate population during population training. The VAE's stored parent-encoder digest must match the encoder actually used to produce real latents, normalization and centring references. Keep discriminator fitting, scoring and nearest-neighbour references within that fold's latent space; nearest-neighbour memorization references use that fold's real **training** latents. Aggregate the approved diagnostics afterward, never raw vectors from independently trained encoders. Held-out development subjects retain the manifest's fixed support/evaluation partition; G1's support rotation remains training-subjects-only.

Reuse saves additional encoder/generator fits beyond D28, not discriminator training. It does not settle G2's discriminator split, acceptance, fold aggregation or conditioning-coverage choices; those remain blocked. A failed gate may require rebuilding several generator lineages and their dependent synthetic banks rather than a guaranteed single rebuild. Final diagnostics use frozen refit artifacts only after development is complete and can never trigger tuning on final subjects.

---

## G3 — Personalization-efficiency curve (the headline)

**Config:** `gen_personalization_efficiency.yaml` · **Runs:** cloud CUDA · **Mode:** `evaluate`

**What it measures.** For each held-out subject, 18-class macro-F1 as a function of the total number of real calibration gesture trials *k* ∈ **{0, 1, 2, 5, 10, 17, 20, 34}**, under four calibration strategies:

| `calibration_strategy` | Operation | Answers |
|---|---|---|
| `population_no_adaptation` | Use the unchanged F1 population head; no held-out-subject data | The horizontal population reference |
| `real_adaptation` | Start from the F1 head and fine-tune it on all valid windows from the *k* selected real trials | **D5 baseline** — what ordinary head adaptation buys |
| `real_plus_population_synthetic_adaptation` | The same, plus synthetic windows conditioned on the **population** embedding | **The control (D29)** — what generic balanced replay buys on its own |
| `real_plus_subject_synthetic_adaptation` | The same, plus synthetic windows conditioned on **this subject's** calibration-derived embedding | **The headline** — does subject conditioning add anything beyond generic replay? |

Four curves on one axis (D29). The headline number is the horizontal gap: how many fewer real gesture trials `real_plus_subject_synthetic_adaptation` needs to reach the macro-F1 `real_adaptation` reaches at a given *k*. Name the result `real_gesture_trials_saved`; never shorten it to "samples saved," which would obscure the unit.

The **unit and name** of that statistic are frozen; its numerical extraction is not. Before the runner is implemented, fix the target-macro-F1 rule, subject-level versus aggregate computation, interpolation or monotonic treatment, uncertainty, and what is reported when a strategy never reaches the target. Those remain Pending rather than being hidden in evaluation code.

D34 applies the same fixed labels 0–17 and `undefined_class_f1: 0.0` rule as F1. At small *k*, gestures absent from support still appear in the fixed evaluation set and the metric denominator; never score only the demonstrated gestures. Accuracy remains a secondary curve, and D30's subgroup recall and rest accuracy explain where changes occur. The trials-saved statistic is derived from the macro-F1 curve; retain its pending numerical extraction gate.

**Why the adaptation strategy is here (D5).** It is the first question anyone asks about the headline: *why synthesize data instead of just fine-tuning on the real trials you already have?* Without `real_adaptation`, the contribution is unquantified. It is cheap because it shares this entire harness and needs no generator. Scope is **head-only fine-tuning**. For *k*>0, all three trainable strategies use the same F1 initialization, trainable surface, optimizer schedule, and optimizer-step budget; full fine-tuning and test-time adaptation are follow-ups. At *k*=0, `real_adaptation` is a no-op and coincides with `population_no_adaptation`. The two replay aliases share one population-conditioned head per recorded `(adaptation_seed, synthetic_draw_id)` across subjects and schedules, because neither supplies an input there; predictions and metrics are still computed once per held-out subject and repeat. That synthetic-only point is a diagnostic, not a matched real-versus-synthetic comparison, because no real-only optimizer-step budget exists at *k*=0.

### The adaptation objective (D28)

$$J_{\text{real}} = L_{\text{real}} + \beta R_{\text{SP}}
\qquad
J_{\text{replay},c} = J_{\text{real}} + \lambda_{\text{syn}} L_{\text{synthetic},c}$$

**Additive, not averaged.** The replay objective embeds the entire real-only objective *with its coefficient unchanged* and adds exactly one intervention. An averaged form, $(L_{\text{real}}+L_{\text{synthetic}})/2$, would halve the real term's coefficient — changing two things at once in a comparison designed to change one.

Be precise about what that buys. The additive form preserves the **algebraic coefficient** of the real-only objective and nothing more. The realized update, the parameter trajectory, the gradient norm and the effective step size all change — intentionally, because that is the intervention. The combined gradient norm is *not* guaranteed to grow. The quantity $\|g_{\text{real}} + \lambda g_{\text{syn}}\|$ can rise or fall depending on gradient alignment. Adam normalizes by its second moment, so loss rescaling does not translate into a proportionally larger update at all.

| Term | Definition | Why |
|---|---|---|
| $L_{\text{real}}$ | $\sum_i w_{y_i}\mathrm{CE}_i \big/ \sum_i w_{y_i}$ | A weighted **mean**. The real set spans ~97 windows at *k*=1 to ~3,300 at *k*=34, so an unnormalized sum would make the loss scale a function of *k* and the curve would partly measure that |
| $L_{\text{synthetic}}$ | $\frac{1}{N_{\text{syn}}}\sum_j \mathrm{CE}_j$, **unweighted** | Inverse-frequency weights exist to flatten a skewed distribution. The synthetic bank is already flat at 50 windows per class, so weighting it would *de-balance* it toward rare population classes |
| $R_{\text{SP}}$ | $\|\theta - \theta_{\text{F1}}\|^2$, summed over head parameters | **L2-SP**: anchor to the pretrained solution, not toward zero. Adam's `weight_decay` is set to **0**, or the two regularizers compete silently with opposite targets |
| $\lambda_{\text{syn}}$ | **Frozen at 1.0** | With both sources normalized to a per-source mean, this is a transparent pre-declared equal-source coefficient; it is not claimed to be uniquely optimal. Freezing it means the replay arms add **no separately tuned loss coefficient** |
| Class weights | F1 population weights, from the development fold's 28 subjects; all 32 for the final pipeline | Defined even when a subject's calibration set omits classes. Fitting on 32 during development would leak the validation subjects' class distribution, the same rule as §3.3's normalization statistics |

At *k*=0 no real support exists, so the shared replay diagnostic minimizes $\beta R_{\text{SP}} + \lambda_{\text{syn}}L_{\text{synthetic,population}}$. The unchanged population head and the real-adaptation no-op are the other shared model variant. Cache the unchanged head once. For replay, each recorded `(adaptation_seed, synthetic_draw_id)` produces one population-conditioned head shared across subjects, schedules, and both replay-strategy aliases; do not collapse distinct recorded repeats, but do not present four independent strategy estimates. Its optimizer-step assignment remains Pending: real adaptation is a no-op at *k*=0, so neither that arm nor a nonzero-*k* mapping can silently select the synthetic-only diagnostic's step count. G3 and G4 remain blocked until the config's explicit `synthetic_only_k0_optimizer_steps` rule is resolved.

**Full-batch, restarted at every *k*.** Head-only latent adaptation at the largest *k* has roughly 3,300 active-gesture real windows before any rest admitted by the still-Pending ownership rule, plus 900 synthetic 256-d latents — still only a few megabytes. At *k*>0, generate each paired population/subject bank set once per recorded `(subject, schedule, k, synthetic_draw_id)`; at *k*=0, generate one population-conditioned bank per `synthetic_draw_id` and share it across subjects and schedules. In both cases reuse the fixed bank across optimizer steps; redrawing 50/class every step would silently multiply the data budget. Full-batch removes minibatch composition and cycling as protocol variables, and every permitted real window contributes to every step. Each *k* restarts from the F1 head with a fresh optimizer; **never warm-start *k*=2 from *k*=1**, which would make the curve cumulative rather than independent *k*-shot adaptation.

Matched optimizer steps means matched *between strategies at the same k* — D15 does not fix one count across all *k*. It also does not mean matched FLOPs: the replay arms process 900 extra examples per step, which is acceptable and must be reported.

**All 18 output rows stay active.** Classes absent from a small-*k* support set receive only negative gradient, and `real_adaptation` may therefore fall *below* `population_no_adaptation` at small *k*. That is a legitimate finding — naive adaptation can hurt before it helps — and masking or freezing absent classes would shrink the very gap the headline reports.

**Tuning, and what is still open.** Shared hyperparameters are selected on `real_adaptation` only, then frozen for both replay arms: treatment-specific tuning is the thing to avoid. The resulting bias direction is **not known** — under the additive objective a fixed anchor is relatively weaker against a larger data loss. This is a shared anti-cherry-picking rule, not an unbiased estimator. Selection runs through the D16 folds: one fixed-budget development model per fold is trained on the manifest's 28-subject complement, with model weights never checkpoint-selected on the outer four. Its common epoch budget is D18's globally development-selected refit budget, which was derived across all eight validation groups; these are therefore hyperparameter-selection diagnostics, not unbiased outer-fold performance estimates. Build and persist the fold-local population model at `checkpoints/baseline_cnn_subject_split/development/fold{fold_index}/fixed_budget.pt` and its generator at `checkpoints/gen_vae_train/development/fold{fold_index}/fixed_budget.pt`. The generator must consume that fold-local encoder. The all-32 `refit.pt` artifacts and validation-selected `folds/fold{fold_index}/best.pt` artifacts are forbidden for this selection machinery.

Normalization, class weights, latent centroids, population embedding and generator are all fitted from the same 28, calibrating from {1,4} and evaluating on {2,3,5,6}. Each development artifact records the manifest hash, fold index, exact training/scoring subject IDs, source-config digests and fixed encoder and generator training budgets and exact parent-encoder digest; the runner verifies that provenance before scoring a candidate. D33/D35 share these artifacts with population-temperature fitting and G2 development diagnostics, without making their scores reportable estimates. *k*=0 is excluded, since real adaptation is a no-op there. Those eight development fits are shared infrastructure for the whole arm, not a cost charged to adaptation tuning — and **D27's centring reference must come from the same 28**, or development silently mixes populations. The numeric `candidate_grid` is still Pending: it must fix whether steps are one global count across nonzero *k* or a pre-declared *k*→steps mapping, the metric's aggregation order, and a deterministic tie-break before results are inspected.

**Design requirement.** `calibration_strategy` must be a **pluggable axis** in the runner from the first line of code. Bolted on afterwards it means rewriting the loop.

### Which repetitions calibrate, and which evaluate

Calibration draws from **repetitions {1, 4}**; evaluation uses **{2, 3, 5, 6}**. The two sets are disjoint, which is what leakage rule 3 requires — windows from the *k* permitted trials cannot overlap the windows the method is scored on. G3/G4 obtain both sets from the verified split manifest; their experiment configs deliberately do not duplicate them.

The reason the calibration set is `{1, 4}` rather than `{1, 2}` is drift. The DB2 descriptor reports "a significant (P<0.05) dependence on the repetition ... in 12.5% (database 2) of the subjects": for roughly one subject in eight, sEMG amplitude moves systematically across the six repetitions, from fatigue or electrode shift. Calibrating on the first two repetitions and scoring on the last four would then measure adaptation to that drift rather than adaptation to the subject, in those subjects. Spanning early and late keeps the calibration set representative of the session it is meant to represent.

This is a **session-representative offline calibration** estimand, not a strictly prospective "calibrate once at startup, then evaluate only later data" estimand: repetition 4 occurs after evaluation repetitions 2 and 3. The sets are disjoint, so this is not sample overlap, but the headline and paper must not imply chronological onboarding that DB2 cannot test under this split.

Repetition indices come from `rerepetition` (README §3.1), not `repetition`.

### The calibration unit and acquisition schedule (D14)

One unit of *k* is one **complete active-gesture trial**: one gesture class performed by one held-out subject at one eligible corrected `rerepetition`. Selection happens at the trial level before windowing. Every 400-sample window wholly contained in that trial is available; no window may straddle gesture/rest, trial, or calibration/evaluation boundaries. A nominal 5 s trial yields roughly 97 windows at stride 100, but those highly overlapping windows are derived training examples, not 97 independent calibration units.

Exercise B supplies 17 active gestures and two eligible calibration repetitions, so each subject's pool contains **34 active-gesture trials**. The *k* grid is therefore **{0, 1, 2, 5, 10, 17, 20, 34}**; values above 34 are impossible. The axis is total trials per subject, not trials per class. At *k* < 17, calibration intentionally omits some gesture classes: G3 then measures whether subject information learned from demonstrated gestures transfers to the held-out subject's undemonstrated gestures.

For each `(subject, schedule_index)`, construct one seeded, nested acquisition schedule. **D43 explicitly amends the former three-field colon-string seed recipe before any reportable run:** use README §3.5's `temporallens_rng_v1` key with `root_seed=experiment.seed`, `scope="g3_g4_personalization"`, `purpose="acquisition_schedule"`, and `indices={"subject_id": subject_id, "schedule_index": schedule_index}`. Use one fresh NumPy PCG64 generator and schedule indices `0..4` for the five declared schedules; retain the dataset's one-based subject IDs. **D45 fixes the remaining ordering mechanics below.** G3/G4 and every strategy reuse the same realized schedule artifact, not separately seeded lookalikes.

1. Call `Generator.permutation` on the ascending integer gesture IDs `1..17`.
2. In the first pass, select exactly one trial per gesture, alternating repetitions 1 and 4 down that class list. If `(subject_id + schedule_index) % 2 == 0`, start with repetition 1; otherwise start with repetition 4. Because 17 is odd, the starting repetition supplies 9 trials and the other 8; that imbalance flips across schedules.
3. In the second pass, select each gesture's other eligible repetition. Group the remaining trials by their repetition. Sort each group's gesture IDs ascending, then call the **same generator** twice more: repetition-1 group first, repetition-4 group second, regardless of subject parity. Interleave the shuffled groups starting with the repetition underrepresented after the first pass. There are exactly three sequential permutation calls, no fresh generator per call and no final shuffle.
4. Define every calibration set as a prefix of that schedule, so a smaller-*k* set is a strict subset of every larger-*k* set.
5. Reuse the identical selected trials for all calibration strategies. Unselected trials from repetitions 1 and 4 remain unused; they never enter evaluation.

This balanced reshuffle removes the deterministic coupling between being demonstrated earliest and receiving a second demonstration earliest. Reusing first-pass order would also be valid and balanced, but is not the selected policy. An unrestricted shuffle of all remaining trials could violate repetition balance at intermediate prefixes. D45 guarantees imbalance at most one for every prefix: *k*=17 is 9/8, *k*=20 is 10/10, and *k*=34 is 17/17. Relative to same-order reuse, only the configured *k*=20 membership changes; all budgets up to 17 and the full 34-trial set are unchanged. The schedule is offline selection from recorded trials, not their acquisition chronology, and model fitting still restarts at every *k*.

`data.assignments.build_g3_schedule` requires an actual recording: exactly one contiguous, windowable segment for each of the 34 eligible `(gesture,rerepetition)` pairs, with no missing, split or duplicate eligible trial. Source provenance binds their sample boundaries and canonical little-endian float32 calibration-signal digest; evaluation-repetition signal values are not hashed or used to choose the schedule. Persist the source ledger, ordered trial IDs and original D43 RNG record before fitting at the shared consumer-neutral locator `data/processed/assignments/g3_g4_personalization/v1/{source_pool_sha256}/subject{subject_id}/schedule{schedule_index}.json`. Both configs name this same artifact; load and verify an existing one against the current source fingerprint rather than redraw it. Material schedule, source-pool, logical-envelope and exact-file digests have separate meanings, as in README §3.2. Distinct schedules are not forced by redrawing coincidental duplicates. Rest ownership and G1's training-time rotation/component identities are not settled by D45.

Repeat schedules over fixed seeds and report the within-subject spread. Schedules are Monte Carlo replicates of the acquisition policy, not additional subjects or independent inferential units. The point summary, spread summary, and joint-versus-ordered reduction across schedules, adaptation seeds and synthetic draws remain Pending in `evaluation.within_subject_repeat_aggregation`. Deduplicate shared population artifacts before that summary, and never treat repeated scoring on the same evaluation windows as independent observations.

**Easily missed.**
- **Class coverage is part of the treatment.** In addition to overall accuracy and macro-F1, report D30's mean per-class recall for calibration-seen and calibration-unseen active gestures, rest accuracy separately, and the number of demonstrated gesture classes. Otherwise which gestures happened to be selected can masquerade as a method effect. An empty subgroup is `NA`, never zero.
- **Report per-subject curves, not only the mean.** A mean over eight held-out subjects can conceal one stuck at chance, and the variance between subjects is a finding in its own right.
- **The real and synthetic units differ.** *k* counts gesture trials; the generator budget counts latent windows per class. Name and report both. D28 uses full-batch steps matched across all three adapted strategies at each *k*>0 and freezes the synthetic source coefficient at 1.0. Replay still processes 900 additional latents per step, so report source counts and compute rather than calling the comparison compute-matched. Only the numerical step/LR/L2-SP grid remains Pending.
- **Rest ownership remains open.** Rest is the eighteenth output class, but it is not an ordinary prompted gesture trial and does not increment *k*. Inspection established that corrected rest intervals carry repetition indices, so unique trial-to-rest assignment is feasible; the owner still must choose the exact rule before implementation. Never give a method all held-out-subject rest data for free.
- **The evaluation set never changes** with *k* (leakage rule 5).

---

## G4 — Calibration efficiency

**Config:** `gen_calibration_efficiency.yaml` · **Runs:** cloud CUDA · **Mode:** `evaluate`

**What it measures.** Expected calibration error and overconfidence-on-wrong-predictions as a function of *k*, for the same four strategies as G3. This is *probability* calibration, over a *subject-calibration* axis — the collision of terms README §2 warns about.

**Why it exists.** Augmentation could raise accuracy while making the model confidently wrong, which for a wearable interface is a poor trade. Improving both makes for a much stronger claim than improving accuracy alone.

**Probability versions and claims (D33).** Temperature-scaled ECE is the primary calibration endpoint, consistent with F1/F2 and the language arm. Raw ECE and raw/scaled Brier are mandatory companions, along with D25's overconfidence outcomes. Raw ECE describes confidence reliability before recalibration; scaled ECE describes the adapted head plus a calibrator a deployer can fit from development subjects. A raw advantage that is no longer detected after scaling is a finding about a largely correctable confidence-scale effect, not proof that synthesis had no benefit. Use the metric meanings and interpretation rules in README §3.4; macro-F1 still measures decoding.

### Development-fitted strategy × k temperature table

Freeze the adaptation settings before this separate probability-calibration pass. Reuse the verified D28 fold-local population models and generators; run **all four strategies**, including the replay arms absent from real-only hyperparameter selection. For each outer development subject, schedule and recorded repeat, adapt only from the selected *k* trials in the manifest calibration repetitions `{1,4}` and collect logits on its disjoint evaluation repetitions `{2,3,5,6}`. Fit one population T per strategy × *k* across those development predictions, using equal subject and within-subject repeat weights and ordinary NLL with natural class proportions (README §3.4). No final-test-subject trial or prediction enters this fit, and no additional test-subject calibration burden is incurred.

Fit cells separately, with no smoothing across *k* and no per-test-subject T. The unchanged population reference has one T across all *k*; `real_adaptation` at *k*=0 uses that same T. The two replay arms at *k*=0 share one T fitted over their shared population-conditioned repeat domain. Deduplicate aliases and fictitious schedule copies before assigning weights. Independent fitting does not make cells independent observations: they reuse subjects and query windows.

Fit the population reference on these **query repetitions**, not by inheriting F1's all-six-repetition T. Different evaluation distributions can justify different fitted parameters under the same procedure. Do not directly compare F1 and G4 ECE values on different evaluation sets. The complement-28-to-refit-32 approximation applies to both the population model and generator.

A new pass on the same development subjects/query repetitions is still reuse of model-selection data. Its outputs have `calibration_fitting_diagnostic_only` status, not independent validation performance. Freeze the completed table before any final-test inference and do not revise it after viewing final results. The fitting rule is shared across strategies; their fitted T values may differ. It minimizes development NLL, not final-test ECE, and does not guarantee optimal calibration.

Persist `temperature_table.json` as a versioned fitted artifact identified by an immutable run ID and SHA-256. It stores the cell values, aliases, per-cell subject/repeat/window counts, normalized fitting weights, fit diagnostics, manifest/repetition identities, exact development subject/fold IDs, settings and source-config digests, encoder/generator artifact and parent digests, query-logit digest and freeze timestamp. Record the intended final refit population-model and generator digests as deployment parents as well as the development digests; the former bind the table to the final pipeline, while the latter identify the fits that estimated T. These digests differ under the declared 28-to-32 approximation. A consumer verifies deployment parents and settings against its adapted-head lineage, not equality between a development and final model's weights. A failed, non-positive, non-finite or unresolved boundary fit blocks reportable scaling. D39's shared float64 inverse-temperature solver, bounds, tolerances and diagnostics in README §3.4 apply to every cell; numerical readiness does not waive a failed fit or the remaining G4 design gates. Raw/scaled uncertainty is conditional on the frozen table, as it is on the frozen models.

**Easily missed.** D23/D39 fix top-label ECE to nominally 10 equal-mass bins, keeping confidence ties together and reporting the effective nonempty count. ECE computed on a small per-subject test set is high-variance, while pooling predictions can hide subject-level miscalibration and overweight subjects or repeated evaluation windows. The schedule-within-subject summary is separately blocked by `evaluation.within_subject_repeat_aggregation`; subject weighting and the role of pooled ECE remain blocked by `evaluation.headline_ece_aggregation_status`. Always show per-subject values.

---

## G5 — Conditional diffusion variant — STRETCH

Not in v1. Replaces the VAE with a conditional diffusion model over the same latent space, run through the identical G2 gate and G3 protocol. Attempt only once the VAE result is in hand and the curve exists.

---

## Metrics reference — the G-series keys

README §3.4 is the shared metric registry. The tables below register generation-specific keys, and every `evaluation.metrics` entry in a `generation/` config appears as a first-column key here or in the shared registry. Registration prevents unnamed config values; it does not by itself make a metric operational while a required contract remains Pending.

**G0/G1 — generator training**

| Key | What it is |
|---|---|
| `negative_elbo` | The fixed-sign name for the minimized conditional-VAE objective; lower is better under a fixed definition. At KL weight 1 it is the negative of the raw evidence lower bound. The scheduled warmup loss is a weighted analogue whose values are not comparable while the weight changes (G0) |
| `reconstruction_error` | The reconstruction term, reported apart from the total. Separates a decoder carrying signal from one reproducing the dataset mean |
| `kl_divergence` | The KL term, reported apart from the total. KL → 0 is posterior collapse: the decoder is ignoring the latent and generating from the conditioning alone (G1) |

This freezes only the optimized objective's sign and name. The reconstruction likelihood/error, component reductions across dimensions, windows, and subjects, units, KL-warmup schedule, whether evaluation uses the fixed weight-1 objective, and raw-versus-weighted KL reporting remain Pending before G0/G1 implementation.

**G2 — the synthetic-quality gate**

| Key | What it is |
|---|---|
| `discriminator_auc` | AUC of the classifier two-sample test separating real encoder latents from generated ones. ≈0.5 means no detected separation under the approved test; strong separation is a failure, with numeric direction set by the Pending orientation. Meaningless unless calibrated against a deliberately poor generator |
| `discriminator_auc_confidence_interval` | Interval on that AUC. A point estimate near 0.5 with a wide interval is not evidence of anything |
| `per_class_discriminator_auc` | The same AUC per class. A pooled AUC hides a generator that is excellent for rest and useless for rare movements |
| `nearest_neighbour_distance` | Distance from each synthetic sample to its closest real training sample. It can expose memorization, a quality and privacy concern, but is not by itself a diversity or coverage guarantee |

**G3 — the personalization-efficiency curve**

| Key | What it is |
|---|---|
| `macro_f1_vs_k_curve` | **Primary decoding curve (D34).** Fixed 18-class macro-F1 against *k*, one curve per `calibration_strategy`; summarize repeats within subject before comparison |
| `accuracy_vs_k_curve` | Secondary overall-correctness curve against *k*, weighted by the natural evaluation class frequencies |
| `real_gesture_trials_saved` | **The headline.** Horizontal macro-F1 gap: how many fewer real gesture trials `real_plus_subject_synthetic_adaptation` needs to match `real_adaptation`. Its numerical extraction is still Pending; D29's shared population-conditioned replay diagnostic at *k*=0 is excluded |
| `calibration_seen_gesture_macro_recall` | **Mean per-class recall** over gesture classes the subject demonstrated in their *k* trials — subgroup balanced accuracy (D30) |
| `calibration_unseen_gesture_macro_recall` | The same over classes they did **not** demonstrate. Below *k* = 17 this is the transfer question the grid exists to ask; `NA` from *k*=17, where one trial per gesture gives full coverage |
| `rest_accuracy` | Accuracy on the rest class, reported separately because rest is an evaluation class that never increments *k* |
| `demonstrated_gesture_count` | How many distinct gesture classes the *k* trials covered. Without it the seen/unseen split cannot be read at a given *k* |

**G4 — calibration efficiency**

| Key | What it is |
|---|---|
| `ece_vs_k_curve` | Expected calibration error against *k*, one curve per strategy and probability version; scaled is primary, raw mandatory alongside, both with explicit suffixes |

Several registered metrics remain non-executable under DECISIONS → Pending. Existing blockers cover G1 embedding operational details, both G2 contracts, G3 headline extraction, rest ownership, schedule reproducibility, the adaptation candidate grid, repeated-run domains and within-subject aggregation, the VAE objective components, calibration execution and G4 headline-ECE aggregation. D25 and D30 have already settled the overconfidence-error and subgroup semantics; implementers must not reopen them or fill the remaining gaps with defaults.

---

## Robustness (D2)

F3–F5 perturbations are also evaluated against the augmented decoders from G3, not only the F1 encoder. The question is whether accuracy bought with synthetic data survives realistic test-time shift — augmentation that improves clean accuracy while degrading robustness has not solved the deployment problem it claims to address.

The registry carries `g3_real_adaptation`, `g3_population_synthetic_adaptation`, and `g3_subject_synthetic_adaptation` slices in [`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml). Unlike encoder and adapter targets they are **run-shaped, not file-shaped**: G3 trains per-subject decoders on the fly, so each target points at a `run_dir` rather than one `refit.pt`. The two replay targets share each population-conditioned *k*=0 head for the same `(adaptation_seed, synthetic_draw_id)`; the runner must resolve that artifact alias across strategies, subjects, and schedules, then still score it once on every held-out subject for that repeat. The unchanged head is likewise one shared artifact at *k*=0, without schedule or adaptation-seed replication, but still has one evaluation per held-out subject. Real adaptation carries an `adaptation_seed` axis at *k*>0, while replay also carries `synthetic_draw_id`; their counts, component keys, pairing and within-subject aggregation remain Pending after D43 fixed the shared seed factory and fail-closed in the registry. See README §6.

For calibration under perturbation, adapted targets resolve their matching frozen G4 temperature table by immutable run ID and settings/model/generator digests. Use the same strategy × *k* T on clean and perturbed predictions; fitting a new T on perturbed final data is forbidden. Registry resolution stays blocked until the calibration artifact is identified and its source protocol is ready. Model-weight aliases and calibrator aliases are distinct: real adaptation at *k*=0 shares F1's head, but G4's query-fitted reference T need not equal F1's all-repetition T. The two replay *k*=0 aliases do share both their weights and their G4 table cell.

---

## Reporting checklist

Before any G-series number leaves this repository:

- [ ] G2 gate passed and reported (AUC with CI, per class, plus memorization and diversity/coverage diagnostics)
- [ ] All five leakage rules verified for the reported configuration
- [ ] All four D29 strategies on the curve, with matched training for the three adapted heads
- [ ] At *k*=0, the unchanged head is cached once and each replay repeat is stored once, with both strategy pairs exposed only as aliases
- [ ] Replay controls share synthetic labels/noise, adaptation seed and fixed-bank draw IDs
- [ ] Per-subject curves shown, not only the mean
- [ ] G3 headline and trials-saved extraction use fixed 18-class macro-F1; accuracy is secondary
- [ ] G4 shows scaled ECE primary, raw ECE and raw/scaled Brier, with the frozen table and fit provenance
- [ ] Multiple nested, class-aware schedules per subject, shared across strategies, with spread
- [ ] *k* stated as total complete active-gesture trials per subject; contained windows also reported
- [ ] Calibration-seen gestures, calibration-unseen gestures, and rest reported separately
- [ ] Rest ownership fixed from an inspected MAT file and enforced without sample overlap
- [ ] The headline "N fewer real gesture trials" claim states the synthetic-window and compute budgets
- [ ] `{1, 4}` result described as session-representative offline calibration, not chronological onboarding
