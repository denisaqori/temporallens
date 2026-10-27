# TemporalLens — Experiment & Ablation Specification

This directory is the **authority** on what each experiment measures and where it can go wrong. Every config file in `configs/` maps to one experiment specified here. If a config and this document disagree, the config is wrong until someone fixes it; without that rule the mapping drifts and nobody notices.

It exists because shorthand ("the adapter", "the LLM test", "generative") had already started to mean different things in different places. Section 2 fixes the vocabulary. **Use the exact terms from Section 2 in configs, code, commit messages, and the paper.**

| Document | Covers |
|---|---|
| This file | Shared protocol, vocabulary, metrics, and the rules both arms obey |
| [generative-arm.md](generative-arm.md) | **Headline.** Conditional generation, synthetic-quality validation, the personalization-efficiency curve |
| [language-arm.md](language-arm.md) | **Secondary.** Soft-prefix conditioning of a frozen LLM, pretraining control, and whole-pipeline engineered-text comparison |

---

## 1. The two arms and what each claims

The project has one shared temporal encoder and two research questions that do not depend on each other. They are evaluated separately, and either can succeed while the other fails.

**Generative arm (headline).** *How many real calibration gesture trials does a new, unseen subject need to reach a target macro-F1, and can synthetic data replace some of them?* The deliverable is a curve, not a single number.

**Language arm (secondary).** *Does mapping signal embeddings into a frozen language model's embedding space improve decoding, calibration, or failure reporting beyond a well-trained encoder?* The deliverable is a comparison with two controls attached. A clean negative result is a valid outcome and must be reported as one.

A result in one arm is never evidence for the other.

---

## 2. Vocabulary — the shorthand that caused confusion

These words were being used loosely. They are now defined, and the definitions are binding.

| Term                          | Precise meaning                                                                                                                                                                                                                                                                                          | Frequently confused with         |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **Encoder**                   | The 1-D CNN mapping a raw EMG window to a 256-d embedding. Trained in Milestone 0, then **frozen** for both arms.                                                                                                                                                                                        | —                                |
| **Projector**                 | The small trainable MLP mapping the 256-d encoder embedding to *N* soft-prefix embeddings of the LLM's hidden size. In configs this is the `adapter:` block, whose `type` is `mlp_projector`.                                                                                                            | "Adapter"                        |
| **Adapter**                   | In this project, a synonym for **projector** — the same object. It is *not* a PEFT/LoRA adapter. Nothing is ever inserted into the language model's own weights.                                                                                                                                         | LoRA adapters                    |
| **Head**                      | The trainable classifier on the frozen model's pooled last-non-padding hidden state. Produces the 18-class logits. Separate from, and downstream of, the projector.                                                                                                                                     | Projector; the LLM's own LM head |
| **Soft prefix**               | Continuous embedding vectors prepended to the text-prompt embeddings, passed via `inputs_embeds`. They are **not** discrete tokens and never index the vocabulary.                                                                                                                                       | "Token space", prompt text       |
| **Frozen**                    | `requires_grad=False` on every parameter of that module, and the module in `eval()` mode. The encoder and the language model are frozen; the projector and head are not.                                                                                                                                 | "Not updated this step"          |
| **Readout**                   | *How* a class prediction is extracted. **Discriminative** = trainable head on a pooled hidden state. **Generative** = the model's own LM head emits a label token. See Section 4.                                                                                                                        | The input path                   |
| **Input path**                | *What* the model receives: soft-prefix embeddings, or engineered text, or nothing (encoder-only). Orthogonal to readout.                                                                                                                                                                                 | The readout                      |
| **Calibration** (two senses!) | (a) **Subject calibration** — the *k* complete labelled gesture trials a new user provides in total; all contained windows are derived training examples, not additional calibration units. (b) **Probability calibration** — whether predicted confidence matches accuracy (ECE). Always qualify which. | each other                       |
| **Generative** (two senses!)  | (a) The **generative arm** — the conditional VAE synthesizing EMG. (b) A **generative readout** — reading a class out of an LM head. Unrelated. Always qualify.                                                                                                                                          | each other                       |

---

## 3. Shared protocol

Every experiment in both arms inherits this unless it explicitly overrides it.

### 3.1 Dataset

NinaPro DB2, **Exercise B** — 17 hand/wrist movements + rest = **18 classes**; 12 channels at 2 kHz, 40 intact subjects. Windows are 400 samples (200 ms). Data is not redistributed with this repository; it is downloaded under its own terms into `data/raw/`.

The reduced class set is deliberate: cross-subject decoding over the full 49+rest set lands near chance and produces illegible curves. The full set is a later extension.

**Labels come from `restimulus` and `rerepetition`. Never `stimulus` or `repetition`.** The DB2 files carry both. `stimulus` is the label the acquisition software *prompted*, and the [data descriptor](https://doi.org/10.1038/sdata.2014.53) is explicit that it does not describe what the subject did: movements "may not perfectly match with the stimuli proposed by our software due to human reaction times and experimental conditions." The dataset authors corrected this offline with a generalized likelihood ratio algorithm that "realigns the movement boundaries by maximizing the likelihood of a rest-movement-rest sequence", and shipped the result as `restimulus`. `rerepetition` is the matching repetition index.

**Easily missed — this fails silently, and it fails worst exactly where it matters.** Loading `stimulus` mislabels every movement onset by the subject's reaction time: windows tagged as a gesture while the arm is still at rest, and rest windows tagged as gesture at the far end. Training runs, loss falls, accuracy looks plausible. Onset is also the region a real-time decoder has to get right, so the damage lands on the most important windows. Assert on load that the label column is `restimulus`.

**Measured on the shipped data, not inferred.** On subject 2's Exercise B recording the two columns disagree on **19.9% of samples** — one in five — and the median gesture onset sits **856 ms later** under `restimulus` than under `stimulus` (range 6 ms to 2.6 s). At a 200 ms window that is roughly the first four windows of *every* gesture trial consisting purely of rest while labelled as movement. "Reaction-time offset" undersells it.

**No baseline subtraction.** The dataset authors do not subtract a rest level, and neither do we. Rest is one of the 18 classes, so removing the rest level would erase the signal that defines it. The only filtering DB2 receives is theirs: a Hampel filter for 50 Hz power-line interference and its harmonics. (The 1 Hz Butterworth low-pass described in the descriptor applies to database 1 only, whose Otto Bock electrodes emit an already RMS-rectified envelope. DB2's Delsys signal is raw.) Normalization is handled separately and per §3.3.

### 3.2 Splits

| `split` value | What it does | Role |
|---|---|---|
| `subject_independent` | Train on one set of subjects, test on subjects never seen in training. Held-out set is `[5, 10, 15, 20, 25, 30, 35, 40]`, frozen in the manifest below. | **The main protocol. Every headline number uses this.** |
| `random_window` | Windows from the **32 development subjects** shuffled without regard to subject; the eight held-out subjects are excluded (D41). | Deliberately optimistic. Exists *only* to demonstrate how much leakage inflates results. Never a headline number. |

**Easily missed:** windows from a single subject are highly correlated. A random split puts near-duplicate windows on both sides and inflates accuracy dramatically. Reporting a `random_window` number without labelling it as the leakage demonstration is a serious error.

**The eight held-out subjects never enter a population fit or a development decision (D41).** No training run that produces a shared or population model, no normalization or class-weight fit, no temperature fit, no hyperparameter or gate decision, and no debug subset uses their data — F2 included, even though nothing downstream consumes F2. They appear only where the protocol is *about* them: the one-shot final evaluation of frozen models (including an instrument fitted inside that evaluation, such as G2's final discriminator), and G3/G4's per-subject adaptation on their own calibration repetitions, scored on their disjoint evaluation repetitions. The leakage F2 demonstrates comes from splitting windows *within* subjects, so it needs no held-out subject to show.

#### F2's fixed random-window assignment (D44)

Build the window table from the data itself, never from a supplied index (D46). Read the processed recordings of exactly the manifest's 32 development subjects, one at a time; index each uncapped at window 400 and stride 100 with the indexer training uses; sort the result by `(subject_id, start_sample)`. Duplicate window IDs and invalid metadata are errors, and held-out subjects' files are never opened. The source identity binds the manifest, every window's corrected label and repetition, the window geometry, and a content digest of each recording's EMG, label and repetition streams, so the frozen split is provably that of the full index of today's data.

Use D43 root 42, scope `f2_random_window`, purpose `outer_test_assignment`, and empty indices `{}`. Call `Generator.permutation(N)` once; the first `(N+4)//5` canonical row IDs are the fixed outer test, exactly `ceil(N/5)`. Canonically re-sort the remaining IDs before applying one independent permutation with purpose `inner_calibration_folds` and empty indices. Divide that result with `numpy.array_split(..., 8)`: for `M=8q+r` non-test windows, folds `0..r-1` contain `q+1`, the others `q`. Fold *i* scores only its assigned windows and fits on the other seven folds; final refit uses the entire non-test pool. Membership arrays return to canonical-ID order and do not prescribe minibatch order.

No subject, gesture, repetition, segment or temporal-neighbour grouping or stratification is applied. Label/repetition values affect source provenance, never random membership or seeds. Audit realized composition descriptively; do not redraw, rebalance or condition acceptance on obtaining preferred proportions. Neighbouring windows may share signal across partitions intentionally. This is D41's seen-subject leakage demonstration, not an unseen-subject estimate.

`data.assignments` builds and saves the assignment, and `data.assignment_registry` freezes it; §3.5 defines its four digests and the freeze. Persist both memberships and the original RNG records before fitting at `data/processed/assignments/f2_random_window/v1/{source_pool_sha256}/assignment.json`, and reuse the registered artifact rather than redrawing it. Source digests are locators and provenance, never RNG keys: changed recordings give a new source digest, which has no frozen assignment until one is built and registered. D44 settles the protocol and the builder, not the still-unwritten eight-fit/refit/temperature runner; `scripts/train_encoder.py` continues to refuse F2.

#### The split manifest

The split is **data, not configuration**. It lives in [`configs/splits/subject_independent_v1.yaml`](../../configs/splits/subject_independent_v1.yaml): the 8 test subjects, the 32 training subjects, the 8 folds of 4 validation subjects (D6), the corrected label columns (D12), and the repetition policy (D13).

Load it through `SplitManifest.load()`, never by reading the YAML directly. That call recomputes a SHA-256 over the split-defining fields and refuses to return anything whose digest does not match, so an edit to the file cannot quietly change what a number means. It also checks the structure: roles partition all 40 subjects, every training subject validates in exactly one fold, no fold validates on a test subject, and the calibration and evaluation repetitions stay disjoint.

Every **reportable subject-independent config** names this file through `dataset.split_manifest`. It does not repeat `held_out_subjects` or the calibration/evaluation repetition lists: consumers derive those values from the verified object and record `manifest_hash` in the run and checkpoint metadata. Tiny/local configs instead declare `dataset.debug_split`; their deliberately reduced subject holdout exists only to exercise the pipeline and can never produce a reportable number.

Fold assignment is **strided, not blocked** — the *i*-th training subject goes to fold *i* mod 8. This gives every fold one subject from each quarter of the ID range instead of one contiguous ID block. It is a deterministic range-balancing rule, not a claim that subject ID is an acquisition timestamp; the DB2 descriptor does not establish that chronology (D17). Each fold's training set is the complement of its validation set and is deliberately not written down, so a subject's role is stated in exactly one place.

**The manifest is frozen.** Changing a split invalidates every number computed under the old one, and the language arm is a comparison against F1's reference row. A new split means a new versioned manifest and a DECISIONS row — never an edit in place.

#### Repetitions

DB2 records **6 repetitions** of every movement by every subject. The rule is short: **the subject is the split unit, so all 6 repetitions follow their subject.** A held-out subject contributes all six to the held-out pool; a training subject contributes all six to training. Repetitions never cross the subject-level train/validation/test boundary. F1 evaluates on the full held-out pool; G3/G4 make the D13 within-subject partition, using repetitions `{1, 4}` only for subject calibration and `{2, 3, 5, 6}` only for evaluation.

This is a deliberate departure from the dataset's own benchmark, which splits *by repetition within* subject — "repetition 2 and 5 in database 2" to test, the rest to training. That measures how well a decoder generalizes to a new session from the same person. We measure generalization to a new person, which is the deployment question for a wearable and the harder of the two. **Our numbers are therefore not comparable to theirs**, and the ~75% they report for DB2 is a different task (50 movements, within subject), not a target to beat.

**Why the rule needs stating rather than assuming.** The descriptor measured amplitude drift across repetitions and found "a significant (P<0.05) dependence on the repetition ... in 12.5% (database 2) of the subjects", warning that "attention should be paid to it while splitting movement repetitions into training set and test set." Keeping whole subjects on one side means that drift never straddles a split boundary here. It does resurface in subject calibration, where complete per-gesture trials indexed by `rerepetition` are the natural unit — see [generative-arm.md](generative-arm.md), G3.

### 3.3 Windowing and normalization

`window_size: 400`, `stride: 100` for real runs (`200` for debug configs — fewer, less overlapping windows).

**Windows never cross a segment boundary (D24).** A segment is a maximal run of samples sharing the same `(restimulus, rerepetition)`. Windows are cut within segments; where fewer than `window_size` samples remain at a segment's tail, they are dropped.

A window spanning rest→gesture contains two states and can carry only one label, so whichever the code assigns is wrong for part of the window. That is the same silent mislabelling D12 exists to prevent, one layer further down, and it concentrates at movement onset — the region a real-time decoder most needs to get right. The alternative, labelling by majority or centre sample, keeps those windows but puts a transition inside them; given how much weight D12 places on onset correctness, that is the worse trade. A dropped window costs a little data; a mislabelled one costs the ability to trust the number.

This generalizes the rule the generative arm already assumed: G3's `forbid_boundary_crossing_windows` is now protocol-wide, not a G3 setting.

| `normalize` value | Statistics come from | Used by |
|---|---|---|
| `train_subject_covered_sample_stats` | The current training **subjects** only: 28 fold-training subjects during cross-validation, all 32 development subjects for the refit, or the explicitly reduced subjects in a debug holdout | Subject-independent and subject-holdout debug runs |
| `train_split_covered_sample_stats` | The training **split** only | Random-window runs |

Both names say `covered_sample` because D31 fixes *which* samples within the partition: the ones training windows actually cover, **each counted once**. Not per-window — an interior sample is inside four windows at stride 100 and would be weighted 4x. Not whole recordings either, which would include the `(L - window) mod stride` tail that D24 leaves windowless: 0.54% of the signal, and measurably quieter at std 0.445–0.928 of the covered samples. On subject 2 the three candidate fitting sets agree on per-channel std to within 1.4%, so this is a freeze decision rather than an accuracy one — covered-once wins because it is stride-independent (0.000% versus 0.230% between stride 100 and 200), and these statistics are reused by consumers that may window differently. An earlier revision called these values `*_global_stats`, which named the partition but left the fitting set unstated — the one thing D31 is about.

**Easily missed — this is the most common silent leak in the whole project.** Normalization and other preprocessing statistics must be computed on **the current training partition only** and then applied unchanged to validation and test. For each cross-validation fold that means its 28 training subjects, not all 32 development subjects; the refit recomputes statistics on all 32. Class weights follow the same partition rule. Computing either quantity over the full dataset before splitting leaks validation/test distribution into training and inflates every downstream number. It leaves no trace in the logs. G3's subject-conditioning embedding is the explicit exception: at *k*>0 it may use only the windows contained in the selected *k* trials, as defined by the generative-arm leakage rules; this never permits subject-specific normalization.

### 3.4 Metrics

| Metric | What it captures | Why it is here |
|---|---|---|
| `accuracy` | Fraction of correctly classified evaluation windows; higher is better | Readable secondary measure, weighted by the observed class frequencies; rest accounts for about half the windows |
| `macro_f1` | Equal-weight mean of the 18 class F1 scores; each F1 combines precision and recall | Primary classification endpoint. Fixed labels 0–17, undefined class F1 = 0.0; missing support classes remain in evaluation |
| `per_subject_accuracy` | Accuracy for each held-out subject separately | **Report the spread, not just the mean.** A mean over 8 subjects can look respectable while one of them sits at chance |
| `per_class_precision` | Precision for each of the 18 classes separately | When the model predicts this gesture, how often is it right? Catches classes the model over-predicts |
| `per_class_recall` | Recall for each of the 18 classes separately | How much of this gesture does the model find? Catches classes it quietly misses, which accuracy hides and macro-F1 averages away |
| `confusion_matrix` | Which movements are mistaken for which | Anatomically adjacent gestures confuse; the pattern is a result |
| `expected_calibration_error` | Occupancy-weighted absolute difference between mean top-label confidence and correctness in each bin; lower is better | Measures a binned confidence–accuracy mismatch. Raw and scaled versions have distinct meanings; low ECE alone does not establish useful decoding |
| `per_subject_expected_calibration_error` | ECE computed separately for each evaluation subject | D23 requires these values wherever ECE is reported; pooling can hide a subject at high miscalibration |
| `per_subject_macro_f1` | Macro-F1 for each evaluation subject separately | D26 makes the subject the unit of inference, so the headline metric needs per-subject values too |
| `per_subject_brier_score` | Brier score for each evaluation subject separately | Same rule as ECE: a pooled proper score can hide one badly-scored subject |
| `per_subject_overconfidence_error` | Overconfidence error for each evaluation subject separately | Paired cross-arm comparison (D26) needs the per-subject values, not a pooled number |
| `overconfidence_error` | Confidence-weighted **positive** gap, summed over bins: `Σ (count(Bₘ)/n)·conf(Bₘ)·max(conf(Bₘ)−acc(Bₘ), 0)` | Isolates the direction that hurts a wearable — confidently wrong. Deliberately one-sided, so read it beside ECE, never alone (D25) |
| `mean_confidence_when_wrong` | Mean confidence over misclassified windows | **Descriptive only.** Answers "when the device is wrong, how sure does it sound?" for the structured-report demo. Never compared across arms — it has no calibrated reference value (D25) |
| `brier_score` | Sum of squared errors across all 18 class probabilities, then mean over evaluation windows; unhalved range [0, 2], lower is better (D39) | Proper score that reflects calibration and predictive discrimination together. Report raw and scaled versions wherever ECE is reported |
| `robustness_drop` | Accuracy loss from clean → perturbed | Robustness experiments only |

The table above is the shared metric registry; the "Metrics reference" tables in [generative-arm.md](generative-arm.md) register generation-specific keys. Every value under an experiment config's `evaluation.metrics` must appear as a first-column key in one of those two registries, and `tests/test_experiment_configs.py` enforces that registration. Registration gives name-level coverage, not proof that every metric is executable; Pending decisions remain binding.

**Metric roles and interpretation (D33–D34).** Macro-F1 is the primary classification endpoint, including G3's personalization curve. Temperature-scaled ECE is the primary probability-calibration endpoint in every reportable experiment that measures calibration. Raw ECE and raw/scaled Brier are mandatory companion outcomes. There is no combined score or rule that a win on any one metric is an overall win; keep their roles fixed before inspecting results.

**Brier reduction (D39).** For each window, sum `(p_c - one_hot_c)^2` over all 18 classes, then average window scores within the subject. This is the unhalved multiclass score in **[0, 2]**: a certain wrong prediction contributes 1 for missing the true class and 1 for predicting the wrong class. Do not divide by 18 or by 2. The familiar binary score in [0, 1] scores one binary probability; halving the full-vector score is another valid convention, but not this project's. The existing `evaluation.metrics.brier_score` already uses the chosen class-sum/window-mean reduction. Across-subject or repeated-run aggregation is a separate rule; this does not resolve G3/G4's Pending aggregation choices. See the [scikit-learn multiclass Brier definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html).

| Outcome | Meaning | Limit |
|---|---|---|
| Raw ECE | Reliability of the head's native confidence, before probability recalibration | Includes scale effects that a development-fitted calibrator may correct; does not directly measure confidence on errors |
| Temperature-scaled ECE | Reliability of the deployed head plus the frozen, development-fitted calibrator | Evaluates that complete procedure, not intrinsic confidence quality or guaranteed optimal test calibration |
| Raw/scaled Brier | Quality of the full probability vector in each corresponding version | A lower score can reflect better discrimination, better calibration, or both; compare it with ECE and macro-F1 |
| Overconfidence error | The one-sided, confidence-weighted positive bin gap (D25) | An underconfident model can score zero; it is meaningful only alongside ECE |
| Mean confidence when wrong | How confident the model sounds on its errors | Descriptive only; `NA` with no errors, and never a cross-arm calibration claim (D25) |
| Per-class and seen/unseen recall | Which gestures are recovered and whether adaptation transfers beyond demonstrated classes | Keep rest separate and empty subgroups `NA`; subgroup recall is not the overall macro-F1 |

Interpret all declared outcomes together. Better raw ECE with no detected scaled-ECE advantage is consistent with a native-confidence benefit that this scaling procedure largely corrects; it does not prove that synthesis has no other benefit. Better scaled ECE with worse Brier or macro-F1 needs an explicit tradeoff explanation rather than a blanket claim of improvement. A predictor can be calibrated while carrying little useful information. Top-label ECE also does not establish that every class probability is calibrated. See [Vaicenavicius et al., AISTATS 2019](https://proceedings.mlr.press/v89/vaicenavicius19a.html) for that distinction and the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) for why proper scores and calibration gaps can disagree.

Config `evaluation.metrics` entries name metric **families**. Reportable probability outputs append `_raw` or `_temperature_scaled` to the family name, including per-subject families and G4's curve: for example `expected_calibration_error_raw`, `expected_calibration_error_temperature_scaled`, `brier_score_raw`, `brier_score_temperature_scaled`, and `ece_vs_k_curve_temperature_scaled`. Recompute confidence and bin assignments from each probability version. Never compare scaled ECE with an unlabelled raw Brier score. F0's existing unsuffixed smoke-test outputs remain non-reportable; the result-schema and runtime work is separate from this protocol amendment.

**Where each metric is computed (D6).** Cross-validation and testing run the same metric set through the same code path, which is what makes them comparable. Each of the 8 fold models is evaluated on the held-out test subjects and those values are averaged over the folds; that is the extended analysis. The refit model is evaluated on the same test subjects, and its numbers are the ones reported, including the F1 reference row.

Produce the confusion matrix for the cross-validation as well as the test evaluation. The cross-validation matrix tells you whether a confusion holds up across folds. Two gestures that collide in all eight are worth writing about; two that collide in one fold are probably noise.

**Easily missed — average fold confusion matrices, never sum them.** All 8 fold models are evaluated on the *same* test subjects, so summing counts every test window eight times and implies eight times the data. Take the element-wise mean instead, which leaves the matrix on the scale of one evaluation. Same goes for any other count-based metric.

#### Measuring calibration (D23)

Calibration is only observable as a *frequency*. A single prediction at confidence 0.7 is right or wrong; there is no such thing as one prediction being "70% right". So predictions making similar claims are pooled and the group's accuracy compared against its mean confidence. **That pooling is the binning, and the bin count is an artifact of estimation, not part of what calibration is** — which is exactly why it has to be fixed in the spec rather than chosen per run.

| | Rule |
|---|---|
| Binning | **Adaptive (equal-mass / quantile)**, not equal-width |
| Bin count | **M = 10** nominal quantile bins, fixed everywhere; confidence ties can leave fewer nonempty bins (D39) |
| Quantity | **Top-label** ECE — the max softmax probability |
| Recalibration | **Temperature scaling**, fitted on development predictions from the corresponding model procedure; sources fixed below (D33 amends D23) |
| Reporting | Scaled ECE primary; raw ECE and **raw/scaled** `brier_score` alongside, with explicit output suffixes |
| Uncertainty | Resample **subjects**, never windows. Cross-arm comparisons are **paired** |
| Per-subject | Always shown, whatever the headline aggregation |

**There is finite-sample bias, and it is not subtractable.** In the simulation used to select M=10, a perfectly calibrated model measured ECE ≈ 0.037. That number is not a universal project threshold: it depends on sample count, the confidence/label distribution, and dependence among windows, all of which differ across pooled, per-subject, F2, and G4 evaluations. A threshold used in a report must be recomputed and retained for that exact design. It can never be subtracted as an offset: in the motivating simulation a true ECE of 0.050 measured 0.058, not 0.087.

**Why M = 10.** In that motivating simulation, the measured bias rose monotonically with M — about 0.026 at M=5, 0.037 at M=10, 0.045 at M=15, 0.058 at M=25 — while the run-to-run spread stayed near 0.009. Below 10, the opposite failure appeared: miscalibration that changed sign across the confidence range was cancelled inside bins that straddled the flips, and M=5 under-reported a true 0.065 as 0.053. M=10 was the smallest tested bin count that still resolved it. These numbers justify the fixed estimator choice; they are not reusable confidence thresholds for every experiment.

**Why adaptive bins.** A confident decoder's confidence distribution is skewed, so equal-width bins strand several on nearly no data — in one check, 2 of 15 bins were empty and the smallest non-empty held a single window, whose "accuracy" is 0 or 1. Adaptive binning fixes *placement* only; it does **not** remove the bias in M, so fixing M is still required.

**Confidence ties (D39).** Compute confidence-only quantiles at `j/10` using linear interpolation, with outer edges 0 and 1. Collapse duplicate cutpoints; a confidence equal to an interior cutpoint belongs to the upper bin, with confidence 1 retained in the final bin. Keep identical confidences together, omit empty bins from the weighted sum, and report the effective nonempty bin count and each occupancy. Do not split ties by input order or correctness labels to manufacture ten populated bins. Ten is the fixed requested estimator setting, not a guarantee of ten distinct confidence groups. Recompute boundaries separately after scaling and for each subject-level metric. The effective bin count belongs to the retained diagnostics and design-specific bias analysis.

**Why temperature scaling, and what it changes.** Use `softmax(logits / T)` with a finite `T > 0`. Fit T by ordinary negative log-likelihood (NLL), not by minimizing the reported binned ECE or by reusing the classifier's inverse-frequency class weights. A scalar can correct part of the global confidence scale; it need not remove class-specific or subject-specific errors. It preserves each example's argmax and class ordering, so accuracy, macro-F1 and confusion matrices do not change. Multiclass confidence ordering **across examples** can change. Scaled ECE therefore does not isolate ranking quality. The claim is *"better calibrated after the same development-fitted scaling procedure"*, not "optimally scaled". The original method fits T on held-out validation logits: [Guo et al., ICML 2017, §4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

**Where T comes from (D33, amending D23).** Fit only from the development population; freeze the model procedure and fitted calibrator before final-test inference. The fit gives each development subject equal weight, each recorded repeat equal weight within that subject, and each query window equal weight within that repeat. Keep natural query-class proportions. Fit on every repeat's logits under that weighting, not on averaged logits or duplicated aliases. The corresponding objective is the mean over subjects of their mean repeat-level NLL. Subjects with more windows or repeats must not silently dominate the population calibrator.

| Consumer | Development predictions used to fit T | Frozen parameter consumed at test |
|---|---|---|
| F1 | Pool subject-OOF predictions from eight **fresh fixed-budget** complement-28 encoders, using all repetitions of each outer four subjects; train every encoder for D18–D19's accepted refit budget | One T in `refit.pt`'s `model_config`, with fitting provenance; eight per-fold Ts are optional heterogeneity diagnostics, never the refit's median estimator |
| F2 | Pool window-OOF predictions from eight folds **inside the non-test 80% window pool**; fold models and final refit use the same declared fixed training budget | One T in the F2 refit `model_config`; retain the fixed final 20% window test split |
| L2–L4 | Held-out predictions from each arm's own model procedure; encoder and trainable modules must respect its eventual fold boundaries | Remains blocked until the cross-validation/refit and repeat contracts define the source; no F1 temperature inheritance |
| G4 | After adaptation settings freeze, run all four strategies on D28's outer development subjects; support comes from manifest calibration repetitions, query from its disjoint evaluation repetitions | One frozen strategy × k population-temperature table, described in generative-arm G4; no fitting on final-subject trials |
| F3–F5 | No new temperature fit | Consume the target's frozen calibration parameter: checkpoint metadata for population/language targets, matching G4 table for adapted heads; never recalibrate on perturbed test data |

F1's calibration encoders reuse the planned D28 development fits (§5.1), so they match the final refit's fixed epoch budget instead of each fold's validation-selected epoch. Population fitting excludes the current outer four subjects; normalization and class weights are fitted on the same complement. G2 and G4 reuse the corresponding generator fits after G1's remaining rules are frozen. These are specified artifacts, not currently implemented training machinery.

The selected `folds/fold{k}/best.pt` models used in extended final-test analysis fit their own T from that model's outer-fold validation logits after checkpoint selection. Those temperatures calibrate those analysis models only; they never replace or feed the fixed-budget OOF refit T. Fitting and model selection reuse the outer fold, so the scaled fitting score is not a reportable validation estimate. **D40 supplies scaled validation diagnostics** by leave-one-subject-out temperature fitting within that outer fold: freeze the selected checkpoint, fit T on three complete validation subjects, score only the fourth, and repeat for all four. Use the same fixed fold model for every fit and D33's equal-subject/repeat weighting, then average the four subject-level metrics equally. This takes four scalar fits per fold (32 overall), with no extra CNN training. Persist the model identity, fitting/scoring subject IDs, temperatures, weights and fit diagnostics. No scored subject enters its own calibrator fit, and these temperatures never feed the refit estimator or a median-temperature substitute. Raw validation/classification metrics retain D20's selected-checkpoint rule. No runner may quietly use the refit's T for every fold model.

These are **development diagnostics conditional on the selected checkpoint**, not unbiased nested-CV performance: the four subjects still influenced checkpoint selection. The cross-fit removes calibrator fit-on-score reuse, not that model-selection reuse. Selected fold models use the all-four outer-validation T above for their extended final-test analysis; the refit keeps its separate fixed-budget pooled-OOF T. The final eight subjects remain untouched by every fitting procedure.

F2's fold preprocessing and class weights use only that fold's training-window fitting set. Persist the outer test assignment and inner fold assignments. Its OOF windows are out of the model's fitting indices, but remain correlated with overlapping training windows: this intentionally demonstrates leakage, not independent subject/trial generalization. Eight folds process about seven full-pool fits' worth of training data, plus the refit, under equal epoch budgets; fitting T itself is cheap. This is a relative data-pass calculation, not a measured runtime. D43/D44 fix the seed factory, PCG64, zero-based indices and exact outer/inner assignments (§3.2). F2 is protocol-ready; its dedicated training/refit/temperature harness is still unimplemented. D39 fixes the shared numerical fitting recipe. F2 currently retains its declared 30-epoch budget for the inner fits and refit; F1's budget is development-selected. This diagnostic therefore does not isolate the split as the sole cause of an F1/F2 difference. OOF calibration followed by an all-development refit is also documented in the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

**Development reuse is explicit.** A new post-selection pass on the same subjects/query repetitions is not independent of model selection. It separates the fitting stages and permits replay-arm predictions to be produced; it does not create a fresh validation population. Guo et al. allow the hyperparameter-validation set to fit the calibrator. Treat its scaled scores as fitting diagnostics, not unbiased evaluation, and keep the final test untouched. Calibrated validation metrics require their own held-out/cross-fitted calibrator if they are to be reported as out-of-fit estimates; do not score the fitted T on its fitting predictions and call that validation performance.

**Fitted-parameter contract.** Population and language refit checkpoints carry normalization plus T and its provenance in `model_config`. G4 carries a frozen external table because it has many adapted heads. Artifact identity includes the prediction-source digest, exact subject/fold and repetition identities, normalized fitting weights, config/settings digests, model/generator parent digests and aliases. Record each T and fit diagnostics; failed, non-finite, non-positive or unresolved boundary fits block reportable scaling. There is no automatic fallback to T=1 or post-result smoothing. **D39 fixes the numerical recipe**, not a promise that every fitted cell will pass.

Minimize the weighted NLL in inverse temperature `beta = 1/T` using `scipy.optimize.minimize_scalar(method="bounded")`, float64 and stable log-sum-exp. The parameter bounds are `[0.01, 100]` (equivalently T in `[0.01, 100]`), `xatol=1e-8`, and `maxiter=500`. The NLL is convex in beta: its second derivative is the weighted variance of the logits under the scaled softmax. The bounds are predeclared numerical guards, not measured optimal limits. Normalize D33's fitting weights; require finite logits/weights/objectives, solver convergence, finite positive T and an interior optimum. Check boundary solutions using the NLL derivative: if the lower-bound derivative is nonnegative or the upper-bound derivative is nonpositive, block reportable scaling instead of clipping or widening the range after seeing results. This also blocks an unidentified flat fit. The fitted NLL must not exceed the feasible T=1 objective by more than `1e-8`. Persist solver/parameter identity, bounds/tolerances, iteration count, convergence, boundary derivatives, T and both NLL values. See the [SciPy bounded-solver controls](https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-bounded.html). The solver, table/checkpoint enforcement and cross-fitted validation runners are still future implementation; resolving their protocol does not make F1's currently refused harness executable.

*Known approximations:* complement-28 model temperatures transfer to a refit trained on 32; G4 also transfers from fold-local generators to the final all-32 generator. Reused development subjects carry model-selection dependence. The final subject bootstrap is conditional on the frozen models and calibration artifacts. One scalar fitted on 32 subjects is a modest parameterization, not a guarantee of stability. Temperature scaling can also deteriorate under distribution shift: [Ovadia et al., NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).

**Easily missed — resample subjects, never windows.** Windows overlap 75% at stride 100, and the ~154 windows from one five-second contraction are one event seen 154 times. Bootstrapping windows treats them as independent and produced a 95% interval **20× too narrow** than the same data resampled by subject (width 0.005 against 0.100, identical point estimate). The subject is the unit because the protocol generalizes to a *new person*. The honest interval is wide; the way to recover power is **pairing** — every arm is scored on the identical 8 test subjects, so per-subject differences control the shared subject variation. It removes the additive subject-level component, not a subject-by-arm interaction, so it reduces the spread rather than eliminating it.

**Report `brier_score` alongside.** It is a proper scoring rule with no binning hyperparameter, so it cannot be a bin artifact. Brier score also reflects probability sharpness and discrimination, whereas top-label ECE estimates a binned calibration gap, so the two can legitimately rank arms differently. Investigate a disagreement for implementation errors, then report it transparently if it remains.

**Two metrics, two names (D25).** `overconfidence_error` is the binned, confidence-weighted positive gap and inherits everything above — same bins, same scaling, same resampling unit. It is one-sided on purpose: a model made uniformly timid scores a perfect 0 on it while being badly miscalibrated, so it is only interpretable next to ECE, which catches both directions.

`mean_confidence_when_wrong` is a separate, **descriptive** quantity. It is reported because it answers a question the structured-report demo needs — *when the device is wrong, how sure does it sound?* — but it is not a calibration metric and must never be compared across arms. Three *perfectly calibrated* models scored 0.25, 0.40 and 0.67 on it purely from differing confidence profiles, so no value of it means "calibrated"; and scaling a model's confidences down improves it from 0.69 to 0.31 without changing a single prediction. Lower is not better.

**The output contract (D25, D23).** Temperature scaling changes the confidence distribution, and adaptive bins are quantile-based, so the edges move with it. The rules:

| | Rule |
|---|---|
| Bin edges | **Recomputed after scaling.** The scaled distribution is what is being assessed, so pre-scaling quantiles would bin the wrong variable |
| Result keys | ECE and `overconfidence_error` are emitted **twice**, raw and temperature-scaled, under distinct keys — never one key whose meaning depends on a config flag |
| The descriptive statistic | `mean_confidence_when_wrong` is reported on the **raw** probabilities only. It is descriptive, and scaling it would invite exactly the cross-arm comparison it must not be used for |
| Zero errors | `mean_confidence_when_wrong` is **`NA`**, never 0. A model with no errors has no confidence-on-errors, and 0 would read as "perfectly humble when wrong" |
| Per-subject | Every calibration metric also emits its `per_subject_*` form (D26) |

#### The inferential unit (D26)

D26 generalizes the resampling rule above from calibration to **every** reported metric.

| | Rule |
|---|---|
| Resampling unit | **Subject.** Never the window, never the trial |
| Per-subject values | **Always reported** — accuracy and macro-F1 as well as the calibration family |
| Fold spread | Reported **separately**, labelled *training* variability (§5.3); never pooled with subject spread |
| Cross-arm claims | **Paired** over the identical 8 test subjects: all 8 per-subject differences, plus a 95% paired bootstrap interval |
| Significance | **No unpaired tests at n=8.** Wilcoxon signed-rank is available as a paired nonparametric check |
| Paired interval | **Percentile bootstrap**, 10,000 resamples of the 8 test subjects, seeded from `experiment.seed` |
| Minimum reportable difference | A difference is reportable when its paired interval **excludes zero**; for ECE it must additionally exceed the estimator floor (§ above). Stated beside every claim |

**Fold spread and subject spread answer different questions and must not merge.** Fold-to-fold variation says how much the result depends on *which 28 subjects trained the model*; subject-to-subject variation says how much it depends on *who is wearing it*. One interval pooling both answers neither.

**Pairing is what makes n=8 workable**, and it is why the wide marginal interval is not fatal: the shared subject variation is the dominant component and pairing controls it. Interval resolution is still limited at n=8, so reports state that rather than implying more precision than 8 subjects can support.

### 3.5 Reproducibility

`seed: 42` everywhere. Every run writes `run.json` (config, git commit, summary) and `metrics.jsonl` through `RunLogger` — this is not optional and not W&B-dependent. W&B is offline by default; nothing may depend on a network service to be reproducible.

**Keyed protocol randomness (D43).** Use a fresh `numpy.random.Generator(numpy.random.PCG64(seed))` for each declared randomized assignment, through `utils.randomness.assignment_rng`. This specifies the assignment backend, not a single mutable global stream and not a replacement for PyTorch's training RNG. Adding an unrelated draw or changing task order must not redraw a split or acquisition schedule. Keep the root `experiment.seed` at 42; D38's existing direct-seed F1 initialization and D26's direct-seed bootstrap contract remain unchanged.

The seed contract is `temporallens_rng_v1`. Form a mapping with exactly `contract`, `root_seed`, `scope`, `purpose`, and `indices`. `contract` is that version string; `root_seed` is an unsigned 64-bit Python integer; `scope` and `purpose` are lowercase ASCII identifiers matching `[a-z][a-z0-9_]*`; `indices` maps identifiers of the same form to nonnegative Python integers, never booleans or floating-point values. Serialize with sorted keys, compact separators `(',', ':')` and `ensure_ascii=True`, encode as UTF-8, take SHA-256 digest bytes `[0:8]`, and interpret them as an unsigned big-endian integer. Never use Python's process-randomized `hash()` or derive protocol assignments from the entire config digest: a logging change must not change the scientific split. `derive_seed` and `randomization_provenance` share this one implementation.

`scope` identifies the **randomization group**, not automatically the literal experiment name. A purpose names one semantic random object; only identifiers defining that object belong in its key. G3 and G4 share the acquisition schedule scope, purpose and IDs. D29's replay pair shares labels/noise and must not include the replay strategy in their shared stream keys; its shared *k*=0 replay object must not include subject or acquisition schedule. Language seed indices are paired under D36, but which component streams are actually shared still needs its own declaration. Separate independent purposes such as outer assignment, inner assignment, support rotation, data order and latent noise; do not couple them through one mutable generator. Generated fold, schedule, seed and draw indices start at zero; existing subject, repetition and class identifiers keep their dataset/manifest numbering.

Persist the exact key, derived seed, PRNG/backend and library version **and the realized assignment**: ordered stable window IDs for folds/splits, ordered trial IDs for acquisition schedules, or a lossless draw ledger for pseudo-calibration choices. Record the source-pool identity and SHA-256 of the assignment artifact; persist before fitting and consume that artifact rather than independently redrawing it for each experiment. High-volume stochastic training records component seeds/backend and draw identities rather than every dropout bit; a resumable checkpoint additionally needs the relevant RNG states. A seed alone is not an archival assignment format: [NumPy's compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html) allows higher-level generator operations to change across versions, whereas [PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html) provides a stronger guarantee for its underlying integer stream.

D43 fixes the common seed/generator/index/persistence convention; D44 fixes F2 allocation/component identities, and D45 fixes G3/G4 acquisition schedules. G1's rotation-selection rule/component keys, language component pairing/counts, and adaptation repeat domains remain explicitly Pending. Their runners must not substitute defaults for those choices.

**Four digests, four questions.** Every assignment artifact carries a *material* digest over the stable IDs and their memberships or order, which any reuse must reproduce; a *source-pool* digest over the inputs it was drawn from, which says whether it belongs to today's data; a *logical-envelope* digest over the whole record including RNG provenance, which says whether the record is internally consistent; and the *exact-file* digest of the saved bytes, returned by `save_assignment`, which says whether this is the file that was frozen. Only the last needs an anchor outside the file: whoever rewrites a file can recompute the other three.

**Frozen assignments are registered in Git (D46).** `configs/splits/assignment_registry.yaml`, beside the split manifest, records each artifact's kind, logical identity (none for F2; subject and schedule index for G3), source-pool, material and exact-file digests, and repository-relative path. Registration is the one-time freeze: it loads the saved file against the digest the writer returned and the current source digest, re-derives the membership under the NumPy version that drew it, refuses any difference, and only then appends an entry. It never replaces an entry, and under a different NumPy version it refuses rather than skipping the check. Later loads resolve the entry and verify against it without redrawing, so a frozen assignment stays authoritative across library versions. Commit the registry before a reportable run consumes an entry; non-debug runs refuse an uncommitted tree, so an uncommitted entry cannot reach a reported number. The artifact files stay under `data/`, untracked, because they carry NinaPro labels.

**Easily missed:** seeding Python/NumPy/PyTorch does not make MPS and CUDA produce identical results. Do not compare a local MPS number against a cloud CUDA number and call the difference an effect. Cross-device comparisons need the same device.

### 3.6 Class imbalance

Rest is over-represented in DB2 Exercise B — measured on subject 2 at `stride: 100`, **rest is 50.2% of all windows**, about **17.5×** the median gesture class (338–761 windows each). A model that predicted rest and nothing else would score 50% accuracy, which is why `macro_f1` and not `accuracy` is the headline. Handle the imbalance in the loss, never by resampling:

|                                                  | Rule                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Population-model training                        | `class_weighted_cross_entropy` — weights inversely proportional to class frequency in the current training partition                                                                                                                                                                                                                                           |
| G3/G4 adaptation (D28; amends D11 for this case) | Reuse the F1 population-weight vector for the real source in all three adapted strategies: fit on each fold's 28 training subjects during development and all 32 for the final pipeline. A small-*k* subject support set omits classes, so recomputing inverse-frequency weights there is undefined. The class-balanced synthetic bank uses unweighted mean CE |
| Resampling                                       | **Never.** No oversampling, no undersampling, on any split                                                                                                                                                                                                                                                                                                     |
| Test set                                         | **Never balanced.** A balanced test set does not estimate deployment performance                                                                                                                                                                                                                                                                               |

The group's [prior work](https://arxiv.org/abs/2303.10336) oversamples within each subject to equalize classes. That is fine for discrete trials, but not for this data. Windows are 400 samples at stride 100, so they already overlap by 75%, and duplicating one drops near-identical copies of the same signal into a batch. §3.2 and §3.3 both warn against that correlation; oversampling would let it back in. Weighting the loss rebalances the classes without duplicating anything.

Do not combine class weighting with class-balancing resampling. Macro-F1 already accounts for the imbalance and the population-model weighted loss already corrects for it. In G3/G4, D28's class-balanced synthetic bank is an explicit intervention with an unweighted source loss; the real support is never resampled. Resampling on top would leave `accuracy` describing a class distribution the model never meets in deployment.

Debug-only `max_windows_per_subject` caps are a non-reportable plumbing exception, not a class-balancing method. Their seeded subsample retains every class present so the smoke test exercises all output logits; no scientific metric may be reported from a capped run.

---

## 4. Readout: discriminative vs. generative — and the rule

This axis is **orthogonal** to which arm an experiment belongs to and to which input path it uses. Conflating the two is what motivated this document.

- **Discriminative readout** — a trainable MLP head pools the frozen model's last non-padding hidden state (`pooling: last_token`) and emits 18 logits. Trainable: projector (if present) + head.
- **Generative readout** — the frozen LLM's own LM head is teacher-forced to emit a label token. Trainable: the projector only.

### The binding rule

> **One readout, held identical across every config in a comparison.**

If the method used a generative readout and a control used a discriminative one, any observed difference could be caused by the readout rather than by the thing under test, which defeats the control's purpose.

**Decision: the discriminative readout is primary for the language arm.** Three reasons:

1. **The random-init control requires it.** A randomly initialized transformer has no usable LM head, so it cannot support a generative readout. A trainable head is the only readout all three language-arm configs can share — and sharing is the point.
2. **It still proves the thesis.** "Does language pretraining help?" is answered by *frozen-pretrained beats matched-random with an identical head* — same readout, weights the only difference.
3. **Calibration metrics.** ECE and overconfidence are standard and well-defined on classifier logits; deriving a calibrated 18-class distribution from label-token probabilities is fiddly and would differ per arm.

The head specification is therefore **byte-identical** in all three language-arm configs:

```yaml
head:
  type: mlp
  pooling: last_token
  input_dim: auto_from_llm_config
  hidden_dim: 256
  dropout: 0.1
  num_classes: 18
```

A generative readout remains available as an explicitly secondary ablation (L5) on the frozen-LLM method only — never as a control's readout.

**Easily missed — `pooling: last_token` means the last non-padding token, never literally the final position.** Defining the pooled quantity this way, rather than by which padding side the tokenizer happens to use, is what keeps the implementation from drifting from the intent.

| How you index it | Correct when |
|---|---|
| `hidden_states[:, -1]` | **Left**-padding only |
| `attention_mask.sum(-1) - 1` | **Right**-padding only, and only with no interior zeros in the mask |
| Rightmost index where the mask is 1 — `L - 1 - mask.to(torch.int64).flip(-1).argmax(-1)` | **Either** padding, and under interior masking |

Prefer the third. Note that `sum(-1) - 1` is *not* a general "find the last real token" formula even though it reads from the mask: it assumes the real tokens occupy indices `0..n-1`, so it returns 1 instead of 2 for a left-padded `[0,1,1]`.

**Then assert**, because the general form still degrades silently on an all-pad row — `argmax` returns 0, the index becomes `L-1`, and you pool a pad again. Check that every row has at least one unmasked token and that the mask at the chosen index is 1.

**Why this hides.** The longest sequence in a batch is correct under either padding, so the bug corrupts only the short rows of mixed-length batches. Training runs and loss falls.

**Soft prefixes make the mask contract load-bearing.** For every tokenizer-backed soft-prefix path (L1–L3), prepend a mask of ones for the *N* soft-prefix embeddings to the tokenizer's text mask: `combined_attention_mask = cat([prefix_mask, text_attention_mask], dim=-1)`. Derive both the rightmost non-padding pooling index and explicit `position_ids` from that same combined mask, then pass the combined mask and positions with `inputs_embeds`. With left-padded text this mask contains interior zeros between the prefix and text, which is exactly why the rightmost-mask index is used. After validating that the mask is binary, compute positions by cumulative non-padding count and set masked slots to a safe value such as zero. Direct Llama calls with `inputs_embeds` must not rely on default sequential positions: they offset real text tokens by the pad count. L4 has no soft prefix and uses its tokenizer mask directly.

**Easily missed — `input_dim: auto_from_llm_config` is load-bearing.** The hidden size must be read from the model's own config at runtime, never hard-coded. Llama 3.2 1B and 3B have different hidden sizes, and the local smoke test uses 1B while the real run uses 3B. A hard-coded value passes locally and fails on the rented GPU.

---

## 5. Config field reference

Fields shared across configs. Arm-specific blocks are documented in each arm's file.

| Block | Field | Meaning |
|---|---|---|
| `experiment` | `name` | Run identifier; becomes the `results/runs/` directory name |
| | `seed` | D43 root seed, fixed at 42; D26 bootstrap and D38 F1 initialization retain their direct-seed contracts (§3.5) |
| | `mode` | `debug` \| `train` \| `evaluate` — controls whether training runs and how much data loads |
| `tracking` | `json_logging` | Always `true`. Local `RunLogger`; never disable |
| | `wandb_mode` | `disabled` (debug) \| `offline` (local) \| `online` (cloud) |
| `dataset` | `exercise`, `input_channels`, `num_classes` | `B`, `12`, `18` — fixed for v1 |
| | `split_manifest` | Required for reportable subject-independent runs; load and verify it as §3.2 specifies |
| | `split`, `test_fraction` | Random-window leakage demonstration only |
| | `debug_split` | Explicit non-reportable `train_subjects` and `held_out_subjects` for tiny/local runs |
| | `window_size`, `stride` | See §3.3 |
| | `normalize` | See §3.3 |
| | `max_windows_per_subject` | Debug-only cap for fast iteration; class-covering and never reportable |
| `encoder` | `checkpoint`, `freeze`, `embedding_dim` | Milestone-0 CNN, frozen, 256-d |
| `training` | `target`, `loss` | `class_label`, `class_weighted_cross_entropy` for classification |
| | `batch_size`, `gradient_accumulation_steps` | Effective batch = product |
| | `epochs`, `learning_rate`, `weight_decay` | Optimization; F1's `epochs` is the initial cross-validation fold horizon |
| | `optimization` | D38 freezes F1's Adam implementation/settings, constant learning rate, initialization, precision and gradient handling (§5.2); not a tunable grid |
| | `epoch_selection`, `horizon_escalation`, `refit` | F1 model-selection and refit contract (§5.2). There is no top-level `early_stopping` key: D18/D19 fix the horizon, so a field for it could only restate the protocol or contradict it |
| | `device` | `auto` (portable) \| `cpu` \| `mps` (local-only) \| `cuda` (cloud-only). `auto` picks CUDA, then MPS, then CPU; any other value is a **pin** and an unavailable pin raises rather than falling back, so a config asking for `cuda` can never silently report a CPU run. The resolved device is recorded in the run log |
| | `save_checkpoint` | Writes `checkpoints/<name>/refit.pt` (final downstream use) and `checkpoints/<name>/folds/fold{k}/best.pt` (analysis only); D28 selection separately writes `development/fold{k}/fixed_budget.pt` |
| `evaluation` | `metrics` | See §3.4 |
| | `save_predictions` | Persist per-window predictions for later analysis |

### 5.1 The three checkpoint kinds

A training run can produce three kinds of checkpoint. The names differ because the selection rule and intended reader differ.

| Path | What it is | Who reads it |
|---|---|---|
| `checkpoints/<name>/refit.pt` | The model refit on the **full training-subject set** using the cross-validated hyperparameters | **Every final/reportable downstream consumer** — the adapters, the generator, the robustness registry |
| `checkpoints/<name>/folds/fold{k}/best.pt` | Per-fold model at the **smoothed validation peak** (§5.2) | Extended analysis only — never consumed by another run |
| `checkpoints/<name>/development/fold{k}/fixed_budget.pt` | A fresh fold-complement fit trained for a fixed, already-selected budget, with no checkpoint selection on that fold's four validation subjects | Shared development infrastructure (D33/D35): F1 temperature fitting, G2 diagnostics, D28 adaptation selection and G4 temperature fitting; never the final pipeline or a reportable fold result |

**Why the refit artifact is not called `best`.** Within a fold, `best` carries its ordinary sense: the epoch that scored highest on that fold's validation subjects. The refit has no validation set at all, since every training subject goes into training, so it runs to a fixed epoch count (§5.2) without early stopping. Giving both files the same name would leave `best` meaning two different things, and the difference is one a reader needs to see. All three satisfy the checkpoint contract: `{model_state, model_config}`.

The D28 development checkpoint is separate from both. Its fold's four validation subjects may score adaptation candidates but may not select the population model's weights. It records the manifest hash, fold index, exact training/scoring subject IDs, source-config digests, and fixed training budget so the complement-28 fit can be reconstructed without consulting an all-32 refit. The fold-local generator must consume the corresponding fold-local encoder. D35 extends their consumer scope without changing the split: each generator records its exact parent encoder's SHA-256, and every consumer verifies it. Equal dimensions or matching fold IDs do not establish equal latent coordinates. Fit discriminators and compute latent distances within each encoder's space, then aggregate diagnostics rather than pooling vectors across independently trained folds. The encoder budget is D18–D19's accepted refit budget, written `d18_d19_accepted_refit_budget` in every config that names this path; the generator budget comes from the frozen G1 config (`frozen_generator_source_config_training_epochs`). Record both separately. Because several experiments name the same file, they must agree on what built it: a config that named the same path with a different budget would be describing a different model, and whichever experiment ran second would silently load the first one's encoder. A structural test enforces one budget per shared path. No second set of encoder/generator fits is required by G2, but its discriminators still require training, and a failed gate can require rebuilding affected generator artifacts and invalidating their dependent banks.

### 5.2 Epoch selection: the smoothed validation peak

**Reference optimization (D38).** F1 keeps F0's optimizer mechanics with F1's already-declared learning-rate/decay settings: Adam with betas `(0.9, 0.999)`, epsilon `1e-8`, `amsgrad=False`, learning rate `3e-4` and coupled L2 weight decay `1e-4` on every trainable parameter (not AdamW). The learning rate is constant, with no scheduler. F0's debug learning rate remains `1e-3`; this does not replace either config's declared value. Each fresh model uses the current CNN modules' PyTorch-default initialization under `experiment.seed`; record the pinned environment and seed with the artifact. Use FP32, no autocast or gradient scaling, and no gradient clipping. A non-finite loss or gradient fails the run before the optimizer step; never silently skip the update and produce a shortened-budget checkpoint. Apply the same recipe to selection folds, fixed-budget development fits and the final refit. This is a reproducible reference freeze, not evidence that these hyperparameters are optimal. The future F1 harness must consume the explicit `training.optimization` block; the current F0 smoke runner is unchanged and is not thereby certified for reportable F1 execution.

Each of the eight cross-validation folds trains for exactly **30 epochs**, with **no early stopping**. At each epoch, compute 18-class macro-F1 separately for each of the fold's four validation subjects, using the fixed class labels 0–17 and zero for an undefined class F1. The raw fold score is the arithmetic mean of those four subject scores, so every validation subject has exactly 25% weight; concatenating their windows before computing macro-F1 is forbidden (D21).

For epoch *e* in {10, …, 30}, the selection score is the arithmetic mean of those raw fold scores over epochs *e*−9 through *e*. Epochs 1–9 are ineligible because they do not have a full 10-epoch trailing window. Select the epoch with the highest score; if two or more scores are exactly equal, select the earliest epoch. The fold checkpoint is the model state from that selected epoch. Record the 10-epoch value as a model-selection diagnostic, not as the fold's reportable `macro_f1`.

All reportable validation metrics — including `macro_f1` — are computed from the predictions made by that one selected fold checkpoint. D40 cross-fits temperature by validation subject for scaled validation diagnostics (§3.4); these are conditional on checkpoint selection, not unbiased nested validation. Do not report the all-four-subject temperature-fitting score as validation performance. Final-test metrics likewise come from predictions made by the frozen selected fold checkpoints and the frozen refit checkpoint. Metrics are never averaged across epochs: doing so would describe no single model, and is especially ill-defined for count-valued or nonlinear outputs such as confusion matrices and ECE. D20 explicitly narrows D9's earlier phrase "fold metrics are the smoothed values": only the validation-macro-F1 **selection score** is smoothed.

After all eight folds finish, sort their selected epochs as *e*<sub>(1)</sub> ≤ … ≤ *e*<sub>(8)</sub> and compute

*E*<sub>refit</sub> = ⌈(*e*<sub>(4)</sub> + *e*<sub>(5)</sub>) / 2⌉.

If *E*<sub>refit</sub> is below 30, refit a fresh model on all 32 development subjects for exactly that many epochs, again with no early stopping. The refit has no validation set: its budget is transferred from cross-validation.

If *E*<sub>refit</sub> equals 30, the central selected epoch is at the observation boundary and the initial horizon is treated as right-censored. Do **not** refit and do **not** inspect any final-test prediction. Instead, rerun all eight folds from scratch with a 60-epoch horizon, keeping every other setting fixed and applying the same complete 10-epoch selection window over epochs 10–60. Recompute *E*<sub>refit</sub> from those eight replacement runs; the 30-epoch fold results are development diagnostics, not reportable alternatives. If the new value is below 60, proceed with the refit. If it equals 60, stop for a new owner decision — there is no second automatic extension.

No final-test inference begins until horizon selection is complete, the fresh refit checkpoint is written and frozen, and the eight selected fold checkpoints from the same accepted horizon are frozen as one set. Test performance can never trigger longer training or a second confirmatory evaluation. D18 fixes the selection constants and D19 fixes this one allowed development-only horizon escalation; neither is a runner default or per-run tuning knob.

Two simpler rules suggest themselves, and both fail. Picking the raw best epoch selects partly for noise: validation macro-F1 jumps around from epoch to epoch, especially with only 4 validation subjects, and whatever noise pushed one epoch to the top also inflates the number you then report from it. Training to a fixed budget and averaging the last N epochs, which is what [the group's prior work](https://arxiv.org/abs/2303.10336) does, avoids that problem, and the instinct behind it is sound. But it measures wherever the run happened to stop. A model that has overfit by epoch 800 of 2000 still gets read off at epoch 2000.

Smoothing takes the peak from the first rule and the averaging from the second.

**Easily missed:** the 10-epoch window, full-window requirement, initial 30-epoch horizon, single 60-epoch contingency, and earliest tie-break are protocol constants, not tuning knobs. Use one horizon consistently across all eight folds; never extend only the folds that reached the boundary.

### 5.3 Sanity check: the refit against the fold distribution

The 8×8 matrix (§3.4) yields eight fold-model test scores. The refit trains on all 32 training subjects while each fold model trains on 28. Those four extra subjects are worth something: in subject-independent EMG the number of training subjects is the binding constraint, so **the refit should land at or above the fold mean.** That is why the check is one-sided rather than a containment test:

| Outcome | Reading |
|---|---|
| Refit ≥ fold mean | **Expected.** Not a flag. The size of the gap is itself a result — see below |
| Refit < fold mean | **Red flag.** The refit strictly dominates every fold model in training data, so underperforming points at a defect: the epoch budget transferring badly from 28 subjects to 32, a class-weight computed on the wrong split, or a bug in the refit path |
| Refit far above the fold maximum | **Soft flag.** Larger than four extra subjects can plausibly explain. Not automatically wrong, but check for leakage before reporting |

Report the refit score beside the fold mean and range, so a reader can check this rather than take it on trust.

The gap itself is worth keeping. Refit-minus-fold-mean estimates what four more training subjects buy in accuracy, which is a direct read on the marginal value of subject data — the quantity the generative arm's personalization-efficiency curve approaches from the other side (§6, `real_gesture_trials_saved`). Record it once the check passes instead of throwing it away.

---

## 6. Experiment index

Config paths below are relative to `configs/experiment/` (D1 layout, now in place): `foundation/`, `generation/`, `language/`. The existing `configs/dataset/` and `configs/model/` siblings are untouched.

### Foundation (Milestone 0) — prerequisite for both arms

| ID | Experiment | Config |
|---|---|---|
| F0 | Pipeline smoke test | `foundation/debug_tiny.yaml` |
| F1 | Encoder baseline, subject-independent | `foundation/baseline_cnn_subject_split.yaml` |
| F2 | Leakage demonstration (random split) | `foundation/baseline_cnn_random_split.yaml` |
| F3 | Robustness — additive noise | `foundation/robustness_noise.yaml` |
| F4 | Robustness — channel dropout | `foundation/robustness_channel_dropout.yaml` |
| F5 | Robustness — amplitude scaling | `foundation/robustness_amplitude_scaling.yaml` |

F1 produces the frozen encoder checkpoint that **both arms depend on**. Neither arm can start until F1 is done and its accuracy is written down as measured.

**F3–F5 run against every arm, not just the encoder (D2).** Each perturbation is evaluated on the F1 encoder, on the language-arm models, and on the generative arm's augmented decoders. "The encoder is robust" says nothing about whether soft-prefix conditioning or synthetic augmentation preserves that robustness — and whether augmentation buys accuracy at the cost of robustness is the first question a reviewer will ask.

**How this is wired.** The perturbation acts on the input window and is target-agnostic, so the two axes are separate files. Each `robustness_*.yaml` describes only the perturbation and carries no checkpoint. The models to evaluate live in one shared registry, [`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml), and `scripts/evaluate.py` evaluates the perturbation against every registry target whose checkpoint or run directory exists.

**Two gates, in order.** First the perturbation config gates *itself*: a `type` and a level list are not an operational definition, and without the application order relative to normalization, the sampling rule, the seed derivation, the repetition count and the level aggregation, every target would be scored against an undefined transform. That is one refusal for the whole run rather than a per-target skip, because the unresolved thing is the axis. F3–F5 declare those five fields as `null` today, so all three currently refuse. Second, per target: an unresolved `pending_*` sentinel, or a source config whose `protocol.status` is not `ready`, blocks before artifact lookup; ready targets with absent artifacts are skipped with a log line.

**Readiness is declared, never inferred from silence.** A config with no `protocol:` block or no `protocol.status`, and a target with no `source_config`, is blocked: read as ready, silence would be indistinguishable from someone having written `status: ready`. `status: ready` beside a non-empty `blocked_on` is refused as a contradiction. Every config named by the registry also declares a status, and a test enforces that. Before this rule the four non-generation targets resolved as dispatchable while blockers stood recorded against them.

The suite runs incrementally, but **today it still cannot execute**: all three perturbations are blocked and the evaluator is a stub. D38–D40 make F1 protocol-ready, not trained: its 8-fold/refit harness and calibration machinery remain unimplemented, and no reportable artifact exists. The other six targets remain blocked. L2–L4 need selection and repeat contracts. G3 slices need repeat/run resolution and the matching immutable G4 table with a ready calibration source protocol. These fields must be consumed and validated when the evaluator is implemented; this protocol amendment does not make the current stub execute them.

A file-shaped target is just `(name, checkpoint)` — no `model_type`. That relies on the **checkpoint contract**: every checkpoint is saved as `{model_state, model_config}`, so any consumer rebuilds the architecture from the checkpoint alone. That matters for the adapter stacks (encoder + projector + backbone + head), which no single `model_type` string could describe. F1's trainer is the first code to honor this contract. G3 targets are run-shaped instead and name a `run_dir`, because their per-subject heads are resolved inside the run.

File-shaped targets point at **`refit.pt`** — see §5 for the three checkpoint kinds and why the refit artifact does not reuse the name `best`.

The evaluation logic itself (data loader, perturbation transforms, metrics) is Milestone-0 work and not written yet; `scripts/evaluate.py` currently resolves and reports the plan.

### Arms

See [generative-arm.md](generative-arm.md) (G-series) and [language-arm.md](language-arm.md) (L-series).

---

## 7. Resolved decisions

Recorded so they are not re-litigated.

**D1 — Three config subdirectories, not two.** `foundation/`, `generation/`, `language/`. F0–F5 are shared infrastructure both arms depend on; filing them under one arm would misrepresent the structure.

**D2 — Robustness evaluations run against every arm.** See the note under §6.

**D3 — The generative readout (L5) is deferred beyond v1.** The core claim is fully answered by the discriminative readout plus the random-init control; a generative readout adds narrative (natural-language output for the structured-report demo), not inferential power, and costs real work — label tokenization, multi-token label handling, and deriving calibrated 18-class probabilities from token logits. If the language arm returns a null — the spec's own expected outcome — that work is wasted.

*Revisit trigger:* if the frozen pretrained LLM beats the matched random-init control under the shared discriminative readout, or if the structured-report demo is needed for outreach.

---

**D4 — Text-summary feature set and formatting (L4).** Confirmed as specified below.

**D5 — The adaptation baseline is in v1**, scoped to head-only fine-tuning, implemented as a `calibration_strategy` on the shared personalization harness. D15 amends its strategy names and matched-training semantics. Details below.

**D14 — G3's subject-calibration unit is a complete gesture trial, total per held-out subject.** The trial budget, grid, acquisition schedule, and class-coverage reporting are specified in [generative-arm.md](generative-arm.md), G3.

**D15 — G3 uses matched F1-head adaptation strategies.** The population reference is not adapted; the adapted arms share initialization, trainable surface, optimizer schedule, and optimizer-step budget for *k*>0. **Amended by D29**: the single synthetic arm became two, split by conditioning source, so there are four curves. See G3 and D5 below.

---

## 8. D4 and D5 in full

### D4 — Text-summary feature set and numeric formatting (L4)

Two problems with the feature list as specified, both of which weaken the baseline. This baseline exists to be *hard to beat*, so a weak version does active damage: it hands the soft prefix a victory that a reviewer will discount.

**Problem 1: `rms` is provably redundant.** For any window, `rms = sqrt(mean² + sd²)` exactly (verified numerically to ~1e-16, for both population and sample SD at N=400). Given `mean` and `standard_deviation`, `rms` adds *zero* information. The specified five features are really four.

**Problem 2 — the more serious one: every specified feature is permutation-invariant.** `mean`, `sd`, `rms`, `min`, `max` are all unchanged if you shuffle the samples within a window. The text baseline therefore receives **no temporal information whatsoever**, while the encoder path sees a 1-D CNN over the ordered signal. Any gap between them would then be partly attributable to *temporal structure vs. none* rather than to *soft prefix vs. text*, so the control would carry a confound of its own.

**Recommended feature set** — drop `rms`, add two standard time-domain EMG features that capture ordering:

| Feature | Adds |
|---|---|
| `mean`, `standard_deviation`, `minimum`, `maximum` | Amplitude distribution (keep) |
| ~~`rms`~~ | *Drop — deterministic function of mean and sd* |
| `waveform_length` (Σ\|xᵢ₊₁ − xᵢ\|) | Signal path length: temporal, canonical for sEMG |
| `zero_crossings` | Coarse frequency content: temporal, canonical for sEMG |

Six features × 12 channels = 72 scalars. These are from the standard Hudgins time-domain set that an EMG practitioner would actually reach for, which is the bar this baseline should clear.

**Recommended formatting:**

1. **Compute on the normalized signal**, using the same `train_subject_covered_sample_stats` statistics as the encoder path — same preprocessing for both arms, and it bounds the numeric range so one format works everywhere.
2. **Fixed-point, 2 decimal places.** Post-normalization values sit roughly in ±5, so `%.2f` preserves ample resolution at ~5 characters each. Avoid scientific notation (`1.2e-05` fragments badly under BPE) and avoid full float repr (17 digits of pure token noise).
3. **One labelled line per channel**, so position is never inferred: `ch01 mean=-0.12 sd=0.98 min=-3.41 max=3.02 wl=41.7 zc=88`. A bare list of 72 comma-separated numbers asks the model to count, which it does poorly — and handicapping the prompt defeats the baseline's purpose.
4. **The prompt template is a hyperparameter.** Freeze it, version it, and record it verbatim in `run.json`. If it drifts between runs the baseline is not comparable to itself.

Budget: roughly 200–250 prompt tokens per window.

### D5 — Adaptation baseline scope

**In v1.** It is the first question any reviewer asks about the headline: *why synthesize data instead of just fine-tuning on the real calibration trials already acquired?*

The adaptation baseline is `real_adaptation`: start from the F1 population head and perform **head-only fine-tuning on every valid window contained in the selected *k* real gesture trials** for that held-out subject. The encoder stays frozen. Full fine-tuning, test-time adaptation (entropy minimization or normalization-stat updates), and a fresh randomly initialized head are follow-ups, not v1 alternatives.

For *k*>0 it has two matched synthetic counterparts (D29) — `real_plus_population_synthetic_adaptation` and `real_plus_subject_synthetic_adaptation` — which start from the same F1 head, train the same parameters under the same schedule and step budget, and differ from each other *only* by the conditioning embedding. The objective, loss reductions, regularizer and full-batch rule are fixed by D28; only the numeric candidate grid remains Pending. The unchanged F1 head is retained as `population_no_adaptation`, a horizontal population reference. At *k*=0, `real_adaptation` is the same no-op reference. The replay aliases share one population-conditioned head per recorded `(adaptation_seed, synthetic_draw_id)`, since subject and schedule provide no input there; each held-out subject is still evaluated once per repeat. That synthetic-only point is a diagnostic, not a matched treatment comparison.

Build the personalization runner with a pluggable `calibration_strategy` axis (`population_no_adaptation` | `real_adaptation` | `real_plus_population_synthetic_adaptation` | `real_plus_subject_synthetic_adaptation`) from the first line of code. These names replace the ambiguous prior labels, under which `real_only` and `real_plus_adaptation` could describe the same operation.
