# Generative arm (G-series) — the headline

**The question.** How many real calibration gesture trials does a new, unseen subject need to
reach a target decoding accuracy — and can synthetic data from a conditional generator replace
some of them?

**The deliverable is a curve, not a number.** Accuracy against *k*, the count of real
calibration gesture trials supplied in total by the held-out subject, plotted for each calibration
strategy. The single headline sentence falls out of the curve: *augmentation reaches the target
accuracy with approximately N fewer real gesture trials per subject.*

Read [README.md](README.md) first: §2 (vocabulary), §3 (shared protocol) are assumed here. Note
especially the two senses of "calibration" in §2 — this document uses **subject calibration**
(the *k* real gesture trials) unless it says *probability calibration*.

**Prerequisite:** F1 must be complete. The generator operates in the frozen encoder's latent
space, so it cannot be built before the encoder exists.

---

## The leakage rules — read before anything else

The generative arm is the part of this project where a positive result is easiest to fake
without noticing. These rules are **non-negotiable**, and a result traceable to a violation is
treated as **no result**.

1. **The generator never trains on held-out-subject data.** Not their test windows, not their
   calibration windows, not their statistics. It is trained on training subjects only.
2. **At k = 0**, generation is class-conditioned only, using a population/default subject
   embedding. There is no subject-specific information available, by definition.
3. **At k > 0**, any subject-specific conditioning is derived **exclusively** from the windows
   wholly contained in those *k* permitted real gesture trials — never from any other window
   belonging to that subject.
4. **Normalization statistics** follow README §3.3: computed on training subjects, applied
   unchanged. A subject-specific normalization fitted on the full held-out recording is a leak,
   and it is invisible in the logs.
5. **The evaluation set is fixed** across all *k* and all strategies. If the test windows shift
   as *k* grows, the curve measures the test set rather than the method.

**Easily missed:** the most common leak is not in the generator at all — it is fitting *any*
statistic (normalization, subject embedding, class prior) on data the protocol has not yet
"paid" for. No strategy may see more than the selected *k* trials: both adapted strategies use
exactly that same set, while `population_no_adaptation` deliberately uses none. Their overlapping
windows are derived examples, not additional calibration units.

---

## G0 — Generator plumbing check

**Config:** `gen_vae_debug.yaml` · **Runs:** locally, minutes · **Mode:** `debug`

**What it measures.** That the conditional VAE trains at all: encoder latents load, the
class-conditioning and subject-conditioning paths accept their inputs, the negative-ELBO
objective is finite and updates parameters, and samples come back at the right shape.

**Why it exists.** Same reason as L0 — find shape and conditioning bugs on 3 subjects locally
before spending cloud time.

**Easily missed.** A finite loss and parameter update prove plumbing, not usefulness. During KL
warmup, the changing weight means objective values are not directly comparable across epochs.
Check that reconstructions are not the dataset mean, and that varying the class label actually
changes the sample. Both failures can accompany a beautiful loss curve.

---

## G1 — Conditional VAE training

**Config:** `gen_vae_train.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** A conditional VAE over the **frozen encoder's latent space**, conditioned
on gesture class and a subject/calibration embedding.

**Why latent space rather than raw signal.** Small, fast, and stable to train; and the
downstream classifier consumes encoder latents anyway, so synthesizing at that level is
sufficient for the personalization question. Raw-waveform synthesis is a larger project and is
not required for the headline claim.

**Easily missed.**
- **Posterior collapse.** The classic conditional-VAE failure: the decoder learns to ignore the
  latent and generate from the conditioning alone. Symptom: KL term → 0. Monitor the KL term
  separately from the total negative-ELBO objective, and warm up its weight rather than applying
  it at full strength from step 0.
- **The conditioning must actually condition.** Verify that sampling with a fixed latent and
  varying the class label produces different outputs. If not, the "conditional" generator is
  unconditional and every downstream result is meaningless.
- **Training subjects only** (leakage rule 1). This constrains the subject-embedding design: it
  must be *computable* for an unseen subject from the windows contained in their *k* permitted
  gesture trials, so it cannot be a learned per-subject lookup table indexed by subject ID. An
  embedding that only exists for training subjects cannot be produced for a new user at all.

**Not yet an executable definition.** The phrase `calibration_derived` fixes the information the
embedding may use, not how to compute it. Before G1 runs, specify the estimator and pooling,
training-subject pseudo-calibration schedules across *k*, the population embedding at *k*=0,
target-window exclusion, and a control against gesture-class content masquerading as subject
identity. G1 fails closed on those fields; a runner must not supply defaults.

---

## G2 — Synthetic quality by discrimination (the gate)

**Config:** `gen_synthetic_quality.yaml` · **Runs:** cloud or local · **Mode:** `evaluate`

**What it measures.** A **classifier two-sample test**: train a discriminator to tell real
encoder latents from generated ones. AUC near chance means the approved discriminator did not
detect separability; it establishes distributional similarity only under the approved grouping,
orientation, uncertainty, and sanity-control contract.

**Why it exists — it is a gate, not a metric.** It runs *before* G3 is interpreted. Without it,
a personalization "gain" could come from a degenerate generator: one emitting near-copies of a
few training examples, or class-typical blobs that happen to be linearly separable. Either
produces an artifact rather than a result, which is why G2 has to pass before the headline claim
means anything.

**Easily missed.**
- **Both failure directions matter.** Strong separation means the generator is unrealistic; its
  numeric AUC direction depends on the approved label/score orientation. AUC ≈ 0.5 from a *weak*
  discriminator means nothing at all — it must be strong enough to separate obviously-bad
  synthetic data, so calibrate it against a deliberately poor generator (e.g. Gaussian noise at
  matched moments) and confirm strong separation there.
- **Fidelity is not diversity.** A generator that reproduces ten real samples perfectly scores
  near chance and is useless. Nearest-neighbour distance from each synthetic sample to its closest
  real training sample can expose memorization, which is both a quality failure *and* a privacy
  consideration, but it cannot establish diversity by itself. G2 needs a separate coverage or
  synthetic–synthetic diversity diagnostic.
- **The iterative development gate cannot use the final test subjects.** The old wording, "test on
  held-out real data," left two different held-out roles collapsed: inner subject validation for
  model development and the eight final test subjects. G2 is fail-closed until the development
  population/aggregation and the one-shot final diagnostic failure policy are approved. A final
  diagnostic may never trigger tuning on those same subjects.
- **Evaluate per class.** A generator can be excellent for rest and useless for rare movements,
  and a pooled AUC hides that.
- Report AUC with a confidence interval. A single number near 0.5 with a wide interval is not
  evidence of anything. Discriminator splitting and interval resampling must respect subject/trial
  dependence rather than treating overlapping windows as independent.

**Gate condition:** if the approved development gate fails, fix the generator before the protocol
is frozen. If the eventual one-shot final diagnostic fails, G3 is not reported; its data cannot be
recycled into another development iteration. The exact two-stage population policy and the
quality-metric/acceptance contract remain Pending.

---

## G3 — Personalization-efficiency curve (the headline)

**Config:** `gen_personalization_efficiency.yaml` · **Runs:** cloud CUDA · **Mode:** `evaluate`

**What it measures.** For each held-out subject, decoding accuracy as a function of the total
number of real calibration gesture trials *k* ∈ **{0, 1, 2, 5, 10, 17, 20, 34}**, under three
calibration strategies:

| `calibration_strategy` | Operation | Answers |
|---|---|---|
| `population_no_adaptation` | Use the unchanged F1 population head; no held-out-subject data | The horizontal population reference |
| `real_adaptation` | Start from the F1 head and fine-tune it on all valid windows from the *k* selected real trials | **D5 baseline** — what ordinary head adaptation buys |
| `real_plus_synthetic_adaptation` | Start from the same F1 head and fine-tune the same parameters on those real windows plus subject-conditioned synthetic latent windows | **The headline** — does synthesis reduce real trial burden beyond matched adaptation? |

Three curves on one axis. The headline number is the horizontal gap: how many fewer real gesture
trials `real_plus_synthetic_adaptation` needs to reach the accuracy `real_adaptation` reaches at a
given *k*. Name the result `real_gesture_trials_saved`; never shorten it to "samples saved," which
would obscure the unit.

The **unit and name** of that statistic are frozen; its numerical extraction is not. Before the
runner is implemented, fix the target-accuracy rule, subject-level versus aggregate computation,
interpolation or monotonic treatment, uncertainty, and what is reported when a strategy never
reaches the target. Those remain Pending rather than being hidden in evaluation code.

**Why the adaptation strategy is here (D5).** It is the first question anyone asks about the
headline: *why synthesize data instead of just fine-tuning on the real trials you already have?*
Without `real_adaptation`, the contribution is unquantified. It is cheap because it shares this
entire harness and needs no generator. Scope is **head-only fine-tuning**. For *k*>0, the two
trainable strategies use the same F1 initialization, trainable surface, optimizer schedule, and
optimizer-step budget; full fine-tuning and test-time adaptation are follow-ups. At *k*=0,
`real_adaptation` is a no-op and coincides with `population_no_adaptation`, while
`real_plus_synthetic_adaptation` may update the head using population-conditioned synthetic
windows. That synthetic-only point is a diagnostic, not a matched real-versus-synthetic treatment
comparison, because no real-only optimizer-step budget exists at *k*=0.

The matched-step rule does not yet define the adaptation objective. Optimizer, loss, class-weight
source, learning rate, regularization, and behavior for classes absent from a small-*k* support set
remain Pending and fail closed. A population-conditioned class-balanced replay control is also
Pending: without it, a gain from `real_plus_synthetic_adaptation` could reflect generic balanced
rehearsal rather than calibration-derived subject information. Until that control is decided, keep
the causal claim narrow.

**Design requirement.** `calibration_strategy` must be a **pluggable axis** in the runner from
the first line of code. Bolted on afterwards it means rewriting the loop.

### Which repetitions calibrate, and which evaluate

Calibration draws from **repetitions {1, 4}**; evaluation uses **{2, 3, 5, 6}**. The two sets are
disjoint, which is what leakage rule 3 requires — windows from the *k* permitted trials cannot
overlap the windows the method is scored on. G3/G4 obtain both sets from the verified split
manifest; their experiment configs deliberately do not duplicate them.

The reason the calibration set is `{1, 4}` rather than `{1, 2}` is drift. The DB2 descriptor
reports "a significant (P<0.05) dependence on the repetition ... in 12.5% (database 2) of the
subjects": for roughly one subject in eight, sEMG amplitude moves systematically across the six
repetitions, from fatigue or electrode shift. Calibrating on the first two repetitions and scoring
on the last four would then measure adaptation to that drift rather than adaptation to the
subject, in those subjects. Spanning early and late keeps the calibration set representative of
the session it is meant to represent.

This is a **session-representative offline calibration** estimand, not a strictly prospective
"calibrate once at startup, then evaluate only later data" estimand: repetition 4 occurs after
evaluation repetitions 2 and 3. The sets are disjoint, so this is not sample overlap, but the
headline and paper must not imply chronological onboarding that DB2 cannot test under this split.

Repetition indices come from `rerepetition` (README §3.1), not `repetition`.

### The calibration unit and acquisition schedule (D14)

One unit of *k* is one **complete active-gesture trial**: one gesture class performed by one
held-out subject at one eligible corrected `rerepetition`. Selection happens at the trial level
before windowing. Every 400-sample window wholly contained in that trial is available; no window
may straddle gesture/rest, trial, or calibration/evaluation boundaries. A nominal 5 s trial yields
roughly 97 windows at stride 100, but those highly overlapping windows are derived training
examples, not 97 independent calibration units.

Exercise B supplies 17 active gestures and two eligible calibration repetitions, so each subject's
pool contains **34 active-gesture trials**. The *k* grid is therefore
**{0, 1, 2, 5, 10, 17, 20, 34}**; values above 34 are impossible. The axis is total trials per
subject, not trials per class. At *k* < 17, calibration intentionally omits some gesture classes:
G3 then measures whether subject information learned from demonstrated gestures transfers to the
held-out subject's undemonstrated gestures.

For each `(subject, schedule_index)`, construct one seeded, nested acquisition schedule. Derive its
seed from the first 64 bits of SHA-256 over the UTF-8 string
`"{experiment.seed}:{subject_id}:{schedule_index}"`; never use Python's process-randomized `hash()`.
The PRNG/permutation algorithm and whether schedule indices start at 0 or 1 remain Pending; both
must be fixed before execution, and every selected `(gesture, rerepetition)` trial ID must be
persisted with the run so another implementation can reconstruct the exact schedule.

1. Randomly permute the 17 active gesture classes.
2. In the first pass, select exactly one trial per gesture, alternating repetitions 1 and 4 down
   the permuted class list. Because 17 is odd, one repetition supplies 9 trials and the other 8;
   choose the starting repetition from `(subject_id + schedule_index) % 2` so the imbalance is at
   most one and flips across schedules.
3. In the second pass, select each gesture's other eligible repetition.
4. Define every calibration set as a prefix of that schedule, so a smaller-*k* set is a strict
   subset of every larger-*k* set.
5. Reuse the identical selected trials for all calibration strategies. Unselected trials from
   repetitions 1 and 4 remain unused; they never enter evaluation.

Repeat schedules over fixed seeds and report the within-subject spread. Schedules are Monte Carlo
replicates of the acquisition policy, not additional subjects or independent inferential units.

**Easily missed.**
- **Class coverage is part of the treatment.** In addition to overall accuracy and macro-F1,
  report accuracy for calibration-seen active gestures, calibration-unseen active gestures, and
  rest separately, plus the number of distinct demonstrated gesture classes. Otherwise which
  gestures happened to be selected can masquerade as a method effect. An empty subgroup is `NA`,
  never zero; the subgroup aggregation rule (pooled-window versus class-macro) remains Pending.
- **Report per-subject curves, not only the mean.** A mean over eight held-out subjects can
  conceal one stuck at chance, and the variance between subjects is a finding in its own right.
- **The real and synthetic units differ.** *k* counts gesture trials; the generator budget counts
  latent windows per class. Name and report both. Match optimizer steps across the two trainable
  strategies so extra compute cannot masquerade as a synthetic-data effect. Exact batch size,
  real/synthetic minibatch composition, real-window exposure, synthetic loss weight, and the
  meaning of an "epoch" over unequal datasets remain Pending and must be fixed before the runner.
- **Rest ownership remains open.** Rest is the eighteenth output class, but it is not an ordinary
  prompted gesture trial and does not increment *k*. Before implementing the runner, inspect a
  real MAT file and decide whether selecting a gesture trial also permits one uniquely assigned
  adjacent rest interval. Never give a method all held-out-subject rest data for free.
- **The evaluation set never changes** with *k* (leakage rule 5).

---

## G4 — Calibration efficiency

**Config:** `gen_calibration_efficiency.yaml` · **Runs:** cloud CUDA · **Mode:** `evaluate`

**What it measures.** Expected calibration error and overconfidence-on-wrong-predictions as a
function of *k*, for the same three strategies as G3. This is *probability* calibration, over a
*subject-calibration* axis — the collision of terms README §2 warns about.

**Why it exists.** Augmentation could raise accuracy while making the model confidently wrong,
which for a wearable interface is a poor trade. Improving both makes for a much stronger claim
than improving accuracy alone.

**Easily missed.** ECE is sensitive to binning; fix the bin count and report it. ECE computed on
a small per-subject test set is high-variance, while pooling predictions can hide subject-level
miscalibration and overweight subjects or repeated evaluation windows. The schedule-within-subject
summary, subject aggregation, and role of pooled ECE remain Pending; always show per-subject values.

---

## G5 — Conditional diffusion variant — STRETCH

Not in v1. Replaces the VAE with a conditional diffusion model over the same latent space, run
through the identical G2 gate and G3 protocol. Attempt only once the VAE result is in hand and
the curve exists.

---

## Metrics reference — the G-series keys

README §3.4 is the shared metric registry. The tables below register generation-specific keys, and
every `evaluation.metrics` entry in a `generation/` config appears as a first-column key here or in
the shared registry. Registration prevents unnamed config values; it does not by itself make a
metric operational while a required contract remains Pending.

**G0/G1 — generator training**

| Key | What it is |
|---|---|
| `negative_elbo` | The fixed-sign name for the minimized conditional-VAE objective; lower is better under a fixed definition. At KL weight 1 it is the negative of the raw evidence lower bound. The scheduled warmup loss is a weighted analogue whose values are not comparable while the weight changes (G0) |
| `reconstruction_error` | The reconstruction term, reported apart from the total. Separates a decoder carrying signal from one reproducing the dataset mean |
| `kl_divergence` | The KL term, reported apart from the total. KL → 0 is posterior collapse: the decoder is ignoring the latent and generating from the conditioning alone (G1) |

This freezes only the optimized objective's sign and name. The reconstruction likelihood/error,
component reductions across dimensions, windows, and subjects, units, KL-warmup schedule, whether
evaluation uses the fixed weight-1 objective, and raw-versus-weighted KL reporting remain Pending
before G0/G1 implementation.

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
| `accuracy_vs_k_curve` | Decoding accuracy against *k*, one curve per `calibration_strategy`. The deliverable |
| `real_gesture_trials_saved` | **The headline.** Horizontal gap between the two adapted curves: how many fewer real gesture trials `real_plus_synthetic_adaptation` needs to match `real_adaptation`. D15's unmatched synthetic *k*=0 diagnostic is excluded |
| `calibration_seen_gesture_accuracy` | Accuracy restricted to gesture classes the subject demonstrated in their *k* trials |
| `calibration_unseen_gesture_accuracy` | Accuracy restricted to classes they did **not** demonstrate. Below *k* = 17 this is the transfer question the grid exists to ask |
| `rest_accuracy` | Accuracy on the rest class, reported separately because rest is an evaluation class that never increments *k* |
| `demonstrated_gesture_count` | How many distinct gesture classes the *k* trials covered. Without it the seen/unseen split cannot be read at a given *k* |

**G4 — calibration efficiency**

| Key | What it is |
|---|---|
| `ece_vs_k_curve` | Expected calibration error against *k*, one curve per strategy |
| `per_subject_expected_calibration_error` | ECE for each held-out subject. These values must be shown; the schedule/subject aggregation and the role of any pooled-prediction ECE remain Pending |

Several registered metrics remain non-executable under DECISIONS → Pending. Existing blockers
cover the G1 subject embedding, G2 development/final-test population, G3 headline and subgroup
semantics, ECE binning, and the overconfidence-error definition. The additional VAE objective,
G2 quality-gate, within-subject repeated-run, and G4 headline-ECE contracts raised by the metric
audit are recorded there as well. Implementers must not fill any of these gaps with defaults.

---

## Robustness (D2)

F3–F5 perturbations are also evaluated against the augmented decoders from G3, not only the F1
encoder. The question is whether accuracy bought with synthetic data survives realistic
test-time shift — augmentation that improves clean accuracy while degrading robustness has not
solved the deployment problem it claims to address.

The `g3_synthetic_adaptation` target is already listed in
[`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml).
Unlike the encoder and adapter targets it is **run-shaped, not file-shaped**: G3 trains
per-subject decoders on the fly, so the target points at a `run_dir` rather than a single
`refit.pt`, and the runner treats it as not-present until that directory is populated. See
README §6.

---

## Reporting checklist

Before any G-series number leaves this repository:

- [ ] G2 gate passed and reported (AUC with CI, per class, plus memorization and
      diversity/coverage diagnostics)
- [ ] All five leakage rules verified for the reported configuration
- [ ] All three D15 strategies on the curve, with matched training for the two adapted heads
- [ ] Per-subject curves shown, not only the mean
- [ ] Multiple nested, class-aware schedules per subject, shared across strategies, with spread
- [ ] *k* stated as total complete active-gesture trials per subject; contained windows also reported
- [ ] Calibration-seen gestures, calibration-unseen gestures, and rest reported separately
- [ ] Rest ownership fixed from an inspected MAT file and enforced without sample overlap
- [ ] The headline "N fewer real gesture trials" claim states the synthetic-window and compute budgets
- [ ] `{1, 4}` result described as session-representative offline calibration, not chronological onboarding
