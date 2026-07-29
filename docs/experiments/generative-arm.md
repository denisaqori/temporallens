# Generative arm (G-series) — the headline

**The question.** How many real calibration samples does a new, unseen subject need to reach a
target decoding accuracy — and can synthetic data from a conditional generator replace some of
them?

**The deliverable is a curve, not a number.** Accuracy against *k*, the count of real
calibration samples from the held-out subject, plotted for each calibration strategy. The
single headline sentence falls out of the curve: *augmentation reaches the target accuracy with
approximately N fewer real samples per subject.*

Read [README.md](README.md) first: §2 (vocabulary), §3 (shared protocol) are assumed here. Note
especially the two senses of "calibration" in §2 — this document uses **subject calibration**
(the *k* real samples) unless it says *probability calibration*.

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
3. **At k > 0**, any subject-specific conditioning is derived **exclusively** from those *k*
   permitted real samples — never from any other window belonging to that subject.
4. **Normalization statistics** follow README §3.3: computed on training subjects, applied
   unchanged. A subject-specific normalization fitted on the full held-out recording is a leak,
   and it is invisible in the logs.
5. **The evaluation set is fixed** across all *k* and all strategies. If the test windows shift
   as *k* grows, the curve measures the test set rather than the method.

**Easily missed:** the most common leak is not in the generator at all — it is fitting *any*
statistic (normalization, subject embedding, class prior) on data the protocol has not yet
"paid" for. At every step ask: *how many of this subject's samples has the method been allowed
to see?* The answer must be exactly *k*.

---

## G0 — Generator plumbing check

**Config:** `gen_vae_debug.yaml` · **Runs:** locally, minutes · **Mode:** `debug`

**What it measures.** That the conditional VAE trains at all: encoder latents load, the
class-conditioning and subject-conditioning paths accept their inputs, the ELBO decreases, and
samples come back at the right shape.

**Why it exists.** Same reason as L0 — find shape and conditioning bugs on 3 subjects locally
before spending cloud time.

**Easily missed.** A decreasing ELBO proves optimization, not usefulness. Check that
reconstructions are not the dataset mean, and that varying the class label actually changes the
sample. Both failures produce a beautiful loss curve.

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
  separately from total ELBO, and warm up its weight rather than applying it at full strength
  from step 0.
- **The conditioning must actually condition.** Verify that sampling with a fixed latent and
  varying the class label produces different outputs. If not, the "conditional" generator is
  unconditional and every downstream result is meaningless.
- **Training subjects only** (leakage rule 1). This constrains the subject-embedding design: it
  must be *computable* for an unseen subject from *k* samples at inference, so it cannot be a
  learned per-subject lookup table indexed by subject ID. An embedding that only exists for
  training subjects cannot be produced for a new user at all.

---

## G2 — Synthetic quality by discrimination (the gate)

**Config:** `gen_synthetic_quality.yaml` · **Runs:** cloud or local · **Mode:** `evaluate`

**What it measures.** A **classifier two-sample test**: train a discriminator to tell real
encoder latents from generated ones. Discrimination near chance (AUC ≈ 0.5) indicates the
synthetic distribution is close to the real one.

**Why it exists — it is a gate, not a metric.** It runs *before* G3 is interpreted. Without it,
a personalization "gain" could come from a degenerate generator: one emitting near-copies of a
few training examples, or class-typical blobs that happen to be linearly separable. Either
produces an artifact rather than a result, which is why G2 has to pass before the headline claim
means anything.

**Easily missed.**
- **Both failure directions matter.** AUC ≈ 1.0 means the generator is unrealistic. AUC ≈ 0.5
  from a *weak* discriminator means nothing at all — it must be strong enough to separate
  obviously-bad synthetic data, so calibrate it against a deliberately poor generator (e.g.
  Gaussian noise at matched moments) and confirm it scores near 1.0 there.
- **Fidelity is not diversity.** A generator that reproduces ten real samples perfectly scores
  near chance and is useless. Report a diversity measure alongside AUC — nearest-neighbour
  distance from each synthetic sample to its closest real training sample will expose
  memorization, which is both a quality failure *and* a privacy consideration.
- **Test on held-out real data**, not on the real data the generator trained on.
- **Evaluate per class.** A generator can be excellent for rest and useless for rare movements,
  and a pooled AUC hides that.
- Report AUC with a confidence interval. A single number near 0.5 with a wide interval is not
  evidence of anything.

**Gate condition:** if G2 fails, G3 is not reported. Fix the generator first.

---

## G3 — Personalization-efficiency curve (the headline)

**Config:** `gen_personalization_efficiency.yaml` · **Runs:** cloud CUDA · **Mode:** `evaluate`

**What it measures.** For each held-out subject, decoding accuracy as a function of the number
of real calibration samples *k* ∈ **{0, 5, 10, 20, 50}**, under three calibration strategies:

| `calibration_strategy` | The subject's calibration set is | Answers |
|---|---|---|
| `real_only` | the *k* real samples | The baseline curve — how much real data is needed |
| `real_plus_synthetic` | the *k* real samples + generated samples | **The headline** — can synthesis substitute for real data? |
| `real_plus_adaptation` | the *k* real samples, used for head-only fine-tuning | **D5** — is generation better than just adapting? |

Three curves on one axis. The headline number is the horizontal gap: how many fewer real samples
`real_plus_synthetic` needs to reach the accuracy `real_only` reaches at a given *k*.

**Why the adaptation strategy is here (D5).** It is the first question anyone asks about the
headline: *why synthesize data instead of just fine-tuning on the k samples you already have?*
Without that curve, the contribution is unquantified. It is cheap because it shares this entire
harness and needs no generator — strictly less machinery than the synthetic arm. Scope is
**head-only fine-tuning**, which keeps the trainable surface identical across strategies; full
fine-tuning and test-time adaptation are follow-ups.

**Design requirement.** `calibration_strategy` must be a **pluggable axis** in the runner from
the first line of code. Bolted on afterwards it means rewriting the loop.

**Easily missed.**
- **Which *k* samples?** Sampling *k* windows from a subject is itself a random choice with real
  variance at *k* = 5. Repeat each (subject, *k*) cell over several draws with fixed seeds and
  report the spread. A single draw per cell produces a curve made largely of noise.
- **Class coverage at small *k*.** With 18 classes and *k* = 5, most classes have **zero**
  calibration samples. Decide and document explicitly: is *k* per class or per subject? The two
  are very different experiments. *(Per subject is the realistic deployment framing — a new user
  provides a handful of examples total, not five per gesture.)*
- **Report per-subject curves, not only the mean.** A mean over eight held-out subjects can
  conceal one stuck at chance, and the variance between subjects is a finding in its own right.
- **Matched budgets.** When comparing `real_only` at *k* = 50 against `real_plus_synthetic` at
  *k* = 10, state clearly what is held constant — real samples, total samples, or compute. The
  claim is about *real* samples; say so explicitly rather than leaving it inferred.
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
a small per-subject test set is high-variance — pool across subjects for the headline ECE and
show per-subject values separately.

---

## G5 — Conditional diffusion variant — STRETCH

Not in v1. Replaces the VAE with a conditional diffusion model over the same latent space, run
through the identical G2 gate and G3 protocol. Attempt only once the VAE result is in hand and
the curve exists.

---

## Robustness (D2)

F3–F5 perturbations are also evaluated against the augmented decoders from G3, not only the F1
encoder. The question is whether accuracy bought with synthetic data survives realistic
test-time shift — augmentation that improves clean accuracy while degrading robustness has not
solved the deployment problem it claims to address.

The `g3_augmented_decoder` target is already listed in
[`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml).
Unlike the encoder and adapter targets it is **run-shaped, not file-shaped**: G3 trains
per-subject decoders on the fly, so the target points at a `run_dir` rather than a single
`refit.pt`, and the runner treats it as not-present until that directory is populated. See
README §6.

---

## Reporting checklist

Before any G-series number leaves this repository:

- [ ] G2 gate passed and reported (AUC with CI, per class, plus a diversity measure)
- [ ] All five leakage rules verified for the reported configuration
- [ ] All three strategies on the curve, including `real_plus_adaptation`
- [ ] Per-subject curves shown, not only the mean
- [ ] Multiple *k*-draws per cell, with spread
- [ ] *k*-per-subject vs. *k*-per-class stated explicitly
- [ ] The headline "N fewer real samples" claim states what was held constant
