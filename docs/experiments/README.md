# TemporalLens — Experiment & Ablation Specification

This directory is the **authority** on what each experiment measures, why it exists, and what
is easy to get wrong. Every config file in `configs/` maps to exactly one experiment specified
here. If a config and this document disagree, this document is wrong until someone fixes it —
that is the only way the mapping stays honest.

It exists because shorthand ("the adapter", "the LLM test", "generative") had already started
to mean different things in different places. Section 2 fixes the vocabulary. **Use the exact
terms from Section 2 in configs, code, commit messages, and the paper.**

| Document | Covers |
|---|---|
| This file | Shared protocol, vocabulary, metrics, and the rules both arms obey |
| [generative-arm.md](generative-arm.md) | **Headline.** Conditional generation, synthetic-quality validation, the personalization-efficiency curve |
| [language-arm.md](language-arm.md) | **Secondary.** Soft-prefix conditioning of a frozen LLM, and the two controls that make the claim un-confoundable |

---

## 1. The two arms, and what each is actually claiming

The project has one shared temporal encoder and two independent research questions. They are
evaluated separately and can succeed or fail independently.

**Generative arm (headline).** *How many real calibration samples does a new, unseen subject
need to reach a target accuracy, and can synthetic data replace some of them?* The deliverable
is a curve, not a single number.

**Language arm (secondary).** *Does mapping signal embeddings into a frozen language model's
embedding space improve decoding, calibration, or failure reporting beyond a well-trained
encoder?* The deliverable is a comparison with two controls attached. A clean negative result
is a valid outcome and must be reported as one.

A result in one arm is never evidence for the other. Do not merge them into a single headline.

---

## 2. Vocabulary — the shorthand that caused confusion

These words were being used loosely. They are now defined, and the definitions are binding.

| Term | Precise meaning | Frequently confused with |
|---|---|---|
| **Encoder** | The 1-D CNN mapping a raw EMG window to a 256-d embedding. Trained in Milestone 0, then **frozen** for both arms. | — |
| **Projector** | The small trainable MLP mapping the 256-d encoder embedding to *N* soft-prefix embeddings of the LLM's hidden size. In configs this is the `adapter:` block, whose `type` is `mlp_projector`. | "Adapter" |
| **Adapter** | In this project, a synonym for **projector** — the same object. It is *not* a PEFT/LoRA adapter. Nothing is ever inserted into the language model's own weights. | LoRA adapters |
| **Head** | The trainable classifier on the frozen model's final hidden state. Produces the 18-class logits. Separate from, and downstream of, the projector. | Projector; the LLM's own LM head |
| **Soft prefix** | Continuous embedding vectors prepended to the text-prompt embeddings, passed via `inputs_embeds`. They are **not** discrete tokens and never index the vocabulary. | "Token space", prompt text |
| **Frozen** | `requires_grad=False` on every parameter of that module, and the module in `eval()` mode. The encoder and the language model are frozen; the projector and head are not. | "Not updated this step" |
| **Readout** | *How* a class prediction is extracted. **Discriminative** = trainable head on a pooled hidden state. **Generative** = the model's own LM head emits a label token. See Section 4. | The input path |
| **Input path** | *What* the model receives: soft-prefix embeddings, or engineered text, or nothing (encoder-only). Orthogonal to readout. | The readout |
| **Calibration** (two senses!) | (a) **Subject calibration** — the *k* real labelled samples a new user provides. (b) **Probability calibration** — whether predicted confidence matches accuracy (ECE). Always qualify which. | each other |
| **Generative** (two senses!) | (a) The **generative arm** — the conditional VAE synthesizing EMG. (b) A **generative readout** — reading a class out of an LM head. Unrelated. Always qualify. | each other |

---

## 3. Shared protocol

Every experiment in both arms inherits this unless it explicitly overrides it.

### 3.1 Dataset

NinaPro DB2, **Exercise B** — 17 hand/wrist movements + rest = **18 classes**; 12 channels at
2 kHz, 40 intact subjects. Windows are 400 samples (200 ms). Data is not redistributed with
this repository; it is downloaded under its own terms into `data/raw/`.

The reduced class set is deliberate: cross-subject decoding over the full 49+rest set lands
near chance and produces illegible curves. The full set is a later extension.

### 3.2 Splits

| `split` value | What it does | Role |
|---|---|---|
| `subject_independent` | Train on one set of subjects, test on subjects never seen in training. Held-out set is `[5, 10, 15, 20, 25, 30, 35, 40]`. | **The main protocol. Every headline number uses this.** |
| `random_window` | Windows shuffled without regard to subject. | Deliberately optimistic. Exists *only* to demonstrate how much leakage inflates results. Never a headline number. |

**Easily missed:** windows from a single subject are highly correlated. A random split puts
near-duplicate windows on both sides and inflates accuracy dramatically. Reporting a
`random_window` number without labelling it as the leakage demonstration is a serious error.

### 3.3 Windowing and normalization

`window_size: 400`, `stride: 100` for real runs (`200` for debug configs — fewer, less
overlapping windows).

| `normalize` value | Statistics come from | Used by |
|---|---|---|
| `train_subjects_global_stats` | Pooled across training subjects only | Subject-independent runs |
| `per_subject_train_stats` | Each subject's own training windows | Debug configs |
| `train_split_global_stats` | The training split only | Random-window runs |

**Easily missed — this is the most common silent leak in the whole project.** Normalization
statistics must be computed on **training data only** and then applied unchanged to
validation and test. Computing mean/std over the full dataset before splitting leaks test
distribution into training and inflates every downstream number. It leaves no trace in the
logs. Any statistic that touches a held-out subject's data is a leak.

### 3.4 Metrics

| Metric | What it captures | Why it is here |
|---|---|---|
| `accuracy` | Overall correctness | Baseline readability; misleading alone under class imbalance |
| `macro_f1` | Mean per-class F1, unweighted | Rest is over-represented; macro-F1 stops a model from coasting on it |
| `per_subject_accuracy` | Accuracy for each held-out subject separately | **Report the spread, not just the mean.** Cross-subject variance is the actual story; a good mean over 8 subjects can hide one at chance |
| `confusion_matrix` | Which movements are mistaken for which | Anatomically adjacent gestures confuse; the pattern is a result |
| `expected_calibration_error` | Gap between confidence and accuracy | Probability calibration. A wearable that is confidently wrong is worse than one that abstains |
| `overconfidence_error` | Confidence specifically on *incorrect* predictions | The language arm's benefit, if any, may live here rather than in accuracy |
| `robustness_drop` | Accuracy loss from clean → perturbed | Robustness experiments only |

### 3.5 Reproducibility

`seed: 42` everywhere. Every run writes `run.json` (config, git commit, summary) and
`metrics.jsonl` through `RunLogger` — this is not optional and not W&B-dependent. W&B is
offline by default; nothing may depend on a network service to be reproducible.

**Easily missed:** seeding Python/NumPy/PyTorch does not make MPS and CUDA produce identical
results. Do not compare a local MPS number against a cloud CUDA number and call the difference
an effect. Cross-device comparisons need the same device.

---

## 4. Readout: discriminative vs. generative — and the rule

This axis is **orthogonal** to which arm an experiment belongs to and to which input path it
uses. Conflating the two is what motivated this document.

- **Discriminative readout** — a trainable MLP head pools the frozen model's final hidden
  state (`pooling: last_token`) and emits 18 logits. Trainable: projector (if present) + head.
- **Generative readout** — the frozen LLM's own LM head is teacher-forced to emit a label
  token. Trainable: the projector only.

### The binding rule

> **One readout, held identical across every config in a comparison.**

If the method used a generative readout and a control used a discriminative one, any observed
difference could be caused by the readout rather than by the thing under test — which destroys
the control's entire purpose.

**Decision: the discriminative readout is primary for the language arm.** Three reasons:

1. **The random-init control requires it.** A randomly initialized transformer has no usable
   LM head, so it cannot support a generative readout. A trainable head is the only readout
   all three language-arm configs can share — and sharing is the point.
2. **It still proves the thesis.** "Does language pretraining help?" is answered by
   *frozen-pretrained beats matched-random with an identical head* — same readout, weights the
   only difference.
3. **Calibration metrics.** ECE and overconfidence are standard and well-defined on classifier
   logits; deriving a calibrated 18-class distribution from label-token probabilities is
   fiddly and would differ per arm.

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

A generative readout remains available as an explicitly secondary ablation (L5) on the
frozen-LLM method only — never as a control's readout.

**Easily missed — `pooling: last_token` assumes right-padding.** With a causal model and
left-padded batches, the final position is a pad token and the pooled vector is garbage;
training still runs and loss still decreases, so this fails silently. Either pin the tokenizer
to right-padding or pool the true last non-pad index per sequence. Assert it in code.

**Easily missed — `input_dim: auto_from_llm_config` is load-bearing.** The hidden size must be
read from the model's own config at runtime, never hard-coded. Llama 3.2 1B and 3B have
different hidden sizes, and the local smoke test uses 1B while the real run uses 3B. A
hard-coded value passes locally and fails on the rented GPU.

---

## 5. Config field reference

Fields shared across configs. Arm-specific blocks are documented in each arm's file.

| Block | Field | Meaning |
|---|---|---|
| `experiment` | `name` | Run identifier; becomes the `results/runs/` directory name |
| | `seed` | Global RNG seed |
| | `mode` | `debug` \| `train` \| `evaluate` — controls whether training runs and how much data loads |
| `tracking` | `json_logging` | Always `true`. Local `RunLogger`; never disable |
| | `wandb_mode` | `disabled` (debug) \| `offline` (local) \| `online` (cloud) |
| `dataset` | `exercise`, `input_channels`, `num_classes` | `B`, `12`, `18` — fixed for v1 |
| | `split`, `held_out_subjects` | See §3.2 |
| | `window_size`, `stride` | See §3.3 |
| | `normalize` | See §3.3 |
| | `max_subjects`, `max_windows_per_subject` | Debug-only caps for fast iteration |
| `encoder` | `checkpoint`, `freeze`, `embedding_dim` | Milestone-0 CNN, frozen, 256-d |
| `training` | `target`, `loss` | `class_label`, `cross_entropy` |
| | `batch_size`, `gradient_accumulation_steps` | Effective batch = product |
| | `epochs`, `learning_rate`, `weight_decay` | Optimization |
| | `device` | `auto` (portable) \| `mps` (local-only) \| `cuda` (cloud-only) |
| | `save_checkpoint` | Writes to `checkpoints/<name>/best.pt` |
| `evaluation` | `metrics` | See §3.4 |
| | `save_predictions` | Persist per-window predictions for later analysis |

---

## 6. Experiment index

Config paths below are relative to `configs/experiment/` (D1 layout, now in place):
`foundation/`, `generation/`, `language/`. The existing `configs/dataset/` and
`configs/model/` siblings are untouched.

### Foundation (Milestone 0) — prerequisite for both arms

| ID | Experiment | Config |
|---|---|---|
| F0 | Pipeline smoke test | `foundation/debug_tiny.yaml` |
| F1 | Encoder baseline, subject-independent | `foundation/baseline_cnn_subject_split.yaml` |
| F2 | Leakage demonstration (random split) | `foundation/baseline_cnn_random_split.yaml` |
| F3 | Robustness — additive noise | `foundation/robustness_noise.yaml` |
| F4 | Robustness — channel dropout | `foundation/robustness_channel_dropout.yaml` |
| F5 | Robustness — amplitude scaling | `foundation/robustness_amplitude_scaling.yaml` |

F1 produces the frozen encoder checkpoint that **both arms depend on**. Neither arm can start
until F1 is done and its accuracy is honestly recorded.

**F3–F5 run against every arm, not just the encoder (D2).** Each perturbation is evaluated on
the F1 encoder, on the language-arm models, and on the generative arm's augmented decoders.
"The encoder is robust" says nothing about whether soft-prefix conditioning or synthetic
augmentation preserves that robustness — and whether augmentation buys accuracy at the cost of
robustness is exactly the question a reviewer asks.

**Implication, not yet done:** the three robustness configs currently hard-code
`checkpoint.path: checkpoints/baseline_cnn_subject_split/best.pt`. To run against all arms they
must take the checkpoint and model type as parameters. Until that change lands, D2 is specified
but not executable.

### Arms

See [generative-arm.md](generative-arm.md) (G-series) and [language-arm.md](language-arm.md)
(L-series).

---

## 7. Resolved decisions

Recorded so they are not re-litigated.

**D1 — Three config subdirectories, not two.** `foundation/`, `generation/`, `language/`.
F0–F5 are shared infrastructure both arms depend on; filing them under one arm would
misrepresent the structure.

**D2 — Robustness evaluations run against every arm.** See the note under §6.

**D3 — The generative readout (L5) is deferred beyond v1.** The core claim is fully answered by
the discriminative readout plus the random-init control; a generative readout adds narrative
(natural-language output for the structured-report demo), not inferential power, and costs real
work — label tokenization, multi-token label handling, and deriving calibrated 18-class
probabilities from token logits. If the language arm returns a null — the spec's own expected
outcome — that work is wasted.

*Revisit trigger:* if the frozen pretrained LLM beats the matched random-init control under the
shared discriminative readout, or if the structured-report demo is needed for outreach.

---

**D4 — Text-summary feature set and formatting (L4).** Confirmed as specified below.

**D5 — The adaptation baseline is in v1**, scoped to head-only fine-tuning, implemented as a
`calibration_strategy` on the shared personalization harness. Details below.

---

## 8. D4 and D5 in full

### D4 — Text-summary feature set and numeric formatting (L4)

Two problems with the feature list as specified, both of which weaken the baseline. Since this
baseline exists to be *hard to beat*, a weak version is worse than useless — it manufactures a
win for the soft prefix that a reviewer will immediately discount.

**Problem 1: `rms` is provably redundant.** For any window,
`rms = sqrt(mean² + sd²)` exactly (verified numerically to ~1e-16, for both population and
sample SD at N=400). Given `mean` and `standard_deviation`, `rms` adds *zero* information. The
specified five features are really four.

**Problem 2 — the more serious one: every specified feature is permutation-invariant.**
`mean`, `sd`, `rms`, `min`, `max` are all unchanged if you shuffle the samples within a window.
The text baseline therefore receives **no temporal information whatsoever**, while the encoder
path sees a 1-D CNN over the ordered signal. Any gap between them would then be partly
attributable to *temporal structure vs. none* rather than to *soft prefix vs. text* — a
confound sitting inside the control that is supposed to remove confounds.

**Recommended feature set** — drop `rms`, add two standard time-domain EMG features that
capture ordering:

| Feature | Adds |
|---|---|
| `mean`, `standard_deviation`, `minimum`, `maximum` | Amplitude distribution (keep) |
| ~~`rms`~~ | *Drop — deterministic function of mean and sd* |
| `waveform_length` (Σ\|xᵢ₊₁ − xᵢ\|) | Signal path length: temporal, canonical for sEMG |
| `zero_crossings` | Coarse frequency content: temporal, canonical for sEMG |

Six features × 12 channels = 72 scalars. These are from the standard Hudgins time-domain set
that an EMG practitioner would actually reach for, which is the bar this baseline should clear.

**Recommended formatting:**

1. **Compute on the normalized signal**, using the same `train_subjects_global_stats` statistics
   as the encoder path — same preprocessing for both arms, and it bounds the numeric range so
   one format works everywhere.
2. **Fixed-point, 2 decimal places.** Post-normalization values sit roughly in ±5, so `%.2f`
   preserves ample resolution at ~5 characters each. Avoid scientific notation (`1.2e-05`
   fragments badly under BPE) and avoid full float repr (17 digits of pure token noise).
3. **One labelled line per channel**, so position is never inferred:
   `ch01 mean=-0.12 sd=0.98 min=-3.41 max=3.02 wl=41.7 zc=88`.
   A bare list of 72 comma-separated numbers asks the model to count, which it does poorly —
   and handicapping the prompt defeats the baseline's purpose.
4. **The prompt template is a hyperparameter.** Freeze it, version it, and record it verbatim
   in `run.json`. If it drifts between runs the baseline is not comparable to itself.

Budget: roughly 200–250 prompt tokens per window.

### D5 — Adaptation baseline scope

**In v1 — it is much cheaper than it looks, provided one design choice is made up
front.** The spec already argues it "would sharpen the contribution considerably," and it is the
first question any reviewer asks about the headline: *why synthesize data instead of just
fine-tuning on the k real samples?*

Scope it as **head-only fine-tuning on the k real calibration samples** for each held-out
subject. Not full fine-tuning, not test-time adaptation (entropy minimization, BN-stat updates)
— those are a follow-up. Head-only keeps the trainable surface identical to the augmented arm,
which is what makes the comparison fair.

The cost is low because it is a *third curve on the same axes*, reusing the entire
personalization-efficiency harness — the held-out-subject loop, the k-selection, the evaluation.
It needs no generator at all, so it is strictly less machinery than the augmented arm.

**The design choice that keeps it cheap:** build the personalization runner with a pluggable
`calibration_strategy` axis (`real_only` | `real_plus_synthetic` | `real_plus_adaptation`) from
the first line of code. Bolted on afterwards, it means rewriting the loop.
