# TemporalLens — Experiment & Ablation Specification

This directory is the **authority** on what each experiment measures and where it can go wrong.
Every config file in `configs/` maps to one experiment specified here. If a config and this
document disagree, the config is wrong until someone fixes it; without that rule the mapping
drifts and nobody notices.

It exists because shorthand ("the adapter", "the LLM test", "generative") had already started
to mean different things in different places. Section 2 fixes the vocabulary. **Use the exact
terms from Section 2 in configs, code, commit messages, and the paper.**

| Document | Covers |
|---|---|
| This file | Shared protocol, vocabulary, metrics, and the rules both arms obey |
| [generative-arm.md](generative-arm.md) | **Headline.** Conditional generation, synthetic-quality validation, the personalization-efficiency curve |
| [language-arm.md](language-arm.md) | **Secondary.** Soft-prefix conditioning of a frozen LLM, and the two controls that make the claim un-confoundable |

---

## 1. The two arms and what each claims

The project has one shared temporal encoder and two research questions that do not depend on
each other. They are evaluated separately, and either can succeed while the other fails.

**Generative arm (headline).** *How many real calibration gesture trials does a new, unseen
subject need to reach a target accuracy, and can synthetic data replace some of them?* The
deliverable is a curve, not a single number.

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
| **Calibration** (two senses!) | (a) **Subject calibration** — the *k* complete labelled gesture trials a new user provides in total; all contained windows are derived training examples, not additional calibration units. (b) **Probability calibration** — whether predicted confidence matches accuracy (ECE). Always qualify which. | each other |
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

**Labels come from `restimulus` and `rerepetition`. Never `stimulus` or `repetition`.** The DB2
files carry both. `stimulus` is the label the acquisition software *prompted*, and the
[data descriptor](https://doi.org/10.1038/sdata.2014.53) is explicit that it does not describe
what the subject did: movements "may not perfectly match with the stimuli proposed by our
software due to human reaction times and experimental conditions." The dataset authors corrected
this offline with a generalized likelihood ratio algorithm that "realigns the movement boundaries
by maximizing the likelihood of a rest-movement-rest sequence", and shipped the result as
`restimulus`. `rerepetition` is the matching repetition index.

**Easily missed — this fails silently, and it fails worst exactly where it matters.** Loading
`stimulus` mislabels every movement onset by the subject's reaction time: windows tagged as a
gesture while the arm is still at rest, and rest windows tagged as gesture at the far end.
Training runs, loss falls, accuracy looks plausible. Onset is also the region a real-time
decoder has to get right, so the damage lands on the most important windows. Assert on load that
the label column is `restimulus`.

**No baseline subtraction.** The dataset authors do not subtract a rest level, and neither do we.
Rest is one of the 18 classes, so removing the rest level would erase the signal that defines it.
The only filtering DB2 receives is theirs: a Hampel filter for 50 Hz power-line interference and
its harmonics. (The 1 Hz Butterworth low-pass described in the descriptor applies to database 1
only, whose Otto Bock electrodes emit an already RMS-rectified envelope. DB2's Delsys signal is
raw.) Normalization is handled separately and per §3.3.

### 3.2 Splits

| `split` value | What it does | Role |
|---|---|---|
| `subject_independent` | Train on one set of subjects, test on subjects never seen in training. Held-out set is `[5, 10, 15, 20, 25, 30, 35, 40]`, frozen in the manifest below. | **The main protocol. Every headline number uses this.** |
| `random_window` | Windows shuffled without regard to subject. | Deliberately optimistic. Exists *only* to demonstrate how much leakage inflates results. Never a headline number. |

**Easily missed:** windows from a single subject are highly correlated. A random split puts
near-duplicate windows on both sides and inflates accuracy dramatically. Reporting a
`random_window` number without labelling it as the leakage demonstration is a serious error.

#### The split manifest

The split is **data, not configuration**. It lives in
[`configs/splits/subject_independent_v1.yaml`](../../configs/splits/subject_independent_v1.yaml):
the 8 test subjects, the 32 training subjects, the 8 folds of 4 validation subjects (D6), the
corrected label columns (D12), and the repetition policy (D13).

Load it through `SplitManifest.load()`, never by reading the YAML directly. That call recomputes a
SHA-256 over the split-defining fields and refuses to return anything whose digest does not match,
so an edit to the file cannot quietly change what a number means. It also checks the structure:
roles partition all 40 subjects, every training subject validates in exactly one fold, no fold
validates on a test subject, and the calibration and evaluation repetitions stay disjoint.

Every **reportable subject-independent config** names this file through `dataset.split_manifest`.
It does not repeat `held_out_subjects` or the calibration/evaluation repetition lists: consumers
derive those values from the verified object and record `manifest_hash` in the run and checkpoint
metadata. Tiny/local configs instead declare `dataset.debug_split`; their deliberately reduced
subject holdout exists only to exercise the pipeline and can never produce a reportable number.

Fold assignment is **strided, not blocked** — the *i*-th training subject goes to fold *i* mod 8.
This gives every fold one subject from each quarter of the ID range instead of one contiguous ID
block. It is a deterministic range-balancing rule, not a claim that subject ID is an acquisition
timestamp; the DB2 descriptor does not establish that chronology (D17). Each fold's training set
is the complement of its validation set and is deliberately not written down, so a subject's role
is stated in exactly one place.

**The manifest is frozen.** Changing a split invalidates every number computed under the old one,
and the language arm is a comparison against F1's reference row. A new split means a new versioned
manifest and a DECISIONS row — never an edit in place.

#### Repetitions

DB2 records **6 repetitions** of every movement by every subject. The rule is short: **the
subject is the split unit, so all 6 repetitions follow their subject.** A held-out subject
contributes all six to the held-out pool; a training subject contributes all six to training.
Repetitions never cross the subject-level train/validation/test boundary. F1 evaluates on the full
held-out pool; G3/G4 make the D13 within-subject partition, using repetitions `{1, 4}` only for
subject calibration and `{2, 3, 5, 6}` only for evaluation.

This is a deliberate departure from the dataset's own benchmark, which splits *by repetition
within* subject — "repetition 2 and 5 in database 2" to test, the rest to training. That measures
how well a decoder generalizes to a new session from the same person. We measure generalization
to a new person, which is the deployment question for a wearable and the harder of the two. **Our
numbers are therefore not comparable to theirs**, and the ~75% they report for DB2 is a different
task (50 movements, within subject), not a target to beat.

**Why the rule needs stating rather than assuming.** The descriptor measured amplitude drift
across repetitions and found "a significant (P<0.05) dependence on the repetition ... in 12.5%
(database 2) of the subjects", warning that "attention should be paid to it while splitting
movement repetitions into training set and test set." Keeping whole subjects on one side means
that drift never straddles a split boundary here. It does resurface in subject calibration, where
complete per-gesture trials indexed by `rerepetition` are the natural unit — see
[generative-arm.md](generative-arm.md), G3.

### 3.3 Windowing and normalization

`window_size: 400`, `stride: 100` for real runs (`200` for debug configs — fewer, less
overlapping windows).

| `normalize` value | Statistics come from | Used by |
|---|---|---|
| `train_subjects_global_stats` | Pooled across training subjects only | Subject-independent runs |
| `per_subject_train_stats` | Each subject's own training windows | Debug configs |
| `train_split_global_stats` | The training split only | Random-window runs |

**Easily missed — this is the most common silent leak in the whole project.** Normalization and
other preprocessing statistics must be computed on **training data only** and then applied
unchanged to validation and test. Computing mean/std over the full dataset before splitting leaks
test distribution into training and inflates every downstream number. It leaves no trace in the
logs. G3's subject-conditioning embedding is the explicit exception: at *k*>0 it may use only the
windows contained in the selected *k* trials, as defined by the generative-arm leakage rules; this
never permits subject-specific normalization.

### 3.4 Metrics

| Metric | What it captures | Why it is here |
|---|---|---|
| `accuracy` | Overall correctness | Baseline readability; misleading alone under class imbalance |
| `macro_f1` | Mean per-class F1, unweighted | Rest is over-represented; macro-F1 stops a model from coasting on it |
| `per_subject_accuracy` | Accuracy for each held-out subject separately | **Report the spread, not just the mean.** A mean over 8 subjects can look respectable while one of them sits at chance |
| `per_class_precision` | Precision for each of the 18 classes separately | When the model predicts this gesture, how often is it right? Catches classes the model over-predicts |
| `per_class_recall` | Recall for each of the 18 classes separately | How much of this gesture does the model find? Catches classes it quietly misses, which accuracy hides and macro-F1 averages away |
| `confusion_matrix` | Which movements are mistaken for which | Anatomically adjacent gestures confuse; the pattern is a result |
| `expected_calibration_error` | Gap between confidence and accuracy | Probability calibration. A wearable that is confidently wrong is worse than one that abstains |
| `overconfidence_error` | Confidence specifically on *incorrect* predictions | The language arm's benefit, if any, may live here rather than in accuracy |
| `robustness_drop` | Accuracy loss from clean → perturbed | Robustness experiments only |

The table above is the shared set. Generation-specific keys — the VAE terms, the G2 gate, and the
*k*-indexed curves — are defined in [generative-arm.md](generative-arm.md), "Metrics reference".
Between the two, every `evaluation.metrics` entry in every config is defined somewhere, and
`tests/test_experiment_configs.py` enforces that.

**Where each metric is computed (D6).** Cross-validation and testing run the same metric set
through the same code path, which is what makes them comparable. Each of the 8 fold models is
evaluated on the held-out test subjects and those values are averaged over the folds; that is
the extended analysis. The refit model is evaluated on the same test subjects, and its numbers
are the ones reported, including the F1 reference row.

Produce the confusion matrix for the cross-validation as well as the test evaluation. The
cross-validation matrix tells you whether a confusion holds up across folds. Two gestures that
collide in all eight are worth writing about; two that collide in one fold are probably noise.

**Easily missed — average fold confusion matrices, never sum them.** All 8 fold models are
evaluated on the *same* test subjects, so summing counts every test window eight times and
implies eight times the data. Take the element-wise mean instead, which leaves the matrix on the
scale of one evaluation. Same goes for any other count-based metric.

### 3.5 Reproducibility

`seed: 42` everywhere. Every run writes `run.json` (config, git commit, summary) and
`metrics.jsonl` through `RunLogger` — this is not optional and not W&B-dependent. W&B is
offline by default; nothing may depend on a network service to be reproducible.

**Easily missed:** seeding Python/NumPy/PyTorch does not make MPS and CUDA produce identical
results. Do not compare a local MPS number against a cloud CUDA number and call the difference
an effect. Cross-device comparisons need the same device.

### 3.6 Class imbalance

Rest is over-represented in DB2 Exercise B. Handle that in the loss, never by resampling:

| | Rule |
|---|---|
| Training | `class_weighted_cross_entropy` — weights inversely proportional to class frequency in the training split |
| Resampling | **Never.** No oversampling, no undersampling, on any split |
| Test set | **Never balanced.** A balanced test set does not estimate deployment performance |

The group's [prior work](https://arxiv.org/abs/2303.10336) oversamples within each subject to
equalize classes. That is fine for discrete trials, but not for this data. Windows are 400
samples at stride 100, so they already overlap by 75%, and duplicating one drops near-identical
copies of the same signal into a batch. §3.2 and §3.3 both warn against that correlation;
oversampling would let it back in. Weighting the loss rebalances the classes without duplicating
anything.

Do not do both. Macro-F1 already accounts for the imbalance and the weighted loss already
corrects for it. Resampling on top would leave `accuracy` describing a class distribution the
model never meets in deployment.

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
difference could be caused by the readout rather than by the thing under test, which defeats the
control's purpose.

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
| | `split_manifest` | Required for reportable subject-independent runs; load and verify it as §3.2 specifies |
| | `split`, `test_fraction` | Random-window leakage demonstration only |
| | `debug_split` | Non-reportable tiny/local subject holdout only |
| | `window_size`, `stride` | See §3.3 |
| | `normalize` | See §3.3 |
| | `max_subjects`, `max_windows_per_subject` | Debug-only caps for fast iteration |
| `encoder` | `checkpoint`, `freeze`, `embedding_dim` | Milestone-0 CNN, frozen, 256-d |
| `training` | `target`, `loss` | `class_label`, `cross_entropy` |
| | `batch_size`, `gradient_accumulation_steps` | Effective batch = product |
| | `epochs`, `learning_rate`, `weight_decay` | Optimization; F1's `epochs` is the initial cross-validation fold horizon |
| | `early_stopping`, `epoch_selection`, `horizon_escalation`, `refit` | F1 model-selection and refit contract (§5.2) |
| | `device` | `auto` (portable) \| `mps` (local-only) \| `cuda` (cloud-only) |
| | `save_checkpoint` | Writes `checkpoints/<name>/refit.pt` (consumed downstream) and `checkpoints/<name>/folds/fold{k}/best.pt` (analysis only) |
| `evaluation` | `metrics` | See §3.4 |
| | `save_predictions` | Persist per-window predictions for later analysis |

### 5.1 The two checkpoint kinds

A training run produces two kinds of checkpoint. The names differ because the selection rule and
the intended reader differ.

| Path | What it is | Who reads it |
|---|---|---|
| `checkpoints/<name>/refit.pt` | The model refit on the **full training-subject set** using the cross-validated hyperparameters | **Every downstream consumer** — the adapters, the generator, the robustness registry |
| `checkpoints/<name>/folds/fold{k}/best.pt` | Per-fold model at the **smoothed validation peak** (§5.2) | Extended analysis only — never consumed by another run |

**Why the refit artifact is not called `best`.** Within a fold, `best` carries its ordinary
sense: the epoch that scored highest on that fold's validation subjects. The refit has no
validation set at all, since every training subject goes into training, so it runs to a fixed
epoch count (§5.2) without early stopping. Giving both files the same name would leave `best`
meaning two different things, and the difference is one a reader needs to see. Both satisfy the
checkpoint contract: `{model_state, model_config}`.

### 5.2 Epoch selection: the smoothed validation peak

Each of the eight cross-validation folds trains for exactly **30 epochs**, with **no early
stopping**. At each epoch, compute 18-class macro-F1 separately for each of the fold's four
validation subjects, using the fixed class labels 0–17 and zero for an undefined class F1. The raw
fold score is the arithmetic mean of those four subject scores, so every validation subject has
exactly 25% weight; concatenating their windows before computing macro-F1 is forbidden (D21).

For epoch *e* in {10, …, 30}, the selection score is the arithmetic mean of those raw fold scores
over epochs *e*−9 through *e*. Epochs 1–9 are ineligible because they do not have a full 10-epoch
trailing window. Select the epoch with the highest score; if two or more scores are exactly equal,
select the earliest epoch. The fold checkpoint is the model state from that selected epoch. Record
the 10-epoch value as a model-selection diagnostic, not as the fold's reportable `macro_f1`.

All reportable validation metrics — including `macro_f1` — are computed from the predictions made
by that one selected fold checkpoint. Final-test metrics likewise come from predictions made by the
frozen selected fold checkpoints and the frozen refit checkpoint. Metrics are never averaged across
epochs: doing so would describe no single model, and is especially ill-defined for count-valued or
nonlinear outputs such as confusion matrices and ECE. D20 explicitly narrows D9's earlier phrase
"fold metrics are the smoothed values": only the validation-macro-F1 **selection score** is smoothed.

After all eight folds finish, sort their selected epochs as
*e*<sub>(1)</sub> ≤ … ≤ *e*<sub>(8)</sub> and compute

*E*<sub>refit</sub> = ⌈(*e*<sub>(4)</sub> + *e*<sub>(5)</sub>) / 2⌉.

If *E*<sub>refit</sub> is below 30, refit a fresh model on all 32 development subjects for exactly
that many epochs, again with no early stopping. The refit has no validation set: its budget is
transferred from cross-validation.

If *E*<sub>refit</sub> equals 30, the central selected epoch is at the observation boundary and the
initial horizon is treated as right-censored. Do **not** refit and do **not** inspect any final-test
prediction. Instead, rerun all eight folds from scratch with a 60-epoch horizon, keeping every other
setting fixed and applying the same complete 10-epoch selection window over epochs 10–60. Recompute
*E*<sub>refit</sub> from those eight replacement runs; the 30-epoch fold results are development
diagnostics, not reportable alternatives. If the new value is below 60, proceed with the refit. If
it equals 60, stop for a new owner decision — there is no second automatic extension.

No final-test inference begins until horizon selection is complete, the fresh refit checkpoint is
written and frozen, and the eight selected fold checkpoints from the same accepted horizon are
frozen as one set. Test performance can never trigger longer training or a second confirmatory
evaluation. D18 fixes the selection constants and D19 fixes this one allowed development-only
horizon escalation; neither is a runner default or per-run tuning knob.

Two simpler rules suggest themselves, and both fail. Picking the raw best epoch
selects partly for noise: validation macro-F1 jumps around from epoch to epoch, especially with
only 4 validation subjects, and whatever noise pushed one epoch to the top also inflates the
number you then report from it. Training to a fixed budget and averaging the last N epochs, which
is what [the group's prior work](https://arxiv.org/abs/2303.10336) does, avoids that problem, and
the instinct behind it is sound. But it measures wherever the run happened to stop. A model that
has overfit by epoch 800 of 2000 still gets read off at epoch 2000.

Smoothing takes the peak from the first rule and the averaging from the second.

**Easily missed:** the 10-epoch window, full-window requirement, initial 30-epoch horizon, single
60-epoch contingency, and earliest tie-break are protocol constants, not tuning knobs. Use one
horizon consistently across all eight folds; never extend only the folds that reached the boundary.

### 5.3 Sanity check: the refit against the fold distribution

The 8×8 matrix (§3.4) yields eight fold-model test scores. The refit trains on all 32 training
subjects while each fold model trains on 28. Those four extra subjects are worth something: in
subject-independent EMG the number of training subjects is the binding constraint, so **the
refit should land at or above the fold mean.** That is why the check is one-sided rather than a
containment test:

| Outcome | Reading |
|---|---|
| Refit ≥ fold mean | **Expected.** Not a flag. The size of the gap is itself a result — see below |
| Refit < fold mean | **Red flag.** The refit strictly dominates every fold model in training data, so underperforming points at a defect: the epoch budget transferring badly from 28 subjects to 32, a class-weight computed on the wrong split, or a bug in the refit path |
| Refit far above the fold maximum | **Soft flag.** Larger than four extra subjects can plausibly explain. Not automatically wrong, but check for leakage before reporting |

Report the refit score beside the fold mean and range, so a reader can check this rather than
take it on trust.

The gap itself is worth keeping. Refit-minus-fold-mean estimates what four more training subjects
buy in accuracy, which is a direct read on the marginal value of subject data — the quantity the
generative arm's personalization-efficiency curve approaches from the other side (§6,
`real_gesture_trials_saved`). Record it once the check passes instead of throwing it away.

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
until F1 is done and its accuracy is written down as measured.

**F3–F5 run against every arm, not just the encoder (D2).** Each perturbation is evaluated on
the F1 encoder, on the language-arm models, and on the generative arm's augmented decoders.
"The encoder is robust" says nothing about whether soft-prefix conditioning or synthetic
augmentation preserves that robustness — and whether augmentation buys accuracy at the cost of
robustness is the first question a reviewer will ask.

**How this is wired.** The perturbation acts on the input window and is target-agnostic, so the
two axes are separate files. Each `robustness_*.yaml` describes only the perturbation and carries
no checkpoint. The models to evaluate live in one shared registry,
[`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml),
and `scripts/evaluate.py` evaluates the perturbation against every registry target whose
checkpoint exists — skipping the rest with a log line. The suite therefore runs incrementally:
today it resolves to F1 only (and F1's checkpoint appears once F1 is trained); the language and
generative targets light up as those arms land, with no config change.

A target is just `(name, checkpoint)` — no `model_type`. That relies on the **checkpoint
contract**: every checkpoint is saved as `{model_state, model_config}`, so any consumer rebuilds
the architecture from the checkpoint alone. That matters for the adapter stacks
(encoder + projector + backbone + head), which no single `model_type` string could describe.
F1's trainer is the first code to honor this contract.

Targets point at **`refit.pt`** — see §5 for the two checkpoint kinds and why the refit artifact
does not reuse the name `best`.

The evaluation logic itself (data loader, perturbation transforms, metrics) is Milestone-0 work
and not written yet; `scripts/evaluate.py` currently resolves and reports the plan.

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
`calibration_strategy` on the shared personalization harness. D15 amends its strategy names and
matched-training semantics. Details below.

**D14 — G3's subject-calibration unit is a complete gesture trial, total per held-out subject.**
The trial budget, grid, acquisition schedule, and class-coverage reporting are specified in
[generative-arm.md](generative-arm.md), G3.

**D15 — G3 uses matched F1-head adaptation strategies.** The population reference is not adapted;
`real_adaptation` and `real_plus_synthetic_adaptation` share initialization, trainable surface,
optimizer schedule, and optimizer-step budget for *k*>0. See G3 and D5 below.

---

## 8. D4 and D5 in full

### D4 — Text-summary feature set and numeric formatting (L4)

Two problems with the feature list as specified, both of which weaken the baseline. This baseline
exists to be *hard to beat*, so a weak version does active damage: it hands the soft prefix a
victory that a reviewer will discount.

**Problem 1: `rms` is provably redundant.** For any window,
`rms = sqrt(mean² + sd²)` exactly (verified numerically to ~1e-16, for both population and
sample SD at N=400). Given `mean` and `standard_deviation`, `rms` adds *zero* information. The
specified five features are really four.

**Problem 2 — the more serious one: every specified feature is permutation-invariant.**
`mean`, `sd`, `rms`, `min`, `max` are all unchanged if you shuffle the samples within a window.
The text baseline therefore receives **no temporal information whatsoever**, while the encoder
path sees a 1-D CNN over the ordered signal. Any gap between them would then be partly
attributable to *temporal structure vs. none* rather than to *soft prefix vs. text*, so the
control would carry a confound of its own.

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

**In v1.** It is the first question any reviewer asks about the headline: *why synthesize data
instead of just fine-tuning on the real calibration trials already acquired?*

The adaptation baseline is `real_adaptation`: start from the F1 population head and perform
**head-only fine-tuning on every valid window contained in the selected *k* real gesture trials**
for that held-out subject. The encoder stays frozen. Full fine-tuning, test-time adaptation
(entropy minimization or normalization-stat updates), and a fresh randomly initialized head are
follow-ups, not v1 alternatives.

For *k*>0, its matched synthetic counterpart, `real_plus_synthetic_adaptation`, starts from the
same F1 head and trains the same head parameters with the same optimizer schedule and
optimizer-step budget; the treatment difference is adding subject-conditioned synthetic latent
windows. Exact batch composition, real-window exposure, synthetic weighting, and the numerical
step budget remain Pending. The unchanged F1 head is retained as `population_no_adaptation`, a
horizontal population reference. At *k*=0, `real_adaptation` is the same no-op reference, while
the synthetic arm may update the head using population-conditioned synthetic windows because no
subject-specific information exists. That synthetic-only point is a diagnostic, not a matched
treatment comparison.

Build the personalization runner with a pluggable `calibration_strategy` axis
(`population_no_adaptation` | `real_adaptation` | `real_plus_synthetic_adaptation`) from the first
line of code. These names replace the ambiguous prior labels, under which `real_only` and
`real_plus_adaptation` could describe the same operation.
