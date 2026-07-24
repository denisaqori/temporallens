# Language arm (L-series) — secondary

**The question.** Does mapping temporal signal embeddings into a *frozen* language model's
embedding space improve decoding, calibration, or failure reporting beyond a well-trained
encoder alone?

**The expected answer is "little, for raw accuracy."** That is not a failure mode. A clean
negative, properly isolated by L3 and L4, is a publishable finding and is more interesting than
an unisolated positive. This arm is written so that a null result is as defensible as a
positive one — which is only true if the controls are airtight.

Read [README.md](README.md) first: §2 (vocabulary), §3 (shared protocol), §4 (the readout rule)
are assumed throughout and are not repeated here.

**Prerequisite:** F1 must be complete. Every L-series run loads the frozen encoder checkpoint
`checkpoints/baseline_cnn_subject_split/best.pt` and reports against F1's held-out-subject
accuracy as the reference row.

---

## The comparison this arm exists to make

| Row | Input path | Frozen backbone | Trainable | Config |
|---|---|---|---|---|
| **Reference** | — (encoder only) | — | head | F1 |
| **L2** | soft prefix | Llama 3.2 3B, **pretrained** | projector + head | `adapter_llama3b_subject_split` |
| **L3** | soft prefix | matched-size, **random init** | projector + head | `adapter_random_transformer` |
| **L4** | engineered text | Llama 3.2 3B, **pretrained** | head only | `adapter_text_summary_only` |

Two differences, each isolated by exactly one comparison:

- **L2 vs. L3** — identical size, identical readout, identical input path; the *only* difference
  is whether the weights are pretrained. This is what licenses any claim about **language
  pretraining**. If L2 ≈ L3, the effect was model capacity, not language.
- **L2 vs. L4** — identical backbone, identical readout; the only difference is whether the
  signal arrives as a *learned soft prefix* or as *engineered text*. This is what licenses any
  claim about the **projector**. If L2 ≈ L4, a prompt of numbers was enough and the learned
  path bought nothing.

Both comparisons are void if the readout differs across rows. See README §4.

---

## L0 — Mock-LLM plumbing check

**Config:** `adapter_mock_debug.yaml` · **Runs:** locally, seconds · **Mode:** `debug`

**What it measures.** Nothing scientific. It verifies that the shapes line up end to end:
encoder → projector → *N* soft-prefix vectors → concatenation with prompt embeddings → backbone
forward → pooled hidden state → head → 18 logits → cross-entropy → backward.

**Why it exists.** Every dimension bug in this arm is cheaper to find against a 128-d mock than
against a 3B model on a rented GPU. The mock backbone is a small random module with a
configurable `embedding_dim`; no Llama weights, no tokenizer, no network.

**Easily missed.**
- `mock_llm.embedding_dim` must equal `adapter.output_dim` (both 128 here). This config is the
  one place `output_dim` is a literal rather than `auto_from_llm_config`, because there is no
  model config to read.
- The encoder is **not** frozen here (`encoder.freeze: false`, `checkpoint: null`) — this is a
  plumbing test, not a measurement. Do not copy that setting into L1–L4.
- Passing L0 proves shapes, not correctness. A projector that emits constant vectors passes L0.

---

## L1 — Real Llama 1B smoke test (local)

**Config:** `adapter_llama1b_local.yaml` · **Runs:** locally on MPS · **Mode:** `debug`

**What it measures.** That the *real* embedding-injection path works: a genuine Llama tokenizer
and embedding table, real `inputs_embeds` assembly, on Apple Silicon.

**Why it exists.** To prove the mechanism before renting a GPU. Cloud time is the project's only
recurring cost, and discovering an injection bug there is the expensive way to find it.

**Easily missed.**
- **This is not a result.** `max_subjects: 4`, `max_windows_per_subject: 100`, one epoch. Never
  report a number from this config.
- Llama weights are **gated**. `hf auth login` and accepting Meta's licence must happen first;
  this is an account action, not a code action.
- `torch_dtype: float16` on MPS: some ops silently fall back or lose precision. If loss is NaN,
  test in float32 before assuming the adapter is wrong.
- `llm.device: mps` and `training.device: mps` are hard-coded here on purpose — this config is
  local-only by design. The 3B config uses `cuda` for the same reason.
- 1B and 3B have **different hidden sizes**. This is exactly why `adapter.output_dim` is
  `auto_from_llm_config`; a value hard-coded to pass here fails on the 3B run.

---

## L2 — Frozen pretrained LLM, subject-independent (the method)

**Config:** `adapter_llama3b_subject_split.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether soft-prefix conditioning of a frozen, *pretrained* language model
improves held-out-subject decoding, calibration, or overconfidence beyond the encoder alone.

**Why it exists.** It is the secondary hypothesis of the whole project. It is also the piece
that demonstrates wiring a modality into a frozen LLM's embedding space — the transferable
skill — independently of whether the result is positive.

**How to read it.** Never alone. L2 is meaningful only as the triple (L2, L3, L4) plus the F1
reference. Reporting L2 against F1 without L3 invites the immediate objection that any gain came
from parameter count.

**Easily missed.**
- The encoder is **frozen** and loaded from F1. If it trains here, this stops being an adapter
  study and becomes a bigger-model study.
- `adapter.output_dim: auto_from_llm_config` — read `hidden_size` from the model config at
  runtime. Hard-coding is the single most likely cause of "worked locally, failed on the pod."
- **Padding side.** `head.pooling: last_token` with a causal model assumes right-padding. Under
  left-padding the pooled vector is a pad embedding; training still runs and loss still falls,
  so this fails *silently*. Pin the tokenizer or pool the true last non-pad index, and assert it.
- Effective batch is `batch_size × gradient_accumulation_steps` = 4 × 8 = 32. Changing only
  `batch_size` to fit memory silently changes the optimization, not just the memory profile.
- `wandb_mode: online` here. Requires `wandb login` on the pod; the local JSON logger is still
  the source of truth.

---

## L3 — Matched-size random-init transformer (rigor baseline) — NEW

**Config:** `adapter_random_transformer.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether the effect in L2 is attributable to **language pretraining** or
merely to routing signals through a large fixed transformer.

**Why it exists.** It is the first question a strong reviewer asks, and without it no claim
about "language" survives contact. A frozen random transformer is still a high-dimensional
non-linear feature transform, and those can help on their own.

**Easily missed.**
- **"Matched" is a real constraint,** not a label. Same architecture, hidden size, layer count,
  and head count as `baseline.match_model` — instantiate from the *config* of
  `meta-llama/Llama-3.2-3B` with random weights, never a differently-shaped stand-in. If sizes
  differ, this controls for nothing.
- It is **frozen**, exactly like L2. A trainable random transformer is a completely different
  experiment.
- This row is *why* the readout is discriminative (README §4): a random model has no usable LM
  head, so a generative readout cannot be run here at all.
- Seed matters more than usual — the "features" are a draw from an initialization distribution.
  If L2 and L3 land close, re-run L3 across several seeds before concluding anything.
- **A null here is the good outcome for rigor.** If L2 ≈ L3, report that plainly: the honest
  conclusion is that language pretraining did not help, and that is the finding.

---

## L4 — Text-summary-only (rigor baseline) — NEW

**Config:** `adapter_text_summary_only.yaml` · **Runs:** cloud CUDA · **Mode:** `train`

**What it measures.** Whether the *learned soft prefix* beats simply telling the same frozen
model the same information in words.

**Why it exists.** If a hand-written prompt of per-channel statistics matches the projector, the
projector is decoration. This baseline must be made **as strong as possible** — a weak version
manufactures a win for L2 that any reviewer will discount.

### Feature set (decision D4)

Six features per channel × 12 channels = **72 scalars**, computed on the **normalized** signal
using the same `train_subjects_global_stats` statistics as the encoder path.

| Feature | Contributes |
|---|---|
| `mean`, `standard_deviation`, `minimum`, `maximum` | Amplitude distribution |
| `waveform_length` — Σ\|xᵢ₊₁ − xᵢ\| | Temporal: signal path length |
| `zero_crossings` | Temporal: coarse frequency content |

**Why `rms` was removed.** `rms = sqrt(mean² + sd²)` exactly, for any window (verified to ~1e-16
for both population and sample SD at N = 400). Given `mean` and `standard_deviation` it carries
**zero** additional information. Including it wastes tokens and misrepresents the baseline as
richer than it is.

**Why two temporal features were added — this is the important one.** `mean`, `sd`, `min`,
`max` (and `rms`) are all **permutation-invariant**: shuffle the samples within a window and
every one of them is unchanged. A baseline built only from those receives *no temporal
information at all*, while L2's encoder path runs a 1-D CNN over the ordered signal. Any L2 − L4
gap would then be partly **temporal structure vs. none**, not **soft prefix vs. text** — a
confound living inside the control whose whole purpose is to remove confounds. Waveform length
and zero crossings are standard Hudgins time-domain sEMG features and restore ordering
information.

### Prompt formatting (decision D4)

1. **Fixed-point, 2 decimals.** Post-normalization values sit around ±5, so `%.2f` keeps ample
   resolution at ~5 characters. **No scientific notation** (`1.2e-05` fragments badly under BPE)
   and no full float repr (17 digits of token noise).
2. **One labelled line per channel**, so position is never inferred:
   `ch01 mean=-0.12 sd=0.98 min=-3.41 max=3.02 wl=41.7 zc=88`
   A bare list of 72 comma-separated numbers forces the model to count positions, which it does
   badly. Handicapping the prompt defeats the baseline's purpose.
3. **The template is a hyperparameter.** Freeze it, version it, and record it verbatim in
   `run.json`. If it drifts between runs, L4 is not comparable to itself.

Budget ≈ 200–250 prompt tokens per window.

**Easily missed.**
- **No projector and no encoder** — deliberately. `adapter:` and `encoder:` are absent, and the
  head is the only trainable module. That absence *is* the experiment; it is not an oversight.
- The backbone must be the **same** frozen pretrained model as L2. Comparing L4 on 1B against L2
  on 3B measures model size.
- Features are computed on the normalized signal, not raw mV — otherwise L2 and L4 see different
  preprocessing and the comparison is unfair before it starts.

---

## L5 — Generative readout — DEFERRED (D3)

Not in v1. The frozen LLM's own LM head emits a label token instead of a head reading the hidden
state. It would add natural-language output for the structured-report demo, but no inferential
power the L2/L3/L4 triple does not already provide, and it costs real work (label tokenization,
multi-token labels, calibrated 18-class probabilities from token logits).

**Revisit if** L2 beats L3 under the shared readout, or if the structured-report demo is needed
for outreach. If it is ever run, it is a secondary ablation **on L2 only** — never a control's
readout.

---

## Robustness (D2)

The three F3–F5 perturbations are also evaluated against the L2, L3, and L4 models, not only
against the F1 encoder. Whether soft-prefix conditioning preserves or degrades robustness under
subject shift is a genuine result, and possibly this arm's most interesting one — the language
arm's value, if any, is more likely to appear in calibration and failure behaviour than in clean
accuracy.

To evaluate them, add `l2_frozen_llm`, `l3_random_transformer`, and `l4_text_summary` to
[`configs/experiment/robustness_targets.yaml`](../../configs/experiment/robustness_targets.yaml)
(already listed) — each becomes active the moment its checkpoint exists. See README §6 for the
target registry and the checkpoint contract; no per-perturbation config change is needed.

---

## Reporting checklist

Before any L-series number leaves this repository:

- [ ] F1 reference accuracy reported alongside it
- [ ] L3 and L4 reported alongside L2 — never L2 versus F1 alone
- [ ] Identical readout confirmed across L2, L3, L4
- [ ] Per-subject accuracies shown with spread, not just the mean over held-out subjects
- [ ] ECE and overconfidence reported, not only accuracy
- [ ] A null result stated plainly, without softening
