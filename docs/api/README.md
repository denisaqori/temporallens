# API reference — `src/temporallens`

This file carries the part of the documentation that cannot be generated: **why each module
exists, and which protocol decision it implements**. Signatures and parameter meanings live in
the docstrings, where they cannot drift from the code; duplicating them here would create a second
source of truth that goes stale silently — the same reason `STATUS.md` carries no commit hashes.

`tests/test_api_docs.py` asserts that every public module appears below and that nothing listed
has been deleted, so this page fails the suite rather than rotting.

Read [`../experiments/README.md`](../experiments/README.md) for the protocol itself. The decision
IDs referenced throughout are in [`../project/DECISIONS.md`](../project/DECISIONS.md).

---

## Layer 1 — data

Everything here is implemented and tested. See [`../figures/system-layers.svg`](../figures/system-layers.svg).

### `data.ninapro`
Reads DB2 Exercise B and defines the processed on-disk format.

The load path is where **D12** is enforced: labels come from `restimulus`/`rerepetition`, never
`stimulus`/`repetition`. On subject 2 those columns disagree on **19.9% of samples** with a median
onset shift of **856 ms** — roughly the first four windows of every gesture trial would be pure
rest labelled as movement. `read_raw_mat` refuses a file carrying only the uncorrected columns,
and `load_processed` refuses a processed file built from them, so a stale artifact fails loudly
instead of training happily.

Windowing is deliberately *not* baked into the processed format: F0 uses stride 200 and F1 stride
100 over the same recordings.

### `data.windows`
Cuts a recording into labelled windows, and materialises them on demand.

Two separate concerns, for two reasons. **D24**: a window never crosses a `(label, repetition)`
boundary, because a window spanning rest→gesture can carry only one label and is therefore wrong
for part of itself — the D12 failure one layer down, concentrated at onset. And **indexing is
separate from materialising** because at F1's stride a subject yields ~15,700 windows; a
materialised 28-subject fold is ~8 GB resident and twice that at peak, while the signals are only
~78 MB per subject, since stride-100 windows overlap 75% and store every sample four times.

### `data.splits`
Loads, verifies and hashes the frozen split manifest (**D16**).

`SplitManifest.load()` recomputes a SHA-256 over the split-defining fields *and* checks it against
a digest pinned in code, so re-hashing an edited manifest does not pass. It also verifies the
structure: roles partition all 40 subjects, every training subject validates in exactly one fold,
no fold validates on a test subject, and calibration and evaluation repetitions stay disjoint
(leakage rule 3). The digest is checked **last**, so structural errors get specific messages.

### `data.dataset`
A torch `Dataset` over a window index, materialising **per item** rather than per fold.

Normalization is applied here, so there is exactly one place a window can reach a model
un-normalized. Every item carries its subject, because every reported metric is also computed
per-subject (**D26**) and a prediction has to be traceable to the person it came from.

---

## Layer 2 — preprocessing and models

### `preprocessing.normalize`
Per-channel statistics, fitted on training data only — §3.3's silent leak.

The fitting set is always an explicit argument; there is no code path that can reach a validation
or test subject. Statistics are fitted over the samples training windows **cover, each counted
once**. Measured on subject 2, the three candidates agree on the per-channel standard deviation to
within 1.4%, so this is not an accuracy choice — it is one that must be *frozen*, because the
statistics are baked into the shared encoder checkpoint:

| candidate | std vs covered-once | depends on stride? |
|---|---:|---|
| whole recording | 0.216% | no |
| window-weighted | 1.432% | yes, 0.230% between stride 100 and 200 |
| **covered once** | — | no |

Covered-once wins on stride-independence (these statistics are reused by consumers that may window
differently) and because the samples the whole recording adds are ones nothing else touches: under
D24 the last `(L − window) mod stride` samples of every segment host no window — 0.54% of the
signal, 9,711 samples on subject 2, matching the formula exactly — and they are measurably quieter
(std 0.445–0.928 of the covered samples), being segment ends where a contraction is decaying.

### `models.encoders.cnn1d`
The Milestone-0 encoder and its head.

The spec fixes only the interface, so **nothing outside this module may depend on the internal
arrangement**. `freeze()` implements AGENTS.md's definition exactly — `requires_grad=False` *and*
`eval()` — and `train()` is overridden so a later `model.train()` cannot quietly undo it. The
`eval()` half is load-bearing rather than pedantic: this encoder has BatchNorm, and a frozen
encoder left in train mode keeps updating its running statistics from whichever arm is using it,
silently changing a checkpoint both arms are meant to share unchanged.

### `models.checkpoint`
The `{model_state, model_config}` contract (**D7**, and a hard constraint in AGENTS.md).

Exactly two top-level keys, because extra keys are how consumers start depending on what the
contract does not promise. `rebuild_model` reconstructs from the file alone — the reason a
robustness target can be just `(name, artifact)` with no `model_type`. Normalization statistics
live inside `model_config` and saving without them is **refused at write time**, since a consumer
rebuilding without them would normalize differently and the F1 reference row would stop being
comparable — the same argument **D23** makes for the temperature.

### `models.llm.masking`
Attention-mask helpers for the frozen-LM input paths.

`pooling: last_token` means the **last non-padding token**, never literally the final position.
`hidden_states[:, -1]` is correct only under left-padding; `attention_mask.sum(-1) - 1` only under
right-padding and only with no interior zeros. `rightmost_nonpadding_indices` is correct under
either, and `mask_derived_position_ids` covers left-padding's second trap, where real tokens would
otherwise sit at positions offset by the pad count.

---

## Layer 3 — training and evaluation

### `training.class_weights`
Inverse-frequency class weights over the **training split only** (**D11**).

Rest is 50.2% of windows at stride 100, about 17.5× the median gesture class, so a model
predicting only rest scores 50% accuracy. Imbalance is handled in the loss and **never by
resampling**: duplicating windows that already overlap 75% would put near-identical copies in a
batch, reintroducing the window-correlation hazard through the sampler. An absent class gets
weight 0, not infinity.

### `training.encoder`
One encoder run: a configuration in, a recorded result out.

`EncoderRunConfig` takes **explicit subject lists**, not a config path, so there is no code path
that can resolve "every subject" and the leakage rule is testable rather than merely documented.
Three rules are enforced here because each fails silently otherwise: statistics and class weights
are fitted after the split from training windows only; no resampling (the result reports total and
unique training-window counts so a test can assert they match); and no early stopping, since the
epoch count is a declared constant (**D18**/**D19**).

F0 is explicitly not a measurement, so a run here **cannot write `refit.pt`** — that filename is
the artifact both arms load.

### `evaluation.metrics`
The metric set, including the decision-bearing cases.

| Function | Decision it implements |
|---|---|
| `macro_f1` | **D21** — divides by every class, scoring an undefined class 0, so a small-*k* support set cannot quietly rescale it |
| `average_confusion_matrices` | **D8** — averages, never sums; all eight fold models score the *same* test subjects |
| `expected_calibration_error` | **D23** — adaptive bins at M=10. There is a noise floor near 0.037 at this project's effective *n*, and it is a detection threshold, not an offset to subtract |
| `overconfidence_error` | **D25** — the one-sided confidence-weighted positive gap, interpretable only beside ECE: a uniformly timid model scores a perfect 0 |
| `mean_confidence_when_wrong` | **D25** — descriptive only, and `None` rather than 0 with no errors. Never compared across arms |
| `per_subject` | **D26** — never pooled; a pooled number can look respectable while one subject sits at chance |
| `paired_bootstrap_interval` | **D26** — seeded percentile bootstrap over subjects, never windows. A window-level bootstrap gave an interval 20× too narrow on identical data |

### `evaluation.robustness`
Validators for robustness-target plan resolution — fail-closed sentinels, duplicate target names,
and shared-*k*=0 alias checks.

### `utils.device`, `utils.run_logger`
Portable device selection, and always-on local JSON logging. §3.5 makes the logger non-optional and
forbids depending on a network service for reproducibility.

---

## Scripts

| Script | Status |
|---|---|
| `prepare_dataset.py` | Raw `.mat` → processed `.npz` per subject. ~75 MB per subject, ~3 GB for all 40 |
| `verify_environment.py` | `make verify` — interpreter, packages, MPS |
| `evaluate.py` | Robustness driver. Resolves the plan; the evaluation logic itself is not written |
| `worktree.sh` | The single concurrency tool for every agent |

`train_encoder.py`, `train_adapter.py` and `make_report.py` are referenced by Makefile targets but
**not yet written**. Do not present those targets as functional.
