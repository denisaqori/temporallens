# API reference — `src/temporallens`

What each part of the package is for, how the parts connect, and the reasoning behind the choices
that are not obvious from the code. Signatures and parameter meanings are in the docstrings, where
they cannot drift from the code; repeating them here would create a second copy that goes stale.

- The protocol — what each experiment measures and why — is in
  [`../experiments/README.md`](../experiments/README.md).
- Decision IDs such as D31 point into [`../project/DECISIONS.md`](../project/DECISIONS.md), which
  records who decided what and when. The explanations here are meant to stand without them.
- [`../figures/system-layers.svg`](../figures/system-layers.svg) draws the layers described below.
- `tests/test_api_docs.py` fails if a module is missing from this page or a documented module no
  longer exists. It checks coverage, not whether an explanation is right.

---

## How the pieces fit

```
raw .mat ──prepare_dataset.py──▶ processed .npz, one file per subject
                                         │
experiment config ──train_encoder.py──▶ EncoderRunConfig ──▶ train_encoder()
                    (consume or refuse)                            │
   load recordings → index windows → fit normalization and class weights on the training
   subjects → WindowDataset → build the model → train a fixed number of epochs → evaluate
   → metrics → run directory (run.json, metrics.jsonl, predictions, optional checkpoint)
```

Data moves one way. Nothing downstream reaches back to choose subjects, and everything fitted from
data — normalization statistics, class weights — is fitted after the split, from the training
subjects alone. That ordering is the main defence against leakage, and much of the code below
exists to make it hard to break by accident.

### Running it

```bash
make prepare-debug-subjects   # raw → processed for subjects 1–3
make debug                    # F0: train on subjects 1–2, evaluate on held-out subject 3
```

Each run writes a directory under `results/runs/`: `run.json` (the configuration, the device
actually used, the git commit, summary metrics), `metrics.jsonl` (training loss per epoch) and
`predictions.npz` (probabilities, label and subject for every evaluated window). F0 checks that the
path works end to end; it is never a measurement, and it says so when it runs.

### Where to start reading

1. `configs/experiment/foundation/debug_tiny.yaml` beside `scripts/train_encoder.py`: every
   setting, and where each one goes.
2. `train_encoder()` in `training.encoder`: one whole run, top to bottom.
3. The leakage section of `tests/test_f0_pipeline.py`: the protocol's rules written as assertions.
4. The modules below, in the order a run calls them, each with its unit tests.

---

## Configuration

### From a config file to a run

`scripts/train_encoder.py` turns an experiment config into an `EncoderRunConfig`. Its rule is
**consume or refuse**: every leaf key in the config is passed to a run setting, checked against the
one value the code can honour, limited to an allowed set, or recorded — and any other key is an
error. A key that nothing reads is worse than a missing one, because it reads as a promise the code
does not keep.

The declarations at the top of the script are the contract itself:

| Declaration | Meaning |
|---|---|
| `FIELD_PATHS` | Config key → run setting, one to one |
| `CHECKED_PATHS` | Keys with exactly one legal value for this script, such as `dataset.exercise: B` |
| `ALLOWED_VALUE_PATHS` | Keys limited to a set; `tracking.wandb_mode` may not be `online`, because nothing logs remotely |
| `MIRRORED_PATHS` | One quantity declared twice — `input_channels` and `num_classes` under both `dataset` and `model` — which must agree |
| `RECORDED_PATHS` | Kept as run metadata, not acted on: `experiment.mode`, `protocol.*` |
| `PRODUCED_METRICS` | What a run computes; a config may ask for a subset, never more |

Two tests keep these honest: every `EncoderRunConfig` setting must have a config key (except
`output_dir` — where runs are written is not a protocol choice), and a real training pass must
produce exactly `PRODUCED_METRICS`.

The script runs one train/held-out split, which is F0's protocol. A config describing another
protocol is refused rather than approximated: `split: random_window` means the leakage
demonstration (F2), and otherwise a `split_manifest` means cross-validation (F1). F2 names the
manifest too — only to draw its windows from the development subjects — which is why the
random-window check comes first. A single-split number computed from either config would carry the
name of a protocol it did not follow.

A config's `protocol:` block governs whether its results may be **reported**, not whether it may
train. A blocked config that trains prints `NOT REPORTABLE`.

### Which config keys reach which module

| Config keys | Consumed by |
|---|---|
| `dataset.data_dir`, `dataset.debug_split.*`, `window_size`, `stride`, `max_windows_per_subject` | `data.ninapro`, `data.windows`, through `training.encoder` |
| `dataset.normalize` | `preprocessing.normalize` |
| `model.type`, `hidden_dim`, `embedding_dim`, `dropout` | `models.encoders.cnn1d` |
| `training.loss`, `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `num_workers` | `training.encoder` |
| `training.device` | `utils.device` |
| `training.save_checkpoint` | `models.checkpoint`, through `training.encoder` |
| `evaluation.metrics`, `evaluation.save_predictions` | `evaluation.metrics`, `training.encoder` |
| `tracking.*` | `utils.run_logger` |
| `protocol.*` | `evaluation.robustness`; printed by `train_encoder.py` |

---

## Data

### `data.ninapro`
Reads NinaPro DB2 Exercise B recordings and defines the processed on-disk format.

DB2 stores two versions of the gesture labels: the prompted ones (`stimulus`, `repetition`) and
corrected ones (`restimulus`, `rerepetition`), re-aligned to the movement the subject actually
made. Only the corrected columns are used. On subject 2 the two disagree on 19.9% of samples, with
a median onset shift of 856 ms — roughly the first four windows of every gesture trial would
otherwise be rest labelled as movement. The raw reader refuses a file that carries only the
uncorrected columns, and the processed loader refuses a file built from them, so a stale artifact
fails loudly instead of training. The reader also checks the exercise field, the subject number,
that the signal is finite, and that every gesture × repetition pair (17 × 6) is present.

The processed format stores signals and labels, not windows, because experiments window the same
recordings differently (stride 200 in F0, 100 in F1).

**Connects to** `scripts/prepare_dataset.py` (writes) · `training.encoder` (reads) ·
**Decision** D12

### `data.windows`
Turns a recording into a window index, and materialises windows from it on demand.

A window never spans two `(label, repetition)` segments. A window running from rest into a gesture
can carry only one label, so it would be wrong for part of itself — the uncorrected-label problem
again, concentrated at onsets. As a result the last `(L − window) mod stride` samples of each
segment host no window.

Indexing is kept apart from materialising because of memory. At stride 100 a subject yields about
15,700 windows, overlapping 75%, so materialising a 28-subject fold takes about 8 GB (twice that at
peak) while the signals themselves are about 78 MB per subject. The index holds positions and
labels, not samples.

The debug cap (`max_windows_per_subject`) takes a seeded subsample that keeps at least one window
of every class present. Truncating instead would drop whole classes.

**Connects to** `training.encoder`, `data.dataset` (use it) · the boundary recount in
`training.encoder` (checks it independently) · **Decision** D24

### `data.splits`
Loads and verifies the frozen subject split.

Eight subjects are the project's final test set and are never used for development. The other 32
form eight cross-validation folds of four validation subjects each. The manifest also fixes which
repetitions personalization experiments use for subject calibration (1 and 4) and which for
evaluation (2, 3, 5 and 6).

`SplitManifest.load()` first checks the structure: the roles partition all 40 subjects, every
development subject validates in exactly one fold, no fold validates on a test subject, and
calibration and evaluation repetitions do not overlap. It then compares a SHA-256 over the
split-defining fields with a digest pinned in code, so editing the manifest and re-hashing it does
not pass. The digest comes last so that structural problems get specific messages.

**Connects to** every reportable config, through `dataset.split_manifest` · **Decisions** D13, D16

### `data.dataset`
A PyTorch `Dataset` over a window index.

It materialises one window per item, so memory stays at the size of the index, and it applies the
normalization, which makes it the single place a window can reach a model. Each item is
`(window, label, subject)`: the subject travels with every prediction because every metric is also
computed per subject, and comparisons between experiments are paired by subject.

**Connects to** `data.windows`, `preprocessing.normalize` (uses) · `training.encoder` (builds it) ·
**Decision** D26

---

## Preprocessing and models

### `preprocessing.normalize`
Per-channel normalization statistics, fitted on training subjects only.

Statistics computed over every subject before the split would leak the validation and test
distributions into training, and nothing in the logs would show it. Here the fitting set is always
an explicit argument, so no code path can reach a held-out subject.

Within the training subjects, the statistics are computed over the samples the training windows
cover, each counted once. Three candidates were measured on subject 2:

| Fitting set | std differs from covered-once by | Depends on stride? |
|---|---:|---|
| Whole recording | 0.216% | No |
| Window-weighted (a sample counted once per window containing it) | 1.432% | Yes — 0.230% between stride 100 and 200 |
| **Covered samples, each once** | — | No |

They agree within 1.4%, so the choice fixes a definition rather than improving accuracy.
Covered-once does not depend on stride, which matters because the statistics travel inside the
encoder checkpoint and are reused by consumers that may window differently. It also leaves out the
windowless segment tails — 0.54% of the signal, 9,711 samples on subject 2 — which are quieter than
the rest (std 0.445–0.928 of the covered samples) because they are where contractions decay.

The config names the policy: `train_subject_covered_sample_stats` for subject splits and
`train_split_covered_sample_stats` for the random-window split.

**Connects to** `training.encoder` (fits) · `data.dataset` (applies) · `models.checkpoint`
(stores) · **Decision** D31

### `models.encoders.cnn1d`
The reference encoder and its classification head.

The encoder is three convolution, batch-norm and ReLU blocks (kernels 7, 5 and 3; `hidden_dim`
channels, then twice that), max-pooling by 4 after the first two, adaptive average pooling, dropout
and a linear projection to `embedding_dim`. The adaptive pool makes the embedding size independent
of window length. The head is dropout plus a linear layer to the 18 classes; the downstream
experiments keep the encoder and replace the head.

The layer stack is a frozen constant, `ARCHITECTURE`, not a configuration option, because those
values are what the name `cnn1d` means. If a config could change them, two runs could both declare
`model.type: cnn1d` and be different networks. `model_config()` records the stack in every
checkpoint, and `build_model` refuses one whose recorded stack or type differs from this code;
otherwise weights would load into right-shaped but wrong layers. Changing the stack means a new
model type with a new name.

`freeze()` sets `requires_grad=False` and switches to eval mode, and `train()` is overridden so a
later `model.train()` cannot undo it. The eval half matters because of batch norm: a frozen encoder
left in train mode keeps updating its running statistics from whichever experiment is using it,
silently changing the checkpoint all of them share.

**Connects to** `training.encoder` (builds) · `models.checkpoint` (rebuilds) · **Decision** D32;
the meaning of *frozen* is fixed in AGENTS.md

### `models.checkpoint`
Saves and rebuilds models under the `{model_state, model_config}` contract.

A checkpoint holds exactly two things: the weights, and a description complete enough to rebuild
the model with no outside information. That is why a robustness target can be just a name and a
file, with no model type alongside it. The normalization statistics are part of the description,
and saving without them is refused: a consumer that normalized differently would get a
working-looking but different model, and the downstream experiments would stop being comparable
with the F1 reference. The fitted temperature belongs in the description for the same reason.

Checkpoint names carry their role:

| File | What it is | Who loads it |
|---|---|---|
| `refit.pt` | The model refit on all 32 development subjects | Every downstream consumer |
| `folds/fold{k}/best.pt` | The best epoch within one cross-validation fold | Analysis only |
| `development/fold{k}/fixed_budget.pt` | Fitted on a fold's 28 training subjects for the fixed budget | Development-time selection and calibration only |

The refit has no validation set, so it cannot be a "best". F0 writes none of these; it saves
`encoder.pt` in its own run directory.

**Connects to** `models.encoders.cnn1d` (describes, rebuilds) · `training.encoder` (saves) · the
robustness registry (loads) · **Decisions** D7, D28, D31, D33; the contract is a hard constraint in
AGENTS.md

### `models.llm.masking`
Attention-mask helpers for the language-model input paths.

`pooling: last_token` means the **last non-padding token**, never simply the final position.
`hidden_states[:, -1]` is correct only with left padding, and `attention_mask.sum(-1) - 1` only with
right padding and no interior zeros. `rightmost_nonpadding_indices` is correct either way, and
`mask_derived_position_ids` handles the second left-padding trap, where real tokens would otherwise
sit at positions offset by the number of pads. Every row must contain at least one real token.

**Connects to** the language-arm adapters (not yet written)

---

## Training and evaluation

### `training.class_weights`
Class weights for the loss, from the training split only.

Rest is 50.2% of windows at stride 100, about 17.5 times the median gesture class, so a model that
only ever predicted rest would score 50% accuracy. The imbalance is handled by weighting the loss
by inverse class frequency (scaled so the weights average 1), never by resampling: windows already
overlap 75%, and duplicating them would put near-identical copies in a batch. A class absent from
the training split gets weight 0 — it has no frequency to invert, and 0 keeps the vector indexable
by label without inventing a target the split cannot teach.

**Connects to** `training.encoder` (calls it after the split) · **Decision** D11

### `training.encoder`
One encoder training run: a run configuration in, a recorded result out.

`EncoderRunConfig` takes explicit lists of training and held-out subjects rather than a config
path, so it has no way to resolve "every subject". It refuses overlapping lists, and checks the
loss, model type, normalization policy and device against what this code can do.

`train_encoder()` runs these steps in order:

| Step | What it guarantees |
|---|---|
| Resolve the device | See `utils.device`; the device used is recorded |
| Load and index recordings | Training and held-out subjects are loaded separately; no window crosses a segment |
| Fit normalization and class weights | From the training subjects' windows only, after the split |
| Build the model | Through `build_model`, so the declared model type is honoured |
| Train | Class-weighted cross-entropy; Adam with PyTorch's default betas and epsilon and coupled L2 weight decay; a fixed number of epochs, no early stopping, no scheduler |
| Evaluate | Softmax probabilities on the held-out subjects, then the metric set |
| Record | The run directory, an optional checkpoint, and an `EncoderRunResult` |

The result carries two audit facts so tests can check that the protocol held: the total and unique
training-window counts (equal means nothing was resampled), and the number of windows crossing a
segment boundary, recounted from the raw labels rather than taken from the windower. The recount
uses a binary search over segment starts; the obvious nested loop gives the same answer about 95
times more slowly, roughly 3 s per 28-subject fold.

A debug run is not a measurement, so it cannot write the checkpoint names other experiments load.

**Connects to** every module above · `scripts/train_encoder.py` (drives it) · **Decisions** D7,
D11, D18/D19 (fixed epochs), D24, D38 (these optimizer mechanics are the F1 reference)

### `evaluation.metrics`
The metric functions every experiment reports.

Each function scores one set of windows and returns a number or a per-class array. Grouping by
subject or by run happens outside them, so no metric can quietly choose an aggregation the protocol
leaves open. Temperature scaling is not part of them either: they score whatever probabilities they
receive, raw or scaled, and how the temperature is fitted is set by the protocol (§3.4).

| Function | What it measures, and the choice behind it |
|---|---|
| `accuracy`, `confusion_matrix` | Standard. Accuracy is secondary because rest is half of all windows |
| `per_class_precision`, `per_class_recall`, `per_class_f1` | Per-gesture behaviour a single number hides (D8) |
| `macro_f1` | The headline classification metric. It averages over all 18 classes and scores an undefined class as 0, so a small calibration set cannot quietly change the denominator (D21, D34) |
| `average_confusion_matrices` | Averages the fold models' matrices, never sums them; every fold model scores the same test subjects (D8) |
| `brier_score` | Squared error of the whole probability vector, summed over classes and averaged over windows, not halved. Its range is 0–2, because a confident wrong answer both misses the true class and puts its mass on another (D39) |
| `calibration_bins`, `CalibrationBins` | The one place confidence bins are formed: boundaries from the confidences alone, a confidence on a boundary in the upper bin, identical confidences always in one bin, empty bins dropped. It records both the nominal and the occupied bin count, because a value computed over 2 occupied bins is not comparable with one over 10 unless the report says so. ECE and the overconfidence error are computed from this object, so a reported occupancy and the number beside it cannot come from two different binnings (D39) |
| `expected_calibration_error` | Top-label ECE over 10 nominal equal-mass bins. Identical confidences always share a bin, so heavy ties — such as confidences that round to exactly 1.0 in float32 — can leave fewer than 10 bins occupied. A perfectly calibrated model still scores above 0 at finite sample size; that threshold depends on the evaluation design and is never subtracted (D23, D39) |
| `overconfidence_error` | The part of the calibration gap where confidence exceeds accuracy, weighted by confidence. Read it only beside ECE: a uniformly under-confident model scores a perfect 0 (D25) |
| `mean_confidence_when_wrong` | Descriptive only; `None` rather than 0 when there are no errors (D25) |
| `per_subject` | Applies a metric to each subject separately. A pooled number can look fine while one subject sits at chance (D26) |
| `paired_bootstrap_interval` | A seeded percentile bootstrap over subjects, never windows, for comparisons on the same subjects. Resampling windows gave an interval 20 times too narrow on identical data, because the overlapping windows of one contraction are one event (D26) |

**Connects to** `training.encoder` (calls it) · the future cross-validation and robustness
evaluators

### `evaluation.robustness`
Decides which robustness evaluations may run.

A robustness evaluation pairs a perturbation (noise, channel dropout, amplitude scaling) with a
trained model from the shared registry. These helpers find the reasons a pair must not run: an
unresolved `pending_*` value anywhere in a registry entry, a source experiment whose
`protocol.status` is not `ready`, duplicate target names, or a `shared_k0_with` reference to a
target that does not exist.

`protocol_blockers` is a single function used by both checks — a perturbation config checks itself,
and each target checks the experiment that produced it — so the two cannot apply different rules.
Readiness has to be declared: a config with no `protocol:` block produces no blockers, which is
indistinguishable from `status: ready`, so every config the registry names is required (by test) to
state its status.

**Connects to** `scripts/evaluate.py` (calls it) · `configs/experiment/robustness_targets.yaml` ·
**Decision** D2

### `utils.device`
Chooses the device a run uses.

`training.device` may be `auto`, `cpu`, `mps` or `cuda`. `auto` picks CUDA, then Apple MPS, then
CPU. Any other value pins the device, and a pinned device that is not available raises instead of
falling back: a config asking for `cuda` that silently ran on the CPU would record one thing and do
another. The device actually used goes into `run.json`, which keeps `auto` reproducible after the
fact. Numbers from different devices are not compared as if they were one: seeding does not make
MPS and CUDA numerically identical.

**Connects to** `training.encoder`

### `utils.run_logger`
Local, always-on JSON logging for every run.

Each run gets its own directory, with `run.json` (configuration, git commit, timestamps, summary,
artifacts) and `metrics.jsonl` (one line per logged step). Logging cannot be switched off and does
not depend on a network service, so the record of a run never depends on one.

**Connects to** `training.encoder` · `tracking.*` in the configs

---

## Scripts

| Script | What it does |
|---|---|
| `prepare_dataset.py` | Converts raw `.mat` recordings to processed `.npz`, one per subject — about 75 MB each, 3 GB for all 40. `make prepare-debug-subjects` prepares the three F0 needs |
| `train_encoder.py` | Turns a config into a run; see [Configuration](#configuration). `make debug` |
| `evaluate.py` | Robustness driver: works out which perturbation × target pairs may run. The evaluation itself is not written yet |
| `verify_environment.py` | `make verify`: interpreter, packages, MPS |
| `worktree.sh` | The one tool every agent uses for parallel work (AGENTS.md) |

`train_adapter.py` and `make_report.py` are named by Makefile targets but are not written yet.

---

## Tests

- **`tests/_synthetic.py`** builds recordings shaped like DB2 — alternating rest and gesture
  segments — for the tests. They are test doubles written only to temporary directories, never
  data, and nothing computed from them is reportable.
- **One unit-test file per module** (`test_normalize.py`, `test_models_cnn1d.py`,
  `test_checkpoint.py`, `test_metrics.py`, …). The metric tests compare against hand-computed
  values.
- **`test_f0_pipeline.py`** is the integration test: a whole run on synthetic subjects. Its leakage
  section asserts the protocol directly — the held-out subject never reaches training, statistics
  and class weights come from the training subjects only, no window crosses a segment, and nothing
  is resampled.
- **`test_train_encoder.py`** covers the config loader: mostly refusals, plus the two tests that
  keep its declarations in step with the run settings and the metrics.
- **`test_experiment_configs.py`, `test_evaluate.py`, `test_project_memory.py`** are structural:
  configs agree with the spec and with each other, robustness dispatch fails closed, and the
  project-memory tables stay well formed.

Many tests assert a refusal. When adding one, check that it fails with the guard removed; a test
that cannot fail protects nothing.
