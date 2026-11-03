# API reference — `src/temporallens`

What each part of the package is for, how the parts connect, and the reasoning behind the choices that are not obvious from the code. Signatures and parameter meanings are in the docstrings, where they cannot drift from the code; repeating them here would create a second copy that goes stale.

- The protocol — what each experiment measures and why — is in [`../experiments/README.md`](../experiments/README.md).
- Decision IDs such as D31 point into [`../project/DECISIONS.md`](../project/DECISIONS.md), which records who decided what and when. The explanations here are meant to stand without them.
- [The system diagram](../figures/system-layers.svg) separates processes, tensor products and saved artifacts, with F1-sized dimensions at each boundary. Its [reading guide](../figures/README.md) explains the parallel arms, control paths and implemented-versus-planned boundary.
- `tests/test_api_docs.py` fails if a module is missing from this page or a documented module no longer exists. It checks coverage, not whether an explanation is right.

---

## How the pieces fit

```
raw .mat ──prepare_dataset.py──▶ processed .npz, one file per subject
                                         │
experiment config ──train_encoder.py──▶ EncoderRunConfig ──▶ train_encoder()          (F0)
                    (consume or refuse)                            │
   load recordings → index windows → fit normalization and class weights on the training
   subjects → WindowDataset → build the model → train a fixed number of epochs → evaluate
   → metrics → run directory (run.json, metrics.jsonl, predictions, optional checkpoint)

F1 config ──train_encoder.py──▶ SubjectCVConfig ──▶ run_subject_cross_validation()     (F1)
   32 development subjects only → 8 fold trajectories (validated and snapshotted each epoch)
   → D18 selection → common budget B → D19 escalation or block → best.pt / fixed_budget.pt
   → pooled out-of-fold T → fresh all-32 refit for B epochs → refit.pt carrying T
   (preliminary on every device; a reportable run is refused until the final-test stage exists)

F1 artifact ──verify_f1_artifact(expectation, pool)──▶ model, statistics, T    (any consumer)
   role, fold, B, horizon, subjects, lineage, digests checked against the consumer's own record
   → class weights, window identity and normalization recomputed from the permitted pool
```

Data moves one way. Nothing downstream reaches back to choose subjects, and everything fitted from data — normalization statistics, class weights — is fitted after the split, from the training subjects alone. That ordering is the main defence against leakage, and much of the code below exists to make it hard to break by accident.

### Running it

```bash
make prepare-debug-subjects   # raw → processed for subjects 1–3
make debug                    # F0: train on subjects 1–2, evaluate on held-out subject 3
```

Each run writes a directory under `results/runs/`: `run.json` (the configuration, the device actually used, the git commit, summary metrics), `metrics.jsonl` (training loss per epoch) and `predictions.npz` (probabilities, label and subject for every evaluated window). F0 checks that the path works end to end; it is never a measurement, and it says so when it runs.

`make train-baseline` runs F1's development harness (`training.subject_cv`) from committed code. It needs all 32 development subjects prepared and never loads the eight final-test subjects. It is a preliminary run (D47) on every device, CUDA included, and writes under `checkpoints/preliminary/`. `--reportable` requests the canonical run the arms read, and is refused before any data loads: the one-shot final-test evaluation is a separate, CUDA-only stage that is not implemented, and it must be rehearsed on development stand-ins first.

### Where to start reading

1. `configs/experiment/foundation/debug_tiny.yaml` beside `scripts/train_encoder.py`: every setting, and where each one goes.
2. `train_encoder()` in `training.encoder`: one whole run, top to bottom.
3. The leakage section of `tests/test_f0_pipeline.py`: the protocol's rules written as assertions.
4. The modules below, in the order a run calls them, each with its unit tests.

---

## Configuration

### From a config file to a run

`scripts/train_encoder.py` turns an experiment config into an `EncoderRunConfig`. Its rule is **consume or refuse**: every leaf key in the config is passed to a run setting, checked against the one value the code can honour, limited to an allowed set, or recorded — and any other key is an error. A key that nothing reads is worse than a missing one, because it reads as a promise the code does not keep.

The declarations at the top of the script are the contract itself:

| Declaration | Meaning |
|---|---|
| `FIELD_PATHS` | Config key → run setting, one to one |
| `CHECKED_PATHS` | Keys with exactly one legal value for this script, such as `dataset.exercise: B` |
| `ALLOWED_VALUE_PATHS` | Keys limited to a set: `tracking.wandb_mode` may not be `online`, because nothing logs remotely; `experiment.mode` must be `debug` or `train` |
| `MIRRORED_PATHS` | One quantity declared twice — `input_channels` and `num_classes` under both `dataset` and `model` — which must agree |
| `RECORDED_PATHS` | Kept as run metadata, not acted on: `protocol.*` |
| `PRODUCED_METRICS` | What a run computes; a config may ask for a subset, never more |

Every config is read once, through `utils.yaml_io`, so a repeated key is an error rather than a silent last-write-wins. Checked keys are required as well as checked — a check that applies only when a key is present can be dodged by leaving the key out. The mode matters beyond validation: only a `debug` run may start from uncommitted code. A value the run config refuses is reported as a config error naming the file.

Two tests keep these honest: every `EncoderRunConfig` setting must have a config key (except `output_dir` — where runs are written is not a protocol choice), and a real training pass must produce exactly `PRODUCED_METRICS`.

The F0 bridge runs one train/held-out split. A config describing another protocol is refused rather than approximated there: `split: random_window` means the leakage demonstration (F2), and a `split_manifest` means cross-validation (F1). F2 names the manifest too — only to draw its windows from the development subjects — which is why the random-window check comes first. A single-split number computed from either config would carry the name of a protocol it did not follow.

An F1 config instead goes to `build_f1_config`, which applies the same rule with its own tables:

| Declaration | Meaning |
|---|---|
| `F1_FIELD_PATHS` | Config key → `SubjectCVConfig` setting; the manifest path is loaded and verified separately |
| `F1_CHECKED_PATHS` | The protocol constants the harness implements one way: D18's window and tie rule, D19's horizons, D38's optimizer, D33/D39's temperature recipe, D40's cross-fit, D48's snapshot rules and required provenance, D85's digest encoding. Each expected value is the code's own constant where one exists (`SELECTION_WINDOW_EPOCHS`, `ADAM_BETAS`, `INVERSE_TEMPERATURE_BOUNDS`, …), so the config and the code cannot drift apart unnoticed |
| `F1_FINAL_TEST_PATHS` | Declarations for the CUDA-only held-out evaluation, which is not implemented and never run here. They are checked and recorded so a changed declaration is still refused, and the run summary states that the final test was not evaluated |
| `F1_RECORDED_PATHS` | Run metadata: `protocol.*` and the tracking mode |

The comparison is strict about types: YAML's `0` is not `false`, and an `amsgrad: 0` would otherwise pass as the declared `false`. `evaluation.temperature_fit.checkpoint_pattern` must equal the canonical path the harness writes, so a consumer config that names that path is naming this file. An F1 config whose `protocol.status` is not ready is refused outright rather than trained with a `NOT REPORTABLE` note, because its artifacts would sit where consumers look.

A config's `protocol:` block governs whether its results may be **reported**, not whether it may train. A blocked config that trains prints `NOT REPORTABLE`. What the loader records, plus the config file's path and hash, goes into the run log as metadata.

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
| F1's `dataset.split_manifest` | `data.splits`, through `build_f1_config` |
| F1's `training.epoch_selection.*`, `training.horizon_escalation.*`, `training.refit.*` | checked against `training.selection`'s constants; executed by `training.subject_cv` |
| F1's `training.optimization.*`, `training.development_checkpoints.*`, `training.checkpoint_metadata.*` | checked; executed by `training.subject_cv` and `training.class_weights` |
| F1's `evaluation.temperature_fit.*`, `evaluation.calibration_execution.*` (except the bootstrap keys) | checked against `evaluation.calibration`'s constants; executed by `training.subject_cv` |
| F1's other `evaluation.*` keys | checked and recorded for the unimplemented CUDA-only final-test stage |
| F2's `evaluation.temperature_fit.execution` assignment keys; G3/G4's `personalization.trial_selection.*` | checked against the executed protocol, not read as parameters, by `data.assignments` (`check_f2_config`, `check_g3_config`) |

---

## Data

### `data.ninapro`
Reads NinaPro DB2 Exercise B recordings and defines the processed on-disk format.

DB2 stores two versions of the gesture labels: the prompted ones (`stimulus`, `repetition`) and corrected ones (`restimulus`, `rerepetition`), re-aligned to the movement the subject actually made. Only the corrected columns are used. On subject 2 the two disagree on 19.9% of samples, with a median onset shift of 856 ms — roughly the first four windows of every gesture trial would otherwise be rest labelled as movement. The raw reader refuses a file that carries only the uncorrected columns, and the processed loader refuses a file built from them, so a stale artifact fails loudly instead of training. The reader also checks the exercise field, the subject number, that the signal is finite, and that every gesture × repetition pair (17 × 6) is present.

The processed format stores signals and labels, not windows, because experiments window the same recordings differently (stride 200 in F0, 100 in F1).

Both readers convert labels, repetitions and identifiers to integers only if nothing is lost: a fraction, a non-finite value or a number outside int16 is refused rather than truncated or wrapped. Casting first and checking afterwards had read 1.9 as 1. MATLAB stores whole numbers as doubles, so whole-number floats are accepted. `save_processed` writes to a temporary file in the same directory and moves it into place in one step, so an interrupted write never leaves a partial archive at the final name; any file `load_processed` cannot read is reported as a format error.

**Connects to** `scripts/prepare_dataset.py` (writes) · `training.encoder` (reads) · **Decision** D12

### `data.windows`
Turns a recording into a window index, and materialises windows from it on demand.

A window never spans two `(label, repetition)` segments. A window running from rest into a gesture can carry only one label, so it would be wrong for part of itself — the uncorrected-label problem again, concentrated at onsets. Windows start at each segment's first sample and step by the stride, so the last `(L − window) mod stride` samples of each segment host no window: 9,711 samples (0.54%) on subject 2 at stride 100, 1.10% at F0's stride 200.

Indexing is kept apart from materialising because of memory. At stride 100 a subject yields about 17,300 windows (17,239–17,374 on subjects 1–3), overlapping 75%, so a materialised subject is about 330 MB and a 28-subject fold about 9.3 GB (twice that at peak), while the signals themselves are about 87 MB per subject. The index holds positions and labels, not samples.

What a recording looks like, measured on subject 2 (about 900 s): 102 gesture segments (17 × 6) and 103 rest segments, one before each gesture and one after the last. Because the labels are corrected, gestures are shorter than the protocol's 5 s prompt (median 4.07 s) and rest longer than its 3 s (median 4.65 s). No segment is shorter than one window, so none is lost whole.

The debug cap (`max_windows_per_subject`) takes a seeded subsample that keeps at least one window of every class present. Truncating instead would drop whole classes.

**Connects to** `training.encoder`, `data.dataset` (use it) · the boundary recount in `training.encoder` (checks it independently) · **Decision** D24

### `data.splits`
Loads and verifies the frozen subject split.

Eight subjects are the project's final test set and are never used for development. The other 32 form eight cross-validation folds of four validation subjects each. The manifest also fixes which repetitions personalization experiments use for subject calibration (1 and 4) and which for evaluation (2, 3, 5 and 6).

`SplitManifest.load()` first checks the structure: the roles partition all 40 subjects, every development subject validates in exactly one fold, no fold validates on a test subject, and calibration and evaluation repetitions do not overlap. It then compares a SHA-256 over the split-defining fields with a digest pinned in code, so editing the manifest and re-hashing it does not pass. The digest comes last so that structural problems get specific messages.

**Connects to** every reportable config, through `dataset.split_manifest` · `utils.yaml_io` (reads the manifest strictly) · **Decisions** D13, D16

### `data.assignments`
Builds the two randomized data assignments the protocol freezes before any fit: F2's window folds and G3/G4's acquisition schedules.

A seed does not archive a split, because NumPy may change how a generator turns a seed into a permutation. So each builder emits the realized membership together with the inputs it was drawn from, and every later consumer loads that artifact instead of redrawing it. The membership is a pure function of fixed keys: `f2_fold_membership` depends only on the window count, `g3_trial_order` only on subject and schedule index. Labels, repetitions and signal therefore change an artifact's provenance, never who lands where, and those two functions are what registration re-runs to verify a saved artifact.

The builders read their real inputs rather than accepting a prepared index, so a frozen split cannot be the split of a debug subsample. `build_f2_assignment` opens the manifest's 32 development recordings one at a time, indexes them uncapped at 400/100 with the same indexer training uses, and hashes each recording's content into the source identity; held-out subjects' files are never opened (D41). `build_g3_schedule` refuses a recording unless each of its 34 eligible trials is one complete windowable segment.

`check_f2_config` and `check_g3_config` compare the config leaves describing these mechanics with what the module executes. The builders deliberately do not read those values — one versioned protocol, not a parameterized one — so without the check an edited config would be silently ignored. Artifacts are write-once and take the mode the user's umask gives any new file. What each of their four digests identifies is defined once, in the experiments README §3.5.

**Connects to** `data.ninapro` (reads the processed recordings) · `data.windows` (indexes them) · `data.splits` (subject and repetition roles) · `utils.randomness` (keys) · `data.assignment_registry` (freezes and serves the artifacts) · **Decisions** D43–D46

### `data.assignment_registry`
The Git-tracked record that makes a saved assignment trustworthy.

A checksum check is only as good as the place the expected checksum comes from: a digest kept beside the file can be rewritten along with it. The registry, `configs/splits/assignment_registry.yaml` beside the split manifest, holds each artifact's logical identity, source-pool digest, material digest, path and exact-file digest, and Git history is what vouches for those entries. The artifacts themselves stay under `data/` because they carry NinaPro labels.

`register_assignment` is the one-time freeze. It loads the saved file against the digest `save_assignment` returned and the current source digest, re-derives the membership, refuses any difference and only then appends an entry. Re-registering the identical artifact is a no-op; a different artifact for an already frozen identity is refused, never substituted. Registration under a NumPy version other than the one that drew the artifact is refused rather than skipped, because skipping would admit exactly the artifacts the check exists to catch. `load_registered_assignment` serves later runs and never redraws, so a frozen assignment stays authoritative whatever NumPy does next.

Entries key on the source digest, so changed inputs raise `UnregisteredAssignmentError` instead of resolving to the assignment frozen for the old inputs. A reportable run can only see committed entries because non-debug runs already refuse a dirty tree, so there is no separate Git check here. Registration assumes one writer: it is a freeze reviewed in a commit, not something concurrent runs do.

**Connects to** `data.assignments` (verifies and re-derives its artifacts) · `utils.yaml_io` (reads the registry strictly) · `utils.run_logger` (the clean-tree rule that keeps uncommitted entries out of reportable runs) · **Decision** D46

### `data.dataset`
A PyTorch `Dataset` over a window index.

It materialises windows on demand, so memory stays at the size of the index, and it applies the normalization, which makes it the single place a window can reach a model. Each item is `(window, label, subject)`: the subject travels with every prediction because every metric is also computed per subject, and comparisons between experiments are paired by subject.

`__getitems__` lets the DataLoader fetch a whole batch in one call, cut with one vectorized gather per subject. It returns exactly the per-item windows (a test compares them bitwise). Measured on subjects 1–3 at batch 64, loading fell from 20 to 11 µs per window, about 9.7 s to 5.3 s for one 484,000-window F1 fold epoch. That is worth having but not the main cost: in a profile of a fold fit, convolution compute dominated.

**Connects to** `data.windows`, `preprocessing.normalize` (uses) · `training.encoder` (builds it) · **Decision** D26

---

## Preprocessing and models

### `preprocessing.normalize`
Per-channel normalization statistics, fitted on training subjects only.

Statistics computed over every subject before the split would leak the validation and test distributions into training, and nothing in the logs would show it. Here the fitting set is always an explicit argument, so no code path can reach a held-out subject.

Within the training subjects, the statistics are computed over the samples the training windows cover, each counted once. Three candidates were measured on subject 2:

| Fitting set | std differs from covered-once by | Depends on stride? |
|---|---:|---|
| Whole recording | 0.216% | No |
| Window-weighted (a sample counted once per window containing it) | 1.432% | Yes — 0.230% between stride 100 and 200 |
| **Covered samples, each once** | — | No |

They agree within 1.4%, so the choice fixes a definition rather than improving accuracy. Covered-once does not depend on stride, which matters because the statistics travel inside the encoder checkpoint and are reused by consumers that may window differently. It also leaves out the windowless segment tails (see `data.windows`), which are quieter than the rest (std 0.445–0.928 of the covered samples) because they are where contractions decay.

The config names the policy: `train_subject_covered_sample_stats` for subject splits and `train_split_covered_sample_stats` for the random-window split.

`ChannelStats.from_dict` checks what it reads back — equal-length finite vectors, a std of at least the fitting floor (1e-8) on every channel, a named fitting set, distinct in-range floored channels — because the statistics arrive from checkpoints written by other runs, and a malformed payload otherwise fails late: a zero std divides every window by zero. The checks run on the float32 values the statistics are used as, not on the payload: a std of 1e-100 is positive as written but stores as zero, and a mean of 1e39 stores as infinity.

**Connects to** `training.encoder` (fits) · `data.dataset` (applies) · `models.checkpoint` (stores) · **Decision** D31

### `models.encoders.cnn1d`
The reference encoder and its classification head.

The encoder is three convolution, batch-norm and ReLU blocks (kernels 7, 5 and 3; `hidden_dim` channels, then twice that), max-pooling by 4 after the first two, adaptive average pooling, dropout and a linear projection to `embedding_dim`. The adaptive pool makes the embedding size independent of window length. The head is dropout plus a linear layer to the 18 classes. Which of the two each downstream experiment uses is in the table below.

The layer stack is a frozen constant, `ARCHITECTURE`, not a configuration option, because those values are what the name `cnn1d` means. If a config could change them, two runs could both declare `model.type: cnn1d` and be different networks. `model_config()` records the stack in every checkpoint, and both builders refuse a recorded stack or type that differs from this code. `rebuild_from_config`, the path a checkpoint takes, also refuses a config that records no stack at all: fresh construction through `build_model` has nothing recorded yet, but a checkpoint must. Otherwise weights would load into right-shaped but wrong layers. Only an absent key counts as unrecorded; a present `architecture` must be the recorded mapping in both builders, so `architecture: null` cannot pass for one. Changing the stack means a new model type with a new name.

`freeze()` sets `requires_grad=False` and switches to eval mode, and `train()` is overridden so a later `model.train()` — including one on a larger model that wraps the encoder — cannot undo it. The eval half matters because of batch norm: a frozen encoder left in train mode keeps updating its running statistics from whichever experiment is using it, silently changing the checkpoint all of them share.

**`embedding_dim` is the interface every consumer reads.** It is the length of the vector the encoder produces for one window — 256 in F1, 64 in F0. The head, the language-arm projector, the generator and G2's discriminator all take this vector as input, so changing its size means a new F1 checkpoint and retraining everything downstream. Training to classify gestures arranges the embedding space so that windows of the same gesture land close together; that is why the generative arm centres embeddings on each gesture's mean before using them to describe a subject (D27). The layer-by-layer shapes are in [the figure guide](../figures/README.md#b--foundation-encoder--classifier).

**Who uses which part.** The checkpoint stores the whole classifier — encoder and head — and each consumer rebuilds it and takes what it needs:

| Consumer | Encoder | Head |
|---|---|---|
| F1 | Trained | Trained |
| L2, L3 (soft prefix) | Frozen; its embedding feeds the projector | Not used; the language arm has its own head on the transformer output |
| L4 (engineered text) | **Not used at all** — the window becomes 72 engineered statistics, then text | Not used |
| L6 (no-LLM control) | Frozen; its embedding feeds a conventional MLP head directly | Not used; L6 trains its own MLP head |
| G1 generator | Frozen; the VAE models and generates encoder embeddings | Not used |
| G2 quality gate | Frozen; the discriminator compares real and generated embeddings | Not used |
| G3/G4 personalization | Frozen; real calibration windows go through it, synthetic data is already in embedding space | F1's head is the starting point: unchanged for `population_no_adaptation`, fine-tuned (head only) by the adapted strategies |
| F3–F5 on the F1 target | Unchanged | Unchanged — the whole classifier is scored under perturbation |

**Easily missed:**

- Only the encoder can be frozen. Wherever a head is used downstream, it is trainable.
- Three different things are called an embedding or latent here and are easy to confuse: the encoder embedding (256-d), the VAE's noise latent (32-d) and the subject embedding the generator is conditioned on (16-d). The language model's hidden width (3,072 for the 3B model) is a fourth size, and the projector maps the first into it.

**Connects to** `training.encoder` (builds) · `models.checkpoint` (rebuilds) · **Decision** D32; the meaning of *frozen* is fixed in AGENTS.md

### `models.checkpoint`
Saves and rebuilds models under the `{model_state, model_config}` contract.

A checkpoint holds exactly two things: the weights, and a description complete enough to rebuild the model with no outside information. That is why a robustness target can be just a name and a file, with no model type alongside it. The normalization statistics are part of the description, and saving without them is refused: a consumer that normalized differently would get a working-looking but different model, and the downstream experiments would stop being comparable with the F1 reference. The fitted temperature belongs in the description for the same reason.

Saving and loading share one `model_config` validator — the current `schema_version`, a model type, and valid normalization statistics for the channel count the model takes — and `save_checkpoint` reads every file back through `load_checkpoint` before returning, deleting one that fails. So the writer cannot produce a file the loader refuses, such as 11-channel statistics for a 12-channel model. The read-back exists because the validator alone missed a real case: `torch.__version__` is a `str` subclass that `weights_only` loading will not unpickle, and the first full F1 test run wrote such a value into provenance. `load_checkpoint` also refuses any top-level key set other than the two and a `model_state` that is not named tensors, and reads with `weights_only=True`, so a file can never make unpickling run code. `rebuild_model` also refuses weights that do not fit the model the config describes, and a config whose architecture is missing or not a recorded mapping (see `models.encoders.cnn1d`).

F1's artifacts add two optional mappings to `model_config`: `probability_calibration` (the frozen temperature, its D39 fit diagnostics and fitting provenance, or a recorded `fit_failed` status for an analysis model) and `provenance` (D48's source trajectory, subjects, digests, device and lineage). They stay optional so F0's smoke checkpoint still loads, which is why the schema version is still 1: a consumer that relies on them checks they are present rather than misreading an old file. `exclusive=True` refuses to replace an existing file, because a frozen artifact may already have been verified by digest somewhere else; `checkpoint_sha256` is that digest.

Checkpoint names carry their role:

| File | What it is | Who loads it |
|---|---|---|
| `refit.pt` | The model refit on all 32 development subjects | Every downstream consumer |
| `folds/fold{k}/best.pt` | The best epoch within one cross-validation fold | Analysis only |
| `development/fold{k}/fixed_budget.pt` | Retained snapshot of the accepted fold trajectory at the common budget (D48) | Development-time selection and calibration only |

`refit_checkpoint_path`, `fold_checkpoint_path` and `development_checkpoint_path` build these names, so the harness that writes them and the configs that name them cannot disagree. The refit has no validation set, so it cannot be a "best". F0 writes none of these; it saves `encoder.pt` in its own run directory.

**Connects to** `models.encoders.cnn1d` (describes, rebuilds) · `training.encoder`, `training.subject_cv` (save) · the robustness registry (loads) · **Decisions** D7, D28, D31, D33, D48, D85; the contract is a hard constraint in AGENTS.md

### `models.llm.masking`
Attention-mask helpers for the language-model input paths.

`pooling: last_token` means the **last non-padding token**, never simply the final position. `hidden_states[:, -1]` is correct only with left padding, and `attention_mask.sum(-1) - 1` only with right padding and no interior zeros. `rightmost_nonpadding_indices` is correct either way, and `mask_derived_position_ids` handles the second left-padding trap, where real tokens would otherwise sit at positions offset by the number of pads. Every row must contain at least one real token.

**Connects to** the language-arm adapters (not yet written)

---

## Training and evaluation

### `training.class_weights`
Class weights for the loss, from the training split only.

Rest is 50.2% of windows at stride 100, about 17.5 times the median gesture class, so a model that only ever predicted rest would score 50% accuracy. The imbalance is handled by weighting the loss by inverse class frequency (scaled so the weights average 1), never by resampling: windows already overlap 75%, and duplicating them would put near-identical copies in a batch. A class absent from the training split gets weight 0 — it has no frequency to invert, and 0 keeps the vector indexable by label without inventing a target the split cannot teach.

`class_weights_sha256` is the digest D85 makes both F1 development parents and the refit carry, so a language fit can recompute its own weights and verify them against the parent's. It hashes the float32 vector exactly as a trainer uses it — class order, little-endian, contiguous — behind a header naming the encoding `d11_class_weights_v1_shape_dtype_little_endian_float32`, its dtype and shape, the same pattern as `data.assignments`' recording digest. It refuses a float64 vector rather than casting it, since that would hash a value no trainer used. Equal weights do not prove equal fitting pools, so `training.subject_cv` records the training windows' identity in a separate digest.

**Connects to** `training.encoder`, `training.subject_cv` (call it after the split) · the language consumers (verify against the digest) · **Decisions** D11, D76, D85

### `training.encoder`
One encoder training run: a run configuration in, a recorded result out.

`EncoderRunConfig` takes explicit lists of training and held-out subjects rather than a config path, so it has no way to resolve "every subject". It refuses overlapping lists, and checks the loss, model type, normalization policy and device against what this code can do. It also refuses values that would quietly train something other than what was declared: a non-positive epoch count, batch size or learning rate, a stride longer than the window, dropout outside [0, 1), channel or class counts other than DB2's 12 and 18, and a subject listed twice. The last matters because recordings are loaded into a dictionary keyed by subject, which would collapse the repeat and train on fewer subjects than the config lists.

`train_encoder()` runs these steps in order:

| Step | What it guarantees |
|---|---|
| Check the working tree | Non-debug runs only: refuses uncommitted code before anything is loaded (see `utils.run_logger`) |
| Resolve the device | See `utils.device`; the device used is recorded |
| Load and index recordings | Training and held-out subjects are loaded separately; no window crosses a segment |
| Fit normalization and class weights | From the training subjects' windows only, after the split |
| Build the model | Through `build_model`, so the declared model type is honoured |
| Train | Class-weighted cross-entropy; Adam with PyTorch's default betas and epsilon and coupled L2 weight decay; a fixed number of epochs, no early stopping, no scheduler |
| Evaluate | Softmax probabilities on the held-out subjects, then the metric set |
| Record | The run directory with its provenance, an optional checkpoint, and an `EncoderRunResult` |

The result carries two audit facts so tests can check that the protocol held: the total and unique training-window counts (equal means nothing was resampled), and the number of windows crossing a segment boundary, recounted from the raw labels rather than taken from the windower. The recount uses a binary search over segment starts; the obvious nested loop gives the same answer about 95 times more slowly, roughly 3 s per 28-subject fold.

The logged training loss is the class-weighted objective over the whole epoch. PyTorch's batch mean divides by the batch's weight sum rather than its size, so each batch contributes its mean times that sum; averaging batch means equally would over-count rest-heavy batches, whose weight sum is small. The loss is only logged — epochs are selected on validation macro-F1.


**Connects to** every module above · `scripts/train_encoder.py` (drives it) · **Decisions** D7, D11, D18/D19 (fixed epochs), D24, D38 (these optimizer mechanics are the F1 reference)

### `training.selection`
F1's epoch selection and horizon rule, as pure functions of validation scores.

These are protocol constants, not knobs, so they are module constants (`SELECTION_WINDOW_EPOCHS = 10`, `INITIAL_HORIZON_EPOCHS = 30`, `ESCALATED_HORIZON_EPOCHS = 60`) and `build_f1_config` refuses a config that declares other values. Keeping them free of training makes every rule testable on a scripted curve.

| Function | Rule |
|---|---|
| `fold_score` | The raw fold score is the unweighted mean of the four validation subjects' macro-F1, never a pooled-window score (D21) |
| `smoothed_selection_scores`, `select_epoch` | The trailing 10-epoch mean; epochs 1–9 are ineligible; the earliest exact maximum wins (D18). Sums use `math.fsum`, so "exactly equal" means equal real sums rather than equal up to NumPy's addition order. The smoothed value is a selection diagnostic only; reportable metrics come from the selected checkpoint (D20) |
| `common_budget` | `ceil((e(4) + e(5)) / 2)` over the eight sorted fold epochs, in integer arithmetic (D18) |
| `decide_horizon` | Accept a budget below the horizon; escalate 30 to 60 once; block a budget equal to 60 (D19) |

**Connects to** `training.subject_cv` (applies them) · `scripts/train_encoder.py` (checks the config against them) · **Decisions** D18–D21

### `training.subject_cv`
F1's development harness: eight subject folds, selection, D48's development artifacts, the temperatures and the all-32 refit.

`run_subject_cross_validation` runs the protocol in order. It loads only the manifest's 32 development subjects, so a final-test subject cannot reach a fit or a selection; the test doubles are written for those 32 alone, which makes that a tested fact rather than a convention. For each fold, `fit_population_model` refits normalization and class weights on the fold's 28 training subjects through `fitting_statistics` — the one definition `training.f1_artifacts` also calls to recompute them at consumption — trains under D38, validates the four validation subjects after every epoch and snapshots every eligible epoch (10 onward) to disk with its SHA-256. After all eight folds, `training.selection` picks each fold's epoch and the common budget B; a budget at 30 discards the 30-epoch runs and reruns all eight from scratch at 60, and a second boundary hit raises `HorizonBlockedError` before any refit. Then, per fold, `best.pt` is the fold's own selected epoch and `fixed_budget.pt` its trajectory at B, each rebuilt from a snapshot whose digest is re-checked (`load_snapshot` refuses a missing or changed one rather than substituting another epoch). When a fold's own peak is B, the two files share a snapshot but keep distinct roles. Every artifact records the campaign's `common_budget`; `best.pt`'s `source_epoch` stays its individually selected epoch, never relabelled as B, so a consumer can tell the two apart without the run log.

Temperatures come from three separate sources and never mix:

| Temperature | Fitted on | Stored in |
|---|---|---|
| The refit's T | Pooled out-of-fold logits of the eight `fixed_budget.pt` models, repetitions 1–6 (D33/D39/D81) | `refit.pt`, with every key `evaluation.temperature_fit.required_provenance` lists |
| Each fold's own T | That `best.pt` model's four validation subjects | That `best.pt`, for extended analysis only |
| D40's diagnostic Ts | Three validation subjects at a time, scoring the fourth | The run log only, with the frozen `best.pt`'s digest and source snapshot as model identity, and per fit its scored and fitting subjects, fitting-logit digest and weights |

The refit T is fitted before the refit trains, so a D39 failure stops the run without spending the refit; an analysis model's failed T is recorded as `fit_failed` for that model alone, with no T=1 fallback. The refit then trains a fresh model on all 32 subjects for exactly B epochs with statistics refitted on the 32.

**Validation cannot change training (D48).** The training loader has its own generator; `predict` runs in eval mode under no-grad through a loader with its own generator, inside `preserved_rng`, which restores the CPU and device RNG states. A loader's iterator draws a base seed when created, so without its own generator a validation pass would advance the training stream. Two tests pin the consequence: a fit validated every epoch ends bitwise equal to an unvalidated one, and epoch 10 of an 11-epoch fit equals epoch 10 of a 12-epoch fit. Initialization is `torch.manual_seed(seed)` before each model build, and the training order generator takes the same seed; both are recorded as `training_rng_provenance`.

**Scope, then lineage (D47).** A run's scope is requested, never inferred. An earlier version made a run canonical whenever it was on CUDA from a clean tree, but neither fact proves the procedure is complete: a CUDA run today would have written the paths the arms read with no final-test stage behind them. `execution_lineage` now maps a `preliminary` scope to the preliminary lineage on every device, and refuses a `reportable` scope while `REPORTABLE_EXECUTION_BLOCKERS` lists anything — today, that the final-test stage is unwritten and unrehearsed. Each blocker is removed only in the change that implements it; after that, the reportable scope still requires CUDA and a clean tree. The clean-tree guard stays a separate argument, so a preliminary run from committed code keeps it. `lineage_root` sends preliminary artifacts under `<root>/preliminary/`, so no preliminary run can write a path a reportable consumer reads; L1's local smoke test reads `checkpoints/preliminary/baseline_cnn_subject_split/refit.pt` for that reason, and a test derives that path from these functions. Every artifact records its device and lineage. Every frozen artifact is written once: the run refuses to start if any target path exists, and each save is exclusive.

Snapshots are about 1.8 MB each for F1's model (443,538 parameters), so the 21 eligible epochs of eight 30-epoch folds hold about 300 MB on disk until B is known, about 730 MB at 60. They live in the run directory and are deleted once the artifacts are written. The final-test evaluation, D8's fold-model test analysis and D83's diagnostic are a separate CUDA-only stage, not implemented; the summary records `final_test_evaluated: false`.

**Connects to** `data.splits`, `data.windows`, `data.dataset`, `preprocessing.normalize`, `training.class_weights` (reads) · `training.selection`, `evaluation.calibration`, `evaluation.metrics` (applies) · `models.checkpoint` (writes) · `utils.run_logger` (records) · `scripts/train_encoder.py` (drives it) · `training.f1_artifacts` (checks what it wrote, using its roles, lineage roots and `fitting_statistics`) · **Decisions** D6, D16, D18–D21, D33, D38–D40, D47, D48, D81, D85

### `training.f1_artifacts`
The consumption side of F1's artifact boundary: a consumer calls it before it reads an F1 parent's weights.

Writing provenance enforces nothing; only a check at consumption does (handoff §6, items 2–3). Every development consumer — G2/G3/G4's fold encoders, the language arm's exact-B parents and refit — needs the same check, so it lives here once rather than as a partial copy in each runner. It is separate from `models.checkpoint` on purpose: the generic loader stays protocol-agnostic so F0's provenance-free smoke checkpoint still loads.

**The expectation comes from the consumer, never from the file.** A self-consistent file proves nothing: `best.pt` copied over `fixed_budget.pt`, or a refit offered as a fold encoder, is internally consistent. An `F1Expectation` states the role, the fold, the accepted run's B and horizon (from the consumer's own record of that run), the lineage it requires, the manifest, and optionally the SHA-256 it references the parent by (D76); it refuses an impossible combination, such as a B at or above its horizon, before any file is read. `verify_f1_artifact` then compares the file with it — role, fold, `common_budget`, `best.pt`'s own selected epoch, horizon, training and scoring subjects, no final-test subject, lineage (a canonical parent must record CUDA and a clean tree), source run, file digest — and requires every provenance key for the role, a superset of what the F1 config declares. A common-budget development model must carry no temperature, the same rule D84 applies to language snapshots. All mismatches are reported together.

**Recomputed, not trusted.** From the permitted fitting pool — the fold's 28 training subjects, or all 32 for the refit — it recomputes, through the trainer's own `fitting_statistics`, the D85 class-weight digest, the source-window identity and the normalization statistics. The window identity is separate because equal weights do not prove equal pools: on the synthetic test doubles every fold's class weights are identical, and only the window identity tells them apart. Normalization is compared within a relative 1e-6. The statistics are accumulated in float64 and stored as float32, so a recomputation on the same machine is bitwise equal, and a different summation order on another machine can flip at most one float32 rounding, about 1.2e-7. Measured on real subjects 1–3 (pool {1, 2} against {1, 3} and against {1, 2, 3}), swapping or adding one subject changed the most-affected channel's standard deviation by 24–61% and its mean, which sits near zero, by over 1000%, so the tolerance absorbs rounding and never a wrong pool. A tampered file with swapped-in statistics and a matching digest is caught by this recomputation, which is the case a digest check alone misses.

`verify_f1_run` checks all seventeen artifacts of one run and the links between them: B is recomputed from the eight selected epochs by D18's rule rather than read from any file, all seventeen must share one source run, and the refit's temperature must name exactly that run's eight development models by digest and source snapshot, in fold order. It fits each fold's statistics once for both of that fold's artifacts.

**Connects to** `training.subject_cv` (the roles, lineage roots and `fitting_statistics` it verifies against) · `models.checkpoint` (loads and rebuilds through it) · `training.class_weights` (the D85 digest) · `training.selection` (D18's budget rule) · future consumers: G2/G3/G4 and the language runners · **Decisions** D18, D19, D47, D48, D76, D84, D85

### `evaluation.metrics`
The metric functions every experiment reports.

Each function scores one set of windows and returns a number or a per-class array, and checks its inputs first: labels must be non-negative 1-D integer arrays of equal length, below `num_classes` wherever the metric takes it, probabilities finite rows in [0, 1] that sum to 1 within 1e-5 (which admits float32 softmax), confidences finite in [0, 1] with a true/false outcome per window, and nothing may be empty. Unchecked, NumPy either fails somewhere unrelated or returns a plausible number — a confidence of 1.7 gave an ECE of 0.33, and a label of -1 was counted as the last class, because negative indices wrap around. Grouping by subject or by run happens outside them, so no metric can quietly choose an aggregation the protocol leaves open. Temperature scaling is not part of them either: they score whatever probabilities they receive, raw or scaled, and how the temperature is fitted is set by the protocol (§3.4).

| Function | What it measures, and the choice behind it |
|---|---|
| `accuracy`, `confusion_matrix` | Standard. Accuracy is secondary because rest is half of all windows |
| `per_class_precision`, `per_class_recall`, `per_class_f1` | Per-gesture behaviour a single number hides (D8) |
| `macro_f1` | The headline classification metric. It averages over all 18 classes and scores an undefined class as 0, so a small calibration set cannot quietly change the denominator (D21, D34) |
| `average_confusion_matrices` | Averages the fold models' matrices, never sums them; every fold model scores the same test subjects (D8) |
| `brier_score` | Squared error of the whole probability vector, summed over classes and averaged over windows, not halved. Its range is 0–2, because a confident wrong answer both misses the true class and puts its mass on another (D39) |
| `calibration_bins`, `CalibrationBins` | The one place confidence bins are formed: boundaries from the confidences alone, a confidence on a boundary in the upper bin, identical confidences always in one bin, empty bins dropped. It records both the nominal and the occupied bin count, because a value computed over 2 occupied bins is not comparable with one over 10 unless the report says so. ECE and the overconfidence error are computed from this object, so a reported occupancy and the number beside it cannot come from two different binnings (D39) |
| `expected_calibration_error` | Top-label ECE over 10 nominal equal-mass bins, formed by `calibration_bins`; heavy ties — such as confidences that round to exactly 1.0 in float32 — leave fewer occupied. A perfectly calibrated model still scores above 0 at finite sample size; that threshold depends on the evaluation design and is never subtracted (D23, D39) |
| `overconfidence_error` | The part of the calibration gap where confidence exceeds accuracy, weighted by confidence. Read it only beside ECE: a uniformly under-confident model scores a perfect 0 (D25) |
| `mean_confidence_when_wrong` | Descriptive only; `None` rather than 0 when there are no errors (D25) |
| `per_subject` | Applies a metric to each subject separately. A pooled number can look fine while one subject sits at chance (D26). Subject IDs must be integers: results are keyed by integer ID, so fractional IDs 1.1 and 1.9 would share a key and one subject's result would overwrite the other's |
| `paired_bootstrap_interval` | A seeded, descriptive percentile bootstrap over subjects, never windows, for comparisons on the same subjects. Resampling windows gave an interval 20 times too narrow on identical data, because the overlapping windows of one contraction are one event (D26). D79 retires the zero-exclusion reportability gate; this helper is not the paired-t interval declared for primary mean differences, whose reporting path remains unimplemented |

**Connects to** `training.encoder` (calls it) · the future cross-validation and robustness evaluators

### `evaluation.calibration`
Temperature scaling: how the one scalar T is fitted, and when a fit must fail.

A temperature never changes an argmax, so it moves the calibration metrics and nothing else. `d33_fitting_weights` gives every fitting subject equal weight, every recording repetition equal weight within its subject and every window equal weight within its repetition, and gives repetition 0 (the terminal rest segment) weight 0 (D33, D81). `fit_temperature` minimizes the weighted NLL in beta = 1/T with SciPy's bounded scalar solver in float64, bounds [0.01, 100], `xatol` 1e-8 and 500 iterations (D39). The NLL is convex in beta, so the derivative at the bounds decides whether the optimum is interior: a non-negative derivative at the lower bound or a non-positive one at the upper raises `TemperatureFitError`, as do a solver that did not converge and a fit worse than T=1 by more than 1e-8. Nothing clips, widens the range or falls back to T=1. `TemperatureFit` carries every diagnostic D39 requires to be persisted. `cross_fitted_temperatures` is D40: for each subject, the T fitted on the others.

The tests recover known temperatures from labels sampled at T = 0.5, 1 and 2.5, check the analytic derivative against a finite difference, and pin the failure cases: an all-confident-and-correct model runs to the upper bound, and identical logits give an unidentified fit.

**Connects to** `training.subject_cv` (fits every F1 temperature) · `scripts/train_encoder.py` (checks the config's recipe against these constants) · **Decisions** D23, D33, D39, D40, D81

### `evaluation.robustness`
Decides which robustness evaluations may run.

A robustness evaluation pairs a perturbation (noise, channel dropout, amplitude scaling) with a trained model from the shared registry. These helpers find the reasons a pair must not run: an unresolved `pending_*` value anywhere in a registry entry, a source experiment whose `protocol.status` is not `ready`, duplicate target names, or a `shared_k0_with` reference to a target that does not exist.

`protocol_blockers` is a single function used by both checks — a perturbation config checks itself, and each target checks the experiment that produced it — so the two cannot apply different rules. Readiness has to be declared. A config with no `protocol:` block, a block with no status, or a target with no `source_config` is a blocker, because silence is indistinguishable from someone having written `ready`; `require_declared=False` exists only for configs that are never reportable. `status: ready` beside a non-empty `blocked_on` is a contradiction — one of the two was not updated — and raises. All of these files are read through `utils.yaml_io`, so a repeated key is an error here too.

**Connects to** `scripts/evaluate.py` (calls it) · `configs/experiment/robustness_targets.yaml` · **Decision** D2

### `utils.randomness`
Derives each protocol assignment's seed from what the assignment is, not from when it is drawn.

Every randomized object gets a fresh PCG64 generator keyed by a versioned canonical key of scope, purpose and named indices. Adding an unrelated draw or reordering work therefore cannot change a fold or a schedule, and nothing global is seeded or consumed. Training keeps its native PyTorch RNG and F1's direct seed (D38); the bootstrap keeps D26's. The key format, and which identities an experiment may put in a key, are specified in the experiments README §3.5; this module is their one implementation. A seed is provenance, not an archive, which is why `data.assignments` persists the realized assignment beside `randomization_provenance`.

**Connects to** `data.assignments` (its only caller so far) · future G1, G3/G4 and language runners · **Decision** D43

### `utils.yaml_io`
Strict YAML reading for every file whose contents decide what a run does.

PyYAML's safe loader keeps the *last* value when a mapping repeats a key, silently. In a config that means `epochs: 2` near the top and `epochs: 500` further down trains for 500 epochs while a reader of the top sees 2, and nothing records that two values existed. `load_yaml` refuses a repeated or unhashable key and names its line. It started as the split manifest's private loader and is shared so that configs and the manifest follow one rule. YAML merge keys (`<<:`) are not supported, and no project file uses them.

**Connects to** `data.splits` · `data.assignment_registry` · `evaluation.robustness` · `scripts/train_encoder.py` · `scripts/evaluate.py`

### `utils.device`
Chooses the device a run uses.

`training.device` may be `auto`, `cpu`, `mps` or `cuda`. `auto` picks CUDA, then Apple MPS, then CPU. Any other value pins the device, and a pinned device that is not available raises instead of falling back: a config asking for `cuda` that silently ran on the CPU would record one thing and do another. The device actually used goes into `run.json`, which keeps `auto` reproducible after the fact. Numbers from different devices are not compared as if they were one: seeding does not make MPS and CUDA numerically identical.

**Connects to** `training.encoder`

### `utils.run_logger`
Local, always-on JSON logging for every run, and the record that traces a run's numbers to their code.

Each run gets its own directory, with `run.json` and `metrics.jsonl` (one line per logged step). Logging cannot be switched off and does not depend on a network service. `run.json` records:

- the configuration and its `config_sha256`, a digest of canonical JSON, so key order does not change its identity;
- the git commit, whether the working tree was dirty, the changed paths, and `git_diff_sha256`, a digest of the tracked diff plus every untracked file — so two runs with the same commit and digest ran the same code;
- the device the run executed on — required, and never the unresolved `auto`;
- the caller's metadata (for the encoder script: mode, protocol state, tracking mode, declared metrics, and the config file's path and hash);
- every artifact the run wrote, with its SHA-256 and size;
- timestamps and the summary.

A commit alone is not enough: the first F0 run recorded a commit from before `train_encoder.py` existed, because the script was still uncommitted. A run that may be reported therefore refuses to start from a dirty tree: `ensure_clean_tree` raises `UncommittedCodeError` before any data is loaded, and the logger checks again when it is created. Untracked files are always listed, whatever the user's Git configuration says: under `status.showUntrackedFiles=no` a plain `git status` omits them, and an uncommitted new module would pass as a clean tree.

**Connects to** `training.encoder` · `tracking.*` in the configs

---

## Scripts

| Script | What it does |
|---|---|
| `prepare_dataset.py` | Converts raw `.mat` recordings to processed `.npz`, one per subject — about 75 MB each, 3 GB for all 40. It opens an existing file before skipping it and rebuilds one it cannot read, so an interrupted run is never skipped forever. `--subjects` needs at least one value; omit the flag for all 40. `make prepare-debug-subjects` prepares the three F0 needs |
| `train_encoder.py` | Turns a config into a run: F0's single split, or F1's development harness for a manifest config; see [Configuration](#configuration). `make debug`, `make train-baseline`. `--checkpoint-root` moves F1's artifacts for tests; it is not a protocol choice. `--reportable` requests F1's canonical scope explicitly and is refused today (see `training.subject_cv`); it is a flag rather than a config key because the same config describes both runs |
| `evaluate.py` | Robustness driver: works out which perturbation × target pairs may run. A run that evaluates nothing exits non-zero, since a fully blocked suite exiting 0 reads as a pass in any script that runs it; `--allow-empty` accepts that for inspecting an incremental suite. The evaluation itself is not written yet |
| `verify_environment.py` | `make verify`: interpreter, packages, MPS |
| `worktree.sh` | The one tool every agent uses for parallel work (AGENTS.md) |

`train_adapter.py` and `make_report.py` are named by Makefile targets but are not written yet.

---

## Tests

- **`tests/_synthetic.py`** builds recordings shaped like DB2 — alternating rest and gesture segments — for the tests. They are test doubles written only to temporary directories, never data, and nothing computed from them is reportable.
- **One unit-test file per module** (`test_normalize.py`, `test_models_cnn1d.py`, `test_checkpoint.py`, `test_metrics.py`, …). The metric tests compare against hand-computed values.
- **`test_f0_pipeline.py`** is the integration test: a whole run on synthetic subjects. Its leakage section asserts the protocol directly — the held-out subject never reaches training, statistics and class weights come from the training subjects only, no window crosses a segment, and nothing is resampled.
- **`test_train_encoder.py`** covers the config loader: mostly refusals, plus the tests that keep its declarations in step with the run settings and the metrics. For F1 it also asserts that every leaf of the real config sits in exactly one table, and that `main` sends an F1 config to the harness as a clean-tree run.
- **`test_subject_cv.py`** runs the whole F1 harness on DB2-shaped doubles (`db2_like_recording`: rest carries its repetition, one terminal repetition-0 rest, two deliberately indistinguishable gestures so temperatures stay interior). A scripted wrapper replaces only the validation curves, so B is known in advance while training, snapshots, artifacts and temperatures run for real. A fake trainer drives D19's escalate-then-block path without training. The same file tests `training.f1_artifacts`, the one exception to one test file per module: the verifier's subject is that run's artifacts, and the eight-fold run takes about 17 s, so a second file would train it twice. Those tests feed it the right artifact, then each substitution — own-best, all-32, wrong fold, budget, horizon, lineage, digest, run — and files tampered after writing. `test_selection.py` and `test_calibration.py` check the pure rules on scripted inputs.
- **`test_experiment_configs.py`, `test_evaluate.py`, `test_project_memory.py`** are structural: configs agree with the spec and with each other, robustness dispatch fails closed, and the project-memory tables stay well formed.
- **`test_markdown_style.py`** parses every repository Markdown document and rejects manual prose wrapping before handoff, while preserving code, math, HTML, tables, nested lists and intentional breaks. This prevents source line widths from becoming arbitrary visible newlines in a viewer.

Many tests assert a refusal. When adding one, check that it fails with the guard removed; a test that cannot fail protects nothing.
