# System architecture figure

[Open the editable, zoomable SVG][diagram].

The figure explains the F1-sized architecture and its downstream contracts. It is a reading aid,
not a second protocol authority: [the experiment specifications][protocol] and
their linked configs govern the experiments. No new model choices or experiment settings are
introduced by this drawing.

## How to read it

- **Rounded white boxes are processes:** reading data, applying a network, fitting a head or
  computing a score. **Teal stacks are products of computation:** recordings, feature maps,
  embeddings, soft-prefix vectors, logits and probabilities. Their labels give tensor dimensions.

- **Folded-corner grey documents are saved products**, not activation layers. In particular,
  `refit.pt` is loaded to reconstruct a model; a batch does not pass through a checkpoint file.
  The grey bands group persisted data or fitted parameters separately from the model flow.

- Solid arrows carry data or predictions. Dashed arrows mean loading a fitted artifact. Amber
  **trainable** and blue **frozen** badges describe parameter updates, not implementation status.
  Frozen means both `requires_grad=False` and evaluation mode.

- Read **A → B**, then the parallel arms **C / D**, then **E**. The CNN's second row runs
  right-to-left to preserve the operation order without compressing it into one unreadable line.
  The outer left path sends the foundation reference directly to evaluation. The outer right
  path sends normalized windows directly to L4, bypassing the CNN and projector.

- The tensor stacks are schematic representations, **not fabricated NinaPro recordings**; their
  plane counts and drawn widths are not to scale. Dimensions are carried by the labels.

## What the dimensions mean

`B` is a minibatch size, `N_s` the number of recorded samples for one subject, and `N_k` the number
of permitted support windows at a particular trial budget. `L` is the tokenized prompt length,
`D` the transformer's hidden width, and `Q` the number of metric columns in the schematic result
matrix. Neither `N_k` nor `L` is a fixed size.

### A — Data preparation

| Representation | Shape |
|----------------|-------|
| Recorded EMG | `N_s × 12` |
| Corrected labels / repetition vectors | `N_s` each |
| Window batch | `B × 12 × 400` |

At 2 kHz, 400 samples cover 200 ms; stride 100 gives a 50 ms hop.

Sources: [DB2 reader and processed format][db2-reader] · [F1 config][f1-config] ·
[window indexer][windows].

### B — Foundation encoder + classifier

| Operation | Output shape |
|-----------|--------------|
| Input window batch | `B × 12 × 400` |
| Conv1d, kernel 7 → batch norm → ReLU | `B × 128 × 400` |
| Max-pool 4 | `B × 128 × 100` |
| Conv1d, kernel 5 → batch norm → ReLU | `B × 256 × 100` |
| Max-pool 4 | `B × 256 × 25` |
| Conv1d, kernel 3 → batch norm → ReLU | `B × 256 × 25` |
| Adaptive average pool | `B × 256 × 1` |
| Flatten → dropout → linear | `H: B × 256` |
| Separate foundation head: dropout → linear | `B × 18` logits |

Sources: [CNN source][cnn-source] · [F1 config][f1-config] · D32.

### C — Generative arm · real-trial burden

- **Subject conditioning:** at `k > 0`, class-centred `N_k × 256` residuals become a mean
  256-vector, then a projected 16-vector. At `k = 0`, use the learned 16-vector.

- **Synthetic features:** noise has 32 dimensions. The decoder conditions on gesture class and a
  16-vector and produces 256-dimensional EMG features.

- **Replay bank:** 18 classes × 50 features gives one bank of shape `900 × 256`.

Sources: [G1 subject-embedding contract][g1-embedding] · [G1 config][g1-config] ·
[G3 config][g3-config] · D27.

### D — Language arm · frozen-backbone test

| Representation / transformation | Shape |
|---------------------------------|-------|
| EMG features entering the projector | `B × 256` |
| Projector output: eight soft-prefix vectors | `B × 8 × D` |
| Prefix + prompt embeddings | `B × (L+8) × D` |
| Last non-padding hidden state | `B × D` |
| Classifier MLP | `D → 256 → 18` |

With `D = 3072`, the eight soft-prefix vectors contain 24,576 scalars per example.

**L4 alternative input:** windows (`B × 12 × 400`) become engineered features (`B × 72`),
then text and a variable-length token sequence, before the frozen transformer and the same head
architecture. The 12 channels × 6 features are **not** 72 tokens.

The model wrapper must still read `D` from the loaded model configuration. The 3B model's
documented width of 3072 and depth of 28 layers explain the figure; they do not replace the
`auto_from_llm_config` settings with hard-coded values.

Sources: [Language-arm specification][language-spec] · [L2 config][l2-config] ·
[L4 config][l4-config] · [Meta's Llama architecture registry][llama-registry].

### E — Calibration, scoring + paired comparison

- Logits and both probability versions have shape `B × 18`.
- The schematic `8 × Q` matrix represents subject-level metrics for one subject-independent
  evaluation cell.

Sources: [Metric definitions][metrics-spec] · [metric implementation][metrics-source] · D26, D33.

## Scientific and implementation boundaries

- **F0 is runnable; F1 is not yet runnable.** Readers, indexing, normalization, the CNN / foundation
  head, metric primitives and local logging are implemented. F0's smoke config uses hidden width
  32, embedding width 64, stride 200 and dropout 0.1. The figure instead shows F1's widths 128 / 256,
  stride 100 and dropout 0.2. Its `refit.pt`, CV/refit runner, temperature fitting and downstream
  artifacts are specified outputs, not claims that those files have been produced.

- The artifact-load fork shows the **final pipeline**. Development-time selection and calibration
  use fold-local complement-28 encoder / generator artifacts that exclude the current four
  validation subjects, not the all-32 `refit.pt`. See D28 / D35 and the shared development-artifact
  contracts in the specs. Final-test predictions never feed parameter fitting or selection.

- **C expands the replay case**, not four independent training pipelines. The unchanged population
  reference skips adaptation; real-only adaptation omits the synthetic loss. The two replay
  strategies differ in population versus calibration-derived subject conditioning, with paired
  noise / labels and the other matched-treatment controls. At `k = 0`, no support mean is computed:
  both replay strategies use the population vector and share a head within a recorded repeat;
  real-only and unchanged population are also aliases. See D28 / D29.

- `k ∈ {0,1,2,5,10,17,20,34}` counts **complete active gesture trials per subject**, not windows or
  repetitions per class. `N_k` is the resulting permitted window count, not a declared `97 × k`
  tensor size. Rest ownership is still Pending. The fixed support/evaluation repetitions in C are
  for held-out-subject personalization; G1's training-only support rotations are described in the
  [generative specification][generative-spec].

- The generated representation is an **encoder-space feature**, not a 400-sample waveform, not the
  32-dimensional VAE noise and not a language token. The decoder is already fitted when adaptation
  starts. The figure does not invent a VAE depth, class-conditioning encoding, reconstruction
  likelihood or projector hidden width that the protocol has not fixed.

- **D expands L2.** L3 uses the same input path / backbone architecture with random frozen weights.
  L4 uses engineered text instead of a learned soft prefix, so its sequence length is its own
  tokenized text length, not `L+8`. The three arms share the classifier-head **architecture**, not
  the same fitted head weights. The LLM vocabulary head is unused in the v1 discriminative readout.
  The head also applies dropout 0.1; the figure omits that label to keep its main dimensional
  transformation legible. Mask-derived pooling and position IDs remain required.

- **E shows the subject-independent reporting contract**, not F2's deliberately leaky random-window
  demonstration. For F1, a scalar temperature is fitted on fixed-budget development OOF predictions
  and stored in checkpoint metadata. G4 instead needs a frozen strategy × `k` temperature artifact.
  Raw and temperature-scaled ECE / Brier are separate outputs; scaling changes probabilities, not
  the predicted class. G4's within-subject repeat and headline aggregation choices remain Pending:
  the schematic `8 × Q` matrix does not settle them. F0 currently emits raw probabilities only.

- Robustness is an **input intervention before inference**, not a final neural-network layer.
  Operational perturbation definitions and the evaluator remain pending.

## Visual references

The redesign uses the representation-first clarity of the author's diagrams: visible intermediate
data products, explicit dimensions, grouped network stages and selective color rather than a chain
of identical generic boxes. It is newly drawn SVG artwork, not a reproduction of a published figure.

- [Recognizing Complex Gestures on Minimalistic Knitted Sensors][gesture-paper], Figures 6–7:
  processing products and the dimensioned CNN–LSTM architecture.

- [Knitted Sensors: Designs and Novel Approaches for Real-Time, Real-World Sensing][ubicomp-paper],
  Figure 3: separated acquisition, feature construction and recognition stages.

- [Toward Accurate Sensing with Knitted Fabric: Applications and Technical Considerations][eics-paper],
  Figure 12: staged system integration and distinct products at each boundary.

[diagram]: system-layers.svg
[protocol]: ../experiments/README.md
[db2-reader]: ../../src/temporallens/data/ninapro.py
[f1-config]: ../../configs/experiment/foundation/baseline_cnn_subject_split.yaml
[windows]: ../../src/temporallens/data/windows.py
[cnn-source]: ../../src/temporallens/models/encoders/cnn1d.py
[g1-embedding]: ../experiments/generative-arm.md#the-subject-embedding-contract-d27
[g1-config]: ../../configs/experiment/generation/gen_vae_train.yaml
[g3-config]: ../../configs/experiment/generation/gen_personalization_efficiency.yaml
[language-spec]: ../experiments/language-arm.md
[l2-config]: ../../configs/experiment/language/adapter_llama3b_subject_split.yaml
[l4-config]: ../../configs/experiment/language/adapter_text_summary_only.yaml
[llama-registry]: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
[metrics-spec]: ../experiments/README.md#34-metrics
[metrics-source]: ../../src/temporallens/evaluation/metrics.py
[generative-spec]: ../experiments/generative-arm.md
[gesture-paper]: https://www.denisa-qori-mcdonald.com/wp-content/uploads/2026/04/arXiv_2024_knitted_sensors.pdf
[ubicomp-paper]: https://www.denisa-qori-mcdonald.com/wp-content/uploads/2026/04/UbiComp_2020_knitted_sensors-1.pdf
[eics-paper]: https://www.denisa-qori-mcdonald.com/wp-content/uploads/2026/04/EICS_2020_knitted_sensors-1.pdf
