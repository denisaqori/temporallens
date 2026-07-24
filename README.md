# TemporalLens

**Reducing cross-subject calibration burden in EMG decoding via generative augmentation — and testing whether language-model conditioning adds anything beyond the encoder.**

<!-- Fill the arXiv ID and Spaces URL once they exist, then uncomment. -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
<!-- [![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live%20demo-blue)](https://huggingface.co/spaces/denisaqori/temporallens) -->
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg?logo=pytorch&logoColor=white)
![Code License](https://img.shields.io/badge/code-Apache%202.0-green.svg)
![Docs License](https://img.shields.io/badge/paper%20%26%20figures-CC%20BY--NC--SA%204.0-lightgrey.svg)

Wearable surface-electromyography (sEMG) interfaces promise device-free input, but a central obstacle to deployment is **calibration burden**: generalized decoders work across users, yet accuracy improves once the model is adapted to an individual. TemporalLens asks a practical question — *how much real per-subject calibration data does a new, unseen user actually need, and can generative augmentation reduce it?* — and evaluates it the way human-signal systems must be evaluated: on held-out subjects, under perturbation, with calibration analysis and leakage-controlled protocols.

<p align="center">
  <img src="results/figures/personalization_efficiency.png" width="620" alt="Decoding accuracy vs. number of real calibration samples per held-out subject, with and without synthetic augmentation.">
  <br>
  <em>Headline result: decoding accuracy for a new, unseen subject as a function of the number of real
  calibration samples, with and without synthetic augmentation. (Figure populates after Phase&nbsp;2.)</em>
</p>

> **Status.** Actively under development. The repository is organized in phases (see [Roadmap](#roadmap)); results tables and figures are populated as each phase completes. Sections marked _(pending)_ are scaffolded but not yet filled.

---

## Contents

- [Key contributions](#key-contributions)
- [The question](#the-question)
- [Method](#method)
- [Results](#results)
- [Evaluation protocol](#evaluation-protocol)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Data](#data)
- [Related work and positioning](#related-work-and-positioning)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Key contributions

1. **A personalization-efficiency curve for cross-subject EMG decoding.** For each held-out subject, we measure decoding accuracy as a function of the number of real calibration samples available (0, 5, 10, 20, 50), *with and without* synthetic samples from a conditional generative model — quantifying how much real calibration a new user actually needs, and how much of it augmentation can replace.
2. **An honest test of language-model conditioning.** We map temporal signal windows into the embedding space of a **frozen** large language model through a learned soft-prefix adapter, and ask whether it improves decoding, calibration, or failure reporting beyond a well-trained encoder — isolating the effect with a **matched-size random-initialized transformer** and a **text-summary-only** baseline, so any effect is attributable to language pretraining rather than to model size or prompt engineering.
3. **Deployment-aware evaluation throughout.** Subject-independent splits, test-time robustness perturbations, expected calibration error, per-subject variance, and leakage-controlled generative protocols — because for wearable interfaces, behavior under subject shift and perturbation matters as much as aggregate accuracy.

---

## The question

Generalized sEMG decoders can work without per-person calibration, but a small amount of individual data measurably improves them — recent work reports handwriting-recognition gains of up to ~16% from limited personalization ([Sussillo, Kaifosh & Reardon, *Nature* 2025](#references)). That raises a concrete, under-studied question:

> **How many real calibration samples does a new subject need to reach a target accuracy — and can a generative model supply that personalization with fewer real samples?**

TemporalLens answers this as an explicit curve rather than a single number, and treats "does an LLM help?" as a separate, deliberately skeptical question rather than an assumption.

---

## Method

TemporalLens has two arms that share a temporal encoder. The **generative arm** (headline) studies calibration efficiency; the **language arm** (secondary) tests whether LLM conditioning adds value.

```
multichannel EMG window
  ├─ preprocessing / normalization / windowing
  └─ temporal encoder (1-D CNN)
        ├─ classifier head ............................ baseline decoder
        ├─ projection MLP → soft-prefix embeddings → frozen LLM
        │                                              → class prediction + structured report
        └─ conditional VAE (latent space) ............. synthetic windows,
              conditioned on gesture class + a calibration-derived subject embedding
              → augmented calibration set → personalization-efficiency evaluation
```

- **Temporal encoder — 1-D CNN.** Suited to local temporal structure, easy to inspect, and a strong baseline. (A patch-based transformer encoder is a planned ablation, not the default.)
- **Projection adapter.** A small MLP maps encoder outputs into the LLM embedding dimension; the result is treated as **soft-prefix / pseudo-token embeddings** prepended to a text prompt — *not* discrete tokens. Prefix concatenation is used for transparency over cross-attention.
- **Frozen LLM.** The language model's weights are frozen; only the adapter is trained. A small model (e.g. Llama 3.2 1B) is used for local iteration; a larger frozen model for final numbers.
- **Conditional generator.** A conditional VAE operating in the encoder's latent space, conditioned on gesture class and a learned subject/calibration embedding. Synthetic quality is validated before use (see [Evaluation protocol](#evaluation-protocol)). A conditional diffusion variant is a planned extension.

---

## Results

_(Populated as phases complete. Numbers below are placeholders.)_

### Personalization efficiency _(pending — Phase 2)_

Real calibration only vs. real + synthetic, across sample counts, averaged over held-out subjects:

| Real calibration samples | 0 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Accuracy — real only        | – | – | – | – | – |
| Accuracy — real + synthetic | – | – | – | – | – |

**Headline number _(pending)_:** synthetic augmentation reaches the target accuracy with approximately **[N]** fewer real calibration samples per subject.

### Subject-independent decoding and ablations _(pending — Phases 1–2)_

| Configuration | Held-out-subject accuracy | Macro F1 | ECE |
|---|---|---|---|
| Encoder + classifier head                                   | – | – | – |
| Encoder + projection + **frozen LLM**                       | – | – | – |
| Encoder + projection + **random-init transformer** (matched)| – | – | – |
| Encoder + **text-summary-only** LLM                         | – | – | – |

> The random-init-transformer and text-summary-only rows exist to answer the first question any reviewer will ask: is any LLM benefit due to *language pretraining*, or merely to having a large transformer / a hand-written prompt of numbers?

### Robustness _(pending — Phase 1)_

Accuracy under test-time perturbation (clean → perturbed), zero-shot:

| Perturbation | Clean | Mild | Moderate | Severe |
|---|---|---|---|---|
| Additive sensor noise | – | – | – | – |
| Channel dropout       | – | – | – | – |
| Amplitude scaling     | – | – | – | – |

---

## Evaluation protocol

Evaluation is the center of this project. All headline results use a **subject-independent** protocol.

> **Full specification:** [`docs/experiments/`](docs/experiments/) is the authority on what every experiment and ablation measures, why it exists, and the failure modes that are easy to miss — the [shared protocol and vocabulary](docs/experiments/README.md), the [generative arm](docs/experiments/generative-arm.md), and the [language arm](docs/experiments/language-arm.md). Each ablation there maps to exactly one config file.

- **Splits.** (i) random-window split — an optimistic baseline that shows how easily leakage inflates results; (ii) **subject-independent split** — train on a set of subjects, test on unseen subjects (the main result); (iii) grouped/leave-subjects-out folds — for per-subject variance.
- **Robustness.** Test-time perturbations applied zero-shot: additive sensor noise, channel dropout (electrode failure / missing channels), amplitude scaling (strength / impedance / placement).
- **Calibration.** Expected calibration error, confidence under perturbation, and overconfidence on incorrect predictions — the LLM arm's value, if any, may lie here rather than in raw accuracy.
- **Generative leakage control _(non-negotiable)_.** The generator never trains on held-out-subject test windows. In the zero-calibration setting, generation is class-conditioned or uses a population/default subject embedding; in the *k*-shot settings, any subject-specific conditioning is derived **only** from the *k* real calibration samples permitted for that subject. A positive result traceable to leakage is treated as no result.
- **Synthetic-quality validation.** A classifier two-sample test (can a discriminator separate real from synthetic?) near chance indicates realistic synthesis — so a personalization gain cannot be attributed to a degenerate generator.

---

## Repository structure

```
temporallens/
├── docs/experiments/         # AUTHORITATIVE spec: what each ablation measures and why
├── configs/experiment/       # one YAML per ablation, mapped 1:1 to docs/experiments/
│   ├── foundation/           #   F-series: encoder baseline, leakage demo, robustness
│   ├── generation/           #   G-series: conditional VAE, synthetic quality, the headline curve
│   └── language/             #   L-series: soft-prefix adapter + the two rigor baselines
├── src/temporallens/
│   ├── data/                 # NinaPro DB2 loading and splits
│   ├── preprocessing/        # windowing, normalization, perturbations
│   ├── models/
│   │   ├── encoders/         # 1-D CNN (+ planned PatchTST)
│   │   ├── projectors/       # trainable MLP → soft-prefix embeddings
│   │   ├── llm/              # frozen LLM wrapper, soft-prefix injection, mock LLM
│   │   ├── generative/       # conditional VAE (+ planned diffusion)
│   │   └── heads/            # classifier heads
│   ├── training/             # encoder / adapter / generative training loops
│   ├── evaluation/           # subject splits, perturbations, calibration,
│   │                         #   personalization_efficiency, synthetic_quality, ablations
│   ├── reporting/            # structured scorecards and templated reports
│   └── utils/                # device selection, run logging
├── scripts/                  # prepare_dataset, train_*, evaluate, make_report
├── notebooks/                # exploration + per-phase result notebooks
├── results/                  # runs/ metrics/ figures/  (JSON logs + plots)
├── tests/
├── requirements.txt          # local (Apple MPS)      — entry point; deps live in pyproject.toml
├── requirements-cloud.txt    # cloud (CUDA)           — adds PEFT, optional quantization
└── uv.lock                   # exact dependency resolution (tracked; `make setup` installs from it)
```

---

## Installation

Developed on Apple Silicon (MPS) for local work, with a CUDA GPU for the language-model and generative runs. Environments are managed with [`uv`](https://github.com/astral-sh/uv).

**Local (macOS, Apple Silicon):**

```bash
git clone https://github.com/denisaqori/temporallens.git
cd temporallens
make setup               # uv sync --python 3.11 --extra dev — installs the exact uv.lock resolution
source .venv/bin/activate
make verify              # Python version, dependency imports, MPS availability, selected device
```

**Cloud (CUDA GPU) — for the frozen-LLM and generative runs:**

```bash
# install CUDA PyTorch via the official selector for the machine's CUDA version first, then:
make setup-cloud         # uv pip install, NOT uv sync — leaves the host's CUDA PyTorch in place
huggingface-cli login    # frozen LLM weights are gated; request access in advance
```

A frozen 3B model runs comfortably on a 24 GB GPU in `bfloat16`; quantization is optional.

---

## Reproducing the results

Common workflows are wrapped in a `Makefile`; each script is config-driven and runs locally on a tiny config before scaling up. Configs live under `configs/experiment/` in three groups — `foundation/`, `generation/`, `language/` — and each one is specified in [`docs/experiments/`](docs/experiments/).

```bash
make debug              # validate the full pipeline on a 3-subject tiny config
make train-baseline     # 1-D CNN, subject-independent split
make debug-adapter      # adapter shape-check against a mock LLM
make smoke-1b           # real Llama-1B adapter smoke test (local)
make eval-noise         # a robustness perturbation evaluation
make report             # structured scorecard for a run
```

Long training runs on a rented GPU should be launched inside `tmux` so they survive disconnection; checkpoint frequently and sync artifacts back.

---

## Data

TemporalLens uses **[NinaPro DB2](http://ninapro.hevs.ch/)** (Atzori et al., *Scientific Data* 2014): surface EMG (12 channels, 2 kHz) with inertial, kinematic, and force data from 40 intact subjects. The first version uses **Exercise B** (17 hand/wrist movements + rest = 18 classes) under a subject-independent split; the full 49-movement set is a planned extension.

NinaPro data must be downloaded from the official site under its terms of use and is **not** redistributed here. Place the raw `.mat` files under `data/raw/` and run `scripts/prepare_dataset.py` to produce windowed, normalized tensors under `data/processed/`.

---

## Related work and positioning

**Cross-modal sensor–language and adapter models.** SensorLM and SensorLLM align wearable sensor data with text; LLaVA connects a vision encoder to an LLM via a projector (BLIP-2's Q-Former and Flamingo's gated cross-attention are the main alternative connectors). TemporalLens borrows the projector idea but replaces the vision encoder with a temporal encoder, and distinguishes itself through subject-independent evaluation, calibration, robustness, and the baselines that isolate whether language pretraining matters at all.

**Cross-modal EMG.** [EmBridge (Cui et al., ICLR 2026)](#references) and its sibling CPEP align EMG with **hand pose** to enable zero-shot recognition of *unseen gestures*, using a Q-Former and contrastive alignment. TemporalLens differs on axis and target: it aligns EMG with a **frozen language model** and studies *unseen subjects* and calibration efficiency rather than unseen gestures. That these methods succeed by aligning EMG with a rich, motor-aligned modality (pose) is itself a reason to be skeptical that a sparser modality (language) will help — precisely the question the language arm tests.

**EMG cross-subject generalization and calibration.** A large literature reduces calibration cost through *adaptation* — transfer learning and domain adaptation for sEMG, few-shot and meta-learning calibration, and test-time adaptation for session/subject drift; EMGBench benchmarks out-of-distribution generalization and adaptation directly. TemporalLens instead asks whether generative *augmentation* of the calibration set achieves the same end, which makes it complementary to — and directly comparable against — a simple adaptation baseline.

**Time-series and LLMs.** GPT4TS ("One Fits All") freezes a pretrained LM for general time-series tasks and is the conceptually closest of this line; Time-LLM, Chronos, MOMENT, and UniTS repurpose or pretrain models for forecasting. TemporalLens is neither a forecaster nor a foundation model — it is a smaller, transparent adapter-and-evaluation study on human-generated gesture signals.

**Strategic anchor.** Meta's generalized sEMG neuromotor interface ([Sussillo, Kaifosh & Reardon, *Nature* 2025](#references)) demonstrates calibration-free decoding that nonetheless improves with limited personalization — the tradeoff TemporalLens quantifies on public data. Meta's released corpora (emg2qwerty, emg2pose, and the *Nature* dataset) are natural targets for a follow-up beyond NinaPro.

### References

- CTRL-labs at Reality Labs, D. Sussillo, P. Kaifosh, T. Reardon. *A generic non-invasive neuromotor interface for human–computer interaction.* **Nature** (2025). doi:10.1038/s41586-025-09255-w
- W. Cui, C. M. Sandino, H. Pouransari, R. Liu, J. Minxha, E. L. Zippi, E. Azemi, B. Mahasseni. *EMBridge: Enhancing Gesture Generalization from EMG Signals through Cross-Modal Representation Learning.* **ICLR** (2026). OpenReview: `LqrWNdceum`
- M. Atzori et al. *Electromyography data for non-invasive naturally-controlled robotic hand prostheses.* **Scientific Data** (2014). [NinaPro]
- H. Liu, C. Li, Q. Wu, Y. J. Lee. *Visual Instruction Tuning (LLaVA).* **NeurIPS** (2023).
- T. Zhou et al. *One Fits All: Power General Time Series Analysis by Pretrained LM (GPT4TS).* **NeurIPS** (2023).

_A full bibliography accompanies the paper._

---

## Limitations

- **Cross-subject EMG is hard.** Held-out-subject accuracy is substantially lower and more variable than intra-subject accuracy; results should be read as a study of *relative* effects (augmentation, LLM conditioning) under a difficult, realistic protocol, not as state-of-the-art absolute numbers.
- **The language arm may show little or no benefit.** This is an explicit hypothesis to test, not a failure mode; a clean negative, properly isolated by the baselines, is a valid finding.
- **Single dataset, reduced class set (v1).** One dataset with a rigorous protocol is prioritized over several shallow ones; NinaPro DB2 Exercise B is a starting point, with the full class set and additional datasets planned.
- **Not a foundation model or a production decoder.** TemporalLens is a transparent, reproducible study, not a system trained on large-scale proprietary data.

---

## Roadmap

- [ ] Data pipeline: NinaPro DB2 Exercise B, subject-independent splits
- [ ] **Milestone 0** — 1-D CNN baseline, robustness perturbations, calibration analysis
- [ ] **Phase 1** — soft-prefix adapter to a frozen LLM; encoder-only vs. encoder+LLM; random-init and text-only baselines
- [ ] **Phase 2** — conditional VAE; synthetic-quality validation; personalization-efficiency curve; calibration-efficiency ablation
- [ ] Adaptation baseline (few-shot / test-time) on the personalization-efficiency curve
- [ ] Interactive demo (Gradio / 🤗 Spaces) and preprint

**Deferred (deliberately out of scope for now):** LoRA-adapted LLM, patch-transformer encoder ablation, vector search / tool-calling agents, full 49-movement class set, additional datasets.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{qorimcdonald2026temporallens,
  title  = {TemporalLens: Reducing Cross-Subject Calibration Burden in EMG
            Decoding via Generative Augmentation},
  author = {Qori McDonald, Denisa},
  year   = {2026},
  note   = {Preprint. arXiv:XXXX.XXXXX},
  url    = {https://github.com/denisaqori/temporallens}
}
```

---

## License

This repository uses a split license, applying the right instrument to each kind of content:

| Content | License |
|---|---|
| **Source code** | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) |
| **Paper, figures, documentation** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — see [`LICENSE-DOCS`](LICENSE-DOCS) |
| **NinaPro data** | Not redistributed here; subject to its own [terms of use](http://ninapro.hevs.ch/) |

In short: the **code** is permissively licensed under Apache 2.0 — free to use, modify, and redistribute (including commercially), with attribution and an explicit patent grant. The **paper and figures** are Creative Commons non-commercial with attribution and share-alike, keeping the intellectual content protected from commercial reuse.

---

## Acknowledgements

Built on the open NinaPro database and the open-source PyTorch and Hugging Face ecosystems.

**Contact:** Denisa Qori McDonald · [denisa-qori-mcdonald.com](https://www.denisa-qori-mcdonald.com/)
