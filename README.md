# LSTM-DynAE

Implementation accompanying the manuscript:

**Sequential Deep Embedded Clustering with Dynamic Autoencoder**

Hasna AlSaeed, Riadh Ksantini, and Faisal AlKhateeb

*Multimedia Tools and Applications*

This repository provides the implementation of LSTM-DynAE, including the LSTM
Autoencoder, adversarial interpolation pretraining, the dynamic clustering
objective, preprocessing utilities, experimental configuration, evaluation
metrics, and statistical significance analysis.

## Table of Contents

1. [Overview](#1-overview)
2. [Method](#2-method)
3. [Repository Structure](#3-repository-structure)
4. [Requirements](#4-requirements)
5. [Installation](#5-installation)
6. [Datasets](#6-datasets)
7. [Configuration](#7-configuration)
8. [Running Pretraining](#8-running-pretraining)
9. [Running Clustering](#9-running-clustering)
10. [Evaluation](#10-evaluation)
11. [Statistical Significance](#11-statistical-significance)
12. [Reproducibility Notes](#12-reproducibility-notes)
13. [Implementation Audit](#13-implementation-audit)
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Overview

Deep Embedded Clustering (DEC) methods jointly learn a latent representation and a
cluster assignment, but are prone to two competing failure modes: **Feature-Randomness
(FR)**, where the encoder chases noisy pseudo-labels, and **Feature-Drift (FD)**, where
the latent space drifts away from cluster-discriminative structure. **DynAE**
addresses this trade-off for static, feed-forward autoencoders via a
*dynamic* loss that smoothly shifts from reconstruction to centroid-oriented
construction. **LSTM-DynAE** extends this idea to *sequential* data by replacing the
feed-forward autoencoder with an LSTM Autoencoder, pretrained with an Adversarially
Constrained Autoencoder Interpolation (ACAI) regularizer.

## 2. Method

LSTM-DynAE has two phases (Fig. 1 of the manuscript):

- **Phase I -- Pretraining.** An LSTM Autoencoder (encoder: LSTM(64) -> LSTM(128)
  bottleneck; decoder: LSTM(128) -> LSTM(64), mirrored) is pretrained with an
  ACAI-regularized reconstruction objective (Eq. 4-7): a critic network learns to
  predict the latent-interpolation coefficient, while the autoencoder learns to fool
  it, yielding a smoother, more cluster-friendly latent space.
- **Phase II -- Clustering.** Cluster centroids are initialized with K-Means in the
  pretrained latent space (Section 3.3.1). A *dynamic* loss `L = L1 + L2` (Eq. 8-14)
  then jointly fine-tunes the **entire encoder-decoder** (it is explicitly **not**
  frozen) plus the centroids: conflicted (low-confidence) samples are still
  reconstructed, while unconflicted samples are pulled toward a 1-NN-substituted
  cluster centroid. Training stops when the dataset-level fraction of conflicted
  samples `tau_p` (Eq. 11, evaluated over all N samples) drops below 1%, or after
  `MaxItr` iterations.

See `lstm_dynae/` docstrings for the exact equation-by-equation mapping; every
non-trivial function cites the manuscript equation it implements.

## 3. Repository Structure

```
LSTM_DynAE/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── lstm_dynae.yaml
├── data/
│   └── README.md
├── lstm_dynae/
│   ├── __init__.py
│   ├── model.py            # top-level LSTMDynAE orchestrator (Algorithm 1)
│   ├── autoencoder.py       # LSTM encoder/decoder (Section 3.1)
│   ├── critic.py             # ACAI critic network
│   ├── acai.py                 # Eq. 4-7 (pretraining loss)
│   ├── dynamic_loss.py          # Eq. 8-14 (clustering loss), global tau_p
│   ├── clustering.py             # Eq. 1-2, K-means init
│   ├── preprocessing.py           # normalization, windowing, augmentation
│   ├── datasets.py                 # dataset registry + loaders (Table 1)
│   ├── metrics.py                   # ACC (Eq.15), NMI (Eq.16), FR/FD (Eq.17-18)
│   └── utils.py                      # seeding, logging, checkpoints
├── scripts/
│   ├── pretrain.py             # Phase I CLI
│   ├── train_clustering.py      # Phase II CLI
│   ├── run_experiment.py         # Phase I + II + eval for one dataset
│   ├── run_all_datasets.py        # runs all five datasets
│   ├── evaluate.py                 # standalone evaluation
│   └── statistical_test.py          # Wilcoxon signed-rank test
├── results/
│   └── clustering_results.csv    # Table 3/4 values, verbatim from the manuscript
└── notebooks/
    └── LSTM_DynAE_Reproduction.ipynb
```

## 4. Requirements

```
tensorflow==2.20.0
numpy
pandas
scikit-learn
scipy
matplotlib
pyyaml
```

(see `requirements.txt`; every package is actually imported somewhere in this repo).

## 5. Installation

```bash
git clone <this-repository-url>
cd LSTM_DynAE
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Tested layout targets both a local Python 3 environment and Google Colab (no
hidden/absolute local paths are used; all scripts take `--data-dir` /
`--checkpoint-dir` arguments).

## 6. Datasets

Table 1 of the manuscript (verbatim):

| Dataset | Samples | Dimensions | Classes |
|---|---|---|---|
| PenDigits | 7494 | 2 | 10 |
| Handwriting | 1000 | 3 | 26 |
| RacketSports | 303 | 6 | 4 |
| EEG Eye State | 14980 | 14 | 2 |
| Ozone Level Detection | 2536 | 24 | 2 |

**This repository does not redistribute any dataset.** See `data/README.md` for
download pointers and notes on exact source files -- in particular, EEG Eye State
and Ozone Level Detection are natively continuous streams / tabular records, not
pre-segmented sequence collections; the windowing scheme used to turn them into
`(N, T, D)` sequences is a configurable implementation choice not explicitly
specified in the manuscript (see Section 7).

Ground-truth labels are used **only** for external evaluation (`lstm_dynae/metrics.py`)
and are never passed into any training step: `lstm_dynae/model.py`'s
`pretrain_step`, `clustering_train_step`, and `compute_global_tau_p` all take only
`x` (or embeddings derived from `x`) as input, never `y`. Labels are read solely by
`scripts/evaluate.py` and by the periodic ACC/NMI logging in
`scripts/train_clustering.py`, both of which call them strictly after a forward
pass, not during gradient computation.

## 7. Configuration

All manuscript hyperparameters are exposed in `configs/lstm_dynae.yaml`, annotated
`[A]`/`[B]`/`[C]`/`[D]` per the classification scheme in the Implementation Audit
(Section 13). Edit this file (not the code) to change any hyperparameter, including:

- **Normalization**: default per-feature min-max scaling to [0, 1] -- a necessary
  implementation choice, not specified by the manuscript.
- **Sequence framing / windowing** for EEG Eye State / Ozone Level Detection --
  configurable implementation choice not explicitly specified in the manuscript;
  window length and stride are exposed as config parameters with no
  manuscript-derived default.
- **Data augmentation** (rotation + rescaling), applied in both the pretraining
  (Fig. 2) and clustering (Fig. 3) phases, per the manuscript's qualitative
  description ("This technique involves rotating and rescaling data..."). The
  magnitude constants are configurable placeholders, not manuscript-reported values.
- **Critic architecture** (`critic:` section) -- configurable implementation choice;
  the manuscript specifies the critic's function/objective but not its exact
  architecture (see Section 13, item 11).

## 8. Running Pretraining

```bash
python scripts/pretrain.py --config configs/lstm_dynae.yaml --dataset PenDigits
```

Runs `1.3 x 10^5` ACAI-regularized pretraining iterations (Eq. 4-7), Adam optimizer
(lr=1e-4, beta_1=0.9, beta_2=0.999 -- Section 4.3), and saves
`checkpoints/PenDigits/pretrained_autoencoder.weights.h5` /
`pretrained_critic.weights.h5`, plus a per-iteration CSV log.

## 9. Running Clustering

```bash
python scripts/train_clustering.py --config configs/lstm_dynae.yaml \
    --dataset PenDigits --max-iterations <N>
```

Loads the pretrained autoencoder, initializes centroids with K-Means (Section 3.3.1),
then jointly fine-tunes the **full encoder-decoder + centroids** under the dynamic
loss `L = L1 + L2` (Eq. 8-14) with mini-batch SGD+momentum (lr=0.001, momentum=0.9,
batch=256).

**Stopping criterion.** The manuscript defines `tau_p = |S_bar| / N` (Eq. 11) over
the full dataset (N = total sample count), not a mini-batch. This script computes
that dataset-level value periodically (`clustering.tau_p_check_interval` in the
config; default: once per epoch-equivalent) by encoding the entire dataset and
counting conflicted samples, and stops when `global_tau_p < tau_stop` (1%, Section
4.3) or after `max_iterations` steps. Per-mini-batch conflicted fractions
(`batch_tau_p`) are logged separately for fast diagnostics only and never drive the
stopping decision.

**Maximum clustering iterations are configurable; the exact numerical value is not
explicitly reported in the manuscript.** `clustering.max_iterations` has no
default: set it in `configs/lstm_dynae.yaml`, or pass `--max-iterations <N>` as
shown above. The script raises a clear error if neither is provided, rather than
silently choosing a value.

## 10. Evaluation

```bash
python scripts/evaluate.py --config configs/lstm_dynae.yaml --dataset PenDigits
```

Reports **Accuracy (Eq. 15)** via Hungarian-algorithm optimal label matching and
**NMI (Eq. 16)** with arithmetic-mean normalization, plus execution time. Evaluation
code (`lstm_dynae/metrics.py`) is fully decoupled from training code.

## 11. Statistical Significance

```bash
python scripts/statistical_test.py --results results/clustering_results.csv
```

Reproduces the manuscript's exact one-sided Wilcoxon signed-rank test (`LSTM-DynAE >
baseline`, separately for ACC and NMI, five paired dataset-level observations per
comparison, alpha=0.05). The script **recomputes** `W+`, `W-`, and the exact p-value
from whatever is in the CSV -- it does not hard-code `p=0.03125`; that value only
appears if the input data genuinely shows LSTM-DynAE winning on all five datasets
for a given (baseline, metric) pair, as it currently does for every row of
`results/clustering_results.csv` (values transcribed and verified against the
manuscript's reported table).

## 12. Reproducibility Notes

This repository implements the LSTM-DynAE architecture, mathematical objectives,
training phases, evaluation procedures, and hyperparameters explicitly reported in
the manuscript. Software-level implementation choices that are not explicitly
specified in the manuscript are exposed transparently through configuration
options and documented in the Implementation Audit (Section 13). Minor numerical
differences may occur depending on dataset versions, preprocessing configuration,
software environment, and stochastic execution.

Quick reference:

| Requirement (per the manuscript's Code Availability statement) | Status |
|---|---|
| Model code (encoder, decoder, critic, ACAI, dynamic loss, clustering) | Present -- implements every stated equation and architecture size |
| Preprocessing scripts | Present -- normalization/augmentation/windowing code exists; several parameters are configurable rather than manuscript-fixed |
| Hyperparameter settings | Present for everything the manuscript states (Section 4.3); MaxItr's numeric value is configurable (not manuscript-stated) |
| Configuration files | Present -- `configs/lstm_dynae.yaml` exposes every manuscript hyperparameter plus every configurable item |
| Evaluation code | Present -- `lstm_dynae/metrics.py`, `scripts/evaluate.py` |
| Statistical significance code | Present -- `scripts/statistical_test.py` |
| Instructions to execute the pipeline | Present -- Sections 8-11 above |

Items configurable rather than manuscript-fixed (see Section 13 for the full
classification): critic architecture, K-Means initialization variant, 1-NN
candidate pool and refresh cadence, `MaxItr`, exact dataset source files, EEG Eye
State / Ozone Level Detection windowing, normalization method, and data-augmentation
magnitudes. Each is exposed as an explicit, documented configuration option rather
than an invented default.

## 13. Implementation Audit

Classification key: **A** = explicitly stated in manuscript · **B** = direct
mathematical consequence of the manuscript · **C** = necessary software
implementation choice not explicitly specified · **D** = numerical/architectural
detail not explicitly reported in the manuscript, exposed as a configurable
implementation choice.

| # | Item | Class | Notes |
|---|---|---|---|
| 1 | Encoder: LSTM(64) -> LSTM(128) | A | Section 3.1 |
| 2 | Decoder: LSTM(128) -> LSTM(64), mirrored | A | Section 3.1 |
| 3 | Reconstruction loss = MSE (Eq. 3) | A | `autoencoder.py::reconstruction_loss` |
| 4 | Decoder reconstructs in **reverse time order** | B | Manuscript notation `x̂ᵢ={x̂ᵢᵀ,...,x̂ᵢ¹}`; implemented via `reverse_time()` in `autoencoder.py` |
| 5 | Latent dim = 128 (encoder's 2nd layer), distinct from raw feature dim `d` | C | Manuscript literally writes both `zᵢ∈ℝᵈ` and `x̂ᵢ∈ℝᵈ`, reusing symbol `d`; the only reading consistent with the stated "128 units" bottleneck (see `autoencoder.py` docstring) |
| 6 | ACAI interpolation Eq. 4-5 (`z_α`, `x_α`) | A | `acai.py` |
| 7 | `L_fg` (Eq. 6), `L_C` (Eq. 7), `lambda=0.5` | A | `acai.py` |
| 8 | tau_1, tau_2 ~ Uniform(0,1) per iteration | B | "randomly selected from the range [0,1]" -- uniform is the natural reading |
| 9 | In-batch random pairing of (x1,x2) for interpolation | C | Fig. 2 shows 3 input streams; batch-pairing is the standard minibatch realization, not explicitly mandated |
| 10 | Critic predicts tau_1 from a sequence, scalar output | A | Section 3.2 |
| 11 | Critic architecture (layers/units/activations) | D | Configurable implementation choice not explicitly specified in the manuscript; `critic.py` uses a small LSTM(64)->LSTM(32)->Dense(1), exposed via the `critic:` section of `configs/lstm_dynae.yaml` |
| 12 | Pretraining: 1.3e5 iterations, Adam, lr=1e-4 | A | Section 3.2 / 4.3 |
| 13 | Adam beta_1=0.9, beta_2=0.999 | A | Section 4.3 |
| 14 | Seed = 42 | A | Section 4.3 |
| 15 | K-Means centroid init | A | Section 3.3.1 |
| 16 | K-Means variant: plain Lloyd's vs. k-means++, `n_init` | D | Manuscript says "K-means... for simplicity," doesn't disambiguate; `clustering.py` defaults to scikit-learn's k-means++, exposed as a config toggle |
| 17 | Student-t soft assignment (Eq. 1), alpha=1 | A | Section 4.3 |
| 18 | `sigma(x_i)` hard assignment (Eq. 2) | A | `clustering.py` |
| 19 | Conflicted set S̄ (Eq. 8), unconflicted set S (Eq. 13) | A | `dynamic_loss.py` |
| 20 | alpha_1 = kappa/K, alpha_2 = alpha_1/2, kappa=3 (Eq. 10) | A | Section 4.3 |
| 21 | L1 (Eq. 9): reconstruction if conflicted, else `‖g(sigma(xi))-x̂i‖²` | A | `dynamic_loss.py` |
| 22 | 1-NN centroid substitution mu'_r | A (existence) / D (mechanics) | *That* a 1-NN substitution happens is explicit; the candidate pool (batch vs. full dataset) and refresh cadence are configurable (`neighbor_pool` toggle) |
| 23 | L2 (Eq. 12), L=L1+L2 (Eq. 14) | A | `dynamic_loss.py` |
| 24 | Encoder+decoder NOT frozen during clustering | A | Explicit manuscript statement (Section 3.1 closing paragraph); `model.py::clustering_train_step` differentiates w.r.t. the full autoencoder |
| 25 | Clustering optimizer: SGD+momentum, lr=0.001, momentum=0.9, batch=256 | A | Section 4.3 |
| 26 | Stopping criterion: tau_p < 1% OR MaxItr reached | A (criterion) / D (MaxItr numerical value) | The stopping criterion is explicitly stated in the manuscript; the numerical value of MaxItr is not explicitly reported and remains configurable (`clustering.max_iterations`, no default -- see Section 9) |
| 27 | ACC via Hungarian algorithm (Eq. 15) | A | `metrics.py` |
| 28 | NMI, arithmetic-mean normalization (Eq. 16) | A / C (library flag) | `average_method="arithmetic"` pinned explicitly |
| 29 | Wilcoxon one-sided test, W+=15/W-=0/p=1/32 | A / B | A: test + directionality + significance level stated explicitly; B: p=1/32 is itself a mathematical consequence of "all 5 differences positive," recomputed (not hard-coded) by `statistical_test.py` |
| 30 | Significance level alpha=0.05 | A | Section 4.2 |
| 31 | Dataset shapes (Table 1) | A | `datasets.py::DATASET_REGISTRY` |
| 32 | Exact dataset source files/versions | D | See `data/README.md`; PenDigits' N=7494 matches only the UEA archive's train split; Ozone's 24 dims doesn't exactly match UCI's 72-feature variant |
| 33 | EEG Eye State / Ozone sequence windowing | D | Both are natively continuous streams/tabular records, not pre-segmented sequences; window length/stride are configurable, not manuscript-specified |
| 34 | Normalization/scaling method | D | Not specified; `preprocessing.py` defaults to per-feature min-max |
| 35 | Data augmentation magnitudes (rotation angle, rescale factor) | D | Qualitatively described ("rotating and rescaling"), no numeric ranges given |
| 36 | Train ordering / shuffling | C | Not specified; standard per-epoch shuffling used, seeded |
| 37 | LSTM activation functions | C | Not specified; Keras defaults (tanh / sigmoid recurrent) used |
| 38 | Weight initialization scheme | C | Not specified; Keras defaults (Glorot-uniform / orthogonal) used |
| 39 | Checkpointing strategy | C | Not specified; `.weights.h5` saved after each phase, path configurable |
| 40 | FR (Eq. 17) / FD (Eq. 18) metrics | A (definition) / D (operational losses) | Definitions given; the concrete "supervised" vs. "pretext" loss pair to differentiate for LSTM-DynAE's own training is configurable, provided as an optional `metrics.gradient_cosine_similarity` helper |
| 41 | Stopping decision uses **dataset-level** tau_p (Eq. 11), not per-batch | B | Eq. 11 defines N as the total dataset size; `dynamic_loss.py::compute_global_tau_p` and `model.py::LSTMDynAE.compute_global_tau_p` implement this over the full dataset, distinct from the cheaper per-batch `batch_tau_p` used only for logging |
| 42 | Frequency of dataset-level tau_p (re-)evaluation during training | D | The manuscript specifies the tau_p<1% criterion itself but not how often it should be checked during mini-batch training; `clustering.tau_p_check_interval` (default: once per epoch-equivalent) is configurable |

## 14. Citation

If you use this code, please cite the manuscript:

```bibtex
@article{alsaeed_lstmdynae,
  title  = {Sequential Deep Embedded Clustering with Dynamic Autoencoder},
  author = {AlSaeed, Hasna and Ksantini, Riadh and AlKhateeb, Faisal},
  note   = {Manuscript under review}
}
```

Publication metadata (journal volume, issue, pages, DOI) will be added once
assigned. For the underlying methods this work builds on:

```bibtex
@article{berthelot2018acai,
  title   = {Understanding and improving interpolation in autoencoders via an adversarial regularizer},
  author  = {Berthelot, David and others},
  journal = {arXiv preprint arXiv:1807.07543},
  year    = {2018}
}
@article{wilcoxon1945,
  title   = {Individual comparisons by ranking methods},
  author  = {Wilcoxon, Frank},
  journal = {Biometrics Bulletin},
  volume  = {1},
  number  = {6},
  pages   = {80--83},
  year    = {1945}
}
```

## 15. License

See `LICENSE`. The manuscript's Code Availability statement does not specify a
license; the MIT License shipped here is a placeholder default, not an
author-approved or institutionally-confirmed choice. Please confirm the intended
license with the authors/institutions before treating this as the final public
release, or set it explicitly if a decision has already been made elsewhere.
