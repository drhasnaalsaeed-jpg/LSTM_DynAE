# Data

This repository does **not** redistribute any of the five benchmark
datasets, consistent with the manuscript's Data Availability Statement
("The datasets analyzed during this study are publicly available
benchmark datasets and are appropriately cited in the manuscript") and
with not knowing each dataset's redistribution license.

Table 1 of the manuscript specifies the following (samples, dimensions,
classes):

| Dataset | Samples | Dimensions | Classes |
|---|---|---|---|
| PenDigits | 7494 | 2 | 10 |
| Handwriting | 1000 | 3 | 26 |
| RacketSports | 303 | 6 | 4 |
| EEG Eye State | 14980 | 14 | 2 |
| Ozone Level Detection | 2536 | 24 | 2 |

## Dataset source files: configurable, not manuscript-specified

The manuscript text does not cite a specific dataset repository, archive
version, or download URL for any of the five datasets. Based on the
(samples, dimensions, classes) triples alone:

- **PenDigits**, **Handwriting**, **RacketSports** — these numbers match
  the UEA & UCR Multivariate Time Series Classification Archive's
  datasets of the same names (Handwriting: 1000/3/26; RacketSports:
  303/6/4). PenDigits' N=7494 matches only the archive's **train** split
  (train=7494, test=3498); it is unconfirmed whether the authors used the
  train split only, a different split, or a custom combination.
- **EEG Eye State** (14980/14/2) — matches the UCI Machine Learning
  Repository's "EEG Eye State" dataset (14980 rows, 14 channels, binary
  label). That dataset is natively a **single continuous recording**, not
  a pre-segmented collection of independent sequences — see the
  windowing gap below.
- **Ozone Level Detection** (2536/24/2) — close to, but not an exact
  feature-count match with, the UCI "Ozone Level Detection" dataset
  (72 raw features for the "onehr"/"eighthr" variants, not 24). The
  feature subset or transformation used to reach 24 dimensions is
  unspecified.

**Until the exact source files are confirmed, treat any locally-prepared
copy of EEG Eye State / Ozone Level Detection in particular as an
approximation, not a guaranteed match to the manuscript's experiments.**

## High-priority item: windowing for EEG Eye State / Ozone Level Detection

An LSTM autoencoder requires each of the N "samples" to already be a
short sequence `(T, features)`. PenDigits/Handwriting/RacketSports are
natively like this. EEG Eye State and Ozone Level Detection, as sourced
from UCI, are **not** — each is a single long stream (or a small number of
per-day tabular records). The manuscript gives no window length, stride,
or segmentation rule for turning such a stream into "N sequential
samples." This is a configurable implementation choice throughout the
codebase (see `lstm_dynae/preprocessing.py`, `lstm_dynae/datasets.py`).

## Expected local layout

Loaders in `lstm_dynae/datasets.py` expect, per dataset key
(`PenDigits`, `Handwriting`, `RacketSports`, `EEGEyeState`,
`OzoneLevelDetection`):

```
data/<DatasetKey>_X.npy   # (N, T, D) float array for natively-sequential
                           # datasets, OR (L, D) float array (a single
                           # continuous stream) for EEG Eye State / Ozone
                           # Level Detection, to be windowed via
                           # preprocessing.frame_sequences().
data/<DatasetKey>_y.npy   # (N,) integer ground-truth labels, used ONLY
                           # for external evaluation (Acc/NMI), never for
                           # training.
```

Convert whatever raw files you download (`.ts`, `.arff`, `.data`, ...)
into this `.npy` convention yourself; the `.npy` convention itself is a
software-packaging choice made for this repository, not something
specified by the manuscript.
