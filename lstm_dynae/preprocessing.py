"""
lstm_dynae/preprocessing.py
==============================

Preprocessing utilities: normalization, fixed-length sequence framing, and
the data-augmentation transform used both in the pretraining phase
(Fig. 2, "Data Augmentation" block) and the clustering phase (Fig. 3,
"Data Augmentation" block).

IMPORTANT -- read before using on EEG Eye State / Ozone Level Detection:
The manuscript states the five datasets are used "exactly as configured"
(Table 1) as multivariate time series with a fixed (samples, dimensions)
shape. For PenDigits, Handwriting and RacketSports this is unambiguous:
these are established multivariate time-series benchmarks (UEA archive)
where each of the N samples is *already* a short discrete sequence.

For EEG Eye State (14980, 14, 2) and Ozone Level Detection (2536, 24, 2),
however, the sample counts match single, continuously-recorded streams
(UCI "EEG Eye State": 14980 rows x 14 EEG channels x 1 label per row;
UCI "Ozone Level Detection": ~2536 daily records) rather than naturally
pre-segmented collections of independent short sequences. The manuscript
gives NO windowing/segmentation procedure (window length, stride, or
whether "samples" here means individual timesteps treated with a sliding
context window, non-overlapping blocks, or something else) for turning
such a stream into the "N sequential samples of length T" required by an
LSTM autoencoder.

  ==> Configurable implementation choice not explicitly specified in the
      manuscript (classification D). This is a high-priority item: without
      the exact windowing scheme, results for EEG Eye State and Ozone
      Level Detection are approximate rather than an exact reproduction of
      the manuscript's experiments. We expose window length/stride as
      explicit config parameters (`sequence_length`, `stride`) rather than
      guessing manuscript-accurate defaults.

Normalization (classification C -- necessary implementation choice, not
specified in the manuscript): we default to per-feature min-max scaling
to [0, 1], a common choice for LSTM autoencoders and compatible with a
sigmoid-free linear decoder output. This is a configurable implementation
choice, not a manuscript-reported value.

Data augmentation (classification D -- configurable implementation choice
not explicitly specified in the manuscript):
Manuscript text (Section 3.2): "This technique involves rotating and
rescaling data, serving as a regularizer..." (describing prior work the
authors build on) and later: "the proposed reconstruction objective
function of LSTM-DynAE is optimized with both an adversarial regularizer
and data augmentation." No rotation-angle range, rescale-factor range, or
per-channel vs. whole-sequence application is specified. We implement a
configurable, conservative default (small random per-channel rescaling
and a small random rotation applied to *pairs* of channels, since
"rotation" is only well-defined for >=2 dimensional data -- for datasets
with a single derived channel this degenerates to rescaling only); all
magnitude constants below are configurable, not manuscript-reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class PreprocessingConfig:
    # --- sequence framing (see module docstring: configurable, high-
    # priority item for EEG Eye State / Ozone Level Detection) ---
    sequence_length: Optional[int] = None  # None => dataset already framed (PenDigits/Handwriting/RacketSports)
    stride: int = 1

    # --- normalization (classification C, default choice) ---
    normalization: str = "minmax"  # "minmax" | "zscore" | "none"

    # --- data augmentation (classification D, defaults are configurable placeholders) ---
    augment: bool = True
    max_rescale_delta: float = 0.10   # +/-10% amplitude jitter (configurable placeholder)
    max_rotation_rad: float = 0.05    # ~2.9 degree channel-pair rotation (configurable placeholder)


def frame_sequences(stream: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
    """Turns a single continuous (L, features) stream into a set of fixed
    length (sequence_length, features) sliding-window sequences.

    Configurable implementation choice for EEG Eye State / Ozone Level
    Detection: see module docstring. `sequence_length`/`stride` must be
    supplied by the user; there is no manuscript-derived default.
    """
    L, feat = stream.shape
    if sequence_length > L:
        raise ValueError(f"sequence_length={sequence_length} exceeds stream length={L}")
    starts = range(0, L - sequence_length + 1, stride)
    windows = np.stack([stream[s:s + sequence_length] for s in starts], axis=0)
    return windows.astype(np.float32)


def normalize(x: np.ndarray, method: str = "minmax", stats: Optional[dict] = None):
    """Per-feature normalization applied over the (N, T, features) array.

    Returns (x_normalized, stats) so the same stats can be re-applied to
    validation splits or new data.
    """
    if method == "none":
        return x.astype(np.float32), {}

    flat = x.reshape(-1, x.shape[-1])
    if method == "minmax":
        if stats is None:
            fmin = flat.min(axis=0)
            fmax = flat.max(axis=0)
            stats = {"min": fmin, "max": fmax}
        denom = np.where(stats["max"] - stats["min"] == 0, 1.0, stats["max"] - stats["min"])
        x_norm = (x - stats["min"]) / denom
    elif method == "zscore":
        if stats is None:
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            stats = {"mean": mean, "std": std}
        denom = np.where(stats["std"] == 0, 1.0, stats["std"])
        x_norm = (x - stats["mean"]) / denom
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return x_norm.astype(np.float32), stats


def augment_batch(x: tf.Tensor, config: PreprocessingConfig) -> tf.Tensor:
    """Applies the rotation + rescaling augmentation described
    qualitatively in Section 3.2. See module docstring: exact magnitudes
    are a configurable implementation choice, not manuscript-reported
    values.
    """
    if not config.augment:
        return x

    batch_size = tf.shape(x)[0]
    num_features = x.shape[-1]

    # --- rescaling: independent multiplicative jitter per sample ---
    scale = 1.0 + tf.random.uniform(
        (batch_size, 1, 1), -config.max_rescale_delta, config.max_rescale_delta, dtype=x.dtype
    )
    x_aug = x * scale

    # --- rotation: applied to the first two channels only (rotation is
    # only defined for >=2 dims); if the data has a single channel this
    # is a no-op. Channel-pairing strategy for datasets with >2 channels
    # (we rotate channels 0,1 only by default) is a configurable
    # implementation choice not specified in the manuscript.
    if num_features is not None and num_features >= 2:
        theta = tf.random.uniform((batch_size,), -config.max_rotation_rad, config.max_rotation_rad, dtype=x.dtype)
        cos_t = tf.reshape(tf.cos(theta), (-1, 1))
        sin_t = tf.reshape(tf.sin(theta), (-1, 1))
        c0 = x_aug[:, :, 0]
        c1 = x_aug[:, :, 1]
        new_c0 = c0 * cos_t - c1 * sin_t
        new_c1 = c0 * sin_t + c1 * cos_t
        rest = x_aug[:, :, 2:]
        x_aug = tf.concat([tf.expand_dims(new_c0, -1), tf.expand_dims(new_c1, -1), rest], axis=-1)

    return x_aug
