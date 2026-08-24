"""
lstm_dynae/datasets.py
=========================

Dataset registry and loaders for the five sequential benchmarks used in
the manuscript (Table 1, Section 4.1):

    Dataset                Samples  Dimensions  Classes
    PenDigits                 7494           2       10
    Handwriting                1000           3       26
    RacketSports                303           6        4
    EEG Eye State             14980          14        2
    Ozone Level Detection      2536          24        2

These exact (samples, dimensions, classes) triples are reproduced below
verbatim from Table 1 and are used only for validating that a locally
downloaded dataset matches the manuscript's configuration -- NOT for
generating synthetic data. Per the user's instructions and the
manuscript's own Data Availability Statement ("The datasets analyzed
during this study are publicly available benchmark datasets"), this
repository does NOT redistribute any dataset; it expects the user to
download the raw files themselves (see data/README.md) and points loaders
at a local path.

Configurable implementation choices not explicitly specified in the
manuscript (classification D), for every dataset:
    - Exact source file / archive version used (e.g. UEA & UCR Time Series
      Classification Archive vs. UCI Machine Learning Repository release).
    - For PenDigits: the manuscript's N=7494 matches only the UEA archive's
      TRAIN split of PenDigits (train=7494, test=3498, total=10992). It is
      unclear whether only the train split, only the test split, or a
      custom split was used to reach exactly 7494.
    - For Handwriting and RacketSports: N/dims/classes match the UEA
      archive's "Handwriting" (1000/3/26) and "RacketSports" (303/6/4)
      datasets exactly, so these are very likely the UEA versions -- but
      the exact archive release/version is still unconfirmed.
    - For EEG Eye State (14980/14/2) and Ozone Level Detection (2536/24/2):
      these numbers match UCI "EEG Eye State" (14980 rows, 14 channels,
      binary label) and are close to (but for Ozone not an exact
      feature-count match with) UCI "Ozone Level Detection" (72 raw
      features, not 24). Both are natively single continuous streams /
      tabular records, not pre-segmented sequence collections -- see the
      windowing note in preprocessing.py. The exact feature subset used
      to reach 24 dimensions for Ozone is unspecified and configurable.

Because of the above, `load_dataset()` below intentionally raises a
clear, informative error rather than silently fabricating or
down/up-sampling data to match Table 1's shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .preprocessing import PreprocessingConfig, frame_sequences, normalize


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    samples: int
    dimensions: int
    classes: int
    # Configurable classification (classification D): True for datasets
    # whose manuscript "samples" count corresponds to a natively
    # pre-segmented sequence collection (PenDigits/Handwriting/
    # RacketSports); False for datasets that must be windowed from a
    # continuous stream, where the windowing scheme is not explicitly
    # specified in the manuscript (EEG Eye State, Ozone).
    natively_sequential: bool


# Table 1, reproduced verbatim (classification A).
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "PenDigits": DatasetSpec("PenDigits", 7494, 2, 10, natively_sequential=True),
    "Handwriting": DatasetSpec("Handwriting", 1000, 3, 26, natively_sequential=True),
    "RacketSports": DatasetSpec("RacketSports", 303, 6, 4, natively_sequential=True),
    "EEGEyeState": DatasetSpec("EEG Eye State", 14980, 14, 2, natively_sequential=False),
    "OzoneLevelDetection": DatasetSpec("Ozone Level Detection", 2536, 24, 2, natively_sequential=False),
}


class DatasetNotFoundError(FileNotFoundError):
    pass


def _require_file(path: Path, dataset_key: str) -> None:
    if not path.exists():
        raise DatasetNotFoundError(
            f"Could not find data file for dataset '{dataset_key}' at {path}.\n"
            f"This repository does not redistribute datasets (see data/README.md "
            f"for download instructions and AUTHOR-CONFIRMATION-REQUIRED notes on "
            f"the exact source files used in the manuscript)."
        )


def load_raw_npy(data_dir: str, dataset_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generic loader: expects <data_dir>/<dataset_key>_X.npy (either
    already-framed (N, T, D) sequences, or a single (L, D) continuous
    stream for datasets with natively_sequential=False) and
    <data_dir>/<dataset_key>_y.npy (ground-truth labels, used for
    evaluation ONLY, never for training -- see Section 4 / user
    instructions).

    This .npy convention is a software packaging choice (classification C):
    the manuscript does not mandate a file format. Users converting the
    original UEA/.ts or UCI/.data files into this format should record the
    conversion steps used, since they are otherwise unrecoverable from the
    manuscript alone.
    """
    spec = DATASET_REGISTRY[dataset_key]
    x_path = Path(data_dir) / f"{dataset_key}_X.npy"
    y_path = Path(data_dir) / f"{dataset_key}_y.npy"
    _require_file(x_path, dataset_key)
    _require_file(y_path, dataset_key)
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def load_dataset(
    data_dir: str,
    dataset_key: str,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    validate_shape: bool = True,
):
    """Loads, (optionally windows,) and normalizes a dataset.

    Returns
    -------
    x: (N, T, D) float32 array, normalized sequences.
    y: (N,) int array, ground-truth labels (evaluation only).
    spec: DatasetSpec describing the manuscript's Table-1 configuration.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_key}'. Known keys: {list(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[dataset_key]
    preprocessing_config = preprocessing_config or PreprocessingConfig()

    x_raw, y = load_raw_npy(data_dir, dataset_key)

    if not spec.natively_sequential:
        if x_raw.ndim != 2:
            raise ValueError(
                f"Expected a (L, {spec.dimensions}) continuous stream for '{dataset_key}' "
                f"(natively_sequential=False) but got array of shape {x_raw.shape}."
            )
        if preprocessing_config.sequence_length is None:
            raise ValueError(
                f"'{dataset_key}' requires an explicit preprocessing_config.sequence_length "
                f"(windowing scheme not explicitly specified in the manuscript; see preprocessing.py)."
            )
        x = frame_sequences(x_raw, preprocessing_config.sequence_length, preprocessing_config.stride)
    else:
        if x_raw.ndim != 3:
            raise ValueError(
                f"Expected already-framed (N, T, {spec.dimensions}) sequences for "
                f"'{dataset_key}' but got array of shape {x_raw.shape}."
            )
        x = x_raw

    if validate_shape and x.shape[-1] != spec.dimensions:
        raise ValueError(
            f"Feature dimension mismatch for '{dataset_key}': expected {spec.dimensions} "
            f"(Table 1) but loaded data has {x.shape[-1]} channels."
        )

    x_norm, _stats = normalize(x, method=preprocessing_config.normalization)
    return x_norm.astype(np.float32), np.asarray(y), spec
