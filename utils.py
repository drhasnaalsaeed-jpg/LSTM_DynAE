"""
lstm_dynae/utils.py
======================

Cross-cutting utilities: deterministic seeding, lightweight CSV logging,
and checkpoint save/load helpers.

Manuscript-specified random seed (classification A, Section 4.3):
    "All experiments were run with a fixed random seed of 42 to ensure
     reproducibility."
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

MANUSCRIPT_SEED = 42


def set_global_seed(seed: int = MANUSCRIPT_SEED) -> None:
    """Sets Python, NumPy and TensorFlow global random seeds.

    NOTE: exact bit-for-bit determinism of LSTM training on GPU is NOT
    guaranteed by seeding alone (cuDNN's LSTM kernels use non-deterministic
    reduction order by default). This function seeds every RNG the
    manuscript could plausibly refer to; achieving GPU bit-exact
    determinism additionally requires `TF_DETERMINISTIC_OPS=1`
    (set below) and is still not guaranteed across TensorFlow versions.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


class CSVLogger:
    """Minimal dependency-free CSV logger for training curves
    (clustering loss, ACC, NMI, tau_p -- user requirement D.5).
    """

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_weights(model, path: str) -> None:
    """Saves Keras model weights to `path` (creates parent dirs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)


def load_weights(model, path: str) -> None:
    """Loads Keras model weights from `path`."""
    model.load_weights(path)
