"""
lstm_dynae/metrics.py
========================

Evaluation metrics, Section 3.4, Eqs. (15)-(18).

    Acc(y_p, y_t) = max_T { sum_i 1{y_t(i) = T(y_p(i))} / N }             (Eq.15)
    NMI(y_p, y_t) = I(y_p, y_t) / ( 0.5 * [H(y_t) + H(y_p)] )             (Eq.16)

Both use the Hungarian algorithm (via scipy.optimize.linear_sum_assignment)
to find the optimal predicted-to-ground-truth cluster label mapping T
(classification A: "The Hungarian Algorithm [13] is used to obtain this
mapping.").

NMI (Eq. 16) uses the ARITHMETIC mean of the two entropies in its
denominator (0.5 * [H(y_t) + H(y_p)]). scikit-learn's
`normalized_mutual_info_score` supports an `average_method` argument;
we set `average_method="arithmetic"` to match Eq. 16 exactly
(classification B -- direct consequence of matching the stated formula,
since scikit-learn's *default* average_method is "arithmetic" as of
recent versions, but this is pinned explicitly here rather than relying
on a library default that could silently change).

Feature-Randomness (Delta FR) and Feature-Drift (Delta FD), Eqs. (17)-(18),
are OPTIONAL / supplementary metrics: they are core contributions
highlighted by the manuscript (Section 3.4 second half) but were not among
the metrics explicitly requested for this repository (evaluation
requirements: ACC, NMI, execution time, seed control, optional loss/ACC/
NMI/tau_p logging). They are included here for completeness/fidelity but
should be treated as a secondary, best-effort addition:

    Delta FR = cos( dL(x,y_t,omega)/domega , dL(x,y_p,omega)/domega )     (Eq.17)
    Delta FD = cos( dL_P(x,y_p,omega)/domega , dL_S(x,y_pretext,omega)/domega )  (Eq.18)

Classification D (configurable implementation choice not explicitly
specified in the manuscript): the manuscript defines these as cosine
similarities between gradients of a "supervised" loss
(w.r.t. ground-truth-like labels y_t / pseudo-labels y_p) and an
"unsupervised"/pretext loss, but does not fully specify which concrete
loss functions L(x, y_t, omega), L(x, y_p, omega), L_P and L_S are used
operationally during LSTM-DynAE's own training. We provide a generic
gradient-cosine-similarity helper (`gradient_cosine_similarity`) that a
training loop can call with whichever two loss tensors it defines as the
"supervised" and "unsupervised" objectives, but we do NOT wire up a
specific choice by default.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. (15): optimal-mapping clustering accuracy via the Hungarian algorithm.

    y_true, y_pred: 1-D integer arrays of the same length (ground-truth
    labels are used only for this external evaluation step, never for
    training -- see Section 4 / user instructions).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_true.size == y_pred.size, "y_true and y_pred must have the same length"

    n_true = int(y_true.max()) + 1
    n_pred = int(y_pred.max()) + 1
    dim = max(n_true, n_pred)

    cost = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)  # maximize matches == minimize -cost
    matched = cost[row_ind, col_ind].sum()
    return float(matched) / y_pred.size


def normalized_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. (16): NMI with arithmetic mean normalization."""
    return float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))


@contextmanager
def timer():
    """Execution-time measurement context manager (user requirement D.3).

    Usage:
        with timer() as t:
            ...
        elapsed_seconds = t["seconds"]
    """
    state = {"seconds": None}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start


def gradient_cosine_similarity(grads_a: Iterable, grads_b: Iterable) -> float:
    """cos(g_a, g_b) between two same-shaped lists of gradient tensors,
    used to instantiate Delta FR (Eq. 17) / Delta FD (Eq. 18). See module
    docstring: which two losses to differentiate is AUTHOR CONFIRMATION
    REQUIRED and left to the caller.
    """
    import tensorflow as tf

    flat_a = tf.concat([tf.reshape(g, [-1]) for g in grads_a if g is not None], axis=0)
    flat_b = tf.concat([tf.reshape(g, [-1]) for g in grads_b if g is not None], axis=0)
    num = tf.reduce_sum(flat_a * flat_b)
    denom = tf.norm(flat_a) * tf.norm(flat_b) + 1e-12
    return float((num / denom).numpy())


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper bundling ACC and NMI (user requirement D.1-D.2)."""
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_information(y_true, y_pred),
    }
