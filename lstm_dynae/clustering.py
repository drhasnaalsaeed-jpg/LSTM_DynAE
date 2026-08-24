"""
lstm_dynae/clustering.py
==========================

Clustering-center initialization and soft/hard cluster assignment,
Section 3 (preliminaries) and Section 3.3.1, Eqs. (1)-(2).

Manuscript equations (classification A -- explicitly stated):

    q_ir = (1 + ||z_i - mu_r||^2 / alpha)^(-(alpha+1)/2)
           / sum_r' (1 + ||z_i - mu_r'||^2 / alpha)^(-(alpha+1)/2)        (Eq.1)

    sigma(x_i) = mu_argmax_r(q_ir)                                        (Eq.2)

Manuscript-specified hyperparameter (Section 4.3): alpha = 1.

Centroid initialization (Section 3.3.1, classification A for the fact that
K-Means is used; classification D for the exact variant):
    "K-means is employed in LSTM-DynAE to initialize the embedded
     clustering centers." K is fixed to the ground-truth class count
     (Section 4.2). The manuscript does not state whether "K-means" here
     means plain Lloyd's K-Means with random initialization, or the
     now-common default of k-means++ seeding (both are commonly referred
     to as simply "K-means" in practice) -- nor does it state the number
     of restarts (`n_init`) or which underlying algorithm variant
     (Lloyd/Elkan) was used. This implementation defaults to
     scikit-learn's standard k-means++ initialization, documented here as
     a configurable implementation choice (`init_method`), not a value
     reported by the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

STUDENT_T_ALPHA_DEFAULT = 1.0  # manuscript Section 4.3


@dataclass
class ClusteringConfig:
    num_clusters: int
    student_t_alpha: float = STUDENT_T_ALPHA_DEFAULT
    # Configurable implementation choice not explicitly specified in the
    # manuscript (see module docstring): K-Means variant/initialization.
    init_method: str = "k-means++"
    random_seed: int = 42


def student_t_soft_assignment(z: tf.Tensor, centroids: tf.Tensor, alpha: float = STUDENT_T_ALPHA_DEFAULT) -> tf.Tensor:
    """Eq. (1): soft cluster assignments q_ir via a Student's t kernel.

    z: (N, latent_dim) embedded points.
    centroids: (K, latent_dim) latent cluster centers mu_r.
    returns: (N, K) matrix Q = (q_ir).
    """
    z_exp = tf.expand_dims(z, axis=1)               # (N, 1, D)
    c_exp = tf.expand_dims(centroids, axis=0)        # (1, K, D)
    sq_dist = tf.reduce_sum(tf.square(z_exp - c_exp), axis=-1)  # (N, K)

    numerator = tf.pow(1.0 + sq_dist / alpha, -((alpha + 1.0) / 2.0))
    denominator = tf.reduce_sum(numerator, axis=1, keepdims=True)
    q = numerator / denominator
    return q


def hard_assignment(q: tf.Tensor, centroids: tf.Tensor) -> tf.Tensor:
    """Eq. (2): sigma(x_i) = mu_argmax_r(q_ir)."""
    idx = tf.argmax(q, axis=1)
    return tf.gather(centroids, idx)


def kmeans_init_centroids(embeddings: np.ndarray, config: ClusteringConfig):
    """K-Means initialization of the embedded clustering centers
    (Section 3.3.1). Returns (centroids: (K, D) float32 array,
    labels: (N,) int array).
    """
    km = KMeans(
        n_clusters=config.num_clusters,
        init=config.init_method,
        n_init=10,
        random_state=config.random_seed,
    )
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_.astype(np.float32)
    return centroids, labels
