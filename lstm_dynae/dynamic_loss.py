"""
lstm_dynae/dynamic_loss.py
============================

Dynamic clustering loss used in Phase II (clustering phase), Section 3.3.2,
Eqs. (8)-(14). Equations (8)-(14) follow the dynamic reconstruction-to-
centroid objective formulated for LSTM-DynAE in the manuscript.

Equations implemented (classification A -- explicitly stated in manuscript):

    S_bar = { x_i : p_i1 < alpha_1  OR  (p_i1 - p_i2) < alpha_2 }        (Eq.8)

    L1 = sum_i  ||x_i - x_hat_i||^2                  if x_i in S_bar
              ||g(sigma(x_i)) - x_hat_i||^2          otherwise            (Eq.9)

    alpha_1 = kappa / K ,  alpha_2 = alpha_1 / 2 , kappa in [1, K]        (Eq.10)

    tau_p = |S_bar| / N                                                  (Eq.11)

    L2 = sum_{x_i in S} || f_phi_e(x_i) - sigma(x_i) ||^2                (Eq.12)

    S = { x_i : p_i1 >= alpha_1  AND  (p_i1 - p_i2) >= alpha_2 }         (Eq.13)

    L = L1 + L2                                                          (Eq.14)

Manuscript-specified hyperparameters (classification A, confirmed again in
Section 4.3 "Implementation Details"):
    kappa = 3
    student_t alpha = 1  (used in clustering.py for q_ir, Eq. 1)

IMPORTANT -- tau_p must be computed at the dataset level for the stopping
decision (Eq. 11 defines N as the total number of samples, not a minibatch
size). `split_conflicted_unconflicted()` below computes tau_p over
whatever `q` it is given; when called once per training step on a
minibatch of size B, the resulting value is a *per-batch* estimate, not
the manuscript's tau_p. `compute_global_tau_p()` at the bottom of this
module is the function that must be used for the actual stopping
decision: it takes the soft assignments for the FULL dataset (N samples)
and returns the true |S_bar|/N. Per-batch values remain useful as a
lightweight training-time diagnostic, but must not replace the
dataset-level criterion.

1-NN centroid substitution (classification C -- necessary implementation
choice not specified in the manuscript):
    Manuscript text (Section 3.3.2, paragraph following Eq. 9):
    "the selected samples for centroid construction do not accurately
     represent real data points. Therefore, the first nearest neighbor
     (1-NN) ... is employed for each embedded center (mu'_r) as a
     substitute for the generated centroids."
    This specifies THAT a 1-NN substitution of each centroid mu_r by its
    nearest real embedded data point mu'_r is performed, but does NOT
    specify:
        (a) the candidate pool searched for the nearest neighbour (the
            full dataset's embeddings? the current minibatch? the
            unconflicted set S only?),
        (b) how often mu'_r is recomputed (every step? every epoch?).
    We implement 1-NN search over the *current minibatch's* embeddings by
    default (the only pool that is always available inside a `tf.function`
    minibatch training step without a full extra forward pass over the
    dataset), and expose `neighbor_pool` as a config toggle
    ("batch" | "full_dataset") so a full-dataset nearest neighbour search
    (closer to a literal, offline reading of the manuscript) can be
    substituted by the user. Configurable implementation choice not
    explicitly specified in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import tensorflow as tf


@dataclass
class DynamicLossConfig:
    kappa: int = 3          # manuscript-specified confidence threshold (Sec. 4.3)
    num_clusters: int = 10  # K, dataset-dependent (see Table 1's "classes" column)
    # Configurable implementation choice not explicitly specified in the
    # manuscript: candidate pool for 1-NN centroid substitution (see
    # module docstring).
    neighbor_pool: str = "batch"  # "batch" | "full_dataset"

    @property
    def alpha_1(self) -> float:
        """alpha_1 = kappa / K  (Eq. 10)."""
        return self.kappa / float(self.num_clusters)

    @property
    def alpha_2(self) -> float:
        """alpha_2 = alpha_1 / 2  (Eq. 10)."""
        return self.alpha_1 / 2.0


class ConflictSplit(NamedTuple):
    conflicted_mask: tf.Tensor    # bool (batch,) -- membership of S_bar, Eq.8
    unconflicted_mask: tf.Tensor  # bool (batch,) -- membership of S, Eq.13
    p1: tf.Tensor                 # (batch,) highest assignment probability
    p2: tf.Tensor                 # (batch,) second-highest assignment probability
    tau_p: tf.Tensor              # scalar, Eq. 11


def split_conflicted_unconflicted(q: tf.Tensor, alpha_1: float, alpha_2: float) -> ConflictSplit:
    """Computes S_bar (Eq. 8), S (Eq. 13) and tau_p (Eq. 11) from soft
    assignments q (M, K) as produced by clustering.student_t_soft_assignment.

    NOTE: the resulting `tau_p` is |S_bar|/M where M = q.shape[0]. Eq. 11
    defines tau_p over the FULL dataset (M = N). If `q` here is a
    minibatch's assignments, the returned tau_p is only a per-batch
    estimate -- see `compute_global_tau_p()` below for the function that
    must be used for the manuscript's actual stopping decision.
    """
    top2 = tf.math.top_k(q, k=2)
    p1 = top2.values[:, 0]
    p2 = top2.values[:, 1]

    conflicted_mask = tf.logical_or(p1 < alpha_1, (p1 - p2) < alpha_2)      # Eq. 8
    unconflicted_mask = tf.logical_and(p1 >= alpha_1, (p1 - p2) >= alpha_2)  # Eq. 13

    n = tf.cast(tf.shape(q)[0], tf.float32)
    n_conflicted = tf.reduce_sum(tf.cast(conflicted_mask, tf.float32))
    tau_p = n_conflicted / n  # Eq. 11 (over whatever set q covers -- see note above)

    return ConflictSplit(conflicted_mask, unconflicted_mask, p1, p2, tau_p)


def nearest_neighbor_substitute(centroids: tf.Tensor, embedding_pool: tf.Tensor) -> tf.Tensor:
    """mu'_r = 1-NN(mu_r) among `embedding_pool` (Section 3.3.2).

    centroids: (K, latent_dim)
    embedding_pool: (M, latent_dim) -- candidate real embedded points.
    returns: (K, latent_dim) substituted centroids mu'_r.
    """
    # pairwise squared distances (K, M)
    c2 = tf.reduce_sum(tf.square(centroids), axis=1, keepdims=True)          # (K,1)
    p2 = tf.reduce_sum(tf.square(embedding_pool), axis=1, keepdims=True)     # (M,1)
    cross = tf.matmul(centroids, embedding_pool, transpose_b=True)           # (K,M)
    dist2 = c2 - 2.0 * cross + tf.transpose(p2)
    nn_idx = tf.argmin(dist2, axis=1)  # (K,)
    return tf.gather(embedding_pool, nn_idx)


def dynamic_loss(
    x: tf.Tensor,
    x_hat: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    centroids: tf.Tensor,
    decode_fn,
    config: DynamicLossConfig,
    full_dataset_embeddings: tf.Tensor = None,
):
    """Computes L1 (Eq.9), L2 (Eq.12), L = L1+L2 (Eq.14) and tau_p (Eq.11).

    Parameters
    ----------
    x: (batch, T, features) input batch.
    x_hat: (batch, T, features) reconstruction of x by the current
        encoder/decoder (used both for the conflicted branch of L1 and as
        the target for the unconflicted branch of L1).
    z: (batch, latent_dim) = f_phi_e(x), current embeddings.
    q: (batch, K) soft cluster assignments (Eq. 1, from clustering.py).
    centroids: (K, latent_dim) current cluster centers mu_r.
    decode_fn: callable, the decoder g_phi_d, used to compute g(sigma(x_i)).
    config: DynamicLossConfig with kappa, K, neighbor_pool.
    full_dataset_embeddings: (N, latent_dim), required only if
        config.neighbor_pool == "full_dataset".
    """
    alpha_1, alpha_2 = config.alpha_1, config.alpha_2
    split = split_conflicted_unconflicted(q, alpha_1, alpha_2)

    # sigma(x_i) = mu_argmax_r(q_ir)  (Eq. 2, clustering.py); here we need
    # both the raw argmax centroid (for L2, Eq. 12) and the 1-NN-substituted
    # centroid (for the "otherwise" branch of L1, Eq. 9).
    hard_assign_idx = tf.argmax(q, axis=1)                      # (batch,)
    sigma_raw = tf.gather(centroids, hard_assign_idx)            # (batch, latent_dim), Eq. 2

    if config.neighbor_pool == "full_dataset":
        if full_dataset_embeddings is None:
            raise ValueError(
                "neighbor_pool='full_dataset' requires full_dataset_embeddings "
                "(configurable implementation choice; see dynamic_loss.py module docstring)."
            )
        pool = full_dataset_embeddings
    else:
        pool = z  # "batch" pool (default; see module docstring)

    centroids_nn = nearest_neighbor_substitute(centroids, pool)  # mu'_r, all K
    sigma_nn = tf.gather(centroids_nn, hard_assign_idx)          # (batch, latent_dim)

    # --- Eq. 9: L1 ---
    recon_conflicted = tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2])       # ||x_i - x_hat_i||^2
    g_sigma = decode_fn(sigma_nn)                                             # g(sigma(x_i)) using 1-NN substitute
    centroid_construction = tf.reduce_sum(tf.square(g_sigma - x_hat), axis=[1, 2])  # ||g(sigma(x_i)) - x_hat_i||^2

    conflicted_mask_f = tf.cast(split.conflicted_mask, x.dtype)
    per_sample_l1 = conflicted_mask_f * recon_conflicted + (1.0 - conflicted_mask_f) * centroid_construction
    l1 = tf.reduce_sum(per_sample_l1)

    # --- Eq. 12: L2, restricted to unconflicted set S ---
    unconflicted_mask_f = tf.cast(split.unconflicted_mask, x.dtype)
    per_sample_l2 = unconflicted_mask_f * tf.reduce_sum(tf.square(z - sigma_raw), axis=1)
    l2 = tf.reduce_sum(per_sample_l2)

    total_loss = l1 + l2  # Eq. 14

    return {
        "loss": total_loss,
        "l1": l1,
        "l2": l2,
        "batch_tau_p": split.tau_p,
        "conflicted_mask": split.conflicted_mask,
        "unconflicted_mask": split.unconflicted_mask,
        "hard_assign_idx": hard_assign_idx,
    }


def compute_global_tau_p(q_all: tf.Tensor, alpha_1: float, alpha_2: float) -> float:
    """Dataset-level tau_p = |S_bar| / N (Eq. 11), evaluated over the soft
    assignments of the FULL dataset (q_all has shape (N, K), not (batch, K)).

    This is the function that must drive the clustering-phase STOPPING
    decision (`global_tau_p < tau_stop`), per Eq. 11's definition of N as
    the total number of samples. The per-minibatch `batch_tau_p` returned
    by `dynamic_loss()` above is a much cheaper approximation, useful for
    per-step logging/diagnostics, but is not a substitute for this value.

    Callers typically obtain `q_all` by encoding the full dataset in
    memory-sized chunks (see `LSTMDynAE.compute_global_tau_p` in
    lstm_dynae/model.py, which handles the chunked encoding) and calling
    `clustering.student_t_soft_assignment` on the concatenated embeddings
    before passing the result here.
    """
    split = split_conflicted_unconflicted(q_all, alpha_1, alpha_2)
    return float(split.tau_p.numpy())
