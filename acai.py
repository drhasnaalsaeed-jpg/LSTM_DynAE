"""
lstm_dynae/acai.py
====================

Adversarially Constrained Autoencoder Interpolation (ACAI), Section 3.2,
Eqs. (4)-(7), Fig. 2 of the manuscript. This is the regularizer used
during PRETRAINING (Phase I) only.

Manuscript equations implemented here (classification A -- explicitly
stated):

    z_alpha  = tau_1 * f_phi_e(x_1) + (1 - tau_1) * f_phi_e(x_2)      (Eq.4)
    x_alpha  = g_phi_d(z_alpha)                                       (Eq.5)
    L_fg(phi_e, phi_d) = ||x - x_hat||_2^2 + lambda * ||C(x_alpha)||_2^2   (Eq.6)
    L_C(phi_C)          = ||C(x_alpha) - tau_1||_2^2
                           + ||C(tau_2 * x + (1 - tau_2) * x_hat)||_2^2   (Eq.7)

Manuscript-specified hyperparameters (classification A):
    lambda = 0.5                      ("Use the exact manuscript
                                        regularization coefficient")
    tau_1, tau_2 ~ "randomly selected from the range [0, 1]" each
                    training iteration (Section 3.2, paragraph
                    preceding Eq. 4). We implement this literally as
                    tau_1, tau_2 ~ Uniform(0, 1), independently per
                    sample in the batch. The manuscript names the
                    *range* explicitly; the specific distribution shape
                    within that range (uniform) is the only natural
                    reading of "randomly selected from the range
                    [0, 1]" and is treated here as classification B
                    (direct, essentially forced, consequence of the
                    stated range) rather than a free invention. Note
                    that the original ACAI paper (Berthelot et al.,
                    references [1]/[2]) actually samples from
                    Uniform(0, 0.5) and mirrors it -- if bit-exact
                    reproduction of that specific detail is required,
                    note that this manuscript's own text states [0, 1]
                    rather than [0, 0.5], and this implementation follows
                    the manuscript's stated range.

Fig. 2 shows three input streams entering the (augmented, shared-weight)
encoder: x_1, x_2 (used for the latent interpolation z_alpha) and a third
stream x (used for the ordinary, non-interpolated reconstruction x_hat
appearing in the first term of Eq. 6 and the tau_2-blend of Eq. 7). Within
a training batch we realize this by:
    - x        = the batch itself (after augmentation, see preprocessing.py)
    - (x_1,x_2)= two independent random shuffles of the same batch, i.e. a
                 random pairing of samples within the batch for the
                 interpolation branch.
This pairing scheme is a necessary implementation choice not pinned down
by the manuscript (classification C) -- an alternative reading would draw
x_1, x_2 from a *separate* augmented copy of the whole dataset rather than
in-batch shuffles. In-batch pairing is the standard, minibatch-compatible
realization of ACAI's original formulation and is documented here as such.
"""

from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from .autoencoder import LSTMAutoencoder
from .critic import Critic

LAMBDA_ACAI_DEFAULT = 0.5  # manuscript-specified regularization coefficient


class ACAILosses(NamedTuple):
    l_fg: tf.Tensor          # autoencoder (generator) loss, Eq. 6
    l_c: tf.Tensor           # critic loss, Eq. 7
    recon_term: tf.Tensor    # ||x - x_hat||^2 term, for logging
    critic_reg_term: tf.Tensor  # lambda * ||C(x_alpha)||^2 term, for logging


def sample_interpolation_coefficients(batch_size: int, dtype=tf.float32) -> tf.Tensor:
    """tau ~ Uniform(0, 1), one scalar per sample in the batch (Section 3.2)."""
    return tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=1.0, dtype=dtype)


def pair_batch_for_interpolation(x: tf.Tensor, seed: int = None):
    """Randomly pairs samples within a batch to form (x_1, x_2) for Eq. 4.

    See module docstring for the rationale (classification C).
    """
    batch_size = tf.shape(x)[0]
    perm1 = tf.random.shuffle(tf.range(batch_size), seed=seed)
    perm2 = tf.random.shuffle(tf.range(batch_size), seed=seed)
    x1 = tf.gather(x, perm1)
    x2 = tf.gather(x, perm2)
    return x1, x2


def acai_losses(
    autoencoder: LSTMAutoencoder,
    critic: Critic,
    x: tf.Tensor,
    lambda_acai: float = LAMBDA_ACAI_DEFAULT,
    training: bool = True,
) -> ACAILosses:
    """Computes L_fg (Eq. 6) and L_C (Eq. 7) for one batch.

    Parameters
    ----------
    autoencoder: shared encoder/decoder f_phi_e, g_phi_d.
    critic: critic network C with parameters phi_C.
    x: (batch, T, features) already-augmented batch (see preprocessing.py).
    lambda_acai: manuscript's lambda in Eq. 6 (default 0.5).
    """
    batch_size = tf.shape(x)[0]

    # --- ordinary (non-interpolated) reconstruction branch, used in the
    # first term of Eq. 6 and the tau_2-blend of Eq. 7 ---
    z = autoencoder.encode(x, training=training)
    x_hat = autoencoder.decode(z, training=training)

    # --- interpolation branch, Eq. 4-5 ---
    x1, x2 = pair_batch_for_interpolation(x)
    z1 = autoencoder.encode(x1, training=training)
    z2 = autoencoder.encode(x2, training=training)
    tau_1 = sample_interpolation_coefficients(batch_size, dtype=x.dtype)
    tau_1_bc = tf.reshape(tau_1, (-1, 1))  # broadcast over latent_dim
    z_alpha = tau_1_bc * z1 + (1.0 - tau_1_bc) * z2                      # Eq. 4
    x_alpha = autoencoder.decode(z_alpha, training=training)              # Eq. 5

    # --- Eq. 6: L_fg = ||x - x_hat||_2^2 + lambda * ||C(x_alpha)||_2^2 ---
    recon_term = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2]))
    c_x_alpha = critic(x_alpha, training=training)
    critic_reg_term = lambda_acai * tf.reduce_mean(tf.square(c_x_alpha))
    l_fg = recon_term + critic_reg_term

    # --- Eq. 7: L_C = ||C(x_alpha) - tau_1||_2^2 + ||C(tau_2*x + (1-tau_2)*x_hat)||_2^2 ---
    # NOTE: critic input for the first term must NOT be back-propagated
    # into the autoencoder when training the critic; callers training the
    # critic and the autoencoder in separate optimizer steps should
    # `tf.stop_gradient` x_alpha / x_hat as appropriate (see scripts/pretrain.py).
    tau_2 = sample_interpolation_coefficients(batch_size, dtype=x.dtype)
    tau_2_bc = tf.reshape(tau_2, (-1, 1, 1))
    blend = tau_2_bc * x + (1.0 - tau_2_bc) * x_hat
    c_x_alpha_for_critic = critic(tf.stop_gradient(x_alpha), training=training)
    c_blend = critic(tf.stop_gradient(blend), training=training)
    l_c = tf.reduce_mean(tf.square(c_x_alpha_for_critic - tau_1)) + tf.reduce_mean(tf.square(c_blend))

    return ACAILosses(l_fg=l_fg, l_c=l_c, recon_term=recon_term, critic_reg_term=critic_reg_term)
