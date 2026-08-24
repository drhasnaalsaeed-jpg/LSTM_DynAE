"""
lstm_dynae/model.py
======================

Top-level LSTM-DynAE model, tying together the LSTM autoencoder
(autoencoder.py), the ACAI critic/pretraining objective (critic.py,
acai.py), and the dynamic clustering objective (clustering.py,
dynamic_loss.py), following Algorithm 1 and Fig. 4 of the manuscript.

Algorithm 1 (verbatim structure):
    Stage 1 -> Pretraining Phase
        Perform LSTM Encoder f_phi_e: x -> z
        Perform LSTM Decoder g_phi_d: z -> x
        Initialize pretrain w via Eq. 6 and Eq. 7
    Stage 2 -> Clustering Phase
        Initialize embedded centroids eta using K-means
        Compute alpha_1, alpha_2 <- kappa via Eq. 10
        while list of conflicted points is not empty <- |X|:
            for i = 0 to MaxItr:
                Update list of conflicted points <- (X, eta, alpha_1, alpha_2)
                Update Centroids eta <- (X, K)
                Update alpha_1 and alpha_2 <- kappa
                if conflicted points < total: End training
                Compute L via Eq. 14
                w <- w - vartheta * dL/dw

Manuscript Section 3, "Preserve the mathematical notation": the encoder is
explicitly NOT frozen after pretraining --

    "Although the proposed framework consists of a pretraining phase
     followed by a clustering phase, LSTM-DynAE does not follow the
     classical two-stage paradigm where the encoder is frozen after
     pretraining. Instead, the encoder-decoder is jointly fine-tuned under
     the dynamic loss, enabling end-to-end adaptation toward the
     clustering objective."

This is implemented by having `clustering_train_step` compute gradients of
the dynamic loss (Eq. 14) with respect to the FULL autoencoder's trainable
variables (encoder AND decoder), not just a clustering head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .autoencoder import AutoencoderConfig, LSTMAutoencoder
from .critic import CriticConfig, Critic
from .acai import acai_losses, LAMBDA_ACAI_DEFAULT
from .clustering import ClusteringConfig, student_t_soft_assignment, kmeans_init_centroids
from .dynamic_loss import DynamicLossConfig, dynamic_loss, compute_global_tau_p


@dataclass
class LSTMDynAEConfig:
    input_timesteps: int
    input_features: int
    num_clusters: int
    seed: int = 42

    encoder_units: tuple = (64, 128)
    decoder_units: tuple = (128, 64)

    # pretraining (Section 3.2 / 4.3, all classification A)
    lambda_acai: float = LAMBDA_ACAI_DEFAULT
    pretrain_iterations: int = 130_000
    pretrain_lr: float = 1e-4
    # Section 4.3: "the first and second-moment exponential decay rates
    # were set to 0.9 and 0.999, respectively."
    pretrain_beta_1: float = 0.9
    pretrain_beta_2: float = 0.999  # manuscript Section 4.3

    # clustering (Section 3.3.3 / 4.3, classification A)
    clustering_lr: float = 0.001
    clustering_momentum: float = 0.9
    batch_size: int = 256
    student_t_alpha: float = 1.0
    kappa: int = 3
    tau_stop: float = 0.01
    # The maximum number of clustering-phase iterations (MaxItr) is not
    # stated numerically in the manuscript (Algorithm 1 / Section 3.3.3
    # reference it symbolically only). Left unset by default; the caller
    # (scripts/train_clustering.py) requires an explicit value before
    # training starts rather than silently choosing one.
    max_iterations: Optional[int] = None

    # Configurable implementation choice not explicitly specified in the
    # manuscript (see dynamic_loss.py module docstring).
    neighbor_pool: str = "batch"  # "batch" | "full_dataset"

    # Critic architecture (critic.py): configurable implementation choice
    # used in this implementation; the exact critic architecture is not
    # explicitly specified in the manuscript (Section 3.2 states only that
    # C predicts a scalar interpolation coefficient from a sequence).
    critic_recurrent_units: Tuple[int, int] = (64, 32)
    critic_output_units: int = 1

    # How often (in clustering-phase iterations) to evaluate the dataset-
    # level tau_p (Eq. 11) that drives the stopping decision. None means
    # "once per epoch-equivalent" (ceil(N / batch_size) iterations).
    # Configurable implementation choice: the manuscript specifies the
    # tau_p < 1% stopping criterion itself, but not how frequently it
    # should be (re-)evaluated during mini-batch training.
    tau_p_check_interval: Optional[int] = None


class LSTMDynAE:
    """Container for the full LSTM-DynAE model and its two training phases."""

    def __init__(self, config: LSTMDynAEConfig):
        self.config = config

        ae_config = AutoencoderConfig(
            input_timesteps=config.input_timesteps,
            input_features=config.input_features,
            encoder_units=config.encoder_units,
            decoder_units=config.decoder_units,
        )
        self.autoencoder = LSTMAutoencoder(ae_config)

        critic_config = CriticConfig(
            input_timesteps=config.input_timesteps,
            input_features=config.input_features,
            hidden_units=config.critic_recurrent_units,
        )
        self.critic = Critic(critic_config)

        self.clustering_config = ClusteringConfig(
            num_clusters=config.num_clusters,
            student_t_alpha=config.student_t_alpha,
            random_seed=config.seed,
        )
        self.dynamic_loss_config = DynamicLossConfig(
            kappa=config.kappa,
            num_clusters=config.num_clusters,
            neighbor_pool=config.neighbor_pool,
        )

        self.ae_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.pretrain_lr,
            beta_1=config.pretrain_beta_1,
            beta_2=config.pretrain_beta_2,
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.pretrain_lr,
            beta_1=config.pretrain_beta_1,
            beta_2=config.pretrain_beta_2,
        )
        self.clustering_optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.clustering_lr,
            momentum=config.clustering_momentum,
        )

        self.centroids: Optional[tf.Variable] = None  # created in init_centroids()

    # ------------------------------------------------------------------
    # Phase I: pretraining (ACAI), Eq. 4-7
    # ------------------------------------------------------------------
    @tf.function
    def pretrain_step(self, x_batch: tf.Tensor):
        # --- critic update (minimize L_C, Eq. 7) ---
        with tf.GradientTape() as tape_c:
            losses = acai_losses(self.autoencoder, self.critic, x_batch, self.config.lambda_acai, training=True)
            l_c = losses.l_c
        critic_grads = tape_c.gradient(l_c, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # --- autoencoder update (minimize L_fg, Eq. 6) ---
        with tf.GradientTape() as tape_g:
            losses = acai_losses(self.autoencoder, self.critic, x_batch, self.config.lambda_acai, training=True)
            l_fg = losses.l_fg
        ae_grads = tape_g.gradient(l_fg, self.autoencoder.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.autoencoder.trainable_variables))

        return {"L_fg": losses.l_fg, "L_C": losses.l_c, "recon": losses.recon_term}

    # ------------------------------------------------------------------
    # Phase II: clustering, Section 3.3, Eq. 8-14, Algorithm 1
    # ------------------------------------------------------------------
    def init_centroids(self, x_all: np.ndarray) -> np.ndarray:
        """K-means initialization of embedded clustering centers
        (Section 3.3.1, Algorithm 1 line 8)."""
        z_all = self.autoencoder.encode(tf.convert_to_tensor(x_all), training=False).numpy()
        centroids, labels = kmeans_init_centroids(z_all, self.clustering_config)
        self.centroids = tf.Variable(centroids, trainable=True, dtype=tf.float32, name="cluster_centroids")
        return labels

    @tf.function
    def clustering_train_step(self, x_batch: tf.Tensor):
        """One mini-batch SGD step of the dynamic loss L = L1 + L2
        (Eq. 14), jointly fine-tuning the ENCODER AND DECODER (not frozen;
        see module docstring) plus the centroids.

        Returns a `batch_tau_p` value for cheap per-step logging only.
        This is NOT the manuscript's dataset-level tau_p (Eq. 11) and must
        not be used to decide when to stop training -- use
        `compute_global_tau_p()` below for that.
        """
        assert self.centroids is not None, "call init_centroids() before clustering_train_step()"

        trainable_vars = self.autoencoder.trainable_variables + [self.centroids]
        with tf.GradientTape() as tape:
            z = self.autoencoder.encode(x_batch, training=True)
            x_hat = self.autoencoder.decode(z, training=True)
            q = student_t_soft_assignment(z, self.centroids, alpha=self.config.student_t_alpha)
            result = dynamic_loss(
                x=x_batch,
                x_hat=x_hat,
                z=z,
                q=q,
                centroids=self.centroids,
                decode_fn=lambda zz: self.autoencoder.decode(zz, training=True),
                config=self.dynamic_loss_config,
            )
            loss = result["loss"]
        grads = tape.gradient(loss, trainable_vars)
        self.clustering_optimizer.apply_gradients(zip(grads, trainable_vars))
        return {
            "loss": loss,
            "l1": result["l1"],
            "l2": result["l2"],
            "batch_tau_p": result["batch_tau_p"],
        }

    def compute_global_tau_p(self, x_all: np.ndarray, encode_batch_size: int = 512) -> float:
        """Dataset-level tau_p = |S_bar| / N (Eq. 11), evaluated over ALL
        N samples, not a minibatch. This is the value that must drive the
        clustering-phase stopping decision (`global_tau_p < tau_stop`).

        The dataset is encoded in fixed-size chunks (`encode_batch_size`)
        purely for memory management on large datasets (e.g. EEG Eye
        State's 14,980 samples); the chunking has no effect on the
        computed value, since it only changes how the forward pass is
        batched, not which points are counted as conflicted.
        """
        assert self.centroids is not None, "call init_centroids() before compute_global_tau_p()"
        n_total = x_all.shape[0]
        z_chunks = []
        for start in range(0, n_total, encode_batch_size):
            end = min(start + encode_batch_size, n_total)
            x_chunk = tf.convert_to_tensor(x_all[start:end])
            z_chunks.append(self.autoencoder.encode(x_chunk, training=False))
        z_all = tf.concat(z_chunks, axis=0)
        q_all = student_t_soft_assignment(z_all, self.centroids, alpha=self.config.student_t_alpha)
        return compute_global_tau_p(q_all, self.dynamic_loss_config.alpha_1, self.dynamic_loss_config.alpha_2)

    def predict_clusters(self, x: np.ndarray) -> np.ndarray:
        """Hard cluster assignment argmax_r q_ir for evaluation (Eq. 1-2)."""
        z = self.autoencoder.encode(tf.convert_to_tensor(x), training=False)
        q = student_t_soft_assignment(z, self.centroids, alpha=self.config.student_t_alpha)
        return tf.argmax(q, axis=1).numpy()
