"""
lstm_dynae/critic.py
=====================

Critic network C(.) used by the Adversarially Constrained Autoencoder
Interpolation (ACAI) regularizer during pretraining (Section 3.2, Fig. 2).

Manuscript specification:
    "The critic network, represented as C with learnable parameters phi_C,
     is trained to predict the interpolation coefficient tau_1 from input
     x_hat_alpha."  (Section 3.2, around Eq. 6-7)

What the manuscript specifies explicitly (classification A):
    - The critic consumes a reconstructed/interpolated *sequence*
      (x_hat_alpha, same shape as the data: (T, features)) and outputs a
      SCALAR prediction of the interpolation coefficient tau_1 in [0, 1].
    - The critic is optimized with the squared-error losses of Eq. (7).

What the manuscript does NOT specify (classification D -- configurable
implementation choice not explicitly specified in the manuscript):
    - The exact critic architecture (number of layers, layer sizes,
      activation functions, whether it is recurrent, convolutional or
      feed-forward after pooling).

The original ACAI paper (Berthelot et al., cited as [1]/[2] in this
manuscript) uses a convolutional critic mirroring the encoder, because ACAI
was designed for image data. Since LSTM-DynAE operates on sequential data,
the most direct analogue -- and the one requiring the fewest un-stated
assumptions -- is a small recurrent critic that mirrors the *encoder's*
recurrent structure and terminates in a scalar output. This is the
configurable implementation choice used in this implementation
(exposed via the `critic:` section of configs/lstm_dynae.yaml); a
different, equally defensible choice (e.g. a Conv1D critic, or an MLP
critic over a flattened sequence) would also be consistent with the
manuscript text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model


@dataclass
class CriticConfig:
    input_timesteps: int
    input_features: int
    # Configurable implementation choice not explicitly specified in the
    # manuscript. Default mirrors the encoder's own two-layer recurrent
    # pattern for architectural symmetry with the rest of the model; see
    # the `critic:` section of configs/lstm_dynae.yaml to change it.
    hidden_units: Tuple[int, int] = (64, 32)
    # Output activation: sigmoid, since the regression target tau_1 (and
    # the implicit target 0 for the realism term of Eq. 7) both lie in
    # [0, 1]. Configurable implementation choice, not stated in the
    # manuscript.
    output_activation: str = "sigmoid"


class Critic(Model):
    """C: sequence -> scalar interpolation-coefficient prediction."""

    def __init__(self, config: CriticConfig, name: str = "critic", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        u1, u2 = config.hidden_units
        self.lstm1 = layers.LSTM(u1, return_sequences=True, name="critic_lstm_1")
        self.lstm2 = layers.LSTM(u2, return_sequences=False, name="critic_lstm_2")
        self.out = layers.Dense(1, activation=config.output_activation, name="critic_output")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """x: (batch, T, features) -> (batch,) scalar prediction of tau_1."""
        h = self.lstm1(x, training=training)
        h = self.lstm2(h, training=training)
        out = self.out(h, training=training)
        return tf.squeeze(out, axis=-1)
