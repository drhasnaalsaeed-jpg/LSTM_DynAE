"""
lstm_dynae/autoencoder.py
==========================

LSTM Autoencoder (LSTM-AE) architecture, Section 3.1 of the manuscript.

Manuscript specification (verbatim, Section 3.1):

    "LSTM-Encoder: The LSTM encoder f_{phi_e}(x_i) has an LSTM-encoder
     input layer and two LSTM-encoder layers. The first LSTM has 64 units,
     while the second layer has 128 units. It performs mapping of data to
     the bottleneck layer. z_i = f_{phi_e}(x_i)."

    "LSTM-Decoder: The LSTM decoder g_{phi_d}(z_i) is a mirror of the
     encoder, with two LSTM layers. The first layer has 128 units, and the
     second layer has 64 units. x_hat_i = g_{phi_d}(z_i)."

Two details from the manuscript text/notation are encoded explicitly here
(classification: B -- direct consequence of the manuscript, not invented):

1. Notation in Section 3: x_i = {x_i^1, ..., x_i^T} and the decoder output
   is written as x_hat_i = {x_hat_i^T, x_hat_i^{T-1}, ..., x_hat_i^1}, i.e.
   the reconstruction is indexed in REVERSE temporal order relative to the
   input. This mirrors the classical LSTM sequence-to-sequence autoencoder
   trick (encoder reads x_1..x_T, decoder reconstructs x_T..x_1). We
   therefore train the decoder to reproduce the time-reversed input
   sequence (see `reverse_time` below), and reverse the prediction back
   before returning it to the caller so that all *external* tensors
   (encoder input, decoder output returned to users) are in natural
   chronological order.

2. The manuscript writes both the latent code and the reconstruction as
   living in R^d, which is the same symbol used for the raw feature
   dimensionality of a timestep. Read literally this would force the
   bottleneck size to equal the number of input channels, which
   contradicts the explicit "128 units" bottleneck stated in the same
   subsection. We resolve this apparent notational inconsistency
   (classification: C -- necessary implementation choice; the inconsistency
   itself is documented in the README's Implementation Audit) by using:
       latent_dim  = second encoder layer's unit count (128)
       feature_dim = number of channels of the raw input sequence (dataset
                     dependent, e.g. 2 for PenDigits, 14 for EEG Eye State)
   This is the only reading consistent with Fig. 1, Fig. 2, Table 1 and the
   explicit "64/128" unit counts, and is the standard construction used by
   LSTM sequence autoencoders (Section 3.1 cites Mobtahej et al. [20] for
   this architecture family).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model


@dataclass
class AutoencoderConfig:
    """Configuration for the LSTM-AE architecture.

    encoder_units / decoder_units default to the exact values given in the
    manuscript (Section 3.1): encoder [64, 128], decoder [128, 64].
    """

    input_timesteps: int
    input_features: int
    encoder_units: Tuple[int, int] = (64, 128)
    decoder_units: Tuple[int, int] = (128, 64)
    # Configurable implementation choice (classification C): the
    # manuscript does not state LSTM activation functions explicitly.
    # Keras/TensorFlow LSTM defaults (tanh activation, sigmoid recurrent
    # activation) are used, the standard convention assumed implicitly
    # whenever a paper says "LSTM layer" without further qualification.
    activation: str = "tanh"
    recurrent_activation: str = "sigmoid"
    # Configurable implementation choice (classification C): weight
    # initialization scheme is not stated in the manuscript. Keras
    # defaults (Glorot-uniform kernel, orthogonal recurrent kernel) are
    # used.
    kernel_initializer: str = "glorot_uniform"
    recurrent_initializer: str = "orthogonal"

    @property
    def latent_dim(self) -> int:
        return self.encoder_units[-1]


def reverse_time(x: tf.Tensor) -> tf.Tensor:
    """Reverse a (batch, T, features) tensor along the time axis.

    Used to implement the manuscript's x_hat_i = {x_hat_i^T, ..., x_hat_i^1}
    reversed-order reconstruction target (see module docstring, point 1).
    """
    return tf.reverse(x, axis=[1])


class LSTMEncoder(Model):
    """f_{phi_e}: X -> Z  (Section 3.1)."""

    def __init__(self, config: AutoencoderConfig, name: str = "lstm_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        u1, u2 = config.encoder_units
        self.lstm1 = layers.LSTM(
            u1,
            return_sequences=True,
            activation=config.activation,
            recurrent_activation=config.recurrent_activation,
            kernel_initializer=config.kernel_initializer,
            recurrent_initializer=config.recurrent_initializer,
            name="encoder_lstm_64",
        )
        self.lstm2 = layers.LSTM(
            u2,
            return_sequences=False,
            activation=config.activation,
            recurrent_activation=config.recurrent_activation,
            kernel_initializer=config.kernel_initializer,
            recurrent_initializer=config.recurrent_initializer,
            name="encoder_lstm_128_bottleneck",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Maps x: (batch, T, input_features) -> z: (batch, latent_dim)."""
        h = self.lstm1(x, training=training)
        z = self.lstm2(h, training=training)
        return z


class LSTMDecoder(Model):
    """g_{phi_d}: Z -> X  (Section 3.1), mirror of the encoder."""

    def __init__(self, config: AutoencoderConfig, name: str = "lstm_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        u1, u2 = config.decoder_units
        self.repeat = layers.RepeatVector(config.input_timesteps)
        self.lstm1 = layers.LSTM(
            u1,
            return_sequences=True,
            activation=config.activation,
            recurrent_activation=config.recurrent_activation,
            kernel_initializer=config.kernel_initializer,
            recurrent_initializer=config.recurrent_initializer,
            name="decoder_lstm_128",
        )
        self.lstm2 = layers.LSTM(
            u2,
            return_sequences=True,
            activation=config.activation,
            recurrent_activation=config.recurrent_activation,
            kernel_initializer=config.kernel_initializer,
            recurrent_initializer=config.recurrent_initializer,
            name="decoder_lstm_64",
        )
        self.output_proj = layers.TimeDistributed(
            layers.Dense(config.input_features, activation=None),
            name="decoder_output_projection",
        )

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Maps z: (batch, latent_dim) -> x_hat: (batch, T, input_features).

        Internally reconstructs in reverse time order (manuscript notation
        x_hat_i = {x_hat_i^T, ..., x_hat_i^1}) and reverses back to natural
        chronological order before returning, so callers always deal with
        forward-time tensors.
        """
        h = self.repeat(z)
        h = self.lstm1(h, training=training)
        h = self.lstm2(h, training=training)
        x_hat_reversed = self.output_proj(h)
        x_hat = reverse_time(x_hat_reversed)
        return x_hat


class LSTMAutoencoder(Model):
    """Full LSTM-AE: encoder + decoder, Eq. (3): min (1/n) sum ||x_i - x_hat_i||^2."""

    def __init__(self, config: AutoencoderConfig, name: str = "lstm_autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)

    def call(self, x: tf.Tensor, training: bool = False):
        z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return x_hat, z

    def encode(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.encoder(x, training=training)

    def decode(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.decoder(z, training=training)


def reconstruction_loss(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """Eq. (3): L1 = (1/n) * sum_i || x_i - x_hat_i ||^2 (mean squared error)."""
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2]))
