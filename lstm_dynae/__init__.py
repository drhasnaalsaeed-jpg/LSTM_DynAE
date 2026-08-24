"""
LSTM-DynAE
==========

Implementation of LSTM-DynAE for the paper:

    Hasna AlSaeed, Riadh Ksantini, Faisal AlKhateeb.
    "Sequential Deep Embedded Clustering with Dynamic Autoencoder."
    Multimedia Tools and Applications (Manuscript MTAP-D-25-01346).

This package is built strictly from the manuscript text, equations and
tables. Wherever the manuscript does not fully specify an implementation
detail, the corresponding code documents this and, where feasible,
exposes the choice as a configuration parameter rather than hard-coding
it.

See ``README.md`` for the full implementation audit against the manuscript.
"""

__version__ = "0.1.0"

__all__ = [
    "autoencoder",
    "critic",
    "acai",
    "dynamic_loss",
    "clustering",
    "preprocessing",
    "datasets",
    "metrics",
    "utils",
    "model",
]
