#!/usr/bin/env python3
"""
scripts/evaluate.py
======================

Standalone evaluation: loads a trained (clustered) LSTM-DynAE checkpoint
and reports ACC (Eq. 15) and NMI (Eq. 16) on the given dataset, without
touching training code (quality requirement: "keep training and
evaluation separate").

Usage
-----
    python scripts/evaluate.py --config configs/lstm_dynae.yaml --dataset PenDigits
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstm_dynae.datasets import DATASET_REGISTRY, load_dataset
from lstm_dynae.metrics import clustering_accuracy, normalized_mutual_information, timer
from lstm_dynae.model import LSTMDynAE, LSTMDynAEConfig
from lstm_dynae.preprocessing import PreprocessingConfig
from lstm_dynae.utils import set_global_seed, load_weights


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained LSTM-DynAE checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_global_seed(cfg["seed"])

    ds_cfg = cfg["datasets"][args.dataset]
    prep_cfg = PreprocessingConfig(
        normalization=cfg["preprocessing"]["normalization"],
        sequence_length=cfg["preprocessing"]["sequence_length"],
        stride=cfg["preprocessing"]["stride"],
        augment=False,  # no augmentation at evaluation time
    )
    x, y, spec = load_dataset(args.data_dir, args.dataset, prep_cfg)

    critic_cfg = cfg.get("critic", {})

    model_cfg = LSTMDynAEConfig(
        input_timesteps=x.shape[1],
        input_features=x.shape[2],
        num_clusters=ds_cfg["classes"],
        seed=cfg["seed"],
        encoder_units=tuple(cfg["model"]["encoder_units"]),
        decoder_units=tuple(cfg["model"]["decoder_units"]),
        critic_recurrent_units=tuple(critic_cfg.get("recurrent_units", (64, 32))),
        critic_output_units=critic_cfg.get("output_units", 1),
        student_t_alpha=cfg["clustering"]["student_t_alpha"],
        kappa=cfg["clustering"]["kappa"],
    )
    model = LSTMDynAE(model_cfg)

    import tensorflow as tf

    ckpt_dir = Path(args.checkpoint_dir) / args.dataset
    ae_ckpt = ckpt_dir / "clustered_autoencoder.weights.h5"
    centroids_path = ckpt_dir / "final_centroids.npy"
    if not ae_ckpt.exists() or not centroids_path.exists():
        raise FileNotFoundError(
            f"Missing clustered checkpoint/centroids in {ckpt_dir}. "
            f"Run scripts/train_clustering.py first."
        )

    _ = model.autoencoder(tf.convert_to_tensor(x[: min(8, x.shape[0])]))
    load_weights(model.autoencoder, str(ae_ckpt))
    model.centroids = tf.Variable(np.load(centroids_path), dtype=tf.float32, name="cluster_centroids")

    with timer() as t:
        y_pred = model.predict_clusters(x)
    acc = clustering_accuracy(y, y_pred)
    nmi = normalized_mutual_information(y, y_pred)

    print(f"[evaluate] {args.dataset}: ACC={acc:.4f}  NMI={nmi:.4f}  "
          f"inference_time={t['seconds']:.4f}s  n_samples={x.shape[0]}")


if __name__ == "__main__":
    main()
