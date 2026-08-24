#!/usr/bin/env python3
"""
scripts/pretrain.py
======================

Phase I: ACAI-regularized LSTM autoencoder pretraining (Section 3.2,
Eq. 4-7, Fig. 2, Algorithm 1 "Stage 1").

Usage
-----
    python scripts/pretrain.py --config configs/lstm_dynae.yaml --dataset PenDigits

Saves pretrained autoencoder + critic weights to
    checkpoints/<dataset>/pretrained_autoencoder.weights.h5
    checkpoints/<dataset>/pretrained_critic.weights.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstm_dynae.datasets import DATASET_REGISTRY, load_dataset
from lstm_dynae.model import LSTMDynAE, LSTMDynAEConfig
from lstm_dynae.preprocessing import PreprocessingConfig
from lstm_dynae.preprocessing import augment_batch
from lstm_dynae.utils import set_global_seed, CSVLogger, save_weights


def parse_args():
    p = argparse.ArgumentParser(description="Phase I: ACAI pretraining for LSTM-DynAE")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-every", type=int, default=100)
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
        augment=cfg["preprocessing"]["augment"],
        max_rescale_delta=cfg["preprocessing"]["max_rescale_delta"],
        max_rotation_rad=cfg["preprocessing"]["max_rotation_rad"],
    )

    x, y, spec = load_dataset(args.data_dir, args.dataset, prep_cfg)
    print(f"[pretrain] Loaded {args.dataset}: x.shape={x.shape} (expected N,T,{spec.dimensions})")

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
        lambda_acai=cfg["pretraining"]["lambda_acai"],
        pretrain_iterations=cfg["pretraining"]["iterations"],
        pretrain_lr=cfg["pretraining"]["learning_rate"],
        pretrain_beta_1=cfg["pretraining"]["first_moment_decay"],
        pretrain_beta_2=cfg["pretraining"]["second_moment_decay"],
        clustering_lr=cfg["clustering"]["learning_rate"],
        clustering_momentum=cfg["clustering"]["momentum"],
        batch_size=cfg["clustering"]["batch_size"],
        student_t_alpha=cfg["clustering"]["student_t_alpha"],
        kappa=cfg["clustering"]["kappa"],
        tau_stop=cfg["clustering"]["tau_stop"],
        neighbor_pool=cfg["clustering"]["neighbor_pool"],
    )
    model = LSTMDynAE(model_cfg)

    import tensorflow as tf

    n = x.shape[0]
    batch_size = cfg["clustering"]["batch_size"]  # manuscript reuses batch_size=256 (Table 2)
    dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(n, seed=cfg["seed"]).repeat().batch(batch_size)
    iterator = iter(dataset)

    logger = CSVLogger(
        f"{args.checkpoint_dir}/{args.dataset}/pretrain_log.csv",
        fieldnames=["iteration", "L_fg", "L_C", "recon"],
    )

    iterations = cfg["pretraining"]["iterations"]  # 1.3e5, Eq. Section 3.2/4.3
    print(f"[pretrain] Running {iterations} iterations (manuscript: 1.3 x 10^5)...")
    for it in range(1, iterations + 1):
        x_batch = next(iterator)
        if prep_cfg.augment:
            x_batch = augment_batch(x_batch, prep_cfg)
        losses = model.pretrain_step(x_batch)

        if it % args.log_every == 0 or it == iterations:
            row = {
                "iteration": it,
                "L_fg": float(losses["L_fg"].numpy()),
                "L_C": float(losses["L_C"].numpy()),
                "recon": float(losses["recon"].numpy()),
            }
            logger.log(row)
            print(f"[pretrain] iter {it}/{iterations}  L_fg={row['L_fg']:.6f}  "
                  f"L_C={row['L_C']:.6f}  recon={row['recon']:.6f}")

    ckpt_dir = Path(args.checkpoint_dir) / args.dataset
    save_weights(model.autoencoder, str(ckpt_dir / "pretrained_autoencoder.weights.h5"))
    save_weights(model.critic, str(ckpt_dir / "pretrained_critic.weights.h5"))
    print(f"[pretrain] Saved pretrained weights to {ckpt_dir}")


if __name__ == "__main__":
    main()
