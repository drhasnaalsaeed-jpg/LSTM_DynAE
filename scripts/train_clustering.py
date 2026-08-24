#!/usr/bin/env python3
"""
scripts/train_clustering.py
==============================

Phase II: dynamic clustering fine-tuning (Section 3.3, Eq. 8-14, Fig. 3,
Algorithm 1 "Stage 2"). Loads pretrained weights produced by
scripts/pretrain.py, initializes centroids with K-means, then optimizes
L = L1 + L2 with mini-batch SGD+momentum until the dataset-level tau_p
(Eq. 11) drops below tau_stop, or max_iterations is reached (Section 3.3.3).

Usage
-----
    python scripts/train_clustering.py --config configs/lstm_dynae.yaml \
        --dataset PenDigits --max-iterations 20000

`clustering.max_iterations` is not stated numerically in the manuscript
(Algorithm 1 / Section 3.3.3 reference it symbolically only) and has no
default here: it must be supplied via configs/lstm_dynae.yaml or
--max-iterations before this script will run.

Stopping criterion: this script evaluates the manuscript-defined
DATASET-LEVEL tau_p = |S_bar| / N (Eq. 11) by periodically encoding the
full dataset and computing the conflicted-sample fraction over all N
samples -- not just the current mini-batch. See
lstm_dynae/model.py::compute_global_tau_p and
lstm_dynae/dynamic_loss.py::compute_global_tau_p for the implementation.
Per-mini-batch conflicted fractions are also logged (as `batch_tau_p`)
for fast per-step diagnostics, but never drive the stopping decision.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstm_dynae.datasets import DATASET_REGISTRY, load_dataset
from lstm_dynae.metrics import clustering_accuracy, normalized_mutual_information, timer
from lstm_dynae.model import LSTMDynAE, LSTMDynAEConfig
from lstm_dynae.preprocessing import PreprocessingConfig, augment_batch
from lstm_dynae.utils import set_global_seed, CSVLogger, save_weights, load_weights


def parse_args():
    p = argparse.ArgumentParser(description="Phase II: dynamic clustering fine-tuning for LSTM-DynAE")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--results-csv", type=str, default="results/clustering_results.csv")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Overrides config's clustering.max_iterations. Required (here or in "
        "the config) because the manuscript does not state this value numerically.",
    )
    p.add_argument(
        "--tau-p-check-interval",
        type=int,
        default=None,
        help="Overrides config's clustering.tau_p_check_interval "
        "(iterations between dataset-level tau_p evaluations; default: once per epoch-equivalent).",
    )
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg["seed"])

    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = cfg["clustering"]["max_iterations"]
    if not isinstance(max_iterations, int):
        raise ValueError(
            "clustering.max_iterations must be specified in the configuration. "
            "The exact numerical MaxItr is not explicitly reported in the "
            "LSTM-DynAE manuscript. Set clustering.max_iterations in "
            "configs/lstm_dynae.yaml, or pass --max-iterations explicitly."
        )

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
    print(f"[cluster] Loaded {args.dataset}: x.shape={x.shape}, K={ds_cfg['classes']}")

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
        max_iterations=max_iterations,
    )
    model = LSTMDynAE(model_cfg)

    ckpt_dir = Path(args.checkpoint_dir) / args.dataset
    ae_ckpt = ckpt_dir / "pretrained_autoencoder.weights.h5"
    if not ae_ckpt.exists():
        raise FileNotFoundError(
            f"No pretrained weights found at {ae_ckpt}. Run scripts/pretrain.py first "
            f"(Algorithm 1, Stage 1, must precede Stage 2)."
        )
    # build the model by running one forward pass before loading weights
    import tensorflow as tf

    _ = model.autoencoder(tf.convert_to_tensor(x[: min(8, x.shape[0])]))
    load_weights(model.autoencoder, str(ae_ckpt))
    print(f"[cluster] Loaded pretrained autoencoder weights from {ae_ckpt}")

    # Algorithm 1, line 8: initialize embedded centroids eta via K-means
    with timer() as t_init:
        model.init_centroids(x)
    print(f"[cluster] K-means centroid init done in {t_init['seconds']:.2f}s "
          f"(alpha_1={model.dynamic_loss_config.alpha_1:.4f}, "
          f"alpha_2={model.dynamic_loss_config.alpha_2:.4f})")

    logger = CSVLogger(
        f"{args.checkpoint_dir}/{args.dataset}/clustering_log.csv",
        fieldnames=["iteration", "loss", "l1", "l2", "batch_tau_p", "global_tau_p", "ACC", "NMI"],
    )

    n = x.shape[0]
    batch_size = cfg["clustering"]["batch_size"]
    x_tf = tf.convert_to_tensor(x)
    dataset = tf.data.Dataset.from_tensor_slices(x_tf).shuffle(n, seed=cfg["seed"]).repeat().batch(batch_size)
    iterator = iter(dataset)

    tau_stop = cfg["clustering"]["tau_stop"]  # Section 4.3: 1%

    tau_p_check_interval = args.tau_p_check_interval
    if tau_p_check_interval is None:
        tau_p_check_interval = cfg["clustering"].get("tau_p_check_interval")
    if tau_p_check_interval is None:
        # Default: once per epoch-equivalent. Configurable implementation
        # choice; see configs/lstm_dynae.yaml.
        tau_p_check_interval = max(1, math.ceil(n / batch_size))

    print(f"[cluster] Training until dataset-level tau_p < {tau_stop} (checked every "
          f"{tau_p_check_interval} iterations) or {max_iterations} iterations (Section 3.3.3).")

    global_tau_p = None
    with timer() as t_train:
        for it in range(1, max_iterations + 1):
            x_batch = next(iterator)
            if prep_cfg.augment:
                x_batch = augment_batch(x_batch, prep_cfg)
            step_out = model.clustering_train_step(x_batch)
            batch_tau_p = float(step_out["batch_tau_p"].numpy())

            check_now = (it % tau_p_check_interval == 0) or (it == max_iterations)
            if check_now:
                global_tau_p = model.compute_global_tau_p(x)

            if it % args.log_every == 0 or check_now:
                y_pred = model.predict_clusters(x)
                acc = clustering_accuracy(y, y_pred)
                nmi = normalized_mutual_information(y, y_pred)
                row = {
                    "iteration": it,
                    "loss": float(step_out["loss"].numpy()),
                    "l1": float(step_out["l1"].numpy()),
                    "l2": float(step_out["l2"].numpy()),
                    "batch_tau_p": batch_tau_p,
                    "global_tau_p": global_tau_p if check_now else "",
                    "ACC": acc,
                    "NMI": nmi,
                }
                logger.log(row)
                gtp_str = f"{global_tau_p:.4f}" if check_now else "n/a"
                print(f"[cluster] iter {it}/{max_iterations}  loss={row['loss']:.4f}  "
                      f"batch_tau_p={batch_tau_p:.4f}  global_tau_p={gtp_str}  "
                      f"ACC={acc:.4f}  NMI={nmi:.4f}")

            # Section 3.3.3 stopping criterion, evaluated at the DATASET
            # level (Eq. 11 defines N as the full dataset size):
            if check_now and global_tau_p < tau_stop:
                print(f"[cluster] Stopping: global_tau_p={global_tau_p:.4f} < tau_stop={tau_stop}")
                break

    final_y_pred = model.predict_clusters(x)
    final_acc = clustering_accuracy(y, final_y_pred)
    final_nmi = normalized_mutual_information(y, final_y_pred)
    print(f"[cluster] FINAL  ACC={final_acc:.4f}  NMI={final_nmi:.4f}  "
          f"elapsed={t_train['seconds']:.1f}s")

    save_weights(model.autoencoder, str(ckpt_dir / "clustered_autoencoder.weights.h5"))
    np.save(ckpt_dir / "final_centroids.npy", model.centroids.numpy())

    # Append to the results CSV (results/clustering_results.csv), user requirement F.
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with open(results_path, "a") as f:
        if write_header:
            f.write("dataset,method,ACC,NMI\n")
        f.write(f"{args.dataset},LSTM-DynAE (this run),{final_acc:.4f},{final_nmi:.4f}\n")
    print(f"[cluster] Appended this run's result to {results_path}")


if __name__ == "__main__":
    main()
