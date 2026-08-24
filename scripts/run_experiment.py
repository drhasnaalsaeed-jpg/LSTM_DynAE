#!/usr/bin/env python3
"""
scripts/run_experiment.py
============================

Runs the full LSTM-DynAE pipeline for a SINGLE dataset:
    Phase I  (scripts/pretrain.py)
    Phase II (scripts/train_clustering.py)
    Evaluation (scripts/evaluate.py)

This is a thin orchestration wrapper: it shells out to the three
underlying scripts so that each phase remains independently runnable and
independently inspectable (per "keep training and evaluation separate",
quality requirement L).

Usage
-----
    python scripts/run_experiment.py --config configs/lstm_dynae.yaml --dataset PenDigits
"""

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Run the full LSTM-DynAE pipeline for one dataset")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--results-csv", type=str, default="results/clustering_results.csv")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--skip-pretrain", action="store_true", help="Reuse existing pretrained checkpoint.")
    return p.parse_args()


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    if not args.skip_pretrain:
        run([
            sys.executable, str(THIS_DIR / "pretrain.py"),
            "--config", args.config,
            "--dataset", args.dataset,
            "--data-dir", args.data_dir,
            "--checkpoint-dir", args.checkpoint_dir,
        ])

    cluster_cmd = [
        sys.executable, str(THIS_DIR / "train_clustering.py"),
        "--config", args.config,
        "--dataset", args.dataset,
        "--data-dir", args.data_dir,
        "--checkpoint-dir", args.checkpoint_dir,
        "--results-csv", args.results_csv,
    ]
    if args.max_iterations is not None:
        cluster_cmd += ["--max-iterations", str(args.max_iterations)]
    run(cluster_cmd)

    run([
        sys.executable, str(THIS_DIR / "evaluate.py"),
        "--config", args.config,
        "--dataset", args.dataset,
        "--data-dir", args.data_dir,
        "--checkpoint-dir", args.checkpoint_dir,
    ])


if __name__ == "__main__":
    main()
