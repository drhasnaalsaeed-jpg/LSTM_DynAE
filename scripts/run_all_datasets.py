#!/usr/bin/env python3
"""
scripts/run_all_datasets.py
==============================

Runs scripts/run_experiment.py for all five benchmark datasets in
Table 1 (PenDigits, Handwriting, RacketSports, EEGEyeState,
OzoneLevelDetection), producing one row per dataset in
results/clustering_results.csv.

Usage
-----
    python scripts/run_all_datasets.py --config configs/lstm_dynae.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lstm_dynae.datasets import DATASET_REGISTRY

THIS_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Run LSTM-DynAE on all five benchmark datasets")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--results-csv", type=str, default="results/clustering_results.csv")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        help="Subset of dataset keys to run (default: all five).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    failures = []
    for key in args.datasets:
        print(f"\n===== Running LSTM-DynAE on {key} =====")
        cmd = [
            sys.executable, str(THIS_DIR / "run_experiment.py"),
            "--config", args.config,
            "--dataset", key,
            "--data-dir", args.data_dir,
            "--checkpoint-dir", args.checkpoint_dir,
            "--results-csv", args.results_csv,
        ]
        if args.max_iterations is not None:
            cmd += ["--max-iterations", str(args.max_iterations)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[run_all_datasets] FAILED on {key}: {e}")
            failures.append(key)

    if failures:
        print(f"\n[run_all_datasets] Completed with failures on: {failures}")
        sys.exit(1)
    print("\n[run_all_datasets] All datasets completed successfully.")


if __name__ == "__main__":
    main()
