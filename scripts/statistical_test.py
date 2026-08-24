#!/usr/bin/env python3
"""
scripts/statistical_test.py
==============================

Reproduces the statistical significance analysis described in the
manuscript (Section 4.9):

    "A Wilcoxon signed-rank test was applied to the paired ACC and NMI
     scores produced by LSTM-DynAE and all baselines across the five
     datasets. The test yielded p-values below the 0.05 significance
     threshold for every comparison..."

For each baseline, this script performs a ONE-SIDED Wilcoxon signed-rank
test with directional alternative "LSTM-DynAE > baseline", separately for
ACC and NMI, using the five paired dataset-level observations (Table 3).

Because n=5 is small, the exact permutation null distribution of the
Wilcoxon signed-rank statistic is used (not the large-sample normal
approximation). When every one of the 5 paired differences has the SAME
sign (all positive, i.e. LSTM-DynAE wins on every dataset), the exact
one-sided p-value has the closed form:

    W+ = sum(1..5) = 15,  W- = 0,  p = (1/2)^5 = 1/32 = 0.03125

This closed form is a MATHEMATICAL CONSEQUENCE of the assumption "all 5
differences share the same sign" (classification B), not an assumption
this script hard-codes: the script always recomputes W+, W-, and the
exact p-value directly from whatever is in the input CSV, and will report
a DIFFERENT value (and a warning) if the signs are not unanimous, if
there are zero differences (ties at zero, which the Wilcoxon signed-rank
test's standard formulation cannot handle), or if fewer/more than 5 paired
observations are available for a given (method, metric) comparison.

The manuscript's own Section 4.9 (current revision) adds an important
scientific caveat that this script's output should be read alongside, not
in place of: "Given the limited number of benchmark datasets (N = 5),
these statistical findings are interpreted alongside the dataset-level
performance results rather than as standalone evidence of general
superiority." The manuscript also specifies the significance level
explicitly (Section 4.2): alpha = 0.05, and cites the test itself as
Wilcoxon (1945), "Individual comparisons by ranking methods" -- reference
[26] in the manuscript's bibliography.

Usage
-----
    python scripts/statistical_test.py --results results/clustering_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def exact_one_sided_p_all_same_sign(n: int) -> float:
    """Closed-form exact one-sided p-value when all n paired differences
    share the same sign: p = (1/2)^n (each dataset independently has a
    50% chance, under H0, of favoring either method)."""
    return (0.5) ** n


def compute_w_stats(diffs: np.ndarray):
    """Manually computes W+ and W- (sum of ranks of positive / negative
    differences) for reporting, independent of the p-value computation
    library used below.
    """
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0.0, 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").values
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    return w_plus, w_minus


def run_one_sided_test(lstm_scores: np.ndarray, baseline_scores: np.ndarray, metric_name: str, baseline_name: str):
    """LSTM-DynAE > baseline, one-sided Wilcoxon signed-rank test."""
    diffs = lstm_scores - baseline_scores
    n = len(diffs)

    warnings = []
    if n != 5:
        warnings.append(
            f"Expected exactly 5 paired dataset-level observations (manuscript uses five "
            f"benchmark datasets) but found {n} for {baseline_name}/{metric_name}."
        )
    n_zero = int(np.sum(diffs == 0))
    if n_zero > 0:
        warnings.append(
            f"{n_zero} of {n} paired differences are exactly zero for {baseline_name}/{metric_name}; "
            f"the Wilcoxon signed-rank test's standard formulation assumes no ties at zero. "
            f"Results below drop these before ranking (conservative)."
        )

    w_plus, w_minus = compute_w_stats(diffs)

    nonzero_diffs = diffs[diffs != 0]
    all_same_sign = bool(np.all(nonzero_diffs > 0) or np.all(nonzero_diffs < 0)) and nonzero_diffs.size > 0

    p_value = None
    method_used = None
    if nonzero_diffs.size == 0:
        warnings.append(f"All differences are zero for {baseline_name}/{metric_name}; test is undefined.")
    else:
        try:
            # scipy's exact method for the one-sided alternative "greater"
            # (LSTM-DynAE scores stochastically greater than baseline).
            stat, p_value = wilcoxon(diffs, alternative="greater", method="exact", zero_method="wilcox")
            method_used = "scipy.stats.wilcoxon(method='exact')"
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.append(f"scipy exact Wilcoxon failed ({exc}); falling back to closed-form same-sign formula "
                             f"if applicable.")
            if all_same_sign:
                p_value = exact_one_sided_p_all_same_sign(nonzero_diffs.size)
                method_used = "closed-form (1/2)^n, all differences share the same sign"

    return {
        "baseline": baseline_name,
        "metric": metric_name,
        "n": n,
        "W_plus": w_plus,
        "W_minus": w_minus,
        "all_same_sign": all_same_sign,
        "p_value": p_value,
        "method": method_used,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser(description="One-sided Wilcoxon signed-rank test, LSTM-DynAE vs. baselines")
    ap.add_argument("--results", type=str, default="results/clustering_results.csv")
    ap.add_argument("--lstm-method-name", type=str, default="LSTM-DynAE")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"ERROR: results file not found at {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    required_cols = {"dataset", "method", "ACC", "NMI"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: {path} must contain columns {required_cols}, found {set(df.columns)}", file=sys.stderr)
        sys.exit(1)

    lstm_df = df[df["method"] == args.lstm_method_name].set_index("dataset")
    if lstm_df.empty:
        print(f"ERROR: no rows found for method == '{args.lstm_method_name}' in {path}", file=sys.stderr)
        sys.exit(1)

    baselines = sorted(set(df["method"]) - {args.lstm_method_name})
    if not baselines:
        print(f"ERROR: no baseline methods found in {path} besides '{args.lstm_method_name}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {path}. LSTM-DynAE rows: {len(lstm_df)}. "
          f"Baselines found: {baselines}\n")

    any_not_significant = False
    for baseline in baselines:
        base_df = df[df["method"] == baseline].set_index("dataset")
        common_datasets = sorted(set(lstm_df.index) & set(base_df.index))
        if not common_datasets:
            print(f"[{baseline}] SKIPPED: no overlapping datasets with LSTM-DynAE rows.\n")
            continue

        for metric in ("ACC", "NMI"):
            lstm_scores = lstm_df.loc[common_datasets, metric].values.astype(float)
            base_scores = base_df.loc[common_datasets, metric].values.astype(float)

            result = run_one_sided_test(lstm_scores, base_scores, metric, baseline)

            print(f"[{baseline} | {metric}]  n={result['n']}  "
                  f"W+={result['W_plus']:.1f}  W-={result['W_minus']:.1f}  "
                  f"all_differences_same_sign={result['all_same_sign']}")
            if result["p_value"] is not None:
                sig = "SIGNIFICANT" if result["p_value"] < args.alpha else "NOT significant"
                print(f"    one-sided p-value (H1: LSTM-DynAE > {baseline}) = {result['p_value']:.6f} "
                      f"[{result['method']}] -> {sig} at alpha={args.alpha}")
                if result["p_value"] >= args.alpha:
                    any_not_significant = True
            else:
                print("    p-value could not be computed (see warnings below).")
                any_not_significant = True
            for w in result["warnings"]:
                print(f"    WARNING: {w}")
            print()

    if any_not_significant:
        print("NOTE: not every comparison reached the manuscript's reported significance / exact "
              "p=0.03125 pattern -- this reflects whatever is actually in the results CSV. Do not "
              "assume p=0.03125 unless the CSV values genuinely show LSTM-DynAE winning on all 5 "
              "datasets for that (baseline, metric) pair, per the user's instruction: "
              "'Do not automatically claim p=0.03125 if the CSV values do not support it.'")

    print("\nReminder (manuscript Section 4.9): with only N=5 paired datasets, these significance "
          "results should be interpreted alongside the dataset-level ACC/NMI numbers themselves, "
          "not as standalone evidence of general superiority.")


if __name__ == "__main__":
    main()
