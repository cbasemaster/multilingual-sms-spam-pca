"""Summarize the grouped Qwen fusion extension with paired inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import matthews_corrcoef


def cluster_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    joined = reference[["row_index", "group_id", "label", "prediction"]].merge(
        comparison[["row_index", "prediction"]],
        on="row_index",
        suffixes=("_reference", "_comparison"),
        validate="one_to_one",
    )
    groups = joined["group_id"].unique()

    def confusion_counts(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
        labels = frame["label"].to_numpy(dtype=np.int8)
        predicted = frame[prediction_column].to_numpy(dtype=np.int8)
        return np.array(
            [
                np.sum((labels == 0) & (predicted == 0)),
                np.sum((labels == 0) & (predicted == 1)),
                np.sum((labels == 1) & (predicted == 0)),
                np.sum((labels == 1) & (predicted == 1)),
            ],
            dtype=np.float64,
        )

    def mcc_from_counts(counts: np.ndarray) -> float:
        tn, fp, fn, tp = counts
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator

    reference_counts = np.stack(
        [confusion_counts(frame, "prediction_reference") for _, frame in joined.groupby("group_id")]
    )
    comparison_counts = np.stack(
        [confusion_counts(frame, "prediction_comparison") for _, frame in joined.groupby("group_id")]
    )
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)))
        differences[index] = mcc_from_counts(weights @ reference_counts) - mcc_from_counts(
            weights @ comparison_counts
        )
    observed = mcc_from_counts(reference_counts.sum(axis=0)) - mcc_from_counts(
        comparison_counts.sum(axis=0)
    )
    return {
        "matched_samples": len(joined),
        "groups": len(groups),
        "observed_mcc_difference": observed,
        "ci95_low": np.quantile(differences, 0.025),
        "ci95_high": np.quantile(differences, 0.975),
        "probability_difference_gt_zero": np.mean(differences > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    runs = pd.read_csv(args.new_dir / "qwen_fusion_runs.csv")
    predictions = pd.read_csv(args.new_dir / "qwen_fusion_predictions.csv")
    old_predictions = pd.read_csv(args.old_dir / "fusion_predictions.csv")
    seeds = sorted(runs["seed"].unique())

    ci_rows = []
    for configuration, frame in runs.groupby("configuration"):
        row: dict[str, float | int | str] = {
            "configuration": configuration,
            "n_seeds": len(frame),
        }
        for metric in ("accuracy", "spam_f1", "macro_f1", "mcc", "roc_auc", "ece_15"):
            values = frame[metric].to_numpy()
            half_width = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_ci95_low"] = values.mean() - half_width
            row[f"{metric}_ci95_high"] = values.mean() + half_width
        for metric in ("train_seconds", "inference_ms_per_message"):
            values = frame[metric].to_numpy()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        row["stored_embedding_mb"] = frame["stored_embedding_mb"].iloc[0]
        row["trainable_parameters"] = int(frame["trainable_parameters"].iloc[0])
        row["peak_gpu_memory_mb"] = frame["peak_gpu_memory_mb"].max()
        ci_rows.append(row)
    confidence_intervals = pd.DataFrame(ci_rows).sort_values("mcc_mean", ascending=False)

    reference_name = "mBERT+Qwen Concat+PCA-768"
    comparisons = [
        "Qwen2.5-7B-Instruct+PCA-768 (NF4 extraction)",
        "mBERT",
    ]
    paired_rows = []
    reference_runs = runs[runs["configuration"] == reference_name].set_index("seed")
    for comparison_name in comparisons:
        comparison_runs = runs[runs["configuration"] == comparison_name].set_index("seed")
        differences = reference_runs.loc[seeds, "mcc"] - comparison_runs.loc[seeds, "mcc"]
        half_width = t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences))
        paired_rows.append(
            {
                "reference": reference_name,
                "comparison": comparison_name,
                "mean_paired_mcc_difference": differences.mean(),
                "ci95_low": differences.mean() - half_width,
                "ci95_high": differences.mean() + half_width,
            }
        )
    paired = pd.DataFrame(paired_rows)

    rng = np.random.default_rng(20260914)
    seed = 42
    reference = predictions[
        (predictions["configuration"] == reference_name) & (predictions["seed"] == seed)
    ].sort_values("row_index")
    bootstrap_rows = []
    for comparison_name in comparisons:
        comparison = predictions[
            (predictions["configuration"] == comparison_name) & (predictions["seed"] == seed)
        ].sort_values("row_index")
        bootstrap_rows.append(
            {
                "seed": seed,
                "reference": reference_name,
                "comparison": comparison_name,
                **cluster_bootstrap(reference, comparison, args.bootstrap_repetitions, rng),
            }
        )
    for comparison_name in ("Standardized concat", "Concat+PCA-256"):
        comparison = old_predictions[
            (old_predictions["configuration"] == comparison_name)
            & (old_predictions["seed"] == seed)
        ].sort_values("row_index")
        bootstrap_rows.append(
            {
                "seed": seed,
                "reference": reference_name,
                "comparison": comparison_name,
                **cluster_bootstrap(reference, comparison, args.bootstrap_repetitions, rng),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    confidence_intervals.to_csv(args.new_dir / "qwen_fusion_confidence_intervals.csv", index=False)
    paired.to_csv(args.new_dir / "qwen_fusion_paired_seed_intervals.csv", index=False)
    bootstrap.to_csv(args.new_dir / "qwen_fusion_group_bootstrap_seed42.csv", index=False)
    print(confidence_intervals.round(4).to_string(index=False))
    print("\nPaired five-seed MCC differences\n", paired.round(4).to_string(index=False))
    print("\nSeed-42 source-group bootstrap\n", bootstrap.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
