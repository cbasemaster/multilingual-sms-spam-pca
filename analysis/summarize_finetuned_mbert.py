"""Summarize fine-tuned mBERT and compare matched seed-42 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from sklearn.metrics import matthews_corrcoef


def cluster_compare(reference: pd.DataFrame, comparison: pd.DataFrame, repetitions: int, seed: int):
    joined = reference[["group_id", "language_column", "label", "prediction"]].merge(
        comparison[["group_id", "language_column", "label", "prediction"]],
        on=["group_id", "language_column", "label"],
        suffixes=("_reference", "_comparison"),
        validate="one_to_one",
    )
    labels = joined["label"].to_numpy()
    reference_correct = joined["prediction_reference"].to_numpy() == labels
    comparison_correct = joined["prediction_comparison"].to_numpy() == labels
    reference_only = int(np.sum(reference_correct & ~comparison_correct))
    comparison_only = int(np.sum(~reference_correct & comparison_correct))
    discordant = reference_only + comparison_only
    group_ids = joined["group_id"].unique()
    by_group = {group_id: indices for group_id, indices in joined.groupby("group_id").indices.items()}
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled_groups = rng.choice(group_ids, len(group_ids), replace=True)
        indices = np.concatenate([by_group[group_id] for group_id in sampled_groups])
        sampled = joined.iloc[indices]
        sampled_labels = sampled["label"].to_numpy()
        differences[repetition] = matthews_corrcoef(
            sampled_labels, sampled["prediction_reference"]
        ) - matthews_corrcoef(sampled_labels, sampled["prediction_comparison"])
    observed = matthews_corrcoef(labels, joined["prediction_reference"]) - matthews_corrcoef(
        labels, joined["prediction_comparison"]
    )
    return {
        "matched_rows": len(joined),
        "groups": len(group_ids),
        "reference_only_correct": reference_only,
        "comparison_only_correct": comparison_only,
        "exact_mcnemar_p": 1.0 if discordant == 0 else binomtest(reference_only, discordant).pvalue,
        "reference_minus_comparison_mcc": observed,
        "cluster_bootstrap_ci95_low": np.quantile(differences, 0.025),
        "cluster_bootstrap_ci95_high": np.quantile(differences, 0.975),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    run_dir = args.audit_dir / "finetuned_mbert_grouped"
    runs = pd.read_csv(run_dir / "finetuned_mbert_runs.csv")
    metrics = ["accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc", "roc_auc", "brier", "ece_15"]
    summary = {}
    for metric in metrics:
        values = runs[metric].to_numpy()
        mean = values.mean()
        std = values.std(ddof=1)
        half = t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))
        summary.update(
            {
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": mean - half,
                f"{metric}_ci95_high": mean + half,
            }
        )
    for metric in ("train_seconds", "inference_ms_per_message", "peak_gpu_memory_mb"):
        summary[f"{metric}_mean"] = runs[metric].mean()
        summary[f"{metric}_std"] = runs[metric].std(ddof=1)
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(run_dir / "finetuned_mbert_summary.csv", index=False)

    finetuned = pd.read_csv(run_dir / "finetuned_mbert_prediction_seed42.csv")
    fusion = pd.read_csv(
        args.audit_dir / "grouped_embedding_fusion" / "prediction_standardized_concat_seed42.csv"
    )
    sparse = pd.read_csv(args.audit_dir / "grouped_baseline_seed42_predictions.csv")
    sparse = sparse[sparse["model"] == "Char-TFIDF + LR"]
    comparisons = []
    for name, frame in (("Standardized concat", fusion), ("Char-TFIDF + LR", sparse)):
        comparisons.append(
            {
                "reference": "Fine-tuned mBERT",
                "comparison": name,
                **cluster_compare(
                    finetuned,
                    frame,
                    args.bootstrap_repetitions,
                    seed=20260908,
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(run_dir / "finetuned_mbert_seed42_comparisons.csv", index=False)
    print(summary_frame.round(4).to_string(index=False))
    print(comparison_frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
