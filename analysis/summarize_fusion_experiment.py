"""Produce paired, cluster-aware statistics for the grouped fusion experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


METRICS = ("accuracy", "spam_f1", "macro_f1", "mcc")


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "spam_f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    runs = pd.read_csv(args.experiment_dir / "fusion_runs.csv")
    predictions = pd.read_csv(args.experiment_dir / "fusion_predictions.csv")

    ci_rows = []
    for configuration, frame in runs.groupby("configuration"):
        row = {"configuration": configuration, "n_seeds": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy()
            mean = values.mean()
            half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": values.std(ddof=1),
                    f"{metric}_ci95_low": mean - half,
                    f"{metric}_ci95_high": mean + half,
                }
            )
        ci_rows.append(row)
    confidence_intervals = pd.DataFrame(ci_rows).sort_values("mcc_mean", ascending=False)

    language_rows = []
    for (configuration, seed, language), frame in predictions.groupby(
        ["configuration", "seed", "language_column"]
    ):
        language_rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "language_column": language,
                "n": len(frame),
                **score(frame["label"].to_numpy(), frame["prediction"].to_numpy()),
            }
        )
    language_runs = pd.DataFrame(language_rows)
    language_summary = (
        language_runs.groupby(["configuration", "language_column"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    language_summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in language_summary.columns
    ]

    reference_name = str(confidence_intervals.iloc[0]["configuration"])
    mcnemar_rows = []
    for seed in sorted(predictions["seed"].unique()):
        reference = predictions[
            (predictions["configuration"] == reference_name) & (predictions["seed"] == seed)
        ].sort_values("row_index")
        for comparison_name in sorted(predictions["configuration"].unique()):
            if comparison_name == reference_name:
                continue
            comparison = predictions[
                (predictions["configuration"] == comparison_name)
                & (predictions["seed"] == seed)
            ].sort_values("row_index")
            labels = reference["label"].to_numpy()
            reference_correct = reference["prediction"].to_numpy() == labels
            comparison_correct = comparison["prediction"].to_numpy() == labels
            reference_only = int(np.sum(reference_correct & ~comparison_correct))
            comparison_only = int(np.sum(~reference_correct & comparison_correct))
            discordant = reference_only + comparison_only
            p_value = (
                1.0
                if discordant == 0
                else binomtest(reference_only, discordant, p=0.5).pvalue
            )
            mcnemar_rows.append(
                {
                    "seed": seed,
                    "reference": reference_name,
                    "comparison": comparison_name,
                    "reference_only_correct": reference_only,
                    "comparison_only_correct": comparison_only,
                    "exact_mcnemar_p": p_value,
                }
            )
    mcnemar = pd.DataFrame(mcnemar_rows)

    rng = np.random.default_rng(20260908)
    seed = 42
    reference = predictions[
        (predictions["configuration"] == reference_name) & (predictions["seed"] == seed)
    ].sort_values("row_index")
    bootstrap_rows = []
    for comparison_name in ("Concat+PCA-256", "DistilBERT", "Raw concat"):
        comparison = predictions[
            (predictions["configuration"] == comparison_name) & (predictions["seed"] == seed)
        ].sort_values("row_index")
        joined = reference[["row_index", "group_id", "label", "prediction"]].merge(
            comparison[["row_index", "prediction"]],
            on="row_index",
            suffixes=("_reference", "_comparison"),
            validate="one_to_one",
        )
        group_ids = joined["group_id"].unique()
        by_group = {group_id: indices for group_id, indices in joined.groupby("group_id").indices.items()}
        differences = np.empty(args.bootstrap_repetitions, dtype=np.float64)
        for repetition in range(args.bootstrap_repetitions):
            sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
            sampled_indices = np.concatenate([by_group[group_id] for group_id in sampled_groups])
            sampled = joined.iloc[sampled_indices]
            labels = sampled["label"].to_numpy()
            differences[repetition] = matthews_corrcoef(
                labels, sampled["prediction_reference"].to_numpy()
            ) - matthews_corrcoef(labels, sampled["prediction_comparison"].to_numpy())
        observed = matthews_corrcoef(
            joined["label"], joined["prediction_reference"]
        ) - matthews_corrcoef(joined["label"], joined["prediction_comparison"])
        bootstrap_rows.append(
            {
                "seed": seed,
                "reference": reference_name,
                "comparison": comparison_name,
                "groups": len(group_ids),
                "repetitions": args.bootstrap_repetitions,
                "observed_mcc_difference": observed,
                "cluster_bootstrap_ci95_low": np.quantile(differences, 0.025),
                "cluster_bootstrap_ci95_high": np.quantile(differences, 0.975),
                "bootstrap_probability_difference_gt_zero": np.mean(differences > 0),
            }
        )

    confidence_intervals.to_csv(args.experiment_dir / "fusion_confidence_intervals.csv", index=False)
    language_runs.to_csv(args.experiment_dir / "fusion_language_runs.csv", index=False)
    language_summary.to_csv(args.experiment_dir / "fusion_language_summary.csv", index=False)
    mcnemar.to_csv(args.experiment_dir / "fusion_mcnemar_all_seeds.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.experiment_dir / "fusion_group_bootstrap_seed42.csv", index=False)

    sparse_path = args.experiment_dir.parent / "grouped_baseline_seed42_predictions.csv"
    cross_family_rows = []
    if sparse_path.exists():
        sparse = pd.read_csv(sparse_path)
        sparse = sparse[sparse["model"] == "Char-TFIDF + LR"].copy()
        fusion = reference.copy()
        joined = fusion[["group_id", "language_column", "label", "prediction"]].merge(
            sparse[["group_id", "language_column", "label", "prediction"]],
            on=["group_id", "language_column", "label"],
            suffixes=("_fusion", "_sparse"),
            validate="one_to_one",
        )
        fusion_correct = joined["prediction_fusion"].to_numpy() == joined["label"].to_numpy()
        sparse_correct = joined["prediction_sparse"].to_numpy() == joined["label"].to_numpy()
        fusion_only = int(np.sum(fusion_correct & ~sparse_correct))
        sparse_only = int(np.sum(~fusion_correct & sparse_correct))
        discordant = fusion_only + sparse_only
        group_ids = joined["group_id"].unique()
        by_group = {group_id: indices for group_id, indices in joined.groupby("group_id").indices.items()}
        differences = np.empty(args.bootstrap_repetitions, dtype=np.float64)
        for repetition in range(args.bootstrap_repetitions):
            sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
            sampled_indices = np.concatenate([by_group[group_id] for group_id in sampled_groups])
            sampled = joined.iloc[sampled_indices]
            labels = sampled["label"].to_numpy()
            differences[repetition] = matthews_corrcoef(
                labels, sampled["prediction_fusion"].to_numpy()
            ) - matthews_corrcoef(labels, sampled["prediction_sparse"].to_numpy())
        observed = matthews_corrcoef(
            joined["label"], joined["prediction_fusion"]
        ) - matthews_corrcoef(joined["label"], joined["prediction_sparse"])
        cross_family_rows.append(
            {
                "seed": seed,
                "fusion_model": reference_name,
                "sparse_model": "Char-TFIDF + LR",
                "matched_rows": len(joined),
                "groups": len(group_ids),
                "fusion_only_correct": fusion_only,
                "sparse_only_correct": sparse_only,
                "exact_mcnemar_p": 1.0
                if discordant == 0
                else binomtest(fusion_only, discordant, p=0.5).pvalue,
                "fusion_minus_sparse_mcc": observed,
                "cluster_bootstrap_ci95_low": np.quantile(differences, 0.025),
                "cluster_bootstrap_ci95_high": np.quantile(differences, 0.975),
            }
        )
    cross_family = pd.DataFrame(cross_family_rows)
    cross_family.to_csv(args.experiment_dir / "fusion_vs_sparse_seed42.csv", index=False)
    print(confidence_intervals.round(4).to_string(index=False))
    print("\nCluster bootstrap\n", bootstrap.round(4).to_string(index=False))
    if not cross_family.empty:
        print("\nFusion versus sparse\n", cross_family.round(4).to_string(index=False))
    best_languages = language_summary[language_summary["configuration"] == reference_name].sort_values(
        "mcc_mean"
    )
    print("\nLowest languages\n", best_languages.head(5).round(4).to_string(index=False))
    print("\nHighest languages\n", best_languages.tail(5).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
