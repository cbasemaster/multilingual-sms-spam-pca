"""Summarize five-fold by five-seed static-representation experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from grouped_embedding_fusion_experiment import evaluate
from revision_audit import load_long_dataset
from run_cv5_multiseed_all_static import slug


METRICS = (
    "accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc",
    "cohen_kappa", "balanced_accuracy", "roc_auc", "brier", "ece_15",
)
SELECTION_FAMILIES = {
    "mBERT+Qwen validation-selected PCA": [
        f"mBERT+Qwen PCA-{width}" for width in (768, 1024, 1536, 2048)
    ],
    "mBERT+Qwen+Char validation-selected PCA": [
        f"mBERT+Qwen+Char PCA-{width}" for width in (768, 1024, 1536, 2048)
    ],
    "Llama-2+Qwen validation-selected PCA": [
        f"Llama-2+Qwen PCA-{width}" for width in (1024, 1536, 2048)
    ],
    "mBERT+Qwen2.5+Llama-2 validation-selected PCA": [
        f"mBERT+Qwen2.5+Llama-2 PCA-{width}" for width in (1024, 1536, 2048)
    ],
}
COMPARISONS = (
    ("mBERT+Qwen validation-selected PCA", "mBERT"),
    ("mBERT+Qwen validation-selected PCA", "Qwen2.5+PCA-1024"),
    ("mBERT+Qwen validation-selected PCA", "mBERT+Qwen concat no PCA"),
    ("mBERT+Qwen validation-selected PCA", "mBERT+Qwen RP-1024"),
    (
        "mBERT+Qwen+Char validation-selected PCA",
        "mBERT+Qwen validation-selected PCA",
    ),
    (
        "mBERT+Qwen+Char validation-selected PCA",
        "mBERT+Qwen+Char concat no PCA",
    ),
    ("Llama-2+Qwen validation-selected PCA", "Llama-2+Qwen concat no PCA"),
    (
        "mBERT+Qwen2.5+Llama-2 validation-selected PCA",
        "mBERT+Qwen2.5+Llama-2 concat no PCA",
    ),
)


def deployed_width(name: str) -> int:
    match = re.search(r"(?:PCA|RP)-(\d+)$", name)
    if match:
        return int(match.group(1))
    if name in {"mBERT", "DistilBERT"}:
        return 768
    if name == "Qwen2.5 raw":
        return 3584
    if "DistilBERT+mBERT" in name:
        return 1536
    if name == "Llama-2 raw":
        return 4096
    if "Llama-2+Qwen" in name and "mBERT" not in name:
        return 7680
    if "mBERT+Qwen2.5+Llama-2" in name:
        return 8448
    if "mBERT+Qwen+Char" in name:
        return 5376
    if "mBERT+Qwen" in name:
        return 4352
    raise ValueError(f"Unknown deployed width: {name}")


def source_width(name: str) -> int:
    if name.startswith("mBERT PCA") or name == "mBERT":
        return 768
    if name.startswith("Qwen2.5"):
        return 3584
    if name.startswith("DistilBERT+mBERT"):
        return 1536
    if name == "DistilBERT":
        return 768
    if name.startswith("Llama-2+Qwen"):
        return 7680
    if name.startswith("Llama-2"):
        return 4096
    if name.startswith("mBERT+Qwen2.5+Llama-2"):
        return 8448
    if name.startswith("mBERT+Qwen+Char"):
        return 5376
    if name.startswith("mBERT+Qwen"):
        return 4352
    raise ValueError(f"Unknown source width: {name}")


def add_validation_selections(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    frequencies = []
    for selected_name, candidates in SELECTION_FAMILIES.items():
        subset = runs[runs["configuration"].isin(candidates)].copy()
        for (fold, seed), group in subset.groupby(["fold", "training_seed"]):
            winner = group.sort_values(
                ["validation_mcc", "final_dimension"], ascending=[False, True]
            ).iloc[0].copy()
            original = str(winner["configuration"])
            winner["selected_source_configuration"] = original
            winner["configuration"] = selected_name
            winner["final_dimension"] = deployed_width(original)
            selected_rows.append(winner)
        family = pd.DataFrame(selected_rows)
        family = family[family["configuration"] == selected_name]
        for width, count in family["final_dimension"].value_counts().sort_index().items():
            frequencies.append({
                "configuration": selected_name,
                "selected_dimension": int(width),
                "runs": int(count),
            })
    selected = pd.DataFrame(selected_rows)
    combined = pd.concat([runs, selected], ignore_index=True, sort=False)
    return combined, pd.DataFrame(frequencies)


def summarize(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    variance_rows = []
    for name, group in combined.groupby("configuration", sort=False):
        row = {
            "configuration": name,
            "source_dimension": source_width(name),
            "final_dimension": int(group["final_dimension"].median()),
            "runs": len(group),
            "folds": group["fold"].nunique(),
            "seeds": group["training_seed"].nunique(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
        fold_seed_means = group.groupby("fold")["mcc"].mean()
        within_fold = group.groupby("fold")["mcc"].std(ddof=1)
        seed_fold_means = group.groupby("training_seed")["mcc"].mean()
        variance_rows.append({
            "configuration": name,
            "mcc_over_25_runs_mean": group["mcc"].mean(),
            "mcc_over_25_runs_std": group["mcc"].std(ddof=1),
            "mcc_fold_mean_std": fold_seed_means.std(ddof=1),
            "mcc_mean_within_fold_seed_std": within_fold.mean(),
            "mcc_seed_mean_std": seed_fold_means.std(ddof=1),
        })
    return pd.DataFrame(rows), pd.DataFrame(variance_rows)


def paired_fold_intervals(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = combined.set_index(["configuration", "fold", "training_seed"])
    for reference, comparison in COMPARISONS:
        a = indexed.loc[reference, "mcc"]
        b = indexed.loc[comparison, "mcc"]
        difference = (a - b).rename("difference").reset_index()
        fold_means = difference.groupby("fold")["difference"].mean()
        margin = t.ppf(0.975, len(fold_means) - 1) * fold_means.std(ddof=1) / np.sqrt(len(fold_means))
        rows.append({
            "reference": reference,
            "comparison": comparison,
            "mean_mcc_difference_25_runs": difference["difference"].mean(),
            "run_level_std": difference["difference"].std(ddof=1),
            "fold_clustered_ci_low": fold_means.mean() - margin,
            "fold_clustered_ci_high": fold_means.mean() + margin,
            "folds": len(fold_means),
            "seeds_per_fold": difference["training_seed"].nunique(),
        })
    return pd.DataFrame(rows)


def probability_path(prediction_dir: Path, fold: int, seed: int, configuration: str) -> Path:
    return prediction_dir / f"fold{fold}_seed{seed}_{slug(configuration)}.npz"


def oof_ensembles(
    combined: pd.DataFrame,
    dataset: Path,
    fold_dir: Path,
    prediction_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data = load_long_dataset(dataset)
    labels = data["label"].to_numpy(dtype=np.int8)
    group_ids = data["group_id"].to_numpy(dtype=np.int64)
    names = list(dict.fromkeys(combined["configuration"].tolist()))
    probabilities = {name: np.full(len(data), np.nan, dtype=np.float32) for name in names}
    lookup = combined.set_index(["configuration", "fold", "training_seed"])
    for fold in range(1, 6):
        with np.load(fold_dir / f"fold_{fold}.npz") as parts:
            test = parts["test"]
        for name in names:
            fold_values = []
            for seed in sorted(combined["training_seed"].unique()):
                source = name
                row = lookup.loc[(name, fold, seed)]
                if name in SELECTION_FAMILIES:
                    source = str(row["selected_source_configuration"])
                path = probability_path(prediction_dir, fold, int(seed), source)
                with np.load(path) as saved:
                    fold_values.append(saved["probability"].astype(np.float32))
            probabilities[name][test] = np.mean(fold_values, axis=0)
    if any(np.isnan(value).any() for value in probabilities.values()):
        raise RuntimeError("Incomplete out-of-fold probabilities")
    rows = []
    for name, probability in probabilities.items():
        logits = np.log(
            np.clip(probability, 1e-7, 1 - 1e-7)
            / np.clip(1 - probability, 1e-7, 1 - 1e-7)
        )
        rows.append({"configuration": name, **evaluate(labels, logits)})
    return pd.DataFrame(rows), probabilities, labels, group_ids


def grouped_confusion(labels: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> np.ndarray:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique_groups), 4), dtype=np.int64)
    category = labels.astype(np.int8) * 2 + prediction.astype(np.int8)
    np.add.at(counts, (inverse, category), 1)
    return counts


def mcc_from_counts(counts: np.ndarray) -> float:
    tn, fp, fn, tp = counts.astype(np.float64)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def grouped_bootstrap(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    n_groups = np.unique(groups).size
    selections = rng.integers(0, n_groups, size=(repetitions, n_groups), dtype=np.int32)
    rows = []
    for reference, comparison in COMPARISONS:
        a = grouped_confusion(labels, probabilities[reference] >= 0.5, groups)
        b = grouped_confusion(labels, probabilities[comparison] >= 0.5, groups)
        observed = mcc_from_counts(a.sum(axis=0)) - mcc_from_counts(b.sum(axis=0))
        values = np.empty(repetitions, dtype=np.float64)
        for index, selection in enumerate(selections):
            values[index] = (
                mcc_from_counts(a[selection].sum(axis=0))
                - mcc_from_counts(b[selection].sum(axis=0))
            )
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({
            "reference": reference,
            "comparison": comparison,
            "five_seed_ensemble_oof_mcc_difference": observed,
            "source_group_bootstrap_ci_low": low,
            "source_group_bootstrap_ci_high": high,
            "source_groups": n_groups,
            "repetitions": repetitions,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs)
    counts = runs.groupby("configuration").size()
    if len(runs) != 33 * 25 or not counts.eq(25).all():
        raise RuntimeError(f"Expected 25 runs for each of 33 configurations; found {counts.to_dict()}")
    combined, frequencies = add_validation_selections(runs)
    summary, variance = summarize(combined)
    paired = paired_fold_intervals(combined)
    ensemble, probabilities, labels, groups = oof_ensembles(
        combined, args.dataset, args.fold_dir, args.prediction_dir
    )
    bootstrap = grouped_bootstrap(
        probabilities, labels, groups, args.bootstrap_repetitions
    )
    combined.to_csv(args.output_dir / "static_multiseed_runs_with_selection.csv", index=False)
    frequencies.to_csv(args.output_dir / "static_multiseed_selection_frequency.csv", index=False)
    summary.to_csv(args.output_dir / "static_multiseed_summary.csv", index=False)
    variance.to_csv(args.output_dir / "static_multiseed_variance.csv", index=False)
    paired.to_csv(args.output_dir / "static_multiseed_paired_fold_ci.csv", index=False)
    ensemble.to_csv(args.output_dir / "static_multiseed_ensemble_oof.csv", index=False)
    bootstrap.to_csv(args.output_dir / "static_multiseed_ensemble_bootstrap.csv", index=False)
    print(summary.sort_values("mcc_mean", ascending=False).head(15).to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
