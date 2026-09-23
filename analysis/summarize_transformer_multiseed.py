"""Summarize contextual transformer baselines over five folds and five seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from grouped_baseline_benchmark import metrics
from revision_audit import load_long_dataset
from summarize_multiseed_static import grouped_confusion, mcc_from_counts


MODEL_SPECS = (
    ("mBERT", "finetuned_mbert", "mbert"),
    ("DistilmBERT", "finetuned_distilmbert", "distilmbert"),
    ("XLM-R Base", "finetuned_xlmr_base", "xlmr_base"),
)
METRIC_COLUMNS = (
    "accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc",
    "cohen_kappa", "balanced_accuracy", "roc_auc", "brier", "ece_15",
)


def model_dir(fold_root: Path, transformer_root: Path, key: str, fold: int) -> Path:
    if key == "mbert":
        return fold_root / f"fold_{fold}" / "finetuned_mbert_tuned_threshold"
    return transformer_root / key / f"fold_{fold}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    prediction_by_model: dict[str, list[pd.DataFrame]] = {}
    included_models = []
    for display_name, prefix, key in MODEL_SPECS:
        directories = [
            model_dir(args.fold_dir, args.transformer_dir, key, fold)
            for fold in range(1, 6)
        ]
        if not all((directory / f"{prefix}_runs.csv").exists() for directory in directories):
            continue
        included_models.append(display_name)
        prediction_by_model[display_name] = []
        for fold in range(1, 6):
            directory = model_dir(args.fold_dir, args.transformer_dir, key, fold)
            frame = pd.read_csv(directory / f"{prefix}_runs.csv")
            frame["fold"] = fold
            frame["model"] = display_name
            frames.append(frame)
            seed_predictions = []
            for seed in sorted(frame["seed"].astype(int).unique()):
                prediction = pd.read_csv(directory / f"{prefix}_prediction_seed{seed}.csv")
                prediction = prediction.sort_values("row_index").reset_index(drop=True)
                seed_predictions.append(prediction)
            base = seed_predictions[0][["row_index", "group_id", "label"]].copy()
            vote = np.mean(
                [value["prediction"].to_numpy(dtype=np.float32) for value in seed_predictions],
                axis=0,
            )
            base["prediction"] = (vote >= 0.5).astype(np.int8)
            base["fold"] = fold
            prediction_by_model[display_name].append(base)

    runs = pd.concat(frames, ignore_index=True)
    if not frames:
        raise RuntimeError("No complete five-fold transformer result set was found")
    counts = runs.groupby("model").size()
    if not counts.eq(25).all():
        raise RuntimeError(f"Incomplete contextual multi-seed suite: {counts.to_dict()}")
    runs.to_csv(args.output_dir / "transformer_multiseed_runs.csv", index=False)

    summary_rows = []
    for name, group in runs.groupby("model", sort=False):
        row = {
            "model": name,
            "runs": len(group),
            "folds": group["fold"].nunique(),
            "seeds": group["seed"].nunique(),
            "mcc_fold_mean_std": group.groupby("fold")["mcc"].mean().std(ddof=1),
            "mcc_mean_within_fold_seed_std": group.groupby("fold")["mcc"].std(ddof=1).mean(),
            "train_seconds_mean": group["train_seconds"].mean(),
            "inference_ms_per_message_mean": group["inference_ms_per_message"].mean(),
            "peak_gpu_memory_mb_mean": group["peak_gpu_memory_mb"].mean(),
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mcc_mean", ascending=False)
    summary.to_csv(args.output_dir / "transformer_multiseed_summary.csv", index=False)

    pairs = tuple(
        pair for pair in (
            ("XLM-R Base", "mBERT"),
            ("DistilmBERT", "mBERT"),
            ("XLM-R Base", "DistilmBERT"),
        )
        if pair[0] in included_models and pair[1] in included_models
    )
    indexed = runs.set_index(["model", "fold", "seed"])
    comparison_rows = []
    for reference, comparison in pairs:
        difference = (
            indexed.loc[reference, "mcc"] - indexed.loc[comparison, "mcc"]
        ).rename("difference").reset_index()
        fold_means = difference.groupby("fold")["difference"].mean()
        margin = t.ppf(0.975, 4) * fold_means.std(ddof=1) / np.sqrt(5)
        comparison_rows.append({
            "reference": reference,
            "comparison": comparison,
            "mean_mcc_difference_25_runs": difference["difference"].mean(),
            "fold_clustered_ci_low": fold_means.mean() - margin,
            "fold_clustered_ci_high": fold_means.mean() + margin,
        })
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "transformer_multiseed_paired_fold_ci.csv", index=False
    )

    data = load_long_dataset(args.dataset)
    labels = data["label"].to_numpy(dtype=np.int8)
    groups = data["group_id"].to_numpy(dtype=np.int64)
    ensemble_predictions = {}
    ensemble_rows = []
    for name, parts in prediction_by_model.items():
        oof = pd.concat(parts, ignore_index=True).sort_values("row_index")
        if not np.array_equal(oof["row_index"].to_numpy(), np.arange(len(data))):
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        prediction = oof["prediction"].to_numpy(dtype=np.int8)
        ensemble_predictions[name] = prediction
        ensemble_rows.append({"model": name, **metrics(labels, prediction)})
    pd.DataFrame(ensemble_rows).to_csv(
        args.output_dir / "transformer_multiseed_majority_oof.csv", index=False
    )

    rng = np.random.default_rng(20260922)
    n_groups = np.unique(groups).size
    selections = rng.integers(
        0, n_groups, size=(args.bootstrap_repetitions, n_groups), dtype=np.int32
    )
    bootstrap_rows = []
    for reference, comparison in pairs:
        a = grouped_confusion(labels, ensemble_predictions[reference], groups)
        b = grouped_confusion(labels, ensemble_predictions[comparison], groups)
        observed = mcc_from_counts(a.sum(axis=0)) - mcc_from_counts(b.sum(axis=0))
        values = np.empty(args.bootstrap_repetitions)
        for index, selection in enumerate(selections):
            values[index] = (
                mcc_from_counts(a[selection].sum(axis=0))
                - mcc_from_counts(b[selection].sum(axis=0))
            )
        low, high = np.quantile(values, [0.025, 0.975])
        bootstrap_rows.append({
            "reference": reference,
            "comparison": comparison,
            "majority_oof_mcc_difference": observed,
            "source_group_bootstrap_ci_low": low,
            "source_group_bootstrap_ci_high": high,
        })
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "transformer_multiseed_majority_bootstrap.csv", index=False
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
