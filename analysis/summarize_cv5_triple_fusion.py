"""Summarize held-out triple-fusion folds and matched source-group contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paired_cv5_bootstrap import grouped_counts, mcc, slug
from revision_audit import load_long_dataset
from run_cv5_triple_fusion import CONFIGURATIONS, FAMILY
from summarize_cv5_llama_fusion import METRICS, complete_oof, read_predictions


SELECTED = f"{FAMILY} PCA (validation-selected)"
COMPARATORS = (
    CONFIGURATIONS[0],
    "mBERT+Qwen PCA (validation-selected)",
    "Llama-2+Qwen PCA (validation-selected)",
    "Llama-2 raw",
    "Qwen2.5+PCA-1024",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    data = load_long_dataset(args.dataset)
    runs = []
    predictions = {name: [] for name in (SELECTED, *COMPARATORS)}
    for fold in range(1, 6):
        folder = args.fold_dir / f"fold_{fold}/triple_fusion"
        frame = pd.read_csv(folder / "runs.csv")
        if (len(frame) != len(CONFIGURATIONS)
                or set(frame["configuration"]) != set(CONFIGURATIONS)
                or frame["configuration"].duplicated().any()
                or frame[list(METRICS)].isna().any().any()):
            raise ValueError(f"Fold {fold}: incomplete triple configurations")
        runs.append(frame)
        width = int(json.loads((folder / "metadata.json").read_text())[
            "selected_dimension"
        ])
        selected_name = f"{FAMILY} PCA-{width}"
        selected_run = frame[frame["configuration"] == selected_name].copy()
        selected_run["configuration"] = SELECTED
        runs.append(selected_run)
        predictions[SELECTED].append(read_predictions(
            folder / f"predictions_{slug(selected_name)}.csv", fold
        ))
        predictions[CONFIGURATIONS[0]].append(read_predictions(
            folder / f"predictions_{slug(CONFIGURATIONS[0])}.csv", fold
        ))
        llama_dir = args.fold_dir / f"fold_{fold}/llama_fusion"
        llama_width = int(json.loads((llama_dir / "metadata.json").read_text())[
            "selected_dimension"
        ])
        for label, name in (
            ("Llama-2+Qwen PCA (validation-selected)",
             f"Llama-2+Qwen PCA-{llama_width}"),
            ("Llama-2 raw", "Llama-2 raw"),
        ):
            predictions[label].append(read_predictions(
                llama_dir / f"predictions_{slug(name)}.csv", fold
            ))
        baseline_dir = args.fold_dir / f"fold_{fold}/fusion_core"
        mbert_width = int(json.loads((baseline_dir / "metadata.json").read_text())[
            "selected_dimension"
        ])
        for label, name in (
            ("mBERT+Qwen PCA (validation-selected)",
             f"mBERT+Qwen PCA-{mbert_width}"),
            ("Qwen2.5+PCA-1024", "Qwen2.5+PCA-1024"),
        ):
            predictions[label].append(read_predictions(
                baseline_dir / f"predictions_{slug(name)}.csv", fold
            ))
    all_runs = pd.concat(runs, ignore_index=True)
    summary = all_runs.groupby("configuration")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)
    all_runs.to_csv(args.fold_dir / "triple_cv5_test_runs.csv", index=False)
    summary.to_csv(args.fold_dir / "triple_cv5_test_summary.csv", index=False)
    oof = {
        name: complete_oof(parts, len(data), name)
        for name, parts in predictions.items()
    }
    reference = oof[SELECTED]
    for name, frame in oof.items():
        if (not np.array_equal(frame["label"], reference["label"])
                or not np.array_equal(frame["group_id"], reference["group_id"])):
            raise ValueError(f"Unmatched test groups or labels: {name}")
    groups = np.sort(reference["group_id"].unique())
    counts = {name: grouped_counts(frame, groups) for name, frame in oof.items()}
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(groups), size=(args.repetitions, len(groups)))
    paired = []
    for name in COMPARATORS:
        left, right = counts[SELECTED], counts[name]
        observed = mcc(left.sum(axis=0)) - mcc(right.sum(axis=0))
        draws = np.array([
            mcc(left[sample].sum(axis=0)) - mcc(right[sample].sum(axis=0))
            for sample in samples
        ])
        low, high = np.quantile(draws, [0.025, 0.975])
        paired.append({
            "reference": SELECTED,
            "comparison": name,
            "pooled_oof_mcc_difference": observed,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "source_groups": len(groups),
            "repetitions": args.repetitions,
        })
    pd.DataFrame(paired).to_csv(
        args.fold_dir / "triple_cv5_paired_bootstrap.csv", index=False
    )
    print(summary[["configuration", "mcc_mean", "mcc_std"]].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
