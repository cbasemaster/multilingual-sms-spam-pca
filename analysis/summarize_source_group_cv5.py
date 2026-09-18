"""Audit completeness and summarize matched source-group test folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("accuracy", "spam_f1", "macro_f1", "mcc", "cohen_kappa")
OPTIONAL_METRICS = ("roc_auc", "ece_15")
EXPECTED_FOLDS = set(range(1, 6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    classical = pd.read_csv(args.fold_dir / "classical/grouped_baseline_runs.csv")
    classical = classical.rename(columns={"model": "configuration"})
    classical["family"] = "sparse unprojected or SVD-128"
    frames.append(classical)

    for fold in range(1, 6):
        fold_dir = args.fold_dir / f"fold_{fold}"
        for folder, family in (
            ("fusion_core", "static embedding CNN"),
            ("sparse_pca1024", "sparse PCA-1024"),
            ("sparse_svd1024", "sparse SVD-1024"),
        ):
            path = fold_dir / folder / "runs.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["fold"] = fold
            frame["family"] = family
            if "model" in frame:
                frame = frame.rename(columns={"model": "configuration"})
            if folder == "fusion_core":
                fusion_frame = frame
            frames.append(frame)
        finetuned = fold_dir / "finetuned_mbert/finetuned_mbert_runs.csv"
        if not finetuned.exists():
            raise FileNotFoundError(finetuned)
        frame = pd.read_csv(finetuned)
        frame["fold"] = fold
        frame["family"] = "fine-tuned contextual model"
        frame["configuration"] = "Fine-tuned mBERT"
        frames.append(frame)

        metadata = json.loads((fold_dir / "fusion_core/metadata.json").read_text())
        selected = metadata["selected_dimension"]
        chosen = fusion_frame[
            fusion_frame["configuration"] == f"mBERT+Qwen PCA-{selected}"
        ].copy()
        if len(chosen) != 1:
            raise ValueError(f"Fold {fold}: selected PCA row missing")
        chosen["configuration"] = "mBERT+Qwen PCA (validation-selected)"
        frames.append(chosen)

    all_runs = pd.concat(frames, ignore_index=True)
    required = ["family", "configuration", "fold", *METRICS]
    if all_runs[required].isna().any().any():
        raise ValueError("Missing fold identifier or primary metric")
    if all_runs.duplicated(["family", "configuration", "fold"]).any():
        raise ValueError("Duplicate configuration-fold result")
    incomplete = {
        f"{family}: {name}": sorted(EXPECTED_FOLDS - set(rows["fold"].astype(int)))
        for (family, name), rows in all_runs.groupby(["family", "configuration"])
        if set(rows["fold"].astype(int)) != EXPECTED_FOLDS
    }
    if incomplete:
        raise ValueError(f"Incomplete cross-validation results: {incomplete}")

    summary = all_runs.groupby(["family", "configuration"])[
        list(METRICS + OPTIONAL_METRICS)
    ].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)
    all_runs.to_csv(args.fold_dir / "all_test_fold_runs.csv", index=False)
    summary.to_csv(args.fold_dir / "all_test_fold_summary.csv", index=False)
    print(summary[["family", "configuration", "mcc_mean", "mcc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
