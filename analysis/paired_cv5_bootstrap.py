"""Paired source-group bootstrap on pooled out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_audit import load_long_dataset


COMPARATORS = (
    "mBERT",
    "Qwen2.5 raw",
    "Qwen2.5+PCA-1024",
    "mBERT+Qwen concat no PCA",
    "mBERT+Qwen RP-1024",
    "mBERT+Qwen PCA-1024",
    "mBERT+Qwen PCA-2048",
    "mBERT+Qwen+Char PCA-1024",
    "Fine-tuned mBERT",
    "Char-TFIDF + LR",
)


def slug(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def mcc(counts: np.ndarray) -> float:
    tn, fp, fn, tp = counts
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(numerator / denominator) if denominator else 0.0


def grouped_counts(frame: pd.DataFrame, group_ids: np.ndarray) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=np.int8)
    predictions = frame["prediction"].to_numpy(dtype=np.int8)
    code = labels * 2 + predictions
    group_index = pd.Categorical(frame["group_id"], categories=group_ids).codes
    if (group_index < 0).any():
        raise ValueError("Prediction contains an unrecognized source group")
    return np.bincount(group_index * 4 + code, minlength=len(group_ids) * 4).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    data = load_long_dataset(args.dataset)
    frames = {}
    for fold in range(1, 6):
        folder = args.fold_dir / f"fold_{fold}/fusion_core"
        selected = int(json.loads((folder / "metadata.json").read_text())[
            "selected_dimension"
        ])
        names = {"mBERT+Qwen PCA (validation-selected)":
                 f"mBERT+Qwen PCA-{selected}"}
        names.update({name: name for name in COMPARATORS
                      if name not in ("Char-TFIDF + LR", "Fine-tuned mBERT")})
        for alias, name in names.items():
            frame = pd.read_csv(folder / f"predictions_{slug(name)}.csv")
            frame["fold"] = fold
            frames.setdefault(alias, []).append(frame)
        contextual = pd.read_csv(
            args.fold_dir / f"fold_{fold}/finetuned_mbert"
            / "finetuned_mbert_prediction_seed42.csv"
        )
        contextual["fold"] = fold
        frames.setdefault("Fine-tuned mBERT", []).append(contextual)
    classical_path = args.fold_dir / "classical/grouped_baseline_all_fold_predictions.csv"
    classical = pd.read_csv(classical_path)
    classical = classical[classical["model"] == "Char-TFIDF + LR"].copy()
    index = data[["group_id", "language_column"]].reset_index().rename(
        columns={"index": "row_index"}
    )
    classical = classical.merge(index, on=["group_id", "language_column"],
                                validate="one_to_one")
    frames["Char-TFIDF + LR"] = [classical]
    combined = {
        name: pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
        for name, parts in frames.items()
    }
    reference = combined["mBERT+Qwen PCA (validation-selected)"]
    expected_index = np.arange(len(data))
    for name, frame in combined.items():
        if not np.array_equal(frame["row_index"].to_numpy(), expected_index):
            raise ValueError(f"Incomplete or duplicated out-of-fold predictions: {name}")
        if not np.array_equal(frame["label"].to_numpy(), reference["label"].to_numpy()):
            raise ValueError(f"Label mismatch: {name}")
    groups = np.sort(reference["group_id"].unique())
    ref_counts = grouped_counts(reference, groups)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(groups), size=(args.repetitions, len(groups)))
    rows = []
    for name in COMPARATORS:
        comparison_counts = grouped_counts(combined[name], groups)
        observed = mcc(ref_counts.sum(axis=0)) - mcc(comparison_counts.sum(axis=0))
        differences = np.empty(args.repetitions, dtype=np.float64)
        for index, selection in enumerate(samples):
            differences[index] = (
                mcc(ref_counts[selection].sum(axis=0))
                - mcc(comparison_counts[selection].sum(axis=0))
            )
        low, high = np.quantile(differences, [0.025, 0.975])
        rows.append({
            "reference": "mBERT+Qwen PCA (validation-selected)",
            "comparison": name,
            "pooled_oof_mcc_difference": observed,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "source_groups": len(groups),
            "repetitions": args.repetitions,
        })
        print(name, f"{observed:.4f} [{low:.4f}, {high:.4f}]", flush=True)
    pd.DataFrame(rows).to_csv(args.fold_dir / "paired_source_group_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
