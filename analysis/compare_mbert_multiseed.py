"""Paired source-group comparison of the five-seed mBERT ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from revision_audit import load_long_dataset
from summarize_multiseed_static import grouped_confusion, mcc_from_counts, oof_ensembles


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "source/sms_spam_multilingual.parquet"
FOLD_DIR = ROOT / "output/source_group_cv5"
STATIC_DIR = ROOT / "output/multiseed_all"
OUTPUT = ROOT / "output/transformer_multiseed/summary/mbert_paired_bootstrap.csv"


def main() -> None:
    data = load_long_dataset(DATASET)
    labels = data["label"].to_numpy(dtype=np.int8)
    groups = data["group_id"].to_numpy(dtype=np.int64)

    combined = pd.read_csv(
        STATIC_DIR / "summary/static_multiseed_runs_with_selection.csv"
    )
    _, static_probabilities, _, _ = oof_ensembles(
        combined, DATASET, FOLD_DIR, STATIC_DIR / "predictions"
    )
    predictions = {
        "mBERT+Qwen validation-selected PCA": (
            static_probabilities["mBERT+Qwen validation-selected PCA"] >= 0.5
        ).astype(np.int8)
    }

    mbert = np.full(len(data), -1, dtype=np.int8)
    for fold in range(1, 6):
        folder = FOLD_DIR / f"fold_{fold}/finetuned_mbert_tuned_threshold"
        runs = pd.read_csv(folder / "finetuned_mbert_runs.csv")
        votes = []
        indices = None
        for seed in sorted(runs["seed"].astype(int)):
            frame = pd.read_csv(folder / f"finetuned_mbert_prediction_seed{seed}.csv")
            frame = frame.sort_values("row_index")
            indices = frame["row_index"].to_numpy(dtype=np.int64)
            votes.append(frame["prediction"].to_numpy(dtype=np.float32))
        mbert[indices] = (np.mean(votes, axis=0) >= 0.5).astype(np.int8)
    if (mbert < 0).any():
        raise RuntimeError("Incomplete mBERT out-of-fold predictions")
    predictions["Fine-tuned mBERT"] = mbert

    classical = pd.read_csv(
        FOLD_DIR / "classical/grouped_baseline_all_fold_predictions.csv"
    )
    classical = classical[classical["model"] == "Char-TFIDF + LR"].copy()
    row_index = data[["group_id", "language_column"]].reset_index().rename(
        columns={"index": "row_index"}
    )
    classical = classical.merge(
        row_index, on=["group_id", "language_column"], validate="one_to_one"
    ).sort_values("row_index")
    predictions["Character TF-IDF + logistic regression"] = classical[
        "prediction"
    ].to_numpy(dtype=np.int8)

    rng = np.random.default_rng(20260923)
    group_count = np.unique(groups).size
    selections = rng.integers(0, group_count, size=(2000, group_count), dtype=np.int32)
    rows = []
    for reference, comparison in (
        ("Fine-tuned mBERT", "Character TF-IDF + logistic regression"),
        ("Fine-tuned mBERT", "mBERT+Qwen validation-selected PCA"),
        (
            "Character TF-IDF + logistic regression",
            "mBERT+Qwen validation-selected PCA",
        ),
    ):
        reference_counts = grouped_confusion(labels, predictions[reference], groups)
        comparison_counts = grouped_confusion(labels, predictions[comparison], groups)
        values = np.empty(len(selections), dtype=np.float64)
        for index, selection in enumerate(selections):
            values[index] = (
                mcc_from_counts(reference_counts[selection].sum(axis=0))
                - mcc_from_counts(comparison_counts[selection].sum(axis=0))
            )
        rows.append({
            "reference": reference,
            "comparison": comparison,
            "reference_oof_mcc": mcc_from_counts(reference_counts.sum(axis=0)),
            "comparison_oof_mcc": mcc_from_counts(comparison_counts.sum(axis=0)),
            "oof_mcc_difference": (
                mcc_from_counts(reference_counts.sum(axis=0))
                - mcc_from_counts(comparison_counts.sum(axis=0))
            ),
            "source_group_bootstrap_ci_low": np.quantile(values, 0.025),
            "source_group_bootstrap_ci_high": np.quantile(values, 0.975),
            "source_groups": group_count,
            "repetitions": len(selections),
        })
    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
