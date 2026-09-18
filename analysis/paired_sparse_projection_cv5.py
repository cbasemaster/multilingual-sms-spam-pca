"""Compare sparse PCA-1024 with matched SVD-1024 using grouped OOF bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_cv5_bootstrap import grouped_counts, mcc
from revision_audit import load_long_dataset


MODELS = (
    ("char-tfidf", "lr"),
    ("char-tfidf", "linear_svm"),
    ("word-tfidf", "lr"),
    ("word-tfidf", "linear_svm"),
    ("word-tfidf", "mlp"),
    ("word-tfidf", "histgb"),
)


def read_oof(fold_dir: Path, projection: str, feature: str,
             classifier: str) -> pd.DataFrame:
    frames = []
    for fold in range(1, 6):
        name = f"{feature}_plus_{projection}-1024_plus_{classifier}_seed42.csv"
        path = fold_dir / f"fold_{fold}" / f"sparse_{projection}1024" / name
        frame = pd.read_csv(path).rename(columns={"sample_index": "row_index"})
        frame["fold"] = fold
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        "row_index").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    data = load_long_dataset(args.dataset)
    groups = np.sort(data["group_id"].unique())
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(groups),
                           size=(args.repetitions, len(groups)))
    expected_index = np.arange(len(data))
    rows = []
    for feature, classifier in MODELS:
        pca = read_oof(args.fold_dir, "pca", feature, classifier)
        svd = read_oof(args.fold_dir, "svd", feature, classifier)
        if not np.array_equal(pca["row_index"], expected_index):
            raise ValueError(f"Incomplete PCA predictions: {feature} {classifier}")
        if not np.array_equal(svd["row_index"], expected_index):
            raise ValueError(f"Incomplete SVD predictions: {feature} {classifier}")
        if not np.array_equal(pca["label"], svd["label"]):
            raise ValueError(f"Label mismatch: {feature} {classifier}")
        pca_counts = grouped_counts(pca, groups)
        svd_counts = grouped_counts(svd, groups)
        observed = mcc(pca_counts.sum(axis=0)) - mcc(svd_counts.sum(axis=0))
        differences = np.array([
            mcc(pca_counts[selection].sum(axis=0))
            - mcc(svd_counts[selection].sum(axis=0))
            for selection in samples
        ])
        low, high = np.quantile(differences, [0.025, 0.975])
        rows.append({
            "feature": feature,
            "classifier": classifier,
            "pooled_oof_mcc_difference_pca_minus_svd": observed,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "source_groups": len(groups),
            "repetitions": args.repetitions,
        })
        print(feature, classifier, f"{observed:.4f} [{low:.4f}, {high:.4f}]",
              flush=True)
    pd.DataFrame(rows).to_csv(args.fold_dir / "paired_sparse_projection_bootstrap.csv",
                              index=False)


if __name__ == "__main__":
    main()
