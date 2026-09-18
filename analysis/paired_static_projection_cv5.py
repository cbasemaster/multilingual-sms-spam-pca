"""Paired source-group MCC intervals for PCA against its no-PCA control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_cv5_bootstrap import grouped_counts, mcc, slug
from revision_audit import load_long_dataset


PAIRS = (
    ("mBERT PCA-256", "mBERT"),
    ("Qwen2.5+PCA-1024", "Qwen2.5 raw"),
    ("mBERT+Qwen PCA-1024", "mBERT+Qwen concat no PCA"),
    ("mBERT+Qwen+Char PCA-1024", "mBERT+Qwen+Char concat no PCA"),
    ("DistilBERT+mBERT PCA-1024", "DistilBERT+mBERT standardized concat"),
)


def predictions(fold_dir: Path, name: str, sample_count: int) -> pd.DataFrame:
    parts = []
    for fold in range(1, 6):
        frame = pd.read_csv(
            fold_dir / f"fold_{fold}/fusion_core/predictions_{slug(name)}.csv"
        )
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True).sort_values("row_index")
    result = result.reset_index(drop=True)
    if not np.array_equal(result["row_index"].to_numpy(), np.arange(sample_count)):
        raise ValueError(f"Incomplete out-of-fold predictions for {name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()

    dataset = load_long_dataset(args.dataset)
    names = {name for pair in PAIRS for name in pair}
    frames = {
        name: predictions(args.fold_dir, name, len(dataset)) for name in names
    }
    first = next(iter(frames.values()))
    groups = np.sort(first["group_id"].unique())
    for name, frame in frames.items():
        if not np.array_equal(frame["label"].to_numpy(), first["label"].to_numpy()):
            raise ValueError(f"Label mismatch for {name}")
        if not np.array_equal(frame["group_id"].to_numpy(), first["group_id"].to_numpy()):
            raise ValueError(f"Source-group mismatch for {name}")
    counts = {name: grouped_counts(frame, groups) for name, frame in frames.items()}
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(groups), size=(args.repetitions, len(groups)))
    rows = []
    for projected, unprojected in PAIRS:
        left, right = counts[projected], counts[unprojected]
        observed = mcc(left.sum(axis=0)) - mcc(right.sum(axis=0))
        draws = np.array([
            mcc(left[sample].sum(axis=0)) - mcc(right[sample].sum(axis=0))
            for sample in samples
        ])
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "projected": projected,
            "unprojected": unprojected,
            "pooled_oof_mcc_difference": observed,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "source_groups": len(groups),
            "repetitions": args.repetitions,
        })
        print(f"{projected}: {observed:.4f} [{low:.4f}, {high:.4f}]")
    pd.DataFrame(rows).to_csv(
        args.fold_dir / "paired_static_projection_bootstrap.csv", index=False
    )


if __name__ == "__main__":
    main()
