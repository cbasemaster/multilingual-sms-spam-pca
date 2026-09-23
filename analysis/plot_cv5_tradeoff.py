"""Plot validation width selection and held-out storage trade-offs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WIDTHS = (768, 1024, 1536, 2048)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = pd.read_csv(args.runs or args.fold_dir / "all_test_fold_runs.csv")
    if "family" in runs:
        runs = runs[runs["family"] == "static embedding CNN"]
    expected_runs = 25 if args.runs else 5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"PCA": "#126a7a", "RP": "#bd6537", "Concat": "#514a74"}

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for width in WIDTHS:
        row = runs[runs["configuration"] == f"mBERT+Qwen PCA-{width}"]
        if len(row) != expected_runs:
            raise ValueError(f"Missing runs for PCA-{width}: {len(row)}")
        ax.errorbar(width, row["validation_mcc"].mean(),
                    yerr=row["validation_mcc"].std(ddof=1), fmt="o",
                    color=colors["PCA"], capsize=3)
    means = [runs.loc[runs["configuration"] == f"mBERT+Qwen PCA-{width}",
                      "validation_mcc"].mean() for width in WIDTHS]
    ax.plot(WIDTHS, means, color=colors["PCA"], label="Concat+PCA")
    rp = runs[runs["configuration"] == "mBERT+Qwen RP-1024"]
    ax.errorbar(1024, rp["validation_mcc"].mean(),
                yerr=rp["validation_mcc"].std(ddof=1), fmt="s",
                color=colors["RP"], capsize=3, label="Concat+RP")
    ax.set(xlabel="Final dimension", ylabel="Inner-validation MCC",
           xticks=WIDTHS)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_mcc_dimension_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = {
        "mBERT+Qwen PCA-768": "PCA 768",
        "mBERT+Qwen PCA-1024": "PCA 1024",
        "mBERT+Qwen PCA-1536": "PCA 1536",
        "mBERT+Qwen PCA-2048": "PCA 2048",
        "mBERT+Qwen RP-1024": "RP 1024",
        "mBERT+Qwen concat no PCA": "No projection",
        "Qwen2.5 raw": "Qwen raw",
        "Qwen2.5+PCA-1024": "Qwen PCA",
    }
    label_offsets = {
        "mBERT+Qwen PCA-768": (8, 10),
        "mBERT+Qwen PCA-1024": (12, 22),
        "Qwen2.5+PCA-1024": (12, -42),
        "mBERT+Qwen concat no PCA": (-10, -12),
    }
    for name, label in labels.items():
        row = runs[runs["configuration"] == name]
        if len(row) != expected_runs:
            raise ValueError(f"Missing runs for {name}: {len(row)}")
        color = (colors["PCA"] if "PCA-" in name else
                 colors["RP"] if "RP" in name else colors["Concat"])
        x, y = row["stored_embedding_mb"].mean(), row["mcc"].mean()
        ax.errorbar(x, y, yerr=row["mcc"].std(ddof=1), fmt="o",
                    color=color, capsize=3)
        offset = label_offsets.get(name, (5, 5))
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    fontsize=8.5, ha="right" if offset[0] < 0 else "left",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6}
                    if name in label_offsets else None)
    ax.set(xlabel="Stored frozen table (MB)", ylabel="Held-out test MCC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mcc_storage_tradeoff.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
