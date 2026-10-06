"""Audit repeated gap controls, paired source bootstrap, and PCA variance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from revision_audit import load_long_dataset


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "output/source_group_cv5"
STATIC = ROOT / "output/multiseed_all"
OUT = ROOT / "output/revision_gap_experiments"
FIGURE = ROOT / "output/revised_manuscript_draft/Fig4_grouped_confusion.pdf"
METRICS = ("accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc",
           "cohen_kappa", "balanced_accuracy", "roc_auc", "ece_15")


def slug(value):
    return "_".join(filter(None, "".join(
        char.lower() if char.isalnum() else " " for char in value).split()))


def variance_report():
    specs = {
        "mBERT+Qwen": "fusion_core/fusion_pca2048.json",
        "Llama-2+Qwen": "llama_fusion/llama_qwen_pca2048.json",
        "mBERT+Qwen2.5+Llama-2": "triple_fusion/pca2048.json",
        "mBERT+Qwen+Char": "fusion_core/three_source_pca2048.json",
        "DistilBERT+mBERT": "fusion_core/distil_mbert_pca1024.json",
        "Qwen2.5": "fusion_core/qwen_pca1024.json",
        "Llama-2": "llama_fusion/llama_pca1024.json",
        "mBERT": "fusion_core/mbert_pca512.json",
    }
    rows = []
    for fold in range(1, 6):
        for family, relative in specs.items():
            metadata = json.loads((FOLDS / f"fold_{fold}" / relative).read_text())
            eigenvalues = np.asarray(metadata["eigenvalues"])
            variance = float(metadata["total_variance"])
            for width in (128, 256, 512, 768, 1024, 1536, 2048):
                if width > len(eigenvalues):
                    continue
                rows.append({"fold": fold, "family": family, "components": width,
                             "explained_variance": float(eigenvalues[:width].sum() / variance)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pca_variance_by_fold.csv", index=False)
    summary = frame.groupby(["family", "components"]).explained_variance.agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(OUT / "pca_variance_summary.csv", index=False)
    return summary


def counts_by_group(labels, predictions, groups):
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), 4), dtype=np.int64)
    np.add.at(counts, (inverse, labels * 2 + predictions), 1)
    return counts


def mcc(counts):
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator),
                     where=denominator != 0)


def paired_bootstrap(probabilities, labels, groups, pairs, repetitions=2000):
    counts = {name: counts_by_group(labels, values >= .5, groups)
              for name, values in probabilities.items()}
    rng = np.random.default_rng(20260922)
    n = np.unique(groups).size
    selections = rng.integers(0, n, size=(repetitions, n), dtype=np.int32)
    needed = list(dict.fromkeys(name for pair in pairs for name in pair))
    draws = {}
    for name in needed:
        draws[name] = np.array([mcc(counts[name][selection].sum(axis=0))
                                for selection in selections])
    rows = []
    for a, b in pairs:
        low, high = np.quantile(draws[a] - draws[b], [.025, .975])
        rows.append({"reference": a, "comparison": b,
                     "ensemble_mcc_difference": float(mcc(counts[a].sum(axis=0))
                                                       - mcc(counts[b].sum(axis=0))),
                     "ci_low": low, "ci_high": high, "groups": n,
                     "bootstrap_repetitions": repetitions})
    return pd.DataFrame(rows)


def ensemble_predictions(frame, prediction_dir, data_size, allow_selections=False):
    probabilities = {}
    for name, group in frame.groupby("configuration", sort=False):
        values = np.full(data_size, np.nan, dtype=np.float32)
        for fold, fold_runs in group.groupby("fold"):
            with np.load(FOLDS / f"fold_{int(fold)}.npz") as split:
                indices = split["test"]
            scores = []
            for _, run in fold_runs.iterrows():
                source = (run["selected_source_configuration"] if allow_selections
                          and pd.notna(run.get("selected_source_configuration")) else name)
                path = prediction_dir / f"fold{int(fold)}_seed{int(run.training_seed)}_{slug(source)}.npz"
                with np.load(path) as saved:
                    if "test_index" in saved:
                        np.testing.assert_array_equal(saved["test_index"], indices)
                    scores.append(saved["probability"].astype(np.float16).astype(np.float32))
            values[indices] = np.mean(scores, axis=0)
        if np.isnan(values).any():
            raise ValueError(f"Incomplete out-of-fold scores: {name}")
        probabilities[name] = values
    return probabilities


def confusion_figure(probabilities, labels):
    names = [
        ("mBERT+Qwen validation-selected PCA", "(a) Proposed Concat+PCA"),
        ("mBERT+Qwen concat no PCA", "(b) Concatenation without PCA"),
        ("mBERT+Qwen Late fusion", "(c) Late fusion"),
        ("mBERT", "(d) mBERT static vectors"),
        ("Qwen2.5+PCA-1024", "(e) Qwen2.5 + PCA-1024"),
        ("Character TF-IDF + LR", "(f) Character TF-IDF + LR"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.7))
    rows = []
    for axis, (name, title) in zip(axes.flat, names):
        matrix = confusion_matrix(labels, probabilities[name] >= .5, labels=[0, 1])
        percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100
        shaded = axis.imshow(percentage, cmap="Blues", vmin=0, vmax=100)
        for actual in range(2):
            for predicted in range(2):
                rate = percentage[actual, predicted]
                axis.text(predicted, actual, f"{matrix[actual, predicted]:,}\n({rate:.2f}%)",
                          ha="center", va="center", fontsize=9,
                          color="white" if rate > 55 else "#152433")
        axis.set_xticks([0, 1], ["Ham", "Spam"], fontsize=9)
        axis.set_yticks([0, 1], ["Ham", "Spam"], fontsize=9)
        axis.set_xlabel("Predicted class", fontsize=9)
        axis.set_ylabel("Actual class", fontsize=9)
        axis.set_title(title, fontsize=9, pad=8)
        tn, fp, fn, tp = matrix.ravel()
        rows.append({"configuration": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "false_positive_percent": fp / (tn + fp) * 100,
                     "false_negative_percent": fn / (fn + tp) * 100,
                     "ensemble_mcc": matthews_corrcoef(labels, probabilities[name] >= .5)})
    fig.tight_layout(pad=1.2, h_pad=2, w_pad=1.5, rect=(0, .08, 1, 1))
    color_axis = fig.add_axes([.33, .025, .34, .02])
    legend = fig.colorbar(shaded, cax=color_axis, orientation="horizontal", ticks=[0, 50, 100])
    legend.set_label("Within-class proportion (%)", fontsize=9)
    legend.ax.tick_params(labelsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    fig.savefig(OUT / "grouped_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "grouped_confusion_counts.csv", index=False)


def add_sparse(probabilities, data):
    sparse = pd.read_csv(FOLDS / "classical/grouped_baseline_all_fold_predictions.csv")
    sparse = sparse[sparse.model == "Char-TFIDF + LR"]
    indexed = data[["group_id", "language_column"]].reset_index().rename(columns={"index": "sample_index"})
    joined = sparse.merge(indexed, on=["group_id", "language_column"], validate="one_to_one")
    values = np.full(len(data), np.nan, dtype=np.float32)
    values[joined.sample_index.to_numpy()] = joined.prediction.to_numpy()
    if np.isnan(values).any():
        raise ValueError("Incomplete sparse confusion baseline")
    probabilities["Character TF-IDF + LR"] = values


def existing_figure(data):
    old = pd.read_csv(STATIC / "summary/static_multiseed_runs_with_selection.csv")
    retained = ["mBERT+Qwen validation-selected PCA", "mBERT+Qwen concat no PCA",
                "mBERT", "Qwen2.5+PCA-1024"]
    probabilities = ensemble_predictions(old[old.configuration.isin(retained)],
                                        STATIC / "predictions", len(data), True)
    late = np.full(len(data), np.nan, dtype=np.float32)
    for fold in range(1, 6):
        with np.load(FOLDS / f"fold_{fold}.npz") as split:
            indices = split["test"]
        scores = []
        for seed in (13, 21, 42, 87, 101):
            branch_scores = []
            for name in ("mBERT", "Qwen2.5 raw"):
                with np.load(STATIC / "predictions" / f"fold{fold}_seed{seed}_{slug(name)}.npz") as saved:
                    branch_scores.append(saved["probability"].astype(np.float32))
            scores.append(np.mean(branch_scores, axis=0).astype(np.float16).astype(np.float32))
        late[indices] = np.mean(scores, axis=0)
    probabilities["mBERT+Qwen Late fusion"] = late
    add_sparse(probabilities, data)
    confusion_figure(probabilities, data.label.to_numpy(dtype=np.int8))


def main():
    global FOLDS, STATIC, OUT, FIGURE
    parser = argparse.ArgumentParser()
    parser.add_argument("--variance-only", action="store_true")
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--fold-dir", type=Path, default=FOLDS)
    parser.add_argument("--static-dir", type=Path, default=STATIC)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--figure-output", type=Path, default=FIGURE)
    args = parser.parse_args()
    FOLDS, STATIC, OUT, FIGURE = args.fold_dir, args.static_dir, args.output_dir, args.figure_output
    OUT.mkdir(exist_ok=True)
    variance = variance_report()
    print(variance[variance.components == 1024].to_string(index=False))
    if args.variance_only:
        return
    if args.figure_only:
        existing_figure(pd.read_parquet(OUT / "processed_dataset.parquet"))
        return
    frame = pd.read_csv(OUT / "runs.csv")
    counts = frame.groupby("configuration").size()
    if len(counts) != 7 or not counts.eq(25).all():
        raise ValueError(f"Expected seven controls with 25 evaluations each, found {counts.to_dict()}")
    summary = frame.groupby("configuration")[[*METRICS, "train_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)
    data = pd.read_parquet(OUT / "processed_dataset.parquet")
    labels, groups = data.label.to_numpy(dtype=np.int8), data.group_id.to_numpy()
    old = pd.read_csv(STATIC / "summary/static_multiseed_runs_with_selection.csv")
    retained = ["mBERT+Qwen validation-selected PCA", "Llama-2+Qwen validation-selected PCA",
                "DistilBERT+mBERT PCA-1024", "mBERT+Qwen concat no PCA", "mBERT", "Qwen2.5+PCA-1024"]
    probabilities = ensemble_predictions(old[old.configuration.isin(retained)],
                                        STATIC / "predictions", len(data), True)
    probabilities.update(ensemble_predictions(frame, OUT / "predictions", len(data)))
    pairs = [(f"{family} validation-selected PCA", f"{family} {method}")
             for family in ("mBERT+Qwen", "Llama-2+Qwen")
             for method in ("Averaging", "MoE", "Late fusion")]
    diagnostic = "DistilBERT+mBERT PCA-1024 full vocabulary diagnostic"
    pairs.append((diagnostic, "DistilBERT+mBERT PCA-1024"))
    bootstrap = paired_bootstrap(probabilities, labels, groups, pairs)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    all_runs = pd.concat([frame, old], ignore_index=True)
    paired = []
    for a, b in pairs:
        left = all_runs[all_runs.configuration == a].set_index(["fold", "training_seed"]).mcc
        right = all_runs[all_runs.configuration == b].set_index(["fold", "training_seed"]).mcc
        difference = (left - right).rename("difference").reset_index()
        fold_means = difference.groupby("fold").difference.mean()
        margin = t.ppf(.975, 4) * fold_means.std(ddof=1) / np.sqrt(5)
        paired.append({"reference": a, "comparison": b, "mean_difference": difference.difference.mean(),
                       "fold_clustered_ci_low": fold_means.mean() - margin,
                       "fold_clustered_ci_high": fold_means.mean() + margin})
    pd.DataFrame(paired).to_csv(OUT / "paired_fold_ci.csv", index=False)
    add_sparse(probabilities, data)
    confusion_figure(probabilities, labels)
    (OUT / "verification.json").write_text(json.dumps({
        "controls": len(counts), "evaluations": len(frame), "folds": 5, "repetitions_per_fold": 5,
        "source_groups": int(np.unique(groups).size), "samples": len(data),
        "full_vocabulary_is_diagnostic_only": True,
        "late_fusion_reuses_matched_individual_branches": True,
    }, indent=2))
    print(summary[["mcc_mean", "mcc_std"]].to_string())
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
