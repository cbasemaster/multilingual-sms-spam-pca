"""Evaluate DistilBERT+mBERT Concat+PCA-1024 on the fixed grouped test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from grouped_embedding_fusion_experiment import (
    evaluate,
    texts_to_padded,
    train_one,
)
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
CONFIGURATION = "DistilBERT+mBERT Concat+PCA-1024"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    vocabulary = pd.read_csv(args.source_dir / "training_vocabulary.csv", keep_default_na=False)
    word_index = dict(zip(vocabulary["word"].astype(str), vocabulary["index"].astype(int)))

    max_length = min(
        int(data.iloc[train]["clean_text"].str.split().str.len().max()),
        args.max_length,
    )
    all_tokens = texts_to_padded(data["clean_text"], word_index, max_length)
    all_labels = data["label"].to_numpy(dtype=np.int64)

    matrix_path = args.output_dir / "embedding_distil_mbert_concat_pca1024.npy"
    variance_path = args.output_dir / "pca1024_explained_variance.csv"
    if matrix_path.exists():
        matrix = np.load(matrix_path)
    else:
        standardized = np.load(args.source_dir / "embedding_concat_standardized.npy", mmap_mode="r")
        pca = PCA(
            n_components=1024,
            svd_solver="randomized",
            iterated_power=4,
            random_state=SPLIT_SEED,
        )
        matrix = pca.fit_transform(standardized).astype(np.float32)
        np.save(matrix_path, matrix)
        pd.DataFrame(
            {
                "component": np.arange(1, 1025),
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            }
        ).to_csv(variance_path, index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.output_dir / "runs_checkpoint.csv"
    run_rows = pd.read_csv(checkpoint_path).to_dict("records") if checkpoint_path.exists() else []
    completed = {int(row["seed"]) for row in run_rows}

    for seed in SEEDS:
        prediction_path = args.output_dir / f"prediction_seed{seed}.csv"
        if seed in completed and prediction_path.exists():
            continue
        logits, costs = train_one(
            matrix,
            all_tokens[train],
            all_labels[train],
            all_tokens[validation],
            all_labels[validation],
            all_tokens[test],
            seed,
            device,
            args.batch_size,
            args.max_epochs,
        )
        metrics = evaluate(all_labels[test], logits)
        run_rows.append({"configuration": CONFIGURATION, "seed": seed, **metrics, **costs})
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        pd.DataFrame(
            {
                "configuration": CONFIGURATION,
                "seed": seed,
                "sample_index": test,
                "group_id": data.iloc[test]["group_id"].to_numpy(),
                "language_column": data.iloc[test]["language_column"].to_numpy(),
                "label": all_labels[test],
                "probability": probabilities,
                "prediction": (probabilities >= 0.5).astype(np.int8),
            }
        ).to_csv(prediction_path, index=False)
        pd.DataFrame(run_rows).to_csv(checkpoint_path, index=False)
        print(seed, metrics, costs, flush=True)

    runs = pd.DataFrame(run_rows).sort_values("seed")
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    metric_columns = [
        "accuracy",
        "spam_f1",
        "macro_f1",
        "mcc",
        "cohen_kappa",
        "roc_auc",
        "ece_15",
    ]
    summary = {"configuration": CONFIGURATION, "seeds": len(runs)}
    for metric in metric_columns:
        summary[f"{metric}_mean"] = runs[metric].mean()
        summary[f"{metric}_std"] = runs[metric].std(ddof=1)
    summary["input_dimension"] = 1536
    summary["final_dimension"] = 1024
    summary["cumulative_explained_variance"] = pd.read_csv(variance_path).iloc[-1][
        "cumulative_explained_variance"
    ]
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
