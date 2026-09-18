"""Evaluate the unprojected three-source token table on the fixed grouped split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import infer_state, train_extended
from revision_audit import SEEDS, load_long_dataset, split_groups


METRICS = ("accuracy", "spam_f1", "macro_f1", "mcc", "cohen_kappa", "roc_auc", "ece_15")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embedding-concat", type=Path, required=True)
    parser.add_argument("--reference-vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-batch-size", type=int, default=256)
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, 42)
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    reference_words = pd.read_csv(args.reference_vocabulary, keep_default_na=False)["word"].tolist()
    if words != reference_words:
        raise ValueError("Vocabulary order does not match the archived embedding table")
    embedding = np.load(args.embedding_concat, mmap_mode="r")
    if embedding.shape != (len(words) + 1, 5376):
        raise ValueError(f"Unexpected embedding table shape: {embedding.shape}")
    tokens = texts_to_padded(data["clean_text"], word_index, args.max_length)
    labels = data["label"].to_numpy(dtype=np.int64)

    runs_path = args.output_dir / "runs.csv"
    records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {int(record["seed"]) for record in records}
    for seed in SEEDS:
        if seed in completed:
            continue
        state, _, validation_metrics, costs = train_extended(
            embedding, tokens[train], labels[train], tokens[validation], labels[validation],
            seed, args.physical_batch_size, args.effective_batch_size, args.max_epochs,
        )
        logits, inference_ms = infer_state(
            embedding, state, tokens[test], args.physical_batch_size * 2,
        )
        test_metrics = evaluate(labels[test], logits)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        pd.DataFrame({
            "sample_index": test,
            "group_id": data.iloc[test]["group_id"].to_numpy(),
            "label": labels[test],
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(np.int8),
        }).to_csv(args.output_dir / f"prediction_seed{seed}.csv", index=False)
        records.append({
            "configuration": "mBERT+Qwen+CharTFIDF concat without PCA",
            "seed": seed,
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
            **test_metrics,
            **costs,
            "inference_ms_per_message": inference_ms,
        })
        pd.DataFrame(records).sort_values("seed").to_csv(runs_path, index=False)
        print(f"seed={seed} test_mcc={test_metrics['mcc']:.6f}", flush=True)

    frame = pd.DataFrame(records)
    if set(frame["seed"].astype(int)) != set(SEEDS):
        raise ValueError("The five-seed control is incomplete")
    summary = frame[list(METRICS)].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
