"""Token-aligned mBERT + Qwen + character TF-IDF Concat+PCA CNN.

Character TF-IDF is computed for each entry of the training-only word
vocabulary, so its rows align with the mBERT and Qwen vocabulary matrices.
The three source blocks are concatenated, jointly projected with PCA, and then
used as a frozen token table by the same token-axis CNN protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import fit_pca_max, infer_state, train_extended
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
DEFAULT_DIMS = (768, 1024, 1536, 2048)


def make_three_source_concat(
    embedding_concat: np.ndarray,
    words: list[str],
    output_path: Path,
    metadata_path: Path,
    char_dimension: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, dict]:
    if output_path.exists() and metadata_path.exists():
        return np.load(output_path, mmap_mode="r"), json.loads(metadata_path.read_text())

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=char_dimension,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_sparse = vectorizer.fit_transform(words)
    scaler = StandardScaler(with_mean=False, copy=False)
    char_sparse = scaler.fit_transform(char_sparse).astype(np.float32)
    actual_dimension = char_sparse.shape[1]

    partial = output_path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(embedding_concat.shape[0], embedding_concat.shape[1] + actual_dimension),
    )
    output[0] = 0
    for start in range(1, embedding_concat.shape[0], chunk_size):
        stop = min(start + chunk_size, embedding_concat.shape[0])
        output[start:stop, : embedding_concat.shape[1]] = embedding_concat[start:stop]
        output[start:stop, embedding_concat.shape[1] :] = char_sparse[
            start - 1 : stop - 1
        ].toarray().astype(np.float16)
    output.flush()
    del output
    partial.replace(output_path)

    metadata = {
        "source_blocks": [
            "column-standardized mBERT token vectors",
            "column-standardized Qwen2.5 token vectors",
            "variance-scaled character TF-IDF token vectors",
        ],
        "embedding_dimension": int(embedding_concat.shape[1]),
        "character_tfidf_dimension": int(actual_dimension),
        "concatenated_dimension": int(embedding_concat.shape[1] + actual_dimension),
        "character_analyzer": "char_wb",
        "character_ngram_range": [2, 5],
        "character_min_df": 2,
        "fit_scope": "training-only vocabulary entries",
        "construction_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del char_sparse, scaler, vectorizer
    gc.collect()
    return np.load(output_path, mmap_mode="r"), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embedding-concat", type=Path, required=True)
    parser.add_argument("--reference-vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--char-dimension", type=int, default=1024)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--physical-batch-size", type=int, default=256)
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    reference_words = pd.read_csv(args.reference_vocabulary, keep_default_na=False)["word"].tolist()
    if words != reference_words:
        raise ValueError("Vocabulary order does not match the mBERT/Qwen table.")
    tokens = texts_to_padded(data["clean_text"], word_index, args.max_length)
    labels = data["label"].to_numpy(dtype=np.int64)

    embedding_concat = np.load(args.embedding_concat, mmap_mode="r")
    if embedding_concat.shape[0] != len(words) + 1:
        raise ValueError("Embedding table and training-only vocabulary do not align.")
    combined, concat_metadata = make_three_source_concat(
        embedding_concat,
        words,
        args.output_dir / f"embedding_mbert_qwen_char{args.char_dimension}_concat.npy",
        args.output_dir / f"embedding_mbert_qwen_char{args.char_dimension}_concat.json",
        args.char_dimension,
    )

    max_dimension = max(args.dims)
    pca_matrix, pca_metadata = fit_pca_max(
        combined,
        args.output_dir / f"embedding_mbert_qwen_char_concat_pca{max_dimension}_max.npy",
        max_dimension,
    )

    selection_path = args.output_dir / "dimension_selection.csv"
    selection = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
    completed_dimensions = {int(row["dimension"]) for row in selection}
    eigenvalues = np.asarray(pca_metadata["eigenvalues"])
    for dimension in args.dims:
        if dimension in completed_dimensions:
            continue
        state, _, validation_metrics, costs = train_extended(
            pca_matrix[:, :dimension],
            tokens[train],
            labels[train],
            tokens[validation],
            labels[validation],
            SPLIT_SEED,
            args.physical_batch_size,
            args.effective_batch_size,
            args.max_epochs,
        )
        torch.save(state, args.output_dir / f"selection_state_pca{dimension}.pt")
        selection.append({
            "dimension": dimension,
            "retained_variance": float(eigenvalues[:dimension].sum() / pca_metadata["total_variance"]),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **costs,
        })
        pd.DataFrame(selection).sort_values("dimension").to_csv(selection_path, index=False)
        print("selection", dimension, validation_metrics, flush=True)

    selection_frame = pd.DataFrame(selection)
    selected_dimension = int(
        selection_frame.sort_values(
            ["validation_mcc", "dimension"], ascending=[False, True]
        ).iloc[0]["dimension"]
    )
    selected_embedding = pca_matrix[:, :selected_dimension]

    runs_path = args.output_dir / "final_runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed_seeds = {int(row["seed"]) for row in runs}
    for seed in SEEDS:
        if seed in completed_seeds:
            continue
        state, _, validation_metrics, costs = train_extended(
            selected_embedding,
            tokens[train],
            labels[train],
            tokens[validation],
            labels[validation],
            seed,
            args.physical_batch_size,
            args.effective_batch_size,
            args.max_epochs,
        )
        torch.save(state, args.output_dir / f"final_state_seed{seed}.pt")
        logits, inference_ms = infer_state(
            selected_embedding, state, tokens[test], args.physical_batch_size * 2
        )
        metrics = evaluate(labels[test], logits)
        runs.append({
            "configuration": f"mBERT+Qwen+CharTFIDF Concat+PCA-{selected_dimension}",
            "seed": seed,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **metrics,
            **costs,
            "inference_ms_per_message": inference_ms,
        })
        pd.DataFrame(runs).to_csv(runs_path, index=False)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        pd.DataFrame({
            "row_index": test,
            "group_id": data.iloc[test]["group_id"].to_numpy(),
            "label": labels[test],
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(np.int8),
        }).to_csv(args.output_dir / f"prediction_seed{seed}.csv", index=False)
        print("final", seed, metrics, flush=True)

    runs_frame = pd.DataFrame(runs)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "mcc", "roc_auc", "ece_15"]
    summary = runs_frame[metric_columns].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary.to_csv(args.output_dir / "final_summary.csv")
    metadata = {
        **concat_metadata,
        "pca_candidate_dimensions": list(args.dims),
        "selected_pca_dimension": selected_dimension,
        "selection_rule": "highest validation MCC; smaller dimension breaks ties",
        "training_seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "cnn_axis": "token positions",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
