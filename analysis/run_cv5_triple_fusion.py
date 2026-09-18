"""Matched source-group evaluation of frozen mBERT, Qwen, and Llama vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import (
    fit_pca_max, infer_state, moments_without_padding, train_extended,
)
from prepare_cv5_llama_embeddings import load_fold_words
from revision_audit import load_long_dataset
from run_cv5_llama_fusion import load_llama_matrix


FAMILY = "mBERT+Qwen2.5+Llama-2"
WIDTHS = (1024, 1536, 2048)
CONFIGURATIONS = (f"{FAMILY} concat no PCA",) + tuple(
    f"{FAMILY} PCA-{width}" for width in WIDTHS
)


def standardized_triple(matrices, path: Path, chunk_size: int = 512):
    rows = matrices[0].shape[0]
    if any(matrix.shape[0] != rows for matrix in matrices):
        raise ValueError("Training-vocabulary matrices must have matching rows")
    columns = sum(matrix.shape[1] for matrix in matrices)
    if path.exists():
        result = np.load(path, mmap_mode="r")
        if result.shape != (rows, columns) or result.dtype != np.float16:
            raise ValueError("Existing triple cache has the wrong shape or type")
        return result
    moments = [moments_without_padding(matrix) for matrix in matrices]
    partial = path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(rows, columns)
    )
    output[0] = 0
    for start in range(1, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        offset = 0
        for matrix, (mean, std) in zip(matrices, moments):
            width = matrix.shape[1]
            values = np.asarray(matrix[start:stop], dtype=np.float32)
            output[start:stop, offset:offset + width] = (
                (values - mean) / std
            ).astype(np.float16)
            offset += width
    output.flush()
    del output
    partial.replace(path)
    return np.load(path, mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    fold_path = args.fold_dir / f"fold_{args.fold}"
    output_dir = fold_path / "triple_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_long_dataset(args.dataset)
    with np.load(args.fold_dir / f"fold_{args.fold}.npz") as parts:
        train, validation, test = (
            parts[key] for key in ("train", "validation", "test")
        )
    if set(data.iloc[train]["group_id"]) & set(data.iloc[test]["group_id"]):
        raise ValueError("Train and test source groups overlap")
    if set(data.iloc[validation]["group_id"]) & set(data.iloc[test]["group_id"]):
        raise ValueError("Validation and test source groups overlap")
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    if words != load_fold_words(args.fold_dir)[args.fold]:
        raise ValueError("Training vocabulary differs from saved fold vocabulary")
    tokens = texts_to_padded(data["clean_text"], word_index, 128)
    labels = data["label"].to_numpy(dtype=np.int64)
    mbert = np.load(fold_path / "mbert.npy", mmap_mode="r")
    qwen = np.load(fold_path / "qwen.npy", mmap_mode="r")
    llama = load_llama_matrix(args.fold_dir, words)
    expected = (len(words) + 1,)
    if (mbert.shape != (*expected, 768) or qwen.shape != (*expected, 3584)
            or llama.shape != (*expected, 4096)):
        raise ValueError("Source dimensions or vocabulary alignment differ")
    matrix = standardized_triple(
        (mbert, qwen, llama), output_dir / "standardized_concat.npy"
    )
    if not np.all(matrix[0] == 0):
        raise ValueError("Padding vector must remain zero")

    runs_path = output_dir / "runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {row["configuration"] for row in runs}
    projected = None
    details = None
    for name in CONFIGURATIONS:
        if name in completed:
            print(f"Fold {args.fold}: already complete: {name}", flush=True)
            continue
        if "PCA-" in name:
            if projected is None:
                projected, details = fit_pca_max(
                    matrix, output_dir / "pca2048.npy", max(WIDTHS)
                )
            width = int(name.rsplit("-", 1)[1])
            vectors = projected[:, :width]
        else:
            width = matrix.shape[1]
            vectors = matrix
        physical_batch_size = 64 if width > 4096 else 256
        state, _, validation_metrics, costs = train_extended(
            vectors, tokens[train], labels[train],
            tokens[validation], labels[validation], args.training_seed,
            physical_batch_size, 1024, args.max_epochs,
        )
        logits, inference_ms = infer_state(
            vectors, state, tokens[test], 128 if width > 4096 else 512
        )
        test_metrics = evaluate(labels[test], logits)
        probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
        pd.DataFrame({
            "fold": args.fold,
            "configuration": name,
            "row_index": test,
            "group_id": data.iloc[test]["group_id"].to_numpy(),
            "label": labels[test],
            "probability": probability,
            "prediction": (probability >= 0.5).astype(np.int8),
        }).to_csv(output_dir / f"predictions_{slug}.csv", index=False)
        runs.append({
            "fold": args.fold,
            "training_seed": args.training_seed,
            "configuration": name,
            "input_dimension": 8448,
            "final_dimension": width,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **test_metrics,
            **costs,
            "inference_ms_per_message": inference_ms,
            "pca_retained_variance": (
                sum(details["eigenvalues"][:width]) / details["total_variance"]
                if details is not None and "PCA-" in name else np.nan
            ),
        })
        pd.DataFrame(runs).to_csv(runs_path, index=False)
        print(f"Fold {args.fold}: {name}: test MCC {test_metrics['mcc']:.4f}", flush=True)

    frame = pd.DataFrame(runs)
    candidates = frame[frame["configuration"].isin(CONFIGURATIONS[1:])]
    if len(candidates) == len(WIDTHS):
        chosen = candidates.sort_values(
            ["validation_mcc", "final_dimension"], ascending=[False, True]
        ).iloc[0]
        (output_dir / "metadata.json").write_text(json.dumps({
            "protocol": "stratified five-fold source-group cross-validation",
            "sources": ["mBERT frozen vectors", "Qwen2.5 frozen vectors",
                        "Llama-2 raw input-token vectors"],
            "fold": args.fold,
            "training_seed": args.training_seed,
            "selection_rule": "highest validation MCC; smaller dimension breaks ties",
            "selected_dimension": int(chosen["final_dimension"]),
            "selected_test_mcc": float(chosen["mcc"]),
        }, indent=2), encoding="utf-8")
        print(f"Fold {args.fold}: selected PCA-{int(chosen['final_dimension'])}", flush=True)


if __name__ == "__main__":
    main()
