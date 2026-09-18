"""Evaluate Llama-2 and Llama-2/Qwen2.5 PCA controls on an existing test fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import (
    fit_pca_max,
    infer_state,
    make_standardized_concat,
    train_extended,
)
from prepare_cv5_llama_embeddings import (
    REPRESENTATION,
    load_fold_words,
    union_words,
    vocabulary_hash,
)
from revision_audit import load_long_dataset


CONFIGURATIONS = (
    "Llama-2 raw",
    "Llama-2 PCA-1024",
    "Llama-2+Qwen concat no PCA",
    "Llama-2+Qwen PCA-1024",
    "Llama-2+Qwen PCA-1536",
    "Llama-2+Qwen PCA-2048",
)


class FoldMatrix:
    """Read fold-ordered vectors from one shared union matrix."""

    def __init__(self, union: np.ndarray, union_words_list: list[str],
                 fold_words: list[str]) -> None:
        if union.shape != (len(union_words_list) + 1, 4096):
            raise ValueError("Llama union matrix and word list do not align")
        lookup = {word: index for index, word in enumerate(union_words_list, 1)}
        self.union = union
        self.index = np.array([0] + [lookup[word] for word in fold_words], dtype=np.int64)
        self.shape = (len(self.index), union.shape[1])
        self.dtype = union.dtype
        self.nbytes = self.shape[0] * self.shape[1] * self.dtype.itemsize

    def __getitem__(self, rows):
        return self.union[self.index[rows]]

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.union[self.index], dtype=dtype)
        return result.copy() if copy else result


def load_llama_matrix(fold_dir: Path, fold_words: list[str]) -> FoldMatrix:
    cache = fold_dir / "llama_cache"
    words = json.loads((cache / "union_words.json").read_text(encoding="utf-8"))
    metadata = json.loads((cache / "llama_union.json").read_text(encoding="utf-8"))
    if metadata["vocabulary_sha256"] != vocabulary_hash(words):
        raise ValueError("Llama vocabulary digest does not match extracted matrix")
    if metadata.get("representation") != REPRESENTATION:
        raise ValueError("Llama matrix must contain raw input-token embeddings")
    union = np.load(cache / "llama_union.npy", mmap_mode="r")
    return FoldMatrix(union, words, fold_words)


def materialize(name: str, llama: FoldMatrix, qwen: np.ndarray,
                output_dir: Path) -> tuple[np.ndarray | FoldMatrix, dict | None]:
    if name == "Llama-2 raw":
        return llama, None
    if name == "Llama-2 PCA-1024":
        return fit_pca_max(llama, output_dir / "llama_pca1024.npy", 1024)
    concat = make_standardized_concat(
        llama, qwen, output_dir / "llama_qwen_standardized_concat.npy"
    )
    if name == "Llama-2+Qwen concat no PCA":
        return concat, None
    pca, details = fit_pca_max(
        concat, output_dir / "llama_qwen_pca2048.npy", 2048
    )
    width = int(name.rsplit("-", 1)[1])
    return pca[:, :width], details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--only-index", type=int, choices=range(1, 7))
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    fold_path = args.fold_dir / f"fold_{args.fold}"
    output_dir = fold_path / "llama_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_long_dataset(args.dataset)
    with np.load(args.fold_dir / f"fold_{args.fold}.npz") as parts:
        train, validation, test = (parts[key] for key in ("train", "validation", "test"))
    if set(data.iloc[train]["group_id"]) & set(data.iloc[test]["group_id"]):
        raise ValueError("Train and test source groups overlap")
    if set(data.iloc[validation]["group_id"]) & set(data.iloc[test]["group_id"]):
        raise ValueError("Validation and test source groups overlap")
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    recorded = load_fold_words(args.fold_dir)[args.fold]
    if words != recorded:
        raise ValueError("Fold training vocabulary differs from saved vocabulary")
    tokens = texts_to_padded(data["clean_text"], word_index, 128)
    labels = data["label"].to_numpy(dtype=np.int64)
    llama = load_llama_matrix(args.fold_dir, words)
    qwen = np.load(fold_path / "qwen.npy", mmap_mode="r")
    if qwen.shape != (len(words) + 1, 3584):
        raise ValueError("Qwen matrix does not align to the fold vocabulary")

    runs_path = output_dir / "runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {row["configuration"] for row in runs}
    selected = ([CONFIGURATIONS[args.only_index - 1]] if args.only_index
                else CONFIGURATIONS)
    for name in selected:
        if name in completed:
            print(f"Fold {args.fold}: already complete: {name}", flush=True)
            continue
        matrix, pca_details = materialize(name, llama, qwen, output_dir)
        width = matrix.shape[1]
        physical_batch_size = 64 if width > 4096 else 256
        state, _, validation_metrics, costs = train_extended(
            matrix, tokens[train], labels[train],
            tokens[validation], labels[validation],
            args.training_seed, physical_batch_size, 1024, args.max_epochs,
        )
        logits, inference_ms = infer_state(matrix, state, tokens[test], 512 if width <= 4096 else 128)
        test_metrics = evaluate(labels[test], logits)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
        pd.DataFrame({
            "fold": args.fold,
            "configuration": name,
            "llama_representation": REPRESENTATION,
            "row_index": test,
            "group_id": data.iloc[test]["group_id"].to_numpy(),
            "label": labels[test],
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(np.int8),
        }).to_csv(output_dir / f"predictions_{slug}.csv", index=False)
        runs.append({
            "fold": args.fold,
            "training_seed": args.training_seed,
            "configuration": name,
            "input_dimension": 7680 if "+Qwen" in name else 4096,
            "final_dimension": width,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **test_metrics,
            **costs,
            "inference_ms_per_message": inference_ms,
            "pca_retained_variance": (
                sum(pca_details["eigenvalues"][:width])
                / pca_details["total_variance"]
                if pca_details is not None and "PCA-" in name else np.nan
            ),
        })
        pd.DataFrame(runs).to_csv(runs_path, index=False)
        print(f"Fold {args.fold}: {name}: test MCC {test_metrics['mcc']:.4f}", flush=True)

    frame = pd.DataFrame(runs)
    candidates = frame[frame["configuration"].str.match(r"Llama-2\+Qwen PCA-\d+$")].copy()
    if len(candidates) == 3:
        chosen = candidates.sort_values(
            ["validation_mcc", "final_dimension"], ascending=[False, True]
        ).iloc[0]
        (output_dir / "metadata.json").write_text(json.dumps({
            "protocol": "stratified five-fold source-group cross-validation",
            "llama_representation": REPRESENTATION,
            "fold": args.fold,
            "training_seed": args.training_seed,
            "selection_rule": "highest validation MCC; smaller dimension breaks ties",
            "selected_dimension": int(chosen["final_dimension"]),
            "selected_test_mcc": float(chosen["mcc"]),
        }, indent=2), encoding="utf-8")
        print(f"Fold {args.fold}: selected PCA-{int(chosen['final_dimension'])}", flush=True)


if __name__ == "__main__":
    main()
