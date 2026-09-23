"""Run every Table 7 static neural configuration over five folds and seeds."""

from __future__ import annotations

import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import infer_state, train_extended
from prepare_cv5_llama_embeddings import load_fold_words
from revision_audit import load_long_dataset
from run_cv5_llama_fusion import load_llama_matrix
from run_cv5_triple_fusion import standardized_triple


SEEDS = (13, 21, 42, 87, 101)


def slug(value: str) -> str:
    return re.sub(r"_+", "_", "".join(
        char.lower() if char.isalnum() else "_" for char in value
    )).strip("_")


def fusion_core(fold_path: Path) -> list[tuple[str, object]]:
    core = fold_path / "fusion_core"
    mbert = np.load(fold_path / "mbert.npy", mmap_mode="r")
    distil = np.load(fold_path / "distil.npy", mmap_mode="r")
    qwen = np.load(fold_path / "qwen.npy", mmap_mode="r")
    pca = np.load(core / "fusion_pca2048.npy", mmap_mode="r")
    char_pca = np.load(core / "three_source_pca2048.npy", mmap_mode="r")
    distil_pca = np.load(core / "distil_mbert_pca1024.npy", mmap_mode="r")
    mbert_pca = np.load(core / "mbert_pca512.npy", mmap_mode="r")
    return [
        ("mBERT", mbert),
        ("DistilBERT", distil),
        ("Qwen2.5 raw", qwen),
        ("Qwen2.5+PCA-1024", np.load(core / "qwen_pca1024.npy", mmap_mode="r")),
        ("mBERT+Qwen concat no PCA", np.load(core / "standardized_concat.npy", mmap_mode="r")),
        ("mBERT+Qwen RP-1024", np.load(core / "fusion_rp1024.npy", mmap_mode="r")),
        *[(f"mBERT+Qwen PCA-{width}", pca[:, :width])
          for width in (768, 1024, 1536, 2048)],
        ("mBERT+Qwen+Char concat no PCA", np.load(core / "three_source_concat.npy", mmap_mode="r")),
        *[(f"mBERT+Qwen+Char PCA-{width}", char_pca[:, :width])
          for width in (768, 1024, 1536, 2048)],
        ("DistilBERT+mBERT raw concat", np.load(core / "distil_mbert_raw_concat.npy", mmap_mode="r")),
        ("DistilBERT+mBERT standardized concat", np.load(core / "distil_mbert_standardized_concat.npy", mmap_mode="r")),
        ("DistilBERT+mBERT RP-256", np.load(core / "distil_mbert_rp256.npy", mmap_mode="r")),
        *[(f"DistilBERT+mBERT PCA-{width}", distil_pca[:, :width])
          for width in (128, 256, 1024)],
        ("mBERT PCA-256", mbert_pca[:, :256]),
        ("mBERT PCA-512", mbert_pca),
    ]


def llama_family(fold_root: Path, fold_path: Path, words: list[str]) -> list[tuple[str, object]]:
    out = fold_path / "llama_fusion"
    llama = load_llama_matrix(fold_root, words)
    pca = np.load(out / "llama_qwen_pca2048.npy", mmap_mode="r")
    return [
        ("Llama-2 raw", llama),
        ("Llama-2 PCA-1024", np.load(out / "llama_pca1024.npy", mmap_mode="r")),
        ("Llama-2+Qwen concat no PCA", np.load(out / "llama_qwen_standardized_concat.npy", mmap_mode="r")),
        *[(f"Llama-2+Qwen PCA-{width}", pca[:, :width])
          for width in (1024, 1536, 2048)],
    ]


def triple_family(fold_root: Path, fold_path: Path, words: list[str]) -> tuple[list[tuple[str, object]], Path]:
    out = fold_path / "triple_fusion"
    temporary = out / "standardized_concat_multiseed.npy"
    matrix = standardized_triple(
        (np.load(fold_path / "mbert.npy", mmap_mode="r"),
         np.load(fold_path / "qwen.npy", mmap_mode="r"),
         load_llama_matrix(fold_root, words)),
        temporary,
    )
    pca = np.load(out / "pca2048.npy", mmap_mode="r")
    return ([
        ("mBERT+Qwen2.5+Llama-2 concat no PCA", matrix),
        *[(f"mBERT+Qwen2.5+Llama-2 PCA-{width}", pca[:, :width])
          for width in (1024, 1536, 2048)],
    ], temporary)


def existing_seed42(fold_path: Path, family: str, name: str) -> tuple[dict, np.ndarray]:
    source = fold_path / family
    frame = pd.read_csv(source / "runs.csv")
    row = frame.loc[frame["configuration"] == name]
    if len(row) != 1:
        raise ValueError(f"Cannot locate seed-42 result: {family}: {name}")
    prediction = pd.read_csv(source / f"predictions_{slug(name)}.csv")["probability"].to_numpy()
    return row.iloc[0].to_dict(), prediction


def save_checkpoint(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).sort_values(["fold", "training_seed", "configuration"]).to_csv(
        path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    checkpoint = args.output_dir / "static_multiseed_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(int(row["fold"]), int(row["training_seed"]), row["configuration"])
                 for row in rows}

    data = load_long_dataset(args.dataset)
    labels = data["label"].to_numpy(dtype=np.int64)
    recorded_words = load_fold_words(args.fold_dir)

    for fold in args.folds:
        fold_path = args.fold_dir / f"fold_{fold}"
        with np.load(args.fold_dir / f"fold_{fold}.npz") as split:
            train, validation, test = (split[key] for key in ("train", "validation", "test"))
        word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
        if words != recorded_words[fold]:
            raise ValueError(f"Fold {fold} training vocabulary mismatch")
        tokens = texts_to_padded(data["clean_text"], word_index, 128)
        families = [
            ("fusion_core", fusion_core(fold_path), None),
            ("llama_fusion", llama_family(args.fold_dir, fold_path, words), None),
        ]
        triples, temporary = triple_family(args.fold_dir, fold_path, words)
        families.append(("triple_fusion", triples, temporary))

        try:
            for family_name, configurations, _ in families:
                for name, matrix in configurations:
                    for seed in args.seeds:
                        key = (fold, seed, name)
                        if key in completed:
                            continue
                        pred_path = prediction_dir / f"fold{fold}_seed{seed}_{slug(name)}.npz"
                        if seed == 42:
                            result, probability = existing_seed42(fold_path, family_name, name)
                            result.update({"fold": fold, "training_seed": seed,
                                           "configuration": name, "reused_seed42": True})
                        else:
                            width = matrix.shape[1]
                            physical = 64 if width > 4096 else 256
                            state, _, validation_metrics, costs = train_extended(
                                matrix, tokens[train], labels[train],
                                tokens[validation], labels[validation],
                                seed, physical, 1024, args.max_epochs,
                            )
                            logits, inference_ms = infer_state(
                                matrix, state, tokens[test], 128 if width > 4096 else 512
                            )
                            probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                            result = {
                                "fold": fold,
                                "training_seed": seed,
                                "configuration": name,
                                "input_dimension": width,
                                "final_dimension": width,
                                **{f"validation_{metric}": value
                                   for metric, value in validation_metrics.items()},
                                **evaluate(labels[test], logits),
                                **costs,
                                "inference_ms_per_message": inference_ms,
                                "reused_seed42": False,
                            }
                            del state, logits
                        np.savez_compressed(
                            pred_path,
                            probability=np.asarray(probability, dtype=np.float16),
                        )
                        rows.append(result)
                        completed.add(key)
                        save_checkpoint(rows, checkpoint)
                        print(
                            f"Fold {fold} seed {seed}: {name}: MCC {float(result['mcc']):.4f}",
                            flush=True,
                        )
                        gc.collect()
        finally:
            matrix = None
            configurations = None
            del families, triples
            gc.collect()
            if temporary.exists():
                for attempt in range(10):
                    try:
                        temporary.unlink()
                        break
                    except PermissionError:
                        if attempt == 9:
                            raise
                        gc.collect()
                        time.sleep(1)
        print(f"Fold {fold} complete", flush=True)


if __name__ == "__main__":
    main()
