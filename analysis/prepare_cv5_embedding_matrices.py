"""Build fold-specific lookup matrices from frozen per-word encoder outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grouped_embedding_fusion_experiment import encode_vocabulary, fit_vocabulary
from grouped_qwen_embedding_fusion_experiment import extract_qwen
from revision_audit import load_long_dataset


def write_matrix(path: Path, words: list[str], word_to_position: dict[str, tuple[int, int]],
                 base: np.ndarray, extra: np.ndarray) -> None:
    if path.exists():
        old = np.load(path, mmap_mode="r")
        if old.shape != (len(words) + 1, base.shape[1]):
            raise ValueError(f"Stale matrix dimensions: {path}")
        return
    partial = path.with_suffix(".partial.npy")
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=base.dtype, shape=(len(words) + 1, base.shape[1])
    )
    result[0] = 0
    for offset in range(0, len(words), 2048):
        chunk = words[offset:offset + 2048]
        base_target, base_source, extra_target, extra_source = [], [], [], []
        for local_index, word in enumerate(chunk, start=offset + 1):
            location, source_index = word_to_position[word]
            if location == 0:
                base_target.append(local_index)
                base_source.append(source_index)
            else:
                extra_target.append(local_index)
                extra_source.append(source_index)
        if base_target:
            result[base_target] = base[base_source]
        if extra_target:
            result[extra_target] = extra[extra_source]
    result.flush()
    del result
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--base-vocabulary", type=Path, required=True)
    parser.add_argument("--base-mbert", type=Path, required=True)
    parser.add_argument("--base-distil", type=Path)
    parser.add_argument("--base-qwen", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    args = parser.parse_args()

    data = load_long_dataset(args.dataset)
    original_words = pd.read_csv(
        args.base_vocabulary, keep_default_na=False, dtype=str
    )["word"].tolist()
    base_mbert = np.load(args.base_mbert, mmap_mode="r")
    base_qwen = np.load(args.base_qwen, mmap_mode="r")
    if base_mbert.shape[0] != len(original_words) + 1 or base_qwen.shape[0] != len(original_words) + 1:
        raise ValueError("Base matrices do not match the base training vocabulary")
    base_index = {word: index for index, word in enumerate(original_words, start=1)}

    fold_words = {}
    missing = set()
    for fold in range(1, 6):
        fold_dir = args.fold_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with np.load(args.fold_dir / f"fold_{fold}.npz") as parts:
            train = parts["train"]
        _, words = fit_vocabulary(data.iloc[train]["clean_text"])
        fold_words[fold] = words
        missing.update(word for word in words if word not in base_index)
        pd.DataFrame({"word": words, "index": np.arange(1, len(words) + 1)}).to_csv(
            fold_dir / "training_vocabulary.csv", index=False
        )
        print(f"Fold {fold}: {len(words)} training words", flush=True)

    missing_words = sorted(missing)
    cache = args.fold_dir / "frozen_encoder_cache"
    cache.mkdir(parents=True, exist_ok=True)
    missing_list = cache / "missing_words.json"
    if missing_list.exists():
        if json.loads(missing_list.read_text(encoding="utf-8")) != missing_words:
            raise ValueError("Cached missing-word list differs from current folds")
    else:
        missing_list.write_text(json.dumps(missing_words, ensure_ascii=False), encoding="utf-8")
    print(f"Additional words across five folds: {len(missing_words)}", flush=True)
    extra_mbert_path = cache / "missing_mbert.npy"
    extra_qwen_path = cache / "missing_qwen.npy"
    encode_vocabulary(
        "bert-base-multilingual-uncased", missing_words, extra_mbert_path,
        torch.device("cuda"), batch_size=256,
    )
    extract_qwen(args.qwen_model, missing_words, extra_qwen_path, args.embedding_batch_size)
    extra_mbert = np.load(extra_mbert_path, mmap_mode="r")
    extra_qwen = np.load(extra_qwen_path, mmap_mode="r")
    if args.base_distil:
        base_distil = np.load(args.base_distil, mmap_mode="r")
        if base_distil.shape[0] != len(original_words) + 1:
            raise ValueError("DistilBERT matrix does not match the base vocabulary")
        extra_distil_path = cache / "missing_distil.npy"
        encode_vocabulary(
            "distilbert-base-multilingual-cased", missing_words, extra_distil_path,
            torch.device("cuda"), batch_size=256,
        )
        extra_distil = np.load(extra_distil_path, mmap_mode="r")
    extra_index = {word: index for index, word in enumerate(missing_words, start=1)}
    positions = {
        **{word: (0, index) for word, index in base_index.items()},
        **{word: (1, index) for word, index in extra_index.items()},
    }
    for fold, words in fold_words.items():
        fold_dir = args.fold_dir / f"fold_{fold}"
        write_matrix(fold_dir / "mbert.npy", words, positions, base_mbert, extra_mbert)
        write_matrix(fold_dir / "qwen.npy", words, positions, base_qwen, extra_qwen)
        if args.base_distil:
            write_matrix(fold_dir / "distil.npy", words, positions, base_distil, extra_distil)
        print(f"Fold {fold}: embedding matrices ready", flush=True)


if __name__ == "__main__":
    main()
