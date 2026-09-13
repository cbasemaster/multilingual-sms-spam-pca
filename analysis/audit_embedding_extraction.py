"""Measure and verify the offline vocabulary-embedding extraction stage."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from grouped_embedding_fusion_experiment import ENCODERS, MAX_SUBTOKENS


@torch.inference_mode()
def audit_model(
    model_name: str,
    words: list[str],
    saved_matrix_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    load_seconds = time.perf_counter() - start
    saved = np.load(saved_matrix_path, mmap_mode="r")
    unknown_only = 0
    truncated = 0
    max_absolute_difference = 0.0
    sum_absolute_difference = 0.0
    compared_values = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    for offset in range(0, len(words), batch_size):
        batch_words = words[offset : offset + batch_size]
        full_lengths = tokenizer(
            batch_words,
            add_special_tokens=False,
            return_length=True,
            truncation=False,
        )["length"]
        truncated += sum(length > MAX_SUBTOKENS for length in full_lengths)
        encoded = tokenizer(
            batch_words,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=MAX_SUBTOKENS,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if tokenizer.unk_token_id is not None:
            for ids, mask in zip(input_ids, encoded["attention_mask"]):
                active = ids[mask.bool()]
                unknown_only += int(len(active) > 0 and torch.all(active == tokenizer.unk_token_id))
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        current = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu().numpy()
        reference = np.asarray(saved[offset + 1 : offset + 1 + len(batch_words)])
        difference = np.abs(current - reference)
        max_absolute_difference = max(max_absolute_difference, float(difference.max()))
        sum_absolute_difference += float(difference.sum())
        compared_values += difference.size

    if device.type == "cuda":
        torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1_000_000 if device.type == "cuda" else float("nan")
    )
    result = {
        "model": model_name,
        "vocabulary_words": len(words),
        "dimension": int(saved.shape[1]),
        "unknown_only_words": unknown_only,
        "truncated_over_32_subtokens": truncated,
        "model_and_tokenizer_load_seconds": load_seconds,
        "embedding_extraction_seconds": extraction_seconds,
        "words_per_second": len(words) / extraction_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "saved_matrix_mb": saved.nbytes / 1_000_000,
        "repeat_max_absolute_difference": max_absolute_difference,
        "repeat_mean_absolute_difference": sum_absolute_difference / compared_values,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    vocabulary = pd.read_csv(
        args.experiment_dir / "training_vocabulary.csv",
        dtype={"word": str},
        keep_default_na=False,
    )
    words = [str(word) for word in vocabulary["word"].tolist()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in ENCODERS:
        path = args.experiment_dir / f"embedding_{model_name.replace('/', '--')}.npy"
        result = audit_model(model_name, words, path, device, args.batch_size)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    (args.experiment_dir / "embedding_extraction_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
