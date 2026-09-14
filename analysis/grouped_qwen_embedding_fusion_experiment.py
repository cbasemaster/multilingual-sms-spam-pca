"""Run the grouped static-fusion evaluation with the historical Qwen encoder.

Qwen2.5-7B-Instruct is loaded in NF4 only for the one-time vocabulary
embedding extraction. The resulting frozen table is then evaluated with the
same grouped split and CNN protocol used by grouped_embedding_fusion_experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from grouped_embedding_fusion_experiment import (
    evaluate,
    fit_vocabulary,
    texts_to_padded,
    train_one,
)
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
MAX_SUBTOKENS = 32
QWEN_LABEL = "Qwen2.5-7B-Instruct+PCA-768 (NF4 extraction)"


def matrix_moments(matrix: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    total_sq = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk_size):
        values = np.asarray(matrix[start : start + chunk_size], dtype=np.float32)
        total += values.sum(axis=0, dtype=np.float64)
        total_sq += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = total / matrix.shape[0]
    variance = np.maximum(total_sq / matrix.shape[0] - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def standardized_concat(
    left: np.ndarray,
    right: np.ndarray,
    output_path: Path,
    chunk_size: int = 1024,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path, mmap_mode="r")
    left_mean, left_std = matrix_moments(left)
    right_mean, right_std = matrix_moments(right)
    partial = output_path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(left.shape[0], left.shape[1] + right.shape[1]),
    )
    for start in range(0, left.shape[0], chunk_size):
        stop = min(start + chunk_size, left.shape[0])
        output[start:stop, : left.shape[1]] = (
            (np.asarray(left[start:stop], dtype=np.float32) - left_mean) / left_std
        ).astype(np.float16)
        output[start:stop, left.shape[1] :] = (
            (np.asarray(right[start:stop], dtype=np.float32) - right_mean) / right_std
        ).astype(np.float16)
    output.flush()
    del output
    partial.replace(output_path)
    return np.load(output_path, mmap_mode="r")


@torch.inference_mode()
def extract_qwen(
    model_path: Path,
    words: list[str],
    output_path: Path,
    batch_size: int,
) -> dict[str, float | int | str | bool]:
    if output_path.exists():
        matrix = np.load(output_path, mmap_mode="r")
        return {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "rows": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "quantization": "NF4, double quantization, FP16 compute",
            "cached": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        device_map={"": 0},
        quantization_config=quantization,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = False
    dimension = int(model.config.hidden_size)
    partial_path = output_path.with_suffix(".partial.npy")
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if partial_path.exists() and checkpoint_path.exists():
        matrix = np.lib.format.open_memmap(partial_path, mode="r+")
        offset = int(json.loads(checkpoint_path.read_text(encoding="utf-8"))["offset"])
    else:
        matrix = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(words) + 1, dimension),
        )
        matrix[0] = 0
        offset = 0

    unknown_only = 0
    truncated = 0
    started = time.perf_counter()
    for start in range(offset, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        lengths = tokenizer(
            batch_words,
            add_special_tokens=False,
            return_length=True,
            truncation=False,
        )["length"]
        truncated += sum(length > MAX_SUBTOKENS for length in lengths)
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
        encoded = {key: value.to("cuda", non_blocking=True) for key, value in encoded.items()}
        hidden = model(**encoded, use_cache=False).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        stop = start + len(batch_words)
        matrix[start + 1 : stop + 1] = pooled.to(torch.float16).cpu().numpy()
        if stop % (batch_size * 20) == 0 or stop == len(words):
            matrix.flush()
            checkpoint_path.write_text(json.dumps({"offset": stop}), encoding="utf-8")
            print(f"Qwen extraction: {stop}/{len(words)}", flush=True)

    elapsed = time.perf_counter() - started
    matrix.flush()
    del matrix, model
    gc.collect()
    torch.cuda.empty_cache()
    partial_path.replace(output_path)
    checkpoint_path.unlink(missing_ok=True)
    return {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "rows": len(words) + 1,
        "dimension": dimension,
        "unknown_only_words": unknown_only,
        "truncated_words": truncated,
        "seconds": elapsed,
        "quantization": "NF4, double quantization, FP16 compute",
        "cached": False,
    }


def pca_lowrank(
    matrix: np.ndarray,
    output_path: Path,
    components: int = 768,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, dict[str, float]]:
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        return np.load(output_path, mmap_mode="r"), json.loads(metadata_path.read_text())

    torch.manual_seed(SPLIT_SEED)
    rows, columns = matrix.shape
    values = torch.empty((rows, columns), dtype=torch.float32, device="cuda")
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        values[start:stop] = torch.from_numpy(
            np.asarray(matrix[start:stop], dtype=np.float32)
        ).to("cuda")
    started = time.perf_counter()
    mean = values.mean(dim=0, keepdim=True)
    centered = values - mean
    _, singular_values, directions = torch.pca_lowrank(
        centered,
        q=components,
        center=False,
        niter=4,
    )
    scores = centered @ directions
    elapsed = time.perf_counter() - started
    total_variance = torch.sum(centered.square()).item() / max(rows - 1, 1)
    retained_variance = torch.sum(singular_values.square()).item() / max(rows - 1, 1)
    projected = scores.cpu().numpy().astype(np.float32)
    np.save(output_path, projected)
    details = {
        "method": "centered torch.pca_lowrank",
        "components": components,
        "niter": 4,
        "seconds": elapsed,
        "cumulative_explained_variance": retained_variance / total_variance,
    }
    metadata_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    del values, centered, mean, scores, directions, singular_values, projected
    torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r"), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mbert-matrix", type=Path, required=True)
    parser.add_argument("--reference-vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    reference_words = pd.read_csv(
        args.reference_vocabulary,
        keep_default_na=False,
        dtype=str,
    )["word"].tolist()
    if words != reference_words:
        raise ValueError("The regenerated grouped vocabulary order does not match the reference.")
    all_tokens = texts_to_padded(data["clean_text"], word_index, args.max_length)
    all_labels = data["label"].to_numpy(dtype=np.int64)
    mbert = np.load(args.mbert_matrix, mmap_mode="r")
    if mbert.shape[0] != len(words) + 1:
        raise ValueError("The mBERT matrix does not match the grouped training vocabulary.")

    qwen_path = args.output_dir / "embedding_qwen2.5-7b-instruct_nf4.npy"
    extraction = extract_qwen(
        args.model_path,
        words,
        qwen_path,
        args.embedding_batch_size,
    )
    qwen = np.load(qwen_path, mmap_mode="r")
    qwen_pca, qwen_pca_details = pca_lowrank(
        qwen,
        args.output_dir / "embedding_qwen_pca768.npy",
    )
    concat_path = args.output_dir / "embedding_mbert_qwen_standardized.npy"
    concat = standardized_concat(mbert, qwen, concat_path)
    pca, fusion_pca_details = pca_lowrank(
        concat,
        args.output_dir / "embedding_mbert_qwen_pca768.npy",
    )

    configurations = {
        "mBERT": mbert,
        QWEN_LABEL: qwen_pca,
        "mBERT+Qwen Concat+PCA-768": pca,
    }
    checkpoint_path = args.output_dir / "qwen_fusion_runs_checkpoint.csv"
    run_rows = pd.read_csv(checkpoint_path).to_dict("records") if checkpoint_path.exists() else []
    completed = {(str(row["configuration"]), int(row["seed"])) for row in run_rows}
    prediction_paths: list[Path] = []
    for name, matrix in configurations.items():
        slug = name.lower().replace("+", "plus").replace(" ", "_").replace("(", "").replace(")", "")
        for seed in SEEDS:
            prediction_path = args.output_dir / f"prediction_{slug}_seed{seed}.csv"
            if (name, seed) in completed and prediction_path.exists():
                prediction_paths.append(prediction_path)
                continue
            logits, costs = train_one(
                matrix,
                all_tokens[train],
                all_labels[train],
                all_tokens[validation],
                all_labels[validation],
                all_tokens[test],
                seed,
                torch.device("cuda"),
                args.batch_size,
                args.max_epochs,
            )
            values = evaluate(all_labels[test], logits)
            run_rows.append({"configuration": name, "seed": seed, **values, **costs})
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            pd.DataFrame(
                {
                    "configuration": name,
                    "seed": seed,
                    "row_index": test,
                    "group_id": data.iloc[test]["group_id"].to_numpy(),
                    "language_column": data.iloc[test]["language_column"].to_numpy(),
                    "label": all_labels[test],
                    "probability": probabilities,
                    "prediction": (probabilities >= 0.5).astype(np.int8),
                }
            ).to_csv(prediction_path, index=False)
            prediction_paths.append(prediction_path)
            pd.DataFrame(run_rows).to_csv(checkpoint_path, index=False)
            print(name, seed, values, costs, flush=True)

    runs = pd.DataFrame(run_rows)
    metric_columns = [
        "accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc",
        "cohen_kappa", "balanced_accuracy", "roc_auc", "brier", "ece_15",
    ]
    summary = runs.groupby("configuration")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    summary.to_csv(args.output_dir / "qwen_fusion_summary.csv", index=False)
    runs.to_csv(args.output_dir / "qwen_fusion_runs.csv", index=False)
    predictions.to_csv(args.output_dir / "qwen_fusion_predictions.csv", index=False)

    metadata = {
        "experiment_type": "grouped historical-encoder extension",
        "encoder_models": ["bert-base-multilingual-uncased", "Qwen/Qwen2.5-7B-Instruct"],
        "qwen_loading": extraction,
        "split_seed": SPLIT_SEED,
        "training_seeds": SEEDS,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "vocabulary_size": len(words),
        "max_sequence_length": args.max_length,
        "pca": {
            "qwen_to_768": qwen_pca_details,
            "mbert_qwen_concat_to_768": fusion_pca_details,
        },
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "best_configuration": str(summary.iloc[0]["configuration"]),
        "best_mean_mcc": float(summary.iloc[0]["mcc_mean"]),
        "best_seed42_mcc": float(
            matthews_corrcoef(
                predictions[(predictions["configuration"] == str(summary.iloc[0]["configuration"])) & (predictions["seed"] == 42)]["label"],
                predictions[(predictions["configuration"] == str(summary.iloc[0]["configuration"])) & (predictions["seed"] == 42)]["prediction"],
            )
        ),
    }
    (args.output_dir / "qwen_fusion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
