"""Fine-tune mBERT on the fixed leakage-controlled source-group split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grouped_embedding_fusion_experiment import expected_calibration_error, set_seed
from revision_audit import SEEDS, load_long_dataset, split_groups


MODEL_NAME = "bert-base-multilingual-uncased"
SPLIT_SEED = 42


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    predictions = (probabilities >= 0.5).astype(np.int8)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "spam_f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "cohen_kappa": cohen_kappa_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "ece_15": expected_calibration_error(labels, probabilities),
    }


def tokenize(tokenizer, texts: pd.Series, max_length: int) -> tuple[torch.Tensor, ...]:
    encoded = tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    keys = [key for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoded]
    return tuple(torch.from_numpy(encoded[key].astype(np.int64)) for key in keys), keys


def forward_logits(model, batch, keys, device):
    inputs = {key: value.to(device, non_blocking=True) for key, value in zip(keys, batch)}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return model(**inputs).logits.squeeze(-1)


@torch.inference_mode()
def predict(model, loader, keys, device) -> np.ndarray:
    model.eval()
    values = []
    for batch in loader:
        values.append(forward_logits(model, batch, keys, device).float().cpu().numpy())
    return np.concatenate(values)


def train_seed(
    seed: int,
    token_tensors: tuple[torch.Tensor, ...],
    keys: list[str],
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    max_train_batches: int | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        local_files_only=True,
    ).to(device)
    train_dataset = TensorDataset(
        *(tensor[train] for tensor in token_tensors),
        torch.from_numpy(labels[train].astype(np.float32)),
    )
    validation_loader = DataLoader(
        TensorDataset(*(tensor[validation] for tensor in token_tensors)),
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(*(tensor[test] for tensor in token_tensors)),
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    positive_weight = (labels[train] == 0).sum() / (labels[train] == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_state = None
    stale = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            inputs = batch[:-1]
            batch_labels = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(
                    **{key: value.to(device, non_blocking=True) for key, value in zip(keys, inputs)}
                ).logits.squeeze(-1)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        validation_logits = predict(model, validation_loader, keys, device)
        validation_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(validation_logits),
            torch.from_numpy(labels[validation].astype(np.float32)),
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32),
        ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 1:
                break

    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_logits = predict(model, test_loader, keys, device)
    inference_seconds = time.perf_counter() - start
    result = {
        "epochs": epoch,
        "batch_size": batch_size,
        "best_validation_loss": best_loss,
        "train_seconds": train_seconds,
        "inference_ms_per_message": inference_seconds * 1000 / len(test),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1_000_000,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return test_logits, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    start = time.perf_counter()
    token_tensors, keys = tokenize(tokenizer, data["clean_text"], args.max_length)
    tokenization_seconds = time.perf_counter() - start
    labels = data["label"].to_numpy(dtype=np.int64)

    checkpoint = args.output_dir / "finetuned_mbert_runs_checkpoint.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {int(row["seed"]) for row in rows}
    for seed in args.seeds:
        prediction_path = args.output_dir / f"finetuned_mbert_prediction_seed{seed}.csv"
        if seed in completed and prediction_path.exists():
            continue
        logits, costs = train_seed(
            seed,
            token_tensors,
            keys,
            labels,
            train,
            validation,
            test,
            device,
            args.batch_size,
            args.max_epochs,
            args.max_train_batches,
        )
        values = metrics(labels[test], logits)
        rows.append({"seed": seed, **values, **costs})
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        pd.DataFrame(
            {
                "seed": seed,
                "row_index": test,
                "group_id": data.iloc[test]["group_id"].to_numpy(),
                "language_column": data.iloc[test]["language_column"].to_numpy(),
                "label": labels[test],
                "probability": probabilities,
                "prediction": (probabilities >= 0.5).astype(np.int8),
            }
        ).to_csv(prediction_path, index=False)
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
        print(seed, values, costs, flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(args.output_dir / "finetuned_mbert_runs.csv", index=False)
    metadata = {
        "model": MODEL_NAME,
        "split_seed": SPLIT_SEED,
        "seeds": args.seeds,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": 2e-5,
        "tokenization_seconds": tokenization_seconds,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (args.output_dir / "finetuned_mbert_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(runs.round(4).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
