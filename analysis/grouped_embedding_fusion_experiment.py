"""Reconstruct a fully traceable grouped embedding-fusion experiment.

The experiment uses two openly accessible encoders listed in the thesis:
DistilBERT multilingual and multilingual BERT. It is a new grouped experiment,
not a reconstruction of unavailable Llama/Qwen matrices or legacy scores.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from revision_audit import SEEDS, load_long_dataset, split_groups


ENCODERS = (
    "distilbert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
)
SPLIT_SEED = 42
MAX_SUBTOKENS = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_vocabulary(texts: pd.Series) -> tuple[dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    cursor = 0
    for text in texts:
        for word in text.split():
            counts[word] += 1
            if word not in first_seen:
                first_seen[word] = cursor
                cursor += 1
    words = sorted(counts, key=lambda word: (-counts[word], first_seen[word]))
    word_index = {word: index + 1 for index, word in enumerate(words)}
    return word_index, words


def texts_to_padded(texts: pd.Series, word_index: dict[str, int], max_length: int) -> np.ndarray:
    result = np.zeros((len(texts), max_length), dtype=np.int32)
    for row, text in enumerate(texts):
        sequence = [word_index[word] for word in text.split() if word in word_index]
        sequence = sequence[-max_length:]
        if sequence:
            result[row, -len(sequence) :] = sequence
    return result


@torch.inference_mode()
def encode_vocabulary(
    model_name: str,
    words: list[str],
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int | str]:
    if output_path.exists():
        matrix = np.load(output_path, mmap_mode="r")
        return {
            "model": model_name,
            "rows": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "cached": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    dimension = int(model.config.hidden_size)
    matrix = np.zeros((len(words) + 1, dimension), dtype=np.float32)
    unknown_only = 0
    truncated = 0
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
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        matrix[offset + 1 : offset + 1 + len(batch_words)] = pooled.float().cpu().numpy()

    elapsed = time.perf_counter() - start
    np.save(output_path, matrix)
    del model
    torch.cuda.empty_cache()
    return {
        "model": model_name,
        "rows": int(matrix.shape[0]),
        "dimension": dimension,
        "unknown_only_words": unknown_only,
        "truncated_words": truncated,
        "seconds": elapsed,
        "cached": False,
    }


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True).clip(min=1e-6)
    return ((matrix - mean) / std).astype(np.float32)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    value = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            value += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(value)


def evaluate(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
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


class FrozenEmbeddingCNN(nn.Module):
    def __init__(self, embedding: np.ndarray, filters: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32))
        self.embedding = nn.Embedding.from_pretrained(tensor, freeze=True, padding_idx=0)
        self.conv = nn.Conv1d(tensor.shape[1], filters, kernel_size)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(filters, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.embedding(tokens).permute(0, 2, 1)
        values = torch.relu(self.conv(values)).amax(dim=-1)
        values = self.dropout1(values)
        values = torch.relu(self.fc1(values))
        return self.fc2(self.dropout2(values)).squeeze(-1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for (tokens,) in loader:
            outputs.append(model(tokens.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(outputs)


def train_one(
    matrix: np.ndarray,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    validation_tokens: np.ndarray,
    validation_labels: np.ndarray,
    test_tokens: np.ndarray,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    set_seed(seed)
    model = FrozenEmbeddingCNN(matrix).to(device)
    effective_batch_size = min(batch_size, 1024) if matrix.shape[1] > 768 else batch_size
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_labels)),
        batch_size=effective_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_tokens)),
        batch_size=effective_batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_tokens)),
        batch_size=effective_batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    positive_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    best_state = None
    patience = 4
    stale = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for tokens, labels in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()

        validation_logits = predict(model, validation_loader, device)
        validation_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(validation_logits),
            torch.from_numpy(validation_labels.astype(np.float32)),
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32),
        ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test_logits = predict(model, test_loader, device)
    inference_seconds = time.perf_counter() - start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1_000_000 if device.type == "cuda" else float("nan")
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    del model
    torch.cuda.empty_cache()
    return test_logits, {
        "epochs": epoch,
        "batch_size": effective_batch_size,
        "best_validation_loss": best_loss,
        "train_seconds": train_seconds,
        "inference_ms_per_message": inference_seconds * 1000 / len(test_tokens),
        "peak_gpu_memory_mb": peak_memory_mb,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "stored_embedding_mb": matrix.nbytes / 1_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    train_lengths = data.iloc[train]["clean_text"].str.split().str.len()
    max_length = min(int(train_lengths.max()), args.max_length)
    all_tokens = texts_to_padded(data["clean_text"], word_index, max_length)
    all_labels = data["label"].to_numpy(dtype=np.int64)
    np.savez_compressed(
        args.output_dir / "grouped_split_seed42.npz",
        train=train,
        validation=validation,
        test=test,
        group_id=data["group_id"].to_numpy(),
        label=all_labels,
    )
    pd.DataFrame({"word": words, "index": np.arange(1, len(words) + 1)}).to_csv(
        args.output_dir / "training_vocabulary.csv", index=False
    )

    extraction = []
    matrices = {}
    for model_name in ENCODERS:
        slug = model_name.replace("/", "--")
        path = args.output_dir / f"embedding_{slug}.npy"
        extraction.append(
            encode_vocabulary(model_name, words, path, device, args.embedding_batch_size)
        )
        matrices[model_name] = np.load(path)

    distil = matrices[ENCODERS[0]]
    mbert = matrices[ENCODERS[1]]
    raw_concat = np.concatenate([distil, mbert], axis=1).astype(np.float32)
    standardized_concat = np.concatenate([standardize(distil), standardize(mbert)], axis=1)
    pca = PCA(n_components=256, svd_solver="randomized", iterated_power=4, random_state=SPLIT_SEED)
    start = time.perf_counter()
    pca_256 = pca.fit_transform(standardized_concat).astype(np.float32)
    pca_seconds = time.perf_counter() - start
    random_projection = GaussianRandomProjection(n_components=256, random_state=SPLIT_SEED)
    start = time.perf_counter()
    rp_256 = random_projection.fit_transform(standardized_concat).astype(np.float32)
    rp_seconds = time.perf_counter() - start

    np.save(args.output_dir / "embedding_concat_raw.npy", raw_concat)
    np.save(args.output_dir / "embedding_concat_standardized.npy", standardized_concat)
    np.save(args.output_dir / "embedding_concat_pca128.npy", pca_256[:, :128])
    np.save(args.output_dir / "embedding_concat_pca256.npy", pca_256)
    np.save(args.output_dir / "embedding_concat_random_projection256.npy", rp_256)
    pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(args.output_dir / "pca_explained_variance.csv", index=False)

    configurations = {
        "DistilBERT": distil,
        "mBERT": mbert,
        "Raw concat": raw_concat,
        "Standardized concat": standardized_concat,
        "Concat+PCA-128": pca_256[:, :128],
        "Concat+PCA-256": pca_256,
        "Concat+RP-256": rp_256,
    }
    checkpoint_path = args.output_dir / "fusion_runs_checkpoint.csv"
    if checkpoint_path.exists():
        run_rows = pd.read_csv(checkpoint_path).to_dict("records")
    else:
        run_rows = []
    completed = {(str(row["configuration"]), int(row["seed"])) for row in run_rows}
    prediction_paths: list[Path] = []
    for name, matrix in configurations.items():
        for seed in SEEDS:
            prediction_path = args.output_dir / (
                "prediction_" + name.lower().replace("+", "plus").replace(" ", "_") + f"_seed{seed}.csv"
            )
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
                device,
                args.batch_size,
                args.max_epochs,
            )
            values = evaluate(all_labels[test], logits)
            run_rows.append({"configuration": name, "seed": seed, **values, **costs})
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            prediction_frame = pd.DataFrame(
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
            )
            prediction_frame.to_csv(prediction_path, index=False)
            prediction_paths.append(prediction_path)
            pd.DataFrame(run_rows).to_csv(checkpoint_path, index=False)
            print(name, seed, values, costs, flush=True)

    runs = pd.DataFrame(run_rows)
    metric_columns = [
        "accuracy",
        "spam_f1",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "cohen_kappa",
        "balanced_accuracy",
        "roc_auc",
        "brier",
        "ece_15",
    ]
    summary = runs.groupby("configuration")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)

    best_name = str(summary.iloc[0]["configuration"])
    reference = predictions[
        (predictions["configuration"] == best_name) & (predictions["seed"] == 42)
    ].sort_values("row_index")
    tests = []
    for name in configurations:
        if name == best_name:
            continue
        comparison = predictions[
            (predictions["configuration"] == name) & (predictions["seed"] == 42)
        ].sort_values("row_index")
        labels = reference["label"].to_numpy()
        reference_correct = reference["prediction"].to_numpy() == labels
        comparison_correct = comparison["prediction"].to_numpy() == labels
        reference_only = int(np.sum(reference_correct & ~comparison_correct))
        comparison_only = int(np.sum(~reference_correct & comparison_correct))
        discordant = reference_only + comparison_only
        tests.append(
            {
                "seed": 42,
                "reference": best_name,
                "comparison": name,
                "reference_only_correct": reference_only,
                "comparison_only_correct": comparison_only,
                "exact_mcnemar_p": 1.0
                if discordant == 0
                else binomtest(reference_only, discordant, 0.5).pvalue,
            }
        )

    runs.to_csv(args.output_dir / "fusion_runs.csv", index=False)
    summary.to_csv(args.output_dir / "fusion_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "fusion_predictions.csv", index=False)
    pd.DataFrame(tests).to_csv(args.output_dir / "fusion_mcnemar_seed42.csv", index=False)
    metadata = {
        "experiment_type": "new leakage-controlled reconstruction",
        "encoder_models": ENCODERS,
        "split_seed": SPLIT_SEED,
        "training_seeds": SEEDS,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "vocabulary_size": len(words),
        "max_sequence_length": max_length,
        "train_rows_truncated": int((train_lengths > max_length).sum()),
        "train_rows_truncated_fraction": float((train_lengths > max_length).mean()),
        "pca_fit_seconds": pca_seconds,
        "random_projection_fit_seconds": rp_seconds,
        "pca_128_cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[127]),
        "pca_256_cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[255]),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "extraction": extraction,
    }
    (args.output_dir / "fusion_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
