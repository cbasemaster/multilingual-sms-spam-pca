"""Complete matched grouped fusion and vocabulary-sensitivity controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grouped_embedding_fusion_experiment import (
    FrozenEmbeddingCNN, encode_vocabulary, evaluate, fit_vocabulary, predict,
    set_seed, texts_to_padded,
)
from grouped_qwen_pca_sweep_experiment import infer_state, train_extended
from revision_audit import load_long_dataset
from run_cv5_llama_fusion import load_llama_matrix
from run_cv5_multiseed_all_static import SEEDS, slug


FAMILIES = {
    "mBERT+Qwen": ("mBERT", "Qwen2.5 raw"),
    "Llama-2+Qwen": ("Llama-2 raw", "Qwen2.5 raw"),
}


def aligned_sources(sources: list[np.ndarray], path: Path) -> np.ndarray:
    width = min(source.shape[1] for source in sources)
    shape = (sources[0].shape[0], len(sources) * width)
    if any(source.shape[0] != shape[0] for source in sources):
        raise ValueError("Source vocabulary entries must align")
    result = np.empty(shape, dtype=np.float16)
    result[0] = 0
    for start in range(1, shape[0], 1024):
        stop = min(start + 1024, shape[0])
        for index, source in enumerate(sources):
            vectors = np.asarray(source[start:stop], dtype=np.float32)[:, :width]
            centered = vectors - vectors.mean(axis=1, keepdims=True)
            normalized = centered / np.sqrt(centered.var(axis=1, keepdims=True) + 1e-5)
            result[start:stop, index * width:(index + 1) * width] = normalized
    return result


def fuse_table(aligned: np.ndarray, sources: int, path: Path,
               weights: np.ndarray | None = None,
               biases: np.ndarray | None = None) -> np.ndarray:
    width = aligned.shape[1] // sources
    result = np.empty((aligned.shape[0], width), dtype=np.float32)
    use_cuda = weights is not None and torch.cuda.is_available()
    if use_cuda:
        gate_weight = torch.from_numpy(weights).to("cuda")
        gate_bias = torch.from_numpy(biases).to("cuda")
    for start in range(0, aligned.shape[0], 1024):
        values = np.asarray(aligned[start:start + 1024], dtype=np.float32)
        values = values.reshape(-1, sources, width)
        if weights is None:
            result[start:start + len(values)] = values.mean(axis=1)
        elif use_cuda:
            with torch.inference_mode():
                vectors = torch.from_numpy(values).to("cuda")
                scores = (vectors * gate_weight).sum(dim=-1) + gate_bias
                fused = (vectors * scores.softmax(dim=-1).unsqueeze(-1)).sum(dim=-2)
                result[start:start + len(values)] = fused.cpu().numpy()
        else:
            scores = np.einsum("nmd,md->nm", values, weights) + biases
            scores -= scores.max(axis=1, keepdims=True)
            alpha = np.exp(scores)
            alpha /= alpha.sum(axis=1, keepdims=True)
            result[start:start + len(values)] = np.einsum("nm,nmd->nd", alpha, values)
    result[0] = 0
    return result


class GatedEmbeddingCNN(FrozenEmbeddingCNN):
    def __init__(self, aligned: np.ndarray, sources: int, device="cpu") -> None:
        width = aligned.shape[1] // sources
        super().__init__(np.zeros((1, width), dtype=np.float32))
        self.embedding = nn.Embedding(aligned.shape[0], aligned.shape[1],
                                      padding_idx=0, device=device)
        self.embedding.weight.requires_grad_(False)
        with torch.no_grad():
            for start in range(0, aligned.shape[0], 1024):
                chunk = np.asarray(aligned[start:start + 1024], dtype=np.float32)
                self.embedding.weight[start:start + len(chunk)].copy_(torch.from_numpy(chunk))
        self.sources = sources
        self.width = width
        self.gate_weight = nn.Parameter(torch.zeros(sources, width))
        self.gate_bias = nn.Parameter(torch.zeros(sources))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        unique, inverse = torch.unique(tokens, return_inverse=True)
        values = self.embedding(unique).reshape(-1, self.sources, self.width)
        scores = (values * self.gate_weight).sum(dim=-1) + self.gate_bias
        alpha = scores.softmax(dim=-1)
        fused = (values * alpha.unsqueeze(-1)).sum(dim=-2)
        values = fused[inverse].reshape(*tokens.shape, self.width).permute(0, 2, 1)
        values = torch.relu(self.conv(values)).amax(dim=-1)
        values = self.dropout1(values)
        values = torch.relu(self.fc1(values))
        return self.fc2(self.dropout2(values)).squeeze(-1)


def train_gate(aligned, sources, train_tokens, train_labels, validation_tokens,
               validation_labels, seed, max_epochs):
    set_seed(seed)
    device = torch.device("cuda")
    model = GatedEmbeddingCNN(aligned, sources, device=device).to(device)
    physical = 64 if aligned.shape[1] > 4096 else 256
    accumulation = 1024 // physical
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_labels)),
        batch_size=physical, shuffle=True, generator=torch.Generator().manual_seed(seed),
        pin_memory=True, num_workers=0,
    )
    validation = DataLoader(TensorDataset(torch.from_numpy(validation_tokens)),
                            batch_size=physical * 2, pin_memory=True, num_workers=0)
    weight = float((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, stale, best_state = float("inf"), 0, {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (tokens, labels) in enumerate(loader, 1):
            loss = criterion(model(tokens.to(device)), labels.to(device).float())
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        logits = predict(model, validation, device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(validation_labels.astype(np.float32)),
            pos_weight=torch.tensor([weight]),
        ).item()
        if loss < best_loss - 1e-5:
            best_loss, stale = loss, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()
                          if key != "embedding.weight"}
        else:
            stale += 1
            if stale >= 4:
                break
        if epoch % 5 == 0:
            print(f"  MoE seed {seed}: epoch {epoch}, validation loss {loss:.4f}", flush=True)
    model.load_state_dict(best_state, strict=False)
    logits = predict(model, validation, device)
    costs = {
        "epochs": epoch, "physical_batch_size": physical, "effective_batch_size": 1024,
        "best_validation_loss": best_loss, "train_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_source_table_mb": aligned.nbytes / 1e6,
    }
    del model
    torch.cuda.empty_cache()
    return best_state, logits, evaluate(validation_labels, logits), costs


def global_matrices(data: pd.DataFrame, fold_dir: Path, out: Path):
    index, words = fit_vocabulary(data["clean_text"])
    word_digest = hashlib.sha256("\n".join(words).encode()).hexdigest()
    metadata = out / "full_vocabulary.json"
    locations = {}
    for fold in range(1, 6):
        fold_words = pd.read_csv(fold_dir / f"fold_{fold}" / "training_vocabulary.csv",
                                 keep_default_na=False, dtype=str)["word"].tolist()
        for position, word in enumerate(fold_words, 1):
            locations.setdefault(word, (fold, position))
    missing = sorted(set(words) - locations.keys())
    if missing:
        print(f"Extracting {len(missing)} full-vocabulary-only words", flush=True)
        (out / "extra_words.json").write_text(json.dumps(missing, ensure_ascii=False), encoding="utf-8")
        encode_vocabulary("bert-base-multilingual-uncased", missing, out / "extra_mbert.npy",
                          torch.device("cuda"), 256)
        encode_vocabulary("distilbert-base-multilingual-cased", missing, out / "extra_distil.npy",
                          torch.device("cuda"), 256)
        locations.update({word: (0, position) for position, word in enumerate(missing, 1)})
    matrices = [np.empty((len(words) + 1, 768), dtype=np.float32) for _ in range(2)]
    for matrix in matrices:
        matrix[0] = 0
    for fold in range(0, 6):
        selected = [(i, locations[word][1]) for i, word in enumerate(words, 1)
                    if locations[word][0] == fold]
        if not selected:
            continue
        target, source = np.array(selected).T
        for name, output in zip(("distil", "mbert"), matrices):
            source_path = (out / f"extra_{name}.npy" if fold == 0
                           else fold_dir / f"fold_{fold}" / f"{name}.npy")
            original = np.load(source_path, mmap_mode="r")
            for start in range(0, len(target), 1024):
                output[target[start:start + 1024]] = original[source[start:start + 1024]]
    metadata.write_text(json.dumps({"entries": len(words), "sha256": word_digest,
        "scope": "all partitions; diagnostic only, not the primary evaluation"}, indent=2))
    return index, words, matrices


def full_vocabulary_pca(sources, metadata_path):
    from grouped_qwen_pca_sweep_experiment import moments_without_padding
    n = sources[0].shape[0] - 1
    cached = metadata_path.with_suffix(".npy")
    if cached.exists() and metadata_path.exists():
        result = np.load(cached, mmap_mode="r")
        if result.shape != (n + 1, 1024):
            raise ValueError("Stale full-vocabulary PCA cache")
        return result
    width = sum(source.shape[1] for source in sources)
    values = torch.empty((n, width), dtype=torch.float32, device="cuda")
    cursor = 0
    for source in sources:
        mean, std = moments_without_padding(source)
        for start in range(1, n + 1, 1024):
            stop = min(start + 1024, n + 1)
            vectors = (source[start:stop] - mean) / std
            # Match the existing fold pipeline's intermediate FP16 quantization.
            vectors = vectors.astype(np.float16).astype(np.float32)
            values[start - 1:stop - 1, cursor:cursor + source.shape[1]] = torch.from_numpy(vectors).to("cuda")
        cursor += source.shape[1]
    values.sub_(values.mean(dim=0, keepdim=True))
    total_variance = sum(float((values[start:start + 1024] ** 2).sum().item())
                         for start in range(0, n, 1024)) / (n - 1)
    torch.manual_seed(42)
    started = time.perf_counter()
    u, singular, _ = torch.pca_lowrank(values, q=1024, center=False, niter=3)
    result = np.zeros((n + 1, 1024), dtype=np.float16)
    for start in range(0, n, 1024):
        scores = u[start:start + 1024, :1024] * singular[:1024]
        result[start + 1:start + 1 + len(scores)] = scores.cpu().numpy().astype(np.float16)
    metadata_path.write_text(json.dumps({
        "scope": "DistilBERT/mBERT full-vocabulary diagnostic only", "max_components": 1024,
        "deployed_components": 1024, "niter": 3, "total_variance": total_variance,
        "eigenvalues": (singular.square().cpu().numpy() / (n - 1)).tolist(),
        "seconds": time.perf_counter() - started,
    }))
    np.save(cached, result)
    del values, u, singular, scores
    torch.cuda.empty_cache()
    return result


def save_run(rows, path, result, pred_path, probability, test):
    np.savez_compressed(pred_path, probability=probability.astype(np.float32), test_index=test)
    rows.append(result)
    frame = pd.DataFrame(rows).sort_values(["fold", "training_seed", "configuration"])
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)
    print(f"fold {result['fold']} seed {result['training_seed']}: "
          f"{result['configuration']}: MCC {result['mcc']:.4f}; "
          f"training {result['train_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("source/sms_spam_multilingual.parquet"))
    parser.add_argument("--fold-dir", type=Path, default=Path("output/source_group_cv5"))
    parser.add_argument("--static-dir", type=Path, default=Path("output/multiseed_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/revision_gap_experiments"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--mode", choices=("fusion", "vocabulary", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    checkpoint = out / "runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    done = {(int(r["fold"]), int(r["training_seed"]), r["configuration"]) for r in rows}
    static = pd.read_csv(args.static_dir / "static_multiseed_runs.csv").set_index(
        ["fold", "training_seed", "configuration"])
    processed_path = out / "processed_dataset.parquet"
    digest_path = out / "processed_dataset_sha256.txt"
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if processed_path.exists() and digest_path.exists() and digest_path.read_text() == digest:
        data = pd.read_parquet(processed_path)
    else:
        data = load_long_dataset(args.dataset)[["group_id", "language_column", "label", "clean_text"]]
        data.to_parquet(processed_path, index=False)
        digest_path.write_text(digest)
    labels = data["label"].to_numpy(dtype=np.int64)
    full_tokens = full_pca = None
    if args.mode in ("vocabulary", "all"):
        full_index, _, full_sources = global_matrices(data, args.fold_dir, out)
        full_tokens = texts_to_padded(data["clean_text"], full_index, 128)
        full_pca = full_vocabulary_pca(full_sources, out / "full_distil_mbert_pca1024.json")
        del full_sources
    for fold in args.folds:
        fold_path = args.fold_dir / f"fold_{fold}"
        cache = out / f"fold_{fold}"
        cache.mkdir(exist_ok=True)
        with np.load(args.fold_dir / f"fold_{fold}.npz") as split:
            train, validation, test = (split[k] for k in ("train", "validation", "test"))
        partition_groups = [set(data.iloc[p]["group_id"]) for p in (train, validation, test)]
        if any(partition_groups[i] & partition_groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Source-group overlap")
        word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
        recorded = pd.read_csv(fold_path / "training_vocabulary.csv", keep_default_na=False,
                               dtype=str)["word"].tolist()
        if words != recorded:
            raise ValueError("Training vocabulary changed")
        tokens = texts_to_padded(data["clean_text"], word_index, 128)
        if args.mode in ("fusion", "all"):
            qwen = np.load(fold_path / "qwen.npy", mmap_mode="r")
            for family, branches in FAMILIES.items():
                left = (np.load(fold_path / "mbert.npy", mmap_mode="r")
                        if family.startswith("mBERT") else load_llama_matrix(args.fold_dir, words))
                aligned = aligned_sources([left, qwen], cache / f"{slug(family)}_aligned.npy")
                average = fuse_table(aligned, 2, cache / f"{slug(family)}_average.npy")
                for method in ("Averaging", "MoE", "Late fusion"):
                    name = f"{family} {method}"
                    for seed in args.seeds:
                        if (fold, seed, name) in done:
                            continue
                        pred_path = out / "predictions" / f"fold{fold}_seed{seed}_{slug(name)}.npz"
                        if method == "Late fusion":
                            values, branch_costs = [], []
                            for branch in branches:
                                path = args.static_dir / "predictions" / f"fold{fold}_seed{seed}_{slug(branch)}.npz"
                                with np.load(path) as saved:
                                    values.append(saved["probability"].astype(np.float32))
                                branch_costs.append(static.loc[(fold, seed, branch)])
                            probability = np.mean(values, axis=0)
                            logits = np.log(np.clip(probability, 1e-7, 1 - 1e-7)
                                            / np.clip(1 - probability, 1e-7, 1 - 1e-7))
                            costs = {k: sum(float(r[k]) for r in branch_costs)
                                     for k in ("train_seconds", "trainable_parameters", "stored_embedding_mb")}
                            inference = sum(float(r["inference_ms_per_message"]) for r in branch_costs)
                            costs["cost_basis"] = "sum of measured independent branch costs; aggregation excluded"
                            validation_metrics = {}
                            width = left.shape[1] + qwen.shape[1]
                        else:
                            if method == "Averaging":
                                matrix = average
                                state, _, validation_metrics, costs = train_extended(
                                    matrix, tokens[train], labels[train], tokens[validation],
                                    labels[validation], seed, 256, 1024, args.max_epochs,
                                )
                            else:
                                gated, gate_logits, validation_metrics, costs = train_gate(
                                    aligned, 2, tokens[train], labels[train], tokens[validation],
                                    labels[validation], seed, args.max_epochs,
                                )
                                matrix = fuse_table(aligned, 2, cache / "deployed_moe.npy",
                                    gated["gate_weight"].numpy(), gated["gate_bias"].numpy())
                                state = {k: v for k, v in gated.items() if not k.startswith("gate_")}
                                torch.save(gated, cache / f"{slug(family)}_seed{seed}_moe_state.pt")
                                deployed_logits, _ = infer_state(matrix, state, tokens[validation], 256)
                                max_error = float(np.max(np.abs(gate_logits - deployed_logits)))
                                if max_error > 1e-3:
                                    raise ValueError(f"MoE deployment mismatch: {max_error}")
                                costs["deployment_max_logit_error"] = max_error
                            logits, inference = infer_state(matrix, state, tokens[test], 512)
                            probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                            width = matrix.shape[1]
                            costs["stored_embedding_mb"] = matrix.nbytes / 1e6
                            del state
                        result = {"fold": fold, "training_seed": seed, "configuration": name,
                            "source_dimension": left.shape[1] + qwen.shape[1], "final_dimension": width,
                            **{f"validation_{k}": v for k, v in validation_metrics.items()},
                            **evaluate(labels[test], logits), **costs,
                            "inference_ms_per_message": inference}
                        save_run(rows, checkpoint, result, pred_path, probability, test)
                        done.add((fold, seed, name))
                        gc.collect()
                del average, aligned, left
        if args.mode in ("vocabulary", "all"):
            name = "DistilBERT+mBERT PCA-1024 full vocabulary diagnostic"
            for seed in args.seeds:
                if (fold, seed, name) in done:
                    continue
                state, _, validation_metrics, costs = train_extended(
                    full_pca, full_tokens[train], labels[train], full_tokens[validation],
                    labels[validation], seed, 256, 1024, args.max_epochs,
                )
                logits, inference = infer_state(full_pca, state, full_tokens[test], 512)
                probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                result = {"fold": fold, "training_seed": seed, "configuration": name,
                    "source_dimension": 1536, "final_dimension": 1024,
                    **{f"validation_{k}": v for k, v in validation_metrics.items()},
                    **evaluate(labels[test], logits), **costs,
                    "inference_ms_per_message": inference}
                pred_path = out / "predictions" / f"fold{fold}_seed{seed}_{slug(name)}.npz"
                save_run(rows, checkpoint, result, pred_path, probability, test)
                done.add((fold, seed, name))
                del state
    print(f"Complete: {len(rows)} saved gap-control evaluations", flush=True)


if __name__ == "__main__":
    main()
