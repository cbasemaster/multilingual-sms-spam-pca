"""Validation-selected PCA sweep for grouped mBERT/Qwen static fusion.

The dimension sweep is selected by validation MCC only. The selected PCA
width and matched controls are then evaluated with five training seeds on the
fixed source-group split. Padding row zero is explicitly reset after every
fitted transformation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grouped_embedding_fusion_experiment import (
    FrozenEmbeddingCNN,
    evaluate,
    fit_vocabulary,
    predict,
    set_seed,
    texts_to_padded,
)
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
DEFAULT_DIMS = (768, 1024, 1536, 2048)


def moments_without_padding(matrix: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    total_sq = np.zeros(matrix.shape[1], dtype=np.float64)
    count = matrix.shape[0] - 1
    for start in range(1, matrix.shape[0], chunk_size):
        values = np.asarray(matrix[start : start + chunk_size], dtype=np.float32)
        total += values.sum(axis=0, dtype=np.float64)
        total_sq += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def make_standardized_concat(
    left: np.ndarray,
    right: np.ndarray,
    output_path: Path,
    chunk_size: int = 1024,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path, mmap_mode="r")
    left_mean, left_std = moments_without_padding(left)
    right_mean, right_std = moments_without_padding(right)
    partial = output_path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(left.shape[0], left.shape[1] + right.shape[1]),
    )
    output[0] = 0
    for start in range(1, left.shape[0], chunk_size):
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


def fit_pca_max(
    matrix: np.ndarray,
    output_path: Path,
    max_components: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, dict]:
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        return np.load(output_path, mmap_mode="r"), json.loads(metadata_path.read_text())

    torch.manual_seed(SPLIT_SEED)
    rows, columns = matrix.shape
    values = torch.empty((rows - 1, columns), dtype=torch.float32, device="cuda")
    for start in range(1, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        values[start - 1 : stop - 1] = torch.from_numpy(
            np.asarray(matrix[start:stop], dtype=np.float32)
        ).to("cuda")
    mean = values.mean(dim=0, keepdim=True)
    values.sub_(mean)
    total_sum_squares = 0.0
    for start in range(0, rows - 1, chunk_size):
        stop = min(start + chunk_size, rows - 1)
        total_sum_squares += float((values[start:stop] * values[start:stop]).sum().item())
    total_variance = total_sum_squares / max(rows - 2, 1)
    started = time.perf_counter()
    u, singular_values, _ = torch.pca_lowrank(
        values,
        q=max_components,
        center=False,
        niter=3,
    )
    scores = u * singular_values.unsqueeze(0)
    output = np.lib.format.open_memmap(
        output_path.with_suffix(".partial.npy"),
        mode="w+",
        dtype=np.float16,
        shape=(rows, max_components),
    )
    output[0] = 0
    for start in range(0, rows - 1, chunk_size):
        stop = min(start + chunk_size, rows - 1)
        output[start + 1 : stop + 1] = scores[start:stop].cpu().numpy().astype(np.float16)
    output.flush()
    del output
    output_path.with_suffix(".partial.npy").replace(output_path)
    eigenvalues = singular_values.square().cpu().numpy() / max(rows - 2, 1)
    details = {
        "method": "centered torch.pca_lowrank over vocabulary rows 1..N",
        "max_components": max_components,
        "niter": 3,
        "seconds": time.perf_counter() - started,
        "total_variance": total_variance,
        "eigenvalues": eigenvalues.tolist(),
    }
    metadata_path.write_text(json.dumps(details), encoding="utf-8")
    del values, mean, u, singular_values, scores
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r"), details


def fit_random_projection_max(
    matrix: np.ndarray,
    output_path: Path,
    max_components: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path, mmap_mode="r")
    generator = torch.Generator(device="cuda").manual_seed(SPLIT_SEED)
    projection = torch.randn(
        (matrix.shape[1], max_components),
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    ) / math.sqrt(max_components)
    partial = output_path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(matrix.shape[0], max_components),
    )
    output[0] = 0
    for start in range(1, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        values = torch.from_numpy(np.asarray(matrix[start:stop], dtype=np.float32)).to("cuda")
        output[start:stop] = (values @ projection).cpu().numpy().astype(np.float16)
    output.flush()
    del output, projection
    partial.replace(output_path)
    torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r")


def train_extended(
    matrix: np.ndarray,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    validation_tokens: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    physical_batch_size: int,
    effective_batch_size: int,
    max_epochs: int,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict, dict]:
    set_seed(seed)
    device = torch.device("cuda")
    model = FrozenEmbeddingCNN(matrix).to(device)
    accumulation = max(1, effective_batch_size // physical_batch_size)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_labels)),
        batch_size=physical_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_tokens)),
        batch_size=physical_batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    positive_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (tokens, labels) in enumerate(train_loader, start=1):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            (criterion(model(tokens), labels) / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_logits = predict(model, validation_loader, device)
        validation_loss = nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(validation_logits),
            torch.from_numpy(validation_labels.astype(np.float32)),
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32),
        ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if key != "embedding.weight"
            }
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state, strict=False)
    validation_logits = predict(model, validation_loader, device)
    costs = {
        "epochs": epoch,
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": physical_batch_size * accumulation,
        "best_validation_loss": best_loss,
        "train_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1_000_000,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stored_embedding_mb": matrix.nbytes / 1_000_000,
    }
    validation_metrics = evaluate(validation_labels, validation_logits)
    del model
    torch.cuda.empty_cache()
    return best_state, validation_logits, validation_metrics, costs


def infer_state(
    matrix: np.ndarray,
    state: dict[str, torch.Tensor],
    tokens: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    device = torch.device("cuda")
    model = FrozenEmbeddingCNN(matrix).to(device)
    model.load_state_dict(state, strict=False)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tokens)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    started = time.perf_counter()
    logits = predict(model, loader, device)
    elapsed = (time.perf_counter() - started) * 1000 / len(tokens)
    del model
    torch.cuda.empty_cache()
    return logits, elapsed


def slug(text: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in text).strip("_")


def perturb_text(text: str, mode: str, seed: int) -> str:
    rng = random.Random(seed)
    words = text.split()
    changed = []
    for word in words:
        if mode == "character_swap" and len(word) >= 5 and word.isalpha() and rng.random() < 0.35:
            index = rng.randrange(1, len(word) - 2)
            word = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
        elif mode == "space_insertion" and len(word) >= 6 and word.isalpha() and rng.random() < 0.35:
            index = rng.randrange(2, len(word) - 2)
            word = word[:index] + " " + word[index:]
        elif mode == "leetspeak" and rng.random() < 0.35:
            word = word.translate(str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}))
        elif mode == "url_variation" and ("http" in word or "www" in word or ".com" in word):
            word = word.replace("http", "hxxp").replace(".", " dot ")
        changed.append(word)
    return " ".join(changed)


def paired_group_bootstrap(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    repetitions: int = 2000,
) -> tuple[float, float, float]:
    merged = frame_a.merge(
        frame_b,
        on=["row_index", "group_id", "label"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    groups = merged["group_id"].unique()
    grouped_indices = {group: values.index.to_numpy() for group, values in merged.groupby("group_id")}
    rng = np.random.default_rng(20260915)
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([grouped_indices[group] for group in sampled])
        sample = merged.loc[indices]
        differences.append(
            matthews_corrcoef(sample["label"], sample["prediction_a"])
            - matthews_corrcoef(sample["label"], sample["prediction_b"])
        )
    observed = matthews_corrcoef(merged["label"], merged["prediction_a"]) - matthews_corrcoef(
        merged["label"], merged["prediction_b"]
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mbert-matrix", type=Path, required=True)
    parser.add_argument("--qwen-matrix", type=Path, required=True)
    parser.add_argument("--reference-vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        raise ValueError("Vocabulary order does not match the reference experiment.")
    tokens = texts_to_padded(data["clean_text"], word_index, args.max_length)
    labels = data["label"].to_numpy(dtype=np.int64)
    mbert = np.load(args.mbert_matrix, mmap_mode="r")
    qwen = np.load(args.qwen_matrix, mmap_mode="r")
    if mbert.shape[0] != qwen.shape[0] or mbert.shape[0] != len(words) + 1:
        raise ValueError("Embedding matrices and grouped vocabulary do not align.")

    concat = make_standardized_concat(
        mbert,
        qwen,
        args.output_dir / "embedding_mbert_qwen_standardized_padding0.npy",
    )
    max_dim = max(args.dims)
    pca_max, pca_metadata = fit_pca_max(
        concat,
        args.output_dir / f"embedding_mbert_qwen_pca{max_dim}_max.npy",
        max_dim,
    )
    rp_max = fit_random_projection_max(
        concat,
        args.output_dir / f"embedding_mbert_qwen_rp{max_dim}_max.npy",
        max_dim,
    )

    selection_path = args.output_dir / "dimension_selection.csv"
    selection_rows = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
    completed_selection = {str(row["configuration"]) for row in selection_rows}
    for method, maximum in (("PCA", pca_max), ("RP", rp_max)):
        for dimension in args.dims:
            name = f"Concat+{method}-{dimension}"
            if name in completed_selection:
                continue
            matrix = maximum[:, :dimension]
            if method == "RP":
                matrix = np.asarray(matrix, dtype=np.float32) * math.sqrt(max_dim / dimension)
                matrix[0] = 0
            state, _, validation_metrics, costs = train_extended(
                matrix,
                tokens[train], labels[train], tokens[validation], labels[validation],
                SPLIT_SEED, args.physical_batch_size, args.effective_batch_size, args.max_epochs,
            )
            torch.save(state, args.output_dir / f"selection_state_{slug(name)}.pt")
            retained = None
            if method == "PCA":
                eigenvalues = np.asarray(pca_metadata["eigenvalues"])
                retained = float(eigenvalues[:dimension].sum() / pca_metadata["total_variance"])
            selection_rows.append({
                "configuration": name,
                "method": method,
                "dimension": dimension,
                "retained_variance": retained,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **costs,
            })
            pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
            print(name, validation_metrics, flush=True)

    selection = pd.DataFrame(selection_rows)
    pca_selection = selection[selection["method"] == "PCA"].sort_values(
        ["validation_mcc", "dimension"], ascending=[False, True]
    )
    selected_dim = int(pca_selection.iloc[0]["dimension"])
    selected_pca = pca_max[:, :selected_dim]
    qwen_pca, qwen_pca_metadata = fit_pca_max(
        qwen,
        args.output_dir / f"embedding_qwen_pca{selected_dim}.npy",
        selected_dim,
    )
    selected_rp = np.asarray(rp_max[:, :selected_dim], dtype=np.float32) * math.sqrt(
        max_dim / selected_dim
    )
    selected_rp[0] = 0
    final_configurations = {
        "mBERT": mbert,
        "Qwen2.5 raw": qwen,
        f"Qwen2.5+PCA-{selected_dim}": qwen_pca,
        "mBERT+Qwen standardized concat": concat,
        f"mBERT+Qwen Concat+PCA-{selected_dim}": selected_pca,
        f"mBERT+Qwen Concat+RP-{selected_dim}": selected_rp,
    }

    runs_path = args.output_dir / "final_runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {(str(row["configuration"]), int(row["seed"])) for row in runs}
    prediction_frames = []
    for name, matrix in final_configurations.items():
        for seed in SEEDS:
            prediction_path = args.output_dir / f"prediction_{slug(name)}_seed{seed}.csv"
            if (name, seed) not in completed:
                state, _, validation_metrics, costs = train_extended(
                    matrix,
                    tokens[train], labels[train], tokens[validation], labels[validation],
                    seed, args.physical_batch_size, args.effective_batch_size, args.max_epochs,
                )
                torch.save(state, args.output_dir / f"final_state_{slug(name)}_seed{seed}.pt")
                logits, inference_ms = infer_state(
                    matrix, state, tokens[test], args.physical_batch_size * 2
                )
                test_metrics = evaluate(labels[test], logits)
                costs["inference_ms_per_message"] = inference_ms
                runs.append({
                    "configuration": name,
                    "seed": seed,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **test_metrics,
                    **costs,
                })
                probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                pd.DataFrame({
                    "configuration": name,
                    "seed": seed,
                    "row_index": test,
                    "group_id": data.iloc[test]["group_id"].to_numpy(),
                    "label": labels[test],
                    "probability": probabilities,
                    "prediction": (probabilities >= 0.5).astype(np.int8),
                }).to_csv(prediction_path, index=False)
                pd.DataFrame(runs).to_csv(runs_path, index=False)
                print(name, seed, test_metrics, flush=True)
            prediction_frames.append(pd.read_csv(prediction_path))

    runs_frame = pd.DataFrame(runs)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "mcc", "roc_auc", "ece_15"]
    summary = runs_frame.groupby("configuration")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)
    summary.to_csv(args.output_dir / "final_summary.csv", index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    seed42 = {name: frame for name, frame in predictions[predictions["seed"] == 42].groupby("configuration")}
    reference_name = f"mBERT+Qwen Concat+PCA-{selected_dim}"
    bootstrap_rows = []
    for comparison_name, comparison_frame in seed42.items():
        if comparison_name == reference_name:
            continue
        observed, low, high = paired_group_bootstrap(seed42[reference_name], comparison_frame)
        bootstrap_rows.append({
            "reference": reference_name,
            "comparison": comparison_name,
            "mcc_difference": observed,
            "ci_low": low,
            "ci_high": high,
        })
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "paired_group_bootstrap_seed42.csv", index=False)

    seed_interval_rows = []
    pivot = runs_frame.pivot(index="seed", columns="configuration", values="mcc")
    for comparison_name in pivot.columns:
        if comparison_name == reference_name:
            continue
        differences = pivot[reference_name] - pivot[comparison_name]
        half_width = t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / math.sqrt(len(differences))
        seed_interval_rows.append({
            "reference": reference_name,
            "comparison": comparison_name,
            "mean_mcc_difference": differences.mean(),
            "ci_low": differences.mean() - half_width,
            "ci_high": differences.mean() + half_width,
        })
    pd.DataFrame(seed_interval_rows).to_csv(args.output_dir / "paired_seed_intervals.csv", index=False)

    # Seed-42 robustness comparison against the strongest sparse baseline.
    train_text = data.iloc[train]["clean_text"]
    test_text = data.iloc[test]["clean_text"].reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=75_000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(train_text)
    sparse = LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=500, random_state=42, solver="liblinear"
    ).fit(x_train, labels[train])
    selected_state = torch.load(
        args.output_dir / f"final_state_{slug(reference_name)}_seed42.pt",
        map_location="cpu",
        weights_only=True,
    )
    robustness_rows = []
    for mode in ("clean", "character_swap", "space_insertion", "leetspeak", "url_variation"):
        if mode == "clean":
            perturbed = test_text
        else:
            perturbed = pd.Series(
                [perturb_text(text, mode, 42_000_000 + int(row)) for row, text in enumerate(test_text)]
            )
        sparse_probability = sparse.predict_proba(vectorizer.transform(perturbed))[:, 1]
        sparse_logits = np.log(np.clip(sparse_probability, 1e-7, 1 - 1e-7) / np.clip(1 - sparse_probability, 1e-7, 1))
        robustness_rows.append({"model": "Char-TFIDF + LR", "perturbation": mode, **evaluate(labels[test], sparse_logits)})
        perturbed_tokens = texts_to_padded(perturbed, word_index, args.max_length)
        logits, _ = infer_state(selected_pca, selected_state, perturbed_tokens, args.physical_batch_size * 2)
        robustness_rows.append({"model": reference_name, "perturbation": mode, **evaluate(labels[test], logits)})
    pd.DataFrame(robustness_rows).to_csv(args.output_dir / "obfuscation_robustness_seed42.csv", index=False)

    plt.figure(figsize=(6.8, 4.2))
    for method, frame in selection.groupby("method"):
        frame = frame.sort_values("dimension")
        plt.plot(frame["dimension"], frame["validation_mcc"], marker="o", label=method)
    plt.xlabel("Projection dimension")
    plt.ylabel("Validation MCC")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "validation_mcc_dimension_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    cost = runs_frame.groupby("configuration")["stored_embedding_mb"].mean()
    plot_labels = {
        "mBERT": ("mBERT", (6, 5), "left"),
        "Qwen2.5 raw": ("Qwen raw", (6, -12), "left"),
        "Qwen2.5+PCA-1024": ("Qwen PCA", (6, 7), "left"),
        "mBERT+Qwen standardized concat": ("Fusion concat", (-6, 7), "right"),
        "mBERT+Qwen Concat+RP-1024": ("Fusion RP", (6, -13), "left"),
        "mBERT+Qwen Concat+PCA-1024": ("Fusion PCA", (6, 11), "left"),
    }
    for _, row in summary.iterrows():
        name = row["configuration"]
        plt.errorbar(cost[name], row["mcc_mean"], yerr=row["mcc_std"], fmt="o",
                     markersize=6, capsize=3, linewidth=1.2)
        label, offset, alignment = plot_labels[name]
        plt.annotate(label, (cost[name], row["mcc_mean"]), xytext=offset,
                     textcoords="offset points", fontsize=9, ha=alignment)
    plt.xlabel("Stored embedding matrix (MB)")
    plt.ylabel("Test MCC (five-seed mean)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_dir / "mcc_storage_tradeoff.png", dpi=300)
    plt.close()

    metadata = {
        "selection_rule": "highest validation MCC; smaller dimension breaks ties",
        "selected_pca_dimension": selected_dim,
        "pca_dimensions": args.dims,
        "qwen_selected_width_retained_variance": float(
            np.asarray(qwen_pca_metadata["eigenvalues"]).sum()
            / qwen_pca_metadata["total_variance"]
        ),
        "training_seeds": SEEDS,
        "split_seed": SPLIT_SEED,
        "padding_row_reset_to_zero": True,
        "physical_batch_size": args.physical_batch_size,
        "effective_batch_size_via_accumulation": args.effective_batch_size,
        "llama_status": "not run; no local Llama-2 checkpoint or legacy vocabulary matrix available",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
