"""Evaluate centered PCA-1024 on the same sparse TF-IDF fixed test split as SVD."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from evaluate_tfidf_svd1024_test import classifiers
from grouped_baseline_benchmark import metrics
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
FINAL_DIMENSION = 1024


def as_torch_csr(x: sp.csr_matrix, device: str) -> torch.Tensor:
    x = x.astype(np.float32, copy=False)
    return torch.sparse_csr_tensor(
        torch.as_tensor(x.indptr, device=device),
        torch.as_tensor(x.indices, device=device),
        torch.as_tensor(x.data, device=device),
        size=x.shape,
        device=device,
    )


def centered_randomized_pca(
    x_train: sp.csr_matrix,
    x_test: sp.csr_matrix,
    *,
    components: int,
    oversamples: int,
    power_iterations: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a centered randomized PCA basis on training data only."""
    n_train, n_features = x_train.shape
    rank = min(components + oversamples, n_train, n_features)
    if rank <= components:
        raise ValueError("PCA requires rank greater than the requested width")
    mean = np.asarray(x_train.mean(axis=0), dtype=np.float32).ravel()
    total_ss = float(np.square(x_train.data, dtype=np.float64).sum()
                     - n_train * np.square(mean, dtype=np.float64).sum())
    x_gpu = as_torch_csr(x_train, device)
    mean_gpu = torch.as_tensor(mean, device=device)

    def forward(v: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(x_gpu, v) - (mean_gpu @ v).unsqueeze(0)

    def backward(q: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(x_gpu.transpose(0, 1), q) - (
            mean_gpu[:, None] * q.sum(dim=0)[None, :]
        )

    generator = torch.Generator(device=device).manual_seed(SPLIT_SEED)
    omega = torch.randn(n_features, rank, generator=generator, device=device)
    y = forward(omega)
    del omega
    for _ in range(power_iterations):
        q = torch.linalg.qr(y, mode="reduced")[0]
        z = backward(q)
        del y, q
        z = torch.linalg.qr(z, mode="reduced")[0]
        y = forward(z)
        del z
    q = torch.linalg.qr(y, mode="reduced")[0]
    del y
    bt = backward(q)
    del q
    gram = bt.T @ bt
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    del gram
    indices = torch.argsort(eigenvalues, descending=True)[:components]
    strengths = eigenvalues[indices].clamp_min(1e-20)
    basis = bt @ (eigenvectors[:, indices] / strengths.sqrt()[None, :])
    retained_variance = float(strengths.sum().item() / total_ss)
    del bt, eigenvalues, eigenvectors, indices, strengths

    train_scores = forward(basis).cpu().numpy()
    del x_gpu
    test_gpu = as_torch_csr(x_test, device)
    test_scores = (
        torch.sparse.mm(test_gpu, basis) - (mean_gpu @ basis).unsqueeze(0)
    ).cpu().numpy()
    del test_gpu, basis, mean_gpu
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_scores).astype(np.float32),
        scaler.transform(test_scores).astype(np.float32),
        retained_variance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--features", nargs="+", choices=("char", "word"),
                        default=("char", "word"))
    parser.add_argument("--oversamples", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    if args.fold_file:
        with np.load(args.fold_file) as parts:
            train, validation, test = (parts[key] for key in ("train", "validation", "test"))
    else:
        train, validation, test = split_groups(data, SPLIT_SEED)
    train_text = data.iloc[train]["clean_text"]
    test_text = data.iloc[test]["clean_text"]
    y_train = data.iloc[train]["label"].to_numpy()
    y_test = data.iloc[test]["label"].to_numpy()
    counts = np.bincount(y_train)
    sample_weight = len(y_train) / (2 * counts[y_train])

    checkpoint = args.output_dir / "runs_checkpoint.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(str(row["model"]), int(row["seed"])) for row in rows}

    for feature in args.features:
        start = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb" if feature == "char" else "word",
            ngram_range=(3, 5) if feature == "char" else (1, 2),
            min_df=2,
            max_features=75_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train_sparse = vectorizer.fit_transform(train_text)
        x_test_sparse = vectorizer.transform(test_text)
        vectorizer_seconds = time.perf_counter() - start
        print(feature, "sparse", x_train_sparse.shape, x_train_sparse.nnz,
              "vectorizer_s", round(vectorizer_seconds, 2), flush=True)

        start = time.perf_counter()
        x_train, x_test, variance = centered_randomized_pca(
            x_train_sparse,
            x_test_sparse,
            components=FINAL_DIMENSION,
            oversamples=args.oversamples,
            power_iterations=args.power_iterations,
            device=args.device,
        )
        projection_seconds = time.perf_counter() - start
        input_dimension = x_train_sparse.shape[1]
        del vectorizer, x_train_sparse, x_test_sparse
        gc.collect()
        print(feature, "pca", "variance", variance, "projection_s",
              round(projection_seconds, 2), flush=True)

        for seed in args.seeds:
            for old_name, model in classifiers(feature, seed).items():
                name = old_name.replace("SVD-1024", "PCA-1024")
                if (name, seed) in completed:
                    continue
                start = time.perf_counter()
                model.fit(x_train, y_train, sample_weight=sample_weight)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = model.predict(x_test)
                inference_ms = (time.perf_counter() - start) * 1000 / len(test)
                rows.append({
                    "feature": feature,
                    "seed": seed,
                    "model": name,
                    "split_seed": SPLIT_SEED,
                    "train_samples": len(train),
                    "validation_samples": len(validation),
                    "test_samples": len(test),
                    "input_dimension": input_dimension,
                    "final_dimension": FINAL_DIMENSION,
                    "retained_variance": variance,
                    "vectorizer_seconds": vectorizer_seconds,
                    "projection_seconds": projection_seconds,
                    "classifier_fit_seconds": fit_seconds,
                    "inference_ms_per_message": inference_ms,
                    **metrics(y_test, prediction),
                })
                pd.DataFrame({
                    "model": name,
                    "seed": seed,
                    "sample_index": test,
                    "group_id": data.iloc[test]["group_id"].to_numpy(),
                    "language_column": data.iloc[test]["language_column"].to_numpy(),
                    "label": y_test,
                    "prediction": prediction,
                }).to_csv(
                    args.output_dir
                    / (name.lower().replace(" ", "_").replace("+", "plus")
                       + f"_seed{seed}.csv"),
                    index=False,
                )
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                completed.add((name, seed))
                print(feature, seed, name, "mcc", rows[-1]["mcc"], flush=True)
        del x_train, x_test
        gc.collect()

    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "mcc", "cohen_kappa"]
    summary = runs.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    dimensions = runs.groupby("model").agg(
        input_dimension=("input_dimension", "first"),
        final_dimension=("final_dimension", "first"),
        retained_variance_mean=("retained_variance", "mean"),
        seeds=("seed", "count"),
    )
    summary = summary.join(dimensions).reset_index().sort_values(
        "mcc_mean", ascending=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
