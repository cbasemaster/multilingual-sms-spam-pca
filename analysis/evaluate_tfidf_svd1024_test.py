"""Evaluate 1,024-dimensional sparse-TF-IDF projections on the fixed grouped test set."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from grouped_baseline_benchmark import metrics
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
FINAL_DIMENSION = 1024


def classifiers(feature_name: str, seed: int) -> dict[str, object]:
    prefix = "Char-TFIDF" if feature_name == "char" else "Word-TFIDF"
    result: dict[str, object] = {
        f"{prefix} + SVD-1024 + LR": LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=500,
            random_state=seed,
            solver="liblinear",
        ),
        f"{prefix} + SVD-1024 + Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    if feature_name == "word":
        result.update(
            {
                f"{prefix} + SVD-1024 + MLP": MLPClassifier(
                    hidden_layer_sizes=(64,),
                    batch_size=512,
                    early_stopping=True,
                    max_iter=50,
                    n_iter_no_change=5,
                    random_state=seed,
                ),
                f"{prefix} + SVD-1024 + HistGB": HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=120,
                    max_leaf_nodes=31,
                    random_state=seed,
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--svd-iterations", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    train, validation, test = split_groups(data, SPLIT_SEED)
    train_text = data.iloc[train]["clean_text"]
    test_text = data.iloc[test]["clean_text"]
    y_train = data.iloc[train]["label"].to_numpy()
    y_test = data.iloc[test]["label"].to_numpy()
    class_counts = np.bincount(y_train)
    sample_weight = len(y_train) / (2 * class_counts[y_train])

    checkpoint = args.output_dir / "runs_checkpoint.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {(str(row["model"]), int(row["seed"])) for row in rows}

    vectorizers = {
        "char": TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=75_000,
            sublinear_tf=True,
        ),
        "word": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=75_000,
            sublinear_tf=True,
        ),
    }

    for feature_name, vectorizer in vectorizers.items():
        start = time.perf_counter()
        x_train_sparse = vectorizer.fit_transform(train_text)
        x_test_sparse = vectorizer.transform(test_text)
        vectorizer_seconds = time.perf_counter() - start

        start = time.perf_counter()
        svd = TruncatedSVD(
            n_components=FINAL_DIMENSION,
            n_iter=args.svd_iterations,
            random_state=SPLIT_SEED,
        )
        x_train = svd.fit_transform(x_train_sparse).astype(np.float32)
        x_test = svd.transform(x_test_sparse).astype(np.float32)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
        projection_seconds = time.perf_counter() - start
        retained_variance = float(svd.explained_variance_ratio_.sum())

        for seed in SEEDS:
            for model_name, model in classifiers(feature_name, seed).items():
                if (model_name, seed) in completed:
                    continue
                start = time.perf_counter()
                model.fit(x_train, y_train, sample_weight=sample_weight)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = model.predict(x_test)
                inference_ms = (time.perf_counter() - start) * 1000 / len(test)
                rows.append(
                    {
                        "feature": feature_name,
                        "seed": seed,
                        "model": model_name,
                        "split_seed": SPLIT_SEED,
                        "train_samples": len(train),
                        "validation_samples": len(validation),
                        "test_samples": len(test),
                        "input_dimension": x_train_sparse.shape[1],
                        "final_dimension": FINAL_DIMENSION,
                        "retained_variance": retained_variance,
                        "vectorizer_seconds": vectorizer_seconds,
                        "projection_seconds": projection_seconds,
                        "classifier_fit_seconds": fit_seconds,
                        "inference_ms_per_message": inference_ms,
                        **metrics(y_test, prediction),
                    }
                )
                pd.DataFrame(
                    {
                        "model": model_name,
                        "seed": seed,
                        "sample_index": test,
                        "group_id": data.iloc[test]["group_id"].to_numpy(),
                        "language_column": data.iloc[test]["language_column"].to_numpy(),
                        "label": y_test,
                        "prediction": prediction,
                    }
                ).to_csv(
                    args.output_dir
                    / (model_name.lower().replace(" ", "_").replace("+", "plus") + f"_seed{seed}.csv"),
                    index=False,
                )
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                print(feature_name, seed, model_name, retained_variance, flush=True)

    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "mcc", "cohen_kappa"]
    summary = runs.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    dimensions = runs.groupby("model").agg(
        input_dimension=("input_dimension", "first"),
        final_dimension=("final_dimension", "first"),
        retained_variance_mean=("retained_variance", "mean"),
        retained_variance_std=("retained_variance", "std"),
        seeds=("seed", "count"),
    )
    summary = summary.join(dimensions).reset_index().sort_values("mcc_mean", ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
