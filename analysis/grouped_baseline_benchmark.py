"""Run leakage-controlled classical baselines for the revision audit."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD

from revision_audit import SEEDS, load_long_dataset, split_groups


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "spam_f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def timed_predict(model, vectorizer, texts: pd.Series) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    transformed = vectorizer.transform(texts)
    predictions = model.predict(transformed)
    elapsed = time.perf_counter() - start
    return predictions, elapsed * 1000 / len(texts)


def fit_sparse_model(model, x_train, y_train) -> tuple[object, float]:
    start = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, seed in enumerate(SEEDS, start=1):
        if args.fold_dir:
            with np.load(args.fold_dir / f"fold_{fold}.npz") as parts:
                train, test = parts["train"], parts["test"]
        else:
            train, _, test = split_groups(data, seed)
        model_seed = 42 if args.fold_dir else seed
        train_text = data.iloc[train]["clean_text"]
        test_text = data.iloc[test]["clean_text"]
        y_train = data.iloc[train]["label"].to_numpy()
        y_test = data.iloc[test]["label"].to_numpy()

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

        matrices = {}
        vectorize_times = {}
        for feature_name, vectorizer in vectorizers.items():
            start = time.perf_counter()
            matrices[feature_name] = vectorizer.fit_transform(train_text)
            vectorize_times[feature_name] = time.perf_counter() - start

        sparse_models = {
            "Char-TFIDF + LR": (
                "char",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=model_seed,
                    solver="liblinear",
                ),
            ),
            "Char-TFIDF + Linear SVM": (
                "char",
                LinearSVC(C=1.0, class_weight="balanced", random_state=model_seed),
            ),
            "Word-TFIDF + LR": (
                "word",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=model_seed,
                    solver="liblinear",
                ),
            ),
            "Word-TFIDF + Linear SVM": (
                "word",
                LinearSVC(C=1.0, class_weight="balanced", random_state=model_seed),
            ),
        }

        seed_predictions: dict[str, np.ndarray] = {}
        for model_name, (feature_name, model) in sparse_models.items():
            model, fit_seconds = fit_sparse_model(model, matrices[feature_name], y_train)
            pred, inference_ms = timed_predict(model, vectorizers[feature_name], test_text)
            seed_predictions[model_name] = pred
            rows.append(
                {
                    "seed": model_seed,
                    "fold": fold if args.fold_dir else None,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "feature_dimension": matrices[feature_name].shape[1],
                    "vectorizer_fit_seconds": vectorize_times[feature_name],
                    "classifier_fit_seconds": fit_seconds,
                    "inference_ms_per_message": inference_ms,
                    "serialized_model_mb": len(pickle.dumps((vectorizers[feature_name], model)))
                    / 1_000_000,
                    **metrics(y_test, pred),
                }
            )

        # Shared latent semantic features make nonlinear baselines tractable.
        svd_start = time.perf_counter()
        svd = TruncatedSVD(n_components=128, n_iter=4, random_state=model_seed)
        x_train_svd = svd.fit_transform(matrices["word"])
        scaler = StandardScaler()
        x_train_dense = scaler.fit_transform(x_train_svd).astype(np.float32)
        svd_fit_seconds = time.perf_counter() - svd_start

        dense_models = {
            "Word-TFIDF + SVD + MLP": MLPClassifier(
                hidden_layer_sizes=(64,),
                batch_size=512,
                early_stopping=True,
                max_iter=50,
                n_iter_no_change=5,
                random_state=model_seed,
            ),
            "Word-TFIDF + SVD + HistGB": HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=120,
                max_leaf_nodes=31,
                random_state=model_seed,
            ),
        }

        class_counts = np.bincount(y_train)
        sample_weight = len(y_train) / (2 * class_counts[y_train])
        for model_name, model in dense_models.items():
            start = time.perf_counter()
            model.fit(x_train_dense, y_train, sample_weight=sample_weight)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            x_test_word = vectorizers["word"].transform(test_text)
            x_test_dense = scaler.transform(svd.transform(x_test_word)).astype(np.float32)
            pred = model.predict(x_test_dense)
            inference_ms = (time.perf_counter() - start) * 1000 / len(test)
            seed_predictions[model_name] = pred
            rows.append(
                {
                    "seed": model_seed,
                    "fold": fold if args.fold_dir else None,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "feature_dimension": 128,
                    "vectorizer_fit_seconds": vectorize_times["word"] + svd_fit_seconds,
                    "classifier_fit_seconds": fit_seconds,
                    "inference_ms_per_message": inference_ms,
                    "serialized_model_mb": len(
                        pickle.dumps((vectorizers["word"], svd, scaler, model))
                    )
                    / 1_000_000,
                    **metrics(y_test, pred),
                }
            )

        if args.fold_dir or seed == 42:
            base = data.iloc[test][["group_id", "language_column", "label"]].reset_index(drop=True)
            for model_name, pred in seed_predictions.items():
                frame = base.copy()
                if args.fold_dir:
                    frame["fold"] = fold
                frame["model"] = model_name
                frame["prediction"] = pred
                prediction_frames.append(frame)

    runs = pd.DataFrame(rows)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc", "cohen_kappa"]
    summary = runs.groupby("model")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("mcc_mean", ascending=False)

    efficiency_columns = [
        "vectorizer_fit_seconds",
        "classifier_fit_seconds",
        "inference_ms_per_message",
        "serialized_model_mb",
    ]
    efficiency = runs.groupby("model")[efficiency_columns].agg(["mean", "std"])
    efficiency.columns = [f"{metric}_{stat}" for metric, stat in efficiency.columns]
    efficiency = efficiency.reset_index()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    language_rows = []
    for (model_name, language), frame in predictions.groupby(["model", "language_column"]):
        language_rows.append(
            {
                "model": model_name,
                "language_column": language,
                "n": len(frame),
                **metrics(frame["label"].to_numpy(), frame["prediction"].to_numpy()),
            }
        )
    language_metrics = pd.DataFrame(language_rows)

    best_model = summary.iloc[0]["model"]
    best = predictions[predictions["model"] == best_model].sort_values(
        ["group_id", "language_column"]
    )
    tests = []
    for model_name in summary["model"]:
        if model_name == best_model:
            continue
        other = predictions[predictions["model"] == model_name].sort_values(
            ["group_id", "language_column"]
        )
        y = best["label"].to_numpy()
        best_correct = best["prediction"].to_numpy() == y
        other_correct = other["prediction"].to_numpy() == y
        best_only = int(np.sum(best_correct & ~other_correct))
        other_only = int(np.sum(~best_correct & other_correct))
        discordant = best_only + other_only
        p_value = 1.0 if discordant == 0 else binomtest(best_only, discordant, 0.5).pvalue
        tests.append(
            {
                "seed": 42,
                "reference_model": best_model,
                "comparison_model": model_name,
                "reference_only_correct": best_only,
                "comparison_only_correct": other_only,
                "exact_mcnemar_p": p_value,
            }
        )

    runs.to_csv(args.output_dir / "grouped_baseline_runs.csv", index=False)
    summary.to_csv(args.output_dir / "grouped_baseline_summary.csv", index=False)
    efficiency.to_csv(args.output_dir / "grouped_baseline_efficiency.csv", index=False)
    prediction_name = ("grouped_baseline_all_fold_predictions.csv" if args.fold_dir
                       else "grouped_baseline_seed42_predictions.csv")
    predictions.to_csv(args.output_dir / prediction_name, index=False)
    language_metrics.to_csv(args.output_dir / "grouped_baseline_language_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(args.output_dir / "grouped_baseline_mcnemar.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("\nEfficiency\n", efficiency.round(4).to_string(index=False))
    print("\nBest model for seed-42 McNemar reference:", best_model)


if __name__ == "__main__":
    main()
