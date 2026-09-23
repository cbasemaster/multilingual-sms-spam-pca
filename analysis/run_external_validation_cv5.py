"""External validation using frozen fold-specific vocabularies and PCA tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from grouped_embedding_fusion_experiment import evaluate, texts_to_padded
from grouped_qwen_pca_sweep_experiment import infer_state, train_extended
from revision_audit import load_long_dataset


STATIC_CONFIGS = (
    "mBERT",
    "Qwen2.5 raw",
    "mBERT+Qwen concat no PCA",
    "mBERT+Qwen validation-selected PCA",
)


def slug(name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def load_external(path: Path, strict: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    key = "eligible_strict" if strict else "eligible_primary"
    values = frame[key]
    if values.dtype != bool:
        values = values.astype(str).str.lower().eq("true")
    return frame.loc[values, ["dataset", "source_record", "clean_text", "label"]].reset_index(drop=True)


def matrices_for_fold(fold_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    core = fold_dir / "fusion_core"
    selected_dimension = int(json.loads((core / "metadata.json").read_text())["selected_dimension"])
    return {
        "mBERT": np.load(fold_dir / "mbert.npy", mmap_mode="r"),
        "Qwen2.5 raw": np.load(fold_dir / "qwen.npy", mmap_mode="r"),
        "mBERT+Qwen concat no PCA": np.load(core / "standardized_concat.npy", mmap_mode="r"),
        "mBERT+Qwen validation-selected PCA": np.load(
            core / "fusion_pca2048.npy", mmap_mode="r"
        )[:, :selected_dimension],
    }, selected_dimension


def metrics_with_confusion(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    logits = np.log(
        np.clip(probabilities, 1e-7, 1 - 1e-7)
        / np.clip(1 - probabilities, 1e-7, 1 - 1e-7)
    )
    metrics: dict[str, float | int] = evaluate(labels, logits)
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def bootstrap_difference(
    labels: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    repetitions: int = 2000,
) -> tuple[float, float, float]:
    prediction_reference = reference >= 0.5
    prediction_comparison = comparison >= 0.5
    observed = matthews_corrcoef(labels, prediction_reference) - matthews_corrcoef(
        labels, prediction_comparison
    )
    by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    rng = np.random.default_rng(42)
    differences = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        sampled = np.concatenate(
            [indices[rng.integers(0, len(indices), len(indices))] for indices in by_class]
        )
        differences[repeat] = matthews_corrcoef(
            labels[sampled], prediction_reference[sampled]
        ) - matthews_corrcoef(labels[sampled], prediction_comparison[sampled])
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--external", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = load_long_dataset(args.development)
    external_frames = {frame.iloc[0]["dataset"]: frame for path in args.external
                       for frame in [load_external(path, args.strict)]}
    run_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for fold in args.folds:
        fold_path = args.fold_dir / f"fold_{fold}"
        with np.load(args.fold_dir / f"fold_{fold}.npz") as split:
            train, validation, internal_test = (
                split[key] for key in ("train", "validation", "test")
            )
        vocabulary = pd.read_csv(
            fold_path / "training_vocabulary.csv", keep_default_na=False, dtype=str
        )["word"].tolist()
        word_index = {word: index + 1 for index, word in enumerate(vocabulary)}
        development_tokens = texts_to_padded(development["clean_text"], word_index, 128)
        labels = development["label"].to_numpy(dtype=np.int64)
        external_tokens: dict[str, np.ndarray] = {}
        for dataset_name, frame in external_frames.items():
            external_tokens[dataset_name] = texts_to_padded(frame["clean_text"], word_index, 128)
            token_lists = frame["clean_text"].str.split()
            total_tokens = int(token_lists.map(len).sum())
            known_tokens = int(token_lists.map(lambda words: sum(word in word_index for word in words)).sum())
            samples_with_known = int(token_lists.map(lambda words: any(word in word_index for word in words)).sum())
            coverage_rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "samples": len(frame),
                "token_coverage": known_tokens / max(total_tokens, 1),
                "samples_with_at_least_one_known_token": samples_with_known,
                "sample_coverage": samples_with_known / max(len(frame), 1),
            })

        matrices, selected_dimension = matrices_for_fold(fold_path)
        expected_runs = pd.read_csv(fold_path / "fusion_core" / "runs.csv").set_index("configuration")
        for configuration in STATIC_CONFIGS:
            matrix = matrices[configuration]
            recorded_configuration = (
                f"mBERT+Qwen PCA-{selected_dimension}"
                if configuration.endswith("validation-selected PCA")
                else configuration
            )
            state_path = args.output_dir / f"fold_{fold}_{slug(configuration)}_trainable_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                validation_metrics = {}
                costs = {}
            else:
                state, _, validation_metrics, costs = train_extended(
                    matrix,
                    development_tokens[train], labels[train],
                    development_tokens[validation], labels[validation],
                    42,
                    256,
                    1024,
                    args.max_epochs,
                )
                trainable_state = {
                    key: value for key, value in state.items() if key != "embedding.weight"
                }
                torch.save(trainable_state, state_path)

            internal_logits, _ = infer_state(matrix, state, development_tokens[internal_test], 512)
            internal_metrics = evaluate(labels[internal_test], internal_logits)
            expected_mcc = float(expected_runs.loc[recorded_configuration, "mcc"])
            if abs(internal_metrics["mcc"] - expected_mcc) > 1e-9:
                raise RuntimeError(
                    f"Fold {fold} {configuration} reconstruction mismatch: "
                    f"{internal_metrics['mcc']} versus {expected_mcc}"
                )

            for dataset_name, frame in external_frames.items():
                logits, inference_ms = infer_state(
                    matrix, state, external_tokens[dataset_name], 512
                )
                probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                prediction_path = args.output_dir / (
                    f"predictions_{slug(dataset_name)}_{slug(configuration)}_fold{fold}.csv"
                )
                pd.DataFrame({
                    "dataset": dataset_name,
                    "source_record": frame["source_record"],
                    "label": frame["label"],
                    "probability": probability,
                    "prediction": (probability >= 0.5).astype(np.int8),
                }).to_csv(prediction_path, index=False)
                run_rows.append({
                    "fold": fold,
                    "dataset": dataset_name,
                    "configuration": configuration,
                    "selected_dimension": (
                        selected_dimension
                        if configuration.endswith("validation-selected PCA")
                        else matrix.shape[1]
                    ),
                    "internal_test_mcc_reproduced": internal_metrics["mcc"],
                    "internal_test_mcc_recorded": expected_mcc,
                    "inference_ms_per_message": inference_ms,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **costs,
                    **metrics_with_confusion(frame["label"].to_numpy(), probability),
                })
            print(
                f"Fold {fold}: {configuration} reconstructed and externally evaluated",
                flush=True,
            )
            del state
            torch.cuda.empty_cache()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=75_000, sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(development.iloc[train]["clean_text"])
        sparse = LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=500,
            random_state=42, solver="liblinear",
        ).fit(x_train, labels[train])
        for dataset_name, frame in external_frames.items():
            probability = sparse.predict_proba(vectorizer.transform(frame["clean_text"]))[:, 1]
            pd.DataFrame({
                "dataset": dataset_name,
                "source_record": frame["source_record"],
                "label": frame["label"],
                "probability": probability,
                "prediction": (probability >= 0.5).astype(np.int8),
            }).to_csv(
                args.output_dir / (
                    "predictions_"
                    f"{slug(dataset_name)}_{slug('Character TF-IDF + logistic regression')}_fold{fold}.csv"
                ),
                index=False,
            )
            run_rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "configuration": "Character TF-IDF + logistic regression",
                "selected_dimension": int(len(vectorizer.vocabulary_)),
                "internal_test_mcc_reproduced": np.nan,
                "internal_test_mcc_recorded": np.nan,
                "inference_ms_per_message": np.nan,
                **metrics_with_confusion(frame["label"].to_numpy(), probability),
            })

        pd.DataFrame(run_rows).to_csv(args.output_dir / "external_validation_runs.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(args.output_dir / "vocabulary_coverage.csv", index=False)
        print(f"External validation fold {fold} complete", flush=True)

    runs = pd.DataFrame(run_rows)
    metrics = ["accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc", "balanced_accuracy", "roc_auc"]
    summary = runs.groupby(["dataset", "configuration"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "external_validation_summary.csv", index=False)

    ensemble_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for dataset_name, frame in external_frames.items():
        probabilities: dict[str, np.ndarray] = {}
        for configuration in [*STATIC_CONFIGS, "Character TF-IDF + logistic regression"]:
            files = sorted(args.output_dir.glob(
                f"predictions_{slug(dataset_name)}_{slug(configuration)}_fold*.csv"
            ))
            values = [pd.read_csv(path)["probability"].to_numpy() for path in files]
            if len(values) != len(args.folds):
                raise RuntimeError(f"Missing predictions for {dataset_name}, {configuration}")
            probabilities[configuration] = np.mean(values, axis=0)
            ensemble_rows.append({
                "dataset": dataset_name,
                "configuration": configuration,
                "fold_models": len(values),
                **metrics_with_confusion(frame["label"].to_numpy(), probabilities[configuration]),
            })
        reference_name = "mBERT+Qwen validation-selected PCA"
        for comparison_name, comparison in probabilities.items():
            if comparison_name == reference_name:
                continue
            observed, low, high = bootstrap_difference(
                frame["label"].to_numpy(), probabilities[reference_name], comparison
            )
            bootstrap_rows.append({
                "dataset": dataset_name,
                "reference": reference_name,
                "comparison": comparison_name,
                "mcc_difference": observed,
                "ci_low": low,
                "ci_high": high,
            })
    pd.DataFrame(ensemble_rows).to_csv(args.output_dir / "external_validation_ensemble.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "external_validation_bootstrap.csv", index=False)
    metadata = {
        "protocol": "external evaluation with five fold-specific models",
        "external_labels_used_for_training_or_selection": False,
        "development_training_seed": 42,
        "cnn_threshold": 0.5,
        "pca_width_selection": "existing fold-specific internal-validation selection",
        "pca_widths": [
            int(json.loads((args.fold_dir / f"fold_{fold}/fusion_core/metadata.json").read_text())["selected_dimension"])
            for fold in args.folds
        ],
        "primary_excludes": "exact cleaned-text overlap with development corpus",
        "strict_mode": args.strict,
        "interpretation": (
            "Mean and standard deviation describe sensitivity to fold-specific training composition; "
            "all fold models are evaluated on the same external samples."
        ),
    }
    (args.output_dir / "external_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(ensemble_rows).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
