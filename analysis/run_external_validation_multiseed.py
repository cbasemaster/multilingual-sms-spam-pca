"""Evaluate the principal static models externally over five folds and five seeds."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from grouped_embedding_fusion_experiment import evaluate, texts_to_padded
from grouped_qwen_pca_sweep_experiment import infer_state, perturb_text, train_extended
from revision_audit import load_long_dataset
from run_external_validation_cv5 import (
    bootstrap_difference,
    load_external,
    metrics_with_confusion,
    slug,
)


CONFIGURATIONS = (
    "mBERT",
    "Qwen2.5 raw",
    "mBERT+Qwen concat no PCA",
    "mBERT+Qwen validation-selected PCA",
)
WIDTHS = (768, 1024, 1536, 2048)
PERTURBATIONS = ("clean", "character_swap", "space_insertion", "leetspeak", "url_variation")


def fold_matrices(fold_path: Path) -> dict[str, np.ndarray]:
    core = fold_path / "fusion_core"
    return {
        "mBERT": np.load(fold_path / "mbert.npy", mmap_mode="r"),
        "Qwen2.5 raw": np.load(fold_path / "qwen.npy", mmap_mode="r"),
        "mBERT+Qwen concat no PCA": np.load(
            core / "standardized_concat.npy", mmap_mode="r"
        ),
        "pca": np.load(core / "fusion_pca2048.npy", mmap_mode="r"),
    }


def selected_widths(runs: pd.DataFrame) -> dict[tuple[int, int], int]:
    candidates = runs[runs["configuration"].isin(
        [f"mBERT+Qwen PCA-{width}" for width in WIDTHS]
    )].copy()
    candidates["width"] = candidates["configuration"].str.extract(r"(\d+)$").astype(int)
    result = {}
    for (fold, seed), group in candidates.groupby(["fold", "training_seed"]):
        winner = group.sort_values(
            ["validation_mcc", "width"], ascending=[False, True]
        ).iloc[0]
        result[(int(fold), int(seed))] = int(winner["width"])
    return result


def probability_path(
    output_dir: Path, dataset: str, configuration: str, fold: int, seed: int
) -> Path:
    return output_dir / "predictions" / (
        f"{slug(dataset)}_{slug(configuration)}_fold{fold}_seed{seed}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--static-runs", type=Path, required=True)
    parser.add_argument("--external", type=Path, nargs="+", required=True)
    parser.add_argument("--seed42-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 101])
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    static_runs = pd.read_csv(args.static_runs)
    if len(static_runs) != 33 * 25:
        raise RuntimeError("The complete 33-configuration by 25-run static suite is required")
    widths = selected_widths(static_runs)
    development = load_long_dataset(args.development)
    labels = development["label"].to_numpy(dtype=np.int64)
    external_frames = {
        frame.iloc[0]["dataset"]: frame
        for path in args.external
        for frame in [load_external(path, strict=False)]
    }
    checkpoint = args.output_dir / "external_multiseed_runs.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    robustness_checkpoint = args.output_dir / "robustness_multiseed_runs.csv"
    robustness_rows = (
        pd.read_csv(robustness_checkpoint).to_dict("records")
        if robustness_checkpoint.exists() else []
    )
    completed = {
        (int(row["fold"]), int(row["training_seed"]), str(row["configuration"]))
        for row in rows
    }

    for fold in args.folds:
        fold_path = args.fold_dir / f"fold_{fold}"
        with np.load(args.fold_dir / f"fold_{fold}.npz") as split:
            train, validation, test = (split[key] for key in ("train", "validation", "test"))
        vocabulary = pd.read_csv(
            fold_path / "training_vocabulary.csv", keep_default_na=False, dtype=str
        )["word"].tolist()
        word_index = {word: index + 1 for index, word in enumerate(vocabulary)}
        development_tokens = texts_to_padded(development["clean_text"], word_index, 128)
        external_tokens = {
            name: texts_to_padded(frame["clean_text"], word_index, 128)
            for name, frame in external_frames.items()
        }
        test_text = development.iloc[test]["clean_text"].reset_index(drop=True)
        perturbed_texts = {"clean": test_text}
        for mode in PERTURBATIONS[1:]:
            perturbed_texts[mode] = pd.Series([
                perturb_text(text, mode, 42_000_000 + int(row))
                for row, text in enumerate(test_text)
            ])
        perturbed_tokens = {
            mode: texts_to_padded(texts, word_index, 128)
            for mode, texts in perturbed_texts.items()
        }
        matrices = fold_matrices(fold_path)

        for seed in args.seeds:
            for configuration in CONFIGURATIONS:
                key = (fold, seed, configuration)
                expected_files = [
                    probability_path(args.output_dir, name, configuration, fold, seed)
                    for name in external_frames
                ]
                expected_robustness_files = [
                    args.output_dir / "predictions" / (
                        f"robustness_{slug(mode)}_{slug(configuration)}_"
                        f"fold{fold}_seed{seed}.npz"
                    )
                    for mode in PERTURBATIONS
                ]
                if key in completed and all(
                    path.exists() for path in [*expected_files, *expected_robustness_files]
                ):
                    continue
                selected_dimension = widths[(fold, seed)]
                if configuration.endswith("validation-selected PCA"):
                    matrix = matrices["pca"][:, :selected_dimension]
                    source_configuration = f"mBERT+Qwen PCA-{selected_dimension}"
                else:
                    matrix = matrices[configuration]
                    source_configuration = configuration

                seed42_state = args.seed42_model_dir / (
                    f"fold_{fold}_{slug(configuration)}_trainable_state.pt"
                )
                if seed == 42 and seed42_state.exists():
                    state = torch.load(seed42_state, map_location="cpu", weights_only=True)
                    validation_metrics: dict[str, float] = {}
                    costs: dict[str, float] = {}
                else:
                    physical = 64 if matrix.shape[1] > 4096 else 256
                    state, _, validation_metrics, costs = train_extended(
                        matrix,
                        development_tokens[train], labels[train],
                        development_tokens[validation], labels[validation],
                        seed, physical, 1024, args.max_epochs,
                    )

                internal_logits, internal_ms = infer_state(
                    matrix, state, development_tokens[test],
                    128 if matrix.shape[1] > 4096 else 512,
                )
                internal_metrics = evaluate(labels[test], internal_logits)
                expected = static_runs[
                    (static_runs["fold"] == fold)
                    & (static_runs["training_seed"] == seed)
                    & (static_runs["configuration"] == source_configuration)
                ]
                if len(expected) != 1 or abs(internal_metrics["mcc"] - expected.iloc[0]["mcc"]) > 1e-7:
                    raise RuntimeError(
                        f"Internal reconstruction mismatch: fold={fold}, seed={seed}, "
                        f"configuration={configuration}"
                    )

                base = {
                    "fold": fold,
                    "training_seed": seed,
                    "configuration": configuration,
                    "source_configuration": source_configuration,
                    "selected_dimension": selected_dimension if configuration.endswith("PCA") else matrix.shape[1],
                    "internal_test_mcc_reproduced": internal_metrics["mcc"],
                    "internal_inference_ms_per_message": internal_ms,
                    **{f"validation_{name}": value for name, value in validation_metrics.items()},
                    **costs,
                }
                for dataset_name, frame in external_frames.items():
                    logits, inference_ms = infer_state(
                        matrix, state, external_tokens[dataset_name],
                        128 if matrix.shape[1] > 4096 else 512,
                    )
                    probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                    np.savez_compressed(
                        probability_path(
                            args.output_dir, dataset_name, configuration, fold, seed
                        ),
                        probability=probability.astype(np.float16),
                    )
                    rows.append({
                        **base,
                        "dataset": dataset_name,
                        "external_inference_ms_per_message": inference_ms,
                        **metrics_with_confusion(frame["label"].to_numpy(), probability),
                    })
                for mode in PERTURBATIONS:
                    logits, inference_ms = infer_state(
                        matrix, state, perturbed_tokens[mode],
                        128 if matrix.shape[1] > 4096 else 512,
                    )
                    probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                    np.savez_compressed(
                        args.output_dir / "predictions" / (
                            f"robustness_{slug(mode)}_{slug(configuration)}_"
                            f"fold{fold}_seed{seed}.npz"
                        ),
                        probability=probability.astype(np.float16),
                    )
                    robustness_rows.append({
                        "fold": fold,
                        "training_seed": seed,
                        "configuration": configuration,
                        "perturbation": mode,
                        "selected_dimension": base["selected_dimension"],
                        "inference_ms_per_message": inference_ms,
                        **metrics_with_confusion(labels[test], probability),
                    })
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                pd.DataFrame(robustness_rows).to_csv(robustness_checkpoint, index=False)
                completed.add(key)
                print(
                    f"Fold {fold} seed {seed}: {configuration} externally evaluated",
                    flush=True,
                )
                del state, internal_logits
                gc.collect()
                torch.cuda.empty_cache()

        completed_sparse = {
            (int(row["fold"]), str(row["perturbation"]))
            for row in robustness_rows
            if row["configuration"] == "Character TF-IDF + logistic regression"
        }
        if any((fold, mode) not in completed_sparse for mode in PERTURBATIONS):
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=75_000, sublinear_tf=True,
            )
            x_train = vectorizer.fit_transform(development.iloc[train]["clean_text"])
            sparse = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=500,
                random_state=42, solver="liblinear",
            ).fit(x_train, labels[train])
            for mode in PERTURBATIONS:
                if (fold, mode) in completed_sparse:
                    continue
                probability = sparse.predict_proba(
                    vectorizer.transform(perturbed_texts[mode])
                )[:, 1]
                robustness_rows.append({
                    "fold": fold,
                    "training_seed": 42,
                    "configuration": "Character TF-IDF + logistic regression",
                    "perturbation": mode,
                    "selected_dimension": len(vectorizer.vocabulary_),
                    "inference_ms_per_message": np.nan,
                    **metrics_with_confusion(labels[test], probability),
                })
            pd.DataFrame(robustness_rows).to_csv(robustness_checkpoint, index=False)
            del vectorizer, sparse, x_train
            gc.collect()

    ensemble_rows = []
    bootstrap_rows = []
    for dataset_name, frame in external_frames.items():
        probabilities: dict[str, np.ndarray] = {}
        for configuration in CONFIGURATIONS:
            values = []
            for fold in args.folds:
                for seed in args.seeds:
                    with np.load(probability_path(
                        args.output_dir, dataset_name, configuration, fold, seed
                    )) as saved:
                        values.append(saved["probability"].astype(np.float32))
            probabilities[configuration] = np.mean(values, axis=0)
            ensemble_rows.append({
                "dataset": dataset_name,
                "configuration": configuration,
                "models": len(values),
                **metrics_with_confusion(
                    frame["label"].to_numpy(), probabilities[configuration]
                ),
            })

        sparse_files = sorted(args.seed42_model_dir.glob(
            f"predictions_{slug(dataset_name)}_"
            f"{slug('Character TF-IDF + logistic regression')}_fold*.csv"
        ))
        if len(sparse_files) == len(args.folds):
            probabilities["Character TF-IDF + logistic regression"] = np.mean(
                [pd.read_csv(path)["probability"].to_numpy() for path in sparse_files], axis=0
            )
            ensemble_rows.append({
                "dataset": dataset_name,
                "configuration": "Character TF-IDF + logistic regression",
                "models": len(sparse_files),
                **metrics_with_confusion(
                    frame["label"].to_numpy(),
                    probabilities["Character TF-IDF + logistic regression"],
                ),
            })

        reference = "mBERT+Qwen validation-selected PCA"
        for comparison, values in probabilities.items():
            if comparison == reference:
                continue
            observed, low, high = bootstrap_difference(
                frame["label"].to_numpy(), probabilities[reference], values
            )
            bootstrap_rows.append({
                "dataset": dataset_name,
                "reference": reference,
                "comparison": comparison,
                "mcc_difference": observed,
                "ci_low": low,
                "ci_high": high,
            })

    pd.DataFrame(ensemble_rows).to_csv(
        args.output_dir / "external_multiseed_ensemble.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "external_multiseed_bootstrap.csv", index=False
    )
    robustness_frame = pd.DataFrame(robustness_rows).drop_duplicates(
        ["fold", "training_seed", "configuration", "perturbation"], keep="last"
    )
    metric_columns = [
        "accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc",
        "balanced_accuracy", "roc_auc", "brier", "ece_15",
    ]
    robustness_summary = robustness_frame.groupby(
        ["configuration", "perturbation"]
    )[metric_columns].agg(["mean", "std"])
    robustness_summary.columns = [
        f"{metric}_{stat}" for metric, stat in robustness_summary.columns
    ]
    robustness_summary.reset_index().to_csv(
        args.output_dir / "robustness_multiseed_summary.csv", index=False
    )
    print(pd.DataFrame(ensemble_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
