"""Evaluate every validation-tested PCA width on the fixed grouped test set.

The validation-selected seed-42 checkpoints are reused for test inference.
Other seeds are trained with the same early-stopping protocol. Existing
five-seed PCA-1024 results are imported without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grouped_embedding_fusion_experiment import evaluate, fit_vocabulary, texts_to_padded
from grouped_qwen_pca_sweep_experiment import infer_state, train_extended
from revision_audit import SEEDS, load_long_dataset, split_groups


SPLIT_SEED = 42
METRICS = (
    "accuracy",
    "spam_f1",
    "macro_f1",
    "mcc",
    "cohen_kappa",
    "roc_auc",
    "ece_15",
)


def existing_pca1024_rows(path: Path, configuration: str, family: str) -> list[dict]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["configuration"] == configuration].copy()
    if set(frame["seed"].astype(int)) != set(SEEDS):
        raise ValueError(f"Incomplete existing PCA-1024 results in {path}")
    frame.insert(0, "dimension", 1024)
    frame.insert(0, "family", family)
    return frame.to_dict("records")


def save_outputs(records: list[dict], output_dir: Path) -> None:
    runs = pd.DataFrame(records).sort_values(["family", "dimension", "seed"])
    runs.to_csv(output_dir / "all_pca_dimension_test_runs.csv", index=False)
    summary = runs.groupby(["family", "dimension"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "all_pca_dimension_test_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-vocabulary", type=Path, required=True)
    parser.add_argument("--qwen-pca-matrix", type=Path, required=True)
    parser.add_argument("--qwen-experiment-dir", type=Path, required=True)
    parser.add_argument("--char-pca-matrix", type=Path, required=True)
    parser.add_argument("--char-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[768, 1024, 1536, 2048])
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
        raise ValueError("Vocabulary order does not match the PCA matrices.")
    tokens = texts_to_padded(data["clean_text"], word_index, args.max_length)
    labels = data["label"].to_numpy(dtype=np.int64)

    run_path = args.output_dir / "all_pca_dimension_test_runs.csv"
    if run_path.exists():
        records = pd.read_csv(run_path).to_dict("records")
    else:
        records = existing_pca1024_rows(
            args.qwen_experiment_dir / "final_runs.csv",
            "mBERT+Qwen Concat+PCA-1024",
            "mBERT+Qwen",
        )
        records.extend(existing_pca1024_rows(
            args.char_experiment_dir / "final_runs.csv",
            "mBERT+Qwen+CharTFIDF Concat+PCA-1024",
            "mBERT+Qwen+CharTFIDF",
        ))
        save_outputs(records, args.output_dir)

    families = (
        {
            "name": "mBERT+Qwen",
            "matrix": args.qwen_pca_matrix,
            "experiment_dir": args.qwen_experiment_dir,
            "state_pattern": "selection_state_concat_pca_{dimension}.pt",
            "selection_file": "dimension_selection.csv",
        },
        {
            "name": "mBERT+Qwen+CharTFIDF",
            "matrix": args.char_pca_matrix,
            "experiment_dir": args.char_experiment_dir,
            "state_pattern": "selection_state_pca{dimension}.pt",
            "selection_file": "dimension_selection.csv",
        },
    )

    completed = {
        (str(record["family"]), int(record["dimension"]), int(record["seed"]))
        for record in records
    }
    for family in families:
        full_matrix = np.load(family["matrix"], mmap_mode="r")
        selection = pd.read_csv(family["experiment_dir"] / family["selection_file"])
        for dimension in args.dims:
            matrix = full_matrix[:, :dimension]
            for seed in SEEDS:
                key = (family["name"], dimension, seed)
                if key in completed:
                    continue

                if seed == SPLIT_SEED:
                    state_path = family["experiment_dir"] / family["state_pattern"].format(
                        dimension=dimension
                    )
                    state = torch.load(state_path, map_location="cpu", weights_only=True)
                    selection_row = selection.loc[selection["dimension"] == dimension].iloc[0]
                    validation_metrics = {
                        metric: float(selection_row[f"validation_{metric}"])
                        for metric in evaluate(labels[validation], np.zeros(len(validation)))
                        if f"validation_{metric}" in selection_row
                    }
                    costs = {
                        name: selection_row[name]
                        for name in (
                            "epochs",
                            "physical_batch_size",
                            "effective_batch_size",
                            "best_validation_loss",
                            "train_seconds",
                            "peak_gpu_memory_mb",
                            "trainable_parameters",
                            "stored_embedding_mb",
                        )
                        if name in selection_row
                    }
                else:
                    state, _, validation_metrics, costs = train_extended(
                        matrix,
                        tokens[train],
                        labels[train],
                        tokens[validation],
                        labels[validation],
                        seed,
                        args.physical_batch_size,
                        args.effective_batch_size,
                        args.max_epochs,
                    )
                    torch.save(
                        state,
                        args.output_dir / f"state_{family['name'].replace('+', '_')}_pca{dimension}_seed{seed}.pt",
                    )

                logits, inference_ms = infer_state(
                    matrix, state, tokens[test], args.physical_batch_size * 2
                )
                test_metrics = evaluate(labels[test], logits)
                record = {
                    "family": family["name"],
                    "dimension": dimension,
                    "configuration": f"{family['name']} Concat+PCA-{dimension}",
                    "seed": seed,
                    **{f"validation_{name}": value for name, value in validation_metrics.items()},
                    **test_metrics,
                    **costs,
                    "inference_ms_per_message": inference_ms,
                }
                records.append(record)
                completed.add(key)
                probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                pd.DataFrame({
                    "sample_index": test,
                    "group_id": data.iloc[test]["group_id"].to_numpy(),
                    "label": labels[test],
                    "probability": probabilities,
                    "prediction": (probabilities >= 0.5).astype(np.int8),
                }).to_csv(
                    args.output_dir
                    / f"prediction_{family['name'].replace('+', '_')}_pca{dimension}_seed{seed}.csv",
                    index=False,
                )
                save_outputs(records, args.output_dir)
                print(
                    f"completed family={family['name']} dimension={dimension} seed={seed} "
                    f"test_mcc={test_metrics['mcc']:.6f}",
                    flush=True,
                )

    save_outputs(records, args.output_dir)
    print(pd.read_csv(args.output_dir / "all_pca_dimension_test_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
