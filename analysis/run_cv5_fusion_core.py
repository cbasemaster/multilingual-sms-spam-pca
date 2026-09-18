"""Evaluate primary static-fusion controls on one held-out source-group fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_embedding_fusion_experiment import (
    evaluate,
    fit_vocabulary,
    texts_to_padded,
)
from grouped_qwen_pca_sweep_experiment import (
    fit_pca_max,
    fit_random_projection_max,
    infer_state,
    make_standardized_concat,
    train_extended,
)
from grouped_mbert_qwen_char_tfidf_concat_pca_cnn import make_three_source_concat
from revision_audit import load_long_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--only-index", type=int)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    fold_dir = args.fold_dir / f"fold_{args.fold}"
    output_dir = fold_dir / "fusion_core"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_dataset(args.dataset)
    with np.load(args.fold_dir / f"fold_{args.fold}.npz") as parts:
        train, validation, test = (parts[key] for key in ("train", "validation", "test"))
    word_index, words = fit_vocabulary(data.iloc[train]["clean_text"])
    recorded_words = pd.read_csv(
        fold_dir / "training_vocabulary.csv", keep_default_na=False, dtype=str
    )["word"].tolist()
    if words != recorded_words:
        raise ValueError("Fold training vocabulary mismatch")
    tokens = texts_to_padded(data["clean_text"], word_index, 128)
    labels = data["label"].to_numpy(dtype=np.int64)
    mbert = np.load(fold_dir / "mbert.npy", mmap_mode="r")
    qwen = np.load(fold_dir / "qwen.npy", mmap_mode="r")
    if mbert.shape[0] != qwen.shape[0] or mbert.shape[0] != len(words) + 1:
        raise ValueError("Fold embedding matrices and vocabulary do not align")

    concat = make_standardized_concat(
        mbert, qwen, output_dir / "standardized_concat.npy"
    )
    pca_max, pca_details = fit_pca_max(
        concat, output_dir / "fusion_pca2048.npy", 2048
    )
    rp_max = fit_random_projection_max(
        concat, output_dir / "fusion_rp1024.npy", 1024
    )
    qwen_pca, qwen_pca_details = fit_pca_max(
        qwen, output_dir / "qwen_pca1024.npy", 1024
    )
    mbert_pca, _ = fit_pca_max(
        mbert, output_dir / "mbert_pca512.npy", 512
    )
    char_concat, _ = make_three_source_concat(
        concat, words, output_dir / "three_source_concat.npy",
        output_dir / "three_source_concat.json", 1024,
    )
    char_pca_max, char_pca_details = fit_pca_max(
        char_concat, output_dir / "three_source_pca2048.npy", 2048
    )
    distil = np.load(fold_dir / "distil.npy", mmap_mode="r")
    if distil.shape[0] != len(words) + 1:
        raise ValueError("Fold DistilBERT matrix and vocabulary do not align")
    distil_concat = make_standardized_concat(
        distil, mbert, output_dir / "distil_mbert_standardized_concat.npy"
    )
    raw_path = output_dir / "distil_mbert_raw_concat.npy"
    if not raw_path.exists():
        raw = np.lib.format.open_memmap(
            raw_path.with_suffix(".partial.npy"), mode="w+", dtype=np.float16,
            shape=(len(words) + 1, 1536),
        )
        raw[0] = 0
        for start in range(1, len(words) + 1, 2048):
            stop = min(start + 2048, len(words) + 1)
            raw[start:stop, :768] = distil[start:stop]
            raw[start:stop, 768:] = mbert[start:stop]
        raw.flush()
        del raw
        raw_path.with_suffix(".partial.npy").replace(raw_path)
    distil_raw = np.load(raw_path, mmap_mode="r")
    distil_pca, distil_pca_details = fit_pca_max(
        distil_concat, output_dir / "distil_mbert_pca1024.npy", 1024
    )
    distil_rp = fit_random_projection_max(
        distil_concat, output_dir / "distil_mbert_rp256.npy", 256
    )
    configs = {
        "mBERT": mbert,
        "DistilBERT": distil,
        "Qwen2.5 raw": qwen,
        "Qwen2.5+PCA-1024": qwen_pca,
        "mBERT+Qwen concat no PCA": concat,
        "mBERT+Qwen RP-1024": rp_max,
        **{f"mBERT+Qwen PCA-{width}": pca_max[:, :width]
           for width in (768, 1024, 1536, 2048)},
        "mBERT+Qwen+Char concat no PCA": char_concat,
        **{f"mBERT+Qwen+Char PCA-{width}": char_pca_max[:, :width]
           for width in (768, 1024, 1536, 2048)},
        "DistilBERT+mBERT raw concat": distil_raw,
        "DistilBERT+mBERT standardized concat": distil_concat,
        "DistilBERT+mBERT RP-256": distil_rp,
        **{f"DistilBERT+mBERT PCA-{width}": distil_pca[:, :width]
           for width in (128, 256, 1024)},
        "mBERT PCA-256": mbert_pca[:, :256],
        "mBERT PCA-512": mbert_pca,
    }
    expected_names = set(configs)
    end_index = args.end_index or len(configs)
    start_index = args.only_index or args.start_index
    if args.only_index:
        end_index = args.only_index
    if not 1 <= start_index <= end_index <= len(configs):
        raise ValueError(f"Configuration range outside 1..{len(configs)}")
    configs = {name: matrix for index, (name, matrix) in enumerate(configs.items(), 1)
               if start_index <= index <= end_index}
    runs_path = output_dir / "runs.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {row["configuration"] for row in runs}
    for name, matrix in configs.items():
        if name in completed:
            continue
        state, _, validation_metrics, costs = train_extended(
            matrix, tokens[train], labels[train], tokens[validation], labels[validation],
            args.training_seed,
            64 if name == "mBERT+Qwen+Char concat no PCA" else 256,
            1024, args.max_epochs,
        )
        logits, inference_ms = infer_state(matrix, state, tokens[test], 512)
        test_metrics = evaluate(labels[test], logits)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
        pd.DataFrame({
            "fold": args.fold,
            "configuration": name,
            "row_index": test,
            "group_id": data.iloc[test]["group_id"].to_numpy(),
            "label": labels[test],
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(np.int8),
        }).to_csv(output_dir / f"predictions_{slug}.csv", index=False)
        runs.append({
            "fold": args.fold,
            "training_seed": args.training_seed,
            "configuration": name,
            "input_dimension": (5376 if "mBERT+Qwen+Char" in name else
                                4352 if "mBERT+Qwen" in name else
                                1536 if "DistilBERT+mBERT" in name else
                                3584 if name.startswith("Qwen2.5") else
                                768 if name.startswith("mBERT PCA") else matrix.shape[1]),
            "final_dimension": matrix.shape[1],
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **test_metrics,
            **costs,
            "inference_ms_per_message": inference_ms,
        })
        pd.DataFrame(runs).to_csv(runs_path, index=False)
        print(f"Fold {args.fold}: {name}: test MCC {test_metrics['mcc']:.4f}", flush=True)

    frame = pd.DataFrame(runs)
    if set(frame["configuration"]) != expected_names:
        print(f"Fold {args.fold}: {len(frame)}/{len(expected_names)} complete", flush=True)
        return
    pca = frame[frame["configuration"].str.match(r"mBERT\+Qwen PCA-\d+$")].copy()
    pca["dimension"] = pca["final_dimension"].astype(int)
    selected = pca.sort_values(
        ["validation_mcc", "dimension"], ascending=[False, True]
    ).iloc[0]
    (output_dir / "metadata.json").write_text(json.dumps({
        "protocol": "stratified five-fold source-group cross-validation",
        "fold": args.fold,
        "training_seed": args.training_seed,
        "selection_rule": "highest validation MCC; smaller dimension breaks ties",
        "selected_dimension": int(selected["dimension"]),
        "pca_retained_variance_at_1024": sum(pca_details["eigenvalues"][:1024])
        / pca_details["total_variance"],
        "qwen_pca_retained_variance": sum(qwen_pca_details["eigenvalues"])
        / qwen_pca_details["total_variance"],
        "three_source_pca_retained_variance_at_1024":
        sum(char_pca_details["eigenvalues"][:1024])
        / char_pca_details["total_variance"],
        "distil_mbert_pca_retained_variance_at_1024":
        sum(distil_pca_details["eigenvalues"][:1024])
        / distil_pca_details["total_variance"],
        "no_train_test_group_overlap": True,
    }, indent=2), encoding="utf-8")
    print(f"Fold {args.fold}: validation selected PCA-{int(selected['dimension'])}", flush=True)


if __name__ == "__main__":
    main()
