"""Audit split leakage and establish reproducible lightweight baselines.

This script intentionally does not reproduce the paper's Concat+PCA CNN. The
original classifier code and embedding matrices are not available locally.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split


SEEDS = (13, 21, 42, 87, 101)


def clean_text(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text).lower())
    value = regex.sub(r"<.*?>", " ", value)
    value = regex.sub(r"http\S+|www\.\S+", " ", value)
    value = regex.sub(r"@\w+|#\w+", " ", value)
    value = regex.sub(r"\b\w*\d\w*\b", " ", value)
    value = regex.sub(r"[^\p{L}\s]", " ", value)
    value = " ".join(word for word in value.split() if len(word) > 1)
    return value


def load_long_dataset(path: Path) -> pd.DataFrame:
    wide = pd.read_parquet(path).copy()
    wide["group_id"] = np.arange(len(wide), dtype=np.int64)
    text_columns = [column for column in wide.columns if column == "text" or column.startswith("text_")]
    long = wide.melt(
        id_vars=["group_id", "labels"],
        value_vars=text_columns,
        var_name="language_column",
        value_name="raw_text",
    )
    # The thesis code removes duplicate language rows before text cleaning.
    # The current public snapshot yields 105,443 rows; the thesis reports
    # 105,423, so the 20-row source-version discrepancy is retained explicitly.
    long = long.dropna(subset=["raw_text"]).drop_duplicates(["labels", "raw_text"]).copy()
    long["label"] = (long["labels"] != "ham").astype(np.int8)
    long["clean_text"] = long["raw_text"].map(clean_text)
    return long.reset_index(drop=True)


def split_rows(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(data))
    train_val, test = train_test_split(
        indices,
        test_size=0.20,
        stratify=data["label"],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.20,
        stratify=data.iloc[train_val]["label"],
        random_state=seed,
    )
    return train, validation, test


def split_groups(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = data[["group_id", "label"]].drop_duplicates("group_id")
    group_train_val, group_test = train_test_split(
        groups,
        test_size=0.20,
        stratify=groups["label"],
        random_state=seed,
    )
    group_train, group_validation = train_test_split(
        group_train_val,
        test_size=0.20,
        stratify=group_train_val["label"],
        random_state=seed,
    )
    by_group = data.groupby("group_id").indices

    def expand(group_ids: pd.Series) -> np.ndarray:
        return np.concatenate([by_group[int(group_id)] for group_id in group_ids])

    return expand(group_train["group_id"]), expand(group_validation["group_id"]), expand(group_test["group_id"])


def overlap_summary(data: pd.DataFrame, split: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, int]:
    train, validation, test = split
    group_sets = {
        "train": set(data.iloc[train]["group_id"]),
        "validation": set(data.iloc[validation]["group_id"]),
        "test": set(data.iloc[test]["group_id"]),
    }
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_groups": len(group_sets["train"]),
        "validation_groups": len(group_sets["validation"]),
        "test_groups": len(group_sets["test"]),
        "train_validation_overlap_groups": len(group_sets["train"] & group_sets["validation"]),
        "train_test_overlap_groups": len(group_sets["train"] & group_sets["test"]),
        "validation_test_overlap_groups": len(group_sets["validation"] & group_sets["test"]),
        "groups_present_in_all_splits": len(
            group_sets["train"] & group_sets["validation"] & group_sets["test"]
        ),
    }


def evaluate_split(
    data: pd.DataFrame,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    split_name: str,
    seed: int,
) -> dict[str, float | int | str]:
    train, _, test = split
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=75_000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(data.iloc[train]["clean_text"])
    x_test = vectorizer.transform(data.iloc[test]["clean_text"])
    y_train = data.iloc[train]["label"].to_numpy()
    y_test = data.iloc[test]["label"].to_numpy()
    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=500,
        random_state=seed,
        solver="liblinear",
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return {
        "split": split_name,
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "accuracy": accuracy_score(y_test, predictions),
        "spam_f1": f1_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "mcc": matthews_corrcoef(y_test, predictions),
        "cohen_kappa": cohen_kappa_score(y_test, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_long_dataset(args.dataset)
    data_summary = {
        "wide_source_rows": int(data["group_id"].nunique()),
        "long_rows_after_nonempty_cleaning": int(len(data)),
        "ham_rows": int((data["label"] == 0).sum()),
        "spam_rows": int((data["label"] == 1).sum()),
        "language_columns": int(data["language_column"].nunique()),
        "empty_after_cleaning_rows": int((data["clean_text"].str.len() == 0).sum()),
        "thesis_reported_long_rows": 105_423,
        "difference_from_thesis_report": int(len(data) - 105_423),
    }

    audits: list[dict[str, int | str]] = []
    results: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        row_split = split_rows(data, seed)
        group_split = split_groups(data, seed)
        audits.append({"split": "row", "seed": seed, **overlap_summary(data, row_split)})
        audits.append({"split": "group", "seed": seed, **overlap_summary(data, group_split)})
        results.append(evaluate_split(data, row_split, "row", seed))
        results.append(evaluate_split(data, group_split, "group", seed))

    audit_frame = pd.DataFrame(audits)
    result_frame = pd.DataFrame(results)
    metric_columns = ["accuracy", "spam_f1", "macro_f1", "weighted_f1", "mcc", "cohen_kappa"]
    aggregate = result_frame.groupby("split")[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()

    data_summary["clean_text_exact_duplicate_rows"] = int(data.duplicated(["clean_text", "label"]).sum())
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(data_summary, indent=2), encoding="utf-8"
    )
    audit_frame.to_csv(args.output_dir / "split_overlap_audit.csv", index=False)
    result_frame.to_csv(args.output_dir / "baseline_runs.csv", index=False)
    aggregate.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    print(json.dumps(data_summary, indent=2))
    print(audit_frame.groupby("split").mean(numeric_only=True).round(3))
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
