"""Create shared, disjoint source-group test folds for the revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from revision_audit import load_long_dataset


def make_folds(data, seed: int = 42):
    groups = data[["group_id", "label"]].drop_duplicates("group_id").reset_index(drop=True)
    by_group = data.groupby("group_id").indices
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    seen_test_groups = set()
    for fold, (trainval_group_rows, test_group_rows) in enumerate(
        outer.split(groups["group_id"], groups["label"]), start=1
    ):
        trainval_groups = groups.iloc[trainval_group_rows]
        test_groups = groups.iloc[test_group_rows]
        train_groups, validation_groups = train_test_split(
            trainval_groups,
            test_size=0.20,
            stratify=trainval_groups["label"],
            random_state=seed + fold,
        )
        ids = {
            "train": set(train_groups["group_id"].astype(int)),
            "validation": set(validation_groups["group_id"].astype(int)),
            "test": set(test_groups["group_id"].astype(int)),
        }
        assert not (ids["train"] & ids["validation"])
        assert not (ids["train"] & ids["test"])
        assert not (ids["validation"] & ids["test"])
        assert not (seen_test_groups & ids["test"])
        seen_test_groups.update(ids["test"])

        def expand(group_ids):
            return np.concatenate([by_group[int(group_id)] for group_id in group_ids])

        indices = {
            name: expand(sorted(group_ids)) for name, group_ids in ids.items()
        }
        yield fold, indices, {name: len(group_ids) for name, group_ids in ids.items()}
    assert seen_test_groups == set(groups["group_id"].astype(int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_long_dataset(args.dataset)
    records = []
    for fold, indices, group_counts in make_folds(data):
        path = args.output_dir / f"fold_{fold}.npz"
        np.savez_compressed(path, **indices)
        records.append({
            "fold": fold,
            "seed": 42,
            "train_samples": len(indices["train"]),
            "validation_samples": len(indices["validation"]),
            "test_samples": len(indices["test"]),
            **{f"{name}_groups": count for name, count in group_counts.items()},
        })
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"protocol": "stratified 5-fold by source-message group",
                    "dataset_samples": len(data), "folds": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
