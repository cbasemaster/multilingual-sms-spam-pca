"""Audit and normalize external SMS datasets without redistributing source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from difflib import SequenceMatcher
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd

from revision_audit import clean_text, load_long_dataset


def read_exais(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".csv"):
                continue
            user = Path(member).stem
            with archive.open(member) as binary:
                reader = csv.reader(TextIOWrapper(binary, encoding="latin-1", newline=""))
                for line_number, row in enumerate(reader, start=1):
                    if len(row) < 8:
                        continue
                    label_text = row[6].strip().lower()
                    if label_text not in {"spam", "ham"}:
                        continue
                    message_fields = list(row[7:])
                    while message_fields and not message_fields[-1].strip():
                        message_fields.pop()
                    message = ",".join(message_fields).strip()
                    rows.append({
                        "dataset": "ExAIS_SMS",
                        "source_record": f"{user}:{line_number}",
                        "raw_text": message,
                        "label": int(label_text == "spam"),
                    })
    return pd.DataFrame(rows)


def read_turkish(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8")
    group = pd.to_numeric(frame["Group"], errors="coerce")
    valid = group.isin([1, 2]) & frame["Message"].notna()
    return pd.DataFrame({
        "dataset": "TurkishSMSCollection",
        "source_record": [f"row:{index + 2}" for index in frame.index[valid]],
        "raw_text": frame.loc[valid, "Message"].astype(str).str.strip(),
        "label": (group.loc[valid] == 1).astype(np.int8),
    }).reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simhash(text: str) -> int:
    compact = f"  {text.replace(' ', '_')}  "
    features = {compact[index : index + 4] for index in range(max(1, len(compact) - 3))}
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
         for feature in features),
        dtype=np.uint64,
    )
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    weights = bits.sum(axis=0, dtype=np.int32) * 2 - len(features)
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def near_overlap_flags(reference_texts: list[str], query_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    reference_hashes: list[int] = []
    for index, text in enumerate(reference_texts):
        value = simhash(text)
        reference_hashes.append(value)
        for band in range(4):
            buckets[band][(value >> (16 * band)) & 0xFFFF].append(index)

    flags = np.zeros(len(query_texts), dtype=bool)
    scores = np.zeros(len(query_texts), dtype=np.float32)
    for query_index, text in enumerate(query_texts):
        value = simhash(text)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(buckets[band].get((value >> (16 * band)) & 0xFFFF, []))
        best = 0.0
        for candidate in candidates:
            if (value ^ reference_hashes[candidate]).bit_count() > 12:
                continue
            score = SequenceMatcher(None, text, reference_texts[candidate], autojunk=False).ratio()
            if score > best:
                best = score
        scores[query_index] = best
        flags[query_index] = best >= 0.90
    return flags, scores


def audit_dataset(frame: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.copy()
    original_class_counts = frame["label"].value_counts().sort_index().to_dict()
    frame["raw_text"] = frame["raw_text"].fillna("").astype(str).str.strip()
    frame["clean_text"] = frame["raw_text"].map(clean_text)
    frame["exact_key"] = frame["raw_text"].str.casefold().str.replace(r"\s+", " ", regex=True)

    raw_rows = len(frame)
    empty_rows = int(frame["clean_text"].eq("").sum())
    conflicting_keys = set(
        frame.groupby("exact_key")["label"].nunique().loc[lambda values: values > 1].index
    )
    conflicting_rows = int(frame["exact_key"].isin(conflicting_keys).sum())
    frame = frame[~frame["exact_key"].isin(conflicting_keys) & frame["clean_text"].ne("")].copy()
    duplicate_rows = int(frame.duplicated(["exact_key", "label"]).sum())
    frame = frame.drop_duplicates(["exact_key", "label"], keep="first").reset_index(drop=True)

    reference_raw = set(reference["raw_text"].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True))
    reference_clean = set(reference.loc[reference["clean_text"].ne(""), "clean_text"])
    frame["raw_overlap_with_development"] = frame["exact_key"].isin(reference_raw)
    frame["clean_overlap_with_development"] = frame["clean_text"].isin(reference_clean)

    unique_reference_clean = list(dict.fromkeys(reference.loc[reference["clean_text"].ne(""), "clean_text"]))
    near_flags, near_scores = near_overlap_flags(unique_reference_clean, frame["clean_text"].tolist())
    frame["near_overlap_with_development"] = near_flags & ~frame["clean_overlap_with_development"].to_numpy()
    frame["nearest_similarity"] = near_scores
    frame["eligible_primary"] = ~frame["clean_overlap_with_development"]
    frame["eligible_strict"] = frame["eligible_primary"] & ~frame["near_overlap_with_development"]

    primary = frame[frame["eligible_primary"]]
    strict = frame[frame["eligible_strict"]]
    summary = {
        "raw_records": raw_rows,
        "raw_class_counts": {str(key): int(value) for key, value in original_class_counts.items()},
        "empty_after_cleaning_records": empty_rows,
        "conflicting_label_records_removed": conflicting_rows,
        "exact_duplicate_records_removed": duplicate_rows,
        "unique_nonconflicting_records": len(frame),
        "raw_overlap_with_development": int(frame["raw_overlap_with_development"].sum()),
        "clean_overlap_with_development": int(frame["clean_overlap_with_development"].sum()),
        "additional_near_overlap_at_0_90": int(frame["near_overlap_with_development"].sum()),
        "primary_records": len(primary),
        "primary_ham": int((primary["label"] == 0).sum()),
        "primary_spam": int((primary["label"] == 1).sum()),
        "strict_records": len(strict),
        "strict_ham": int((strict["label"] == 0).sum()),
        "strict_spam": int((strict["label"] == 1).sum()),
    }
    return frame.drop(columns=["exact_key"]), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--exais", type=Path, required=True)
    parser.add_argument("--turkish", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_long_dataset(args.development)
    inputs = {
        "ExAIS_SMS": (read_exais(args.exais), args.exais),
        "TurkishSMSCollection": (read_turkish(args.turkish), args.turkish),
    }
    report: dict[str, object] = {
        "development_samples": len(reference),
        "near_overlap_definition": (
            "SimHash candidate retrieval followed by SequenceMatcher ratio >= 0.90 on the "
            "development cleaning function; exact cleaned-text matches are reported separately."
        ),
        "source_files": {},
        "datasets": {},
    }
    for name, (frame, source_path) in inputs.items():
        audited, summary = audit_dataset(frame, reference)
        audited.to_csv(args.output_dir / f"{name.lower()}_audited.csv", index=False)
        report["source_files"][name] = {
            "path": str(source_path),
            "sha256": sha256(source_path),
        }
        report["datasets"][name] = summary
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
