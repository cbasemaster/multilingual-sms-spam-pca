"""Normalize and audit social-media and email spam corpora."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd

from audit_external_validation_datasets import audit_dataset, sha256
from revision_audit import load_long_dataset


def read_youtube(path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".csv") or member.startswith("__MACOSX/"):
                continue
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, encoding="latin-1")
            rows.append(pd.DataFrame({
                "dataset": "YouTubeSpamCollection",
                "source_record": [f"{Path(member).name}:{value}" for value in frame["COMMENT_ID"]],
                "raw_text": frame["CONTENT"].astype(str),
                "label": frame["CLASS"].astype("int8"),
            }))
    return pd.concat(rows, ignore_index=True)


def decode_part(part) -> str:
    try:
        return part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True)
        if payload is None:
            return str(part.get_payload())
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")


def email_text(raw: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    pieces = [str(message.get("subject", ""))]
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "text" and not part.get_filename():
                pieces.append(decode_part(part))
    else:
        pieces.append(decode_part(message))
    return "\n".join(piece for piece in pieces if piece)


def read_spamassassin(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        label = int("spam" in path.stem and "ham" not in path.stem)
        with tarfile.open(path, mode="r:bz2") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                rows.append({
                    "dataset": "SpamAssassinPublicCorpus",
                    "source_record": f"{path.name}:{member.name}",
                    "raw_text": email_text(stream.read()),
                    "label": label,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--youtube", type=Path, required=True)
    parser.add_argument("--spamassassin", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_long_dataset(args.development)
    inputs = {
        "YouTubeSpamCollection": (read_youtube(args.youtube), [args.youtube]),
        "SpamAssassinPublicCorpus": (read_spamassassin(args.spamassassin), args.spamassassin),
    }
    report: dict[str, object] = {
        "development_samples": len(reference),
        "external_labels_used_for_training_or_selection": False,
        "near_overlap_definition": (
            "SimHash candidate retrieval followed by SequenceMatcher ratio >= 0.90 on the "
            "development cleaning function; exact cleaned-text matches are reported separately."
        ),
        "sources": {
            "YouTubeSpamCollection": {
                "url": "https://archive.ics.uci.edu/dataset/380/youtube+spam+collection",
                "doi": "10.24432/C58885",
            },
            "SpamAssassinPublicCorpus": {
                "url": "https://spamassassin.apache.org/old/publiccorpus/",
            },
        },
        "source_files": {},
        "datasets": {},
    }
    for name, (frame, source_paths) in inputs.items():
        audited, summary = audit_dataset(frame, reference)
        audited.to_csv(args.output_dir / f"{name.lower()}_audited.csv", index=False)
        report["source_files"][name] = [
            {"path": str(path), "sha256": sha256(path)} for path in source_paths
        ]
        report["datasets"][name] = summary
    (args.output_dir / "extra_dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
