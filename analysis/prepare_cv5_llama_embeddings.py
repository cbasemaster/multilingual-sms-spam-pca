"""Cache mean input-token embeddings for the union of five training vocabularies."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


MODEL_ID = "meta-llama/Llama-2-7b"
REPRESENTATION = "mean of raw input token embeddings over native SentencePiece subtokens"
MAX_SUBTOKENS = 32


def load_fold_words(fold_dir: Path) -> dict[int, list[str]]:
    words_by_fold = {}
    for fold in range(1, 6):
        path = fold_dir / f"fold_{fold}" / "training_vocabulary.csv"
        words = pd.read_csv(path, keep_default_na=False, dtype=str)["word"].tolist()
        if len(words) != len(set(words)):
            raise ValueError(f"Duplicate words in fold {fold}")
        words_by_fold[fold] = words
    return words_by_fold


def union_words(words_by_fold: dict[int, list[str]]) -> list[str]:
    return list(dict.fromkeys(word for fold in sorted(words_by_fold)
                              for word in words_by_fold[fold]))


def vocabulary_hash(words: list[str]) -> str:
    payload = json.dumps(words, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_or_write_vocabulary(cache: Path, words: list[str]) -> str:
    digest = vocabulary_hash(words)
    path = cache / "union_words.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != words:
            raise ValueError("Existing Llama union vocabulary differs from current folds")
    else:
        path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return digest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_input_embeddings(table: torch.Tensor, token_ids: list[list[int]]) -> torch.Tensor:
    """Mean-pool table rows for a batch without loading transformer layers."""
    if not token_ids or any(not ids for ids in token_ids):
        raise ValueError("Every vocabulary word must produce at least one subtoken")
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
    indices = torch.tensor([index for ids in token_ids for index in ids], dtype=torch.long)
    if indices.min() < 0 or indices.max() >= table.shape[0]:
        raise ValueError("Tokenizer produced an ID outside the embedding table")
    vectors = table.index_select(0, indices).float()
    destinations = torch.repeat_interleave(torch.arange(len(token_ids)), lengths)
    pooled = torch.zeros((len(token_ids), table.shape[1]), dtype=torch.float32)
    pooled.index_add_(0, destinations, vectors)
    return pooled / lengths.unsqueeze(1)


def load_input_embedding_table(weights_path: Path) -> tuple[torch.Tensor, str]:
    """Read the one ZIP storage backing tok_embeddings.weight, not all 7B weights."""
    with FakeTensorMode():
        state = torch.load(
            weights_path, map_location="cpu", weights_only=True, mmap=True
        )
    if "tok_embeddings.weight" not in state:
        raise KeyError("Raw Llama checkpoint lacks tok_embeddings.weight")
    fake = state["tok_embeddings.weight"]
    if fake.storage_offset() != 0 or not fake.is_contiguous():
        raise ValueError("Input embedding table is not a contiguous storage")
    offset = getattr(fake.untyped_storage(), "_checkpoint_offset", None)
    if offset is None:
        raise ValueError("PyTorch did not expose the embedding storage offset")
    expected_bytes = fake.numel() * fake.element_size()
    if fake.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Unexpected raw embedding dtype: {fake.dtype}")

    with zipfile.ZipFile(weights_path) as archive, weights_path.open("rb") as stream:
        matches = []
        for entry in archive.infolist():
            stream.seek(entry.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid PyTorch checkpoint ZIP header")
            name_length, extra_length = struct.unpack("<HH", header[26:30])
            data_offset = entry.header_offset + 30 + name_length + extra_length
            if data_offset == offset:
                matches.append(entry)
        if len(matches) != 1 or matches[0].file_size != expected_bytes:
            raise ValueError("Cannot identify the exact input embedding storage")
        if matches[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError("Input embedding storage must be uncompressed")
        payload = archive.read(matches[0])
    if len(payload) != expected_bytes:
        raise ValueError("Incomplete input embedding storage")
    digest = hashlib.sha256(payload).hexdigest()
    buffer = np.frombuffer(payload, dtype="<u2").copy()
    table = torch.from_numpy(buffer).view(fake.dtype).reshape(fake.shape)
    return table, digest


@torch.inference_mode()
def extract(model_path: Path, words: list[str], output_path: Path,
            batch_size: int, vocab_sha256: str) -> dict:
    weights_path = model_path / "consolidated.00.pth"
    tokenizer_path = model_path / "tokenizer.model"
    for path in (weights_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download the approved raw {MODEL_ID} checkpoint "
                "and its tokenizer; no HF model conversion is required."
            )
    checkpoint_bytes = weights_path.stat().st_size
    tokenizer_sha256 = file_sha256(tokenizer_path)
    if output_path.exists():
        matrix = np.load(output_path, mmap_mode="r")
        metadata_path = output_path.with_suffix(".json")
        if not metadata_path.exists():
            raise ValueError("Cached Llama matrix has no provenance metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if matrix.shape != (len(words) + 1, 4096):
            raise ValueError(f"Cached Llama matrix has unexpected shape: {matrix.shape}")
        if (metadata.get("vocabulary_sha256") != vocab_sha256
                or metadata.get("representation") != REPRESENTATION
                or metadata.get("checkpoint_bytes") != checkpoint_bytes
                or metadata.get("tokenizer_sha256") != tokenizer_sha256):
            raise ValueError("Cached Llama matrix has a different vocabulary or source")
        return metadata

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    table, embedding_sha256 = load_input_embedding_table(weights_path)
    if table.ndim != 2 or table.shape != (tokenizer.get_piece_size(), 4096):
        raise ValueError(
            f"Embedding table {tuple(table.shape)} does not match tokenizer "
            f"size {tokenizer.get_piece_size()} and Llama-2-7B width 4096"
        )

    partial = output_path.with_suffix(".partial.npy")
    checkpoint = output_path.with_suffix(".checkpoint.json")
    if partial.exists() != checkpoint.exists():
        raise ValueError("Incomplete Llama extraction checkpoint; inspect it before resuming")
    if partial.exists():
        matrix = np.lib.format.open_memmap(partial, mode="r+")
        status = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (status.get("vocabulary_sha256") != vocab_sha256
                or status.get("representation") != REPRESENTATION
                or status.get("checkpoint_bytes") != checkpoint_bytes
                or status.get("tokenizer_sha256") != tokenizer_sha256
                or status.get("embedding_sha256") != embedding_sha256):
            raise ValueError("Llama extraction checkpoint uses a different input")
        if matrix.shape != (len(words) + 1, 4096) or matrix.dtype != np.float16:
            raise ValueError("Llama extraction checkpoint has wrong matrix format")
        offset = int(status["offset"])
        if not 0 <= offset <= len(words):
            raise ValueError("Llama extraction checkpoint has invalid offset")
        truncated = int(status["truncated_words"])
    else:
        matrix = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16,
            shape=(len(words) + 1, 4096),
        )
        matrix[0] = 0
        offset = truncated = 0

    started = time.perf_counter()
    for start in range(offset, len(words), batch_size):
        batch = words[start:start + batch_size]
        encoded = [tokenizer.encode(word, out_type=int) for word in batch]
        truncated += sum(len(ids) > MAX_SUBTOKENS for ids in encoded)
        values = pool_input_embeddings(
            table, [ids[:MAX_SUBTOKENS] for ids in encoded]
        ).to(torch.float16).numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite Llama embedding in vocabulary batch {start}")
        stop = start + len(batch)
        matrix[start + 1:stop + 1] = values
        if stop % (batch_size * 20) == 0 or stop == len(words):
            matrix.flush()
            checkpoint.write_text(json.dumps({
                "offset": stop,
                "truncated_words": truncated,
                "vocabulary_sha256": vocab_sha256,
                "representation": REPRESENTATION,
                "checkpoint_bytes": checkpoint_bytes,
                "tokenizer_sha256": tokenizer_sha256,
                "embedding_sha256": embedding_sha256,
            }), encoding="utf-8")
            print(f"Llama input embeddings: {stop}/{len(words)}", flush=True)

    elapsed = time.perf_counter() - started
    matrix.flush()
    del matrix, table
    partial.replace(output_path)
    checkpoint.unlink()
    metadata = {
        "model": MODEL_ID,
        "checkpoint_path": str(weights_path.resolve()),
        "checkpoint_bytes": checkpoint_bytes,
        "tokenizer_sha256": tokenizer_sha256,
        "embedding_sha256": embedding_sha256,
        "rows": len(words) + 1,
        "dimension": 4096,
        "vocabulary_sha256": vocab_sha256,
        "max_subtokens": MAX_SUBTOKENS,
        "representation": REPRESENTATION,
        "special_tokens": "none; each word encoded independently",
        "storage_dtype": "float16",
        "truncated_words": truncated,
        "extraction_seconds": elapsed,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    cache = args.fold_dir / "llama_cache"
    cache.mkdir(parents=True, exist_ok=True)
    words = union_words(load_fold_words(args.fold_dir))
    digest = verify_or_write_vocabulary(cache, words)
    print(json.dumps({"union_words": len(words), "vocabulary_sha256": digest}), flush=True)
    metadata = extract(
        args.model_path, words, cache / "llama_union.npy", args.batch_size, digest
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
