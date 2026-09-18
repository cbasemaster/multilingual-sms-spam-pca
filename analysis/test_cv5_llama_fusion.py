"""Focused checks for shared Llama vocabulary alignment."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sentencepiece as spm
import torch

from prepare_cv5_llama_embeddings import (
    REPRESENTATION,
    extract,
    load_input_embedding_table,
    pool_input_embeddings,
    union_words,
    vocabulary_hash,
)
from run_cv5_llama_fusion import FoldMatrix, materialize


class LlamaFoldMappingTests(unittest.TestCase):
    def test_input_token_means_and_invalid_ids(self):
        table = torch.tensor([[0., 0.], [2., 4.], [6., 8.], [10., 12.]])
        pooled = pool_input_embeddings(table, [[1], [1, 2, 3]])
        torch.testing.assert_close(pooled, torch.tensor([[2., 4.], [6., 8.]]))
        with self.assertRaisesRegex(ValueError, "at least one subtoken"):
            pool_input_embeddings(table, [[]])
        with self.assertRaisesRegex(ValueError, "outside the embedding table"):
            pool_input_embeddings(table, [[4]])

    def test_extracts_raw_table_and_reuses_verified_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            corpus = root / "corpus.txt"
            corpus.write_text("hello world\nspam message\nhello spam\n", encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(model / "tokenizer"),
                vocab_size=32, hard_vocab_limit=False, model_type="bpe",
                minloglevel=2,
            )
            tokenizer = spm.SentencePieceProcessor(
                model_file=str(model / "tokenizer.model")
            )
            width = 4096
            table = torch.arange(tokenizer.get_piece_size(), dtype=torch.float32)
            table = table.to(torch.bfloat16)
            table = table[:, None].expand(-1, width).contiguous()
            torch.save({"tok_embeddings.weight": table}, model / "consolidated.00.pth")
            selected, digest = load_input_embedding_table(model / "consolidated.00.pth")
            torch.testing.assert_close(selected, table)
            self.assertEqual(len(digest), 64)
            words = ["hello", "spam"]
            digest = vocabulary_hash(words)
            output = root / "llama_union.npy"
            metadata = extract(model, words, output, 1, digest)
            self.assertEqual(metadata["representation"], REPRESENTATION)
            actual = np.load(output)
            self.assertEqual(actual.shape, (3, width))
            self.assertTrue(np.all(actual[0] == 0))
            for row, word in enumerate(words, 1):
                ids = tokenizer.encode(word, out_type=int)
                self.assertAlmostEqual(float(actual[row, 0]), np.mean(ids), places=2)
            self.assertEqual(extract(model, words, output, 1, digest), metadata)
            with self.assertRaisesRegex(ValueError, "different vocabulary or source"):
                extract(model, words, output, 1, "wrong digest")

    def test_union_preserves_first_seen_fold_order(self):
        folds = {1: ["alpha", "beta"], 2: ["beta", "gamma"],
                 3: ["alpha"], 4: ["delta"], 5: ["gamma"]}
        self.assertEqual(union_words(folds), ["alpha", "beta", "gamma", "delta"])
        self.assertNotEqual(vocabulary_hash(union_words(folds)),
                            vocabulary_hash(["beta", "alpha", "gamma", "delta"]))

    def test_fold_matrix_reorders_rows_and_preserves_padding(self):
        source = np.zeros((4, 4096), dtype=np.float16)
        source[1, 0] = 10
        source[2, 0] = 20
        source[3, 0] = 30
        matrix = FoldMatrix(source, ["alpha", "beta", "gamma"],
                            ["gamma", "alpha"])
        self.assertEqual(matrix.shape, (3, 4096))
        self.assertEqual(matrix.nbytes, 3 * 4096 * 2)
        np.testing.assert_array_equal(matrix[:3][:, 0], [0, 30, 10])
        np.testing.assert_array_equal(np.asarray(matrix)[:, 0], [0, 30, 10])

    def test_fusion_uses_aligned_fold_rows_and_zero_padding(self):
        source = np.zeros((4, 4096), dtype=np.float16)
        source[1:, 0] = [10, 20, 30]
        llama = FoldMatrix(source, ["alpha", "beta", "gamma"],
                           ["gamma", "alpha"])
        qwen = np.zeros((3, 3584), dtype=np.float16)
        qwen[1:, 0] = [3, 1]
        with TemporaryDirectory() as directory:
            fused, details = materialize(
                "Llama-2+Qwen concat no PCA", llama, qwen, Path(directory)
            )
            self.assertIsNone(details)
            self.assertEqual(fused.shape, (3, 7680))
            self.assertTrue(np.isfinite(fused).all())
            self.assertTrue(np.all(fused[0] == 0))
            self.assertGreater(fused[1, 0], fused[2, 0])
            self.assertGreater(fused[1, 4096], fused[2, 4096])
            fused._mmap.close()


if __name__ == "__main__":
    unittest.main()
