"""Small alignment and scaling checks for the triple-fusion cache."""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_cv5_triple_fusion import standardized_triple


class TripleFusionTests(unittest.TestCase):
    def test_standardizes_each_source_and_keeps_padding_zero(self):
        sources = (
            np.array([[0, 0], [1, 10], [2, 20], [3, 30]], dtype=np.float16),
            np.array([[0], [100], [200], [300]], dtype=np.float16),
            np.array([[0], [-2], [0], [2]], dtype=np.float16),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triple.npy"
            result = standardized_triple(sources, path, chunk_size=2)
            self.assertEqual(result.shape, (4, 4))
            np.testing.assert_array_equal(result[0], 0)
            np.testing.assert_allclose(result[1:].mean(axis=0), 0, atol=1e-3)
            np.testing.assert_allclose(result[1:].std(axis=0), 1, atol=1e-3)
            cached = standardized_triple(sources, path)
            np.testing.assert_array_equal(cached, result)
            del cached, result
            gc.collect()

    def test_rejects_misaligned_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "matching rows"):
                standardized_triple(
                    (np.zeros((3, 2)), np.zeros((4, 2))),
                    Path(directory) / "triple.npy",
                )


if __name__ == "__main__":
    unittest.main()
