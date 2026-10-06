"""Check aligned fusion and equivalence of trained/deployed MoE."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from grouped_embedding_fusion_experiment import FrozenEmbeddingCNN
from run_revision_gap_experiments import GatedEmbeddingCNN, aligned_sources, fuse_table


class FusionTests(unittest.TestCase):
    def test_unique_lookup_matches_dense_outputs_and_gate_gradients(self):
        rng = np.random.default_rng(12)
        aligned = rng.normal(size=(8, 12)).astype(np.float32)
        aligned[0] = 0
        model = GatedEmbeddingCNN(aligned, 2).eval()
        tokens = torch.tensor([[0, 1, 2, 1, 4], [2, 2, 3, 0, 1]])
        optimized = model(tokens)
        optimized.sum().backward()
        gradient = model.gate_weight.grad.clone()
        model.zero_grad(set_to_none=True)
        values = model.embedding(tokens).reshape(*tokens.shape, 2, 6)
        scores = (values * model.gate_weight).sum(dim=-1) + model.gate_bias
        fused = (values * scores.softmax(dim=-1).unsqueeze(-1)).sum(dim=-2)
        features = torch.relu(model.conv(fused.permute(0, 2, 1))).amax(dim=-1)
        dense = model.fc2(torch.relu(model.fc1(features))).squeeze(-1)
        dense.sum().backward()
        torch.testing.assert_close(optimized, dense, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(gradient, model.gate_weight.grad, atol=1e-6, rtol=1e-5)

    def test_alignment_and_mean(self):
        sources = [np.array([[0., 0.], [1., 3.]], dtype=np.float32),
                   np.array([[0., 0., 0.], [3., 1., 99.]], dtype=np.float32)]
        with TemporaryDirectory() as folder:
            path = Path(folder)
            aligned = aligned_sources(sources, path / "aligned.npy")
            self.assertEqual(aligned.shape, (2, 4))
            np.testing.assert_allclose(aligned[0], 0)
            np.testing.assert_allclose(aligned[1], [-1, 1, 1, -1], atol=1e-3)
            average = fuse_table(aligned, 2, path / "average.npy")
            np.testing.assert_allclose(average, 0, atol=1e-5)
            del average, aligned

    def test_input_dependent_gate_and_deployed_equivalence(self):
        rng = np.random.default_rng(2)
        aligned = rng.normal(size=(8, 12)).astype(np.float32)
        aligned[0] = 0
        model = GatedEmbeddingCNN(aligned, 2).eval()
        with torch.no_grad():
            model.gate_weight.copy_(torch.from_numpy(rng.normal(size=(2, 6)).astype(np.float32)))
            model.gate_bias.copy_(torch.tensor([.3, -.1]))
        tokens = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 0, 1]])
        with TemporaryDirectory() as folder:
            matrix = fuse_table(aligned, 2, Path(folder) / "moe.npy",
                                model.gate_weight.detach().numpy(), model.gate_bias.detach().numpy())
            deployed = FrozenEmbeddingCNN(matrix).eval()
            state = {k: v for k, v in model.state_dict().items()
                     if k != "embedding.weight" and not k.startswith("gate_")}
            deployed.load_state_dict(state, strict=False)
            with torch.inference_mode():
                torch.testing.assert_close(model(tokens), deployed(tokens), atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(matrix[0], 0)
            del deployed, matrix

    def test_rejects_misaligned_sources(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "align"):
                aligned_sources([np.zeros((3, 2)), np.zeros((4, 2))], Path(folder) / "a.npy")


if __name__ == "__main__":
    unittest.main()
