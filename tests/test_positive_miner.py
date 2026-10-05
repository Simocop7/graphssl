"""AFGRL's chunked top-k neighbour search — runs without faiss (only k-means needs it)."""

import pytest
import torch
import torch.nn.functional as F

from graphssl.config.schema import AFGRLConfig
from graphssl.utils.positive_miner import topk_similar


def _embeddings(n=300, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    student = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    teacher = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    return student, teacher


class TestTopkSimilar:
    @pytest.mark.parametrize("chunk_size", [1, 7, 64, 299, 300, 1000])
    def test_chunked_matches_full_matrix(self, chunk_size):
        student, teacher = _embeddings()
        full = topk_similar(student, teacher, k=5, chunk_size=None)
        assert torch.equal(topk_similar(student, teacher, k=5, chunk_size=chunk_size), full)

    def test_matches_dense_reference(self):
        student, teacher = _embeddings()
        sim = student @ teacher.T + 10.0 * torch.eye(student.shape[0])
        expected = sim.topk(k=5, dim=1).indices
        assert torch.equal(topk_similar(student, teacher, k=5, chunk_size=32), expected)

    def test_every_node_is_its_own_top_neighbour(self):
        student, teacher = _embeddings()
        idx = topk_similar(student, teacher, k=3, chunk_size=50)
        assert idx.shape == (300, 3)
        assert torch.equal(idx[:, 0], torch.arange(300))

    def test_invalid_chunk_size_raises(self):
        student, teacher = _embeddings(n=10)
        with pytest.raises(ValueError, match="chunk_size"):
            topk_similar(student, teacher, k=2, chunk_size=0)


class TestAFGRLConfigChunk:
    def _cfg(self, **overrides):
        d = {"encoder": {"name": "gin", "hidden_dim": 16, "num_layers": 2, "pool": False}}
        d.update(overrides)
        return AFGRLConfig.from_dict(d)

    def test_default_chunk_size(self):
        assert self._cfg().knn_chunk_size == 4096

    def test_none_allowed(self):
        assert self._cfg(knn_chunk_size=None).knn_chunk_size is None

    def test_non_positive_raises(self):
        with pytest.raises(ValueError, match="knn_chunk_size"):
            self._cfg(knn_chunk_size=0)
