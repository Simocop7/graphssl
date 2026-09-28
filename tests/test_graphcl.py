"""Smoke tests for the GraphCL pipeline."""

import pytest
import torch

from graphssl.losses import NTXentLoss
from graphssl.models import GraphCL
from helpers import make_batch as _make_batch
from helpers import make_graph as _make_graph

_AUG = [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}]


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True},
        "augment": _AUG,
        "proj_dim": 64,
        "tau": 0.5,
    }
    cfg.update(overrides)
    return cfg


class TestGraphCL:
    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        assert model.encoder is not None
        assert model.projector is not None

    def test_projector_output_dim(self):
        """Projector must map hidden_dim → proj_dim."""
        model = GraphCL(self.config, in_channels=self.in_channels)
        x = torch.randn(4, self.config["encoder"]["hidden_dim"])
        out = model.projector(x)
        assert out.shape == (4, self.config["proj_dim"])

    def test_forward_shape(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        emb = model(batch)
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_compute_loss_non_negative(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.item() >= 0.0

    def test_student_parameters_all_params(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        assert student_ids == all_ids

    def test_invalid_tau_raises(self):
        bad = _make_config()
        bad["tau"] = -0.1
        with pytest.raises(ValueError, match="tau"):
            GraphCL(bad, in_channels=self.in_channels)

    def test_invalid_proj_dim_raises(self):
        bad = _make_config()
        bad["proj_dim"] = 0
        with pytest.raises(ValueError, match="proj_dim"):
            GraphCL(bad, in_channels=self.in_channels)

    def test_invalid_loss_chunk_size_raises(self):
        bad = _make_config(loss_chunk_size=0)
        with pytest.raises(ValueError, match="loss_chunk_size"):
            GraphCL(bad, in_channels=self.in_channels)

    def test_loss_chunk_size_forwarded(self):
        model = GraphCL(_make_config(loss_chunk_size=16), in_channels=self.in_channels)
        assert model.loss_fn.chunk_size == 16


class TestNTXentChunked:
    """Chunked NT-Xent must be exact: same value and gradients as the full [2N, 2N] version."""

    @pytest.mark.parametrize("chunk_size", [1, 7, 16, 39])
    def test_chunked_matches_full(self, chunk_size):
        torch.manual_seed(0)
        z1 = torch.randn(20, 8, requires_grad=True)
        z2 = torch.randn(20, 8, requires_grad=True)

        full = NTXentLoss(tau=0.5, chunk_size=None)(z1, z2)
        g1_full, g2_full = torch.autograd.grad(full, (z1, z2))

        chunked = NTXentLoss(tau=0.5, chunk_size=chunk_size)(z1, z2)
        g1_chunk, g2_chunk = torch.autograd.grad(chunked, (z1, z2))

        torch.testing.assert_close(chunked, full)
        torch.testing.assert_close(g1_chunk, g1_full)
        torch.testing.assert_close(g2_chunk, g2_full)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            NTXentLoss(chunk_size=0)


class TestGraphCLNodeLevel:
    def setup_method(self):
        cfg = _make_config()
        cfg["encoder"]["pool"] = False
        self.config = cfg
        self.in_channels = 7

    def test_forward_shape(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        emb = model(graph)
        assert emb.shape == (20, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = GraphCL(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
