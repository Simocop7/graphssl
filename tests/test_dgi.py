"""Smoke tests for the DGI pipeline."""

import sys, os
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graphssl.models import DGI
from helpers import make_batch as _make_batch, make_graph as _make_graph


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": False},
        "corruption": "shuffle_nodes",
        "shuffle_ratio": 1.0,
    }
    cfg.update(overrides)
    return cfg


class TestDGI:

    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = DGI(self.config, in_channels=self.in_channels)
        assert model.encoder is not None
        assert hasattr(model, "W")

    def test_W_is_learnable_parameter(self):
        model = DGI(self.config, in_channels=self.in_channels)
        assert isinstance(model.W, torch.nn.Parameter)
        assert model.W.requires_grad
        assert model.W.shape == (32, 32)

    def test_forward_shape(self):
        model = DGI(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_feat=self.in_channels)
        emb = model(graph)
        assert emb.shape == (20, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = DGI(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_feat=self.in_channels)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_student_parameters_includes_W(self):
        """student_parameters() must include all params, including the discriminator W."""
        model = DGI(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        assert id(model.W) in student_ids


class TestDGIGraphLevel:

    def setup_method(self):
        cfg = _make_config()
        cfg["encoder"]["pool"] = True
        self.config = cfg
        self.in_channels = 7

    def test_compute_loss_graph_level_finite(self):
        model = DGI(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestDGICorruption:

    def test_shuffle_nodes_mode(self):
        cfg = _make_config(corruption="shuffle_nodes")
        model = DGI(cfg, in_channels=7)
        graph = _make_graph()
        loss = model.compute_loss(graph)
        assert torch.isfinite(loss)

    def test_shuffle_edges_mode(self):
        cfg = _make_config(corruption="shuffle_edges")
        model = DGI(cfg, in_channels=7)
        graph = _make_graph()
        loss = model.compute_loss(graph)
        assert torch.isfinite(loss)

    def test_invalid_corruption_raises(self):
        bad = _make_config()
        bad["corruption"] = "random_walk"
        with pytest.raises(ValueError, match="corruption"):
            DGI(bad, in_channels=7)

    def test_invalid_shuffle_ratio_raises(self):
        bad = _make_config()
        bad["shuffle_ratio"] = 0.0
        with pytest.raises(ValueError, match="shuffle_ratio"):
            DGI(bad, in_channels=7)

    def test_partial_shuffle_ratio(self):
        """shuffle_ratio < 1.0 only corrupts a subset of nodes/edges."""
        cfg = _make_config(shuffle_ratio=0.3)
        model = DGI(cfg, in_channels=7)
        graph = _make_graph()
        loss = model.compute_loss(graph)
        assert torch.isfinite(loss)
