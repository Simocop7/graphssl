"""Smoke tests for the GraphCL pipeline."""

import pytest
import torch

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
