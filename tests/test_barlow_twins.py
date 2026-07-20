"""Smoke tests for the Barlow Twins pipeline."""

import sys, os
import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graphssl.models import BarlowTwins
from helpers import make_batch as _make_batch, make_graph as _make_graph


_AUG = [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}]


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True},
        "augment": _AUG,
        "proj_dim": 64,
        "lambda_param": None,
    }
    cfg.update(overrides)
    return cfg


class TestBarlowTwins:

    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        assert model.encoder is not None
        assert model.projector is not None

    def test_projector_is_three_layers(self):
        """Barlow Twins uses a 3-layer projector, same as VICReg."""
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        linear_layers = [m for m in model.projector.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3

    def test_projector_output_dim(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        x = torch.randn(4, self.config["encoder"]["hidden_dim"])
        out = model.projector(x)
        assert out.shape == (4, self.config["proj_dim"])

    def test_lambda_defaults_to_one_over_proj_dim(self):
        """lambda_param=None must default to 1/proj_dim inside BarlowTwinsLoss."""
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        expected = 1.0 / self.config["proj_dim"]
        assert model.loss_fn.lambda_param == pytest.approx(expected)

    def test_custom_lambda(self):
        cfg = _make_config(lambda_param=0.005)
        model = BarlowTwins(cfg, in_channels=self.in_channels)
        assert model.loss_fn.lambda_param == pytest.approx(0.005)

    def test_forward_shape(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        emb = model(batch)
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_compute_loss_non_negative(self):
        """Barlow Twins loss is a sum of squared terms — always non-negative."""
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.item() >= 0.0

    def test_student_parameters_all_params(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        assert student_ids == all_ids

    def test_invalid_lambda_raises(self):
        bad = _make_config(lambda_param=-0.1)
        with pytest.raises(ValueError, match="lambda_param"):
            BarlowTwins(bad, in_channels=self.in_channels)


class TestBarlowTwinsNodeLevel:

    def setup_method(self):
        cfg = _make_config()
        cfg["encoder"]["pool"] = False
        self.config = cfg
        self.in_channels = 7

    def test_forward_shape(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        emb = model(graph)
        assert emb.shape == (20, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = BarlowTwins(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
