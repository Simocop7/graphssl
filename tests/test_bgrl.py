"""Smoke tests for the BGRL pipeline."""

import pytest
import torch

from graphssl.models import BGRL
from helpers import make_batch as _make_batch
from helpers import make_graph as _make_graph

_AUG = [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}]


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True},
        "augment": _AUG,
        "pred_hidden": 64,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": 0,
    }
    cfg.update(overrides)
    return cfg


class TestBGRL:
    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        assert model.online_enc is not None
        assert model.target_enc is not None
        assert model.online_pred is not None

    def test_teacher_is_frozen(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        for p in model.target_enc.parameters():
            assert not p.requires_grad

    def test_target_reset_differs_from_online(self):
        """target_enc is deepcopied then reset_parameters() is called → weights differ."""
        model = BGRL(self.config, in_channels=self.in_channels)
        any_different = any(
            not torch.equal(po.data, pt.data)
            for po, pt in zip(
                model.online_enc.parameters(), model.target_enc.parameters(), strict=True
            )
        )
        assert any_different, "Target encoder must have different weights after reset"

    def test_forward_shape(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        emb = model(batch)
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_student_parameters_excludes_teacher(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        teacher_ids = {id(p) for p in model.target_enc.parameters()}
        assert student_ids.isdisjoint(teacher_ids)

    def test_post_step_updates_teacher(self):
        """EMA update must change target_enc weights (online and target start different)."""
        model = BGRL(self.config, in_channels=self.in_channels)
        before = {n: p.data.clone() for n, p in model.target_enc.named_parameters()}
        model.post_step()
        any_changed = any(
            not torch.equal(before[n], p.data) for n, p in model.target_enc.named_parameters()
        )
        assert any_changed, "post_step() must update target_enc via EMA"

    def test_invalid_ema_tau_raises(self):
        bad = _make_config()
        bad["ema_tau"] = 1.5
        with pytest.raises(ValueError, match="ema_tau"):
            BGRL(bad, in_channels=self.in_channels)

    def test_invalid_hidden_dim_raises(self):
        bad = _make_config()
        bad["encoder"]["hidden_dim"] = -1
        with pytest.raises(ValueError, match="hidden_dim"):
            BGRL(bad, in_channels=self.in_channels)


class TestBGRLNodeLevel:
    def setup_method(self):
        cfg = _make_config()
        cfg["encoder"]["pool"] = False
        self.config = cfg
        self.in_channels = 7

    def test_forward_shape(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        emb = model(graph)
        assert emb.shape == (20, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = BGRL(self.config, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestBGRLEMAScheduler:
    def test_tau_increases_over_steps(self):
        """CosineEMAScheduler must produce monotonically increasing tau."""
        cfg = _make_config()
        cfg["ema_tau"] = 0.9
        cfg["ema_tau_end"] = 1.0
        cfg["total_steps"] = 100
        model = BGRL(cfg, in_channels=7)
        assert model._ema_scheduler is not None
        tau_0 = model._ema_scheduler.get(0)
        tau_50 = model._ema_scheduler.get(50)
        tau_100 = model._ema_scheduler.get(100)
        assert tau_0 <= tau_50 <= tau_100

    def test_no_scheduler_when_total_steps_zero(self):
        model = BGRL(_make_config(), in_channels=7)
        assert model._ema_scheduler is None
        assert model._ema_tau == pytest.approx(0.99)

    def test_step_counter_increments(self):
        model = BGRL(_make_config(), in_channels=7)
        assert model._step == 0
        model.post_step()
        assert model._step == 1
        model.post_step()
        assert model._step == 2
