"""Smoke tests for the AFGRL pipeline.

Requires faiss-cpu: pip install faiss-cpu
All tests in this file are skipped automatically if faiss is not available.
"""

import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed — skipping AFGRL tests")

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from graphssl.models import AFGRL  # noqa: E402
from helpers import make_batch as _make_batch  # noqa: E402


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True},
        "pred_hidden": 64,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": 0,
        "topk": 3,
        "num_centroids": 5,
        "num_kmeans": 2,
        "clus_num_iters": 5,
    }
    cfg.update(overrides)
    return cfg


def _make_graph(n_nodes=30, n_feat=7):
    # Denser graph (4x edges) needed for AFGRL's kNN mining to find valid positives.
    return Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.stack(
            [
                torch.randint(0, n_nodes, (n_nodes * 4,)),
                torch.randint(0, n_nodes, (n_nodes * 4,)),
            ]
        ),
    )


class TestAFGRL:
    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = AFGRL(self.config, in_channels=self.in_channels)
        assert model.online_enc is not None
        assert model.target_enc is not None
        assert model.online_pred is not None
        assert model.positive_miner is not None

    def test_teacher_is_frozen(self):
        model = AFGRL(self.config, in_channels=self.in_channels)
        for p in model.target_enc.parameters():
            assert not p.requires_grad

    def test_target_same_weights_as_online_at_init(self):
        """Unlike BGRL, AFGRL's target starts with identical weights (no reset)."""
        model = AFGRL(self.config, in_channels=self.in_channels)
        all_equal = all(
            torch.equal(po.data, pt.data)
            for po, pt in zip(
                model.online_enc.parameters(), model.target_enc.parameters(), strict=True
            )
        )
        assert all_equal, "AFGRL target encoder must start with same weights as online"

    def test_student_parameters_excludes_teacher(self):
        model = AFGRL(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        teacher_ids = {id(p) for p in model.target_enc.parameters()}
        assert student_ids.isdisjoint(teacher_ids)

    def test_forward_shape_graph_level(self):
        model = AFGRL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        emb = model(batch)
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_graph_level_finite(self):
        """Graph-level path: simple teacher-student loss, no PositiveMiner."""
        model = AFGRL(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_feat=self.in_channels)
        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_compute_loss_node_level_finite(self):
        """Node-level path: PositiveMiner mines pairs via local adjacency + k-means."""
        cfg = _make_config()
        cfg["encoder"]["pool"] = False
        model = AFGRL(cfg, in_channels=self.in_channels)
        graph = _make_graph(n_nodes=30, n_feat=self.in_channels)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_post_step_updates_teacher_after_student_change(self):
        """EMA must move target toward online when online params have changed."""
        model = AFGRL(self.config, in_channels=self.in_channels)
        # Simulate a gradient update on the online encoder
        with torch.no_grad():
            for p in model.online_enc.parameters():
                p.add_(torch.ones_like(p))
        before = {n: p.data.clone() for n, p in model.target_enc.named_parameters()}
        model.post_step()
        any_changed = any(
            not torch.equal(before[n], p.data) for n, p in model.target_enc.named_parameters()
        )
        assert any_changed, "post_step() must update target_enc via EMA"

    def test_invalid_topk_raises(self):
        bad = _make_config()
        bad["topk"] = 0
        with pytest.raises(ValueError, match="topk"):
            AFGRL(bad, in_channels=self.in_channels)

    def test_invalid_num_centroids_raises(self):
        bad = _make_config()
        bad["num_centroids"] = -1
        with pytest.raises(ValueError, match="num_centroids"):
            AFGRL(bad, in_channels=self.in_channels)
