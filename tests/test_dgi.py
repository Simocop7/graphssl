"""Smoke tests for the DGI pipeline."""

import pytest
import torch
from torch_geometric.data import Data

import graphssl.models.dgi as dgi_module
from graphssl.models import DGI
from helpers import make_batch as _make_batch
from helpers import make_graph as _make_graph


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


class TestDGISharedForwardPass:
    """The real and the corrupted graph go through the encoder in one forward pass."""

    N_ISOLATED = 5

    def _graph_with_isolated_nodes(self, n_connected=30, n_feat=7):
        g = torch.Generator().manual_seed(0)
        n = self.N_ISOLATED + n_connected
        edge_index = torch.randint(self.N_ISOLATED, n, (2, 4 * n_connected), generator=g)
        return Data(x=torch.randn(n, n_feat, generator=g), edge_index=edge_index)

    def test_batch_statistics_do_not_tell_the_two_graphs_apart(self, monkeypatch):
        # A node without edges whose features the corruption leaves alone is the same node
        # in both graphs. Its two embeddings can differ only if each graph is normalised
        # with its own batch statistics: the shortcut that gave the discriminator 100%
        # accuracy on such nodes in mini-batch training (ogbn-arxiv stress test).
        cfg = _make_config()
        cfg["encoder"]["drop"] = 0.0
        model = DGI(cfg, in_channels=7)
        model.train()  # BatchNorm uses the statistics of the batch
        graph = self._graph_with_isolated_nodes()
        k = self.N_ISOLATED

        def corrupt_connected_nodes_only(data):
            x = data.x.clone()
            x[k:] = data.x[k + torch.randperm(data.x.size(0) - k)]
            return Data(x=x, edge_index=data.edge_index, batch=data.batch)

        monkeypatch.setattr(model, "_corrupt", corrupt_connected_nodes_only)
        h_pos, h_neg = model._embed_pair(graph)

        assert torch.allclose(h_pos[:k], h_neg[:k], atol=1e-6)
        assert not torch.allclose(h_pos[k:], h_neg[k:], atol=1e-3)


class TestDGIMiniBatch:
    def _logits_in_loss(self, monkeypatch, graph):
        model = DGI(_make_config(), in_channels=7)
        seen = {}
        bce = dgi_module.F.binary_cross_entropy_with_logits

        def spy(logits, labels):
            seen["n"] = logits.numel()
            return bce(logits, labels)

        monkeypatch.setattr(dgi_module.F, "binary_cross_entropy_with_logits", spy)
        assert torch.isfinite(model.compute_loss(graph))
        return seen["n"]

    def test_loss_uses_seed_nodes_only(self, monkeypatch):
        """On a NeighborLoader batch the loss covers the first batch_size nodes."""
        graph = _make_graph(n_nodes=20)
        graph.batch_size = 8  # as NeighborLoader sets it: the seed nodes come first
        assert self._logits_in_loss(monkeypatch, graph) == 2 * 8  # real + corrupted

    def test_full_batch_loss_uses_every_node(self, monkeypatch):
        graph = _make_graph(n_nodes=20)
        assert self._logits_in_loss(monkeypatch, graph) == 2 * 20


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
