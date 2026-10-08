"""Smoke tests for the Supervised pipeline."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from graphssl.models import Supervised


def _make_config(**overrides):
    cfg = {
        "encoder": {"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": False},
    }
    cfg.update(overrides)
    return cfg


def _make_graph(n_nodes=20, n_feat=7, num_classes=3):
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[: n_nodes // 2] = True
    return Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.stack(
            [
                torch.randint(0, n_nodes, (n_nodes * 3,)),
                torch.randint(0, n_nodes, (n_nodes * 3,)),
            ]
        ),
        y=torch.randint(0, num_classes, (n_nodes,)),
        train_mask=train_mask,
    )


def _make_batch(n_graphs=4, n_nodes=10, n_feat=7, num_classes=3):
    """Graph-level batch: one label per graph."""
    graphs = []
    for _ in range(n_graphs):
        graphs.append(
            Data(
                x=torch.randn(n_nodes, n_feat),
                edge_index=torch.stack(
                    [
                        torch.randint(0, n_nodes, (n_nodes * 2,)),
                        torch.randint(0, n_nodes, (n_nodes * 2,)),
                    ]
                ),
                y=torch.randint(0, num_classes, (1,)),
            )
        )
    return Batch.from_data_list(graphs)


class TestSupervised:
    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7
        self.num_classes = 3

    def test_instantiation(self):
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        assert model.encoder is not None
        assert model.head is not None

    def test_head_output_dim(self):
        """Linear head must output exactly num_classes logits."""
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        assert model.head.out_features == self.num_classes

    def test_forward_shape(self):
        """forward() returns encoder embeddings, not logits."""
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels)
        emb = model(graph)
        assert emb.shape == (20, self.config["encoder"]["hidden_dim"])

    def test_compute_loss_finite(self):
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels, num_classes=self.num_classes)
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_compute_loss_non_negative(self):
        """Cross-entropy loss is always non-negative."""
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels, num_classes=self.num_classes)
        loss = model.compute_loss(graph)
        assert loss.item() >= 0.0

    def test_student_parameters_all_params(self):
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        student_ids = {id(p) for p in model.student_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        assert student_ids == all_ids

    def test_mini_batch_crop(self):
        """In mini-batch mode, loss must use only the first batch_size seed nodes."""
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels, num_classes=self.num_classes)
        # Simulate NeighborLoader: batch_size marks how many are seed nodes
        graph.batch_size = 8
        loss = model.compute_loss(graph)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_different_num_classes(self):
        for nc in [2, 5, 10]:
            model = Supervised(self.config, in_channels=self.in_channels, num_classes=nc)
            assert model.head.out_features == nc
            graph = _make_graph(n_nodes=15, n_feat=self.in_channels, num_classes=nc)
            loss = model.compute_loss(graph)
            assert torch.isfinite(loss)

    def test_full_batch_uses_only_train_mask(self):
        """Full-batch loss must see train labels only: val/test labels can't change it."""
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        model.eval()  # deterministic: no dropout, BatchNorm running stats
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels, num_classes=self.num_classes)
        mask = graph.train_mask

        with torch.no_grad():
            loss = model.compute_loss(graph)
            expected = F.cross_entropy(model.head(model(graph))[mask], graph.y[mask])
            graph.y[~mask] = (graph.y[~mask] + 1) % self.num_classes
            loss_other_labels = model.compute_loss(graph)

        assert torch.allclose(loss, expected)
        assert torch.allclose(loss, loss_other_labels)

    def test_full_batch_without_train_mask_raises(self):
        model = Supervised(self.config, in_channels=self.in_channels, num_classes=self.num_classes)
        graph = _make_graph(n_nodes=20, n_feat=self.in_channels, num_classes=self.num_classes)
        del graph.train_mask
        with pytest.raises(ValueError, match="train_mask"):
            model.compute_loss(graph)

    def test_graph_level_pools_and_uses_graph_labels(self):
        """pool=True: one embedding and one loss term per graph, not per node."""
        config = _make_config(
            encoder={"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True}
        )
        model = Supervised(config, in_channels=self.in_channels, num_classes=self.num_classes)
        model.eval()
        batch = _make_batch(n_graphs=4, n_feat=self.in_channels, num_classes=self.num_classes)

        with torch.no_grad():
            emb = model(batch)
            loss = model.compute_loss(batch)
            expected = F.cross_entropy(model.head(emb), batch.y)

        assert model.graph_level
        assert emb.shape == (4, 32)
        assert torch.allclose(loss, expected)

    def test_regression_task_uses_l1_on_float_targets(self):
        """task='regression' (e.g. ZINC): one output per target, mean absolute error."""
        config = _make_config(
            encoder={"name": "gin", "hidden_dim": 32, "num_layers": 2, "pool": True},
            task="regression",
        )
        model = Supervised(config, in_channels=self.in_channels, num_classes=1)
        model.eval()
        batch = _make_batch(n_graphs=4, n_feat=self.in_channels)
        batch.y = torch.randn(4)  # one float target per graph

        with torch.no_grad():
            loss = model.compute_loss(batch)
            expected = (model.head(model(batch)).squeeze(-1) - batch.y).abs().mean()

        assert torch.allclose(loss, expected)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task"):
            Supervised(_make_config(task="ranking"), in_channels=7, num_classes=3)
