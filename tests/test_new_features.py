"""Tests for new encoder features: edge_emb_num_classes, norm_type API uniformity, CombinedLoss."""

from __future__ import annotations

import sys, os
import pytest
import torch
from torch_geometric.data import Data, Batch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graphssl.encoders import GINEncoder, GCNEncoder, TransformerEncoder
from graphssl.losses import CombinedLoss, NTXentLoss, VICRegLoss, BarlowTwinsLoss
from graphssl.config.schema import EncoderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zinc_batch(n_graphs: int = 4, n_nodes: int = 8, n_edge_types: int = 4):
    """Mimics a ZINC batch: integer node features, integer edge features."""
    graphs = []
    for _ in range(n_graphs):
        n_edges = n_nodes * 2
        graphs.append(Data(
            x=torch.randint(0, 28, (n_nodes, 1)),        # atom type (categorical, 28 classes)
            edge_index=torch.stack([
                torch.randint(0, n_nodes, (n_edges,)),
                torch.randint(0, n_nodes, (n_edges,)),
            ]),
            edge_attr=torch.randint(0, n_edge_types, (n_edges, 1)),   # bond type (categorical)
        ))
    return Batch.from_data_list(graphs)


def _make_continuous_batch(n_graphs: int = 4, n_nodes: int = 8, n_feat: int = 128):
    """Mimics an ogbn-arxiv batch: continuous node features, no edge features."""
    graphs = []
    for _ in range(n_graphs):
        n_edges = n_nodes * 2
        graphs.append(Data(
            x=torch.randn(n_nodes, n_feat),
            edge_index=torch.stack([
                torch.randint(0, n_nodes, (n_edges,)),
                torch.randint(0, n_nodes, (n_edges,)),
            ]),
        ))
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# GINEncoder: edge_emb_num_classes (ZINC-style)
# ---------------------------------------------------------------------------

class TestGINEncoderEdgeEmbedding:

    def test_zinc_forward_shape(self):
        """GINEncoder with node + edge embeddings produces correct output shape."""
        enc = GINEncoder(
            in_channels=1,      # ignored when node_emb_num_classes is set
            hidden_dim=32,
            num_layers=3,
            pool=True,
            node_emb_num_classes=28,
            edge_dim=32,
            edge_emb_num_classes=4,
        )
        batch = _make_zinc_batch(n_graphs=4)
        out = enc(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        assert out.shape == (4 * 8, 32), f"Expected node-level shape, got {out.shape}"

    def test_zinc_no_edge_attr_still_works(self):
        """When edge_attr is None but edge_proj exists, encoder must not crash."""
        enc = GINEncoder(
            in_channels=1,
            hidden_dim=16,
            num_layers=2,
            pool=False,
            node_emb_num_classes=28,
            edge_dim=16,
            edge_emb_num_classes=4,
        )
        batch = _make_zinc_batch()
        # Pass edge_attr=None to simulate missing edge features
        out = enc(batch.x, batch.edge_index, batch.batch, edge_attr=None)
        assert out.shape[1] == 16

    def test_edge_emb_requires_edge_dim(self):
        with pytest.raises(ValueError, match="edge_emb_num_classes requires edge_dim"):
            GINEncoder(
                in_channels=7,
                hidden_dim=32,
                num_layers=2,
                edge_emb_num_classes=4,
                edge_dim=None,    # missing → must raise
            )

    def test_edge_emb_config_validation(self):
        """EncoderConfig must reject edge_emb_num_classes without edge_dim."""
        with pytest.raises(ValueError, match="edge_emb_num_classes requires edge_dim"):
            EncoderConfig(
                name="gin",
                hidden_dim=32,
                num_layers=2,
                edge_emb_num_classes=4,
                edge_dim=None,
            )

    def test_encoder_config_builds_zinc_gin(self):
        """EncoderConfig.build() must correctly forward edge_emb_num_classes to GINEncoder."""
        cfg = EncoderConfig(
            name="gin",
            hidden_dim=32,
            num_layers=2,
            node_emb_num_classes=28,
            edge_dim=32,
            edge_emb_num_classes=4,
        )
        enc = cfg.build(in_channels=1)
        batch = _make_zinc_batch()
        out = enc(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        assert out.shape[1] == 32


# ---------------------------------------------------------------------------
# GCNEncoder: uniform norm_type API
# ---------------------------------------------------------------------------

class TestGCNEncoderNormType:

    @pytest.mark.parametrize("norm_type", ["batch", "layer", "none"])
    def test_norm_type_api(self, norm_type):
        enc = GCNEncoder(in_channels=7, hidden_dim=32, norm_type=norm_type)
        data = _make_continuous_batch(n_feat=7, n_graphs=1)[0]
        out = enc(data.x, data.edge_index)
        assert out.shape == (8, 32)

    def test_backward_compat_batchnorm_true(self):
        """Old batchnorm=True flag must still work."""
        enc = GCNEncoder(in_channels=7, hidden_dim=32, batchnorm=True)
        data = _make_continuous_batch(n_feat=7, n_graphs=1)[0]
        assert enc(data.x, data.edge_index).shape == (8, 32)

    def test_backward_compat_layernorm_true(self):
        enc = GCNEncoder(in_channels=7, hidden_dim=32, layernorm=True)
        data = _make_continuous_batch(n_feat=7, n_graphs=1)[0]
        assert enc(data.x, data.edge_index).shape == (8, 32)

    def test_batchnorm_and_layernorm_raises(self):
        with pytest.raises(ValueError):
            GCNEncoder(in_channels=7, hidden_dim=32, batchnorm=True, layernorm=True)


# ---------------------------------------------------------------------------
# TransformerEncoder: edge features, norm_type, node embedding
# ---------------------------------------------------------------------------

class TestTransformerEncoderNewFeatures:

    def test_norm_type_batch(self):
        enc = TransformerEncoder(in_channels=7, hidden_dim=16, heads=2, norm_type="batch")
        batch = _make_continuous_batch(n_feat=7, n_graphs=2)
        out = enc(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (2 * 8, 16)

    def test_norm_type_layer(self):
        enc = TransformerEncoder(in_channels=7, hidden_dim=16, heads=2, norm_type="layer")
        batch = _make_continuous_batch(n_feat=7, n_graphs=2)
        out = enc(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (2 * 8, 16)

    def test_edge_features(self):
        """TransformerEncoder with edge_dim correctly processes ZINC-style edge_attr."""
        enc = TransformerEncoder(
            in_channels=1,
            hidden_dim=16,
            heads=2,
            node_emb_num_classes=28,
            edge_dim=8,
            edge_emb_num_classes=4,
        )
        batch = _make_zinc_batch(n_graphs=2)
        out = enc(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        assert out.shape == (2 * 8, 16)

    def test_edge_emb_requires_edge_dim(self):
        with pytest.raises(ValueError, match="edge_emb_num_classes requires edge_dim"):
            TransformerEncoder(in_channels=7, hidden_dim=16, edge_emb_num_classes=4, edge_dim=None)


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss:

    def _make_embeddings(self, n: int = 8, d: int = 16):
        z1 = torch.randn(n, d)
        z2 = torch.randn(n, d)
        return z1, z2

    def test_single_loss_passthrough(self):
        """CombinedLoss with one component at weight=1.0 should equal the raw loss."""
        z1, z2 = self._make_embeddings()
        raw = NTXentLoss(tau=0.5)(z1, z2)
        combined = CombinedLoss([(NTXentLoss(tau=0.5), 1.0)])(z1, z2)
        assert torch.allclose(raw, combined, atol=1e-5)

    def test_weighted_sum(self):
        """Weighted combination must match manual computation."""
        z1, z2 = self._make_embeddings()
        loss_a = NTXentLoss(tau=0.5)
        loss_b = VICRegLoss()
        w_a, w_b = 0.7, 0.3

        combined = CombinedLoss([(loss_a, w_a), (loss_b, w_b)])(z1, z2)
        expected = w_a * loss_a(z1, z2) + w_b * loss_b(z1, z2)
        assert torch.allclose(combined, expected, atol=1e-5)

    def test_from_config(self):
        """CombinedLoss.from_config must build and run without error."""
        z1, z2 = self._make_embeddings()
        config = [
            {"name": "nt_xent", "weight": 0.6, "tau": 0.5},
            {"name": "vicreg",  "weight": 0.4, "invariance": 25.0, "variance": 25.0, "covariance": 1.0},
        ]
        fn = CombinedLoss.from_config(config)
        loss = fn(z1, z2)
        assert torch.isfinite(loss)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CombinedLoss([])

    def test_output_is_finite(self):
        z1, z2 = self._make_embeddings()
        fn = CombinedLoss([
            (NTXentLoss(tau=0.5), 0.5),
            (BarlowTwinsLoss(), 0.5),
        ])
        assert torch.isfinite(fn(z1, z2))
