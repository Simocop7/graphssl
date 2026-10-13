"""Tests for new encoder features: edge_emb_num_classes, norm_type API uniformity, CombinedLoss."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphssl.config.schema import EncoderConfig
from graphssl.encoders import GCNEncoder, GINEncoder, TransformerEncoder
from graphssl.losses import BarlowTwinsLoss, CombinedLoss, NTXentLoss, VICRegLoss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zinc_batch(n_graphs: int = 4, n_nodes: int = 8, n_edge_types: int = 4):
    """Mimics a ZINC batch: integer node features, integer edge features."""
    graphs = []
    for _ in range(n_graphs):
        n_edges = n_nodes * 2
        graphs.append(
            Data(
                x=torch.randint(0, 28, (n_nodes, 1)),  # atom type (categorical, 28 classes)
                edge_index=torch.stack(
                    [
                        torch.randint(0, n_nodes, (n_edges,)),
                        torch.randint(0, n_nodes, (n_edges,)),
                    ]
                ),
                edge_attr=torch.randint(0, n_edge_types, (n_edges, 1)),  # bond type (categorical)
            )
        )
    return Batch.from_data_list(graphs)


def _make_continuous_batch(n_graphs: int = 4, n_nodes: int = 8, n_feat: int = 128):
    """Mimics an ogbn-arxiv batch: continuous node features, no edge features."""
    graphs = []
    for _ in range(n_graphs):
        n_edges = n_nodes * 2
        graphs.append(
            Data(
                x=torch.randn(n_nodes, n_feat),
                edge_index=torch.stack(
                    [
                        torch.randint(0, n_nodes, (n_edges,)),
                        torch.randint(0, n_nodes, (n_edges,)),
                    ]
                ),
            )
        )
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# GINEncoder: edge_emb_num_classes (ZINC-style)
# ---------------------------------------------------------------------------


class TestGINEncoderEdgeEmbedding:
    def test_zinc_forward_shape(self):
        """GINEncoder with node + edge embeddings produces correct output shape."""
        enc = GINEncoder(
            in_channels=1,  # ignored when node_emb_num_classes is set
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
                edge_dim=None,  # missing → must raise
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

    def test_eval_statistics_follow_a_short_training(self):
        """After 300 full-batch passes eval mode gives what training saw.

        Small-variance inputs, as bag-of-words features give: the running variance starts
        at 1, far above the batch variance. With a BatchNorm momentum of 0.01 (the default
        until 0.2.0) 5% of that initial value was still there after 300 passes and the
        eval-mode embeddings were not the ones the model had been trained with.
        """
        torch.manual_seed(0)
        n = 400
        x = 0.01 * torch.randn(n, 7)
        edge_index = torch.randint(0, n, (2, 4 * n))
        enc = GCNEncoder(in_channels=7, hidden_dim=32)
        enc.train()
        with torch.no_grad():
            for _ in range(300):
                trained_with = enc(x, edge_index)
            enc.eval()
            in_eval = enc(x, edge_index)
        gap = (in_eval - trained_with).abs().max()
        assert gap < 0.05 * trained_with.std()


# ---------------------------------------------------------------------------
# GCNEncoder: depth and weight standardization
# ---------------------------------------------------------------------------


class TestGCNEncoderDepth:
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_num_layers(self, num_layers):
        enc = GCNEncoder(in_channels=7, hidden_dim=32, out_dim=16, num_layers=num_layers)
        data = _make_continuous_batch(n_feat=7, n_graphs=1)[0]
        assert len(enc.convs) == num_layers
        assert enc(data.x, data.edge_index).shape == (8, 16)

    def test_default_is_two_layers(self):
        assert len(GCNEncoder(in_channels=7, hidden_dim=32).convs) == 2

    def test_config_num_layers_reaches_the_encoder(self):
        # It used to be dropped on the way: every GCN had two layers whatever the config said.
        enc = EncoderConfig(name="gcn", hidden_dim=32, num_layers=3, pool=False).build(7)
        assert len(enc.convs) == 3

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            GCNEncoder(in_channels=7, hidden_dim=32, num_layers=0)

    def test_reset_parameters_changes_every_layer(self):
        enc = GCNEncoder(in_channels=7, hidden_dim=32, num_layers=3)
        before = [conv.lin.weight.detach().clone() for conv in enc.convs]
        enc.reset_parameters()
        after = [conv.lin.weight for conv in enc.convs]
        assert all(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


class TestGCNWeightStandardization:
    def _forward(self, **kwargs):
        enc = GCNEncoder(in_channels=7, hidden_dim=32, num_layers=3, **kwargs)
        before = [conv.lin.weight.detach().clone() for conv in enc.convs]
        data = _make_continuous_batch(n_feat=7, n_graphs=1)[0]
        enc(data.x, data.edge_index)
        return enc, before

    def test_standardizes_every_layer_but_the_first(self):
        # The helper used to look for a direct `weight` parameter, which GCNConv does not
        # have (it is `lin.weight`): the option did nothing.
        enc, before = self._forward(weight_standardization=True)
        assert torch.equal(enc.convs[0].lin.weight, before[0])
        for conv in enc.convs[1:]:
            weight = conv.lin.weight
            assert torch.allclose(weight.mean(dim=1), torch.zeros(32), atol=1e-6)
            assert torch.allclose(weight.var(dim=1), torch.ones(32), atol=1e-2)

    def test_off_by_default(self):
        enc, before = self._forward()
        after = [conv.lin.weight for conv in enc.convs]
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_config_field_reaches_the_encoder(self):
        cfg = {"hidden_dim": 32, "num_layers": 2, "pool": False, "weight_standardization": True}
        assert EncoderConfig(name="gcn", **cfg).build(7).weight_standardization
        # An encoder without the option refuses it instead of ignoring it.
        with pytest.raises(ValueError, match="does not take weight_standardization"):
            EncoderConfig(name="gin", **cfg).build(7)


# ---------------------------------------------------------------------------
# TransformerEncoder: edge features, norm_type, node embedding
# ---------------------------------------------------------------------------


class TestEncoderDropout:
    """``EncoderConfig.drop`` must reach every encoder.

    The GCN encoder had no dropout and the Transformer calls its rate ``dropout``, so a
    ``drop`` in the config was discarded by ``build()`` for both: the benchmark's supervised
    reference "with dropout 0.5" trained its GCN without any.
    """

    @staticmethod
    def _two_train_passes(enc):
        torch.manual_seed(0)
        x = torch.randn(40, 7)
        edge_index = torch.randint(0, 40, (2, 160))
        enc.train()
        return enc(x, edge_index), enc(x, edge_index), x, edge_index

    @pytest.mark.parametrize("name", ["gcn", "gin", "transformer"])
    def test_drop_is_applied_in_training_only(self, name):
        cfg = EncoderConfig(name=name, hidden_dim=32, num_layers=2, pool=False, drop=0.5)
        enc = cfg.build(7)
        first, second, x, edge_index = self._two_train_passes(enc)
        assert not torch.allclose(first, second), "two training passes should differ with dropout"
        enc.eval()
        assert torch.equal(enc(x, edge_index), enc(x, edge_index))

    @pytest.mark.parametrize("name", ["gcn", "gin", "transformer"])
    def test_zero_drop_means_no_dropout(self, name):
        cfg = EncoderConfig(name=name, hidden_dim=32, num_layers=2, pool=False, drop=0.0)
        first, second, _, _ = self._two_train_passes(cfg.build(7))
        assert torch.equal(first, second)

    def test_unset_drop_keeps_each_encoder_default(self):
        """None = the encoder's own default: no dropout for GCN, 0.2 for GIN, 0.1 for Transformer."""
        cfg = {"hidden_dim": 32, "num_layers": 2, "pool": False}
        gcn = EncoderConfig(name="gcn", **cfg).build(7)
        assert gcn.drop == 0.0
        first, second, _, _ = self._two_train_passes(gcn)
        assert torch.equal(first, second)
        for name in ("gin", "transformer"):
            first, second, _, _ = self._two_train_passes(EncoderConfig(name=name, **cfg).build(7))
            assert not torch.allclose(first, second), name

    def test_gcn_without_dropout_is_unchanged(self):
        """drop=0 gives exactly GCNConv → norm → PReLU, the encoder before the parameter existed."""
        torch.manual_seed(0)
        enc = GCNEncoder(in_channels=7, hidden_dim=32, num_layers=2).eval()
        x, edge_index = torch.randn(40, 7), torch.randint(0, 40, (2, 160))
        expected = x
        for conv, norm, act in zip(enc.convs, enc.norms, enc.acts, strict=True):
            expected = act(norm(conv(expected, edge_index)))
        assert torch.equal(enc(x, edge_index), expected)

    @pytest.mark.parametrize("drop", [-0.1, 1.0])
    def test_invalid_drop_raises(self, drop):
        with pytest.raises(ValueError, match="drop must be in"):
            EncoderConfig(name="gcn", hidden_dim=32, num_layers=2, drop=drop)
        with pytest.raises(ValueError, match="drop must be in"):
            GCNEncoder(in_channels=7, hidden_dim=32, drop=drop)
        with pytest.raises(ValueError, match="drop must be in"):
            TransformerEncoder(in_channels=7, hidden_dim=32, drop=drop)


class TestEncoderConfigRefusesIgnoredFields:
    """A field that was set and that the encoder cannot use must not be dropped in silence."""

    @pytest.mark.parametrize(
        "field, value",
        [("edge_dim", 16), ("node_emb_num_classes", 28), ("mlp_ratio", 4.0)],
    )
    def test_gcn_refuses_what_it_cannot_use(self, field, value):
        cfg = EncoderConfig(name="gcn", hidden_dim=32, num_layers=2, **{field: value})
        with pytest.raises(ValueError, match=f"'gcn' does not take {field}"):
            cfg.build(7)

    def test_defaults_and_model_level_fields_pass(self):
        """Fields left at their default, and pool / readout (read by the models), are fine."""
        for name in ("gcn", "gin", "transformer"):
            EncoderConfig(name=name, hidden_dim=32, num_layers=2, pool=True, readout="sum").build(7)

    def test_required_field_the_encoder_drops_is_refused(self):
        """num_layers is always set: an encoder of fixed depth must not drop it in silence."""
        from graphssl.registry import ENCODERS

        @ENCODERS.register("_test_fixed_depth_encoder")
        class _Fixed(torch.nn.Module):
            def __init__(self, in_channels: int, hidden_dim: int):
                super().__init__()

        try:
            cfg = EncoderConfig(name="_test_fixed_depth_encoder", hidden_dim=32, num_layers=2)
            with pytest.raises(ValueError, match="does not take num_layers"):
                cfg.build(7)
        finally:
            ENCODERS._builders.pop("_test_fixed_depth_encoder", None)

    def test_encoder_with_kwargs_takes_everything(self):
        from graphssl.registry import ENCODERS

        seen = {}

        @ENCODERS.register("_test_kwargs_encoder")
        class _Any(torch.nn.Module):
            def __init__(self, in_channels: int, hidden_dim: int, **kwargs):
                super().__init__()
                seen.update(kwargs)

        try:
            EncoderConfig(
                name="_test_kwargs_encoder", hidden_dim=32, num_layers=2, edge_dim=16
            ).build(7)
        finally:
            ENCODERS._builders.pop("_test_kwargs_encoder", None)
        assert seen["edge_dim"] == 16 and seen["num_layers"] == 2 and "drop" not in seen


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
            {
                "name": "vicreg",
                "weight": 0.4,
                "invariance": 25.0,
                "variance": 25.0,
                "covariance": 1.0,
            },
        ]
        fn = CombinedLoss.from_config(config)
        loss = fn(z1, z2)
        assert torch.isfinite(loss)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CombinedLoss([])

    def test_output_is_finite(self):
        z1, z2 = self._make_embeddings()
        fn = CombinedLoss(
            [
                (NTXentLoss(tau=0.5), 0.5),
                (BarlowTwinsLoss(), 0.5),
            ]
        )
        assert torch.isfinite(fn(z1, z2))
