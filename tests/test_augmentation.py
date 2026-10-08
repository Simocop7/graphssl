"""Tests for the augmentation functions, compose() and protected nodes."""

import pytest
import torch
from torch_geometric.data import Data

from graphssl.augmentation import (
    EdgeAdd,
    EdgeDrop,
    FeatMask,
    FeatNoise,
    FeatShuffle,
    MultiView,
    NodeDrop,
    Subgraph,
    compose,
)
from graphssl.augmentation import functional as F
from graphssl.models import BGRL, GraphCL

N_NODES = 40
N_SEEDS = 8


def _graph(seed: int = 0) -> Data:
    """A graph whose node i has feature row [i, i, i, i]: rows identify nodes."""
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(N_NODES, dtype=torch.float).unsqueeze(1).repeat(1, 4)
    edge_index = torch.randint(0, N_NODES, (2, 4 * N_NODES), generator=g)
    return Data(x=x, edge_index=edge_index)


def _node_ids(data: Data) -> torch.Tensor:
    return data.x[:, 0].long()


class TestNodeDropProtectedNodes:
    """Seed nodes of a NeighborLoader batch come first and the loss reads ``z[:batch_size]``:
    a view that drops or moves one pairs the wrong nodes across views."""

    def test_protected_nodes_survive_in_place(self):
        data = _graph()
        seeds = torch.arange(N_SEEDS)
        for trial in range(20):
            torch.manual_seed(trial)
            out = F.node_drop(data, p=0.9, protected_nodes=seeds)
            assert torch.equal(_node_ids(out)[:N_SEEDS], seeds)

    def test_without_protection_the_leading_rows_change(self):
        # What the protection prevents: with p = 0.9 some seed is dropped almost surely, and
        # the rows read as "the seeds" then belong to other nodes.
        data = _graph()
        seeds = torch.arange(N_SEEDS)
        moved = 0
        for trial in range(20):
            torch.manual_seed(trial)
            out = F.node_drop(data, p=0.9)
            moved += int(not torch.equal(_node_ids(out)[:N_SEEDS], seeds))
        assert moved == 20

    def test_two_views_keep_the_seeds_aligned(self):
        data = _graph()
        seeds = torch.arange(N_SEEDS)
        augments = [("edge_drop", {"p": 0.5}), ("node_drop", {"p": 0.5})]
        torch.manual_seed(0)
        v1 = compose(data, augments, protected_nodes=seeds)
        v2 = compose(data, augments, protected_nodes=seeds)
        assert v1.num_nodes != v2.num_nodes or not torch.equal(v1.x, v2.x)  # different views
        assert torch.equal(_node_ids(v1)[:N_SEEDS], _node_ids(v2)[:N_SEEDS])

    def test_edges_are_remapped_to_the_surviving_nodes(self):
        data = _graph()
        torch.manual_seed(0)
        out = F.node_drop(data, p=0.5, protected_nodes=torch.arange(N_SEEDS))
        ids = _node_ids(out)
        assert int(out.edge_index.max()) < out.num_nodes
        original = set(map(tuple, data.edge_index.t().tolist()))
        remapped = set(map(tuple, ids[out.edge_index].t().tolist()))
        assert remapped <= original

    def test_subgraph_refuses_protected_nodes(self):
        # It keeps the neighbourhood of one random node: it would silently replace the seeds.
        with pytest.raises(ValueError, match="protected"):
            compose(_graph(), [("subgraph", {"num_hops": 1})], protected_nodes=torch.arange(4))

    def test_multiview_registry_style_forwards_protected_nodes(self):
        data = _graph()
        seeds = torch.arange(N_SEEDS)
        views = MultiView([("node_drop", {"p": 0.9})], n_views=3)(data, protected_nodes=seeds)
        assert len(views) == 3
        assert all(torch.equal(_node_ids(v)[:N_SEEDS], seeds) for v in views)

    @pytest.mark.parametrize("model_cls", [GraphCL, BGRL])
    def test_models_protect_the_seeds_of_a_mini_batch(self, model_cls, monkeypatch):
        # The models pass the seeds of a NeighborLoader batch to compose(): with a node_drop
        # augmentation every view must still start with them.
        import graphssl.models.bgrl as bgrl_module
        import graphssl.models.graphcl as graphcl_module

        module = {GraphCL: graphcl_module, BGRL: bgrl_module}[model_cls]
        views = []

        def spy(data, augments, protected_nodes=None):
            view = compose(data, augments, protected_nodes=protected_nodes)
            views.append(view)
            return view

        monkeypatch.setattr(module, "compose", spy)
        config = {
            "encoder": {"name": "gin", "hidden_dim": 16, "num_layers": 2, "pool": False},
            "augment": [{"name": "node_drop", "p": 0.9}],
        }
        model = model_cls(config, in_channels=4)
        batch = _graph()
        batch.batch_size = N_SEEDS  # as NeighborLoader sets it: the seed nodes come first

        assert torch.isfinite(model.compute_loss(batch))
        assert len(views) == 2
        assert all(torch.equal(_node_ids(v)[:N_SEEDS], torch.arange(N_SEEDS)) for v in views)


class TestAugmentations:
    def test_none_modifies_its_input(self):
        data = _graph()
        x, edge_index = data.x.clone(), data.edge_index.clone()
        for name, kwargs in [
            ("edge_drop", {"p": 0.5}),
            ("edge_add", {"p": 0.5}),
            ("feat_mask", {"p": 0.5}),
            ("feat_noise", {"std": 1.0}),
            ("feat_shuffle", {"p": 0.5}),
            ("node_drop", {"p": 0.5}),
            ("subgraph", {"num_hops": 1}),
        ]:
            compose(data, [(name, kwargs)])
            assert torch.equal(data.x, x) and torch.equal(data.edge_index, edge_index), name

    def test_edge_drop_extremes(self):
        data = _graph()
        assert F.edge_drop(data, p=0.0).edge_index.size(1) == data.edge_index.size(1)
        assert F.edge_drop(data, p=1.0).edge_index.size(1) == 0

    def test_edge_drop_keeps_a_subset_of_the_edges(self):
        data = _graph()
        torch.manual_seed(0)
        out = F.edge_drop(data, p=0.5)
        assert 0 < out.edge_index.size(1) < data.edge_index.size(1)
        original = set(map(tuple, data.edge_index.t().tolist()))
        assert set(map(tuple, out.edge_index.t().tolist())) <= original

    def test_edge_add_appends_edges(self):
        data = _graph()
        out = F.edge_add(data, p=0.25)
        n_edges = data.edge_index.size(1)
        assert out.edge_index.size(1) == n_edges + n_edges // 4
        assert torch.equal(out.edge_index[:, :n_edges], data.edge_index)

    def test_feat_mask_zeroes_whole_columns(self):
        data = Data(x=torch.ones(N_NODES, 64), edge_index=_graph().edge_index)
        torch.manual_seed(0)
        out = F.feat_mask(data, p=0.5)
        column_sums = out.x.sum(dim=0)
        assert set(column_sums.tolist()) == {0.0, float(N_NODES)}  # the same mask for every node

    def test_feat_noise_changes_features_only(self):
        data = _graph()
        out = F.feat_noise(data, std=0.5)
        assert not torch.equal(out.x, data.x)
        assert torch.equal(out.edge_index, data.edge_index)

    def test_feat_shuffle_moves_existing_rows(self):
        data = _graph()
        torch.manual_seed(0)
        out = F.feat_shuffle(data, p=0.5)
        assert not torch.equal(out.x, data.x)
        assert set(_node_ids(out).tolist()) <= set(range(N_NODES))

    def test_node_drop_without_dropping_is_the_identity(self):
        data = _graph()
        out = F.node_drop(data, p=0.0)
        assert torch.equal(out.x, data.x) and torch.equal(out.edge_index, data.edge_index)

    def test_subgraph_returns_a_relabelled_subgraph(self):
        data = _graph()
        torch.manual_seed(0)
        out = F.subgraph(data, num_hops=1)
        assert 0 < out.num_nodes <= N_NODES
        assert out.edge_index.numel() == 0 or int(out.edge_index.max()) < out.num_nodes

    def test_class_based_transforms_wrap_the_functions(self):
        data = _graph()
        transforms = [
            EdgeDrop(p=0.5),
            EdgeAdd(p=0.5),
            FeatMask(p=0.5),
            FeatNoise(std=0.5),
            FeatShuffle(p=0.5),
            NodeDrop(p=0.5),
            Subgraph(num_hops=1),
        ]
        for transform in transforms:
            out = transform(data)
            assert isinstance(out, Data) and out.x.size(1) == data.x.size(1)

    def test_multiview_with_class_based_transforms(self):
        data = _graph()
        torch.manual_seed(0)
        views = MultiView([EdgeDrop(p=0.5), FeatNoise(std=0.5)], n_views=3)(data)
        assert len(views) == 3
        assert not torch.equal(views[0].x, views[1].x)  # independent views
        assert all(v.edge_index.size(1) < data.edge_index.size(1) for v in views)

    def test_unknown_augmentation_raises(self):
        with pytest.raises(KeyError):
            compose(_graph(), [("no_such_augmentation", {})])
