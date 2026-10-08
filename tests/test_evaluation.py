"""Tests for LogRegEvaluator and KNNEvaluator (incl. OGB-style 2D labels) and effective_rank."""

import pytest
import torch
from torch_geometric.data import Data

from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, effective_rank, extract_embeddings
from graphssl.evaluation.linear_probe import DEFAULT_WEIGHT_DECAYS


def _make_splits(n=60, num_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n, 16, generator=g)
    labels_1d = torch.randint(0, num_classes, (n,), generator=g)
    idx = torch.randperm(n, generator=g)
    train_idx, val_idx, test_idx = idx[:30], idx[30:45], idx[45:]
    return embeddings, labels_1d, train_idx, val_idx, test_idx


def _make_clusters(n=240, num_classes=3, dim=16, seed=0, dtype=torch.float64):
    """Overlapping Gaussian clusters: a problem a linear probe can partly solve."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    centers = torch.randn(num_classes, dim, generator=g, dtype=dtype)
    embeddings = centers[labels] + 2.0 * torch.randn(n, dim, generator=g, dtype=dtype)
    idx = torch.randperm(n, generator=g)
    return embeddings, labels, idx[: n // 2], idx[n // 2 : 3 * n // 4], idx[3 * n // 4 :]


class TestLogRegEvaluator:
    def test_1d_labels_planetoid_style(self):
        embeddings, labels, train_idx, val_idx, test_idx = _make_splits()
        results = LogRegEvaluator().evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=3
        )
        assert 0.0 <= results["val_acc"] <= 1.0
        assert 0.0 <= results["test_acc"] <= 1.0
        assert results["weight_decay"] in DEFAULT_WEIGHT_DECAYS
        assert results["converged"]

    def test_2d_single_label_ogb_style_matches_1d(self):
        # OGB node datasets store y as [N, 1]; results must match the 1D case
        # exactly since it's the same labels, only reshaped.
        embeddings, labels_1d, train_idx, val_idx, test_idx = _make_splits(seed=1)
        labels_2d = labels_1d.unsqueeze(-1)

        results_1d = LogRegEvaluator().evaluate(
            embeddings, labels_1d, train_idx, val_idx, test_idx, num_classes=3
        )
        results_2d = LogRegEvaluator().evaluate(
            embeddings, labels_2d, train_idx, val_idx, test_idx, num_classes=3
        )
        assert results_1d == results_2d

    def test_result_does_not_depend_on_the_random_seed(self):
        # The 0.1.0 probe (random init, 100 Adam steps) gave 45-54% on the same embeddings.
        splits = _make_clusters()
        runs = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            runs.append(LogRegEvaluator().evaluate(*splits, num_classes=3))
        assert runs[0] == runs[1] == runs[2]

    def test_weight_decay_is_selected_on_validation(self):
        splits = _make_clusters(seed=3)
        grid = (1.0, 1e-2, 1e-4)
        single = {
            wd: LogRegEvaluator(weight_decay=wd).evaluate(*splits, num_classes=3) for wd in grid
        }
        # Best validation accuracy, the strongest regularisation on ties.
        expected = max(grid, key=lambda wd: (single[wd]["val_acc"], wd))

        selected = LogRegEvaluator(weight_decay=grid).evaluate(*splits, num_classes=3)

        assert selected["weight_decay"] == expected
        assert selected["val_acc"] == single[expected]["val_acc"]
        assert selected["test_acc"] == single[expected]["test_acc"]

    def test_standardization_removes_feature_scale_and_offset(self):
        embeddings, labels, train_idx, val_idx, test_idx = _make_clusters(seed=4)
        g = torch.Generator().manual_seed(0)
        scale = torch.logspace(-3, 3, embeddings.size(1), dtype=embeddings.dtype)
        offset = 100 * torch.randn(embeddings.size(1), generator=g, dtype=embeddings.dtype)
        rescaled = embeddings * scale + offset

        plain = LogRegEvaluator().evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=3
        )
        moved = LogRegEvaluator().evaluate(
            rescaled, labels, train_idx, val_idx, test_idx, num_classes=3
        )
        assert plain == moved

    def test_converged_reports_an_exhausted_budget(self):
        splits = _make_clusters(seed=5)
        short = LogRegEvaluator(weight_decay=1e-4, max_iter=2).evaluate(*splits, num_classes=3)
        full = LogRegEvaluator(weight_decay=1e-4).evaluate(*splits, num_classes=3)
        assert not short["converged"]
        assert full["converged"]

    def test_constant_feature_does_not_produce_nans(self):
        embeddings, labels, train_idx, val_idx, test_idx = _make_splits(seed=6)
        embeddings[:, 0] = 3.0  # zero variance on every split
        results = LogRegEvaluator().evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=3
        )
        assert 0.0 <= results["test_acc"] <= 1.0

    def test_multilabel_labels_not_squeezed(self):
        n, n_classes = 40, 5
        g = torch.Generator().manual_seed(2)
        embeddings = torch.randn(n, 16, generator=g)
        labels = (torch.rand(n, n_classes, generator=g) > 0.5).float()
        labels[0, 0] = float("nan")  # missing labels are skipped
        idx = torch.randperm(n, generator=g)
        train_idx, val_idx, test_idx = idx[:20], idx[20:30], idx[30:]

        results = LogRegEvaluator(weight_decay=1e-2, multilabel=True).evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=n_classes
        )
        assert "val_ap" in results and "test_ap" in results

    def test_multilabel_selection_needs_average_precision(self, monkeypatch):
        import graphssl.evaluation.linear_probe as linear_probe

        monkeypatch.setattr(linear_probe, "HAS_SKLEARN", False)
        embeddings = torch.randn(20, 4)
        labels = (torch.rand(20, 3) > 0.5).float()
        idx = torch.arange(20)
        with pytest.raises(ImportError, match="scikit-learn"):
            LogRegEvaluator(multilabel=True).evaluate(
                embeddings, labels, idx[:10], idx[10:15], idx[15:], num_classes=3
            )

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="weight_decay"):
            LogRegEvaluator(weight_decay=[])
        with pytest.raises(ValueError, match="weight_decay"):
            LogRegEvaluator(weight_decay=-1.0)
        with pytest.raises(ValueError, match="max_iter"):
            LogRegEvaluator(max_iter=0)
        # 0.1.0 took (lr, epochs, ...) positionally: fail loudly instead of reading them
        # as (weight_decay, max_iter).
        with pytest.raises(TypeError):
            LogRegEvaluator(0.01, 100)


class TestKNNEvaluator:
    def test_1d_labels_planetoid_style(self):
        embeddings, labels, train_idx, val_idx, test_idx = _make_splits()
        results = KNNEvaluator(k=5).evaluate(embeddings, labels, train_idx, val_idx, test_idx)
        assert 0.0 <= results["val_acc"] <= 1.0
        assert 0.0 <= results["test_acc"] <= 1.0

    def test_2d_single_label_ogb_style_matches_1d(self):
        embeddings, labels_1d, train_idx, val_idx, test_idx = _make_splits(seed=3)
        labels_2d = labels_1d.unsqueeze(-1)

        results_1d = KNNEvaluator(k=5).evaluate(embeddings, labels_1d, train_idx, val_idx, test_idx)
        results_2d = KNNEvaluator(k=5).evaluate(embeddings, labels_2d, train_idx, val_idx, test_idx)
        assert results_1d == results_2d

    def test_chunked_matches_single_pass(self):
        """Scoring the split in row chunks must not change any prediction."""
        splits = _make_splits(n=200, seed=1)
        whole = KNNEvaluator(k=5, chunk_size=10_000).evaluate(*splits)
        for chunk_size in (1, 7, 64):
            assert KNNEvaluator(k=5, chunk_size=chunk_size).evaluate(*splits) == whole

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            KNNEvaluator(chunk_size=0)


class _SpyData(Data):
    """Records whether .to() was called on this very object."""

    def to(self, *args, **kwargs):
        self.__dict__["moved"] = True
        return super().to(*args, **kwargs)


class _Identity(torch.nn.Module):
    def forward(self, data):
        return data.x


class TestExtractEmbeddings:
    def test_full_batch_extraction_leaves_the_datamodule_graph_in_place(self):
        """Data.to() mutates in place: extraction must move a copy, or a NeighborLoader built
        on the datamodule's graph would suddenly sample from GPU tensors."""
        data = _SpyData(
            x=torch.randn(6, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            y=torch.arange(6),
        )
        dm = DataModule(data=data, is_graph_level=False)

        z, y = extract_embeddings(_Identity(), dm, device="cpu")

        assert "moved" not in data.__dict__
        assert torch.equal(z, data.x) and torch.equal(y, data.y)


class TestEffectiveRank:
    def test_isotropic_embeddings_use_every_dimension(self):
        z = torch.randn(4000, 16, generator=torch.Generator().manual_seed(0))
        assert effective_rank(z) > 15.5

    def test_rank_one_embeddings(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(500, 1, generator=g) @ torch.randn(1, 32, generator=g)
        assert effective_rank(z) == pytest.approx(1.0, abs=1e-3)

    def test_counts_only_directions_with_variance(self):
        g = torch.Generator().manual_seed(0)
        z = torch.cat([torch.randn(4000, 3, generator=g), torch.zeros(4000, 13)], dim=1)
        assert effective_rank(z) == pytest.approx(3.0, abs=0.05)

    def test_invariant_to_shift_and_scale(self):
        z = torch.randn(300, 8, generator=torch.Generator().manual_seed(0))
        assert effective_rank(3.0 * z + 5.0) == pytest.approx(effective_rank(z), rel=1e-5)

    def test_no_variance_is_zero(self):
        assert effective_rank(torch.ones(50, 8)) == 0.0
