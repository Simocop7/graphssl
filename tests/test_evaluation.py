"""Tests for LogRegEvaluator and KNNEvaluator, incl. OGB-style 2D labels."""

import torch

from graphssl.evaluation import KNNEvaluator, LogRegEvaluator


def _make_splits(n=60, num_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n, 16, generator=g)
    labels_1d = torch.randint(0, num_classes, (n,), generator=g)
    idx = torch.randperm(n, generator=g)
    train_idx, val_idx, test_idx = idx[:30], idx[30:45], idx[45:]
    return embeddings, labels_1d, train_idx, val_idx, test_idx


class TestLogRegEvaluator:
    def test_1d_labels_planetoid_style(self):
        embeddings, labels, train_idx, val_idx, test_idx = _make_splits()
        results = LogRegEvaluator(epochs=5).evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=3
        )
        assert 0.0 <= results["val_acc"] <= 1.0
        assert 0.0 <= results["test_acc"] <= 1.0

    def test_2d_single_label_ogb_style_matches_1d(self):
        # OGB node datasets store y as [N, 1]; results must match the 1D case
        # exactly since it's the same labels, only reshaped.
        embeddings, labels_1d, train_idx, val_idx, test_idx = _make_splits(seed=1)
        labels_2d = labels_1d.unsqueeze(-1)

        torch.manual_seed(0)
        results_1d = LogRegEvaluator(epochs=5).evaluate(
            embeddings, labels_1d, train_idx, val_idx, test_idx, num_classes=3
        )
        torch.manual_seed(0)
        results_2d = LogRegEvaluator(epochs=5).evaluate(
            embeddings, labels_2d, train_idx, val_idx, test_idx, num_classes=3
        )
        assert results_1d == results_2d

    def test_multilabel_labels_not_squeezed(self):
        n, n_classes = 40, 5
        g = torch.Generator().manual_seed(2)
        embeddings = torch.randn(n, 16, generator=g)
        labels = (torch.rand(n, n_classes, generator=g) > 0.5).float()
        idx = torch.randperm(n, generator=g)
        train_idx, val_idx, test_idx = idx[:20], idx[20:30], idx[30:]

        results = LogRegEvaluator(epochs=5, multilabel=True).evaluate(
            embeddings, labels, train_idx, val_idx, test_idx, num_classes=n_classes
        )
        assert "val_ap" in results and "test_ap" in results


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
