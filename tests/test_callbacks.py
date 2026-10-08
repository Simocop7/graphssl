"""Tests for the built-in trainer callbacks."""

import torch
from torch.optim import AdamW
from torch_geometric.data import Data

from graphssl.core.callback import Callback
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator
from graphssl.models import BarlowTwins
from graphssl.training import DINOTrainer
from graphssl.training.callbacks import EmbeddingLoggerCallback, LinearEvalCallback

N_NODES, N_FEAT, N_CLASSES, HIDDEN = 60, 8, 3, 16


def _datamodule() -> DataModule:
    g = torch.Generator().manual_seed(0)
    data = Data(
        x=torch.randn(N_NODES, N_FEAT, generator=g),
        edge_index=torch.randint(0, N_NODES, (2, 4 * N_NODES), generator=g),
        y=torch.randint(0, N_CLASSES, (N_NODES,), generator=g),
    )
    idx = torch.randperm(N_NODES, generator=g)
    return DataModule(
        data=data, is_graph_level=False, train_idx=idx[:30], val_idx=idx[30:45], test_idx=idx[45:]
    )


def _train(dm: DataModule, callbacks: list, num_epochs: int) -> BarlowTwins:
    torch.manual_seed(0)
    config = {
        "encoder": {"name": "gin", "hidden_dim": HIDDEN, "num_layers": 2, "pool": False},
        "augment": [{"name": "edge_drop", "p": 0.2}],
        "proj_dim": 16,
    }
    model = BarlowTwins(config, in_channels=N_FEAT)
    optimizer = AdamW(model.student_parameters(), lr=1e-3)
    trainer = DINOTrainer(grad_clip_norm=None, callbacks=callbacks)
    trainer.train(model, dm.train_dataloader(), optimizer, num_epochs=num_epochs)
    return model


class TestEmbeddingLoggerCallback:
    def test_saves_embeddings_and_labels_at_its_interval(self, tmp_path):
        dm = _datamodule()
        callback = EmbeddingLoggerCallback(dm, save_dir=str(tmp_path), every_n_epochs=2)
        _train(dm, [callback], num_epochs=4)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["epoch_0000.pt", "epoch_0002.pt"]
        saved = torch.load(tmp_path / "epoch_0002.pt")
        assert saved["embeddings"].shape == (N_NODES, HIDDEN)
        assert torch.equal(saved["labels"], dm.data.y)


class TestLinearEvalCallback:
    def test_records_one_result_per_evaluated_epoch(self):
        dm = _datamodule()
        callback = LinearEvalCallback(dm, num_classes=N_CLASSES, every_n_epochs=2)
        _train(dm, [callback], num_epochs=4)

        history = callback.results_history
        assert [r["epoch"] for r in history] == [0, 2]
        assert all(0.0 <= r["test_acc"] <= 1.0 and "weight_decay" in r for r in history)

    def test_uses_the_evaluator_it_is_given(self):
        dm = _datamodule()
        evaluator = LogRegEvaluator(weight_decay=0.5)
        callback = LinearEvalCallback(
            dm, num_classes=N_CLASSES, every_n_epochs=1, evaluator=evaluator
        )
        _train(dm, [callback], num_epochs=1)
        assert callback.results_history[0]["weight_decay"] == 0.5

    def test_training_continues_in_train_mode_after_an_evaluation(self):
        # extract_embeddings puts the model in eval mode; the next epoch must train again.
        dm = _datamodule()
        modes = []

        class RecordMode(Callback):
            def on_batch_end(self, trainer, model, loss, batch):
                modes.append(model.training)

        callback = LinearEvalCallback(dm, num_classes=N_CLASSES, every_n_epochs=1)
        _train(dm, [callback, RecordMode()], num_epochs=3)
        assert modes == [True, True, True]
