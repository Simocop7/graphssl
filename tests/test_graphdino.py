"""Smoke tests for the GraphDINO pipeline."""

import pytest
import torch
from torch_geometric.data import Data

from graphssl.models import GraphDINO
from graphssl.training import DINOTrainer
from graphssl.utils import update_ema_params
from helpers import make_batch


def _make_batch(n_graphs=4, n_nodes_per_graph=10, n_features=7):
    return make_batch(n_graphs=n_graphs, n_nodes=n_nodes_per_graph, n_feat=n_features)


def _make_config(**overrides):
    cfg = {
        "encoder": {
            "name": "gin",
            "hidden_dim": 32,
            "num_layers": 2,
        },
        "head": {
            "name": "dino",
            "proj_hidden": 64,
            "bottleneck_dim": 32,
            "n_prototypes": 128,
            "student_temp": 0.1,
            "teacher_temp": 0.07,
            "center_momentum": 0.9,
            "warmup_teacher_temp": 0.04,
            "warmup_teacher_temp_epochs": 0,
        },
        "augment_teacher": [
            {"name": "edge_drop", "p": 0.2},
            {"name": "feat_mask", "p": 0.1},
        ],
        "augment_student": [
            {"name": "edge_drop", "p": 0.3},
            {"name": "feat_mask", "p": 0.2},
        ],
        "ema_tau": 0.996,
        "freeze_last_layer_epochs": 0,
    }
    cfg.update(overrides)
    return cfg


class TestGraphDINO:
    def setup_method(self):
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        model = GraphDINO(self.config, in_channels=self.in_channels)
        assert model.student_enc is not None
        assert model.teacher_enc is not None
        assert model.student_head is not None
        assert model.teacher_head is not None

    def test_teacher_is_frozen(self):
        model = GraphDINO(self.config, in_channels=self.in_channels)
        for p in model.teacher_enc.parameters():
            assert not p.requires_grad
        for p in model.teacher_head.parameters():
            assert not p.requires_grad

    def test_forward_shapes(self):
        model = GraphDINO(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_features=self.in_channels)

        emb = model(batch)
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])

        loss = model.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_last_teacher_out_stored(self):
        model = GraphDINO(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_features=self.in_channels)
        model.compute_loss(batch)
        assert model.last_teacher_out is not None
        assert model.last_teacher_out.shape[0] == 2 * 4

    def test_invalid_config_raises(self):
        bad = _make_config()
        bad["encoder"]["hidden_dim"] = -1
        with pytest.raises(ValueError, match="hidden_dim"):
            GraphDINO(bad, in_channels=self.in_channels)

    def test_student_parameters_excludes_teacher(self):
        model = GraphDINO(self.config, in_channels=self.in_channels)
        student_ids = {id(p) for p in model.student_parameters()}
        teacher_ids = {id(p) for p in model.teacher_enc.parameters()} | {
            id(p) for p in model.teacher_head.parameters()
        }
        assert student_ids.isdisjoint(teacher_ids)


class TestFreezeLastLayer:
    def test_proto_grads_zeroed_during_freeze(self):
        """post_backward() cancels proto gradients when epoch < freeze_last_layer_epochs."""
        cfg = _make_config()
        cfg["freeze_last_layer_epochs"] = 2
        model = GraphDINO(cfg, in_channels=7)
        batch = _make_batch(n_features=7)

        model.train()
        loss = model.compute_loss(batch)
        loss.backward()

        # epoch 0 < 2 → freeze active
        model.post_backward()
        for p in model.student_head.proto.parameters():
            assert p.grad is None

    def test_proto_grads_intact_after_freeze_window(self):
        """post_backward() leaves proto gradients alone once freeze_last_layer_epochs is past."""
        cfg = _make_config()
        cfg["freeze_last_layer_epochs"] = 1
        model = GraphDINO(cfg, in_channels=7)
        model.on_epoch_end(0)  # advances _epoch to 1

        batch = _make_batch(n_features=7)
        model.train()
        loss = model.compute_loss(batch)
        loss.backward()

        model.post_backward()
        # _epoch == 1 == freeze_last_layer_epochs → freeze no longer active
        has_grad = any(p.grad is not None for p in model.student_head.proto.parameters())
        assert has_grad

    def test_no_freeze_when_zero(self):
        """freeze_last_layer_epochs=0 means the prototype trains from epoch 0."""
        model = GraphDINO(_make_config(), in_channels=7)  # freeze=0 in default config
        batch = _make_batch(n_features=7)

        model.train()
        loss = model.compute_loss(batch)
        loss.backward()
        model.post_backward()

        has_grad = any(p.grad is not None for p in model.student_head.proto.parameters())
        assert has_grad


class TestTeacherTempWarmup:
    def test_warmup_starts_at_warmup_temp(self):
        """At epoch 0 the effective teacher temp equals warmup_teacher_temp."""
        cfg = _make_config()
        cfg["head"]["warmup_teacher_temp"] = 0.04
        cfg["head"]["teacher_temp"] = 0.07
        cfg["head"]["warmup_teacher_temp_epochs"] = 10
        model = GraphDINO(cfg, in_channels=7)

        model.on_epoch_start(0)
        assert model.teacher_head._current_teacher_temp == pytest.approx(0.04)

    def test_warmup_reaches_final_temp(self):
        """At epoch >= warmup_teacher_temp_epochs the effective temp equals teacher_temp."""
        cfg = _make_config()
        cfg["head"]["warmup_teacher_temp"] = 0.04
        cfg["head"]["teacher_temp"] = 0.07
        cfg["head"]["warmup_teacher_temp_epochs"] = 10
        model = GraphDINO(cfg, in_channels=7)

        model.on_epoch_start(10)
        assert model.teacher_head._current_teacher_temp == pytest.approx(0.07)

    def test_warmup_is_linear(self):
        """Halfway through warmup the temp is the midpoint."""
        cfg = _make_config()
        cfg["head"]["warmup_teacher_temp"] = 0.04
        cfg["head"]["teacher_temp"] = 0.08
        cfg["head"]["warmup_teacher_temp_epochs"] = 10
        model = GraphDINO(cfg, in_channels=7)

        model.on_epoch_start(5)
        assert model.teacher_head._current_teacher_temp == pytest.approx(0.06)

    def test_no_warmup_by_default(self):
        """warmup_teacher_temp_epochs=0 means teacher_temp is used from the start."""
        model = GraphDINO(_make_config(), in_channels=7)
        assert model.teacher_head._current_teacher_temp == pytest.approx(0.07)


class TestEMAUpdate:
    def test_teacher_moves_toward_student(self):
        student = torch.nn.Linear(4, 4)
        teacher = torch.nn.Linear(4, 4)
        with torch.no_grad():
            teacher.weight.fill_(0.0)
            teacher.bias.fill_(0.0)

        tau = 0.5
        update_ema_params(student, teacher, tau)

        expected = student.weight.data * (1.0 - tau)
        assert torch.allclose(teacher.weight.data, expected, atol=1e-6)


class TestDINOTrainer:
    def test_train_epoch_runs(self):
        from torch_geometric.loader import DataLoader

        model = GraphDINO(_make_config(), in_channels=7)
        optimizer = torch.optim.Adam(model.student_parameters(), lr=1e-3)

        graphs = [
            Data(
                x=torch.randn(10, 7),
                edge_index=torch.stack(
                    [
                        torch.randint(0, 10, (20,)),
                        torch.randint(0, 10, (20,)),
                    ]
                ),
            )
            for _ in range(8)
        ]
        loader = DataLoader(graphs, batch_size=4)
        trainer = DINOTrainer(grad_clip_norm=3.0)
        avg_loss = trainer.train_epoch(model, loader, optimizer)

        assert isinstance(avg_loss, float)
        assert not (avg_loss != avg_loss)  # not NaN

    def test_train_calls_epoch_hooks(self):
        """train() must call on_epoch_start/end so freeze and warmup advance correctly."""
        from torch_geometric.loader import DataLoader

        cfg = _make_config()
        cfg["freeze_last_layer_epochs"] = 1
        model = GraphDINO(cfg, in_channels=7)

        graphs = [
            Data(
                x=torch.randn(10, 7),
                edge_index=torch.stack(
                    [
                        torch.randint(0, 10, (20,)),
                        torch.randint(0, 10, (20,)),
                    ]
                ),
            )
            for _ in range(8)
        ]
        loader = DataLoader(graphs, batch_size=4)
        optimizer = torch.optim.Adam(model.student_parameters(), lr=1e-3)
        trainer = DINOTrainer(grad_clip_norm=3.0)
        trainer.train(model, loader, optimizer, num_epochs=2)

        # After 2 epochs, _epoch should be 2
        assert model._epoch == 2
