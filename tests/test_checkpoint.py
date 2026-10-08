"""Tests for save_model / load_model and for reusing a pre-trained encoder."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphssl.config import build_model, load_model, save_model
from graphssl.core import pretrained_encoder
from graphssl.utils.positive_miner import HAS_FAISS

HIDDEN, IN_CHANNELS = 16, 7
AUGMENT = [{"name": "edge_drop", "p": 0.2}]
MODELS = [
    "dgi",
    "graphcl",
    "vicreg",
    "barlow_twins",
    "bgrl",
    pytest.param("afgrl", marks=pytest.mark.skipif(not HAS_FAISS, reason="needs faiss-cpu")),
    "graphdino",
]


def _config(model: str) -> dict:
    cfg: dict = {
        "name": model,
        "encoder": {"name": "gin", "hidden_dim": HIDDEN, "num_layers": 2, "pool": True},
    }
    if model in ("graphcl", "vicreg", "barlow_twins", "bgrl"):
        cfg["augment"] = AUGMENT
    if model == "graphdino":
        cfg["head"] = {
            "name": "dino",
            "proj_hidden": HIDDEN,
            "bottleneck_dim": 8,
            "n_prototypes": 8,
        }
        cfg["augment_teacher"] = AUGMENT
        cfg["augment_student"] = AUGMENT
    return cfg


def _batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    graphs = [
        Data(
            x=torch.randn(10, IN_CHANNELS, generator=g),
            edge_index=torch.randint(0, 10, (2, 30), generator=g),
            y=torch.randn(1, generator=g),
        )
        for _ in range(4)
    ]
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("model_name", MODELS)
def test_save_and_load_give_the_same_embeddings(model_name, tmp_path):
    torch.manual_seed(0)
    config = _config(model_name)
    model = build_model(config, in_channels=IN_CHANNELS)
    model.train()
    model.compute_loss(_batch()).backward()  # touch BatchNorm statistics and step counters
    model.eval()

    path = tmp_path / "model.pt"
    save_model(model, path, config, in_channels=IN_CHANNELS)
    loaded = load_model(path)
    loaded.eval()

    assert type(loaded) is type(model)
    assert torch.equal(loaded(_batch()), model(_batch()))


def test_supervised_checkpoint_keeps_num_classes(tmp_path):
    config = {**_config("supervised"), "task": "regression"}
    model = build_model(config, in_channels=IN_CHANNELS, num_classes=2)
    save_model(model, tmp_path / "sup.pt", config, in_channels=IN_CHANNELS, num_classes=2)
    assert load_model(tmp_path / "sup.pt").head.out_features == 2


@pytest.mark.parametrize("model_name", MODELS)
def test_pretrained_encoder_is_the_one_forward_uses(model_name):
    # Fine-tuning must start from the encoder that produced the evaluated embeddings.
    torch.manual_seed(0)
    model = build_model(_config(model_name), in_channels=IN_CHANNELS)
    model.eval()
    batch = _batch()
    before = model(batch)
    with torch.no_grad():
        for p in pretrained_encoder(model).parameters():
            p.add_(1.0)
    assert not torch.allclose(model(batch), before)


def test_fine_tuning_starts_from_the_pretrained_weights():
    torch.manual_seed(0)
    pretrained = build_model(_config("bgrl"), in_channels=IN_CHANNELS)
    supervised = build_model(
        {**_config("supervised"), "task": "regression"}, in_channels=IN_CHANNELS, num_classes=1
    )
    supervised.encoder.load_state_dict(pretrained_encoder(pretrained).state_dict())

    pretrained.eval()
    supervised.eval()
    batch = _batch()
    assert torch.equal(supervised(batch), pretrained(batch))


def test_pretrained_encoder_rejects_other_modules():
    with pytest.raises(ValueError, match="encoder"):
        pretrained_encoder(torch.nn.Linear(2, 2))
