"""Every model must build and train with every registered encoder.

Models get their encoder from ``cfg.encoder.build()``, which forwards only the kwargs each
encoder accepts. A model that builds its encoder another way only breaks with some encoders
(GraphDINO once crashed with GCN and Transformer while working with GIN), so the whole
model × encoder matrix is exercised here, node-level and graph-level.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphssl.config.load import build_model
from graphssl.core import encode, pretrained_encoder
from graphssl.nn.pooling import pool_graph_embeddings
from graphssl.utils.positive_miner import HAS_FAISS

HIDDEN = 16
IN_CHANNELS = 7
NUM_CLASSES = 3
ENCODERS = ["gin", "gcn", "transformer"]
AUGMENT = [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}]


def _config(model: str, encoder: str, pool: bool) -> dict:
    cfg: dict = {
        "name": model,
        "encoder": {"name": encoder, "hidden_dim": HIDDEN, "num_layers": 2, "pool": pool},
    }
    if model in ("graphcl", "vicreg", "barlow_twins", "bgrl"):
        cfg["augment"] = AUGMENT
    if model in ("bgrl", "afgrl"):
        cfg["pred_hidden"] = HIDDEN
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


def _graph(n: int = 30, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: n // 2] = True
    return Data(
        x=torch.randn(n, IN_CHANNELS, generator=g),
        edge_index=torch.randint(0, n, (2, 4 * n), generator=g),
        y=torch.randint(0, NUM_CLASSES, (n,), generator=g),
        train_mask=train_mask,
    )


def _graph_batch(n_graphs: int = 4) -> Batch:
    graphs = []
    for i in range(n_graphs):
        d = _graph(n=10, seed=i)
        graphs.append(Data(x=d.x, edge_index=d.edge_index, y=torch.tensor([i % NUM_CLASSES])))
    return Batch.from_data_list(graphs)


MODELS = [
    "dgi",
    "graphcl",
    "vicreg",
    "barlow_twins",
    "bgrl",
    pytest.param("afgrl", marks=pytest.mark.skipif(not HAS_FAISS, reason="needs faiss-cpu")),
    "graphdino",
    "supervised",
]


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("model_name", MODELS)
def test_node_level_builds_and_trains(model_name, encoder):
    torch.manual_seed(0)
    model = build_model(
        _config(model_name, encoder, pool=False), in_channels=IN_CHANNELS, num_classes=NUM_CLASSES
    )
    data = _graph()
    model.train()
    loss = model.compute_loss(data)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    model.eval()
    assert model(data).shape == (data.num_nodes, HIDDEN)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("model_name", MODELS)
def test_graph_level_builds_and_trains(model_name, encoder):
    """forward() must return one embedding per graph: evaluation pairs it with graph labels."""
    torch.manual_seed(0)
    model = build_model(
        _config(model_name, encoder, pool=True), in_channels=IN_CHANNELS, num_classes=NUM_CLASSES
    )
    batch = _graph_batch()
    model.train()
    loss = model.compute_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    model.eval()
    assert model(batch).shape == (batch.num_graphs, HIDDEN)


# ---------------------------------------------------------------------------
# Edge features (molecules: one bond type per edge)
# ---------------------------------------------------------------------------

EDGE_AWARE_ENCODERS = ["gin", "transformer"]
N_ATOM_TYPES, N_BOND_TYPES = 5, 3


def _molecule_batch(n_graphs: int = 4, n: int = 8) -> Batch:
    """ZINC-like graphs: an integer atom type per node, an integer bond type per edge."""
    g = torch.Generator().manual_seed(0)
    graphs = [
        Data(
            x=torch.randint(0, N_ATOM_TYPES, (n, 1), generator=g),
            edge_index=torch.randint(0, n, (2, 3 * n), generator=g),
            edge_attr=torch.randint(0, N_BOND_TYPES, (3 * n,), generator=g),
            y=torch.tensor([i % NUM_CLASSES]),
        )
        for i in range(n_graphs)
    ]
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("encoder", EDGE_AWARE_ENCODERS)
@pytest.mark.parametrize("model_name", MODELS)
def test_edge_features_reach_the_encoder(model_name, encoder):
    """Bond types must reach the encoder, in training and in forward().

    Every model used to call its encoder without ``edge_attr``; an edge-aware encoder then
    ran on zeros, so the bond types of a molecule were ignored without any error.
    """
    torch.manual_seed(0)
    cfg = _config(model_name, encoder, pool=True)
    cfg["encoder"].update(
        node_emb_num_classes=N_ATOM_TYPES,
        edge_dim=HIDDEN,
        edge_emb_num_classes=N_BOND_TYPES,
        drop=0.0,
    )
    model = build_model(cfg, in_channels=1, num_classes=NUM_CLASSES)
    batch = _molecule_batch()

    got_edge_attr = []

    def record(module, args, kwargs):
        got_edge_attr.append(kwargs.get("edge_attr") is not None or len(args) > 3)

    encoders = [m for m in model.modules() if hasattr(m, "edge_proj")]
    assert encoders
    for enc in encoders:
        enc.register_forward_pre_hook(record, with_kwargs=True)

    model.train()
    assert torch.isfinite(model.compute_loss(batch))
    assert got_edge_attr and all(got_edge_attr)

    model.eval()
    other_bonds = batch.clone()
    other_bonds.edge_attr = (batch.edge_attr + 1) % N_BOND_TYPES
    assert not torch.allclose(model(batch), model(other_bonds))


# ---------------------------------------------------------------------------
# Graph readout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readout", ["mean", "sum", "max"])
@pytest.mark.parametrize("model_name", MODELS)
def test_graph_readout(model_name, readout):
    """Graph embeddings are the configured readout of the encoder's node embeddings."""
    torch.manual_seed(0)
    cfg = _config(model_name, "gin", pool=True)
    cfg["encoder"]["readout"] = readout
    model = build_model(cfg, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    batch = _graph_batch()

    model.train()
    assert torch.isfinite(model.compute_loss(batch))

    model.eval()
    with torch.no_grad():
        nodes = encode(pretrained_encoder(model), batch)
        expected = pool_graph_embeddings(nodes, batch.batch, readout)
    assert torch.allclose(model(batch), expected, atol=1e-6)


def test_readouts_differ_and_sum_sees_the_graph_size():
    x = torch.tensor([[1.0, 4.0], [3.0, 0.0], [5.0, 5.0]])
    batch = torch.tensor([0, 0, 1])
    assert pool_graph_embeddings(x, batch, "mean").tolist() == [[2.0, 2.0], [5.0, 5.0]]
    assert pool_graph_embeddings(x, batch, "sum").tolist() == [[4.0, 4.0], [5.0, 5.0]]
    assert pool_graph_embeddings(x, batch, "max").tolist() == [[3.0, 4.0], [5.0, 5.0]]
    assert pool_graph_embeddings(x, None, "sum").tolist() == [[9.0, 9.0]]  # one graph
    with pytest.raises(ValueError, match="readout"):
        pool_graph_embeddings(x, batch, "median")
    with pytest.raises(ValueError, match="readout"):
        build_model(
            {"name": "dgi", "encoder": {**_config("dgi", "gin", True)["encoder"], "readout": "x"}},
            in_channels=IN_CHANNELS,
        )
