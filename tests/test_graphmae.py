"""GraphMAE: masking, re-masking, the scaled cosine error and the loss on masked nodes only."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from graphssl.config.load import build_model
from graphssl.config.schema import GraphMAEConfig
from graphssl.core import encode
from graphssl.losses import SCELoss

HIDDEN, IN_CHANNELS = 16, 6


def _config(encoder: str = "gcn", **kwargs) -> dict:
    return {
        "name": "graphmae",
        "encoder": {"name": encoder, "hidden_dim": HIDDEN, "num_layers": 2, "pool": False},
        **kwargs,
    }


def _graph(n: int = 40, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(n, IN_CHANNELS, generator=g),
        edge_index=torch.randint(0, n, (2, 4 * n), generator=g),
    )


# ------------------------------------------------------------------ config and loss


@pytest.mark.parametrize(
    "field, value",
    [
        ("mask_rate", 0.0),
        ("mask_rate", 1.0),
        ("replace_rate", -0.1),
        ("replace_rate", 1.0),
        ("alpha", 0.5),
    ],  # fmt: skip
)
def test_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValueError, match=field):
        GraphMAEConfig.from_dict(_config(**{field: value}))


def test_config_defaults_are_the_reference_cora_values():
    cfg = GraphMAEConfig.from_dict(_config())
    assert (cfg.mask_rate, cfg.replace_rate, cfg.alpha) == (0.5, 0.05, 3.0)


def test_categorical_node_features_are_refused():
    cfg = _config("gin")
    cfg["encoder"]["node_emb_num_classes"] = 5
    with pytest.raises(ValueError, match="categorical"):
        build_model(cfg, in_channels=1)


def test_sce_values():
    x = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    same_direction = torch.tensor([[3.0, 0.0], [0.5, 0.5], [0.0, 0.1]])
    orthogonal = torch.tensor([[0.0, 1.0], [1.0, -1.0], [5.0, 0.0]])
    loss = SCELoss(alpha=3.0)
    assert loss(same_direction, x).item() == pytest.approx(0.0, abs=1e-6)  # scale-invariant
    assert loss(orthogonal, x).item() == pytest.approx(1.0)
    # (1 - cos)^alpha: the exponent shrinks small errors more than large ones
    rec = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])
    one_minus_cos = 1 - F.cosine_similarity(rec, target).item()
    assert SCELoss(alpha=1.0)(rec, target).item() == pytest.approx(one_minus_cos)
    assert SCELoss(alpha=2.0)(rec, target).item() == pytest.approx(one_minus_cos**2)


def test_sce_rejects_alpha_below_one():
    with pytest.raises(ValueError, match="alpha"):
        SCELoss(alpha=0.9)


# ------------------------------------------------------------------ masking


def test_masked_nodes_full_batch():
    model = build_model(_config(mask_rate=0.5), in_channels=IN_CHANNELS)
    masked = model._masked_nodes(40, None, "cpu")
    assert masked.numel() == 20 and masked.unique().numel() == 20
    # at least one node on any graph (the reference masks int(0.5 * 1) = 0 nodes)
    assert model._masked_nodes(1, None, "cpu").numel() == 1


def test_masked_nodes_mini_batch_masks_seeds_and_neighbors_in_proportion():
    model = build_model(_config(mask_rate=0.5), in_channels=IN_CHANNELS)
    masked = model._masked_nodes(100, 10, "cpu")
    seeds = masked[masked < 10]
    assert seeds.numel() == 5 and (masked >= 10).sum() == 45
    assert masked.unique().numel() == masked.numel()
    assert model._masked_nodes(100, 1, "cpu")[0] == 0  # one seed: masked, never zero seeds


def test_corruption_replaces_only_masked_rows():
    torch.manual_seed(0)
    model = build_model(_config(replace_rate=0.25), in_channels=IN_CHANNELS)
    with torch.no_grad():
        model.mask_token.copy_(torch.full((1, IN_CHANNELS), 7.0))
    x = torch.randn(40, IN_CHANNELS)
    masked = model._masked_nodes(40, None, "cpu")
    x_in = model._corrupt(x, masked)
    keep = torch.ones(40, dtype=torch.bool)
    keep[masked] = False
    assert torch.equal(x_in[keep], x[keep])
    is_token = (x_in[masked] == 7.0).all(dim=1)
    assert is_token.sum() == masked.numel() - int(0.25 * masked.numel())
    # the others carry the features of some node of the graph
    for row in x_in[masked][~is_token]:
        assert (x == row).all(dim=1).any()


def test_no_replacement_masks_every_masked_node_with_the_token():
    model = build_model(_config(replace_rate=0.0), in_channels=IN_CHANNELS)
    with torch.no_grad():
        model.mask_token.copy_(torch.full((1, IN_CHANNELS), 7.0))
    masked = model._masked_nodes(40, None, "cpu")
    assert (model._corrupt(torch.randn(40, IN_CHANNELS), masked)[masked] == 7.0).all()


# ------------------------------------------------------------------ the training step


def _capture(module, store):
    def hook(_module, args):
        store.append([a.detach().clone() if torch.is_tensor(a) else a for a in args])

    return module.register_forward_pre_hook(hook)


def test_decoder_sees_masked_nodes_re_masked_and_loss_uses_masked_nodes_only():
    torch.manual_seed(0)
    model = build_model(_config(), in_channels=IN_CHANNELS)
    data = _graph()
    decoder_inputs, loss_inputs = [], []
    _capture(model.decoder, decoder_inputs)
    _capture(model.loss_fn, loss_inputs)
    torch.manual_seed(1)
    masked = model._masked_nodes(data.num_nodes, None, "cpu")
    torch.manual_seed(1)
    model.compute_loss(data)
    rep = decoder_inputs[0][0]
    assert (rep[masked] == 0).all() and (rep.abs().sum(1) > 0).sum() == data.num_nodes - 20
    reconstruction, target = loss_inputs[0]
    assert reconstruction.shape == target.shape == (masked.numel(), IN_CHANNELS)
    assert torch.equal(target, data.x[masked])  # the clean features, not the corrupted ones


def test_mini_batch_loss_uses_masked_seeds_only():
    torch.manual_seed(0)
    model = build_model(_config(), in_channels=IN_CHANNELS)
    data = _graph(n=60)
    data.batch_size = 12
    loss_inputs = []
    _capture(model.loss_fn, loss_inputs)
    assert torch.isfinite(model.compute_loss(data))
    assert loss_inputs[0][0].shape[0] == 6  # half of the 12 seeds


def test_every_part_receives_gradients():
    torch.manual_seed(0)
    model = build_model(_config(), in_channels=IN_CHANNELS)
    model.compute_loss(_graph()).backward()
    for name in ("mask_token", "encoder_to_decoder.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert all(p.grad is not None for p in model.decoder.parameters())
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_forward_embeds_the_clean_graph():
    torch.manual_seed(0)
    model = build_model(_config(), in_channels=IN_CHANNELS)
    model.eval()
    data = _graph()
    assert torch.equal(model(data), encode(model.encoder, data))
    assert torch.equal(model(data), model(data))  # no masking at evaluation


@pytest.mark.parametrize("encoder", ["gcn", "gin", "transformer"])
def test_training_lowers_the_reconstruction_error(encoder):
    """On a graph whose features follow its communities, the masked features can be rebuilt
    from the neighbors', and a few hundred steps must lower the error clearly."""
    torch.manual_seed(0)
    n, communities = 60, 3
    labels = torch.arange(n) % communities
    centers = torch.randn(communities, IN_CHANNELS) * 3
    x = centers[labels] + 0.3 * torch.randn(n, IN_CHANNELS)
    src, dst = torch.randint(0, n, (2, 2000))
    keep = labels[src] == labels[dst]
    data = Data(x=x, edge_index=torch.stack([src[keep], dst[keep]]))
    model = build_model(_config(encoder), in_channels=IN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def mean_loss(k: int = 20) -> float:
        with torch.no_grad():
            return sum(model.compute_loss(data).item() for _ in range(k)) / k

    before = mean_loss()
    for _ in range(300):
        optimizer.zero_grad()
        model.compute_loss(data).backward()
        optimizer.step()
    assert mean_loss() < 0.5 * before


# ------------------------------------------------------------------ the decoder


def test_gat_decoder_is_gatconv_with_one_head():
    """Same function as PyG's GATConv(heads=1), outputs and gradients, with self-loops already
    present on some nodes and nodes without any edge."""
    from torch_geometric.nn import GATConv

    from graphssl.nn import GATDecoder

    torch.manual_seed(0)
    n, d_in, d_out = 30, 8, 50
    x = torch.randn(n, d_in, requires_grad=True)
    edge_index = torch.randint(0, n - 5, (2, 90))  # the last five nodes have no edge
    edge_index = torch.cat([edge_index, torch.tensor([[0, 3], [0, 3]])], dim=1)  # self-loops
    reference = GATConv(d_in, d_out, heads=1)
    ours = GATDecoder(d_in, d_out)
    with torch.no_grad():
        reference.bias.normal_()
        ours.lin.weight.copy_(reference.lin.weight)
        ours.att_src.copy_(reference.att_src.view(1, -1))
        ours.att_dst.copy_(reference.att_dst.view(1, -1))
        ours.bias.copy_(reference.bias)
    out_ref, out = reference(x, edge_index), ours(x, edge_index)
    assert torch.allclose(out, out_ref, atol=1e-5)
    (grad_ref,) = torch.autograd.grad(out_ref.square().sum(), x)
    (grad,) = torch.autograd.grad(out.square().sum(), x)
    assert torch.allclose(grad, grad_ref, atol=1e-4)
