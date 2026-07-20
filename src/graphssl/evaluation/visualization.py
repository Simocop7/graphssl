"""Embedding extraction and UMAP visualization utilities."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor

try:
    import umap
    import matplotlib.pyplot as plt
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def extract_embeddings(
    model: "torch.nn.Module",
    datamodule,
    encoder_source: Literal["auto", "teacher", "student", "online", "target", "encoder"] = "auto",
    device: str = "cpu",
    mini_batch: bool = False,
    num_neighbors: Optional[List[int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Extract embeddings from a model using a DataModule.

    Three extraction paths:

    - **Graph-level** (``datamodule.is_graph_level=True``): iterates
      ``eval_dataloader()``, one embedding per graph.
    - **Node full-batch** (default, ``mini_batch=False``): single forward pass
      over the full graph stored in ``datamodule.data``.
    - **Node mini-batch** (``mini_batch=True``): ``NeighborLoader`` with
      ``global_to_local`` reassembly via ``batch.n_id``. Use this for graphs
      too large to fit in memory for a single forward pass.

    When ``encoder_source="auto"`` (default), calls ``model.forward()``
    directly. This respects each model's own evaluation convention — e.g.
    teacher encoder for GraphDINO, online encoder for BGRL/AFGRL — without
    requiring the caller to know the internal architecture.

    Pass an explicit ``encoder_source`` to bypass ``model.forward()`` and
    extract from a specific sub-encoder (useful for ablations).

    Args:
        model: Any ``BaseSSLModel`` or compatible ``nn.Module``.
        datamodule: ``DataModule`` exposing ``eval_dataloader()``, ``.data``,
            and ``neighbor_loader()`` (required for ``mini_batch=True``).
        encoder_source: ``"auto"`` calls ``model.forward()``; one of
            ``"teacher"``, ``"student"``, ``"online"``, ``"target"``,
            ``"encoder"`` selects a specific sub-encoder directly.
        device: Inference device string, e.g. ``"cuda:0"``.
        mini_batch: If ``True``, use ``NeighborLoader`` for node-level
            extraction. Ignored when ``is_graph_level=True``.
        num_neighbors: Neighbours per hop for ``NeighborLoader``. Defaults to
            ``[-1, -1]`` (all neighbours, 2 hops). Used only when
            ``mini_batch=True``.

    Returns:
        ``(embeddings, labels)`` — embeddings tensor and optional labels tensor.
        Labels are ``None`` when the dataset has no ``y`` attribute.
    """
    _device = torch.device(device)
    model = model.to(_device)
    model.eval()

    use_model_forward = (encoder_source == "auto")
    if not use_model_forward:
        _enc = _select_encoder(model, encoder_source)

    def _extract(batch) -> Tensor:
        if use_model_forward:
            return model(batch)
        return _enc(batch.x, batch.edge_index, getattr(batch, "batch", None))

    with torch.no_grad():
        if datamodule.is_graph_level:
            return _extract_graph_level(datamodule, _extract, _device)
        if mini_batch:
            return _extract_node_mini_batch(datamodule, _extract, _device, num_neighbors)
        return _extract_node_full_batch(datamodule, _extract, _device)


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_graph_level(datamodule, extract_fn, device) -> Tuple[Tensor, Optional[Tensor]]:
    emb_list: list[Tensor] = []
    lbl_list: list[Tensor] = []
    for batch in datamodule.eval_dataloader():
        batch = batch.to(device)
        emb_list.append(extract_fn(batch).cpu())
        if hasattr(batch, "y") and batch.y is not None:
            lbl_list.append(batch.y.cpu())
    embeddings = torch.cat(emb_list, dim=0)
    labels = torch.cat(lbl_list, dim=0) if lbl_list else None
    return embeddings, labels


def _extract_node_full_batch(datamodule, extract_fn, device) -> Tuple[Tensor, Optional[Tensor]]:
    data = datamodule.data.to(device)
    embeddings = extract_fn(data).cpu()
    labels = data.y.cpu() if hasattr(data, "y") and data.y is not None else None
    return embeddings, labels


def _extract_node_mini_batch(
    datamodule,
    extract_fn,
    device,
    num_neighbors: Optional[List[int]],
) -> Tuple[Tensor, Optional[Tensor]]:
    """NeighborLoader extraction with global position reassembly via batch.n_id.

    Each batch from NeighborLoader contains seed nodes at local positions
    ``[0, batch.batch_size)`` with their global IDs in ``batch.n_id``.
    Neighbour nodes beyond ``batch_size`` exist only for message passing and
    are discarded here.
    """
    data = datamodule.data
    n_nodes = data.x.shape[0]
    _nbrs = num_neighbors if num_neighbors is not None else [-1, -1]
    loader = datamodule.neighbor_loader(num_neighbors=_nbrs)

    store: Optional[Tensor] = None
    for batch in loader:
        batch = batch.to(device)
        h = extract_fn(batch)
        # Crop to seed nodes — neighbours are used for message passing only.
        seed_h = h[: batch.batch_size].cpu()
        seed_ids = batch.n_id[: batch.batch_size]
        if store is None:
            store = torch.zeros(n_nodes, seed_h.shape[1])
        store[seed_ids] = seed_h

    embeddings = store if store is not None else torch.empty(0)
    labels = data.y.cpu() if hasattr(data, "y") and data.y is not None else None
    return embeddings, labels


# ---------------------------------------------------------------------------
# Encoder selector (used when encoder_source != "auto")
# ---------------------------------------------------------------------------

def _select_encoder(model: "torch.nn.Module", source: str) -> "torch.nn.Module":
    if source in ("student", "online"):
        for attr in ("student_enc", "online_enc"):
            if hasattr(model, attr):
                return getattr(model, attr)
    if source in ("teacher", "target"):
        for attr in ("teacher_enc", "target_enc"):
            if hasattr(model, attr):
                return getattr(model, attr)
    if source == "encoder" and hasattr(model, "encoder"):
        return model.encoder
    return model


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_embeddings(
    embeddings: Tensor,
    labels: Optional[Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Embeddings",
) -> None:
    """UMAP 2-D scatter plot of embeddings.

    Requires optional dependencies: ``pip install umap-learn matplotlib``
    """
    if not HAS_VIZ:
        raise ImportError(
            "Visualization requires optional deps: pip install umap-learn matplotlib"
        )

    reducer = umap.UMAP(n_components=2)
    z2d = reducer.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(8, 8))
    if labels is not None:
        scatter = plt.scatter(
            z2d[:, 0], z2d[:, 1],
            c=labels.cpu().numpy(), cmap="tab10", s=3, alpha=0.7,
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(z2d[:, 0], z2d[:, 1], s=3, alpha=0.7)

    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
