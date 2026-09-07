from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch.nn as nn
from torch_geometric.loader import DataLoader, NeighborLoader

from graphssl.registry import LOADERS


@LOADERS.register("graph")
def build_graph_loader(*, dataset, params):
    # Standard graph-level loader, batches whole graphs together.
    return DataLoader(
        dataset,
        batch_size=params.get("batch_size", 32),
        shuffle=params.get("shuffle", True),
        num_workers=params.get("num_workers", 0),
    )


@LOADERS.register("neighbor")
def build_neighbor_loader(*, dataset, params):
    # Samples neighbor subgraphs per node; useful for large single-graph datasets.
    return NeighborLoader(
        dataset,
        num_neighbors=params.get("num_neighbors", [10, 10]),
        batch_size=params.get("batch_size", 1024),
        shuffle=True,
    )


# ---------------------------------------------------------------------------
# Config loading and model construction
# ---------------------------------------------------------------------------


def load_config(path: Union[str, Path]) -> dict:
    """Load a GraphSSL YAML config file and return it as a plain dict.

    The returned dict can be passed directly to ``build_model()`` or to any
    model constructor (``ModelClass(config, in_channels)``).

    Requires PyYAML: ``pip install pyyaml``
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load config files: pip install pyyaml") from e
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(
    config: dict,
    in_channels: int,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """Instantiate a GraphSSL model from a config dict.

    ``config`` must contain a top-level ``name`` key identifying the model.
    For ``supervised``, ``num_classes`` is required. All other fields are
    forwarded to the model constructor and validated by its config dataclass.

    Example::

        cfg = load_config("configs/bgrl.yaml")
        model = build_model(cfg, in_channels=dataset.num_features)

        cfg = load_config("configs/supervised.yaml")
        model = build_model(cfg, in_channels=dataset.num_features,
                            num_classes=dataset.num_classes)

    Args:
        config: Config dict with a ``name`` key and model-specific fields.
        in_channels: Node feature dimensionality from the dataset.
        num_classes: Required only for the ``supervised`` model.

    Returns:
        Instantiated model as an ``nn.Module``.
    """
    # Deferred import to avoid circular dependency (models import from config).
    from graphssl.models import (
        AFGRL,
        BGRL,
        DGI,
        BarlowTwins,
        GraphCL,
        GraphDINO,
        Supervised,
        VICReg,
    )

    _MODELS = {
        "dgi": DGI,
        "graphcl": GraphCL,
        "vicreg": VICReg,
        "barlow_twins": BarlowTwins,
        "bgrl": BGRL,
        "afgrl": AFGRL,
        "graphdino": GraphDINO,
        "supervised": Supervised,
    }

    name = config.get("name", "").lower().replace("-", "_")
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_MODELS)}")
    cls = _MODELS[name]
    if name == "supervised":
        if num_classes is None:
            raise ValueError("num_classes is required for the Supervised model")
        return cls(config, in_channels, num_classes)
    return cls(config, in_channels)
