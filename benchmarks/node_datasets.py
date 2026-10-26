"""Node-classification datasets of the factor study, loaded the same way by every script.

- Planetoid (Cora, CiteSeer, PubMed): the public split.
- Amazon Computers / Photo, Coauthor CS: no public split. Random split 0 (10/10/80, the
  convention of GRACE, PyGCL and BGRL) stands in for it and is written into the masks, so the
  supervised references train on it.
- ogbn-arxiv, ogbn-products: the official split. ogbn-arxiv is made undirected, as in the
  stress test and the BGRL reproduction; ogbn-products already is.
- arxiv-year (Lim et al., NeurIPS 2021): the ogbn-arxiv graph and features, labelled with the
  publication year cut into 5 classes at its quantiles (their ``even_quantile_labels``), with
  random 50/25/25 splits. SSL training never sees labels, so the models trained on ogbn-arxiv
  are scored on it without retraining.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

PLANETOID = ["Cora", "CiteSeer", "PubMed"]
# Name in the results -> (PyG dataset class, its name argument).
RANDOM_SPLIT = {
    "AmazonComputers": ("Amazon", "Computers"),
    "AmazonPhoto": ("Amazon", "Photo"),
    "CoauthorCS": ("Coauthor", "CS"),
}
OGB = ["ogbn-arxiv", "ogbn-products"]
ARXIV_YEAR = "arxiv-year"
FULL_BATCH_DATASETS = [*PLANETOID, *RANDOM_SPLIT]
ALL_DATASETS = [*FULL_BATCH_DATASETS, *OGB, ARXIV_YEAR]

DEFAULT_SPLIT_FRACTIONS = (0.1, 0.1)
# Train and validation fractions of a random split, where a dataset's convention differs.
SPLIT_FRACTIONS = {ARXIV_YEAR: (0.5, 0.25)}
ARXIV_YEAR_CLASSES = 5


def split_fractions(name: str) -> Tuple[float, float]:
    """(train, validation) fractions of the random splits of ``name``; the rest is test."""
    return SPLIT_FRACTIONS.get(name, DEFAULT_SPLIT_FRACTIONS)


def random_split(num_nodes: int, fractions: Sequence[float], seed: int) -> tuple:
    """(train_idx, val_idx, test_idx): a random permutation of the nodes cut by ``fractions``."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    n_train = int(round(fractions[0] * num_nodes))
    n_val = int(round(fractions[1] * num_nodes))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def even_quantile_labels(values: np.ndarray, num_classes: int) -> np.ndarray:
    """Class k for values in [q_k, q_{k+1}), q the quantiles; as in Lim et al.'s code."""
    labels = -np.ones(values.shape[0], dtype=np.int64)
    lower = -np.inf
    for k in range(num_classes - 1):
        upper = np.nanquantile(values, (k + 1) / num_classes)
        labels[(values >= lower) & (values < upper)] = k
        lower = upper
    labels[values >= lower] = num_classes - 1
    return labels


def _masks(num_nodes: int, splits: List[Tensor]) -> List[Tensor]:
    masks = []
    for idx in splits:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        masks.append(mask)
    return masks


def _ogb(name: str, data_dir: str):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError('Install ogb: pip install "graphssl[benchmark]"') from e

    # ogb (<= 1.3.6) loads its processed file with a bare torch.load(); since PyTorch 2.6
    # that defaults to weights_only=True and rejects PyG's Data classes unless allow-listed.
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return PygNodePropPredDataset(name=name, root=f"{data_dir}/{name}")


def load_node_dataset(name: str, data_dir: str = "data"):
    """(data, num_classes, train_idx, val_idx, test_idx) of one dataset of the study."""
    if name in PLANETOID:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=f"{data_dir}/{name}", name=name)
        data = dataset[0]
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        val_idx = data.val_mask.nonzero(as_tuple=True)[0]
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]
        return data, dataset.num_classes, train_idx, val_idx, test_idx

    if name in RANDOM_SPLIT:
        import torch_geometric.datasets as pyg_datasets

        cls_name, pyg_name = RANDOM_SPLIT[name]
        dataset = getattr(pyg_datasets, cls_name)(root=f"{data_dir}/{name}", name=pyg_name)
        data = dataset[0]
        splits = list(random_split(data.num_nodes, split_fractions(name), seed=0))
        data.train_mask, data.val_mask, data.test_mask = _masks(data.num_nodes, splits)
        return data, dataset.num_classes, *splits

    from torch_geometric.transforms import ToUndirected

    if name in OGB:
        dataset = _ogb(name, data_dir)
        data = dataset[0]
        if name == "ogbn-arxiv":
            data = ToUndirected()(data)  # directed citations -> undirected
        data.y = data.y.view(-1)  # OGB labels are [N, 1]
        split = dataset.get_idx_split()
        return data, dataset.num_classes, split["train"], split["valid"], split["test"]

    if name == ARXIV_YEAR:
        data = ToUndirected()(_ogb("ogbn-arxiv", data_dir)[0])
        years = data.node_year.view(-1).numpy()
        data.y = torch.from_numpy(even_quantile_labels(years, ARXIV_YEAR_CLASSES))
        splits = list(random_split(data.num_nodes, split_fractions(name), seed=0))
        return data, ARXIV_YEAR_CLASSES, *splits

    raise ValueError(f"unknown dataset {name!r}; choose from {ALL_DATASETS}")
