"""Pure-Python DataModule (no PyTorch Lightning)."""

from __future__ import annotations
from typing import List, Optional
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, NeighborLoader


class DataModule:
    """Wraps a dataset and exposes train/eval DataLoaders.

    Node-level task (is_graph_level=False):
        - ``data``: single PyG Data object with train_mask / val_mask / test_mask
          encoded as index tensors (train_idx, val_idx, test_idx).
        - ``train_dataloader()`` returns a single-element iterable yielding the
          raw full-graph ``Data`` object (not routed through PyG's ``Batch``
          collation, so no spurious ``.batch_size`` is attached).
        - ``neighbor_loader()`` returns a NeighborLoader for mini-batch training.

    Graph-level task (is_graph_level=True):
        - ``dataset_obj``: a PyG Dataset or list of Data objects.
        - ``train_dataloader()`` / ``eval_dataloader()`` iterate over the dataset.

    Attributes exposed for evaluation utilities:
        data, labels, train_idx, val_idx, test_idx,
        is_graph_level, batch_size, dataset_obj.
    """

    def __init__(
        self,
        data: Optional[Data] = None,
        labels: Optional[torch.Tensor] = None,
        train_idx: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None,
        test_idx: Optional[torch.Tensor] = None,
        is_graph_level: bool = False,
        batch_size: int = 32,
        dataset_obj: Optional[Dataset] = None,
        num_workers: int = 0,
    ):
        self.data = data
        self.labels = labels
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.is_graph_level = is_graph_level
        self.batch_size = batch_size
        self.dataset_obj = dataset_obj
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def train_dataloader(self, **kwargs):
        if self.is_graph_level:
            return DataLoader(
                self.dataset_obj,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                **kwargs,
            )
        # Node-level full batch: yield the raw Data object directly rather than
        # routing it through DataLoader/Batch.from_data_list(), which would attach
        # a spurious `.batch_size` attribute (PyG's "number of graphs in this
        # batch", always 1 here) that models mistake for NeighborLoader's
        # "number of seed nodes", silently cropping the graph down to 1 node.
        return [self.data]

    def eval_dataloader(self, **kwargs):
        if self.is_graph_level:
            return DataLoader(
                self.dataset_obj,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                **kwargs,
            )
        return [self.data]

    def neighbor_loader(
        self,
        num_neighbors: List[int],
        input_nodes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NeighborLoader:
        """Mini-batch NeighborLoader for node-level training."""
        assert self.data is not None, "data must be set for node-level mini-batch loading"
        return NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )
