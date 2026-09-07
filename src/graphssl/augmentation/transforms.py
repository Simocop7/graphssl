"""Class-based graph augmentation transforms (torchvision style).

Each transform is a callable Data → Data that wraps the functional API.
They satisfy the BaseAugmentation Protocol and can be used with Compose/MultiView.
"""

from torch_geometric.data import Data

from graphssl.augmentation import functional as F


class EdgeDrop:
    """Randomly drop edges with probability p."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, data: Data) -> Data:
        return F.edge_drop(data, p=self.p)


class EdgeAdd:
    """Randomly add edges (fraction p of existing edges)."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data: Data) -> Data:
        return F.edge_add(data, p=self.p)


class Subgraph:
    """Keep a k-hop subgraph around a random seed node."""

    def __init__(self, num_hops: int = 2):
        self.num_hops = num_hops

    def __call__(self, data: Data) -> Data:
        return F.subgraph(data, num_hops=self.num_hops)


class FeatMask:
    """Zero-out a fraction p of feature dimensions."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, data: Data) -> Data:
        return F.feat_mask(data, p=self.p)


class FeatNoise:
    """Add Gaussian noise with standard deviation std to node features."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, data: Data) -> Data:
        return F.feat_noise(data, std=self.std)


class FeatShuffle:
    """Randomly swap features between a fraction p of nodes."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data: Data) -> Data:
        return F.feat_shuffle(data, p=self.p)


class NodeDrop:
    """Drop nodes with probability p, remapping edge_index.

    Note: when used with protected_nodes (seed nodes in mini-batch training),
    use the registry-style compose() API which propagates protected_nodes
    automatically. The class-based API does not support protected_nodes.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data: Data) -> Data:
        return F.node_drop(data, p=self.p)
