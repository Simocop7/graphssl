from __future__ import annotations
from typing import List, Optional
import torch
from torch_geometric.data import Data

from graphssl.registry import AUGMENTS


def compose(data: Data, augments: list, protected_nodes: Optional[torch.Tensor] = None) -> Data:
    """Apply a sequence of registry-based augmentations to a Data object.

    Args:
        augments: List of (name, kwargs) tuples referencing AUGMENTS registry.
        protected_nodes: Optional tensor of node indices that must not be removed
            or dropped. Passed to every augmentation function; functions that do
            not use it accept and ignore the parameter.
    """
    for name, kwargs in augments:
        data = AUGMENTS.build(name, data=data, protected_nodes=protected_nodes, **kwargs)
    return data


class MultiView:
    """Generate n independent augmented views of a Data object.

    Supports both registry-style augments (list of (name, kwargs) tuples) and
    class-based transforms from augmentation/transforms.py (callables Data → Data).

    Args:
        transforms: Either a list of (name, kwargs) tuples OR a list of callables.
        n_views: Number of independent views to generate.
    """

    def __init__(self, transforms: list, n_views: int = 2):
        self.transforms = transforms
        self.n_views = n_views
        # Detect which API is being used
        self._registry_style = (
            transforms
            and isinstance(transforms[0], tuple)
            and isinstance(transforms[0][0], str)
        )

    def __call__(
        self,
        data: Data,
        protected_nodes: Optional[torch.Tensor] = None,
    ) -> List[Data]:
        if self._registry_style:
            return [
                compose(data, self.transforms, protected_nodes=protected_nodes)
                for _ in range(self.n_views)
            ]
        views = []
        for _ in range(self.n_views):
            view = data
            for t in self.transforms:
                view = t(view)
            views.append(view)
        return views
