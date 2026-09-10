# Augmentation System

A torchvision-style composable system in `augmentation/`:

- `functional.py` — pure functions: `(data, *, protected_nodes=None, **kwargs) -> Data`
- `transforms.py` — callable classes wrapping those functions
- `compose.py` — `compose(data, aug_list, protected_nodes=None)` applies a list in sequence

## Available operations

| Name | Parameters | Description |
|---|---|---|
| `edge_drop` | `p` | drops edges with probability `p` |
| `edge_add` | `p` | adds random edges (fraction `p` of existing ones) |
| `feat_mask` | `p` | masks node features with probability `p` |
| `feat_noise` | `std` | adds Gaussian noise to features |
| `feat_shuffle` | `p` | swaps features between random nodes |
| `subgraph` | `num_hops` | extracts a k-hop subgraph from a random seed |
| `node_drop` | `p` | drops nodes; `protected_nodes` are never removed |

## Two APIs

**Class-based** (torchvision style), used with `MultiView`:

```python
from graphssl.augmentation import EdgeDrop
view = EdgeDrop(p=0.2)(data)
```

**Registry-based**, used internally by every model's `compute_loss()` (supports
`protected_nodes`):

```python
from graphssl.augmentation import compose
view = compose(data, [("edge_drop", {"p": 0.2})])
```

`MultiView(transforms, n_views)` accepts either API and generates `n_views` independent views.

## Protected nodes

In mini-batch node training, `batch.batch_size` gives the seed-node count, and the loss must
only be computed on `z[:batch.batch_size]`. If augmentation is allowed to drop a seed node, the
loss computation silently breaks (index mismatch or a seed node that no longer exists).

`compose()` accepts `protected_nodes=torch.arange(batch.batch_size)` and propagates it to every
augmentation in the list; `node_drop` is the one operation that consults it, to guarantee seed
nodes are never removed. Every model that supports mini-batch training constructs this
argument automatically — you don't need to pass it yourself unless you're calling `compose()`
directly.

!!! note "Current granularity"
    Protected-node enforcement is active for `node_drop` (the only node-destructive
    augmentation). Per-augmentation protection for feature-level transforms (e.g. exempting
    seed nodes from `feat_mask`) isn't implemented yet.

## Reference

See the [Encoders/Augmentation source](https://github.com/Simocop7/graphssl/tree/main/src/graphssl/augmentation)
for the full function signatures.
