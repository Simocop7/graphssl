# Loss Functions

Every loss is a standalone `nn.Module` in `losses/`, independently testable and usable outside
any model — registered in the `LOSSES` registry via `@LOSSES.register("name")`.

| Registry name | Class | Used by |
|---|---|---|
| `nt_xent` | `NTXentLoss` | [GraphCL](models/graphcl.md) |
| `vicreg` | `VICRegLoss` | [VICReg](models/vicreg.md) |
| `barlow_twins` | `BarlowTwinsLoss` | [Barlow Twins](models/barlow-twins.md) |
| `cosine_regression` | `CosineRegressionLoss` | [BGRL](models/bgrl.md), [AFGRL](models/afgrl.md) |
| — | `DINOLoss` | [GraphDINO](models/graphdino.md) |

## Combining losses

`CombinedLoss` mixes and weights any set of registered losses without touching model code:

```python
from graphssl.losses import CombinedLoss

fn = CombinedLoss.from_config([
    {"name": "nt_xent", "weight": 0.7, "tau": 0.5},
    {"name": "vicreg",  "weight": 0.3, "invariance": 25.0, "variance": 25.0, "covariance": 1.0},
])
loss = fn(z1, z2)
```

Or programmatically:

```python
from graphssl.losses import CombinedLoss, NTXentLoss, VICRegLoss

fn = CombinedLoss([(NTXentLoss(tau=0.5), 0.7), (VICRegLoss(), 0.3)])
```

## Reference

See the [Losses API](api/losses.md) for full signatures and the formula for each loss (also
documented on each model's own page).
