"""Barlow Twins cross-correlation loss."""

from torch import Tensor, nn

from graphssl.registry import LOSSES


@LOSSES.register("barlow_twins")
class BarlowTwinsLoss(nn.Module):
    """Cross-correlation matrix loss.

    On-diagonal terms are pushed to 1; off-diagonal terms are pushed to 0.

    Args:
        lambda_param: Weight for off-diagonal terms. Defaults to 1/out_dim.
        out_dim: Projection dimension used to compute the default lambda.
    """

    def __init__(self, lambda_param: float | None = None, out_dim: int = 256):
        super().__init__()
        self.lambda_param = lambda_param if lambda_param is not None else 1.0 / out_dim

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        N = z1.size(0)

        # Normalise along the batch dimension
        z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
        z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)

        C = z1_n.T @ z2_n / N  # [D, D] cross-correlation matrix

        on_diag = (1 - C.diagonal()).pow(2).sum()
        off_diag = C.pow(2).sum() - C.diagonal().pow(2).sum()

        return on_diag + self.lambda_param * off_diag
