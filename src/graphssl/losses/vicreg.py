"""VICReg: Variance-Invariance-Covariance Regularization loss."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphssl.registry import LOSSES


@LOSSES.register("vicreg")
class VICRegLoss(nn.Module):
    """Triple loss: invariance + variance + covariance.

    Args:
        invariance: Weight for the MSE invariance term (default 25.0).
        variance: Weight for the per-dimension variance term (default 25.0).
        covariance: Weight for the off-diagonal covariance term (default 1.0).
        gamma: Target std per dimension for the variance term (default 1.0).
    """

    def __init__(
        self,
        invariance: float = 25.0,
        variance: float = 25.0,
        covariance: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.inv_w = invariance
        self.var_w = variance
        self.cov_w = covariance
        self.gamma = gamma

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        N, D = z1.shape

        # 1. Invariance: MSE(z1, z2)
        inv_loss = F.mse_loss(z1, z2)

        # 2. Variance: push each dimension's std toward gamma
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = F.relu(self.gamma - std_z1).mean() / 2 + F.relu(self.gamma - std_z2).mean() / 2

        # 3. Covariance: penalise off-diagonal entries
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        C1 = (z1_c.T @ z1_c) / (N - 1)
        C2 = (z2_c.T @ z2_c) / (N - 1)

        def off_diag_sq(C: Tensor) -> Tensor:
            return (C.pow(2).sum() - C.diagonal().pow(2).sum()) / D

        cov_loss = off_diag_sq(C1) + off_diag_sq(C2)

        return self.inv_w * inv_loss + self.var_w * var_loss + self.cov_w * cov_loss
