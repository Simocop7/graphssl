import torch
from torch import nn


@torch.no_grad()
def update_ema_params(student: nn.Module, teacher: nn.Module, tau: float) -> None:
    """EMA update of teacher parameters from the student.

    Buffers (e.g. BatchNorm running stats) are copied directly, not averaged.
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)
