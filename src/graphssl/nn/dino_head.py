import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

from graphssl.registry import HEADS


@HEADS.register("dino")
class DINOHead(nn.Module):
    """Projection head for GraphDINO (student and teacher share this architecture).

    Args:
        hidden_dim: Encoder output dimension.
        proj_hidden: Hidden dim of the 2-layer projector MLP.
        bottleneck_dim: Bottleneck dim fed to the prototype layer.
        n_prototypes: Number of prototype (output) dimensions.
        student_temp: Temperature for student log-softmax.
        teacher_temp: Final teacher temperature (after warmup).
        center_momentum: EMA momentum for center update.
        warmup_teacher_temp: Initial teacher temperature. Linearly interpolated
            to teacher_temp over warmup_teacher_temp_epochs epochs.
        warmup_teacher_temp_epochs: Number of epochs for the linear warmup.
            Set to 0 to disable warmup (use teacher_temp from epoch 0).
    """

    def __init__(
        self,
        hidden_dim: int,
        proj_hidden: int,
        bottleneck_dim: int,
        n_prototypes: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs

        # Start at warmup_teacher_temp if warmup is active, else jump straight to teacher_temp.
        self._current_teacher_temp: float = (
            warmup_teacher_temp if warmup_teacher_temp_epochs > 0 else teacher_temp
        )

        self.projector = Sequential(
            Linear(hidden_dim, proj_hidden),
            BN(proj_hidden),
            ReLU(),
            Linear(proj_hidden, bottleneck_dim),
        )

        # weight_norm keeps prototype rows unit-norm; using parametrizations API
        # since the old weight_norm is deprecated and breaks deepcopy.
        proto_linear = Linear(bottleneck_dim, n_prototypes, bias=False)
        self.proto = nn.utils.parametrizations.weight_norm(proto_linear, dim=0)

        self.center: torch.Tensor
        self.register_buffer("center", torch.zeros(1, n_prototypes))

    def set_epoch(self, epoch: int) -> None:
        """Update the effective teacher temperature for the given epoch.

        Linear warmup from warmup_teacher_temp (epoch 0) to teacher_temp
        (epoch >= warmup_teacher_temp_epochs). Call this at the start of each epoch.
        """
        if self.warmup_teacher_temp_epochs <= 0 or epoch >= self.warmup_teacher_temp_epochs:
            self._current_teacher_temp = self.teacher_temp
        else:
            frac = epoch / self.warmup_teacher_temp_epochs
            self._current_teacher_temp = self.warmup_teacher_temp + frac * (
                self.teacher_temp - self.warmup_teacher_temp
            )

    def forward(self, x: torch.Tensor, use_teacher_temp: bool = False) -> torch.Tensor:
        h = self.projector(x)
        h = F.normalize(h, dim=-1)
        logits = self.proto(h)

        if use_teacher_temp:
            return F.softmax((logits - self.center) / self._current_teacher_temp, dim=-1)
        else:
            return F.log_softmax(logits / self.student_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        self.center = self.center * self.center_momentum + teacher_out.mean(0, keepdim=True) * (
            1 - self.center_momentum
        )

    def cancel_last_layer_gradients(self) -> None:
        """Zero gradients of the prototype (last) layer. Call after backward()."""
        for p in self.proto.parameters():
            p.grad = None
