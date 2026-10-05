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
        norm_last_layer: Fix the weight-norm scale ``g`` of every prototype at 1, so the
            logits are cosine similarities in [-1, 1] (DINO's ``norm_last_layer``). When
            False, ``g`` is trained, and the network can sharpen its outputs by growing the
            prototype norms.
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
        norm_last_layer: bool = False,
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

        # weight_norm splits each prototype row into a direction and a trainable scale g
        # (parametrizations API: the old weight_norm is deprecated and breaks deepcopy).
        # With norm_last_layer, g is fixed at 1 as in DINO.
        proto_linear = Linear(bottleneck_dim, n_prototypes, bias=False)
        self.proto = nn.utils.parametrizations.weight_norm(proto_linear, dim=0)
        if norm_last_layer:
            scale = self.proto.parametrizations.weight.original0
            with torch.no_grad():
                scale.fill_(1.0)
            scale.requires_grad_(False)

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

    def prototype_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw prototype scores, before centering, temperature and (log-)softmax."""
        h = self.projector(x)
        h = F.normalize(h, dim=-1)
        return self.proto(h)

    def teacher_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Teacher output: softmax((logits - center) / teacher_temp_eff)."""
        return F.softmax((logits - self.center) / self._current_teacher_temp, dim=-1)

    def forward(self, x: torch.Tensor, use_teacher_temp: bool = False) -> torch.Tensor:
        logits = self.prototype_logits(x)

        if use_teacher_temp:
            return self.teacher_probs(logits)
        else:
            return F.log_softmax(logits / self.student_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        """EMA of the batch-mean *raw* teacher logits (from ``prototype_logits``).

        The center is subtracted from the logits, so it must live in logit space — as in
        the reference DINO implementation. Feeding it the post-softmax probabilities
        instead would give a center that always sums to 1, on the wrong scale.
        """
        self.center = self.center * self.center_momentum + teacher_logits.mean(0, keepdim=True) * (
            1 - self.center_momentum
        )

    def cancel_last_layer_gradients(self) -> None:
        """Zero gradients of the prototype (last) layer. Call after backward()."""
        for p in self.proto.parameters():
            p.grad = None
