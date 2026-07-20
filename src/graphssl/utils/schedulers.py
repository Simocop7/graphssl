"""Learning rate and EMA momentum schedulers."""

from __future__ import annotations
import math


class CosineDecayScheduler:
    """Cosine decay from max_val to min_val over total_steps with optional linear warmup.

    At step 0 (after warmup): max_val. At step total_steps: min_val.
    """

    def __init__(
        self,
        max_val: float,
        min_val: float = 0.0,
        total_steps: int = 1,
        warmup_steps: int = 0,
    ):
        assert total_steps > 0, "total_steps must be > 0"
        assert warmup_steps >= 0
        self.max_val = max_val
        self.min_val = min_val
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_val * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_val + (self.max_val - self.min_val) * cosine


class CosineEMAScheduler:
    """EMA momentum annealing from ema_base to ema_end over total_steps.

    Momentum increases (ema_base < ema_end) following a cosine schedule.
    Typical use: ema_base=0.99, ema_end=1.0 for DINO/BGRL teacher momentum.
    """

    def __init__(self, ema_base: float, ema_end: float, total_steps: int):
        assert 0.0 < ema_base < 1.0 and 0.0 < ema_end <= 1.0
        assert total_steps > 0
        self.ema_base = ema_base
        self.ema_end = ema_end
        self.total_steps = total_steps

    def get(self, step: int) -> float:
        # cosine goes from 1 (step=0) to 0 (step=total_steps)
        progress = min(step / self.total_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.ema_end - (self.ema_end - self.ema_base) * cosine
