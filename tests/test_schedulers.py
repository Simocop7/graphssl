"""Tests for the learning-rate and EMA-momentum schedules."""

import math

import pytest

from graphssl.utils.schedulers import CosineDecayScheduler, CosineEMAScheduler


class TestCosineDecayScheduler:
    def test_warmup_is_linear_from_zero(self):
        sched = CosineDecayScheduler(max_val=1e-2, total_steps=10_000, warmup_steps=1_000)
        assert sched.get(0) == 0.0
        assert sched.get(500) == pytest.approx(5e-3)
        assert sched.get(1_000) == pytest.approx(1e-2)

    def test_cosine_decay_after_warmup(self):
        # BGRL's schedule (its Appendix F): eta * (1 + cos((i - warmup) * pi / (total - warmup))) / 2
        sched = CosineDecayScheduler(max_val=1e-2, total_steps=10_000, warmup_steps=1_000)
        for step in (1_000, 2_500, 5_500, 9_999):
            expected = 1e-2 * (1 + math.cos((step - 1_000) * math.pi / 9_000)) / 2
            assert sched.get(step) == pytest.approx(expected)
        assert sched.get(5_500) == pytest.approx(5e-3)  # halfway through the decay

    def test_ends_at_min_val_and_stays_there(self):
        sched = CosineDecayScheduler(max_val=1.0, min_val=0.1, total_steps=100, warmup_steps=10)
        assert sched.get(100) == pytest.approx(0.1)
        assert sched.get(1_000) == pytest.approx(0.1)

    def test_without_warmup_starts_at_max_val(self):
        assert CosineDecayScheduler(max_val=3.0, total_steps=50).get(0) == pytest.approx(3.0)

    def test_decay_is_monotonic(self):
        sched = CosineDecayScheduler(max_val=1.0, total_steps=200, warmup_steps=20)
        values = [sched.get(step) for step in range(20, 201)]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_invalid_arguments(self):
        with pytest.raises(AssertionError):
            CosineDecayScheduler(max_val=1.0, total_steps=0)
        with pytest.raises(AssertionError):
            CosineDecayScheduler(max_val=1.0, total_steps=10, warmup_steps=-1)


class TestCosineEMAScheduler:
    def test_goes_from_base_to_end(self):
        sched = CosineEMAScheduler(ema_base=0.99, ema_end=1.0, total_steps=1_000)
        assert sched.get(0) == pytest.approx(0.99)
        assert sched.get(1_000) == pytest.approx(1.0)
        assert sched.get(5_000) == pytest.approx(1.0)  # clamped after the last step

    def test_matches_the_bgrl_formula(self):
        # tau_i = 1 - (1 - tau_base) / 2 * (cos(i * pi / n) + 1)
        sched = CosineEMAScheduler(ema_base=0.99, ema_end=1.0, total_steps=10_000)
        for step in (0, 1, 2_500, 5_000, 9_999):
            expected = 1 - (1 - 0.99) / 2 * (math.cos(step * math.pi / 10_000) + 1)
            assert sched.get(step) == pytest.approx(expected)

    def test_momentum_only_increases(self):
        sched = CosineEMAScheduler(ema_base=0.9, ema_end=0.996, total_steps=300)
        values = [sched.get(step) for step in range(301)]
        assert all(a <= b for a, b in zip(values, values[1:], strict=False))

    def test_invalid_arguments(self):
        with pytest.raises(AssertionError):
            CosineEMAScheduler(ema_base=1.0, ema_end=1.0, total_steps=10)
        with pytest.raises(AssertionError):
            CosineEMAScheduler(ema_base=0.99, ema_end=1.0, total_steps=0)
