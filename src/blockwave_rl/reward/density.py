"""Bounded-memory density estimators — the model of "normal" the reward is built on.

Both estimators share one discipline, and it is the whole point of this module:

**Causal.** ``score_then_update(x)`` computes log p(x) under θ_{t-1} and only
*then* folds x into θ_t. Updating first would let the model partially explain
away the very state it is scoring, weakening the surprise signal — worst of all
for exactly the rare states that matter most.

**Bounded memory.** θ is an exponentially-weighted estimate with an effective
window W, via ``alpha = max(1/(t+1), 1/W)``: an exact running average during
warm-up, then a constant-rate forgetting estimate. Without the bound, the chaos
of early random play would define "normal" for the entire run, and later
competent play would score *worse* than the flailing that preceded it.
``window=None`` gives the unbounded estimator, kept for the ablation.

**Floored.** A variance (or probability) allowed to collapse turns a constant
trajectory into log p → +∞: a numerical blowup and a reward hack at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


def _alpha(t: int, window: int | None) -> float:
    exact = 1.0 / (t + 1)
    return exact if window is None else max(exact, 1.0 / window)


@dataclass(slots=True)
class DensityState:
    """A fixed-shape snapshot of θ, for the augmented policy input."""

    mean: np.ndarray
    log_scale: np.ndarray
    t: int


class GaussianDensity:
    """Diagonal Gaussian over a D-dimensional vector (a VAE latent)."""

    def __init__(self, dim: int, window: int | None = 1024, var_floor: float = 1e-4) -> None:
        if var_floor <= 0:
            raise ValueError("var_floor must be positive")
        self.dim = dim
        self.window = window
        self.var_floor = var_floor
        self.reset()

    def reset(self) -> None:
        self.mu = np.zeros(self.dim, dtype=np.float64)
        self.var = np.ones(self.dim, dtype=np.float64)
        self.t = 0

    def log_prob(self, z: np.ndarray) -> float:
        """log N(z; μ, σ²) under the *current* parameters. Does not update."""
        z = np.asarray(z, dtype=np.float64)
        delta = z - self.mu
        return float(-0.5 * np.sum(LOG_2PI + np.log(self.var) + delta * delta / self.var))

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64)
        alpha = _alpha(self.t, self.window)
        delta = z - self.mu
        self.mu = self.mu + alpha * delta
        # Exponentially-weighted variance (West, 1979).
        self.var = np.maximum((1.0 - alpha) * (self.var + alpha * delta * delta), self.var_floor)
        self.t += 1

    def score_then_update(self, z: np.ndarray) -> float:
        """The causal primitive: log p(z | θ_{t-1}), then θ_{t-1} → θ_t."""
        logp = self.log_prob(z)
        self.update(z)
        return logp

    def state(self) -> DensityState:
        return DensityState(self.mu.copy(), 0.5 * np.log(self.var), self.t)


class BernoulliDensity:
    """Product of independent Bernoullis over binary cells.

    The paper's actual Tetris model. Used by the board-state feasibility track.
    """

    def __init__(self, shape: tuple[int, ...], window: int | None = 1024, eps: float = 1e-3) -> None:
        if not 0 < eps < 0.5:
            raise ValueError("eps must be in (0, 0.5)")
        self.shape = shape
        self.window = window
        self.eps = eps
        self.reset()

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))

    def reset(self) -> None:
        self.p = np.full(self.shape, 0.5, dtype=np.float64)
        self.t = 0

    def log_prob(self, cells: np.ndarray) -> float:
        b = np.asarray(cells, dtype=np.float64)
        return float(np.sum(b * np.log(self.p) + (1.0 - b) * np.log1p(-self.p)))

    def update(self, cells: np.ndarray) -> None:
        b = np.asarray(cells, dtype=np.float64)
        alpha = _alpha(self.t, self.window)
        self.p = np.clip(self.p + alpha * (b - self.p), self.eps, 1.0 - self.eps)
        self.t += 1

    def score_then_update(self, cells: np.ndarray) -> float:
        logp = self.log_prob(cells)
        self.update(cells)
        return logp

    def state(self) -> DensityState:
        # log-odds is the natural unbounded-but-floored parameterization.
        return DensityState(self.p.copy(), np.log(self.p) - np.log1p(-self.p), self.t)
