"""The SMiRL reward: r_t = log p_θ(s_t), with θ learned from the agent's own experience.

Nothing in here knows what Tetris is. There is no notion of lines, holes, height
or score — only "how likely is this state, given what I have seen?". Whatever
competence emerges comes from the game making chaos the default and familiarity
something that has to be earned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .density import BernoulliDensity, GaussianDensity


@dataclass(slots=True)
class SmirlConfig:
    #: Effective memory of the density model, in agent steps. None = unbounded
    #: (ablation only — lets early chaos define "normal" forever).
    window: int | None = 1024
    #: Steps during which θ is updated but the reward is held at zero, because
    #: the statistics are not yet meaningful.
    warmup: int = 16
    #: Symmetric clip on the per-dimension log-likelihood.
    clip: float = 10.0


class SmirlReward:
    """Wraps a density estimator with the causal, warmed-up, clipped reward.

    θ is **not** reset on a top-out. The board clears, but the model of normal
    persists — that is what denies the agent a fresh-start bonus for dying.
    """

    def __init__(self, density: GaussianDensity | BernoulliDensity, config: SmirlConfig | None = None) -> None:
        self.density = density
        self.config = config or SmirlConfig()

    @classmethod
    def for_board(cls, shape: tuple[int, int], config: SmirlConfig | None = None) -> SmirlReward:
        config = config or SmirlConfig()
        return cls(BernoulliDensity(shape, window=config.window), config)

    @classmethod
    def for_latent(cls, dim: int, config: SmirlConfig | None = None) -> SmirlReward:
        config = config or SmirlConfig()
        return cls(GaussianDensity(dim, window=config.window), config)

    def reset(self) -> None:
        """Only at the start of a *training run* — never on a top-out."""
        self.density.reset()

    def __call__(self, x: np.ndarray) -> float:
        in_warmup = self.density.t < self.config.warmup
        logp = self.density.score_then_update(x)      # causal: θ_{t-1}, then update
        if in_warmup:
            return 0.0
        per_dim = logp / self.density.dim
        return float(np.clip(per_dim, -self.config.clip, self.config.clip))

    def progress(self) -> float:
        """Bounded timestep fraction for the augmented state."""
        window = self.config.window or 4096
        return min(self.density.t / window, 1.0)
