"""Empowerment from pixels: how many distinct futures the keys can still reach.

The board-state reward was log(number of distinct reachable boards), counted
exactly with the simulator. Here the same quantity is estimated from the agent's
own learned model, which brings two problems the exact version did not have.

**What counts as distinct.** Two predicted latents are never bit-identical, so
counting needs a notion of "the same outcome". This uses the effective number of
distinct points under a Gaussian kernel,

    N_eff = (sum_i w_i)^2 / sum_ij w_i w_j k(z_i, z_j)

which is the count when outcomes are either coincident or far apart, and
interpolates sensibly in between. Its one knob is the kernel width, and
`calibrate` sets that from the agent's own data rather than by hand.

**Futures that end the game are not futures.** The model predicts how likely
each program is to end the game, and each outcome is weighted by its chance of
surviving. A board where every program kills has no reachable futures at all,
which is what a dead board is worth.

The reward is the log of that count, minus the same quantity on an empty board —
the queue correction that made the board-state reward readable, kept here for
the same reason: it depends on what is coming, which the agent cannot influence.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import MacroModel

#: Guards log(0) when every future is predicted to end the game.
FLOOR = 1e-3


def effective_count(points: torch.Tensor, weights: torch.Tensor, width: float) -> torch.Tensor:
    """(B, N, D), (B, N) -> (B,) effective number of distinct points."""
    distance = torch.cdist(points, points)
    kernel = torch.exp(-0.5 * (distance / width) ** 2)
    weighted = torch.einsum("bi,bj,bij->b", weights, weights, kernel)
    return weights.sum(dim=1).pow(2) / weighted.clamp(min=1e-9)


class PixelEmpowerment:
    """The reward, computed through a learned model of the agent's own keys.

    ``horizon`` is in pieces. One is blind — on any board with headroom every
    program still reaches somewhere different, which is what the board-state
    track found by enumeration — so the default looks two pieces ahead. The
    second piece is expanded over a sample of programs rather than all of them,
    which costs accuracy in the tail and a great deal less compute.
    """

    def __init__(
        self,
        model: MacroModel,
        *,
        horizon: int = 2,
        width: float = 1.0,
        branch: int = 12,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.horizon = horizon
        self.width = width
        self.branch = branch
        self.device = device
        # A fixed sample, drawn once. Re-drawing it per call would make the
        # reward for one state differ between two evaluations of it, which is
        # noise in a training signal for no gain in what it measures.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._branch = torch.randperm(model.n_macros, generator=generator)[:branch].to(device)
        self._baseline: float | None = None

    def _reachable(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted outcomes and survival weights, (B, N, D) and (B, N)."""
        points, weights = self.model.outcomes(z)
        for _ in range(self.horizon - 1):
            batch, n, dim = points.shape
            macros = self._branch
            nxt, survive = self.model.outcomes(points.reshape(batch * n, dim), macros)
            points = nxt.reshape(batch, n * len(macros), dim)
            weights = (weights.unsqueeze(-1) * survive.reshape(batch, n, len(macros))).reshape(batch, -1)
        return points, weights

    def raw(self, latents: np.ndarray) -> np.ndarray:
        """log(effective number of distinct reachable futures), per state."""
        z = torch.as_tensor(np.atleast_2d(latents), dtype=torch.float32, device=self.device)
        points, weights = self._reachable(z)
        count = effective_count(points, weights, self.width)
        return torch.log(count.clamp(min=FLOOR)).cpu().numpy()

    def calibrate(self, empty_latent: np.ndarray, width: float) -> None:
        """Set the kernel width and the empty-board baseline.

        The width is the model's own held-out error: two predicted futures count
        as the same outcome when they are closer together than the model can
        resolve. That is not a free knob — it falls out of how well the agent
        has learned its own keys, and it tightens as the model improves.

        Many programs genuinely do land on the same placement (five steps left
        and four, when only three fit), and those coincidences are what the
        count is *for*: they are the futures the agent cannot tell apart, and
        therefore cannot choose between.
        """
        self.width = max(float(width), 1e-3)
        self._baseline = None
        self._baseline = float(self.raw(empty_latent)[0])

    @property
    def baseline(self) -> float:
        if self._baseline is None:
            raise RuntimeError("calibrate() first: the reward is relative to an empty board")
        return self._baseline

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        """The reward: zero with a board's worth of headroom, negative as it shrinks."""
        return self.raw(latents) - self.baseline
