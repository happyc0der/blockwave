"""The board-state training environment for the empowerment feasibility track.

Wraps `BlockwaveEnv` to supply two things the policy and learner need.

**The policy observation.** The locked board alone is not enough to act on: it
omits the piece being controlled. So the policy sees two 24x10 planes — locked
cells and the active piece, over all 24 rows so a piece is visible from the
moment it spawns in the buffer — plus the next pieces as one-hot. That queue is
exactly what the reward is conditioned on, so the policy sees what its reward
depends on. This track reads true occupancy and is **not** the pixel deliverable.

**The reward, scored on lock events.** Queue-corrected horizon empowerment of
the locked board. Death is handled here, and handled carefully: the base env
soft-resets inside `step()`, so a topped-out board has already become a fresh
empty one — the *maximum* empowerment — by the time `step()` returns. Scoring
that board would reward every death as full control. Game over is precisely
zero control: no futures remain, so raw empowerment is log(1 + 0) = 0. That is
not an injected death penalty; it is what the principle says a dead board is
worth.

*How long* it is worth nothing is the ``death`` setting. ``"one_step"`` charges
zero control for one placement. It was the first accounting, and a learner
that credited moves better found its flaw: once a board was doomed, dying fast
beat lingering, because the soft reset hands back full control — the
fresh-start bonus the soft reset exists to deny, returning through the value
function. ``"absorbing"`` (the default) charges zero control for every
placement the dead game would have had; see `HorizonEmpowermentReward.death`.
The episode still does not terminate and the learner still bootstraps through
the reset.

Reads `top_out` from `info` — an observable event, not score. Never reads
score, lines or level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blockwave.core.constants import BOARD_WIDTH, TOTAL_HEIGHT, PieceType

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from .empowerment import HorizonEmpowermentReward

N_PIECES = len(PieceType)


@dataclass(slots=True)
class StepResult:
    grid: np.ndarray       # (2, 24, 10) float32
    queue: np.ndarray      # (horizon * 7,) float32
    reward: float
    locked: bool           # a placement (or top-out) happened: the discount clock ticks
    top_out: bool
    info: dict             # evaluation only — the learner must not read it


DEATH_ACCOUNTING = ("absorbing", "one_step")


class EmpowermentEnv:
    def __init__(
        self,
        horizon: int = 2,
        config: EnvConfig | None = None,
        death: str = "absorbing",
        gamma: float = 0.99,
    ) -> None:
        if death not in DEATH_ACCOUNTING:
            raise ValueError(f"death must be one of {DEATH_ACCOUNTING}")
        self.horizon = horizon
        self.death = death
        self.gamma = gamma
        base = config or EnvConfig(obs_mode=ObsMode.BOARD_STATE, agent_hz=20, gravity_scale=4.0)
        base.obs_mode = ObsMode.BOARD_STATE
        self.env = BlockwaveEnv(base)
        self.reward_fn = HorizonEmpowermentReward(horizon)
        self._placed = 0

    @property
    def n_actions(self) -> int:
        return self.env.n_actions

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (2, TOTAL_HEIGHT, BOARD_WIDTH)

    @property
    def queue_size(self) -> int:
        return self.horizon * N_PIECES

    def _queue(self) -> list[PieceType]:
        engine = self.env.engine
        if engine.piece is None:
            return engine.preview(self.horizon)
        return [engine.piece.type] + engine.preview(self.horizon - 1)

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        engine = self.env.engine
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        grid[0] = engine.board.colors != 0
        if engine.piece is not None:
            for x, y in engine.piece.cells():
                if 0 <= y < TOTAL_HEIGHT and 0 <= x < BOARD_WIDTH:
                    grid[1, y, x] = 1.0
        queue = np.zeros(self.queue_size, dtype=np.float32)
        for slot, piece in enumerate(self._queue()[: self.horizon]):
            queue[slot * N_PIECES + int(piece) - 1] = 1.0
        return grid, queue

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.env.reset(seed=seed)
        self._placed = self.env.engine.stats.pieces_placed
        return self._observe()

    def step(self, action: int) -> StepResult:
        _, _, _, _, info = self.env.step(action)
        top_out = bool(info["top_out"])
        placed = self.env.engine.stats.pieces_placed
        locked = top_out or placed != self._placed
        self._placed = placed

        reward = 0.0
        if top_out:
            # Zero reachable futures. The board on screen is already the fresh
            # one, so score the *event*, not the board.
            gamma = self.gamma if self.death == "absorbing" else None
            reward = self.reward_fn.death(self._queue(), gamma)
        elif locked:
            reward = self.reward_fn(list(self.env.engine.board.rows), self._queue())

        grid, queue = self._observe()
        return StepResult(grid, queue, float(reward), locked, top_out, info)
