"""BLOCKWAVE as a reinforcement-learning environment.

Gym-shaped — `reset() -> (obs, info)` and `step(a) -> (obs, reward, terminated,
truncated, info)` — without depending on gymnasium.

**This environment emits no reward.** `step` always returns `0.0`. The intrinsic
reward is computed downstream, from observations alone, by `blockwave_rl.reward`.
That keeps the firewall trivially checkable: there is no code path here by which
score could become reward, because there is no reward here at all.

Step semantics
--------------
* One agent decision is one engine `Action`, applied through `apply_tick` — the
  same function the live game and replays use. The agent plays the real game.
* One action per step, as a single pulse for one agent tick. No autorepeat: DAS
  exists so a human can hold a key, and an agent emitting one action per tick has
  no use for it.
* **A top-out is not terminal.** The board resets, `terminated` stays False, and
  the episode continues. An empty board is the most familiar state there is, so
  treating game over as an episode boundary hands a surprise-minimizer a
  fresh-start bonus for dying — the paper's documented failure mode.
* `truncated` is True only at the configured rollout horizon.
* `info` carries score, lines and level for logging. The agent must never read it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from blockwave.app.replay import apply_tick
from blockwave.core.constants import Action, BOARD_WIDTH, TOTAL_HEIGHT, VISIBLE_TOP
from blockwave.core.engine import Engine, EngineConfig
from blockwave.render.compositor import Compositor
from blockwave.render.layout import Layout

from .crops import Crop, Variant, crop_for

N_ACTIONS = len(Action)


class ObsMode(str, Enum):
    #: Rendered pixels. The deliverable.
    PIXELS = "pixels"
    #: The true occupancy grid, read from the engine. **Not pixel-only.** Exists
    #: solely for the Stage 1 feasibility checkpoint, which asks whether the
    #: objective survives a full-size board independently of whether it can be
    #: learned from pixels.
    BOARD_STATE = "board_state"


@dataclass(slots=True)
class EnvConfig:
    obs_mode: ObsMode = ObsMode.PIXELS
    variant: Variant = Variant.BOARD_ONLY
    cell_px: int = 4
    #: `flat` because shake, bloom and particles are frame variance unrelated to
    #: the board — under a log-likelihood reward, that variance is surprise.
    profile: str = "flat"
    #: Agent decisions per second. Sets the length of the credit-assignment
    #: chain; it gives the agent no information.
    agent_hz: float = 20.0
    gravity_scale: float = 1.0
    start_level: int = 1
    #: Stationary dynamics. With progression on, an improving agent levels up
    #: until gravity outruns its decision rate — measured: at 20 Hz a scripted
    #: player dies every time around level 11-13, where ~7 rows fall per
    #: decision against the ~12 decisions positioning takes. That would make a
    #: learning-curve plateau indistinguishable from the objective failing.
    fixed_level: bool = True
    frame_stack: int = 4
    #: Rollout horizon. `truncated` fires here and nowhere else.
    max_steps: int | None = None


@dataclass(slots=True)
class _Counters:
    steps: int = 0
    top_outs: int = 0
    reset_seeds: list[int] = field(default_factory=list)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Integer luminance — bit-exact on every platform, unlike float weights."""
    r = frame[..., 0].astype(np.uint32)
    g = frame[..., 1].astype(np.uint32)
    b = frame[..., 2].astype(np.uint32)
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


class BlockwaveEnv:
    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        c = self.config
        self.dt = 1.0 / c.agent_hz
        self.engine = Engine(
            EngineConfig(
                seed=0,
                start_level=c.start_level,
                gravity_scale=c.gravity_scale,
                fixed_level=c.fixed_level,
            )
        )
        self.layout = Layout(c.cell_px)
        self.compositor = Compositor(self.layout, c.profile)
        self.crop: Crop = crop_for(c.variant, self.layout)
        self._frames: deque[np.ndarray] = deque(maxlen=max(1, c.frame_stack))
        self._rng = np.random.default_rng(0)
        self._counters = _Counters()

    # -- spaces -----------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self.config.obs_mode is ObsMode.BOARD_STATE:
            return (TOTAL_HEIGHT - VISIBLE_TOP, BOARD_WIDTH)
        h, w = self.crop.shape
        return (max(1, self.config.frame_stack), h, w)

    # -- observation ------------------------------------------------------

    def _pixels(self) -> np.ndarray:
        """Render, crop and convert one frame. No shake: the env passes none."""
        frame = self.compositor.render(self.engine, shake=(0, 0))
        return _to_gray(self.crop.apply(frame))

    def occupancy(self) -> np.ndarray:
        """The true visible occupancy grid, as uint8 0/1.

        Reads engine state directly. Used by the BOARD_STATE feasibility track
        and by offline diagnostics — never by the pixel agent.
        """
        colors = self.engine.board.colors[VISIBLE_TOP:TOTAL_HEIGHT]
        return (colors != 0).astype(np.uint8)

    def _observe(self) -> np.ndarray:
        if self.config.obs_mode is ObsMode.BOARD_STATE:
            return self.occupancy()
        return np.stack(tuple(self._frames), axis=0)

    # -- lifecycle --------------------------------------------------------

    def _new_game_seed(self) -> int:
        seed = int(self._rng.integers(0, 2**31 - 1))
        self._counters.reset_seeds.append(seed)
        return seed

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(seed)
        self._counters = _Counters()
        self.engine.reset(seed=self._new_game_seed())

        self._frames.clear()
        if self.config.obs_mode is ObsMode.PIXELS:
            first = self._pixels()
            for _ in range(self._frames.maxlen or 1):
                self._frames.append(first)
        return self._observe(), self._info(top_out=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not 0 <= int(action) < N_ACTIONS:
            raise ValueError(f"action {action} outside 0..{N_ACTIONS - 1}")

        apply_tick(self.engine, [Action(int(action))], self.dt)
        self._counters.steps += 1

        top_out = self.engine.game_over
        if top_out:
            # Soft reset: a new board, but the MDP episode — and whatever the
            # reward module has learned about "normal" — carries on.
            self._counters.top_outs += 1
            self.engine.reset(seed=self._new_game_seed())

        if self.config.obs_mode is ObsMode.PIXELS:
            self._frames.append(self._pixels())

        max_steps = self.config.max_steps
        truncated = max_steps is not None and self._counters.steps >= max_steps

        # Reward is always zero. The environment has no opinion about what is good.
        return self._observe(), 0.0, False, truncated, self._info(top_out=top_out)

    def _info(self, *, top_out: bool) -> dict:
        """For logging and post-hoc evaluation ONLY. The agent must never read this."""
        stats = self.engine.stats
        return {
            "score": stats.score,
            "lines": stats.lines,
            "level": stats.level,
            "pieces": stats.pieces_placed,
            "top_out": top_out,
            "top_outs": self._counters.top_outs,
            "steps": self._counters.steps,
        }
