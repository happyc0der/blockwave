"""Score a fixed policy under the intrinsic reward, without training anything.

This is how the reward is judged *before* compute is spent on it. A policy is
run through the real environment and the SMiRL reward is computed exactly as it
would be during training — θ fitted online, causally, on that policy's own
experience. If a competent player does not out-score random play and every
exploit here, the objective is wrong and no amount of training will fix it.

Reads only occupancy and the reward. Never reads score, lines or level.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from blockwave.core.constants import Action
from blockwave.core.engine import Engine

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from ..scripted.heuristic import HeuristicPlayer
from .smirl import SmirlConfig, SmirlReward

Policy = Callable[[Engine, np.random.Generator, int], int]


@dataclass(slots=True)
class RolloutResult:
    name: str
    mean_reward: float
    total_reward: float
    top_outs: int
    steps: int


def run(
    name: str,
    policy: Policy,
    *,
    steps: int = 6_000,
    seed: int = 0,
    env_config: EnvConfig | None = None,
    smirl_config: SmirlConfig | None = None,
) -> RolloutResult:
    config = env_config or EnvConfig(obs_mode=ObsMode.BOARD_STATE, agent_hz=20, gravity_scale=4.0)
    env = BlockwaveEnv(config)
    env.reset(seed=seed)
    reward = SmirlReward.for_board(env.observation_shape, smirl_config)
    rng = np.random.default_rng(seed)

    rewards: list[float] = []
    top_outs = 0
    for t in range(steps):
        *_, info = env.step(policy(env.engine, rng, t))
        rewards.append(reward(env.occupancy()))
        top_outs += int(info["top_out"])

    scored = np.array(rewards[reward.config.warmup :])
    return RolloutResult(name, float(scored.mean()), float(scored.sum()), top_outs, steps)


# -- the reference and the exploits ---------------------------------------


def heuristic() -> Policy:
    player = HeuristicPlayer()
    return lambda engine, rng, t: player.act(engine)


def random_policy(engine: Engine, rng: np.random.Generator, t: int) -> int:
    return int(rng.integers(len(Action)))


def all_noop(engine: Engine, rng: np.random.Generator, t: int) -> int:
    """Let gravity do everything: pieces lock wherever they spawn."""
    return int(Action.NOOP)


def hard_drop_spam(engine: Engine, rng: np.random.Generator, t: int) -> int:
    """Death farming: slam every piece down at the spawn column."""
    return int(Action.HARD_DROP)


def oscillate(engine: Engine, rng: np.random.Generator, t: int) -> int:
    """Wiggle left and right forever, locking only when the reset cap forces it."""
    return int(Action.LEFT if t % 2 else Action.RIGHT)


def rotate_spam(engine: Engine, rng: np.random.Generator, t: int) -> int:
    return int(Action.ROTATE_CW)


def hold_spam(engine: Engine, rng: np.random.Generator, t: int) -> int:
    return int(Action.HOLD)


EXPLOITS: dict[str, Policy] = {
    "random": random_policy,
    "all_noop": all_noop,
    "hard_drop_spam": hard_drop_spam,
    "oscillate": oscillate,
    "rotate_spam": rotate_spam,
    "hold_spam": hold_spam,
}
