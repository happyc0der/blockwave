"""Score a fixed policy under the intrinsic reward, without training anything.

This is how a reward is judged *before* compute is spent on it. A policy is run
through the real environment and the intrinsic reward is computed exactly as it
would be during training — for SMiRL, θ fitted online and causally on that
policy's own experience (`run`); for empowerment, the exact future count
(`run_empowerment`). If a competent player does not out-score random play and
every exploit here, the objective is wrong and no amount of training will fix it.

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
    embed: Callable[[np.ndarray], np.ndarray] | None = None,
    clock: str = "tick",
) -> RolloutResult:
    """Run ``policy`` and score it under the intrinsic reward.

    ``embed``   map a board to a latent; if given, the density is a Gaussian
                over latents rather than the per-cell Bernoulli.
    ``clock``   "tick" scores every agent step; "placement" scores only when
                the locked board changes. Under frame-level control a per-tick
                reward measures how often the board *changes*, so slow play
                earns more ticks of an unchanging board — "placement" removes
                that incentive.
    """
    if clock not in {"tick", "placement"}:
        raise ValueError(f"unknown clock {clock!r}")
    config = env_config or EnvConfig(obs_mode=ObsMode.BOARD_STATE, agent_hz=20, gravity_scale=4.0)
    env = BlockwaveEnv(config)
    env.reset(seed=seed)
    if embed is None:
        reward = SmirlReward.for_board(env.observation_shape, smirl_config)
        featurize = lambda board: board
    else:
        dim = int(np.asarray(embed(env.occupancy())).reshape(-1).shape[0])
        reward = SmirlReward.for_latent(dim, smirl_config)
        featurize = lambda board: np.asarray(embed(board)).reshape(-1)
    rng = np.random.default_rng(seed)

    rewards: list[float] = []
    top_outs = 0
    placed_before = env.engine.stats.pieces_placed
    for t in range(steps):
        *_, info = env.step(policy(env.engine, rng, t))
        top_outs += int(info["top_out"])
        placed = env.engine.stats.pieces_placed
        changed = placed != placed_before or info["top_out"]
        placed_before = placed
        if clock == "tick" or changed:
            rewards.append(reward(featurize(env.occupancy())))

    scored = np.array(rewards[reward.config.warmup :]) if len(rewards) > reward.config.warmup else np.zeros(1)
    return RolloutResult(name, float(scored.mean()), top_outs, steps)


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


def run_empowerment(
    name: str,
    policy: Policy,
    *,
    horizon: int = 2,
    max_scored: int = 300,
    max_steps: int = 200_000,
    seed: int = 0,
) -> RolloutResult:
    """Score a policy by horizon empowerment of each locked board.

    The queue is what the agent could see: the piece now in play plus the
    preview. Empowerment is expensive, so only the first ``max_scored`` lock
    events are scored — enough to estimate the mean.
    """
    from .empowerment import horizon_empowerment

    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.BOARD_STATE, agent_hz=20, gravity_scale=4.0))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    top_outs = 0
    prev = env.engine.stats.pieces_placed
    t = 0
    while len(values) < max_scored and t < max_steps:
        *_, info = env.step(policy(env.engine, rng, t))
        t += 1
        top_outs += int(info["top_out"])
        placed = env.engine.stats.pieces_placed
        if placed == prev and not info["top_out"]:
            continue
        prev = placed
        engine = env.engine
        queue = engine.preview(horizon) if engine.piece is None else (
            [engine.piece.type] + engine.preview(horizon - 1)
        )
        values.append(horizon_empowerment(list(engine.board.rows), queue[:horizon]))
    arr = np.array(values)
    return RolloutResult(name, float(arr.mean()), top_outs, t)
