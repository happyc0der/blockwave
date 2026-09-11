"""Collect states for representation learning: board grids, or rendered frames.

The corpus is **unsupervised data only**: board states, with the policies that
produced them discarded. It is never a reward, a label or a policy init.

Its *composition* matters more than it looks. The representation is judged by
whether a density over it ranks competent play above exploits — so the encoder
must represent every state any policy under test can reach, and represent them
comparably well. If towers were missing from the corpus, a tower would be
encoded badly, and a badly encoded state can land anywhere in latent space:
artificially tight (scoring high) or scattered (scoring low). The ranking would
then be decided by a representation gap rather than by the objective. So the
corpus deliberately covers the exploits too.

States are sampled on lock events, matching the per-placement reward clock.
Frames are sampled there too, plus one mid-flight frame per piece: the pixel
encoder has to represent a falling piece as well as a settled board, because the
reward will be computed from frames at both points.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from blockwave.core.engine import Engine

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from ..env.crops import Variant

Policy = Callable[[Engine, np.random.Generator, int], int]


def collect(policy: Policy, n_states: int, *, seed: int, early_only: bool = False) -> np.ndarray:
    """Board states, one per lock event, from one policy.

    ``early_only`` keeps only the first few placements after each reset, which
    is where the near-empty boards live.
    """
    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.BOARD_STATE, agent_hz=20, gravity_scale=4.0))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    prev = 0
    t = 0
    while len(states) < n_states:
        *_, info = env.step(policy(env.engine, rng, t))
        t += 1
        placed = env.engine.stats.pieces_placed
        locked = placed != prev or info["top_out"]
        prev = placed
        if not locked:
            continue
        if early_only and placed > 6:
            continue
        states.append(env.occupancy().copy())
        if t > 5_000_000:  # a policy that never locks would loop forever
            raise RuntimeError("policy produced too few lock events")
    return np.stack(states)


def build(sources: dict[str, tuple[Policy, int, bool]], *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stack a mixture. Returns (states, source index per state)."""
    chunks, labels = [], []
    for index, (name, (policy, n, early)) in enumerate(sources.items()):
        chunk = collect(policy, n, seed=seed + index, early_only=early)
        chunks.append(chunk)
        labels.append(np.full(len(chunk), index))
    return np.concatenate(chunks), np.concatenate(labels)


# -- pixels ---------------------------------------------------------------


def collect_frames(
    policy: Policy,
    n_frames: int,
    *,
    seed: int,
    early_only: bool = False,
    variant: Variant | str = Variant.BOARD_PLUS_PREVIEW,
    cell_px: int = 4,
    agent_hz: float = 5.0,
    gravity_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rendered frames from one policy, with the board behind each one.

    Returns ``(frames, boards, is_lock)``. The boards are the true occupancy
    grids, kept **only** so offline probes can ask whether the latents encode
    stack height, holes and bumpiness. They never reach the agent, and
    `tests/rl/test_firewall.py` keeps it that way.

    One frame per lock event, plus one frame from a uniformly chosen moment
    while each piece was falling.
    """
    env = BlockwaveEnv(
        EnvConfig(
            obs_mode=ObsMode.PIXELS, variant=Variant(variant), cell_px=cell_px,
            agent_hz=agent_hz, gravity_scale=gravity_scale, frame_stack=1,
        )
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    boards: list[np.ndarray] = []
    is_lock: list[bool] = []

    def keep(lock: bool) -> None:
        frames.append(env._pixels())
        boards.append(env.occupancy().copy())
        is_lock.append(lock)

    prev = 0
    t = 0
    in_flight: list[int] = []
    while len(frames) < n_frames:
        *_, info = env.step(policy(env.engine, rng, t))
        t += 1
        placed = env.engine.stats.pieces_placed
        locked = placed != prev or info["top_out"]
        if not locked:
            in_flight.append(t)
            continue
        prev = placed
        if not (early_only and placed > 6):
            keep(True)
            # One moment from the flight that just ended, chosen uniformly.
            if in_flight and len(frames) < n_frames:
                pick = int(rng.integers(len(in_flight)))
                if pick == len(in_flight) - 1:
                    keep(False)
        in_flight = []
        if t > 5_000_000:
            raise RuntimeError("policy produced too few lock events")
    return np.stack(frames), np.stack(boards), np.array(is_lock)


def build_frames(
    sources: dict[str, tuple[Policy, int, bool]], *, seed: int = 0, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack a mixture of frame sources. Returns (frames, boards, is_lock, source)."""
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labels = []
    for index, (name, (policy, n, early)) in enumerate(sources.items()):
        chunk = collect_frames(policy, n, seed=seed + index, early_only=early, **kwargs)
        chunks.append(chunk)
        labels.append(np.full(len(chunk[0]), index))
    return (
        np.concatenate([c[0] for c in chunks]),
        np.concatenate([c[1] for c in chunks]),
        np.concatenate([c[2] for c in chunks]),
        np.concatenate(labels),
    )
