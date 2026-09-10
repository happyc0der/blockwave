"""Collect board states for representation learning.

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
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from blockwave.core.engine import Engine

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode

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
