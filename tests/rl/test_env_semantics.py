"""Step semantics, action mapping and determinism.

Determinism is non-negotiable: without byte-identical replay, no result in this
project is reproducible and no failure is debuggable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blockwave.core.constants import Action
from blockwave_rl.env.base import N_ACTIONS, BlockwaveEnv, EnvConfig, ObsMode
from blockwave_rl.env.crops import Variant


def rollout(config: EnvConfig, seed: int, actions: list[int]) -> list[np.ndarray]:
    env = BlockwaveEnv(config)
    obs, _ = env.reset(seed=seed)
    frames = [obs.copy()]
    for action in actions:
        obs, *_ = env.step(action)
        frames.append(obs.copy())
    return frames


# -- determinism ----------------------------------------------------------


@pytest.mark.parametrize("variant", list(Variant), ids=[v.value for v in Variant])
def test_same_seed_and_actions_give_byte_identical_observations(variant):
    actions = list(np.random.default_rng(0).integers(0, N_ACTIONS, 1_500))
    config = EnvConfig(variant=variant, gravity_scale=8.0)
    a = rollout(config, 42, actions)
    b = rollout(config, 42, actions)
    assert len(a) == len(b)
    for index, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(x, y), f"diverged at step {index}"


def test_determinism_survives_soft_resets():
    """Top-outs reseed the board; that must be deterministic too."""
    actions = [int(Action.HARD_DROP)] * 600  # tops out repeatedly
    config = EnvConfig(gravity_scale=8.0)
    env = BlockwaveEnv(config)
    env.reset(seed=5)
    tops = 0
    for action in actions:
        *_, info = env.step(action)
        tops = info["top_outs"]
    assert tops >= 2, "the scenario must actually exercise soft resets"
    assert np.array_equal(rollout(config, 5, actions)[-1], rollout(config, 5, actions)[-1])


def test_different_seeds_diverge():
    actions = [int(Action.HARD_DROP)] * 20
    a = rollout(EnvConfig(gravity_scale=8.0), 1, actions)
    b = rollout(EnvConfig(gravity_scale=8.0), 2, actions)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


# -- step semantics -------------------------------------------------------


def test_top_out_is_never_terminal():
    """Game over is a soft reset. Terminating would hand a surprise-minimizer a
    fresh-start bonus for dying — the paper's documented failure mode."""
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0))
    env.reset(seed=1)
    saw_top_out = False
    for _ in range(800):
        _, _, terminated, _, info = env.step(int(Action.HARD_DROP))
        assert terminated is False
        saw_top_out |= info["top_out"]
    assert saw_top_out, "hard-drop spam should have topped out"


def test_soft_reset_leaves_a_playable_board():
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0))
    env.reset(seed=1)
    for _ in range(800):
        _, _, _, _, info = env.step(int(Action.HARD_DROP))
        if info["top_out"]:
            assert not env.engine.game_over, "the board must be live after a soft reset"
            assert env.engine.piece is not None
            return
    pytest.fail("never topped out")


def test_truncation_only_at_the_horizon():
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0, max_steps=50))
    env.reset(seed=1)
    for step in range(1, 51):
        *_, truncated, _ = env.step(int(Action.NOOP))
        assert truncated is (step == 50), f"truncated={truncated} at step {step}"


def test_no_horizon_means_never_truncated():
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0))
    env.reset(seed=1)
    for _ in range(500):
        *_, truncated, _ = env.step(int(Action.HARD_DROP))
        assert truncated is False


# -- action mapping -------------------------------------------------------


def test_action_space_is_the_engine_action_set():
    assert BlockwaveEnv().n_actions == len(Action) == 8


@pytest.mark.parametrize("bad", [-1, 8, 99])
def test_out_of_range_actions_are_rejected(bad):
    env = BlockwaveEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(bad)


def test_one_action_is_one_pulse():
    """LEFT moves the piece exactly one column — no autorepeat, no held state."""
    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.BOARD_STATE))
    env.reset(seed=1)
    x0 = env.engine.piece.x
    env.step(int(Action.LEFT))
    assert env.engine.piece.x == x0 - 1
    env.step(int(Action.NOOP))
    assert env.engine.piece.x == x0 - 1, "NOOP must not continue the previous move"


def test_hold_lockout_is_enforced():
    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.BOARD_STATE))
    env.reset(seed=1)
    env.step(int(Action.HOLD))
    held_piece = env.engine.piece.type
    env.step(int(Action.HOLD))  # second hold on the same piece: refused
    assert env.engine.piece.type == held_piece


# -- observation shape ----------------------------------------------------


@pytest.mark.parametrize("stack", [1, 4])
def test_frame_stack_shape(stack):
    env = BlockwaveEnv(EnvConfig(frame_stack=stack))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_shape
    assert obs.shape[0] == stack
    assert obs.dtype == np.uint8


def test_frame_stack_carries_motion():
    """The whole point of stacking: consecutive frames differ as a piece falls."""
    env = BlockwaveEnv(EnvConfig(frame_stack=4, gravity_scale=20.0))
    env.reset(seed=1)
    for _ in range(4):
        obs, *_ = env.step(int(Action.NOOP))
    assert not np.array_equal(obs[0], obs[-1])


def test_empowerment_env_defaults_match_the_recorded_runs():
    """Every run logged before --agent-hz existed used 20 Hz and gravity x4."""
    from blockwave_rl.reward.empowerment_env import EmpowermentEnv

    env = EmpowermentEnv()
    assert env.env.config.agent_hz == 20.0
    assert env.env.config.gravity_scale == 4.0


def test_a_slower_agent_gets_fewer_decisions_per_piece():
    """Same fall speed, fewer decisions: the credit chain shortens, nothing is revealed."""
    from blockwave_rl.reward.empowerment_env import EmpowermentEnv

    def steps_to_first_lock(hz: float) -> int:
        env = EmpowermentEnv(agent_hz=hz)
        env.reset(seed=1)
        for t in range(1, 2_000):
            if env.step(0).locked:
                return t
        raise AssertionError("no piece locked")

    fast, slow = steps_to_first_lock(20.0), steps_to_first_lock(5.0)
    assert slow * 3 < fast < slow * 5
