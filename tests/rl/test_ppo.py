"""PPO's per-placement discounting — subtle enough to get wrong silently."""

from __future__ import annotations

import numpy as np
import pytest

from blockwave_rl.agents.ppo import gae


def one_env(*values):
    return np.array(values, dtype=np.float32).reshape(-1, 1)


def test_no_discount_between_placements():
    """With no lock, a later reward reaches earlier steps undiminished."""
    rewards = one_env(0, 0, 0, 1)
    values = one_env(0, 0, 0, 0)
    locked = np.zeros((4, 1), dtype=bool)
    adv, _ = gae(rewards, values, np.zeros(1), locked, gamma=0.5, lam=0.5)
    np.testing.assert_allclose(adv[:, 0], [1, 1, 1, 1])


def test_discount_applies_across_a_placement():
    """A lock at step 1 discounts everything after it, once."""
    rewards = one_env(0, 0, 0, 1)
    values = one_env(0, 0, 0, 0)
    locked = np.array([[False], [True], [False], [False]])
    adv, _ = gae(rewards, values, np.zeros(1), locked, gamma=0.5, lam=1.0)
    # Steps 2-3 are after the lock: full credit. Steps 0-1 bootstrap across it.
    np.testing.assert_allclose(adv[:, 0], [0.5, 0.5, 1.0, 1.0])


def test_stalling_is_neutral():
    """The same placement outcome, reached in 2 ticks or 6, earns the same credit."""
    fast_r, fast_l = one_env(0, 1), np.array([[False], [True]])
    slow_r = one_env(0, 0, 0, 0, 0, 1)
    slow_l = np.array([[False]] * 5 + [[True]])
    fast, _ = gae(fast_r, one_env(0, 0), np.zeros(1), fast_l, 0.9, 0.9)
    slow, _ = gae(slow_r, one_env(0, 0, 0, 0, 0, 0), np.zeros(1), slow_l, 0.9, 0.9)
    assert fast[0, 0] == pytest.approx(slow[0, 0])


def test_returns_equal_advantage_plus_value():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(16, 3)).astype(np.float32)
    values = rng.normal(size=(16, 3)).astype(np.float32)
    locked = rng.random((16, 3)) < 0.3
    adv, ret = gae(rewards, values, np.zeros(3), locked, 0.99, 0.95)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-6)


def test_matches_standard_gae_when_every_step_locks():
    """If every step is a placement, this must reduce to ordinary GAE."""
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=(20, 1)).astype(np.float32)
    values = rng.normal(size=(20, 1)).astype(np.float32)
    nv = np.array([0.3], dtype=np.float32)
    ours, _ = gae(rewards, values, nv, np.ones((20, 1), dtype=bool), 0.99, 0.95)

    ref = np.zeros(20)
    last = 0.0
    for t in reversed(range(20)):
        nxt = nv[0] if t == 19 else values[t + 1, 0]
        delta = rewards[t, 0] + 0.99 * nxt - values[t, 0]
        last = delta + 0.99 * 0.95 * last
        ref[t] = last
    np.testing.assert_allclose(ours[:, 0], ref, rtol=1e-5)


def test_lam_step_defaults_to_the_sampled_return_inside_a_piece():
    rng = np.random.default_rng(2)
    rewards = rng.normal(size=(12, 2)).astype(np.float32)
    values = rng.normal(size=(12, 2)).astype(np.float32)
    locked = rng.random((12, 2)) < 0.25
    a, _ = gae(rewards, values, np.zeros(2), locked, 0.99, 0.95)
    b, _ = gae(rewards, values, np.zeros(2), locked, 0.99, 0.95, lam_step=1.0)
    np.testing.assert_array_equal(a, b)


def test_lam_step_is_ordinary_undiscounted_gae_inside_a_piece():
    """Between locks there is no discount, so the trace is gamma=1, lambda=lam_step."""
    rng = np.random.default_rng(3)
    rewards = np.zeros((10, 1), dtype=np.float32)
    values = rng.normal(size=(10, 1)).astype(np.float32)
    nv = np.array([0.7], dtype=np.float32)
    ours, _ = gae(rewards, values, nv, np.zeros((10, 1), dtype=bool), 0.5, 0.5, lam_step=0.8)

    ref, last = np.zeros(10), 0.0
    for t in reversed(range(10)):
        nxt = nv[0] if t == 9 else values[t + 1, 0]
        last = (nxt - values[t, 0]) + 0.8 * last
        ref[t] = last
    np.testing.assert_allclose(ours[:, 0], ref, rtol=1e-5)


def test_lam_step_never_touches_the_discount():
    """With a perfect critic every TD error is zero, whatever the trace decay."""
    locked = np.array([[False], [False], [True], [False], [True]])
    rewards = one_env(0, 0, -1, 0, -2)
    # Exact per-placement values for this trajectory, with gamma = 0.5.
    values = one_env(-1 + 0.5 * -2, -1 + 0.5 * -2, -1 + 0.5 * -2, -2, -2)
    for lam_step in (1.0, 0.9, 0.5):
        adv, _ = gae(rewards, values, np.zeros(1), locked, 0.5, 0.95, lam_step=lam_step)
        np.testing.assert_allclose(adv[:, 0], 0.0, atol=1e-6)
