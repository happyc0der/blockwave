"""The reward arithmetic: causality, bounded memory, and numerical edge cases.

Every one of these is a way a density reward fails without crashing — it keeps
producing numbers, they are just the wrong numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from blockwave_rl.reward.density import BernoulliDensity, GaussianDensity
from blockwave_rl.reward.smirl import SmirlConfig, SmirlReward

BOARD = (20, 10)


# -- causality ------------------------------------------------------------


def test_gaussian_scores_under_the_previous_parameters():
    """r_t must use θ_{t-1}. Proven by showing the non-causal order differs."""
    rng = np.random.default_rng(0)
    history = rng.normal(size=(50, 4))
    x = np.array([3.0, -3.0, 3.0, -3.0])  # an unusual point

    causal = GaussianDensity(4, window=None)
    for h in history:
        causal.update(h)
    expected = causal.log_prob(x)                 # θ_{t-1}
    assert causal.score_then_update(x) == pytest.approx(expected)

    leaky = GaussianDensity(4, window=None)
    for h in history:
        leaky.update(h)
    leaky.update(x)                               # absorb x first...
    non_causal = leaky.log_prob(x)                # ...then score it

    assert non_causal > expected, (
        "scoring after the update must look less surprising — if these were equal "
        "the test would not be distinguishing the two orderings at all"
    )


def test_bernoulli_scores_under_the_previous_parameters():
    density = BernoulliDensity(BOARD, window=None)
    empty = np.zeros(BOARD)
    for _ in range(30):
        density.update(empty)
    full_row = empty.copy()
    full_row[-1] = 1
    before = density.log_prob(full_row)
    assert density.score_then_update(full_row) == pytest.approx(before)


# -- numerical edge cases --------------------------------------------------


def test_identical_repeated_latents_stay_finite():
    """A constant trajectory drives variance to zero; the floor must hold."""
    density = GaussianDensity(8, window=64, var_floor=1e-4)
    z = np.full(8, 0.37)
    values = [density.score_then_update(z) for _ in range(5_000)]
    assert all(math.isfinite(v) for v in values)
    assert density.var.min() >= 1e-4
    # Bounded above: per-dim log-likelihood of a point at the mean with var=floor.
    ceiling = -0.5 * (math.log(2 * math.pi) + math.log(1e-4)) * 8
    assert max(values) <= ceiling + 1e-6


def test_long_constant_board_stays_finite():
    density = BernoulliDensity(BOARD, window=256)
    board = np.zeros(BOARD)
    values = [density.score_then_update(board) for _ in range(20_000)]
    assert all(math.isfinite(v) for v in values)
    assert density.p.min() >= density.eps and density.p.max() <= 1 - density.eps


def test_all_empty_and_all_full_boards_are_finite():
    density = BernoulliDensity(BOARD)
    for _ in range(200):
        density.update(np.zeros(BOARD))
    assert math.isfinite(density.score_then_update(np.ones(BOARD)))
    assert math.isfinite(density.score_then_update(np.zeros(BOARD)))


def test_first_step_is_finite_and_warmup_is_silent():
    reward = SmirlReward.for_board(BOARD, SmirlConfig(warmup=8))
    board = np.zeros(BOARD)
    early = [reward(board) for _ in range(8)]
    assert early == [0.0] * 8, "reward must be held at zero while θ is meaningless"
    assert math.isfinite(reward(board))


def test_reward_is_clipped():
    reward = SmirlReward.for_latent(4, SmirlConfig(warmup=0, clip=10.0))
    for _ in range(500):
        reward(np.zeros(4))
    extreme = reward(np.full(4, 1e6))
    assert extreme == -10.0


def test_extreme_inputs_never_produce_nan():
    reward = SmirlReward.for_latent(6, SmirlConfig(warmup=0))
    for value in (0.0, 1e-12, -1e-12, 1e6, -1e6, 1e-300):
        assert math.isfinite(reward(np.full(6, value)))


def test_invalid_floors_are_rejected():
    with pytest.raises(ValueError):
        GaussianDensity(4, var_floor=0.0)
    with pytest.raises(ValueError):
        BernoulliDensity(BOARD, eps=0.0)
    with pytest.raises(ValueError):
        BernoulliDensity(BOARD, eps=0.6)


# -- bounded memory -------------------------------------------------------


def test_bounded_memory_forgets_early_chaos():
    """The failure unbounded memory causes: early chaos defines normal forever."""
    rng = np.random.default_rng(1)
    chaotic = (rng.random((2_000, *BOARD)) < 0.5).astype(float)
    tidy = np.zeros(BOARD)

    bounded = BernoulliDensity(BOARD, window=200)
    unbounded = BernoulliDensity(BOARD, window=None)
    for frame in chaotic:
        bounded.update(frame)
        unbounded.update(frame)
    for _ in range(600):  # the agent gets good
        bounded.update(tidy)
        unbounded.update(tidy)

    assert bounded.log_prob(tidy) > unbounded.log_prob(tidy), (
        "after sustained tidy play, a bounded model should find tidy boards more "
        "normal than an unbounded one still anchored to the chaos"
    )


def test_warmup_is_an_exact_running_mean():
    density = GaussianDensity(3, window=1_000)
    points = np.random.default_rng(2).normal(size=(10, 3))
    for p in points:
        density.update(p)
    np.testing.assert_allclose(density.mu, points.mean(axis=0), rtol=1e-12)


# -- the model of normal is not reset on death -----------------------------


def test_density_state_is_fixed_shape():
    gauss = GaussianDensity(16)
    for _ in range(5):
        gauss.update(np.random.default_rng(3).normal(size=16))
    state = gauss.state()
    assert state.mean.shape == state.log_scale.shape == (16,)
    assert np.all(np.isfinite(state.log_scale))

    bern = BernoulliDensity(BOARD)
    bern.update(np.zeros(BOARD))
    assert bern.state().mean.shape == BOARD
    assert np.all(np.isfinite(bern.state().log_scale))


def test_progress_is_bounded():
    reward = SmirlReward.for_board(BOARD, SmirlConfig(window=100))
    for _ in range(1_000):
        reward(np.zeros(BOARD))
    assert reward.progress() == 1.0
