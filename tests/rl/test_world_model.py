"""The learned model, and the count of distinct futures built on it.

The board-state reward could enumerate futures exactly. This one predicts them,
so two things can go wrong quietly: the model can be too blunt to tell its own
key presses apart, and the count can stop counting.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from blockwave_rl.world.babble import Babble
from blockwave_rl.world.empowerment import PixelEmpowerment, effective_count
from blockwave_rl.world.macros import MACROS, N_MACROS, Macro
from blockwave_rl.world.model import MacroModel, report, train

from blockwave.core.constants import Action


# -- the vocabulary -------------------------------------------------------


def test_every_macro_is_a_runnable_key_sequence():
    for macro in MACROS:
        actions = macro.actions
        assert all(0 <= a < len(Action) for a in actions)
        assert actions[-1] == int(Action.HARD_DROP), "a macro must resolve the piece"


def test_the_vocabulary_has_no_duplicates():
    assert len({m.actions for m in MACROS}) == N_MACROS


def test_a_macro_needs_no_knowledge_of_the_board():
    """Steps are blind: into a wall they simply do nothing."""
    assert Macro(0, -1, 5).actions == (int(Action.LEFT),) * 5 + (int(Action.HARD_DROP),)
    assert Macro(2, 0, 0).actions == (int(Action.ROTATE_CW),) * 2 + (int(Action.HARD_DROP),)


# -- counting distinct futures --------------------------------------------


def test_identical_futures_count_as_one():
    points = torch.zeros(1, 8, 4)
    weights = torch.ones(1, 8)
    assert float(effective_count(points, weights, width=1.0)) == pytest.approx(1.0, rel=1e-3)


def test_separated_futures_count_as_many():
    points = (torch.arange(8, dtype=torch.float32) * 100).reshape(1, 8, 1).repeat(1, 1, 4)
    weights = torch.ones(1, 8)
    assert float(effective_count(points, weights, width=1.0)) == pytest.approx(8.0, rel=1e-3)


def test_futures_that_end_the_game_do_not_count():
    points = (torch.arange(8, dtype=torch.float32) * 100).reshape(1, 8, 1).repeat(1, 1, 4)
    alive = torch.ones(1, 8)
    alive[0, 4:] = 0.0
    assert float(effective_count(points, alive, width=1.0)) == pytest.approx(4.0, rel=1e-3)


def test_a_wider_kernel_counts_fewer_futures():
    """Two futures closer than the model can resolve are one future."""
    points = (torch.arange(6, dtype=torch.float32)).reshape(1, 6, 1).repeat(1, 1, 3)
    weights = torch.ones(1, 6)
    sharp = float(effective_count(points, weights, width=0.05))
    blunt = float(effective_count(points, weights, width=50.0))
    assert sharp > 5.5 and blunt < 1.5


# -- the model ------------------------------------------------------------


def fake_babble(n: int = 2048, dim: int = 8, seed: int = 0) -> Babble:
    """Each macro shifts the latent its own way, so a model can learn it."""
    rng = np.random.default_rng(seed)
    shifts = rng.normal(size=(N_MACROS, dim)).astype(np.float32) * 3.0
    before = rng.normal(size=(n, dim)).astype(np.float32)
    macro = rng.integers(N_MACROS, size=n)
    after = before + shifts[macro] + rng.normal(scale=0.05, size=(n, dim)).astype(np.float32)
    # The tallest boards are the fatal ones, so death is learnable too.
    died = before[:, 0] > 1.5
    return Babble(before, macro, after, died)


def test_the_model_learns_to_tell_its_own_keys_apart():
    fit, held_out = fake_babble().split(0.9, seed=1)
    model = train(fit, epochs=30, seed=0)
    quality = report(model, held_out)
    assert quality["skill"] > 0.9, "a learnable shift was not learned"
    assert quality["macro_top1"] > 0.8, "the model cannot identify which macro ran"
    assert quality["macro_rank"] < 1.0


def test_a_state_where_everything_is_fatal_has_no_choice_left():
    """One outcome, its own: log(1) = 0, matching the exact reward's dead board."""
    model = MacroModel(latent_dim=4)
    with torch.no_grad():
        model.death.bias.fill_(20.0)  # everything kills
    reward = PixelEmpowerment(model, horizon=1, width=1.0)
    reward.calibrate(np.zeros((1, 4), dtype=np.float32), width=1.0)
    assert reward.raw(np.zeros((1, 4), dtype=np.float32))[0] == pytest.approx(0.0)


def test_the_death_charge_makes_one_more_placement_never_worse():
    """Section 3's finding, carried over: charged once, dying fast pays."""
    model = MacroModel(latent_dim=4)
    reward = PixelEmpowerment(model, horizon=1, width=1.0)
    reward.calibrate(np.zeros((1, 4), dtype=np.float32), width=1.0)
    reward._baseline = 1.25
    gamma = 0.99
    absorbing = reward.death_charge(gamma)
    once = -reward.baseline
    # The worst a live placement can score is minus the baseline.
    assert once + gamma * absorbing >= absorbing - 1e-9, "absorbing: lingering is never worse"
    assert once + gamma * once < once, "charged once: dying now beats one more placement"


def test_the_reward_is_zero_on_the_board_it_was_calibrated_against():
    model = MacroModel(latent_dim=4)
    empty = np.zeros((1, 4), dtype=np.float32)
    reward = PixelEmpowerment(model, horizon=2, width=1.0, branch=4)
    reward.calibrate(empty, width=1.0)
    assert reward(empty)[0] == pytest.approx(0.0, abs=1e-6)


def test_the_reward_refuses_to_run_before_calibration():
    reward = PixelEmpowerment(MacroModel(latent_dim=4), horizon=1)
    with pytest.raises(RuntimeError, match="calibrate"):
        reward(np.zeros((1, 4), dtype=np.float32))


def test_the_reward_is_the_same_twice():
    """Two evaluations of one state must agree: a noisy reward is a worse one."""
    model = MacroModel(latent_dim=4)
    reward = PixelEmpowerment(model, horizon=2, width=1.0, branch=6)
    reward.calibrate(np.zeros((1, 4), dtype=np.float32), width=1.0)
    state = np.linspace(-1, 1, 4, dtype=np.float32)[None]
    assert reward(state)[0] == reward(state)[0]
