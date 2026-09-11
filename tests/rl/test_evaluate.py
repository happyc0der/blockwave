"""The yardstick itself. A miscounting tracker would make every curve a lie."""

from __future__ import annotations

import numpy as np

from blockwave.core.constants import Action
from blockwave_rl.evaluate import BASELINES, GameTracker


def info(lines: int, pieces: int, top_out: bool = False) -> dict:
    return {"lines": lines, "pieces": pieces, "top_out": top_out}


def test_lines_survive_a_soft_reset():
    """A top-out zeroes the per-game count; the lines already made still count."""
    t = GameTracker(1)
    for update in ([info(2, 10)], [info(5, 20)], [info(0, 0, top_out=True)], [info(1, 3)]):
        t.update(update)
    assert t.lines == 6
    assert t.pieces == 23
    assert t.top_outs == 1


def test_envs_are_tracked_separately():
    t = GameTracker(2)
    t.update([info(1, 4), info(0, 4)])
    t.update([info(1, 8), info(3, 8)])
    assert list(t.env_lines) == [1, 3]
    assert list(t.env_pieces) == [8, 8]
    assert t.summary()["lines_per_piece"] == 4 / 16


def test_a_window_reset_keeps_the_running_counts():
    """Only the window's totals clear; the per-env baselines must not re-count."""
    t = GameTracker(1)
    t.update([info(4, 10)])
    t.reset_window()
    t.update([info(4, 12)])
    assert t.lines == 0 and t.pieces == 2


def test_standard_error_shrinks_with_more_envs():
    rng = np.random.default_rng(0)

    def se(n: int) -> float:
        t = GameTracker(n)
        t.env_pieces = rng.poisson(100, n).astype(float)
        t.env_lines = rng.binomial(t.env_pieces.astype(int), 0.1).astype(float)
        return t.standard_errors()["lines_per_piece"]

    assert se(400) < se(25) / 2


def test_identical_envs_have_zero_error():
    t = GameTracker(4)
    t.env_pieces[:] = 50
    t.env_lines[:] = 5
    assert t.standard_errors()["lines_per_piece"] == 0.0


def test_drift_is_random_without_hard_drop():
    assert set(BASELINES["random"]) == {int(a) for a in Action}
    assert set(BASELINES["drift"]) == set(BASELINES["random"]) - {int(Action.HARD_DROP)}


def feed(tracker: GameTracker, events: list[tuple[int, int, bool, float, bool]]) -> None:
    """One env: (lines, pieces, top_out, reward, locked) per step."""
    for lines, pieces, top_out, reward, locked in events:
        tracker.update([info(lines, pieces, top_out)], np.array([reward]), np.array([locked]))


def test_complete_games_drop_both_partial_edges():
    t = GameTracker(1)
    feed(t, [
        (1, 5, False, -1.0, True),      # partial game the window started inside
        (0, 0, True, -6.0, True),       # its death: the complete span starts here
        (2, 10, False, -0.5, True),     # one complete game...
        (0, 0, True, -6.0, True),       # ...ending in a death
        (1, 3, False, -9.0, True),      # partial game the window ends inside
    ])
    g = t.complete_games()
    assert g["games"] == 1
    assert g["lines_per_piece"] == 2 / 10
    assert g["pieces_per_game"] == 10
    # The death step belongs to the game it ended; the partial edges do not count.
    assert g["intrinsic_per_placement"] == (-0.5 - 6.0) / 2


def test_an_env_with_one_top_out_contributes_nothing():
    t = GameTracker(2)
    t.update([info(0, 4, False), info(0, 4, False)])
    t.update([info(0, 0, True), info(0, 0, True)])
    t.update([info(1, 9, False), info(0, 9, False)])
    t.update([info(0, 0, True), info(0, 12, False)])
    g = t.complete_games()
    assert g["games"] == 1
    assert g["min_games_per_env"] == 0
    assert g["lines_per_piece"] == 1 / 9


def test_no_complete_games_is_nan_not_zero():
    t = GameTracker(1)
    t.update([info(3, 30, False)])
    g = t.complete_games()
    assert g["games"] == 0
    assert np.isnan(g["lines_per_piece"]) and np.isnan(g["top_outs_per_piece"])


def test_burn_in_is_played_but_not_measured():
    from blockwave_rl.evaluate import baseline_actor, play

    result = play(baseline_actor("random", seed=1), steps_per_env=60, burn_in=40, envs=2, workers=1)
    assert result["steps"] == 2 * 60
