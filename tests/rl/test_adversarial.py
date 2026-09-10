"""The reward-hacking suite. No training run launches until this is green.

A competent scripted player must out-score every exploit under the intrinsic
reward, with each policy's density fit online on its own experience exactly as
it would be during training.

**Current state, recorded rather than hidden:** the factorized-Bernoulli board
density (the paper's own Tetris model) is hackable on a 10-wide board. Policies
that never move a piece build a narrow tower at the spawn columns, leaving six
of ten columns permanently empty. A per-cell density predicts those empty cells
perfectly — they cost nothing — and that free likelihood outweighs what the
doomed tower costs. Measured: edge columns cost the tower -0.006 against the
heuristic's -1.362, while the centre costs it -1.808 against -0.830.

The paper's board was 4 wide, so there were no spare columns to exploit.

The concentration exploits are therefore `xfail(strict=True)`: if a change makes
them pass, the test flips and forces someone to notice and update this record.
"""

from __future__ import annotations

import pytest

from blockwave_rl.reward.rollout import EXPLOITS, heuristic, run

STEPS = 8_000

#: Beaten by the heuristic under the per-cell density.
CLOSED = ["random", "oscillate", "rotate_spam"]

#: Win by spatial concentration under a density that factorizes over cells.
CONCENTRATION = ["all_noop", "hold_spam", "hard_drop_spam"]


@pytest.fixture(scope="module")
def reference():
    return run("heuristic", heuristic(), steps=STEPS, seed=0)


@pytest.mark.parametrize("name", CLOSED)
def test_heuristic_beats_exploit(reference, name):
    exploit = run(name, EXPLOITS[name], steps=STEPS, seed=0)
    assert reference.mean_reward > exploit.mean_reward, (
        f"{name} ({exploit.mean_reward:.4f}) out-scores competent play "
        f"({reference.mean_reward:.4f}) — the reward is hackable"
    )


@pytest.mark.xfail(
    strict=True,
    reason="factorized per-cell density rewards spatial concentration on a 10-wide board",
)
@pytest.mark.parametrize("name", CONCENTRATION)
def test_heuristic_beats_concentration_exploit(reference, name):
    exploit = run(name, EXPLOITS[name], steps=STEPS, seed=0)
    assert reference.mean_reward > exploit.mean_reward


def test_rotate_spam_actually_places_pieces():
    """Regression guard for the engine bug this suite found.

    Holding rotate used to hold a piece aloft forever — zero pieces locked in
    8,000 ticks — which handed rotate-spam a perfectly static board and a
    near-maximal reward. That was a game bug, fixed in the engine.
    """
    result = run("rotate_spam", EXPLOITS["rotate_spam"], steps=STEPS, seed=0)
    assert result.mean_reward < -0.05, "a static board would score near zero"
