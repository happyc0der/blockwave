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

**A learned density does not rescue it — it makes the answer arbitrary.** With
a Gaussian over the latents of a VAE trained on board states (a density that does
not factorize over cells), the ranking between competent play and the
concentration exploits is not decided by the objective at all. Its *sign* flips
with the VAE's corpus size and training length, and not monotonically: measured
margins ranged from +0.43 (competent play wins clearly) to -0.61 (the exploit
wins clearly) across ordinary training configurations. A Gaussian density
rewards whatever the representation happens to cluster tightly.

That instability is encoded below as a test in its own right, because it is the
real finding: a reward whose preference between competent and degenerate play
depends on how long an upstream network trained is not a usable reward.

(An earlier version of this docstring concluded "it is the objective, not the
density", from a robustness check that varied only the VAE seed. Varying corpus
size and epochs overturned that.)
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



# -- a learned density makes the ranking arbitrary ------------------------


@pytest.fixture(scope="module")
def instability_corpus():
    from blockwave_rl.repr.corpus import build
    from blockwave_rl.reward.rollout import all_noop, hard_drop_spam, hold_spam, random_policy

    states, _ = build(
        {
            "random": (random_policy, 3000, False),
            "heuristic": (heuristic(), 3000, False),
            "all_noop": (all_noop, 1200, False),
            "hard_drop_spam": (hard_drop_spam, 1200, False),
            "hold_spam": (hold_spam, 1200, False),
            "early": (random_policy, 1200, True),
        },
        seed=0,
    )
    return states


def _margin_over_noop(states, epochs: int) -> float:
    from blockwave_rl.repr.board_vae import train

    vae = train(states, latent_dim=16, epochs=epochs, seed=0)
    embed = lambda board: vae.embed(board)[0]  # noqa: E731
    ref = run("heuristic", heuristic(), steps=STEPS, seed=0, embed=embed, clock="placement")
    noop = run("all_noop", EXPLOITS["all_noop"], steps=STEPS, seed=0, embed=embed, clock="placement")
    return ref.mean_reward - noop.mean_reward


def test_latent_density_ranking_flips_with_training_length(instability_corpus):
    """Same corpus, same seed, same objective — only the VAE's epochs differ.

    If the objective determined the ranking, these would agree in sign. They
    do not. This is the finding that keeps the training gate closed.
    """
    short = _margin_over_noop(instability_corpus, epochs=15)
    long = _margin_over_noop(instability_corpus, epochs=60)
    assert short > 0 > long, (
        f"expected the ranking to flip (short={short:+.3f}, long={long:+.3f}); "
        "if it no longer does, the instability finding needs re-examining"
    )
