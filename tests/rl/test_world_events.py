"""Reading placements and game-overs out of the pixels.

The board-state track took both from `info`. A pixel agent cannot, so it reads
them off the screen — and if it reads them wrongly, everything downstream is
wrong in a way nothing else would catch. These tests check the detectors
against the engine's own accounting, over several policies including the ones
built to break things.

Two real bugs are pinned here:

* **The first hold of a game is not a placement.** It draws a piece from the
  queue to replace the one it stores, so the NEXT panel shifts with nothing
  having landed. Counted as a placement it would hand a policy a free reward
  tick for pressing one key, and under per-placement discounting a tick pushes
  its future deaths further away. Later holds swap with the piece already held
  and shift nothing. A HOLD press is not disqualifying by itself: pressed while
  the hold is spent it does nothing, and gravity can land a piece on that step.
* **A line clear is not a game-over.** The board blanks the cleared rows before
  collapsing them, which is a large relative drop in filled area. Only a
  top-out empties the playfield outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockwave_rl.env.base import BlockwaveEnv, EnvConfig, ObsMode
from blockwave_rl.env.crops import Variant
from blockwave_rl.repr.pixel_gate import drift_policy
from blockwave_rl.reward.rollout import EXPLOITS, heuristic
from blockwave_rl.world.events import FrameEvents

#: The spawn of the next piece waits for the line-clear pause, so a detection
#: can legitimately land a few steps after the lock that caused it.
LAG = 4

POLICIES = {
    "random": EXPLOITS["random"],
    "drift": drift_policy(),
    "competent": heuristic(),
    "hold_spam": EXPLOITS["hold_spam"],
    "rotate_spam": EXPLOITS["rotate_spam"],
}


def rollout(policy, steps: int = 1500, seed: int = 5):
    """Returns (true placement steps, detected steps, true deaths, detected deaths)."""
    env = BlockwaveEnv(
        EnvConfig(
            obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
            agent_hz=5, gravity_scale=4.0, frame_stack=1,
        )
    )
    env.reset(seed=seed)
    events = FrameEvents(env.layout, env.crop)
    events.calibrate(env._pixels())
    rng = np.random.default_rng(0)
    previous = env._pixels()
    truth, detected, deaths, found = [], [], [], []
    placed = 0
    for step in range(steps):
        action = policy(env.engine, rng, step)
        *_, info = env.step(action)
        current = env._pixels()
        # A new piece enters play after a lock *and* after a reset, which is
        # how the board-state env counts placements too.
        if env.engine.stats.pieces_placed != placed or info["top_out"]:
            truth.append(step)
        placed = env.engine.stats.pieces_placed
        if info["top_out"]:
            deaths.append(step)
        saw_placement, saw_end = events.observe(previous, current, action)
        if saw_placement:
            detected.append(step)
        if saw_end:
            found.append(step)
        previous = current
    return truth, detected, deaths, found


def unmatched(reference: list[int], candidates: list[int], lag: int = LAG) -> int:
    """How many of ``reference`` have no candidate within ``lag`` steps after."""
    pool = sorted(candidates)
    misses = 0
    for step in reference:
        hit = next((c for c in pool if step <= c <= step + lag), None)
        if hit is None:
            misses += 1
        else:
            pool.remove(hit)
    return misses


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_every_placement_is_seen_and_nothing_else_is(name):
    truth, detected, _, _ = rollout(POLICIES[name])
    assert truth, "the policy placed no pieces: the test would be vacuous"
    assert unmatched(truth, detected) == 0, "a placement went unseen"
    assert abs(len(detected) - len(truth)) <= max(2, int(0.02 * len(truth))), "spurious placements"


def test_holding_is_not_counted_as_a_placement():
    """A policy that presses HOLD constantly must not bank extra reward ticks."""
    truth, detected, _, _ = rollout(POLICIES["hold_spam"], steps=2500)
    assert len(detected) <= len(truth) + 2


@pytest.mark.parametrize("name", ["random", "drift", "rotate_spam"])
def test_every_game_over_is_seen_and_nothing_else_is(name):
    _, _, deaths, found = rollout(POLICIES[name], steps=3000)
    assert deaths, "the policy never topped out: the test would be vacuous"
    assert unmatched(deaths, found, lag=0) == 0
    assert len(found) == len(deaths)


def test_line_clears_are_not_game_overs():
    """The competent player clears constantly and never tops out."""
    _, _, deaths, found = rollout(POLICIES["competent"], steps=3000)
    assert not deaths
    assert not found


def test_a_crop_without_the_next_panel_is_refused():
    """BOARD_ONLY cannot carry this reward, and says so rather than misreading."""
    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_ONLY, cell_px=4))
    with pytest.raises(ValueError, match="not inside this crop"):
        FrameEvents(env.layout, env.crop)
