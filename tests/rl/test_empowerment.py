"""Empowerment: the first intrinsic objective that ranks competent play correctly.

SMiRL rewards predictability, and on a full-width board a doomed repetitive
policy is predictable — so SMiRL rewarded it, under every density tried.
Empowerment rewards control over the future: how many distinct outcomes the
agent's actions can still reach. Within one piece's life Tetris is
deterministic, so the channel capacity reduces exactly to
log(#distinct reachable outcomes).

These tests pin three findings:

1. **One step ahead it is blind.** An empty board, a low stack and a sixteen-
   high tower offer identical placement counts — a tower removes no choices, a
   piece just lands higher on it. Options only vanish at death.
2. **A longer horizon sees danger coming,** monotonically: the taller the
   tower, the fewer futures survive.
3. **It ranks competent play above every exploit**, including the three
   concentration exploits that beat SMiRL. Consistent in sign across seeds —
   but the margin is small (~1-2%), because empowerment saturates on any board
   with headroom and every policy spends time on those.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockwave.core.constants import PieceType
from blockwave_rl.reward.empowerment import (
    board_empowerment,
    horizon_empowerment,
    occupancy_to_rows,
    placement_count,
)
from blockwave_rl.reward.rollout import EXPLOITS, heuristic, run_empowerment

QUEUE = [PieceType.T, PieceType.S, PieceType.L]


def board(fill: dict[tuple[slice, slice], int] | None = None) -> list[int]:
    grid = np.zeros((20, 10), dtype=np.uint8)
    for (rows, cols), value in (fill or {}).items():
        grid[rows, cols] = value
    return occupancy_to_rows(grid)


EMPTY = board()
FLAT = board({(slice(-3, None), slice(0, 9)): 1})
TOWER_16 = board({(slice(4, None), slice(3, 7)): 1})
TOWER_18 = board({(slice(2, None), slice(3, 7)): 1})
AT_SPAWN = board({(slice(1, None), slice(3, 7)): 1})


# -- finding 1: one step ahead, it is blind -------------------------------


def test_one_step_empowerment_cannot_see_a_tall_tower():
    assert board_empowerment(EMPTY) == board_empowerment(FLAT) == board_empowerment(TOWER_16)


def test_placement_counts_are_the_standard_figures():
    # Rotation-and-column placements on an empty board: the familiar numbers.
    counts = {p: placement_count(EMPTY, p) for p in PieceType}
    assert counts[PieceType.O] == 9
    assert counts[PieceType.I] == 17
    assert counts[PieceType.T] == counts[PieceType.J] == counts[PieceType.L] == 34
    assert counts[PieceType.S] == counts[PieceType.Z] == 17


def test_a_topped_out_placement_is_not_a_future():
    full = [0b1111111111] * 24
    assert placement_count(full, PieceType.O) == 0


# -- finding 2: a longer horizon sees danger -------------------------------


def test_longer_horizon_orders_boards_by_danger():
    values = [horizon_empowerment(rows, QUEUE) for rows in (EMPTY, TOWER_16, TOWER_18, AT_SPAWN)]
    assert values == sorted(values, reverse=True), values
    assert values[0] > values[-1]


def test_headroom_saturates():
    """A board with plenty of room genuinely offers as much control as an empty
    one over a short horizon. Correct behaviour — and the reason the margin
    against exploits is small."""
    assert horizon_empowerment(EMPTY, QUEUE) == horizon_empowerment(FLAT, QUEUE)


def test_horizon_is_needed_to_separate_the_tower():
    assert horizon_empowerment(EMPTY, QUEUE[:1]) == horizon_empowerment(TOWER_18, QUEUE[:1])
    assert horizon_empowerment(EMPTY, QUEUE) > horizon_empowerment(TOWER_18, QUEUE)


# -- finding 3: it ranks competent play above every exploit ----------------


@pytest.fixture(scope="module")
def reference():
    return run_empowerment("heuristic", heuristic(), horizon=2, max_scored=300, seed=0)


@pytest.mark.parametrize("name", sorted(EXPLOITS))
def test_heuristic_beats_every_exploit(reference, name):
    """Including all_noop, hold_spam and hard_drop_spam — the three that beat
    SMiRL under both a per-cell and a learned density."""
    exploit = run_empowerment(name, EXPLOITS[name], horizon=2, max_scored=300, seed=0)
    assert reference.mean_reward > exploit.mean_reward, (
        f"{name} ({exploit.mean_reward:.4f}) beat competent play ({reference.mean_reward:.4f})"
    )


# -- the queue-corrected reward --------------------------------------------


def test_corrected_reward_is_zero_on_an_empty_board():
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=2)
    for queue in ([PieceType.I, PieceType.O], [PieceType.T, PieceType.S], [PieceType.Z, PieceType.L]):
        assert reward(EMPTY, queue) == 0.0


def test_corrected_reward_penalizes_danger():
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=3)
    assert reward(FLAT, QUEUE) == 0.0, "headroom means full control"
    assert reward(TOWER_18, QUEUE) < 0.0
    assert reward(AT_SPAWN, QUEUE) < reward(TOWER_18, QUEUE)


def test_correction_removes_queue_noise_without_reordering():
    """The baseline depends only on the queue, so it cannot change a ranking."""
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=2)
    queues = [[PieceType.I, PieceType.O], [PieceType.T, PieceType.T], [PieceType.O, PieceType.O]]
    raw_empty = [horizon_empowerment(EMPTY, q) for q in queues]
    assert max(raw_empty) - min(raw_empty) > 1.0, "raw empowerment swings with the queue"
    assert all(reward(EMPTY, q) == 0.0 for q in queues), "corrected does not"


# -- death accounting ------------------------------------------------------
#
# A better learner (per-tick credit, `lam_step`) found that the first
# accounting made dying fast the best use of a doomed board: it spent 1.8
# placements in danger before each death, against drift's 5.9. These tests pin
# the flaw and the fix as properties of the reward, independent of any learner.

GAMMA = 0.99
BOARDS = {"empty": EMPTY, "flat": FLAT, "tower_16": TOWER_16, "tower_18": TOWER_18, "at_spawn": AT_SPAWN}


def _one_more_then_die_minus_die_now(reward, rows, gamma_for_death, worst: bool = True):
    """(place, then die next) - (die now), worst case or mean over current queues."""
    from blockwave_rl.reward.empowerment import bag_windows

    windows = bag_windows(2)
    next_death = sum(p * reward.death(list(w), gamma_for_death) for w, p in windows.items())
    gaps = {q: reward(rows, list(q)) + GAMMA * next_death - reward.death(list(q), gamma_for_death) for q in windows}
    if worst:
        return min(gaps.values())
    return sum(p * gaps[q] for q, p in windows.items())


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_absorbing_death_never_rewards_dying_sooner(name):
    """On every board and every queue, one more placement alive is worth at least dying now."""
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=2)
    assert _one_more_then_die_minus_die_now(reward, BOARDS[name], GAMMA) >= -1e-9


def test_one_step_death_rewards_dying_sooner_on_a_dangerous_board():
    """The flaw, recorded: charged once, death is cheaper than lingering in danger.

    In expectation over the queue, so this is the board speaking, not the queue.
    (Worst case over the current queue it fails even on an empty board: dying
    while holding a cheap queue like O-O costs less than an average later death.)
    """
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=2)
    assert _one_more_then_die_minus_die_now(reward, AT_SPAWN, None, worst=False) < 0
    assert _one_more_then_die_minus_die_now(reward, EMPTY, None, worst=False) > 0, "headroom: no incentive"


def test_absorbing_death_is_the_dead_games_discounted_stream():
    from blockwave_rl.reward.empowerment import HorizonEmpowermentReward

    reward = HorizonEmpowermentReward(horizon=2)
    q = [PieceType.T, PieceType.S]
    once = reward.death(q)
    assert once == -reward._baseline(tuple(q))
    stream = once - GAMMA * reward.mean_baseline() / (1 - GAMMA)
    assert reward.death(q, GAMMA) == pytest.approx(stream)
    assert reward.death(q, GAMMA) < 50 * once, "about 100 placements of zero control"


def test_bag_windows_are_exact():
    from blockwave_rl.reward.empowerment import bag_windows

    for horizon in (1, 2, 3):
        assert sum(bag_windows(horizon).values()) == pytest.approx(1.0)
    pairs = bag_windows(2)
    # Within a bag consecutive pieces differ; only a bag boundary (1 in 7) repeats.
    assert pairs[(PieceType.T, PieceType.T)] == pytest.approx(1 / 343)
    assert pairs[(PieceType.T, PieceType.S)] == pytest.approx(1 / 49 + 1 / 343)


def test_bag_windows_match_the_real_randomizer():
    from collections import Counter

    from blockwave.core.randomizer import SevenBag
    from blockwave_rl.reward.empowerment import bag_windows

    bag = SevenBag(seed=3)
    seq = [bag.next() for _ in range(70_001)]
    counts = Counter(zip(seq, seq[1:]))
    same = sum(v for (a, b), v in counts.items() if a == b) / (len(seq) - 1)
    assert same == pytest.approx(7 / 343, abs=0.003)
    assert set(counts) == set(bag_windows(2))


# -- the placement enumerator against its own reference ---------------------


def test_fast_placements_match_the_reference_enumerator():
    """`_placements` is the hand-optimised version; `_placements_reference` is the
    straightforward one its docstring says it is kept to be checked against.
    Until this test existed, nothing checked it.

    Random low stacks with the top rows clear, so every piece can spawn and most
    placements survive; the two must agree on the exact set of resulting boards,
    line clears included.
    """
    from blockwave.core.constants import BOARD_WIDTH, PieceType, TOTAL_HEIGHT

    from blockwave_rl.reward.empowerment import _placements, _placements_reference

    rng = np.random.default_rng(0)
    full = (1 << BOARD_WIDTH) - 1
    compared = 0
    for _ in range(40):
        depth = int(rng.integers(0, 12))
        rows = [0] * (TOTAL_HEIGHT - depth)
        for _ in range(depth):
            row = int(rng.integers(0, full + 1))
            if row == full:              # a pre-filled full row would clear on spawn
                row &= ~(1 << int(rng.integers(BOARD_WIDTH)))
            rows.append(row)
        board = tuple(rows)
        for piece in PieceType:
            fast = set(_placements(board, piece))
            slow = set(_placements_reference(board, piece))
            assert fast == slow, (
                f"{piece.name} on a {depth}-deep stack: fast enumerator disagrees with "
                f"reference ({len(fast)} vs {len(slow)} boards)"
            )
            compared += len(fast)
    assert compared > 1000, "the sample was too shallow to mean anything"
