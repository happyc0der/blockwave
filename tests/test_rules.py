"""Difficulty curve, T-spin recognition and scoring."""

from __future__ import annotations

import pytest

from tetris.core.constants import (
    LINES_PER_LEVEL,
    LOCK_DELAY_FLOOR_MS,
    LOCK_DELAY_START_MS,
    MAX_GRAVITY_LEVEL,
    PieceType,
)
from tetris.core.piece import Piece
from tetris.core.rules import (
    TSpin,
    detect_tspin,
    gravity_seconds_per_row,
    is_difficult,
    level_for_lines,
    lock_delay_ms,
    score_lock,
)

from helpers import make_board


# -- difficulty curve — requirement 3 -------------------------------------


def test_level_one_gravity_is_one_second_per_row():
    assert gravity_seconds_per_row(1) == pytest.approx(1.0)


def test_gravity_strictly_accelerates_until_it_goes_instant():
    previous = gravity_seconds_per_row(1)
    for level in range(2, 15):
        current = gravity_seconds_per_row(level)
        assert current < previous, f"level {level} should be faster than {level - 1}"
        previous = current


def test_gravity_becomes_instant_at_high_levels():
    # Past this point the piece is on the stack the moment it spawns, and the
    # lock delay is the only thing still making the game harder.
    assert gravity_seconds_per_row(20) == 0.0


def test_lock_delay_shrinks_then_holds_at_the_floor():
    assert lock_delay_ms(1) == pytest.approx(LOCK_DELAY_START_MS)
    assert lock_delay_ms(MAX_GRAVITY_LEVEL) == pytest.approx(200.0)
    assert lock_delay_ms(50) == pytest.approx(LOCK_DELAY_FLOOR_MS)
    assert lock_delay_ms(999) >= LOCK_DELAY_FLOOR_MS


def test_lock_delay_is_monotonically_decreasing():
    values = [lock_delay_ms(level) for level in range(1, 40)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_level_advances_every_ten_lines():
    assert level_for_lines(0) == 1
    assert level_for_lines(LINES_PER_LEVEL - 1) == 1
    assert level_for_lines(LINES_PER_LEVEL) == 2
    assert level_for_lines(LINES_PER_LEVEL * 3) == 4


def test_start_level_offsets_the_curve():
    # The RL difficulty curriculum leans on this: drop an agent straight into
    # level 10 without making it clear ninety lines first.
    assert level_for_lines(0, start_level=10) == 10
    assert level_for_lines(LINES_PER_LEVEL, start_level=10) == 11


# -- T-spin recognition ---------------------------------------------------


def test_tspin_requires_a_t_piece():
    board = make_board("##########")
    piece = Piece(PieceType.L, x=3, y=21, rotation=2)
    assert detect_tspin(board, piece, True, 0) is TSpin.NONE


def test_tspin_requires_the_last_move_to_be_a_rotation():
    board = make_board(
        ".....#....",
        "###...####",
        "####.#####",
    )
    piece = Piece(PieceType.T, x=3, y=21, rotation=2)
    assert detect_tspin(board, piece, False, -1) is TSpin.NONE


def test_tspin_requires_three_filled_corners():
    board = make_board("##########")
    # A T sitting flat on the floor has only its two lower corners filled.
    piece = Piece(PieceType.T, x=3, y=21, rotation=0)
    assert detect_tspin(board, piece, True, 0) is TSpin.NONE


def test_full_tspin_when_both_front_corners_are_filled():
    board = make_board(
        ".....#....",  # the overhang that makes this a real spin setup
        "###...####",
        "####.#####",
    )
    piece = Piece(PieceType.T, x=3, y=21, rotation=2)
    assert detect_tspin(board, piece, True, 0) is TSpin.FULL


def test_mini_tspin_when_only_one_front_corner_is_filled():
    board = make_board(
        "...#......",  # one front corner filled, the other open
        "..........",
        "...#.#....",  # both back corners filled
    )
    piece = Piece(PieceType.T, x=3, y=21, rotation=0)
    assert detect_tspin(board, piece, True, 0) is TSpin.MINI


def test_final_kick_candidate_always_scores_as_a_full_tspin():
    # This override is what makes T-spin triples score correctly.
    board = make_board(
        "...#......",
        "..........",
        "...#.#....",
    )
    piece = Piece(PieceType.T, x=3, y=21, rotation=0)
    assert detect_tspin(board, piece, True, 4) is TSpin.FULL


# -- scoring --------------------------------------------------------------


@pytest.mark.parametrize(
    "lines,expected", [(1, 100), (2, 300), (3, 500), (4, 800)]
)
def test_basic_line_clear_scores(lines, expected):
    outcome = score_lock(
        lines=lines, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    assert outcome.score == expected


def test_score_scales_with_level():
    at_one = score_lock(
        lines=4, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    at_ten = score_lock(
        lines=4, tspin=TSpin.NONE, level=10, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    assert at_ten.score == at_one.score * 10


@pytest.mark.parametrize(
    "lines,expected", [(0, 400), (1, 800), (2, 1200), (3, 1600)]
)
def test_full_tspin_scores(lines, expected):
    outcome = score_lock(
        lines=lines, tspin=TSpin.FULL, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    assert outcome.score == expected


def test_tspin_double_beats_a_plain_double_by_a_wide_margin():
    plain = score_lock(
        lines=2, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    spin = score_lock(
        lines=2, tspin=TSpin.FULL, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    assert spin.score == 4 * plain.score


def test_back_to_back_multiplies_a_difficult_clear():
    solo = score_lock(
        lines=4, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    chained = score_lock(
        lines=4, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=True,
        perfect_clear=False,
    )
    assert chained.score == int(solo.score * 1.5)
    assert chained.b2b is True


def test_easy_clear_breaks_the_back_to_back_chain():
    outcome = score_lock(
        lines=1, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=True,
        perfect_clear=False,
    )
    assert outcome.b2b is False
    # ...and gets no multiplier for the clear that broke it.
    assert outcome.score == 100


def test_a_lock_without_a_clear_preserves_the_chain():
    # Placing a piece that clears nothing must not cost you a chain you have
    # already earned — only an easy clear does that.
    outcome = score_lock(
        lines=0, tspin=TSpin.NONE, level=1, combo=3, b2b_active=True,
        perfect_clear=False,
    )
    assert outcome.b2b is True
    assert outcome.combo == -1, "the combo chain does break, though"


def test_combo_adds_a_growing_bonus():
    first = score_lock(
        lines=1, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    assert first.combo == 0
    assert first.score == 100  # no bonus on the first clear of a chain

    second = score_lock(
        lines=1, tspin=TSpin.NONE, level=1, combo=0, b2b_active=False,
        perfect_clear=False,
    )
    assert second.combo == 1
    assert second.score == 100 + 50


def test_perfect_clear_pays_a_bonus():
    plain = score_lock(
        lines=4, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=False,
    )
    perfect = score_lock(
        lines=4, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=True,
    )
    assert perfect.score > plain.score
    assert perfect.perfect_clear is True


def test_perfect_clear_flag_ignored_when_nothing_cleared():
    outcome = score_lock(
        lines=0, tspin=TSpin.NONE, level=1, combo=-1, b2b_active=False,
        perfect_clear=True,
    )
    assert outcome.perfect_clear is False


@pytest.mark.parametrize(
    "lines,tspin,expected",
    [
        (0, TSpin.NONE, False),
        (0, TSpin.FULL, False),
        (1, TSpin.NONE, False),
        (3, TSpin.NONE, False),
        (4, TSpin.NONE, True),
        (1, TSpin.FULL, True),
        (1, TSpin.MINI, True),
    ],
)
def test_difficulty_classification(lines, tspin, expected):
    assert is_difficult(lines, tspin) is expected
