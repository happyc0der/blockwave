"""Bitboard correctness, line clearing, and stack metrics."""

from __future__ import annotations

from tetris.core.board import Board
from tetris.core.constants import BOARD_WIDTH, FULL_ROW, TOTAL_HEIGHT, PieceType

from helpers import make_board


def test_new_board_is_empty():
    board = Board()
    assert not any(board.rows)
    assert board.colors.sum() == 0
    board.check_invariants()


def test_collision_detects_walls_floor_and_ceiling():
    board = Board()
    assert board.collides(((-1, 5),))
    assert board.collides(((BOARD_WIDTH, 5),))
    assert board.collides(((5, TOTAL_HEIGHT),))
    assert board.collides(((5, -1),))
    assert not board.collides(((0, 0), (9, TOTAL_HEIGHT - 1)))


def test_is_occupied_treats_out_of_bounds_as_solid():
    # The T-spin corner test relies on this: walls and floor count as filled,
    # so a spin in the corner of the board scores the same as one mid-board.
    board = Board()
    assert board.is_occupied(-1, 10)
    assert board.is_occupied(BOARD_WIDTH, 10)
    assert board.is_occupied(5, TOTAL_HEIGHT)
    assert not board.is_occupied(5, 10)


def test_lock_updates_both_representations():
    board = Board()
    board.lock(((0, 23), (1, 23)), PieceType.T)
    assert board.rows[23] == 0b11
    assert board.colors[23, 0] == int(PieceType.T)
    assert board.colors[23, 1] == int(PieceType.T)
    board.check_invariants()


def test_clear_single_row_drops_stack_down():
    board = make_board(
        ".........#",
        "##########",
    )
    full = board.full_rows()
    assert full == [TOTAL_HEIGHT - 1]

    board.clear_rows(full)
    board.check_invariants()

    # The lone block was above the cleared row, so it should now be on the floor.
    assert board.rows[TOTAL_HEIGHT - 1] == 1 << 9
    assert board.rows[TOTAL_HEIGHT - 2] == 0


def test_clear_multiple_non_adjacent_rows():
    board = make_board(
        "##########",
        "#.........",
        "##########",
    )
    board.clear_rows(board.full_rows())
    board.check_invariants()

    assert board.rows[TOTAL_HEIGHT - 1] == 1
    assert board.rows[TOTAL_HEIGHT - 2] == 0
    assert sum(1 for row in board.rows if row) == 1


def test_clear_preserves_colors_of_surviving_rows():
    board = Board()
    board.lock(tuple((x, TOTAL_HEIGHT - 1) for x in range(BOARD_WIDTH)), PieceType.I)
    board.lock(((3, TOTAL_HEIGHT - 2),), PieceType.S)

    board.clear_rows(board.full_rows())
    board.check_invariants()

    assert board.colors[TOTAL_HEIGHT - 1, 3] == int(PieceType.S)
    assert board.rows[TOTAL_HEIGHT - 1] == 1 << 3


def test_full_row_constant_matches_a_filled_row():
    board = Board()
    board.lock(tuple((x, 5) for x in range(BOARD_WIDTH)), PieceType.O)
    assert board.rows[5] == FULL_ROW


def test_column_heights_and_aggregate():
    board = make_board(
        "#.........",
        "#........#",
    )
    heights = board.column_heights()
    assert heights[0] == 2
    assert heights[9] == 1
    assert heights[5] == 0
    assert board.aggregate_height() == 3
    assert board.max_height() == 2


def test_holes_counts_only_covered_gaps():
    board = make_board(
        "#.........",
        "..........",
        "#........#",
    )
    # Column 0 has a filled cell, then a gap, then filled: exactly one hole.
    # Column 9's block has nothing above it, so it is not a hole.
    assert board.holes() == 1


def test_bumpiness_is_zero_for_a_flat_stack():
    board = make_board("##########")
    assert board.bumpiness() == 0

    board = make_board("#.........")
    # One column of height 1 next to a column of height 0, and nothing else.
    assert board.bumpiness() == 1


def test_invariant_check_catches_desync():
    board = Board()
    board.rows[10] = 0b1  # set the bitboard without the colour grid
    try:
        board.check_invariants()
    except AssertionError as exc:
        assert "desync" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("check_invariants failed to notice a desync")
