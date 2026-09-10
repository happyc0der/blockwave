"""SRS rotation and wall kicks.

Getting SRS wrong is the classic way a falling-block game feels subtly bad
without ever visibly crashing, so these tests check the kick tables structurally
*and* exercise real kicks against real boards.
"""

from __future__ import annotations

import pytest

from blockwave.core.board import Board
from blockwave.core.constants import (
    KICKS_I,
    LAST_KICK_INDEX,
    KICKS_JLSTZ,
    SHAPES,
    PieceType,
    kick_table,
)
from blockwave.core.piece import Piece, try_rotate

from helpers import make_board

ALL_TRANSITIONS = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]


# -- table structure ------------------------------------------------------


@pytest.mark.parametrize("table", [KICKS_JLSTZ, KICKS_I], ids=["jlstz", "i"])
def test_kick_table_covers_every_transition(table):
    assert set(table) == set(ALL_TRANSITIONS)
    for transition, offsets in table.items():
        assert len(offsets) == 5, f"{transition} should have 5 candidates"
        assert offsets[0] == (0, 0), f"{transition} must try the identity first"


@pytest.mark.parametrize("table", [KICKS_JLSTZ, KICKS_I], ids=["jlstz", "i"])
@pytest.mark.parametrize("a,b", [(0, 1), (1, 2), (2, 3), (3, 0)])
def test_reverse_transition_offsets_are_negated(table, a, b):
    """The reverse of a transition must undo it, candidate for candidate.

    This is a property of the published tables and it catches transcription
    slips that a spot-check of one or two entries would sail straight past.
    """
    forward = table[(a, b)]
    backward = table[(b, a)]
    for (fx, fy), (bx, by) in zip(forward, backward):
        assert (fx, fy) == (-bx, -by)


def test_i_piece_has_its_own_table():
    # Sharing the JLSTZ table with I is the single most common SRS bug.
    assert KICKS_I != KICKS_JLSTZ
    assert kick_table(PieceType.I) is KICKS_I
    for piece in (PieceType.J, PieceType.L, PieceType.S, PieceType.T, PieceType.Z):
        assert kick_table(piece) is KICKS_JLSTZ


def test_o_piece_has_no_kick_table():
    assert kick_table(PieceType.O) is None


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_every_piece_has_four_states_of_four_cells(piece_type):
    states = SHAPES[piece_type]
    assert len(states) == 4
    for cells in states:
        assert len(cells) == 4
        assert len(set(cells)) == 4, "a piece cannot occupy a cell twice"


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_spawn_orientation_fits_on_an_empty_board(piece_type):
    board = Board()
    piece = Piece.spawn(piece_type)
    assert not board.collides(piece.cells())


# -- rotation behaviour ---------------------------------------------------


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_open_space_rotation_needs_no_kick(piece_type):
    board = Board()
    piece = Piece(piece_type, x=3, y=10, rotation=0)
    result = try_rotate(piece, board, +1)
    assert result.success
    assert result.kick_index == 0
    assert (piece.x, piece.y) == (3, 10)
    assert piece.rotation == 1


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_four_rotations_return_to_start(piece_type):
    board = Board()
    piece = Piece(piece_type, x=3, y=10, rotation=0)
    start = piece.cells()
    for _ in range(4):
        assert try_rotate(piece, board, +1).success
    assert piece.rotation == 0
    assert piece.cells() == start


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_rotation_is_reversible_in_open_space(piece_type):
    board = Board()
    piece = Piece(piece_type, x=3, y=10, rotation=0)
    start = piece.cells()
    assert try_rotate(piece, board, +1).success
    assert try_rotate(piece, board, -1).success
    assert piece.cells() == start


def test_o_piece_never_moves_when_rotated():
    board = Board()
    piece = Piece(PieceType.O, x=3, y=10, rotation=0)
    start = piece.cells()
    for _ in range(4):
        assert try_rotate(piece, board, +1).success
        assert piece.cells() == start


def test_rotation_rejected_when_every_candidate_collides():
    # A T sealed into a pocket that exactly fits its spawn orientation, with
    # solid rock on every side. All five kick candidates land in stone, so the
    # rotation must be refused and the piece left exactly where it was.
    board = make_board(
        "##########",
        "####.#####",
        "###...####",
        "##########",
    )
    piece = Piece(PieceType.T, x=3, y=21, rotation=0)
    assert not board.collides(piece.cells())

    before = (piece.x, piece.y, piece.rotation)
    result = try_rotate(piece, board, +1)
    assert not result.success
    assert (piece.x, piece.y, piece.rotation) == before


def test_i_piece_floor_kick():
    # The classic I floor kick: a flat I resting on the floor cannot stand up in
    # place, because three of its cells would land below the board. Only the
    # final candidate (+1, -2) fits, lifting it clear of the floor.
    board = Board()
    piece = Piece(PieceType.I, x=3, y=22, rotation=0)
    assert piece.cells() == ((3, 23), (4, 23), (5, 23), (6, 23))

    result = try_rotate(piece, board, +1)
    assert result.success
    assert result.kick_index == LAST_KICK_INDEX
    assert piece.rotation == 1
    assert piece.cells() == ((6, 20), (6, 21), (6, 22), (6, 23))
    assert not board.collides(piece.cells())


def test_t_piece_left_wall_kick():
    # A T hugging the left wall, pointing right. Rotating it to point down would
    # push a cell through the wall, so SRS kicks it one column inward.
    board = Board()
    piece = Piece(PieceType.T, x=-1, y=10, rotation=1)
    assert piece.cells() == ((0, 10), (0, 11), (1, 11), (0, 12))

    result = try_rotate(piece, board, +1)
    assert result.success
    assert result.kick_index == 1, "should take the (+1, 0) candidate"
    assert piece.x == 0
    assert piece.rotation == 2
    assert not board.collides(piece.cells())
