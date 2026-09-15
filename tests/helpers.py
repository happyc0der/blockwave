"""Shared test helpers."""

from __future__ import annotations

import numpy as np

from blockwave.core.board import Board
from blockwave.core.constants import BOARD_WIDTH, TOTAL_HEIGHT, PieceType


def make_board(*rows: str) -> Board:
    """Build a board from ASCII art, bottom-aligned.

    Each string is one row of exactly ``BOARD_WIDTH`` characters, ``#`` filled
    and ``.`` empty. The last string given is the floor row, so a test only has
    to describe the part of the stack it cares about.
    """
    board = Board()
    for offset, line in enumerate(reversed(rows)):
        assert len(line) == BOARD_WIDTH, f"row {line!r} must be {BOARD_WIDTH} wide"
        y = TOTAL_HEIGHT - 1 - offset
        for x, char in enumerate(line):
            if char == "#":
                board.rows[y] |= 1 << x
                board.colors[y, x] = np.uint8(int(PieceType.J))
            else:
                assert char == ".", f"unexpected character {char!r}"
    return board
