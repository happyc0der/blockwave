"""The playfield: a bitboard for collision, plus a colour grid for rendering.

Two parallel representations, deliberately:

``rows``
    One integer per row, bit ``x`` set meaning column ``x`` is occupied.
    Collision is a bitwise AND and a full row is an equality test, which makes
    the hot path both fast and hard to get wrong.

``colors``
    A ``(TOTAL_HEIGHT, BOARD_WIDTH)`` uint8 array holding the ``PieceType`` that
    filled each cell (0 = empty). Only touched when a piece locks or lines
    clear, never during collision checks, so it costs nothing per step.

The two must agree at all times: ``colors[y][x] != 0`` exactly when bit ``x`` of
``rows[y]`` is set. :meth:`Board.check_invariants` asserts this and is used by
the test suite after every mutation.
"""

from __future__ import annotations

import numpy as np

from .constants import BOARD_WIDTH, FULL_ROW, TOTAL_HEIGHT, VISIBLE_TOP, PieceType


class Board:
    __slots__ = ("rows", "colors")

    def __init__(self) -> None:
        self.rows: list[int] = [0] * TOTAL_HEIGHT
        self.colors: np.ndarray = np.zeros((TOTAL_HEIGHT, BOARD_WIDTH), dtype=np.uint8)

    # -- queries ----------------------------------------------------------

    def collides(self, cells: tuple[tuple[int, int], ...]) -> bool:
        """Whether any of ``cells`` is out of bounds or overlaps a filled cell."""
        rows = self.rows
        for x, y in cells:
            if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= TOTAL_HEIGHT:
                return True
            if rows[y] & (1 << x):
                return True
        return False

    def is_occupied(self, x: int, y: int) -> bool:
        """Whether cell ``(x, y)`` is filled. Out-of-bounds counts as filled.

        Treating walls and floor as solid is what makes the T-spin corner test
        work at the edges of the board without a special case.
        """
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= TOTAL_HEIGHT:
            return True
        return bool(self.rows[y] & (1 << x))

    def full_rows(self) -> list[int]:
        """Row indices that are completely filled, top to bottom."""
        return [y for y, row in enumerate(self.rows) if row == FULL_ROW]

    # -- mutation ---------------------------------------------------------

    def lock(self, cells: tuple[tuple[int, int], ...], piece: PieceType) -> None:
        """Write a locked piece into both representations."""
        colors = self.colors
        rows = self.rows
        value = np.uint8(int(piece))
        for x, y in cells:
            rows[y] |= 1 << x
            colors[y, x] = value

    def clear_rows(self, ys: list[int]) -> None:
        """Remove the given rows and drop everything above them down.

        Rebuilds both representations from scratch rather than shifting in
        place; at 24 rows that is cheap, and it cannot leave the two out of
        sync the way an in-place shift can.
        """
        if not ys:
            return
        doomed = set(ys)
        kept = [row for y, row in enumerate(self.rows) if y not in doomed]
        blanks = TOTAL_HEIGHT - len(kept)
        self.rows = [0] * blanks + kept

        kept_colors = np.delete(self.colors, list(doomed), axis=0)
        self.colors = np.vstack(
            [np.zeros((blanks, BOARD_WIDTH), dtype=np.uint8), kept_colors]
        )

    # -- stack metrics ----------------------------------------------------
    # Used for RL telemetry and optional reward shaping. All measured over the
    # visible playfield only, so buffer rows never distort them.

    def column_heights(self) -> list[int]:
        """Height of each column, measured up from the floor."""
        heights = [0] * BOARD_WIDTH
        for x in range(BOARD_WIDTH):
            bit = 1 << x
            for y in range(VISIBLE_TOP, TOTAL_HEIGHT):
                if self.rows[y] & bit:
                    heights[x] = TOTAL_HEIGHT - y
                    break
        return heights

    def holes(self) -> int:
        """Empty cells that have at least one filled cell somewhere above them."""
        total = 0
        for x in range(BOARD_WIDTH):
            bit = 1 << x
            covered = False
            for y in range(VISIBLE_TOP, TOTAL_HEIGHT):
                if self.rows[y] & bit:
                    covered = True
                elif covered:
                    total += 1
        return total

    def bumpiness(self) -> int:
        """Total absolute height difference between neighbouring columns."""
        heights = self.column_heights()
        return sum(abs(a - b) for a, b in zip(heights, heights[1:]))

    def aggregate_height(self) -> int:
        return sum(self.column_heights())

    def max_height(self) -> int:
        return max(self.column_heights())

    # -- debugging --------------------------------------------------------

    def check_invariants(self) -> None:
        """Assert the bitboard and colour grid agree. Used by the test suite."""
        for y in range(TOTAL_HEIGHT):
            for x in range(BOARD_WIDTH):
                occupied = bool(self.rows[y] & (1 << x))
                colored = bool(self.colors[y, x])
                if occupied != colored:
                    raise AssertionError(
                        f"board desync at ({x}, {y}): "
                        f"bitboard={occupied} colors={colored}"
                    )
            if self.rows[y] == FULL_ROW:
                raise AssertionError(f"row {y} is full but was never cleared")

    def to_ascii(self, active: tuple[tuple[int, int], ...] = ()) -> str:
        """Render the visible playfield as text, for debugging and doctests."""
        marks = set(active)
        lines = []
        for y in range(VISIBLE_TOP, TOTAL_HEIGHT):
            cells = []
            for x in range(BOARD_WIDTH):
                if (x, y) in marks:
                    cells.append("@")
                elif self.rows[y] & (1 << x):
                    cells.append("#")
                else:
                    cells.append(".")
            lines.append("|" + "".join(cells) + "|")
        lines.append("+" + "-" * BOARD_WIDTH + "+")
        return "\n".join(lines)
