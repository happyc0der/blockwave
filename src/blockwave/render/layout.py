"""Scene layout, measured in cells rather than pixels.

The whole scene is described once in cell units and scaled by a single
``cell_px``, so the same layout code produces a chunky 30 px arcade window and a
tiny 12 px one without a second set of numbers anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.constants import BOARD_WIDTH, TOTAL_HEIGHT, VISIBLE_TOP

#: How many of the hidden buffer rows to draw, dimmed, above the playfield.
#:
#: Pieces spawn inside the buffer, which is guideline-correct but means a newly
#: spawned piece is invisible for the first second of its life. Showing the
#: bottom two buffer rows makes the active piece visible from the moment it
#: appears.
SPAWN_ROWS_SHOWN = 2

#: First board row that appears on screen.
RENDER_TOP = VISIBLE_TOP - SPAWN_ROWS_SHOWN

#: Rows drawn: the visible playfield plus the spawn zone.
RENDER_ROWS = TOTAL_HEIGHT - RENDER_TOP

# Scene metrics, in cells.
#
# Panels are five cells wide rather than four so that a five-character label
# ("SCORE", "LEVEL", "LINES") fits inside its panel at the HUD's text scale.
_MARGIN = 1
_PANEL_W = 5
_HEADER_H = 2
_FOOTER_H = 1

SCENE_COLS = _MARGIN + _PANEL_W + _MARGIN + BOARD_WIDTH + _MARGIN + _PANEL_W + _MARGIN
SCENE_ROWS = _HEADER_H + RENDER_ROWS + _FOOTER_H


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def inset(self, amount: int) -> Rect:
        return Rect(self.x + amount, self.y + amount, self.w - 2 * amount, self.h - 2 * amount)


@dataclass(frozen=True, slots=True)
class Layout:
    """Pixel geometry for one ``cell_px``."""

    cell_px: int

    @property
    def width(self) -> int:
        return SCENE_COLS * self.cell_px

    @property
    def height(self) -> int:
        return SCENE_ROWS * self.cell_px

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def _rect(self, col: int, row: int, cols: int, rows: int) -> Rect:
        c = self.cell_px
        return Rect(col * c, row * c, cols * c, rows * c)

    @property
    def board(self) -> Rect:
        """The playfield, including the dimmed spawn zone at its top."""
        return self._rect(
            _MARGIN + _PANEL_W + _MARGIN, _HEADER_H, BOARD_WIDTH, RENDER_ROWS
        )

    @property
    def spawn_zone(self) -> Rect:
        """The dimmed strip above the playfield proper."""
        return self._rect(
            _MARGIN + _PANEL_W + _MARGIN, _HEADER_H, BOARD_WIDTH, SPAWN_ROWS_SHOWN
        )

    @property
    def playfield(self) -> Rect:
        """The 20 rows that actually count, excluding the spawn zone."""
        return self._rect(
            _MARGIN + _PANEL_W + _MARGIN,
            _HEADER_H + SPAWN_ROWS_SHOWN,
            BOARD_WIDTH,
            TOTAL_HEIGHT - VISIBLE_TOP,
        )

    @property
    def hold_panel(self) -> Rect:
        return self._rect(_MARGIN, _HEADER_H, _PANEL_W, 4)

    @property
    def next_panel(self) -> Rect:
        # Tall enough for a label plus four preview slots of three cells each.
        return self._rect(
            _MARGIN + _PANEL_W + _MARGIN + BOARD_WIDTH + _MARGIN, _HEADER_H, _PANEL_W, 16
        )

    @property
    def stats_panel(self) -> Rect:
        return self._rect(_MARGIN, _HEADER_H + 5, _PANEL_W, 9)

    @property
    def header(self) -> Rect:
        return self._rect(0, 0, SCENE_COLS, _HEADER_H)

    def cell(self, col: int, row: int) -> Rect:
        """Pixel rect of one board cell, in *board* row coordinates."""
        board = self.board
        c = self.cell_px
        return Rect(board.x + col * c, board.y + (row - RENDER_TOP) * c, c, c)


#: Chunky cells for the arcade window.
HUMAN_CELL_PX = 30
