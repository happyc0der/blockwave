"""The Miami Vice palette — the single source of colour for the whole project.

Dark UI: everything sits on a near-black indigo ground and neon is the only
bright thing on screen. Nothing else in the codebase should contain a colour
literal.

The seven piece colours are drawn from this palette rather than the
conventional ones, so the board reads as a single piece of art. They are still
spread around the hue circle so that no two are easy to confuse at a glance, or
at a small cell size.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import PieceType

RGB = tuple[int, int, int]


def _hex(value: str) -> RGB:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


# -- ground and chrome ----------------------------------------------------

VOID: RGB = _hex("0D0221")          # the darkest ground, behind everything
DEEP: RGB = _hex("150829")          # gradient midpoint
PANEL: RGB = _hex("1A0B2E")         # hold / next panel fill
WELL: RGB = _hex("120626")          # the playfield floor, a shade off the void
GRID: RGB = _hex("2A1247")          # playfield grid lines
SPAWN_ZONE: RGB = _hex("1E0A33")    # the dimmed rows above the playfield

# -- neon accents ---------------------------------------------------------

MAGENTA: RGB = _hex("FF1E8E")
CYAN: RGB = _hex("00F0FF")
PURPLE: RGB = _hex("B026FF")
SUNSET_TOP: RGB = _hex("FFD319")
SUNSET_BOTTOM: RGB = _hex("FF1E8E")

TEXT: RGB = _hex("00F0FF")
TEXT_DIM: RGB = _hex("6E5A9E")
TEXT_HOT: RGB = _hex("FF1E8E")

# -- tetromino colours ----------------------------------------------------
# Spread around the hue circle for separability, and all recognisably Miami.

PIECE_COLORS: dict[PieceType, RGB] = {
    PieceType.I: _hex("00E5FF"),  # cyan
    PieceType.J: _hex("3D5AFE"),  # indigo
    PieceType.L: _hex("FF6B1F"),  # orange
    PieceType.O: _hex("FFD319"),  # sunset yellow
    PieceType.S: _hex("2BFF88"),  # mint
    PieceType.T: _hex("C13DFF"),  # purple
    PieceType.Z: _hex("FF2E63"),  # hot red
}


def piece_lut() -> np.ndarray:
    """An ``(8, 3)`` uint8 lookup table indexed by :class:`PieceType`.

    Index 0 is the empty cell, so a board's colour grid can be turned into
    pixels with a single fancy-index rather than a per-cell loop.
    """
    lut = np.zeros((len(PieceType) + 1, 3), dtype=np.uint8)
    for piece, color in PIECE_COLORS.items():
        lut[int(piece)] = color
    return lut
