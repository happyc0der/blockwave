"""A procedural 5x7 bitmap font, drawn straight into a numpy frame.

Bundling a TTF means a licence to track and a rasteriser to call; generating the
glyphs here means neither, plus pixel-crisp output at any integer scale and the
blocky arcade look the rest of the design is going for.

Each glyph is seven rows of five bits, most significant bit on the left.
"""

from __future__ import annotations

import numpy as np

from .palette import RGB

GLYPH_W = 5
GLYPH_H = 7

_GLYPHS: dict[str, tuple[int, ...]] = {
    " ": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "-": (0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000),
    ".": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100),
    # Thousands separators run through here on every score, so a missing comma
    # showed up as "128?450" on the game-over panel.
    ",": (0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b01000),
    "'": (0b01100, 0b01100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000),
    "%": (0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011),
    ":": (0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000),
    "!": (0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100),
    "/": (0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000),
    "x": (0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001),
    "+": (0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000),
}

_FALLBACK = _GLYPHS["?"] = (
    0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100,
)


def _glyph_mask(char: str) -> np.ndarray:
    """A ``(7, 5)`` boolean mask for one character."""
    rows = _GLYPHS.get(char.upper() if char.upper() in _GLYPHS else char, _FALLBACK)
    mask = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
    for y, bits in enumerate(rows):
        for x in range(GLYPH_W):
            if bits & (1 << (GLYPH_W - 1 - x)):
                mask[y, x] = True
    return mask


#: Masks are tiny and reused every frame, so build them once at import.
_MASK_CACHE: dict[str, np.ndarray] = {}


def glyph_mask(char: str) -> np.ndarray:
    mask = _MASK_CACHE.get(char)
    if mask is None:
        mask = _glyph_mask(char)
        _MASK_CACHE[char] = mask
    return mask


def text_size(text: str, scale: int = 1, tracking: int = 1) -> tuple[int, int]:
    """Pixel width and height of ``text`` at ``scale``."""
    if not text:
        return (0, 0)
    advance = (GLYPH_W + tracking) * scale
    return (advance * len(text) - tracking * scale, GLYPH_H * scale)


def fit_scale(text: str, max_width: int, preferred: int, tracking: int = 1) -> int:
    """The largest scale at or below ``preferred`` whose text fits ``max_width``.

    Every line of a panel goes through this, so a long string — a nine-digit
    score, say — shrinks to fit rather than running off the edge of its plate.
    """
    for scale in range(max(1, preferred), 0, -1):
        if text_size(text, scale, tracking)[0] <= max_width:
            return scale
    return 1


def draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: RGB,
    scale: int = 1,
    tracking: int = 1,
) -> None:
    """Blit ``text`` into ``frame`` at ``(x, y)``, clipped to the frame.

    Writes directly into the array — no pygame surface, so this works
    identically in a window and in a headless render.
    """
    height, width = frame.shape[:2]
    advance = (GLYPH_W + tracking) * scale
    paint = np.array(color, dtype=np.uint8)

    for index, char in enumerate(text):
        gx = x + index * advance
        if gx >= width or gx + GLYPH_W * scale <= 0:
            continue

        mask = glyph_mask(char)
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)

        gh, gw = mask.shape
        # Clip against every edge so callers never have to think about it.
        sy0, sx0 = max(0, -y), max(0, -gx)
        dy0, dx0 = max(0, y), max(0, gx)
        dy1, dx1 = min(height, y + gh), min(width, gx + gw)
        if dy1 <= dy0 or dx1 <= dx0:
            continue

        clipped = mask[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
        frame[dy0:dy1, dx0:dx1][clipped] = paint


def draw_text_centered(
    frame: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    color: RGB,
    scale: int = 1,
    tracking: int = 1,
) -> None:
    width, _ = text_size(text, scale, tracking)
    draw_text(frame, text, center_x - width // 2, y, color, scale, tracking)
