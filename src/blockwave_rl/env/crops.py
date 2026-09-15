"""What the agent is allowed to see.

The crop is not a neutral choice. The game's HUD shows SCORE, LEVEL and LINES,
and the whole claim of this package is that the agent never sees them: not as
reward, and not as pixels a reward could be read back out of. (Under the
surprise-minimizing objective first tried, a changing score display was also
surprise to be *avoided*, which would have punished progress outright.) Every
variant here therefore excludes it, and `tests/rl/test_firewall.py` asserts that
two states differing only in score produce byte-identical observations.

Two variants, because frame-stacking cannot recover what is not in the frame:

BOARD_ONLY
    The playfield alone. Next piece and hold are invisible, so the agent cannot
    plan ahead. The hardest task and the cleanest claim.

BOARD_PLUS_PREVIEW
    Playfield, next queue and hold slot, with the stats panel blanked. The stats
    panel sits directly below HOLD in the same column, so a bounding box around
    the three regions would include it — it is masked out explicitly rather than
    relied on to fall outside the crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from blockwave.render.layout import Layout, Rect


class Variant(str, Enum):
    BOARD_ONLY = "board_only"
    BOARD_PLUS_PREVIEW = "board_plus_preview"


@dataclass(frozen=True, slots=True)
class Crop:
    """A rectangle to keep, and rectangles within it to blank."""

    keep: Rect
    masks: tuple[Rect, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.keep.h, self.keep.w)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        keep = self.keep
        out = frame[keep.y : keep.bottom, keep.x : keep.right].copy()
        for mask in self.masks:
            # Translate the mask into crop coordinates and clip it to the crop.
            y0 = max(0, mask.y - keep.y)
            x0 = max(0, mask.x - keep.x)
            y1 = min(out.shape[0], mask.bottom - keep.y)
            x1 = min(out.shape[1], mask.right - keep.x)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = 0
        return out


def crop_for(variant: Variant | str, layout: Layout) -> Crop:
    variant = Variant(variant)

    if variant is Variant.BOARD_ONLY:
        return Crop(keep=layout.board)

    hold, board, nxt, stats = layout.hold_panel, layout.board, layout.next_panel, layout.stats_panel
    x0 = min(hold.x, board.x, nxt.x)
    y0 = min(hold.y, board.y, nxt.y)
    x1 = max(hold.right, board.right, nxt.right)
    y1 = max(hold.bottom, board.bottom, nxt.bottom)
    return Crop(keep=Rect(x0, y0, x1 - x0, y1 - y0), masks=(stats,))
