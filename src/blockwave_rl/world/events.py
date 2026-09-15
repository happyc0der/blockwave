"""Reading the two events the reward needs out of the pixels themselves.

The board-state track took both from `info`: a placement from the engine's
piece counter, a top-out from its flag. A pixel agent has neither. It does have
a screen, and both events are plainly visible on it:

**A new piece is in play** — the NEXT panel shifts up by one slot. A lock is
the only thing that does that, with one exception: the *first* HOLD of a game
draws a piece from the queue to replace the one it stores. Every later hold
swaps with the piece already held and draws nothing, so it shifts nothing. The
agent tracks that one exception itself — it knows when it pressed HOLD and it
can see the game restart — and needs no help from the screen for it.

**The game ended** — the playfield empties. A soft reset clears a board that was
by definition nearly full, so the filled area collapses in a single frame.

Both read fixed rectangles of the frame, the same `Layout` geometry that defines
the crop the agent sees. That is screen layout, not game rules: nothing here
knows what a piece is, how it rotates or what a line is worth.

`tests/rl/test_world_events.py` checks both against the engine's own accounting
over long rollouts of several policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blockwave.core.constants import Action
from blockwave.render.layout import Layout, Rect

from ..env.crops import Crop

#: Grayscale value above which a playfield pixel is a block rather than grid.
#: Measured: grid lines sit at 12-60, block colours at 100-203.
BRIGHT = 80

#: The same question for the HOLD and NEXT panels, which draw their previews as
#: miniature blocks — a whole piece is only ~12 pixels — and dim the hold to 45%
#: while it is spent. A cut this low sees a dimmed piece, so dimming changes no
#: pixel's membership while adding, removing or swapping a piece changes ~12.
PANEL_BRIGHT = 25


def _relative(rect: Rect, crop: Crop) -> tuple[slice, slice]:
    """A layout rectangle, in the cropped frame's coordinates."""
    keep = crop.keep
    y0 = max(0, rect.y - keep.y)
    x0 = max(0, rect.x - keep.x)
    y1 = min(keep.h, rect.bottom - keep.y)
    x1 = min(keep.w, rect.right - keep.x)
    return slice(y0, y1), slice(x0, x1)


@dataclass
class FrameEvents:
    """Detects placements and game-overs in a stream of cropped frames."""

    layout: Layout
    crop: Crop
    #: A placement is called when this many pixels of the NEXT panel change.
    #: One slot's worth of tetromino is far more than this; rendering noise is
    #: far less, since the flat profile has neither shake nor bloom.
    panel_tolerance: int = 6
    #: A game-over: the playfield was this full and is now this empty, in
    #: pixels above the empty-board baseline. Swept over five policies and
    #: 40,000 steps: this pair catches every top-out with no false alarm, and
    #: the gap between the two is what keeps a line clear from reading as one.
    min_filled: int = 160
    empty_slack: int = 32

    def __post_init__(self) -> None:
        self._next = _relative(self.layout.next_panel, self.crop)
        self._board = _relative(self.layout.board, self.crop)
        if self._next[0].stop <= self._next[0].start or self._next[1].stop <= self._next[1].start:
            raise ValueError("the next panel is not inside this crop: no placement signal")
        self._baseline = 0.0
        self._hold_filled = False

    def calibrate(self, empty_frame: np.ndarray) -> None:
        """Take the playfield's bright area on an empty board as zero.

        The playfield is never literally blank: it has a border, and a piece
        with its ghost is already on screen a frame after any reset.
        """
        self._baseline = float((empty_frame[self._board] > BRIGHT).sum())

    def _filled(self, frame: np.ndarray) -> float:
        return max(0.0, float((frame[self._board] > BRIGHT).sum()) - self._baseline)

    def _shape_changed(self, region, previous: np.ndarray, current: np.ndarray) -> int:
        """Pixels that gained or lost a preview block — blind to brightness alone.

        The hold preview dims to 45% while the hold is spent and brightens when
        a lock frees it. Comparing raw values would call that a change at every
        placement and reject every one of them; comparing membership at a cut
        below the dimmed level does not.
        """
        a = previous[region] > PANEL_BRIGHT
        b = current[region] > PANEL_BRIGHT
        return int((a != b).sum())

    def restart(self) -> None:
        """A new game began: the hold slot is empty again."""
        self._hold_filled = False

    def placed(self, previous: np.ndarray, current: np.ndarray, action: int | None = None) -> bool:
        """A new piece entered play *because the last one landed*.

        Pass the key the agent pressed. Counting the first hold of a game as a
        placement would hand a policy a free reward tick for pressing one key,
        and under per-placement discounting a tick pushes its future deaths
        further away.

        Pressing HOLD is not disqualifying by itself: a press made while the
        hold is already spent does nothing, and gravity can land a piece on that
        very step. Only the press that *fills an empty hold* draws from the
        queue, and that is the first one of each game.
        """
        if self._shape_changed(self._next, previous, current) <= self.panel_tolerance:
            return False
        if action is not None and int(action) == int(Action.HOLD) and not self._hold_filled:
            self._hold_filled = True
            return False
        return True

    def ended(self, previous: np.ndarray, current: np.ndarray) -> bool:
        """The playfield went from occupied to empty: a soft reset.

        A line clear blanks its rows before collapsing them, which is a large
        *relative* drop and fooled an earlier version of this. Only a top-out
        empties the board outright — or a perfect clear, which no policy
        measured here has ever produced.
        """
        return self._filled(previous) >= self.min_filled and self._filled(current) <= self.empty_slack

    def observe(self, previous: np.ndarray, current: np.ndarray, action: int | None = None) -> tuple[bool, bool]:
        """One step of the stream: (a piece was placed, the game ended).

        Keeps the hold-slot state in step with the game, so callers do not have
        to remember to call `restart` after a top-out.
        """
        ended = self.ended(previous, current)
        placed = self.placed(previous, current, action)
        if ended:
            self.restart()
        return placed, ended
