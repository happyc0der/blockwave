"""Static game data: board geometry, piece shapes, SRS kick tables, difficulty curve.

Nothing here depends on pygame or numpy. Everything is a plain constant so it can
be diffed against a published SRS reference by eye.

Coordinate convention used throughout the whole project
-------------------------------------------------------
``x`` is the column, 0 at the left edge, increasing right.
``y`` is the row, 0 at the *top*, increasing **downward**.

Downward-positive ``y`` matches both the bitboard row list and numpy image
indexing, so no axis flip is ever needed between logic and rendering. The one
place it costs us is the SRS kick tables, which are published with ``y``
positive *upward*; see ``KICKS_JLSTZ`` for how that is handled.
"""

from __future__ import annotations

from enum import IntEnum

# --------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------

BOARD_WIDTH = 10
VISIBLE_HEIGHT = 20

#: Hidden rows above the visible playfield. Pieces spawn here, so a piece that
#: locks entirely inside this region is a "lock out" (see rules.py). Four rows
#: is enough for every spawn orientation including the vertical I piece.
BUFFER_HEIGHT = 4

TOTAL_HEIGHT = VISIBLE_HEIGHT + BUFFER_HEIGHT

#: Row index of the first *visible* row. Rows [0, VISIBLE_TOP) are the buffer.
VISIBLE_TOP = BUFFER_HEIGHT

#: A row with every column occupied — the line-clear test.
FULL_ROW = (1 << BOARD_WIDTH) - 1

#: Top-left corner of a newly spawned piece's bounding box.
#: x=3 puts the 3-wide pieces at columns 3-5 and the I piece at columns 3-6,
#: and y=2 puts every spawn orientation entirely inside the buffer.
SPAWN_X = 3
SPAWN_Y = 2


class PieceType(IntEnum):
    """Tetromino identity. Values 1-7 so that 0 can mean "empty cell"."""

    I = 1
    J = 2
    L = 3
    O = 4
    S = 5
    T = 6
    Z = 7


class Action(IntEnum):
    """The complete action set, shared by human input and the RL agent."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    SOFT_DROP = 3
    HARD_DROP = 4
    ROTATE_CW = 5
    ROTATE_CCW = 6
    HOLD = 7


NUM_ACTIONS = len(Action)


# --------------------------------------------------------------------------
# Piece shapes
# --------------------------------------------------------------------------
# Each piece is four rotation states; each state is the four occupied cells as
# (bx, by) offsets inside the piece's bounding box. States are ordered
#   0 = spawn, 1 = one step clockwise, 2 = 180 degrees, 3 = one step
#   counter-clockwise
# which is the ordering the SRS kick tables are indexed by.
#
# Listing the states explicitly rather than rotating a matrix at runtime is both
# faster and far harder to get subtly wrong.

#: Bounding box edge length per piece. Only the I piece needs a 4x4 box; O uses
#: a 3x3 box with its cells parked at bx 1-2 so that it spawns on columns 4-5.
BOX_SIZE: dict[PieceType, int] = {
    PieceType.I: 4,
    PieceType.J: 3,
    PieceType.L: 3,
    PieceType.O: 3,
    PieceType.S: 3,
    PieceType.T: 3,
    PieceType.Z: 3,
}

SHAPES: dict[PieceType, tuple[tuple[tuple[int, int], ...], ...]] = {
    # ....  ..X.  ....  .X..
    # XXXX  ..X.  ....  .X..
    # ....  ..X.  XXXX  .X..
    # ....  ..X.  ....  .X..
    PieceType.I: (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    # X..  .XX  ...  .X.
    # XXX  .X.  XXX  .X.
    # ...  .X.  ..X  XX.
    PieceType.J: (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    # ..X  .X.  ...  XX.
    # XXX  .X.  XXX  .X.
    # ...  .XX  X..  .X.
    PieceType.L: (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
    # .XX  -- all four states identical: in SRS the O piece never shifts when
    # .XX     rotated, and it has no kick table.
    # ...
    PieceType.O: (
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
    ),
    # .XX  .X.  ...  X..
    # XX.  .XX  .XX  XX.
    # ...  ..X  XX.  .X.
    PieceType.S: (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    # .X.  .X.  ...  .X.
    # XXX  .XX  XXX  XX.
    # ...  .X.  .X.  .X.
    PieceType.T: (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    # XX.  ..X  ...  .X.
    # .XX  .XX  XX.  XX.
    # ...  .X.  .XX  X..
    PieceType.Z: (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
}


# --------------------------------------------------------------------------
# SRS wall kick tables
# --------------------------------------------------------------------------
# Keyed by (from_state, to_state). Each value is the five candidate offsets
# tried in order; the first that does not collide wins, and if all five collide
# the rotation is rejected.
#
# These literals are written in the *published* SRS form, with y positive
# UPWARD, so they can be compared against any SRS reference without mental
# arithmetic. They are flipped to this project's downward-positive y exactly
# once, immediately below, and only the flipped tables are ever used.

_KICKS_JLSTZ_Y_UP: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (1, 2): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (2, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (0, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
}

_KICKS_I_Y_UP: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (1, 0): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    (2, 1): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
}


def _flip_y(
    table: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Convert a published y-up kick table to this project's y-down convention."""
    return {
        transition: tuple((dx, -dy) for dx, dy in offsets)
        for transition, offsets in table.items()
    }


#: Kick table for J, L, S, T and Z, in y-down coordinates.
KICKS_JLSTZ = _flip_y(_KICKS_JLSTZ_Y_UP)

#: Kick table for I, in y-down coordinates. The I piece genuinely needs its own
#: table — sharing the JLSTZ one is the single most common SRS bug.
KICKS_I = _flip_y(_KICKS_I_Y_UP)

#: Index of the final kick candidate. A T piece that rotates into place using
#: this offset always scores as a full T-spin, never a mini (see rules.py).
LAST_KICK_INDEX = 4


def kick_table(piece: PieceType) -> dict[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    """Return the kick table for ``piece``, or ``None`` for O (which never kicks)."""
    if piece is PieceType.O:
        return None
    if piece is PieceType.I:
        return KICKS_I
    return KICKS_JLSTZ


# --------------------------------------------------------------------------
# Difficulty curve
# --------------------------------------------------------------------------

#: Lines cleared per level. Requirement 3 — the game gets harder as you play.
LINES_PER_LEVEL = 10

#: Gravity follows the guideline formula
#:     seconds_per_row = (0.8 - (level - 1) * 0.007) ** (level - 1)
#: which gives 1.0 s/row at level 1 and about 0.009 s/row at level 15.
GRAVITY_BASE = 0.8
GRAVITY_DECAY = 0.007

#: At or below this many seconds per row, gravity is treated as instantaneous
#: (true 20G: a piece reaches the stack the moment it spawns). Reached at
#: roughly level 15, which is where the formula stops being meaningful.
INSTANT_GRAVITY_THRESHOLD = 1.0 / 600.0

#: Levels past this point no longer speed gravity up — it is already instant.
#: They keep getting harder by shrinking the lock delay instead.
MAX_GRAVITY_LEVEL = 20

#: Lock delay shrinks linearly from LOCK_DELAY_START_MS at level 1 to
#: LOCK_DELAY_END_MS at MAX_GRAVITY_LEVEL, then continues down to the floor.
#: Once gravity is instant this is the *only* thing still ramping difficulty,
#: and it is what makes the late levels genuinely hard rather than just fast.
LOCK_DELAY_START_MS = 500.0
LOCK_DELAY_END_MS = 200.0
LOCK_DELAY_FLOOR_MS = 150.0

#: How many times a move or rotation may reset the lock delay for one piece.
#: Without this cap a piece can be juggled above the stack forever.
MAX_LOCK_RESETS = 15

#: The pause after a line clear, before the stack collapses and the next piece
#: arrives. This is a real mechanic, not just an animation window — it is a beat
#: of rest the player gets for clearing, and it is where the clear animation and
#: its sound live. Shrinks with level so late play keeps its pressure.
CLEAR_DELAY_START_MS = 400.0
CLEAR_DELAY_END_MS = 180.0
CLEAR_DELAY_FLOOR_MS = 150.0
