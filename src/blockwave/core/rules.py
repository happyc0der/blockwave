"""Difficulty curve, T-spin recognition and scoring.

Pure functions over plain data — no engine state is mutated here, so every rule
can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .board import Board
from .constants import (
    CLEAR_DELAY_END_MS,
    CLEAR_DELAY_FLOOR_MS,
    CLEAR_DELAY_START_MS,
    GRAVITY_BASE,
    GRAVITY_DECAY,
    INSTANT_GRAVITY_THRESHOLD,
    LAST_KICK_INDEX,
    LINES_PER_LEVEL,
    LOCK_DELAY_END_MS,
    LOCK_DELAY_FLOOR_MS,
    LOCK_DELAY_START_MS,
    MAX_GRAVITY_LEVEL,
    PieceType,
)
from .piece import Piece


class TSpin(IntEnum):
    NONE = 0
    MINI = 1
    FULL = 2


# --------------------------------------------------------------------------
# Difficulty curve — requirement 3
# --------------------------------------------------------------------------


def level_for_lines(total_lines: int, start_level: int = 1) -> int:
    """The level after clearing ``total_lines`` lines from ``start_level``."""
    return start_level + total_lines // LINES_PER_LEVEL


def gravity_seconds_per_row(level: int) -> float:
    """Seconds a piece takes to fall one row at ``level``.

    Follows the guideline curve, which is 1.0 s at level 1 and decays sharply.
    Returns ``0.0`` once the curve passes the instant-gravity threshold — from
    there the piece reaches the stack the frame it spawns (true 20G) and the
    only remaining difficulty lever is the lock delay.
    """
    level = max(1, min(level, MAX_GRAVITY_LEVEL))
    seconds = (GRAVITY_BASE - (level - 1) * GRAVITY_DECAY) ** (level - 1)
    if seconds <= INSTANT_GRAVITY_THRESHOLD:
        return 0.0
    return seconds


def clear_delay_ms(level: int) -> float:
    """How long the board pauses after a line clear, in milliseconds.

    Shrinks on the same schedule as the lock delay, so the rhythm of the game
    tightens as a whole rather than one timing lagging behind the other.
    """
    span = MAX_GRAVITY_LEVEL - 1
    per_level = (CLEAR_DELAY_START_MS - CLEAR_DELAY_END_MS) / span
    value = CLEAR_DELAY_START_MS - (max(1, level) - 1) * per_level
    return max(CLEAR_DELAY_FLOOR_MS, value)


def lock_delay_ms(level: int) -> float:
    """How long a grounded piece may rest before locking, in milliseconds.

    Shrinks linearly from level 1 to :data:`MAX_GRAVITY_LEVEL`, then keeps
    shrinking at the same rate down to a hard floor. This is what continues to
    ramp difficulty after gravity has bottomed out.
    """
    span = MAX_GRAVITY_LEVEL - 1
    per_level = (LOCK_DELAY_START_MS - LOCK_DELAY_END_MS) / span
    value = LOCK_DELAY_START_MS - (max(1, level) - 1) * per_level
    return max(LOCK_DELAY_FLOOR_MS, value)


# --------------------------------------------------------------------------
# T-spin recognition
# --------------------------------------------------------------------------

#: The four corners of a T piece's 3x3 bounding box.
_T_CORNERS = ((0, 0), (2, 0), (0, 2), (2, 2))

#: The two corners on the side the T points, per rotation state. A T-spin counts
#: as full rather than mini when both of these are occupied.
_T_FRONT_CORNERS: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    0: ((0, 0), (2, 0)),  # pointing up
    1: ((2, 0), (2, 2)),  # pointing right
    2: ((0, 2), (2, 2)),  # pointing down
    3: ((0, 0), (0, 2)),  # pointing left
}


def detect_tspin(
    board: Board,
    piece: Piece,
    last_move_was_rotation: bool,
    kick_index: int,
) -> TSpin:
    """Classify a lock as a T-spin using the standard three-corner rule.

    A T-spin requires that the piece is a T, that the move immediately before
    locking was a rotation, and that at least three of the four corners of its
    bounding box are occupied (walls and floor count as occupied).

    It is a *full* T-spin when both front corners are filled, or when the
    rotation only fitted via the final kick candidate — that last case is the
    one that makes T-spin triples score correctly.
    """
    if piece.type is not PieceType.T or not last_move_was_rotation:
        return TSpin.NONE

    filled = sum(
        board.is_occupied(piece.x + bx, piece.y + by) for bx, by in _T_CORNERS
    )
    if filled < 3:
        return TSpin.NONE

    if kick_index == LAST_KICK_INDEX:
        return TSpin.FULL

    front = _T_FRONT_CORNERS[piece.rotation]
    if all(board.is_occupied(piece.x + bx, piece.y + by) for bx, by in front):
        return TSpin.FULL
    return TSpin.MINI


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
_TSPIN_FULL_SCORES = {0: 400, 1: 800, 2: 1200, 3: 1600}
_TSPIN_MINI_SCORES = {0: 100, 1: 200, 2: 400}
_PERFECT_CLEAR_SCORES = {1: 800, 2: 1200, 3: 1800, 4: 2000}

#: Score per cell for the two drop actions.
SOFT_DROP_POINTS = 1
HARD_DROP_POINTS = 2

#: Multiplier applied to a difficult clear that follows another difficult clear.
B2B_MULTIPLIER = 1.5

COMBO_POINTS = 50


@dataclass(slots=True)
class ClearOutcome:
    """Everything scoring produced for one lock."""

    lines: int
    tspin: TSpin
    score: int
    combo: int
    b2b: bool
    perfect_clear: bool


def is_difficult(lines: int, tspin: TSpin) -> bool:
    """Whether a clear extends a back-to-back chain.

    Quads and any T-spin that clears lines are difficult; everything else
    breaks the chain.
    """
    if lines == 0:
        return False
    return lines == 4 or tspin is not TSpin.NONE


def score_lock(
    *,
    lines: int,
    tspin: TSpin,
    level: int,
    combo: int,
    b2b_active: bool,
    perfect_clear: bool,
) -> ClearOutcome:
    """Score one locked piece and advance the combo and back-to-back state.

    ``combo`` is the count *before* this lock (-1 meaning no chain in progress).
    """
    if tspin is TSpin.FULL:
        base = _TSPIN_FULL_SCORES[lines]
    elif tspin is TSpin.MINI:
        base = _TSPIN_MINI_SCORES.get(lines, _LINE_SCORES[lines])
    else:
        base = _LINE_SCORES[lines]

    difficult = is_difficult(lines, tspin)
    chained = difficult and b2b_active
    if chained:
        base = int(base * B2B_MULTIPLIER)

    if perfect_clear and lines:
        base += _PERFECT_CLEAR_SCORES[lines]

    new_combo = combo + 1 if lines else -1
    if new_combo > 0:
        base += COMBO_POINTS * new_combo

    # A no-clear lock leaves the back-to-back chain alone; only an easy clear
    # breaks it.
    if lines == 0:
        new_b2b = b2b_active
    else:
        new_b2b = difficult

    return ClearOutcome(
        lines=lines,
        tspin=tspin,
        score=base * max(1, level),
        combo=new_combo,
        b2b=new_b2b,
        perfect_clear=perfect_clear and bool(lines),
    )
