"""Empowerment: reward the agent for how much control it still has over its future.

SMiRL rewards *predictability*, and on a full-width board a doomed repetitive
policy — stack at the spawn point, die on schedule — is very predictable, so
SMiRL rewards it. Empowerment asks a different question: how many distinct
futures can the agent's actions still reach? A board that is running out of
room offers fewer and fewer, and game over offers none.

Formally, empowerment is the channel capacity between action sequences and
resulting states, ``max_p I(A; S' | s)``. Within one piece's life Tetris is
deterministic — only the *next* piece is random — and for deterministic
dynamics the capacity reduces exactly to ``log |reachable outcomes|``. So no
variational estimator is needed:

    E(board) = mean over piece types  log(1 + #distinct placements reachable)

**This uses the real simulator as a perfect world model.** It is therefore a
test of the *principle*, in the same way the board-state track tests SMiRL with
true occupancy. A pixel agent would need a learned dynamics model to estimate
it. It never reads score, lines or level, and knows nothing about what makes a
placement good — only how many there are.
"""

from __future__ import annotations

import math

import numpy as np

from blockwave.core.constants import BOARD_WIDTH, SHAPES, SPAWN_Y, TOTAL_HEIGHT, VISIBLE_TOP, PieceType

#: Every rotation state's cells, precomputed as (bx, by) tuples.
_SHAPES = {ptype: tuple(SHAPES[ptype]) for ptype in PieceType}


def _collides(rows: list[int], cells) -> bool:
    for x, y in cells:
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= TOTAL_HEIGHT:
            return True
        if rows[y] & (1 << x):
            return True
    return False


def placement_count(rows: list[int], piece: PieceType) -> int:
    """Distinct lock positions reachable by rotate, shift, drop from spawn.

    Dedupes by final cell set, so the O piece's identical rotations and other
    equivalent routes count once. Placements that would lock entirely inside the
    hidden buffer are a top-out, not a future, and are excluded.
    """
    seen: set[frozenset] = set()
    for rotation in _SHAPES[piece]:
        for x in range(-3, BOARD_WIDTH):
            cells = [(x + bx, SPAWN_Y + by) for bx, by in rotation]
            if _collides(rows, cells):
                continue
            drop = 0
            while not _collides(rows, [(cx, cy + drop + 1) for cx, cy in cells]):
                drop += 1
            final = frozenset((cx, cy + drop) for cx, cy in cells)
            if all(cy < VISIBLE_TOP for _, cy in final):
                continue
            seen.add(final)
    return len(seen)


def board_empowerment(rows: list[int]) -> float:
    """log(1 + reachable placements), averaged over the seven piece types."""
    return float(np.mean([math.log1p(placement_count(rows, p)) for p in PieceType]))


def occupancy_to_rows(board: np.ndarray) -> list[int]:
    """Rebuild bitboard rows from a visible (20, 10) occupancy grid."""
    rows = [0] * VISIBLE_TOP
    for row in board:
        value = 0
        for x, cell in enumerate(row):
            if cell:
                value |= 1 << x
        rows.append(value)
    return rows


class EmpowermentReward:
    """Per-placement empowerment of the locked board. Stateless: no θ to fit."""

    def __call__(self, board: np.ndarray) -> float:
        return board_empowerment(occupancy_to_rows(board))


# -- multi-step: the horizon at which danger becomes visible ---------------
#
# One step ahead, empowerment is blind: an empty board, a flat stack and a
# sixteen-high tower all offer exactly the same placements, because a tower
# does not remove choices — the piece still fits in every column, it just lands
# higher. Options only vanish at the instant a column is unreachable, which is
# the instant of death. Danger is only visible at longer horizons.

from blockwave.core.constants import FULL_ROW  # noqa: E402


def _first_filled(rows: tuple[int, ...] | list[int]) -> list[list[int]]:
    """``table[c][r]``: the first filled row at or below ``r`` in column ``c``.

    Built once per board, it turns a straight drop into an O(4) lookup instead
    of stepping the piece down a row at a time. Exact including overhangs: a
    falling cell stops at the first obstruction *below its own starting row*,
    which is what the table indexes, not the column's topmost filled cell.
    """
    table = []
    for col in range(BOARD_WIDTH):
        bit = 1 << col
        column = [TOTAL_HEIGHT] * (TOTAL_HEIGHT + 1)
        for row in range(TOTAL_HEIGHT - 1, -1, -1):
            column[row] = row if rows[row] & bit else column[row + 1]
        table.append(column)
    return table


def _placements_reference(rows: tuple[int, ...], piece: PieceType):
    """The straightforward version. Kept so the fast one can be checked against it."""
    base = list(rows)
    seen: set[tuple[int, ...]] = set()
    for rotation in _SHAPES[piece]:
        for x in range(-3, BOARD_WIDTH):
            cells = [(x + bx, SPAWN_Y + by) for bx, by in rotation]
            if _collides(base, cells):
                continue
            drop = 0
            while not _collides(base, [(cx, cy + drop + 1) for cx, cy in cells]):
                drop += 1
            final = [(cx, cy + drop) for cx, cy in cells]
            if all(cy < VISIBLE_TOP for _, cy in final):
                continue
            after = base[:]
            for cx, cy in final:
                after[cy] |= 1 << cx
            kept = [r for r in after if r != FULL_ROW]
            board = tuple([0] * (TOTAL_HEIGHT - len(kept)) + kept)
            if board not in seen:
                seen.add(board)
                yield board


def _placements(rows: tuple[int, ...], piece: PieceType):
    """Resulting boards (line clears applied), excluding top-outs. Deduped."""
    base = list(rows)
    below = _first_filled(rows)
    seen: set[tuple[int, ...]] = set()
    for rotation in _SHAPES[piece]:
        for x in range(-3, BOARD_WIDTH):
            cells = [(x + bx, SPAWN_Y + by) for bx, by in rotation]
            if _collides(base, cells):
                continue
            drop = min(below[cx][cy + 1] - 1 - cy for cx, cy in cells)
            final = [(cx, cy + drop) for cx, cy in cells]
            if all(cy < VISIBLE_TOP for _, cy in final):
                continue
            after = base[:]
            for cx, cy in final:
                after[cy] |= 1 << cx
            kept = [r for r in after if r != FULL_ROW]
            board = tuple([0] * (TOTAL_HEIGHT - len(kept)) + kept)
            if board not in seen:
                seen.add(board)
                yield board


def reachable(rows: list[int], queue: list[PieceType]) -> set[tuple[int, ...]]:
    """Distinct boards reachable by placing ``queue`` in order."""
    frontier = {tuple(rows)}
    for piece in queue:
        frontier = {board for current in frontier for board in _placements(current, piece)}
        if not frontier:
            break
    return frontier


def horizon_empowerment(rows: list[int], queue: list[PieceType]) -> float:
    """log(1 + distinct surviving boards) over the known upcoming pieces."""
    return math.log1p(len(reachable(rows, queue)))


class HorizonEmpowermentReward:
    """Queue-corrected horizon empowerment: the reward the agent would optimize.

    Raw empowerment swings with the upcoming pieces — an I piece has 17
    placements, a T has 34, an O has 9 — regardless of board quality. That is
    exogenous noise: the agent cannot influence the queue. Subtracting the
    empty board's empowerment for the *same* queue cancels it:

        r = E(board, queue) - E(empty, queue)

    This is a control variate, not a preference. It depends only on the queue,
    so it shifts every policy's expected return by the same amount and changes
    no ranking. Measured effect: per-placement std fell from 0.632 to 0.002 for
    competent play, and pooled z against the exploits rose from 2.5-4.5 to
    12.5-15.6, with the ordering unchanged.

    The result reads as a loss-of-control signal: zero while the board has
    headroom, negative as the agent's options shrink.
    """

    def __init__(self, horizon: int = 2) -> None:
        self.horizon = horizon
        self._baselines: dict[tuple[PieceType, ...], float] = {}

    def _baseline(self, queue: tuple[PieceType, ...]) -> float:
        value = self._baselines.get(queue)
        if value is None:
            value = horizon_empowerment([0] * TOTAL_HEIGHT, list(queue))
            self._baselines[queue] = value
        return value

    def __call__(self, rows: list[int], queue: list[PieceType]) -> float:
        key = tuple(queue[: self.horizon])
        return horizon_empowerment(rows, list(key)) - self._baseline(key)
