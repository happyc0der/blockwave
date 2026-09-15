"""A scripted reference player.

**Never a reward, a label, or a policy initialization.** It exists for two jobs:

1. Collecting *visual* data for the representation corpus, where only the
   rendered frames are kept and its decisions are discarded.
2. Serving as a competent reference in the adversarial reward tests — a
   well-formed intrinsic reward must rank this player above random play.

It reads engine state to choose a placement, which is fine for a reference and
precisely why it must never feed the agent. It *acts* only through the same
`Action` key presses the agent has, one per step.

Placement is scored with the widely used four-feature linear evaluation
(aggregate height, lines, holes, bumpiness).
"""

from __future__ import annotations

from dataclasses import dataclass

from blockwave.core.board import Board
from blockwave.core.constants import Action
from blockwave.core.engine import Engine
from blockwave.core.piece import Piece, drop_distance

WEIGHTS = {"height": -0.510066, "cleared": 0.760666, "holes": -0.35663, "bumpiness": -0.184483}

#: If the piece cannot reach its target this many steps after spawning, drop it.
STUCK_LIMIT = 24


@dataclass(slots=True)
class _Target:
    #: The piece object itself, not its id(). Python reuses the id of a
    #: garbage-collected object, so a freshly spawned piece can inherit the
    #: previous piece's id — and an id-keyed plan then silently carries over,
    #: step count and all. Holding the reference makes that impossible: the old
    #: piece cannot be collected while we point at it.
    piece: Piece
    rotation: int
    x: int
    steps: int = 0


def _copy(board: Board) -> Board:
    clone = Board()
    clone.rows = list(board.rows)
    clone.colors = board.colors.copy()
    return clone


def evaluate(board: Board, piece: Piece, rotation: int, x: int) -> float | None:
    """Score dropping ``piece`` at (rotation, x) from its current height."""
    trial = Piece(piece.type, x, piece.y, rotation)
    if board.collides(trial.cells()):
        return None
    trial.y += drop_distance(trial, board)
    after = _copy(board)
    after.lock(trial.cells(), trial.type)
    full = after.full_rows()
    after.clear_rows(full)
    return (
        WEIGHTS["height"] * after.aggregate_height()
        + WEIGHTS["cleared"] * len(full)
        + WEIGHTS["holes"] * after.holes()
        + WEIGHTS["bumpiness"] * after.bumpiness()
    )


class HeuristicPlayer:
    """Chooses a placement per piece, then walks to it one key press at a time."""

    def __init__(self) -> None:
        self._target: _Target | None = None

    def reset(self) -> None:
        self._target = None

    def _plan(self, engine: Engine) -> _Target | None:
        piece = engine.piece
        if piece is None:
            return None
        best: tuple[float, int, int] | None = None
        for rotation in range(4):
            for x in range(-3, 10):
                score = evaluate(engine.board, piece, rotation, x)
                if score is not None and (best is None or score > best[0]):
                    best = (score, rotation, x)
        if best is None:
            return None
        return _Target(piece, best[1], best[2])

    def act(self, engine: Engine) -> int:
        piece = engine.piece
        if piece is None:  # clear pause or top-out: nothing to steer
            return int(Action.NOOP)

        if self._target is None or self._target.piece is not piece:
            self._target = self._plan(engine)
            if self._target is None:
                return int(Action.HARD_DROP)

        target = self._target
        target.steps += 1
        if target.steps > STUCK_LIMIT:
            return int(Action.HARD_DROP)
        if piece.rotation != target.rotation:
            return int(Action.ROTATE_CW)
        if piece.x < target.x:
            return int(Action.RIGHT)
        if piece.x > target.x:
            return int(Action.LEFT)
        return int(Action.HARD_DROP)
