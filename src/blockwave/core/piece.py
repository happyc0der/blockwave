"""The active falling piece, and SRS rotation against a board."""

from __future__ import annotations

from dataclasses import dataclass

from .board import Board
from .constants import (
    SHAPES,
    SPAWN_X,
    SPAWN_Y,
    PieceType,
    kick_table,
)


@dataclass(slots=True)
class Piece:
    """A tetromino in play: its type, its bounding-box origin, its rotation."""

    type: PieceType
    x: int = SPAWN_X
    y: int = SPAWN_Y
    rotation: int = 0

    @classmethod
    def spawn(cls, piece_type: PieceType) -> Piece:
        return cls(type=piece_type, x=SPAWN_X, y=SPAWN_Y, rotation=0)

    def cells(self) -> tuple[tuple[int, int], ...]:
        """Absolute board cells this piece currently occupies."""
        x, y = self.x, self.y
        return tuple((x + bx, y + by) for bx, by in SHAPES[self.type][self.rotation])

    def cells_at(self, x: int, y: int, rotation: int) -> tuple[tuple[int, int], ...]:
        """Absolute cells the piece *would* occupy at a hypothetical placement."""
        return tuple((x + bx, y + by) for bx, by in SHAPES[self.type][rotation])

    def copy(self) -> Piece:
        return Piece(self.type, self.x, self.y, self.rotation)


class RotationResult:
    """Outcome of a rotation attempt.

    ``kick_index`` records *which* SRS candidate succeeded. That matters beyond
    bookkeeping: a T piece that rotates in using the final candidate always
    scores as a full T-spin rather than a mini.
    """

    __slots__ = ("kick_index", "success")

    def __init__(self, success: bool, kick_index: int = -1) -> None:
        self.success = success
        self.kick_index = kick_index

    def __bool__(self) -> bool:
        return self.success


FAILED_ROTATION = RotationResult(False)


def try_move(piece: Piece, board: Board, dx: int, dy: int) -> bool:
    """Translate ``piece`` by ``(dx, dy)`` if that does not collide.

    Mutates the piece in place and returns whether the move happened.
    """
    cells = piece.cells_at(piece.x + dx, piece.y + dy, piece.rotation)
    if board.collides(cells):
        return False
    piece.x += dx
    piece.y += dy
    return True


def try_rotate(piece: Piece, board: Board, direction: int) -> RotationResult:
    """Rotate ``piece`` by ``direction`` (+1 clockwise, -1 anticlockwise).

    Walks the SRS kick candidates for this transition in order and takes the
    first that fits. If all five collide the rotation is rejected and the piece
    is left untouched.
    """
    from_state = piece.rotation
    to_state = (from_state + direction) % 4

    table = kick_table(piece.type)
    if table is None:
        # The O piece has no kick table and its four states are identical, so
        # rotating it is a no-op that always "succeeds".
        piece.rotation = to_state
        return RotationResult(True, 0)

    offsets = table[(from_state, to_state)]
    for index, (dx, dy) in enumerate(offsets):
        nx, ny = piece.x + dx, piece.y + dy
        if not board.collides(piece.cells_at(nx, ny, to_state)):
            piece.x = nx
            piece.y = ny
            piece.rotation = to_state
            return RotationResult(True, index)

    return FAILED_ROTATION


def drop_distance(piece: Piece, board: Board) -> int:
    """How many rows the piece can fall before landing."""
    distance = 0
    while not board.collides(piece.cells_at(piece.x, piece.y + distance + 1, piece.rotation)):
        distance += 1
    return distance


def ghost_cells(piece: Piece, board: Board) -> tuple[tuple[int, int], ...]:
    """Cells the piece would occupy after a hard drop.

    This is defined in terms of :func:`drop_distance`, the same function the
    hard drop itself uses, so the ghost can never disagree with where the piece
    actually lands.
    """
    return piece.cells_at(piece.x, piece.y + drop_distance(piece, board), piece.rotation)


def is_grounded(piece: Piece, board: Board) -> bool:
    """Whether the piece is resting on the stack or the floor."""
    return board.collides(piece.cells_at(piece.x, piece.y + 1, piece.rotation))
