"""Piece selection: the guideline 7-bag randomizer.

Every permutation of the seven tetrominoes is dealt before any repeats, which
bounds the worst-case drought at 12 pieces. That bound is what makes the game
fair for a human and what makes the learning problem stationary for an agent —
a naive uniform randomizer can starve an agent of I pieces for long stretches
and inject variance that has nothing to do with its policy.
"""

from __future__ import annotations

import random
from collections import deque

from .constants import PieceType

ALL_PIECES: tuple[PieceType, ...] = tuple(PieceType)


class SevenBag:
    """Seeded 7-bag generator with a lookahead queue."""

    __slots__ = ("_rng", "_bag", "_queue", "_preview")

    def __init__(self, seed: int | None = None, preview: int = 5) -> None:
        self._rng = random.Random(seed)
        self._preview = preview
        self._bag: list[PieceType] = []
        self._queue: deque[PieceType] = deque()
        self._fill()

    def _refill_bag(self) -> None:
        self._bag = list(ALL_PIECES)
        self._rng.shuffle(self._bag)

    def _fill(self) -> None:
        """Top the lookahead queue up to the preview length."""
        while len(self._queue) <= self._preview:
            if not self._bag:
                self._refill_bag()
            self._queue.append(self._bag.pop())

    def next(self) -> PieceType:
        """Take the next piece and advance the queue."""
        piece = self._queue.popleft()
        self._fill()
        return piece

    def peek(self, count: int | None = None) -> list[PieceType]:
        """The upcoming pieces, without consuming them."""
        if count is None:
            count = self._preview
        return list(self._queue)[:count]

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._bag = []
        self._queue.clear()
        self._fill()
