"""Sparks thrown off by a line clear.

Presentation state, like the screen shake, so it lives in the app layer rather
than the compositor — the renderer draws the game, not the celebration.

The one rule these obey: **a particle is only drawn outside the playfield**.
They spawn at the cleared row and fly outward past the board's edges, so the
effect reads as energy bursting out of the well while the cells themselves stay
completely unobscured. Clipping them this way makes it structural rather than a
matter of tuning velocities until nothing happens to overlap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..render.layout import Rect

#: Pixels per second squared, in cell units, pulling sparks back down.
GRAVITY_CELLS = 26.0

#: How many sparks each cleared row throws from each side.
PER_ROW_PER_SIDE = 7


@dataclass(slots=True)
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]


class ParticleField:
    """A small pool of sparks, driven from the engine's event stream."""

    def __init__(self, seed: int = 0xBEEF) -> None:
        self._rng = random.Random(seed)
        self.sparks: list[Spark] = []

    def burst(self, board: Rect, cell: int, rows_y: list[int], color: tuple[int, int, int]) -> None:
        """Throw sparks out of both edges of each cleared row."""
        speed = cell * 9.0
        for y in rows_y:
            centre = y + cell / 2.0
            for side, edge in ((-1, board.x), (1, board.right)):
                for _ in range(PER_ROW_PER_SIDE):
                    self.sparks.append(
                        Spark(
                            x=float(edge),
                            y=centre + self._rng.uniform(-cell * 0.4, cell * 0.4),
                            vx=side * speed * self._rng.uniform(0.45, 1.4),
                            vy=self._rng.uniform(-speed * 0.5, speed * 0.2),
                            life=self._rng.uniform(0.30, 0.65),
                            max_life=0.65,
                            color=color,
                        )
                    )

    def update(self, dt: float, cell: int) -> None:
        if not self.sparks:
            return
        pull = GRAVITY_CELLS * cell * dt
        alive: list[Spark] = []
        for spark in self.sparks:
            spark.life -= dt
            if spark.life <= 0.0:
                continue
            spark.x += spark.vx * dt
            spark.y += spark.vy * dt
            spark.vy += pull
            alive.append(spark)
        self.sparks = alive

    def draw(self, frame: np.ndarray, board: Rect, cell: int) -> None:
        """Additively paint every spark that lies outside the playfield."""
        if not self.sparks:
            return
        height, width = frame.shape[:2]
        size = max(1, cell // 8)

        for spark in self.sparks:
            x, y = int(spark.x), int(spark.y)
            # The invariant: never inside the board.
            if board.x <= x < board.right and board.y <= y < board.bottom:
                continue
            if not (0 <= x < width - size and 0 <= y < height - size):
                continue

            fade = max(0.0, min(1.0, spark.life / spark.max_life))
            tint = np.array(spark.color, dtype=np.float32) * fade
            region = frame[y : y + size, x : x + size]
            region[:] = np.minimum(region.astype(np.float32) + tint, 255.0).astype(frame.dtype)

    def clear(self) -> None:
        self.sparks.clear()
