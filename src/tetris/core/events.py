"""The event stream the engine emits each step.

Audio and visual effects consume these and never read engine state directly.
That one-way flow is what keeps the renderer and the sound bank from quietly
coupling themselves to internals, and it means a replay of the event stream is
enough to reproduce everything a player saw and heard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto


class EventType(IntEnum):
    PIECE_SPAWN = auto()
    PIECE_MOVE = auto()
    PIECE_ROTATE = auto()
    PIECE_HOLD = auto()
    HOLD_DENIED = auto()
    SOFT_DROP = auto()
    HARD_DROP = auto()
    PIECE_LOCK = auto()
    LINE_CLEAR = auto()
    TSPIN = auto()
    PERFECT_CLEAR = auto()
    COMBO = auto()
    LEVEL_UP = auto()
    GAME_OVER = auto()


@dataclass(slots=True, frozen=True)
class GameEvent:
    """A single thing that happened.

    ``value`` carries the one number each event type needs — lines cleared for
    ``LINE_CLEAR``, the new level for ``LEVEL_UP``, rows fallen for
    ``HARD_DROP``, the T-spin grade for ``TSPIN`` — and is zero otherwise.

    ``rows`` carries the board rows a ``LINE_CLEAR`` removed. The renderer needs
    to know *which* rows went in order to animate them, and a count alone cannot
    say that.
    """

    type: EventType
    value: int = 0
    rows: tuple[int, ...] = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.type.name}({self.value})" if self.value else self.type.name
