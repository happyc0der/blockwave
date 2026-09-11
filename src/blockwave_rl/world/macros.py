"""Key-press programs the agent can run blind, and learn the effect of.

The board-state reward counted reachable placements by enumerating them with
the simulator. A pixel agent has no simulator, so it has to find out what its
keys do by pressing them and watching. These macros are the vocabulary it
babbles in: each is a fixed sequence of key presses, chosen without reference to
the board, the piece or the rules —

    rotate r times, step k cells one way, hard drop

Every one is executable from any state. Some will do nothing (a rotation that
will not fit, a step into a wall), and that is the point: which macros still
lead somewhere *different* is exactly what "how much control do I have" means
here, and it is what the learned model has to predict.

These exist only inside the reward's imagination. The policy still presses one
key per decision and never sees a macro.
"""

from __future__ import annotations

from dataclasses import dataclass

from blockwave.core.constants import Action

#: Cells to step sideways. Five reaches most of a ten-wide board from spawn.
MAX_STEPS = 5


@dataclass(frozen=True, slots=True)
class Macro:
    rotations: int
    direction: int  # -1 left, +1 right, 0 when steps == 0
    steps: int

    @property
    def actions(self) -> tuple[int, ...]:
        turn = (int(Action.ROTATE_CW),) * self.rotations
        key = int(Action.LEFT) if self.direction < 0 else int(Action.RIGHT)
        return turn + (key,) * self.steps + (int(Action.HARD_DROP),)

    def __str__(self) -> str:
        side = "" if self.steps == 0 else f"{'L' if self.direction < 0 else 'R'}{self.steps}"
        return f"cw{self.rotations}{side}"


def build() -> tuple[Macro, ...]:
    """Every program: 4 rotations x (stay, or step 1..MAX_STEPS either way)."""
    out = []
    for rotations in range(4):
        out.append(Macro(rotations, 0, 0))
        for steps in range(1, MAX_STEPS + 1):
            out.append(Macro(rotations, -1, steps))
            out.append(Macro(rotations, +1, steps))
    return tuple(out)


#: The fixed vocabulary. Its order is part of the model's input encoding, so it
#: must not change between training a model and using it.
MACROS = build()
N_MACROS = len(MACROS)
