"""Keyboard handling: classic controls with proper DAS and ARR.

The engine takes one action per tick, so this module's job is turning the
*held* state of the keyboard into that stream — which is where a falling-block game
either feels crisp or feels like mud.

Two timings do the work, both in milliseconds and both tunable:

``DAS`` (delayed auto-shift)
    How long you hold left or right before the piece starts repeating.
``ARR`` (auto-repeat rate)
    How fast it repeats once it starts.

The DAS charge deliberately survives a piece locking, so holding a direction
while pieces spawn slides each one to the wall immediately instead of stalling
for another DAS interval on every new piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from ..core.constants import Action

#: Defaults tuned to feel like a modern guideline game.
DEFAULT_DAS_MS = 133.0
DEFAULT_ARR_MS = 10.0
DEFAULT_SOFT_DROP_MS = 25.0


@dataclass(slots=True)
class KeyBinds:
    """Classic controls — requirement 2. Every entry is remappable."""

    left: tuple[int, ...] = (pygame.K_LEFT,)
    right: tuple[int, ...] = (pygame.K_RIGHT,)
    soft_drop: tuple[int, ...] = (pygame.K_DOWN,)
    hard_drop: tuple[int, ...] = (pygame.K_SPACE,)
    rotate_cw: tuple[int, ...] = (pygame.K_UP, pygame.K_x)
    rotate_ccw: tuple[int, ...] = (pygame.K_z, pygame.K_LCTRL, pygame.K_RCTRL)
    hold: tuple[int, ...] = (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT)
    pause: tuple[int, ...] = (pygame.K_ESCAPE, pygame.K_p)
    confirm: tuple[int, ...] = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
    quit: tuple[int, ...] = (pygame.K_q,)


@dataclass(slots=True)
class InputConfig:
    das_ms: float = DEFAULT_DAS_MS
    arr_ms: float = DEFAULT_ARR_MS
    soft_drop_ms: float = DEFAULT_SOFT_DROP_MS
    binds: KeyBinds = field(default_factory=KeyBinds)


class InputState:
    """Turns held keys into a per-tick action stream."""

    def __init__(self, config: InputConfig | None = None) -> None:
        self.config = config or InputConfig()
        self._direction = 0          # -1 left, +1 right, 0 neither
        self._das_timer = 0.0
        self._arr_timer = 0.0
        self._charged = False
        self._soft_timer = 0.0
        self._pending: list[Action] = []

    # -- events -----------------------------------------------------------

    def key_down(self, key: int) -> None:
        """Queue the one-shot actions and (re)start auto-shift."""
        binds = self.config.binds

        if key in binds.left:
            self._start_shift(-1)
            self._pending.append(Action.LEFT)
        elif key in binds.right:
            self._start_shift(+1)
            self._pending.append(Action.RIGHT)
        elif key in binds.rotate_cw:
            self._pending.append(Action.ROTATE_CW)
        elif key in binds.rotate_ccw:
            self._pending.append(Action.ROTATE_CCW)
        elif key in binds.hard_drop:
            self._pending.append(Action.HARD_DROP)
        elif key in binds.hold:
            self._pending.append(Action.HOLD)
        elif key in binds.soft_drop:
            self._pending.append(Action.SOFT_DROP)
            self._soft_timer = 0.0

    def key_up(self, key: int) -> None:
        binds = self.config.binds
        # Only the key for the direction currently held releases the shift;
        # lifting the other one changes nothing.
        if (key in binds.left and self._direction == -1) or (key in binds.right and self._direction == +1):
            self._release_shift()

    def _start_shift(self, direction: int) -> None:
        # The newest direction wins, so tapping the opposite way while holding
        # one direction reverses immediately rather than being ignored.
        self._direction = direction
        self._das_timer = 0.0
        self._arr_timer = 0.0
        self._charged = False

    def _release_shift(self) -> None:
        self._direction = 0
        self._das_timer = 0.0
        self._arr_timer = 0.0
        self._charged = False

    # -- per tick ---------------------------------------------------------

    def poll(self, dt: float, keys) -> list[Action]:
        """Actions for this tick: one-shots first, then any auto-repeats."""
        actions = self._pending
        self._pending = []

        ms = dt * 1000.0
        binds = self.config.binds

        if self._direction:
            actions.extend(self._auto_shift(ms))

        if any(keys[key] for key in binds.soft_drop):
            self._soft_timer += ms
            while self._soft_timer >= self.config.soft_drop_ms:
                self._soft_timer -= self.config.soft_drop_ms
                actions.append(Action.SOFT_DROP)
        else:
            self._soft_timer = 0.0

        return actions

    def _auto_shift(self, ms: float) -> list[Action]:
        action = Action.LEFT if self._direction < 0 else Action.RIGHT
        repeats: list[Action] = []

        if not self._charged:
            self._das_timer += ms
            if self._das_timer < self.config.das_ms:
                return repeats
            # Carry the overshoot into the repeat clock so DAS stays accurate
            # regardless of how the frame boundaries happened to fall.
            self._arr_timer = self._das_timer - self.config.das_ms
            self._charged = True
        else:
            self._arr_timer += ms

        if self.config.arr_ms <= 0.0:
            # ARR 0 means "teleport to the wall": the caller repeats until the
            # move stops having an effect.
            repeats.append(action)
            return repeats

        while self._arr_timer >= self.config.arr_ms:
            self._arr_timer -= self.config.arr_ms
            repeats.append(action)
        return repeats

    def on_piece_locked(self) -> None:
        """Keep the DAS charge across a lock.

        Without this, holding a direction stalls for a fresh DAS interval on
        every new piece, which is the single most common way a falling-block game
        feels sluggish.
        """
        # Deliberately does nothing: the charge simply is not reset. The method
        # exists so the game loop can express the intent, and so the test suite
        # has something to assert against.