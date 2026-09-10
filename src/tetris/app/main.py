"""The playable game: a fixed-timestep loop driving the engine and compositor.

Requirement 1 — "high tick rate" — is met here, and it means a high *logic*
rate rather than merely a high frame rate. Logic advances in fixed
:data:`LOGIC_DT` slices at 240 Hz while rendering happens at whatever the
display refreshes at, so the game feels identical on a 60 Hz laptop panel and a
144 Hz monitor, and DAS timings stay honest on both.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pygame

from ..core.constants import Action
from ..core.engine import EngineConfig, TetrisEngine
from ..core.events import EventType, GameEvent
from ..render.compositor import Compositor
from ..render.display import Display, refresh_rate
from ..render.font import GLYPH_W, draw_text_centered, text_size
from ..render.layout import HUMAN_CELL_PX, Layout
from ..render.palette import CYAN, TEXT_DIM, TEXT_HOT
from .input import InputConfig, InputState

#: Logic ticks per second. Well above any display rate, so input timing and
#: gravity are sampled far more finely than the eye is refreshed.
LOGIC_HZ = 240
LOGIC_DT = 1.0 / LOGIC_HZ

#: Never advance more than this much simulated time in one frame. Without the
#: clamp, a stall (dragging the window, waking from sleep) hands the loop a
#: huge delta and the game fast-forwards through several pieces.
MAX_FRAME_TIME = 0.25


class Scene(Enum):
    TITLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()


@dataclass(slots=True)
class ShakeState:
    """Screen shake, driven from the engine's event stream.

    Kept here rather than in the compositor because it is presentation state,
    not scene content — and because the RL environment drives the same
    compositor without it.
    """

    magnitude: float = 0.0
    decay: float = 9.0
    _rng: random.Random = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = random.Random(0xC0FFEE)

    def kick(self, amount: float) -> None:
        self.magnitude = max(self.magnitude, amount)

    def update(self, dt: float) -> None:
        self.magnitude = max(0.0, self.magnitude - self.decay * dt)

    def offset(self, scale: float) -> tuple[int, int]:
        if self.magnitude <= 0.0 or scale <= 0.0:
            return (0, 0)
        span = self.magnitude * scale
        return (
            int(round(self._rng.uniform(-span, span))),
            int(round(self._rng.uniform(-span, span))),
        )


class Game:
    def __init__(
        self,
        *,
        seed: int | None = None,
        start_level: int = 1,
        profile: str = "arcade",
        cell_px: int = HUMAN_CELL_PX,
        input_config: InputConfig | None = None,
    ) -> None:
        pygame.init()

        self.engine = TetrisEngine(EngineConfig(seed=seed, start_level=start_level))
        self.layout = Layout(cell_px=cell_px)
        self.compositor = Compositor(self.layout, profile)
        self.display = Display(self.compositor.frame_size[::-1], title="TETRIS // MIAMI")
        self.input = InputState(input_config)
        self.shake = ShakeState()

        self.scene = Scene.TITLE
        self.running = True
        self._start_level = start_level

    # -- loop -------------------------------------------------------------

    def run(self) -> None:
        clock = pygame.time.Clock()
        target_fps = refresh_rate(60)
        accumulator = 0.0

        while self.running:
            frame_time = min(clock.tick(target_fps) / 1000.0, MAX_FRAME_TIME)
            self._handle_events()

            if self.scene is Scene.PLAYING:
                accumulator += frame_time
                keys = pygame.key.get_pressed()
                while accumulator >= LOGIC_DT:
                    self._tick(LOGIC_DT, keys)
                    accumulator -= LOGIC_DT
            else:
                accumulator = 0.0

            self.shake.update(frame_time)
            self._present()

        pygame.quit()

    def _tick(self, dt: float, keys) -> None:
        """One fixed logic slice."""
        actions = self.input.poll(dt, keys)

        if not actions:
            self._apply(Action.NOOP, dt)
            return

        # Time passes once per tick no matter how many actions landed in it,
        # otherwise a burst of auto-repeat would also accelerate gravity.
        for index, action in enumerate(actions):
            self._apply(action, dt if index == 0 else 0.0)
            if self.engine.game_over:
                break

    def _apply(self, action: Action, dt: float) -> None:
        events = self.engine.step(action, dt)
        if events:
            self._react(events)

    def _react(self, events: list[GameEvent]) -> None:
        for event in events:
            if event.type is EventType.HARD_DROP and event.value:
                self.shake.kick(0.35)
            elif event.type is EventType.LINE_CLEAR:
                self.shake.kick(0.5 if event.value < 4 else 1.4)
            elif event.type is EventType.PIECE_LOCK:
                self.input.on_piece_locked()
            elif event.type is EventType.GAME_OVER:
                self.scene = Scene.GAME_OVER

    # -- events -----------------------------------------------------------

    def _handle_events(self) -> None:
        binds = self.input.config.binds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type != pygame.KEYDOWN:
                if event.type == pygame.KEYUP and self.scene is Scene.PLAYING:
                    self.input.key_up(event.key)
                continue

            key = event.key
            if self.scene is Scene.TITLE:
                if key in binds.quit:
                    self.running = False
                elif key in binds.confirm:
                    self._new_game()
            elif self.scene is Scene.PLAYING:
                if key in binds.pause:
                    self.scene = Scene.PAUSED
                else:
                    self.input.key_down(key)
            elif self.scene is Scene.PAUSED:
                if key in binds.quit:
                    self.scene = Scene.TITLE
                elif key in binds.pause or key in binds.confirm:
                    self.scene = Scene.PLAYING
            elif self.scene is Scene.GAME_OVER:
                if key in binds.quit:
                    self.scene = Scene.TITLE
                elif key in binds.confirm:
                    self._new_game()

    def _new_game(self) -> None:
        self.engine.reset(seed=random.randrange(1 << 30))
        self.engine.stats.level = self._start_level
        self.input = InputState(self.input.config)
        self.shake.magnitude = 0.0
        self.scene = Scene.PLAYING

    # -- presentation -----------------------------------------------------

    def _present(self) -> None:
        shake = self.shake.offset(self.layout.cell_px * 0.35)
        frame = self.compositor.render(self.engine, shake=shake)

        if self.scene is Scene.TITLE:
            self._overlay(frame, "TETRIS", "PRESS ENTER", TEXT_HOT)
        elif self.scene is Scene.PAUSED:
            self._overlay(frame, "PAUSED", "ENTER TO RESUME", CYAN)
        elif self.scene is Scene.GAME_OVER:
            self._overlay(frame, "", f"SCORE {self.engine.stats.score}", TEXT_HOT, dim=False)

        self.display.present(frame)

    def _overlay(
        self, frame: np.ndarray, title: str, subtitle: str, color, dim: bool = True
    ) -> None:
        height, width = frame.shape[:2]
        if dim:
            frame //= 3

        cx = width // 2
        cy = height // 2
        cell = self.layout.cell_px

        if title:
            scale = max(2, (width // 2) // ((GLYPH_W + 1) * max(1, len(title))))
            _, th = text_size(title, scale)
            draw_text_centered(frame, title, cx, cy - th, color, scale)

        sub_scale = max(1, cell // 8)
        draw_text_centered(frame, subtitle, cx, cy + cell, TEXT_DIM if dim else color, sub_scale)


def main(argv: list[str] | None = None) -> int:
    Game().run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
