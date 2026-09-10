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
from ..render.compositor import Compositor, stroke_rect
from ..render.display import Display, refresh_rate
from ..render.font import draw_text_centered, fit_scale, text_size
from ..render.layout import HUMAN_CELL_PX, Layout, Rect
from ..render.palette import CYAN, PURPLE, RGB, TEXT_DIM, TEXT_HOT
from ..audio.bank import SoundBank
from ..audio.synth import tempo_band
from . import scores as scores_store
from .input import InputConfig, InputState

#: Logic ticks per second. Well above any display rate, so input timing and
#: gravity are sampled far more finely than the eye is refreshed.
LOGIC_HZ = 240
LOGIC_DT = 1.0 / LOGIC_HZ

#: Never advance more than this much simulated time in one frame. Without the
#: clamp, a stall (dragging the window, waking from sleep) hands the loop a
#: huge delta and the game fast-forwards through several pieces.
MAX_FRAME_TIME = 0.25

#: How long an announcement (level up, T-spin, perfect clear) stays on screen.
TOAST_SECONDS = 1.2


class Scene(Enum):
    TITLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()


@dataclass(frozen=True, slots=True)
class PanelLine:
    """One line of an overlay panel.

    ``size`` is a role rather than a number so the panel scales with ``cell_px``
    and the score can be the visual hero without any call site hard-coding a
    pixel value.
    """

    text: str
    size: str  # "hero" | "title" | "body" | "label"
    color: RGB
    gap_before: float = 0.0  # extra space above this line, in cell units


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
        audio: bool = True,
        music: bool = True,
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

        self.audio = SoundBank(enabled=audio)
        self.music_enabled = music and self.audio.enabled

        self.scores = scores_store.load()
        self._new_record = False
        self._toast: tuple[str, float] | None = None

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
            self._update_toast(frame_time)
            self._present()

        self.audio.close()
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
        self.audio.handle(events)

        for event in events:
            if event.type is EventType.HARD_DROP and event.value:
                self.shake.kick(0.35)
            elif event.type is EventType.LINE_CLEAR:
                self.shake.kick(0.5 if event.value < 4 else 1.4)
            elif event.type is EventType.PIECE_LOCK:
                self.input.on_piece_locked()
            elif event.type is EventType.LEVEL_UP:
                # Requirement 3 should be felt, not read off a number.
                self._toast = (f"LEVEL {event.value}", TOAST_SECONDS)
                self.shake.kick(0.8)
                self._sync_music(event.value)
            elif event.type is EventType.TSPIN:
                self._toast = ("T-SPIN", TOAST_SECONDS * 0.7)
            elif event.type is EventType.PERFECT_CLEAR:
                self._toast = ("PERFECT CLEAR", TOAST_SECONDS)
            elif event.type is EventType.GAME_OVER:
                self._new_record = scores_store.submit(self.scores, self.engine.stats)
                scores_store.save(self.scores)
                self.audio.stop_music()
                self.scene = Scene.GAME_OVER

    def _sync_music(self, level: int) -> None:
        """Move the loop to the tempo band this level belongs to."""
        if self.music_enabled:
            self.audio.play_music(tempo_band(level))

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
                    self.audio.play("menu_select")
                    self._new_game()
            elif self.scene is Scene.PLAYING:
                if key in binds.pause:
                    self.audio.play("pause")
                    self.audio.pause_music()
                    self.scene = Scene.PAUSED
                else:
                    self.input.key_down(key)
            elif self.scene is Scene.PAUSED:
                if key in binds.quit:
                    self.audio.play("menu_back")
                    self.audio.stop_music()
                    self.scene = Scene.TITLE
                elif key in binds.pause or key in binds.confirm:
                    self.audio.play("menu_select")
                    self.audio.resume_music()
                    self.scene = Scene.PLAYING
            elif self.scene is Scene.GAME_OVER:
                if key in binds.quit:
                    self.audio.play("menu_back")
                    self.scene = Scene.TITLE
                elif key in binds.confirm:
                    self._new_game()

    def _update_toast(self, dt: float) -> None:
        if self._toast is None:
            return
        text, remaining = self._toast
        remaining -= dt
        self._toast = None if remaining <= 0.0 else (text, remaining)

    def _new_game(self) -> None:
        # reset() already applies start_level from the engine config.
        self.engine.reset(seed=random.randrange(1 << 30))
        self.input = InputState(self.input.config)
        self.shake.magnitude = 0.0
        self._new_record = False
        self._toast = None
        self._sync_music(self.engine.stats.level)
        self.scene = Scene.PLAYING

    # -- presentation -----------------------------------------------------

    def _present(self) -> None:
        shake = self.shake.offset(self.layout.cell_px * 0.35)
        frame = self.compositor.render(self.engine, shake=shake)

        if self._toast is not None and self.scene is Scene.PLAYING:
            self._draw_toast(frame)

        lines = self._panel_lines()
        if lines:
            self._draw_panel(frame, lines)

        self.display.present(frame)

    def _draw_toast(self, frame: np.ndarray) -> None:
        """A brief announcement over the board, fading out.

        Unlike the scene panels this does not dim the frame — the game is still
        being played underneath it and must stay readable.
        """
        text, remaining = self._toast
        board = self.layout.board
        cell = self.layout.cell_px

        fade = min(1.0, remaining / (TOAST_SECONDS * 0.5))
        scale = fit_scale(text, board.w - cell, max(2, cell // 7))
        width, height = text_size(text, scale)

        cx = board.x + board.w // 2
        y = board.y + board.h // 4

        pad = max(2, cell // 4)
        plate = Rect(cx - width // 2 - pad, y - pad, width + 2 * pad, height + 2 * pad)
        region = frame[plate.y : plate.bottom, plate.x : plate.right]
        region[:] = (region * (1.0 - 0.75 * fade)).astype(frame.dtype)

        color = tuple(int(c * fade) for c in TEXT_HOT)
        draw_text_centered(frame, text, cx, y, color, scale)

    def _panel_lines(self) -> list[PanelLine]:
        """The overlay for the current scene, or nothing while playing."""
        if self.scene is Scene.TITLE:
            lines = [
                PanelLine("TETRIS", "hero", TEXT_HOT),
                PanelLine("MIAMI", "body", PURPLE),
            ]
            if self.scores.best:
                lines.append(PanelLine(f"BEST {self.scores.best:,}", "body", CYAN, 0.5))
            lines.append(PanelLine("PRESS ENTER", "label", TEXT_DIM, 0.7))
            return lines

        if self.scene is Scene.PAUSED:
            return [
                PanelLine("PAUSED", "title", CYAN),
                PanelLine("ENTER TO RESUME", "label", TEXT_DIM, 0.7),
                PanelLine("Q TO QUIT", "label", TEXT_DIM),
            ]

        if self.scene is Scene.GAME_OVER:
            stats = self.engine.stats
            lines = [
                PanelLine("GAME OVER", "title", TEXT_HOT),
                PanelLine("SCORE", "label", TEXT_DIM, 0.6),
                # The hero line: the score is the largest thing on screen.
                PanelLine(f"{stats.score:,}", "hero", CYAN),
            ]
            if self._new_record:
                lines.append(PanelLine("NEW RECORD", "body", TEXT_HOT, 0.3))
            elif self.scores.best:
                lines.append(PanelLine(f"BEST {self.scores.best:,}", "body", TEXT_DIM, 0.3))
            lines.append(
                PanelLine(f"LINES {stats.lines}  LEVEL {stats.level}", "body", TEXT_DIM, 0.3)
            )
            # Two short lines rather than one long one: at small cell sizes a
            # single 22-character prompt cannot shrink far enough to fit a
            # 10-cell-wide board, since the font bottoms out at scale 1.
            lines.append(PanelLine("ENTER RESTART", "label", TEXT_DIM, 0.7))
            lines.append(PanelLine("Q QUIT", "label", TEXT_DIM))
            return lines

        return []

    def _scale_for(self, size: str) -> int:
        cell = self.layout.cell_px
        return {
            "hero": max(3, cell // 5),
            "title": max(2, cell // 9),
            "body": max(1, cell // 12),
            "label": max(1, cell // 14),
        }[size]

    def _draw_panel(self, frame: np.ndarray, lines: list[PanelLine], dim: float = 0.30) -> None:
        """Lay a stack of lines out on a plate centred on the board.

        Every line is measured and placed from a running cursor, so lines cannot
        collide however long the text or however small the cells — which is
        exactly the failure this replaced.
        """
        board = self.layout.board
        cell = self.layout.cell_px
        pad = max(3, cell // 2)
        max_text_w = board.w - 2 * pad - 2

        # Measure first: resolve each line's scale, then the block it occupies.
        measured: list[tuple[PanelLine, int, int, int]] = []
        block_h = 0
        block_w = 0
        for index, line in enumerate(lines):
            scale = fit_scale(line.text, max_text_w, self._scale_for(line.size))
            width, height = text_size(line.text, scale)
            gap = int(line.gap_before * cell) if index else 0
            measured.append((line, scale, width, height))
            block_h += gap + height
            block_w = max(block_w, width)

        # Clamp to the board. The font bottoms out at scale 1, so a line that is
        # still too wide there would otherwise push the plate out over the side
        # panels; better to let that one line touch the border than to have the
        # panel escape the playfield.
        plate_w = min(block_w + 2 * pad, board.w)
        plate = Rect(
            board.x + (board.w - plate_w) // 2,
            board.y + (board.h - block_h) // 2 - pad,
            plate_w,
            block_h + 2 * pad,
        )

        frame[:] = (frame * dim).astype(frame.dtype)
        region = frame[plate.y : plate.bottom, plate.x : plate.right]
        region[:] = (region * 0.35).astype(frame.dtype)
        stroke_rect(frame, plate, CYAN, max(1, cell // 12))

        cx = board.x + board.w // 2
        y = plate.y + pad
        for index, (line, scale, _width, height) in enumerate(measured):
            if index:
                y += int(line.gap_before * cell)
            draw_text_centered(frame, line.text, cx, y, line.color, scale)
            y += height


def main() -> int:
    Game().run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
