"""The scene compositor: one renderer, painted in numpy.

This is the only place the game is drawn. The arcade window and the agent's
observation both come out of :meth:`Compositor.render`, differing only in
``cell_px`` and :class:`VisualProfile` — so whatever the agent learns to read,
the player is looking at too.

Layer order, and why it makes the legibility invariant structural:

1. **Background** — gradient, sun, perspective grid. Painted first, then the
   playfield well is laid over the top of it, so background art can never sit
   on a cell.
2. **Chrome** — panels, strokes, static labels. All outside the playfield rect.
3. **Cells** — ghost, locked stack, active piece. The information layer.
4. **Post-process** — bloom, scanlines, vignette, grade. Global passes that
   modulate the whole frame uniformly and cannot occlude anything.

Steps 1 and 2 never change, so they are rendered once and cached; a frame costs
one array copy plus the cell paint and the post pass.

Note what is *not* here: title, paused and game-over messages. Scene
presentation belongs to the app layer, which owns the scene state anyway. This
module draws the game and never a message over it.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import BOARD_WIDTH, TOTAL_HEIGHT, PieceType
from ..core.constants import SHAPES
from ..core.engine import TetrisEngine
from . import effects, randomize
from .font import draw_text, draw_text_centered, fit_scale
from .layout import RENDER_ROWS, RENDER_TOP, SPAWN_ROWS_SHOWN, Layout, Rect
from .palette import (
    CYAN,
    GRID,
    MAGENTA,
    PANEL,
    PURPLE,
    RGB,
    SPAWN_ZONE,
    SUNSET_BOTTOM,
    SUNSET_TOP,
    TEXT,
    TEXT_DIM,
    TEXT_HOT,
    VOID,
    DEEP,
    WELL,
    piece_lut,
)
from .profiles import VisualProfile, get_profile

#: How much of the background glow is allowed to bleed through the playfield
#: well. Small on purpose: enough that the well feels lit from behind, far too
#: little to compete with a neon block for the dominant colour of a cell. At
#: 0.15 the sun behind the board turned the well a muddy brown, which is
#: exactly the kind of contrast loss the legibility invariant exists to stop.
WELL_BLEED = 0.06

#: Ghost piece brightness, as a fraction of the piece's own colour.
GHOST_ALPHA = 0.26


def fill_rect(frame: np.ndarray, rect: Rect, color: RGB | np.ndarray) -> None:
    frame[rect.y : rect.bottom, rect.x : rect.right] = color


def stroke_rect(frame: np.ndarray, rect: Rect, color: RGB, width: int = 1) -> None:
    frame[rect.y : rect.y + width, rect.x : rect.right] = color
    frame[rect.bottom - width : rect.bottom, rect.x : rect.right] = color
    frame[rect.y : rect.bottom, rect.x : rect.x + width] = color
    frame[rect.y : rect.bottom, rect.right - width : rect.right] = color


class Compositor:
    def __init__(
        self,
        layout: Layout | None = None,
        profile: str | VisualProfile = "arcade",
        *,
        include_hud: bool = True,
    ) -> None:
        self.layout = layout or Layout(cell_px=7)
        self.profile = get_profile(profile)
        self.include_hud = include_hud

        self._lut = piece_lut().astype(np.float32)
        self._bevel = self._build_bevel()
        self._static = self._build_static()

    # -- output geometry --------------------------------------------------

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(height, width)`` of what :meth:`render` returns."""
        if self.include_hud:
            return (self.layout.height, self.layout.width)
        board = self.layout.board
        return (board.h, board.w)

    # -- cached layers ----------------------------------------------------

    def _build_bevel(self) -> np.ndarray:
        """Per-cell shading, tiled across the playfield.

        A lit top-left edge, a shaded bottom-right edge and a dark gutter at the
        cell boundary. The gutter is what keeps two adjacent blocks of the same
        colour from merging into one shape at small cell sizes.
        """
        size = self.layout.cell_px
        bevel = np.ones((size, size, 1), dtype=np.float32)
        edge = max(1, size // 7)

        bevel[:edge, :, 0] *= 1.45
        bevel[:, :edge, 0] *= 1.35
        bevel[-edge:, :, 0] *= 0.62
        bevel[:, -edge:, 0] *= 0.70
        # The outermost pixel row/column is the gutter between cells.
        bevel[0, :, 0] = 0.30
        bevel[:, 0, 0] = 0.30
        return bevel

    def _background(self, frame: np.ndarray) -> None:
        """Vertical gradient, then the sun and grid floor if the profile wants them."""
        height, width = frame.shape[:2]
        top = np.array(VOID, dtype=np.float32)
        bottom = np.array(DEEP, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        frame[:] = top + (bottom - top) * ramp

        if not self.profile.background:
            return

        self._draw_sun(frame)
        self._draw_grid_floor(frame)

    def _draw_sun(self, frame: np.ndarray) -> None:
        """A sunset disc with horizontal slits — the synthwave signature."""
        height, width = frame.shape[:2]
        radius = width * 0.30
        cx, cy = width * 0.5, height * 0.42

        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        disc = distance <= radius

        # Slits widen toward the bottom of the disc, the way the classic art does.
        row = np.arange(height)[:, None]
        band = (row // max(1, int(radius * 0.10))) % 2 == 0
        lower = ys > cy - radius * 0.25
        disc &= ~(band & lower)

        vertical = np.clip((ys - (cy - radius)) / (2 * radius), 0.0, 1.0)[:, :, None]
        top = np.array(SUNSET_TOP, dtype=np.float32)
        bot = np.array(SUNSET_BOTTOM, dtype=np.float32)
        sun = top + (bot - top) * vertical

        mask = disc[:, :, None]
        frame[:] = np.where(mask, frame * 0.25 + sun * 0.55, frame)

    def _draw_grid_floor(self, frame: np.ndarray) -> None:
        """A perspective grid receding to the horizon."""
        height, width = frame.shape[:2]
        horizon = int(height * 0.52)
        color = np.array(PURPLE, dtype=np.float32)

        # Horizontal lines, bunching up toward the horizon.
        step = 1.0
        y = float(height)
        while y > horizon:
            row = int(y)
            if horizon < row < height:
                fade = (row - horizon) / max(1.0, height - horizon)
                frame[row] = frame[row] * (1.0 - 0.35 * fade) + color * (0.35 * fade)
            step *= 1.28
            y -= step

        # Verticals radiating from a vanishing point on the horizon.
        vanish = width * 0.5
        for offset in range(-12, 13):
            x_bottom = vanish + offset * width * 0.11
            for row in range(horizon, height):
                t = (row - horizon) / max(1.0, height - horizon)
                x = int(vanish + (x_bottom - vanish) * t)
                if 0 <= x < width:
                    frame[row, x] = frame[row, x] * 0.60 + color * 0.40

    def _build_static(self) -> np.ndarray:
        """Background and chrome — everything that never changes."""
        layout = self.layout
        frame = np.zeros((layout.height, layout.width, 3), dtype=np.float32)
        self._background(frame)

        board = layout.board
        # The well is laid over the background, so background art is physically
        # incapable of reaching a playfield cell. A little bleed keeps it from
        # looking like a hole punched in the artwork.
        behind = frame[board.y : board.bottom, board.x : board.right].copy()
        well = np.array(WELL, dtype=np.float32)
        frame[board.y : board.bottom, board.x : board.right] = (
            well * (1.0 - WELL_BLEED) + behind * WELL_BLEED
        )

        spawn = layout.spawn_zone
        fill_rect(frame, spawn, np.array(SPAWN_ZONE, dtype=np.float32) * 0.9)

        self._draw_well_grid(frame)
        stroke_rect(frame, board, CYAN, max(1, layout.cell_px // 10))

        if self.include_hud:
            self._draw_chrome(frame)
        return frame

    def _draw_well_grid(self, frame: np.ndarray) -> None:
        board = self.layout.board
        cell = self.layout.cell_px
        grid = np.array(GRID, dtype=np.float32)
        for col in range(1, BOARD_WIDTH):
            x = board.x + col * cell
            frame[board.y : board.bottom, x] = grid
        for row in range(1, RENDER_ROWS):
            y = board.y + row * cell
            frame[y, board.x : board.right] = grid

        # A brighter line marking where the hidden spawn zone ends and the
        # playfield the player is judged on begins.
        y = board.y + SPAWN_ROWS_SHOWN * cell
        frame[y, board.x : board.right] = np.array(MAGENTA, dtype=np.float32) * 0.55

    def _draw_chrome(self, frame: np.ndarray) -> None:
        layout = self.layout
        cell = layout.cell_px
        panel = np.array(PANEL, dtype=np.float32)
        stroke_w = max(1, cell // 12)
        label_scale = max(1, cell // 7)

        for rect, label, color in (
            (layout.hold_panel, "HOLD", MAGENTA),
            (layout.next_panel, "NEXT", CYAN),
            (layout.stats_panel, "", PURPLE),
        ):
            fill_rect(frame, rect, panel)
            stroke_rect(frame, rect, color, stroke_w)
            if label:
                draw_text_centered(
                    frame, label, rect.x + rect.w // 2, rect.y + cell // 3, color, label_scale
                )

        draw_text_centered(
            frame,
            "TETRIS",
            layout.width // 2,
            max(2, cell // 2),
            TEXT_HOT,
            max(1, cell // 5),
        )

    # -- the frame --------------------------------------------------------

    def render(
        self,
        engine: TetrisEngine,
        *,
        randomization: randomize.Randomization = randomize.NONE,
        shake: tuple[int, int] = (0, 0),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw ``engine`` and return an ``(H, W, 3)`` uint8 frame."""
        frame = self._static.copy()

        self._draw_cells(frame, engine)
        if self.include_hud:
            self._draw_hud(frame, engine)

        frame = self._post_process(frame, randomization, shake, rng)
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        if not self.include_hud:
            board = self.layout.board
            frame = frame[board.y : board.bottom, board.x : board.right]
        return np.ascontiguousarray(frame)

    def _draw_cells(self, frame: np.ndarray, engine: TetrisEngine) -> None:
        """Ghost, locked stack and active piece — the information layer."""
        layout = self.layout
        cell = layout.cell_px
        board = layout.board

        grid = engine.board.colors[RENDER_TOP:TOTAL_HEIGHT].copy()

        # Ghost first, so the real piece paints over it where they overlap.
        if engine.piece is not None:
            ghost = np.zeros_like(grid)
            for x, y in engine.ghost():
                row = y - RENDER_TOP
                if 0 <= row < RENDER_ROWS and not grid[row, x]:
                    ghost[row, x] = int(engine.piece.type)
            if ghost.any():
                self._blit_cells(frame, board, ghost, cell, alpha=GHOST_ALPHA, bevel=False)

            for x, y in engine.active_cells():
                row = y - RENDER_TOP
                if 0 <= row < RENDER_ROWS:
                    grid[row, x] = int(engine.piece.type)

        self._blit_cells(frame, board, grid, cell)

    def _blit_cells(
        self,
        frame: np.ndarray,
        rect: Rect,
        grid: np.ndarray,
        cell: int,
        *,
        alpha: float = 1.0,
        bevel: bool = True,
    ) -> None:
        """Scale a cell grid up to pixels in one vectorised step.

        ``np.repeat`` twice beats a per-cell loop by a wide margin, which is
        what keeps the agent's frame rate up.
        """
        mask = grid != 0
        if not mask.any():
            return

        colors = self._lut[grid]
        colors = np.repeat(np.repeat(colors, cell, axis=0), cell, axis=1)
        if bevel:
            colors = colors * np.tile(self._bevel, (grid.shape[0], grid.shape[1], 1))
        if alpha != 1.0:
            colors = colors * alpha

        big_mask = np.repeat(np.repeat(mask, cell, axis=0), cell, axis=1)[:, :, None]
        region = frame[rect.y : rect.bottom, rect.x : rect.right]
        np.copyto(region, colors, where=big_mask)

    def _draw_hud(self, frame: np.ndarray, engine: TetrisEngine) -> None:
        layout = self.layout
        cell = layout.cell_px
        scale = max(1, cell // 7)
        preview_cell = max(2, cell * 2 // 3)

        if engine.hold is not None:
            self._draw_mini_piece(
                frame, engine.hold, layout.hold_panel, preview_cell,
                dim=0.45 if engine.hold_used else 1.0,
            )

        panel = layout.next_panel
        for index, piece in enumerate(engine.preview(4)):
            slot = Rect(panel.x, panel.y + (index + 1) * 3 * cell, panel.w, 3 * cell)
            self._draw_mini_piece(frame, piece, slot, preview_cell, dim=1.0 - index * 0.15)

        stats = layout.stats_panel
        line_h = 9 * scale
        inset = cell // 3
        # Values are shrunk to fit rather than allowed to run past the panel —
        # a seven-figure score with separators is wider than the panel at the
        # label's own scale.
        value_room = stats.w - 2 * inset
        y = stats.y + inset
        for label, value, color in (
            ("SCORE", f"{engine.stats.score:,}", TEXT),
            ("LEVEL", f"{engine.stats.level}", TEXT_HOT),
            ("LINES", f"{engine.stats.lines:,}", TEXT),
        ):
            draw_text(frame, label, stats.x + inset, y, TEXT_DIM, scale)
            draw_text(
                frame, value, stats.x + inset, y + line_h, color,
                fit_scale(value, value_room, scale),
            )
            y += line_h * 2 + scale * 2

        if engine.stats.combo > 0:
            draw_text_centered(
                frame, f"{engine.stats.combo}x COMBO", layout.width // 2,
                layout.height - 2 * cell, TEXT_HOT, scale,
            )
    def _draw_mini_piece(
        self, frame: np.ndarray, piece: PieceType, rect: Rect, cell: int, dim: float = 1.0
    ) -> None:
        """Draw a piece preview centred in ``rect``."""
        cells = SHAPES[piece][0]
        xs = [bx for bx, _ in cells]
        ys = [by for _, by in cells]
        span_x = max(xs) - min(xs) + 1
        span_y = max(ys) - min(ys) + 1

        origin_x = rect.x + (rect.w - span_x * cell) // 2 - min(xs) * cell
        origin_y = rect.y + (rect.h - span_y * cell) // 2 - min(ys) * cell
        color = self._lut[int(piece)] * dim

        for bx, by in cells:
            x0 = origin_x + bx * cell
            y0 = origin_y + by * cell
            if x0 < 0 or y0 < 0 or x0 + cell > frame.shape[1] or y0 + cell > frame.shape[0]:
                continue
            frame[y0 : y0 + cell, x0 : x0 + cell] = color
            frame[y0, x0 : x0 + cell] = color * 0.35
            frame[y0 : y0 + cell, x0] = color * 0.35

    def _post_process(
        self,
        frame: np.ndarray,
        rand: randomize.Randomization,
        shake: tuple[int, int],
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        profile = self.profile

        dx = shake[0] + rand.offset_x
        dy = shake[1] + rand.offset_y
        frame = effects.shake_offset(frame, dx, dy)

        if profile.bloom > 0.0:
            radius = max(1, self.layout.cell_px // 3)
            frame = effects.bloom(frame, profile.bloom * rand.bloom_scale, radius)
        if profile.chromatic > 0.0:
            frame = effects.chromatic(frame, profile.chromatic)
        if profile.scanlines > 0.0:
            frame = effects.scanlines(
                frame, profile.scanlines * rand.scanline_scale, rand.scanline_phase
            )
        if profile.vignette > 0.0:
            frame = effects.vignette(frame, profile.vignette)

        frame = effects.grade(frame, rand.brightness, rand.contrast)
        if rand.hue_degrees:
            frame = effects.hue_rotate(frame, rand.hue_degrees)
        if rand.noise > 0.0 and rng is not None:
            frame = effects.add_noise(frame, rand.noise, rng)
        return frame
