"""Overlay panel layout.

These exist because of a real bug: the game-over score was drawn on top of the
`GAME OVER` text, because the compositor centred a banner on the board while the
app centred the score on the frame and neither knew about the other.

The fix was to lay every line out from a running cursor over measured heights.
These tests assert the property that fix guarantees — **no two lines ever
overlap** — across cell sizes and score magnitudes, so it cannot regress.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from tetris.app.main import Game, Scene  # noqa: E402
from tetris.render.font import fit_scale, text_size  # noqa: E402

CELL_SIZES = [12, 20, 30, 40]


@pytest.fixture
def game():
    instance = Game(seed=1, cell_px=30)
    yield instance
    pygame.display.quit()


def line_rects(game: Game, lines) -> list[tuple[int, int, int, int]]:
    """Each line's (x, y, w, h), from the drawer's own layout pass.

    Calls `_layout_panel` rather than reimplementing it, so these assertions
    can never drift from what actually gets drawn.
    """
    _plate, placed = game._layout_panel(lines)
    return [(r.x, r.y, r.w, r.h) for _line, _scale, r in placed]


def overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


# -- the regression -------------------------------------------------------


@pytest.mark.parametrize("cell", CELL_SIZES)
@pytest.mark.parametrize("score", [0, 999, 128_450, 999_999_999])
def test_game_over_lines_never_overlap(cell, score):
    game = Game(seed=1, cell_px=cell)
    try:
        game.scene = Scene.GAME_OVER
        game.engine.stats.score = score
        game.engine.stats.lines = 402
        game.engine.stats.level = 20
        game.scores.best = 204_900

        rects = line_rects(game, game._panel_lines())
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                assert not overlaps(a, b), f"lines overlap at cell={cell}: {a} vs {b}"
    finally:
        pygame.display.quit()


@pytest.mark.parametrize("cell", CELL_SIZES)
@pytest.mark.parametrize(
    "scene", [Scene.TITLE, Scene.PAUSED, Scene.GAME_OVER]
)
def test_every_panel_stays_inside_its_plate(cell, scene):
    game = Game(seed=1, cell_px=cell)
    try:
        game.scene = scene
        game.engine.stats.score = 128_450
        game.scores.best = 204_900

        lines = game._panel_lines()
        plate, _ = game._layout_panel(lines)

        # The plate deliberately overhangs the board a little, but must stay
        # well inside the scene and never reach the window edge.
        assert plate.x >= game.layout.cell_px - 1
        assert plate.right <= game.layout.width - game.layout.cell_px + 1
        assert plate.y >= 0 and plate.bottom <= game.layout.height

        for x, y, w, h in line_rects(game, lines):
            assert y >= plate.y and y + h <= plate.bottom, "line escaped the plate"
            assert x >= plate.x - 1 and x + w <= plate.right + 1, "line escaped sideways"
    finally:
        pygame.display.quit()


def test_playing_scene_draws_no_panel(game):
    game.scene = Scene.PLAYING
    assert game._panel_lines() == []


# -- content --------------------------------------------------------------


def test_score_is_the_largest_line_on_the_game_over_panel(game):
    """The whole point of the fix: the score must dominate."""
    game.scene = Scene.GAME_OVER
    game.engine.stats.score = 128_450
    lines = game._panel_lines()

    heights = {
        line.text: text_size(line.text, game._scale_for(line.size))[1] for line in lines
    }
    score_height = heights["128,450"]
    for text, height in heights.items():
        if text != "128,450":
            assert height < score_height, f"{text!r} is not smaller than the score"


def test_new_record_replaces_best(game):
    game.scene = Scene.GAME_OVER
    game.scores.best = 204_900

    game._new_record = True
    texts = [line.text for line in game._panel_lines()]
    assert "NEW RECORD" in texts
    assert not any(t.startswith("BEST") for t in texts)

    game._new_record = False
    texts = [line.text for line in game._panel_lines()]
    assert "NEW RECORD" not in texts
    assert "BEST 204,900" in texts


def test_scores_use_thousands_separators(game):
    game.scene = Scene.GAME_OVER
    game.engine.stats.score = 1_234_567
    assert any(line.text == "1,234,567" for line in game._panel_lines())


def test_panel_renders_without_error_at_every_cell_size():
    # Exercises the real drawing path, not just the measuring maths.
    for cell in CELL_SIZES:
        game = Game(seed=1, cell_px=cell)
        try:
            for scene in (Scene.TITLE, Scene.PAUSED, Scene.GAME_OVER):
                game.scene = scene
                frame = game.compositor.render(game.engine)
                game._draw_panel(frame, game._panel_lines())
                assert frame.shape == (game.layout.height, game.layout.width, 3)
        finally:
            pygame.display.quit()
