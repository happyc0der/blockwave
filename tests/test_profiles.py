"""Visual profiles, and the rule that no profile field may be dead config.

This file exists because the same bug happened three times: a field is added to
:class:`VisualProfile`, documented, given per-profile values — and the code that
should read it never lands. Nothing fails, because a setting that does nothing
looks exactly like a setting that is switched off.

The general test below closes that off. For every field the compositor is meant
to consume, it renders one frame with the field at its default and one with it
turned up, everything else held constant, and asserts **the frames differ**. A
field that changes no pixel is dead by definition.

It is parametrised over an explicit list, so a newly added field that nobody
wired up fails here rather than shipping.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from blockwave.core.constants import PieceType
from blockwave.core.engine import EngineConfig, Engine
from blockwave.core.piece import Piece
from blockwave.render.compositor import Compositor
from blockwave.render.layout import Layout
from blockwave.render.profiles import (
    ARCADE,
    ARCADE_MAX,
    FLAT,
    PROFILES,
    VisualProfile,
    get_profile,
)

#: Every field the compositor consumes, with a value that should visibly change
#: the frame. `particles` is absent on purpose — it is consumed by the app layer,
#: and is covered separately below.
COMPOSITOR_FIELDS = [
    ("background", True),
    ("scanlines", 0.5),
    ("bloom", 0.9),
    ("vignette", 0.6),
    ("chromatic", 2.0),
    ("shake", 1.0),
]


def populated_engine(seed: int = 4) -> Engine:
    """A board with a stack on it, so effects have something to act on."""
    engine = Engine(EngineConfig(seed=seed))
    for y in (20, 21, 22, 23):
        engine.board.rows[y] = 0b1011011011
        for x in range(10):
            engine.board.colors[y, x] = (x % 7) + 1 if (0b1011011011 >> x) & 1 else 0
    engine.piece = Piece(PieceType.T, x=3, y=10)
    return engine


def render(profile: VisualProfile, shake: tuple[int, int] = (0, 0)) -> np.ndarray:
    return Compositor(Layout(14), profile).render(populated_engine(), shake=shake)


# -- the anti-dead-config test --------------------------------------------


@pytest.mark.parametrize("field,value", COMPOSITOR_FIELDS, ids=[f for f, _ in COMPOSITOR_FIELDS])
def test_every_profile_field_changes_the_frame(field, value):
    """A profile field that changes nothing is not a setting, it is a lie."""
    off = VisualProfile(name="off", background=False)
    on = replace(off, **{field: value})

    # `shake` scales an offset the app hands in, so it needs a nonzero one to
    # have anything to scale.
    kick = (5, 3) if field == "shake" else (0, 0)

    assert not np.array_equal(render(off, kick), render(on, kick)), (
        f"VisualProfile.{field} is declared but changes no pixel — dead config"
    )


def test_the_field_list_covers_the_dataclass():
    """Guards the guard: a new field must be added to COMPOSITOR_FIELDS."""
    covered = {name for name, _ in COMPOSITOR_FIELDS} | {"name", "particles"}
    declared = set(VisualProfile.__dataclass_fields__)
    assert declared == covered, (
        f"VisualProfile fields not covered by a test: {declared - covered}"
    )


# -- shake, the specific bug ----------------------------------------------


def test_shake_is_honoured_per_profile():
    """`flat` declares no shake and must genuinely not move.

    Previously the compositor applied whatever offset it was handed and never
    consulted the profile, so every profile shook identically.
    """
    kick = (6, 4)
    assert np.array_equal(render(FLAT, kick), render(FLAT, (0, 0))), "flat should be still"
    assert not np.array_equal(render(ARCADE, kick), render(ARCADE, (0, 0)))
    assert not np.array_equal(render(ARCADE_MAX, kick), render(ARCADE_MAX, (0, 0)))


def test_arcade_max_shakes_harder_than_arcade():
    assert ARCADE_MAX.shake > ARCADE.shake > FLAT.shake


# -- particles, consumed by the app ---------------------------------------


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_particles_follow_the_profile_flag(name):
    """Sparks must be gated on `particles`, not on some other field.

    They used to be gated on `shake`, which happened to work only because
    `arcade_max` was the one profile with a nonzero value.
    """
    import pygame

    from blockwave.app.main import Game

    game = Game(seed=1, cell_px=14, profile=name, audio=False)
    try:
        game.particles.clear()
        game._spark((22, 23))
        expected = get_profile(name).particles
        assert bool(game.particles.sparks) is expected, (
            f"{name}: particles={expected} but sparks={len(game.particles.sparks)}"
        )
    finally:
        pygame.display.quit()


def test_sparks_never_land_inside_the_playfield():
    """The invariant that lets particles exist at all."""
    import pygame

    from blockwave.app.main import Game

    game = Game(seed=1, cell_px=20, profile="arcade_max", audio=False)
    try:
        board = game.layout.board
        game._spark((20, 21, 22, 23))
        assert game.particles.sparks

        for _ in range(60):
            game.particles.update(1 / 60, game.layout.cell_px)
            frame = game.compositor.render(game.engine)
            before = frame[board.y : board.bottom, board.x : board.right].copy()
            game.particles.draw(frame, board, game.layout.cell_px)
            after = frame[board.y : board.bottom, board.x : board.right]
            assert np.array_equal(before, after), "a spark painted inside the playfield"
    finally:
        pygame.display.quit()


# -- the profile table ----------------------------------------------------


def test_profiles_are_ordered_by_intensity():
    for field in ("scanlines", "bloom", "vignette"):
        assert getattr(FLAT, field) <= getattr(ARCADE, field) <= getattr(ARCADE_MAX, field)


def test_flat_really_is_flat():
    assert FLAT.background is False
    assert FLAT.particles is False
    assert (FLAT.scanlines, FLAT.bloom, FLAT.vignette, FLAT.chromatic, FLAT.shake) == (
        0.0, 0.0, 0.0, 0.0, 0.0,
    )


def test_unknown_profile_name_is_rejected():
    with pytest.raises(ValueError, match="unknown visual profile"):
        get_profile("neon_disco")
