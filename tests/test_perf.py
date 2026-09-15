"""Throughput floors for requirement 1.

Performance rots quietly: nothing fails when someone adds a per-cell loop to the
renderer, the game just gets sluggish. These assert order-of-magnitude floors,
not targets — they are set far below measured performance so they catch a real
regression rather than machine noise or a busy CI box.

Measured on the development machine at the time of writing:
engine 340k steps/sec (floor 50k), arcade render 145 fps (floor 120).
"""

from __future__ import annotations

import pytest

from blockwave.bench import (
    ENGINE_FLOOR_SPS,
    RENDER_FLOOR_FPS,
    engine_steps_per_second,
    render_frames_per_second,
)
from blockwave.render.layout import HUMAN_CELL_PX


def test_engine_steps_fast_enough():
    rate = engine_steps_per_second(steps=20_000)
    assert rate >= ENGINE_FLOOR_SPS, f"engine at {rate:,.0f} steps/sec"


@pytest.mark.parametrize("profile", ["flat", "arcade", "arcade_max"])
def test_render_fast_enough(profile):
    """Every profile must clear a 120 Hz budget at human cell size.

    The floor is 120 rather than 60 deliberately: a profile that only just
    clears 60 drops frames on a high-refresh display, which is how the default
    `arcade` profile was found to be spending 71% of its frame in a full
    resolution bloom blur.

    Best of three, because this is a throughput measurement on a machine that
    may be doing other things. Contention can only ever make a render look
    slower, never faster, so the maximum is the better estimate of what the
    renderer can do and the mean is biased by whatever else is running.

    Measured best-of-three on the development machine, otherwise idle:

        flat        252.4 fps   2.10x the floor
        arcade      134.5 fps   1.12x
        arcade_max  119.7 fps   1.00x   <-- does not clear it

    `arcade_max` does not meet this budget. It is not a regression and not
    machine noise: the heaviest profile sits exactly on the line, so the test
    passes or fails on a coin-flip. The docstring above this one claims the
    floors sit "far below measured performance"; that is true of `flat` and
    roughly true of `arcade`, and false here. Either `arcade_max` needs
    optimising or it needs its own, honestly lower, floor -- and until that is
    decided this test tells the truth by failing.
    """
    fps = max(render_frames_per_second(profile, HUMAN_CELL_PX, frames=60) for _ in range(3))
    assert fps >= RENDER_FLOOR_FPS, f"{profile} at {fps:,.0f} fps (best of 3)"


def test_logic_tick_budget_is_comfortable():
    """240 Hz logic must cost a small fraction of a 60 fps frame.

    Each rendered frame advances four logic ticks; if those cost anything like a
    frame's worth of time the fixed-timestep loop spirals.
    """
    rate = engine_steps_per_second(steps=20_000)
    ticks_per_frame = 240 / 60
    budget_ms = ticks_per_frame / rate * 1000
    assert budget_ms < 1.0, f"logic costs {budget_ms:.3f} ms per rendered frame"
