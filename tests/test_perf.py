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
    """
    fps = render_frames_per_second(profile, HUMAN_CELL_PX, frames=60)
    assert fps >= RENDER_FLOOR_FPS, f"{profile} at {fps:,.0f} fps"


def test_logic_tick_budget_is_comfortable():
    """240 Hz logic must cost a small fraction of a 60 fps frame.

    Each rendered frame advances four logic ticks; if those cost anything like a
    frame's worth of time the fixed-timestep loop spirals.
    """
    rate = engine_steps_per_second(steps=20_000)
    ticks_per_frame = 240 / 60
    budget_ms = ticks_per_frame / rate * 1000
    assert budget_ms < 1.0, f"logic costs {budget_ms:.3f} ms per rendered frame"
