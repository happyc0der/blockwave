"""Throughput floors for requirement 1.

Performance rots quietly: nothing fails when someone adds a per-cell loop to the
renderer, the game just gets sluggish. These assert floors, not targets. The
engine floor is an order of magnitude below what is measured. The render floor
is not: it is a real 120 Hz budget, and the heaviest profile clears it by about
a quarter, not by an order of magnitude (see `test_render_fast_enough`).

Measured on the development machine, idle, best of three (2026-09-15):
engine 340k steps/sec (floor 50k); render flat 282 fps, arcade 162 fps,
arcade_max 148 fps (floor 120).
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

    Measured best-of-three on the development machine, otherwise idle
    (2026-09-15, after the vignette fix in `effects.py`):

        flat        281.9 fps   2.35x the floor
        arcade      161.6 fps   1.35x
        arcade_max  147.6 fps   1.23x

    Before that fix `arcade_max` measured 119.7 against this 120 and passed or
    failed on a coin-flip. The cause was not the profile's effects budget but a
    numpy slow path: the vignette mask was stored (H, W, 1) and broadcast over
    the colour axis, which cost 0.88 ms a frame against 0.16 ms for the same
    mask stored (H, W, 3). That one change is worth about 0.7 ms of an 8 ms
    frame. The fps figures above also carry machine-load variance -- `flat`,
    which has no vignette, moved too between measurements -- so treat the
    isolated 0.7 ms as the finding and the fps as its consequence on that day.

    A 1.23x margin is real but not generous. If this fails again on an idle
    machine, the next lever is `_blit_cells`, which upsamples the cell grid
    twice per call (colours and mask) where once would do.
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
