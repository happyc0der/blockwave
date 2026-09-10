"""Throughput measurement for the engine and the renderer.

Requirement 1 is a high *logic* tick rate, and the way that requirement rots is
quietly: someone adds a per-cell loop to the renderer or an allocation to
``step`` and nothing fails, the game just gets sluggish under load. This module
gives a number to check, and ``tests/test_perf.py`` asserts floors against it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .core.constants import Action
from .core.engine import EngineConfig, TetrisEngine

#: Floors, not targets. Set roughly an order of magnitude below what the code
#: actually does, so these catch a real regression rather than machine noise.
ENGINE_FLOOR_SPS = 50_000
RENDER_FLOOR_FPS = 120


@dataclass(slots=True)
class Result:
    name: str
    per_second: float
    floor: float

    @property
    def ok(self) -> bool:
        return self.per_second >= self.floor

    def line(self) -> str:
        mark = "ok  " if self.ok else "SLOW"
        return f"  [{mark}] {self.name:<34} {self.per_second:>12,.0f}/s   floor {self.floor:>9,.0f}/s"


def engine_steps_per_second(steps: int = 200_000, seed: int = 0) -> float:
    """Time ``step`` under random play, resetting whenever a game ends."""
    rng = random.Random(seed)
    engine = TetrisEngine(EngineConfig(seed=seed))
    actions = [rng.choice(list(Action)) for _ in range(steps)]

    start = time.perf_counter()
    for index, action in enumerate(actions):
        engine.step(action, 1.0 / 60.0)
        if engine.game_over:
            engine.reset(seed=index)
    elapsed = time.perf_counter() - start
    return steps / elapsed if elapsed else float("inf")


def render_frames_per_second(profile: str, cell_px: int, frames: int = 300, seed: int = 0) -> float:
    """Time full-frame rendering, with a realistic stack on the board.

    Imported lazily so the engine benchmark stays usable in environments
    without a working numpy/render stack.
    """
    from .render.compositor import Compositor
    from .render.layout import Layout

    engine = TetrisEngine(EngineConfig(seed=seed))
    rng = random.Random(seed)
    for _ in range(30):
        target = rng.randrange(10)
        for _ in range(10):
            if engine.piece is None:
                break
            current = min(x for x, _ in engine.piece.cells())
            if current == target:
                break
            engine.step(Action.LEFT if current > target else Action.RIGHT, 0.0)
        engine.step(Action.HARD_DROP, 0.0)
        if engine.game_over:
            engine.reset(seed=seed)

    compositor = Compositor(Layout(cell_px), profile)
    compositor.render(engine)  # warm the cached static layer

    start = time.perf_counter()
    for _ in range(frames):
        compositor.render(engine)
    elapsed = time.perf_counter() - start
    return frames / elapsed if elapsed else float("inf")


def run_benchmarks(steps: int = 200_000) -> list[Result]:
    """Measure everything and print a report. Returns the results."""
    from .render.layout import HUMAN_CELL_PX

    results = [
        Result("engine step (random play)", engine_steps_per_second(steps), ENGINE_FLOOR_SPS),
    ]
    for profile in ("flat", "arcade", "arcade_max"):
        results.append(
            Result(
                f"render {profile} @ {HUMAN_CELL_PX}px cells",
                render_frames_per_second(profile, HUMAN_CELL_PX),
                RENDER_FLOOR_FPS,
            )
        )

    print("\ntetris benchmarks")
    print("-" * 78)
    for result in results:
        print(result.line())
    print("-" * 78)

    slow = [r.name for r in results if not r.ok]
    if slow:
        print(f"below floor: {', '.join(slow)}")
    else:
        print("all above floor")
    print()
    return results
