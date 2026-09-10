"""Domain randomization — the lever that buys distortion robustness.

The agent trains on the same renderer it plays in, so the way to make it robust
to natural variation is to *vary* that renderer rather than to strip it. Every
parameter here perturbs the finished frame; none of them can move, hide or
recolour a cell badly enough to break the legibility invariant, which
``test_legibility`` checks at full strength.

Strength is a single 0-1 dial, ramped on a curriculum: near zero while the
agent is still learning to clear lines, high once it can. Ordering the two in
time is what delivers fast learning *and* robustness, instead of trading one
against the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bounds at strength 1.0. Chosen so the picture still looks like the game.
MAX_BRIGHTNESS_JITTER = 0.22
MAX_CONTRAST_JITTER = 0.18
MAX_HUE_DEGREES = 12.0
MAX_OFFSET_PX = 2
MAX_NOISE_SIGMA = 6.0
MAX_SCANLINE_JITTER = 0.5
MAX_BLOOM_JITTER = 0.5


@dataclass(frozen=True, slots=True)
class Randomization:
    """One episode's worth of visual perturbation."""

    brightness: float = 1.0
    contrast: float = 1.0
    hue_degrees: float = 0.0
    offset_x: int = 0
    offset_y: int = 0
    noise: float = 0.0
    scanline_scale: float = 1.0
    scanline_phase: int = 0
    bloom_scale: float = 1.0

    @property
    def is_identity(self) -> bool:
        return (
            self.brightness == 1.0
            and self.contrast == 1.0
            and self.hue_degrees == 0.0
            and self.offset_x == 0
            and self.offset_y == 0
            and self.noise == 0.0
            and self.scanline_scale == 1.0
            and self.scanline_phase == 0
            and self.bloom_scale == 1.0
        )


NONE = Randomization()


def sample(strength: float, rng: np.random.Generator) -> Randomization:
    """Draw a perturbation at ``strength`` in 0-1.

    Drawn from the episode's own generator, so a seeded run stays reproducible
    even with randomization switched on.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return NONE

    def jitter(scale: float) -> float:
        return float(rng.uniform(-scale, scale) * strength)

    offset_range = int(round(MAX_OFFSET_PX * strength))
    return Randomization(
        brightness=1.0 + jitter(MAX_BRIGHTNESS_JITTER),
        contrast=1.0 + jitter(MAX_CONTRAST_JITTER),
        hue_degrees=jitter(MAX_HUE_DEGREES),
        offset_x=int(rng.integers(-offset_range, offset_range + 1)) if offset_range else 0,
        offset_y=int(rng.integers(-offset_range, offset_range + 1)) if offset_range else 0,
        noise=float(rng.uniform(0.0, MAX_NOISE_SIGMA) * strength),
        scanline_scale=1.0 + jitter(MAX_SCANLINE_JITTER),
        scanline_phase=int(rng.integers(0, 3)),
        bloom_scale=1.0 + jitter(MAX_BLOOM_JITTER),
    )
