"""Visual profiles: one renderer, a graded effects budget.

There is one renderer. A profile does not change *which* code runs, only how
much decoration it is allowed to spend — and, crucially, decoration is confined
to layers that cannot occlude an occupied playfield cell.

That invariant is what keeps the arcade loud and the information clean, and it
is enforced by ``tests/test_legibility.py`` rather than by good intentions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisualProfile:
    name: str

    #: Animated sun and perspective grid behind the board. Background layer, so
    #: it is painted over by every locked cell.
    background: bool = True

    #: Post-process passes, all applied to the finished frame.
    scanlines: float = 0.0
    bloom: float = 0.0
    vignette: float = 0.0
    chromatic: float = 0.0

    #: Whole-frame translation on impact, as a multiplier on the offset the app
    #: hands in. Safe for legibility because it moves the board rather than
    #: covering it.
    shake: float = 0.0

    #: Particle bursts on line clears. Drawn only outside the playfield, so
    #: they cannot occlude a cell.
    particles: bool = False


#: Neon blocks on a dark ground, nothing else. The cheapest profile to render,
#: and completely still.
FLAT = VisualProfile(name="flat", background=False)

#: The default for human play *and* for training. Full Miami identity with the
#: motion effects held back, which is the "not too much visual clutter" middle
#: ground the whole design turns on.
ARCADE = VisualProfile(
    name="arcade",
    background=True,
    scanlines=0.28,
    bloom=0.45,
    vignette=0.35,
    # Full strength rather than something gentler: this is exactly the shake
    # the game has always had, since the field used to be ignored entirely.
    shake=1.0,
)

#: Everything at once. The showpiece.
ARCADE_MAX = VisualProfile(
    name="arcade_max",
    background=True,
    scanlines=0.38,
    bloom=0.75,
    vignette=0.5,
    chromatic=1.0,
    shake=1.5,
    particles=True,
)

PROFILES: dict[str, VisualProfile] = {
    profile.name: profile for profile in (FLAT, ARCADE, ARCADE_MAX)
}


def get_profile(profile: str | VisualProfile) -> VisualProfile:
    if isinstance(profile, VisualProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown visual profile {profile!r}; expected one of {sorted(PROFILES)}"
        ) from None
