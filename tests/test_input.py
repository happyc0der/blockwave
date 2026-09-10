"""DAS and ARR — the timings that decide whether the game feels crisp or muddy."""

from __future__ import annotations

import os
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from blockwave.app.input import InputConfig, InputState  # noqa: E402
from blockwave.core.constants import Action  # noqa: E402

NO_KEYS = defaultdict(bool)


def make_input(das: float = 133.0, arr: float = 10.0) -> InputState:
    return InputState(InputConfig(das_ms=das, arr_ms=arr))


def test_keypress_moves_immediately():
    state = make_input()
    state.key_down(pygame.K_LEFT)
    assert state.poll(0.0, NO_KEYS) == [Action.LEFT]


def test_no_repeat_before_das_elapses():
    state = make_input(das=133.0)
    state.key_down(pygame.K_LEFT)
    state.poll(0.0, NO_KEYS)  # consume the initial tap

    assert state.poll(0.100, NO_KEYS) == []  # 100 ms < 133 ms


def test_repeat_starts_once_das_elapses():
    state = make_input(das=133.0, arr=10.0)
    state.key_down(pygame.K_LEFT)
    state.poll(0.0, NO_KEYS)

    state.poll(0.100, NO_KEYS)
    assert state.poll(0.050, NO_KEYS) == [Action.LEFT]  # 150 ms total


def test_das_overshoot_carries_into_the_repeat_clock():
    """DAS accuracy must not depend on where frame boundaries happen to fall."""
    state = make_input(das=100.0, arr=10.0)
    state.key_down(pygame.K_RIGHT)
    state.poll(0.0, NO_KEYS)

    # One big tick lands 45 ms past the DAS threshold, which is four whole ARR
    # intervals — those repeats are owed and must not be swallowed.
    repeats = state.poll(0.145, NO_KEYS)
    assert repeats == [Action.RIGHT] * 4


def test_arr_repeats_at_the_configured_rate():
    state = make_input(das=100.0, arr=20.0)
    state.key_down(pygame.K_RIGHT)
    state.poll(0.0, NO_KEYS)
    state.poll(0.100, NO_KEYS)  # charge exactly

    assert state.poll(0.060, NO_KEYS) == [Action.RIGHT] * 3


def test_releasing_the_key_stops_the_repeat():
    state = make_input(das=100.0)
    state.key_down(pygame.K_LEFT)
    state.poll(0.0, NO_KEYS)
    state.poll(0.200, NO_KEYS)

    state.key_up(pygame.K_LEFT)
    assert state.poll(0.100, NO_KEYS) == []


def test_opposite_direction_takes_over_immediately():
    # Tapping the other way while holding one direction should reverse now, not
    # be ignored until the first key is released.
    state = make_input(das=100.0)
    state.key_down(pygame.K_LEFT)
    state.poll(0.0, NO_KEYS)
    state.poll(0.200, NO_KEYS)  # left is charged

    state.key_down(pygame.K_RIGHT)
    assert state.poll(0.0, NO_KEYS) == [Action.RIGHT]
    # ...and the new direction starts its own DAS from scratch.
    assert state.poll(0.050, NO_KEYS) == []


def test_das_charge_survives_a_piece_lock():
    """The behaviour `on_piece_locked` exists to name.

    It is deliberately a no-op — the charge is preserved by *not* being reset.
    Without this, holding a direction stalls for a fresh DAS interval on every
    new piece, which is the most common way a falling-block game feels sluggish.
    """
    state = make_input(das=133.0, arr=10.0)
    state.key_down(pygame.K_LEFT)
    state.poll(0.0, NO_KEYS)
    state.poll(0.140, NO_KEYS)  # fully charged

    state.on_piece_locked()

    # The next piece should shift on the very next tick, with no fresh delay.
    assert state.poll(0.010, NO_KEYS) == [Action.LEFT]


def test_soft_drop_repeats_while_held():
    state = make_input()
    keys = defaultdict(bool)
    keys[pygame.K_DOWN] = True

    state.key_down(pygame.K_DOWN)
    actions = state.poll(0.100, keys)  # 100 ms at 25 ms per repeat
    assert actions.count(Action.SOFT_DROP) >= 4


def test_soft_drop_stops_when_released():
    state = make_input()
    keys = defaultdict(bool)
    keys[pygame.K_DOWN] = True
    state.poll(0.100, keys)

    keys[pygame.K_DOWN] = False
    assert state.poll(0.100, keys) == []


def test_one_shot_actions_fire_once_each():
    state = make_input()
    for key, expected in (
        (pygame.K_SPACE, Action.HARD_DROP),
        (pygame.K_x, Action.ROTATE_CW),
        (pygame.K_z, Action.ROTATE_CCW),
        (pygame.K_c, Action.HOLD),
    ):
        state.key_down(key)
        assert state.poll(0.0, NO_KEYS) == [expected]
        assert state.poll(0.0, NO_KEYS) == []
