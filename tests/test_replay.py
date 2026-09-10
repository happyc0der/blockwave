"""Replay recording and playback.

A replay that does not reproduce its session exactly is worse than no replay at
all — it would send someone chasing a bug that never happened. So the property
under test is always the same one: same seed, same actions, same fixed timestep
gives a byte-identical final state.
"""

from __future__ import annotations

import json
import random

import pytest

from tetris.app.replay import (
    FORMAT_VERSION,
    Recorder,
    Replay,
    apply_tick,
    playback,
    state_digest,
    verify,
)
from tetris.core.constants import Action
from tetris.core.engine import EngineConfig, TetrisEngine

DT = 1.0 / 240.0


def record_a_game(seed: int = 7, ticks: int = 6_000, density: float = 0.35) -> tuple[Replay, TetrisEngine]:
    """Play a scripted game through the same path the real loop uses."""
    engine = TetrisEngine(EngineConfig(seed=seed, start_level=1))
    recorder = Recorder(seed, 1, DT)
    rng = random.Random(seed)

    for _ in range(ticks):
        if engine.game_over:
            break
        actions = [rng.choice(list(Action))] if rng.random() < density else []
        recorder.tick(actions)
        apply_tick(engine, actions, DT)

    return recorder.finish(engine), engine


# -- the core property ----------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 777])
def test_a_replay_reproduces_its_session_exactly(seed):
    replay, live = record_a_game(seed=seed)
    matched, engine = verify(replay)

    assert matched, "replay diverged from the session it recorded"
    assert state_digest(engine) == state_digest(live)
    assert engine.stats.score == live.stats.score
    assert engine.stats.pieces_placed == live.stats.pieces_placed
    assert engine.board.rows == live.board.rows


def test_playback_is_repeatable():
    replay, _ = record_a_game(seed=11)
    first = state_digest(playback(replay))
    second = state_digest(playback(replay))
    assert first == second


def test_a_tampered_replay_is_detected():
    # The digest exists to catch exactly this: a file that no longer describes
    # the game it claims to.
    replay, _ = record_a_game(seed=5)
    assert verify(replay)[0]

    # Injected mid-game, where it still changes a placement. An action appended
    # after the top-out would legitimately do nothing.
    replay.actions.append((replay.ticks // 2, int(Action.HARD_DROP)))

    matched, engine = verify(replay)
    assert not matched
    assert state_digest(engine) != replay.digest


def test_a_replay_of_a_finished_game_ends_finished():
    replay, live = record_a_game(seed=3, ticks=200_000)
    assert live.game_over, "the scripted game should have topped out"
    _, engine = verify(replay)
    assert engine.game_over


# -- recording ------------------------------------------------------------


def test_idle_ticks_are_not_stored():
    """A 240 Hz loop is almost entirely idle; storing it all would be absurd."""
    recorder = Recorder(1, 1, DT)
    for _ in range(1_000):
        recorder.tick([])
    replay = recorder.replay

    assert replay.ticks == 1_000
    assert replay.actions == []


def test_noop_actions_are_not_stored():
    recorder = Recorder(1, 1, DT)
    recorder.tick([Action.NOOP])
    recorder.tick([Action.LEFT])
    assert recorder.replay.actions == [(1, int(Action.LEFT))]


def test_multiple_actions_in_one_tick_are_kept_in_order():
    recorder = Recorder(1, 1, DT)
    recorder.tick([Action.LEFT, Action.LEFT, Action.ROTATE_CW])
    assert recorder.replay.actions == [
        (0, int(Action.LEFT)),
        (0, int(Action.LEFT)),
        (0, int(Action.ROTATE_CW)),
    ]


def test_recording_stays_small():
    replay, _ = record_a_game(seed=9, ticks=20_000)
    # Sparse storage should keep this far below one entry per tick.
    assert len(replay.actions) < replay.ticks // 2


# -- the shared tick ------------------------------------------------------


def test_time_passes_once_per_tick_regardless_of_action_count():
    """A burst of auto-repeat must not also accelerate gravity."""
    one = TetrisEngine(EngineConfig(seed=4))
    many = TetrisEngine(EngineConfig(seed=4))

    apply_tick(one, [Action.NOOP], 1.0)
    apply_tick(many, [Action.ROTATE_CW, Action.ROTATE_CCW, Action.ROTATE_CW, Action.ROTATE_CCW], 1.0)

    assert one.piece is not None and many.piece is not None
    assert one.piece.y == many.piece.y, "extra actions bought extra gravity"


def test_apply_tick_returns_the_events_it_produced():
    engine = TetrisEngine(EngineConfig(seed=4))
    events = apply_tick(engine, [Action.HARD_DROP], DT)
    assert events, "the game loop needs these for audio and effects"


def test_apply_tick_stops_at_game_over():
    engine = TetrisEngine(EngineConfig(seed=4))
    engine.game_over = True
    assert apply_tick(engine, [Action.LEFT, Action.RIGHT], DT) == []


# -- the file format ------------------------------------------------------


def test_round_trip_through_a_file(tmp_path):
    replay, _ = record_a_game(seed=13)
    path = tmp_path / "run.json"
    replay.save(path)

    loaded = Replay.load(path)
    assert loaded == replay
    assert verify(loaded)[0]


def test_unknown_version_is_rejected():
    payload = json.dumps(
        {"version": 999, "seed": 1, "start_level": 1, "dt": DT, "ticks": 0, "actions": []}
    )
    with pytest.raises(ValueError, match="unsupported replay version"):
        Replay.from_json(payload)


def test_current_version_is_accepted():
    replay, _ = record_a_game(seed=1, ticks=100)
    assert json.loads(replay.to_json())["version"] == FORMAT_VERSION
    assert Replay.from_json(replay.to_json()) == replay
