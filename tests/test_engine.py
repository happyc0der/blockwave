"""End-to-end engine behaviour.

These are the tests that discharge requirement 5. Each one targets a specific
way a falling-block implementation classically goes wrong: pieces that never lock,
hold used twice, a top-out that is missed, a ghost that lies about where the
piece will land.
"""

from __future__ import annotations

import random

import pytest

from blockwave.core.constants import (
    BOARD_WIDTH,
    MAX_LOCK_RESETS,
    TOTAL_HEIGHT,
    VISIBLE_TOP,
    Action,
    PieceType,
)
from blockwave.core.engine import EngineConfig, Engine
from blockwave.core.events import EventType
from blockwave.core.piece import Piece, is_grounded
from blockwave.core.rules import lock_delay_ms

from helpers import make_board


def make_engine(**kwargs) -> Engine:
    kwargs.setdefault("seed", 12345)
    return Engine(EngineConfig(**kwargs))


def event_types(events) -> list[EventType]:
    return [event.type for event in events]


# -- lifecycle ------------------------------------------------------------


def test_reset_produces_a_playable_piece():
    engine = make_engine()
    assert engine.piece is not None
    assert not engine.game_over
    assert engine.stats.score == 0
    engine.board.check_invariants()


def test_reset_is_repeatable_for_a_given_seed():
    a = make_engine(seed=7)
    b = make_engine(seed=7)
    assert a.piece is not None and b.piece is not None
    assert a.piece.type == b.piece.type
    assert a.preview() == b.preview()


def test_different_seeds_diverge():
    a = make_engine(seed=1)
    b = make_engine(seed=2)
    assert (a.piece.type, a.preview()) != (b.piece.type, b.preview())


def test_steps_after_game_over_are_inert():
    engine = make_engine()
    engine.game_over = True
    assert engine.step(Action.LEFT, 1.0) == []


# -- gravity --------------------------------------------------------------


def test_piece_falls_under_gravity():
    engine = make_engine()
    start_y = engine.piece.y
    engine.step(Action.NOOP, 1.0)  # exactly one row at level 1
    assert engine.piece.y == start_y + 1


def test_gravity_scale_speeds_the_fall_up():
    engine = make_engine(gravity_scale=4.0)
    start_y = engine.piece.y
    engine.step(Action.NOOP, 1.0)
    assert engine.piece.y == start_y + 4


def test_instant_gravity_drops_the_piece_on_spawn():
    # At high levels gravity is 20G: the piece is on the stack immediately.
    engine = make_engine(start_level=20)
    engine.step(Action.NOOP, 0.001)
    assert engine.piece is not None
    # Nothing below it, so it must be resting on the floor.
    lowest = max(y for _, y in engine.piece.cells())
    assert lowest == TOTAL_HEIGHT - 1


# -- lock delay -----------------------------------------------------------


def ground_an_o_piece(engine: Engine) -> None:
    """Park an O piece flat on the floor, ready to lock."""
    engine.piece = Piece(PieceType.O, x=3, y=22)
    engine._lock_timer = 0.0
    engine._lock_resets = 0


def test_grounded_piece_waits_out_the_lock_delay():
    engine = make_engine()
    ground_an_o_piece(engine)
    delay = lock_delay_ms(engine.stats.level)

    # Four ticks of 100 ms is 400 ms — not yet enough.
    for _ in range(4):
        engine.step(Action.NOOP, 0.1)
    assert engine.stats.pieces_placed == 0

    events = engine.step(Action.NOOP, 0.1)  # crosses the 500 ms threshold
    assert engine.stats.pieces_placed == 1
    assert EventType.PIECE_LOCK in event_types(events)
    assert delay == pytest.approx(500.0)


def test_moving_a_grounded_piece_resets_the_lock_delay():
    engine = make_engine()
    ground_an_o_piece(engine)

    for _ in range(4):
        engine.step(Action.NOOP, 0.1)
    # 400 ms accumulated; a successful slide should put it back to zero.
    engine.step(Action.LEFT, 0.0)
    assert engine._lock_timer == 0.0
    assert engine.stats.pieces_placed == 0


def test_lock_resets_are_capped_so_a_piece_cannot_float_forever():
    # Without MAX_LOCK_RESETS a player (or an agent) can shuffle a piece left
    # and right on top of the stack indefinitely and the game never advances.
    engine = make_engine()
    ground_an_o_piece(engine)

    steps = 0
    while engine.stats.pieces_placed == 0 and steps < 200:
        engine.step(Action.LEFT if steps % 2 else Action.RIGHT, 0.1)
        steps += 1

    assert engine.stats.pieces_placed == 1, "piece never locked despite the cap"
    assert steps <= MAX_LOCK_RESETS + 10


def test_lock_timer_clears_while_a_piece_is_airborne():
    engine = make_engine()
    # A ledge covering the left half of the floor only.
    engine.board = make_board("#####.....")
    engine.piece = Piece(PieceType.O, x=2, y=21)  # resting on the ledge
    engine.step(Action.NOOP, 0.2)
    assert engine._lock_timer == pytest.approx(200.0)

    # Slide off the end of the ledge and out over the gap.
    engine.step(Action.RIGHT, 0.0)
    engine.step(Action.RIGHT, 0.0)
    assert not is_grounded(engine.piece, engine.board)

    engine.step(Action.NOOP, 0.05)
    assert engine._lock_timer == 0.0


# -- hard drop ------------------------------------------------------------


def test_hard_drop_locks_immediately_and_scores_per_cell():
    engine = make_engine()
    piece = engine.piece
    distance = TOTAL_HEIGHT - 1 - max(y for _, y in piece.cells())

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.pieces_placed == 1
    assert engine.stats.score >= 2 * distance
    types = event_types(events)
    assert EventType.HARD_DROP in types
    assert EventType.PIECE_LOCK in types
    assert types.index(EventType.HARD_DROP) < types.index(EventType.PIECE_LOCK)


def test_ghost_matches_where_a_hard_drop_actually_lands():
    # A ghost that disagrees with the drop is worse than no ghost at all.
    rng = random.Random(0)
    for _ in range(50):
        engine = make_engine(seed=rng.randrange(10_000))
        for _ in range(rng.randrange(6)):
            engine.step(rng.choice([Action.LEFT, Action.RIGHT, Action.ROTATE_CW]), 0.0)

        predicted = set(engine.ghost())
        engine.step(Action.HARD_DROP, 0.0)
        landed = {
            (x, y)
            for y in range(TOTAL_HEIGHT)
            for x in range(BOARD_WIDTH)
            if engine.board.rows[y] & (1 << x)
        }
        assert predicted <= landed, "ghost pointed somewhere the piece did not go"


def test_soft_drop_scores_one_per_cell():
    engine = make_engine()
    before = engine.stats.score
    engine.step(Action.SOFT_DROP, 0.0)
    assert engine.stats.score == before + 1


# -- hold -----------------------------------------------------------------


def test_hold_stashes_the_current_piece():
    engine = make_engine()
    held = engine.piece.type
    upcoming = engine.preview()[0]

    engine.step(Action.HOLD, 0.0)

    assert engine.hold == held, "the held piece should be the one we stashed"
    assert engine.piece.type == upcoming, "an empty hold pulls from the queue"
    assert engine.hold_used


def test_hold_swaps_on_the_second_use():
    engine = make_engine()
    first = engine.piece.type
    engine.step(Action.HOLD, 0.0)  # stash `first`, take the next from the queue

    engine.step(Action.HARD_DROP, 0.0)  # lock it; hold becomes available again
    third = engine.piece.type

    engine.step(Action.HOLD, 0.0)

    assert engine.piece.type == first, "the stashed piece should come back out"
    assert engine.hold == third, "and the piece it displaced takes its place"


def test_hold_is_refused_twice_for_the_same_piece():
    engine = make_engine()
    engine.step(Action.HOLD, 0.0)
    after_first = engine.piece.type

    events = engine.step(Action.HOLD, 0.0)
    assert EventType.HOLD_DENIED in event_types(events)
    assert engine.piece.type == after_first


def test_hold_becomes_available_again_after_a_lock():
    engine = make_engine()
    engine.step(Action.HOLD, 0.0)
    assert engine.hold_used
    engine.step(Action.HARD_DROP, 0.0)
    assert not engine.hold_used


def test_hold_can_be_disabled():
    engine = make_engine(allow_hold=False)
    events = engine.step(Action.HOLD, 0.0)
    assert EventType.HOLD_DENIED in event_types(events)
    assert engine.hold is None


# -- line clears ----------------------------------------------------------


def test_line_clear_updates_stats_and_emits_an_event():
    engine = make_engine()
    # One row needing only columns 4 and 5, which is exactly an O piece.
    engine.board = make_board("####..####")
    engine.piece = Piece(PieceType.O, x=3, y=2)

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.lines == 1
    assert engine.stats.line_clears[1] == 1
    assert EventType.LINE_CLEAR in event_types(events)
    engine.board.check_invariants()


def test_clearing_ten_lines_levels_up():
    engine = make_engine()
    engine.stats.lines = 9
    engine.board = make_board("####..####")
    engine.piece = Piece(PieceType.O, x=3, y=2)

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.level == 2
    assert EventType.LEVEL_UP in event_types(events)


def test_perfect_clear_is_detected():
    # Two rows, each missing exactly the two columns an O piece fills, so the
    # drop sweeps the board completely empty rather than leaving a remainder.
    engine = make_engine()
    engine.board = make_board(
        "####..####",
        "####..####",
    )
    engine.piece = Piece(PieceType.O, x=3, y=2)

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.lines == 2
    assert not any(engine.board.rows), "the board should be swept clean"
    assert EventType.PERFECT_CLEAR in event_types(events)
    assert engine.stats.perfect_clears == 1


def test_tspin_double_scores_as_a_tspin():
    # The setup from test_rules, driven through the real engine: a T rotated
    # into a covered notch, clearing two rows.
    engine = make_engine()
    engine.board = make_board(
        ".....#....",
        "###...####",
        "####.#####",
    )
    engine.piece = Piece(PieceType.T, x=3, y=21, rotation=1)

    rotate = engine.step(Action.ROTATE_CW, 0.0)
    assert EventType.PIECE_ROTATE in event_types(rotate)
    assert engine.piece.rotation == 2

    events = engine.step(Action.HARD_DROP, 0.0)
    types = event_types(events)
    assert EventType.TSPIN in types
    assert EventType.LINE_CLEAR in types
    assert engine.stats.lines == 2
    assert engine.stats.tspins == 1
    # A T-spin double is worth 1200 at level 1; a plain double is only 300.
    assert engine.stats.score >= 1200


# -- losing ---------------------------------------------------------------


def test_lock_out_ends_the_game():
    # A stack that reaches the ceiling: the next piece has to come to rest
    # entirely inside the hidden buffer, which is a loss even though nothing
    # actually overlapped when it spawned.
    engine = make_engine()
    engine.board = make_board(*["#########." for _ in range(TOTAL_HEIGHT - VISIBLE_TOP)])
    engine.piece = Piece(PieceType.O, x=3, y=2)

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.game_over
    assert EventType.GAME_OVER in event_types(events)


def test_block_out_ends_the_game():
    # The spawn area is buried, so the next piece has nowhere to appear.
    board = make_board(*(["#########."] * 2 + [".........."] * 20))
    engine = make_engine()
    engine.board = board
    engine.piece = Piece(PieceType.O, x=7, y=20)  # parked clear of the spawn zone

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.game_over
    assert EventType.GAME_OVER in event_types(events)


# -- fuzz -----------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_play_never_violates_an_invariant(seed):
    """Ten thousand random actions with the board audited throughout.

    This is the net that catches whatever the targeted tests above did not
    think of: desynced representations, uncleaned full rows, a score that goes
    backwards, or a game that runs forever without ending.
    """
    rng = random.Random(seed)
    engine = make_engine(seed=seed)
    actions = list(Action)

    last_score = 0
    games = 0

    for _ in range(10_000):
        engine.step(rng.choice(actions), 1.0 / 60.0)
        engine.board.check_invariants()

        assert engine.stats.score >= last_score, "score went backwards"
        last_score = engine.stats.score

        if engine.piece is not None and not engine.game_over:
            assert not engine.board.collides(engine.piece.cells()), (
                "active piece is overlapping the stack"
            )

        if engine.game_over:
            games += 1
            last_score = 0
            engine.reset(seed=rng.randrange(10_000))

    assert games > 0, "random play should have lost at least once in 10k steps"


# -- fixed level ----------------------------------------------------------


def test_fixed_level_never_advances():
    engine = make_engine(start_level=3, fixed_level=True)
    engine.stats.lines = 9
    engine.board = make_board("####..####")
    engine.piece = Piece(PieceType.O, x=3, y=2)

    events = engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.lines == 10, "lines still count"
    assert engine.stats.level == 3, "but the level must not move"
    assert EventType.LEVEL_UP not in event_types(events)


def test_level_advances_by_default():
    engine = make_engine(start_level=3)
    engine.stats.lines = 9
    engine.board = make_board("####..####")
    engine.piece = Piece(PieceType.O, x=3, y=2)
    engine.step(Action.HARD_DROP, 0.0)
    assert engine.stats.level == 4


# -- the infinite-rotation stall ------------------------------------------


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_rotating_in_place_cannot_hold_a_piece_forever(piece_type):
    """Holding rotate used to freeze the game.

    A floor kick bumps a grounded piece into the air. Being airborne zeroed the
    lock timer, and a move that ended airborne did not count against the
    fifteen-move budget — so rotating in place reset the timer for free,
    forever. Found by the RL reward-hacking suite, where a rotate-spamming
    policy locked zero pieces in 8,000 ticks.
    """
    engine = make_engine(gravity_scale=4.0)
    engine.piece = Piece(piece_type, x=3, y=20)
    engine._lowest_row = max(y for _, y in engine.piece.cells())

    for _ in range(200):
        engine.step(Action.ROTATE_CW, 1 / 20)
        if engine.stats.pieces_placed:
            break
    assert engine.stats.pieces_placed == 1, f"{piece_type.name} never locked"


def test_alternating_rotation_directions_cannot_stall_either():
    engine = make_engine(gravity_scale=4.0)
    engine.piece = Piece(PieceType.T, x=3, y=21)
    engine._lowest_row = max(y for _, y in engine.piece.cells())
    for step in range(300):
        engine.step(Action.ROTATE_CW if step % 2 else Action.ROTATE_CCW, 1 / 20)
        if engine.stats.pieces_placed:
            break
    assert engine.stats.pieces_placed == 1
