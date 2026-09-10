"""The pause after a line clear.

A clear is not instantaneous: the rows blank, the board holds for a beat, then
the stack collapses and the next piece arrives. That pause is a real mechanic —
it is rest the player earned — and it is the window the clear animation and its
sound live in.

The invariant that makes it safe is that the rows are *blanked* at lock rather
than left full, so the board is never sitting on an uncollapsed full row.
"""

from __future__ import annotations

import random

import pytest

from tetris.core.board import Board
from tetris.core.constants import (
    CLEAR_DELAY_FLOOR_MS,
    CLEAR_DELAY_START_MS,
    MAX_LOCK_RESETS,
    MAX_GRAVITY_LEVEL,
    TOTAL_HEIGHT,
    Action,
    PieceType,
)
from tetris.core.engine import EngineConfig, TetrisEngine
from tetris.core.events import EventType
from tetris.core.piece import Piece
from tetris.core.rules import clear_delay_ms

from helpers import make_board

#: A row needing only columns 4 and 5 — exactly what one O piece fills.
GAP_ROW = 0b1111001111


def engine_primed_for_a_clear(rows: tuple[int, ...] = (23,), level: int = 1) -> TetrisEngine:
    """An engine one hard drop away from clearing ``rows``."""
    engine = TetrisEngine(EngineConfig(seed=1, start_level=level))
    for y in rows:
        engine.board.rows[y] = GAP_ROW
        for x in (0, 1, 2, 3, 6, 7, 8, 9):
            engine.board.colors[y, x] = int(PieceType.J)
    engine.piece = Piece(PieceType.O, x=3, y=2)
    return engine


# -- the timing curve -----------------------------------------------------


def test_clear_delay_shrinks_with_level():
    assert clear_delay_ms(1) == pytest.approx(CLEAR_DELAY_START_MS)
    assert clear_delay_ms(MAX_GRAVITY_LEVEL) == pytest.approx(180.0)
    assert clear_delay_ms(99) == pytest.approx(CLEAR_DELAY_FLOOR_MS)


def test_clear_delay_is_monotonically_decreasing():
    values = [clear_delay_ms(level) for level in range(1, 40)]
    assert all(b <= a for a, b in zip(values, values[1:]))


# -- blanking -------------------------------------------------------------


def test_blank_rows_empties_without_collapsing():
    board = Board()
    board.lock(tuple((x, 20) for x in range(10)), PieceType.I)
    board.lock(((0, 19),), PieceType.T)

    board.blank_rows([20])

    assert board.rows[20] == 0
    assert board.colors[20].sum() == 0
    # The block above must not have moved — that is the whole point.
    assert board.rows[19] == 1
    board.check_invariants()


# -- the pause ------------------------------------------------------------


def test_clear_blanks_rows_and_holds_the_board():
    engine = engine_primed_for_a_clear()
    events = engine.step(Action.HARD_DROP, 0.0)

    assert engine.stats.lines == 1, "scoring happens at lock, not after the pause"
    assert engine.board.rows[23] == 0, "the row should be empty immediately"
    assert engine.piece is None, "no piece is in play during the pause"
    assert engine.clearing_rows == (23,)
    assert not engine.game_over

    clear = next(e for e in events if e.type is EventType.LINE_CLEAR)
    assert clear.rows == (23,), "the renderer needs to know which rows went"

    engine.board.check_invariants()


def test_board_never_holds_a_full_uncollapsed_row():
    # If the engine deferred the clear instead of blanking, the board would sit
    # on a full row and this check would fail.
    engine = engine_primed_for_a_clear(rows=(22, 23))
    engine.step(Action.HARD_DROP, 0.0)
    engine.board.check_invariants()


def test_collapse_happens_once_the_delay_expires():
    engine = engine_primed_for_a_clear()
    # Debris above the cleared row, so the collapse is observable.
    engine.board.rows[21] = 0b1
    engine.board.colors[21, 0] = int(PieceType.T)

    engine.step(Action.HARD_DROP, 0.0)
    delay = clear_delay_ms(engine.stats.level) / 1000.0

    engine.step(Action.NOOP, delay * 0.5)
    assert engine.clearing_rows == (23,), "should still be paused halfway through"
    assert engine.board.rows[21] == 0b1, "stack must not move until the pause ends"

    events = engine.step(Action.NOOP, delay)
    assert engine.clearing_rows == ()
    assert engine.piece is not None, "the next piece arrives with the collapse"
    assert EventType.PIECE_SPAWN in [e.type for e in events]
    # The debris fell one row into the space the cleared line left.
    assert engine.board.rows[22] == 0b1
    engine.board.check_invariants()


def test_clear_progress_runs_from_zero_to_one():
    engine = engine_primed_for_a_clear()
    engine.step(Action.HARD_DROP, 0.0)
    assert engine.clear_progress == 0.0

    delay = clear_delay_ms(engine.stats.level) / 1000.0
    engine.step(Action.NOOP, delay * 0.5)
    assert 0.4 < engine.clear_progress < 0.6

    engine.step(Action.NOOP, delay * 0.4)
    assert engine.clear_progress > 0.85


def test_progress_is_zero_when_nothing_is_clearing():
    engine = TetrisEngine(EngineConfig(seed=1))
    assert engine.clear_progress == 0.0
    assert engine.clearing_rows == ()


def test_input_is_ignored_during_the_pause():
    engine = engine_primed_for_a_clear()
    engine.step(Action.HARD_DROP, 0.0)
    score = engine.stats.score
    placed = engine.stats.pieces_placed

    for action in (Action.LEFT, Action.HARD_DROP, Action.ROTATE_CW, Action.HOLD):
        engine.step(action, 0.0)

    assert engine.stats.score == score
    assert engine.stats.pieces_placed == placed
    assert engine.piece is None


def test_finish_clear_skips_the_pause():
    # The escape hatch for tools that step with dt=0 and would otherwise sit on
    # a board with no active piece forever.
    engine = engine_primed_for_a_clear()
    engine.step(Action.HARD_DROP, 0.0)

    events = engine.finish_clear()
    assert engine.clearing_rows == ()
    assert engine.piece is not None
    assert EventType.PIECE_SPAWN in [e.type for e in events]


def test_finish_clear_is_a_no_op_when_nothing_is_pending():
    engine = TetrisEngine(EngineConfig(seed=1))
    piece = engine.piece
    assert engine.finish_clear() == []
    assert engine.piece is piece


def test_perfect_clear_still_detected_through_the_pause():
    engine = engine_primed_for_a_clear(rows=(22, 23))
    events = engine.step(Action.HARD_DROP, 0.0)

    assert EventType.PERFECT_CLEAR in [e.type for e in events]
    assert not any(engine.board.rows)


def test_higher_levels_pause_for_less_time():
    slow = engine_primed_for_a_clear(level=1)
    fast = engine_primed_for_a_clear(level=20)
    for engine in (slow, fast):
        engine.step(Action.HARD_DROP, 0.0)

    # A tick that leaves level 1 still paused should already have freed level 20.
    tick = 0.25
    slow.step(Action.NOOP, tick)
    fast.step(Action.NOOP, tick)
    assert slow.clearing_rows != ()
    assert fast.clearing_rows == ()


# -- lock pulse -----------------------------------------------------------


def test_lock_progress_tracks_the_lock_delay():
    engine = TetrisEngine(EngineConfig(seed=1))
    engine.piece = Piece(PieceType.O, x=3, y=22)  # resting on the floor
    engine._lock_timer = 0.0
    assert engine.lock_progress == 0.0

    engine.step(Action.NOOP, 0.25)  # half of the 500 ms level-1 delay
    assert 0.4 < engine.lock_progress < 0.6


def test_lock_progress_is_zero_for_a_falling_piece():
    engine = TetrisEngine(EngineConfig(seed=1))
    assert engine.lock_progress == 0.0


# -- the move-reset budget ------------------------------------------------


def exhaust_resets(engine: TetrisEngine) -> None:
    for index in range(MAX_LOCK_RESETS + 3):
        engine.step(Action.LEFT if index % 2 else Action.RIGHT, 0.001)


def test_descending_to_a_new_row_refills_the_move_budget():
    """Guideline Extended Placement: a new lowest row restores the counter.

    Without this, a piece adjusted on a ledge and then slid off into a well
    arrives at the bottom with its budget spent and locks with no chance to
    adjust — punishing a perfectly ordinary maneuver.
    """
    engine = TetrisEngine(EngineConfig(seed=1))
    engine.board = make_board(
        "#####.....",
        "..........",
        "..........",
        "..........",
    )
    engine.piece = Piece(PieceType.O, x=2, y=18)  # resting on the ledge
    engine._lowest_row = max(y for _, y in engine.piece.cells())

    exhaust_resets(engine)
    assert engine._lock_resets == MAX_LOCK_RESETS, "the budget should be spent"

    # Slide off the end of the ledge and let it fall to the floor.
    for _ in range(6):
        engine.step(Action.RIGHT, 0.001)
    for _ in range(400):
        engine.step(Action.NOOP, 0.01)
        if engine.piece is None:
            break

    assert engine._lock_resets == 0, "landing lower must restore the budget"


def test_soft_drop_refills_the_move_budget():
    engine = TetrisEngine(EngineConfig(seed=1))
    engine.piece = Piece(PieceType.O, x=3, y=10)
    engine._lowest_row = max(y for _, y in engine.piece.cells())
    engine._lock_resets = 12

    engine.step(Action.SOFT_DROP, 0.0)
    assert engine._lock_resets == 0


def test_moving_sideways_does_not_refill_the_budget():
    # Only *descending* restores it; otherwise the cap would mean nothing.
    engine = TetrisEngine(EngineConfig(seed=1))
    engine.piece = Piece(PieceType.O, x=3, y=22)  # on the floor, cannot descend
    engine._lowest_row = max(y for _, y in engine.piece.cells())
    engine._lock_resets = 0

    engine.step(Action.LEFT, 0.0)
    engine.step(Action.RIGHT, 0.0)
    assert engine._lock_resets == 2


def test_a_floor_bound_piece_still_locks_despite_the_refill():
    """The refill must not reopen the infinite-stall hole the cap closed."""
    engine = TetrisEngine(EngineConfig(seed=1))
    engine.piece = Piece(PieceType.O, x=3, y=22)
    engine._lowest_row = max(y for _, y in engine.piece.cells())
    engine._lock_timer = 0.0
    engine._lock_resets = 0

    placed = engine.stats.pieces_placed
    steps = 0
    while engine.stats.pieces_placed == placed and steps < 500:
        engine.step(Action.LEFT if steps % 2 else Action.RIGHT, 0.1)
        steps += 1

    assert engine.stats.pieces_placed == placed + 1, "piece never locked"
    assert steps <= MAX_LOCK_RESETS + 10


def test_lowest_row_starts_from_the_spawned_piece():
    engine = TetrisEngine(EngineConfig(seed=1))
    assert engine.piece is not None
    assert engine._lowest_row == max(y for _, y in engine.piece.cells())


# -- fuzz -----------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_play_with_clears_keeps_invariants(seed):
    """The pause must not let the board reach an inconsistent state."""
    rng = random.Random(seed)
    engine = TetrisEngine(EngineConfig(seed=seed))
    actions = list(Action)
    games = 0

    for _ in range(8_000):
        engine.step(rng.choice(actions), 1.0 / 60.0)
        engine.board.check_invariants()

        if engine.clearing_rows:
            # While paused: no piece, and every clearing row genuinely empty.
            assert engine.piece is None
            for row in engine.clearing_rows:
                assert engine.board.rows[row] == 0
                assert 0 <= row < TOTAL_HEIGHT

        if engine.game_over:
            games += 1
            engine.reset(seed=rng.randrange(10_000))

    assert games > 0
