"""The game engine: the single source of truth for what is happening.

Pure Python and numpy. Nothing in this module imports pygame, opens a window or
touches an audio device, which is what lets the same code run a 60 fps arcade
cabinet and a headless training loop at tens of thousands of steps per second.

The engine advances by ``step(action, dt)``. Time is passed in explicitly rather
than read from a clock so that a fixed-timestep game loop, an RL environment and
a replay can all drive it identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board
from .constants import (
    MAX_LOCK_RESETS,
    VISIBLE_TOP,
    Action,
    PieceType,
)
from .events import EventType, GameEvent
from .piece import (
    Piece,
    drop_distance,
    ghost_cells,
    is_grounded,
    try_move,
    try_rotate,
)
from .randomizer import SevenBag
from .rules import (
    HARD_DROP_POINTS,
    clear_delay_ms,
    SOFT_DROP_POINTS,
    TSpin,
    detect_tspin,
    gravity_seconds_per_row,
    level_for_lines,
    lock_delay_ms,
    score_lock,
)


@dataclass(slots=True)
class EngineConfig:
    """Everything tunable about a game.

    ``start_level`` and ``gravity_scale`` are the difficulty-curriculum knobs the
    RL environment exposes: start an agent on a slow board and ramp it up.
    """

    seed: int | None = None
    start_level: int = 1
    preview: int = 5
    gravity_scale: float = 1.0
    #: Set false to make the engine a pure drop-only game (used by tests).
    allow_hold: bool = True


@dataclass(slots=True)
class Stats:
    """Cumulative telemetry. Mirrored into the RL ``info`` dict every step."""

    score: int = 0
    lines: int = 0
    level: int = 1
    combo: int = -1
    b2b: bool = False
    pieces_placed: int = 0
    tspins: int = 0
    tetrises: int = 0
    perfect_clears: int = 0
    line_clears: dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0}
    )


class TetrisEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.board = Board()
        self.bag = SevenBag(self.config.seed, self.config.preview)
        self.stats = Stats()
        self.piece: Piece | None = None
        self.hold: PieceType | None = None
        self.hold_used = False
        self.game_over = False
        self.reset()

    # -- lifecycle --------------------------------------------------------

    def reset(self, seed: int | None = None) -> list[GameEvent]:
        if seed is not None:
            self.config.seed = seed
        self.board = Board()
        self.bag.reset(self.config.seed)
        self.stats = Stats(level=self.config.start_level)
        self.hold = None
        self.hold_used = False
        self.game_over = False

        self._gravity_accum = 0.0
        self._lock_timer = 0.0
        self._lock_resets = 0
        self._lowest_row = -1
        self._last_move_was_rotation = False
        self._last_kick_index = -1
        self._pending_collapse: list[int] = []
        self._clear_timer = 0.0

        self.piece = None
        return self._spawn()

    # -- the step ---------------------------------------------------------

    def step(self, action: Action | int, dt: float) -> list[GameEvent]:
        """Advance the game by ``dt`` seconds while applying ``action``.

        Order matters: the action is applied first, then gravity, then the lock
        check. Doing it this way means a player who slides a piece into a gap on
        the very last tick of lock delay gets the placement, which is the
        behaviour that makes the game feel fair.
        """
        if self.game_over:
            return []

        events: list[GameEvent] = []

        # A line clear is a pause: the rows sit empty, no piece is in play, and
        # the only thing advancing is the clear timer. Input is ignored for its
        # duration, which is the beat of rest the player earned.
        if self._pending_collapse:
            self._advance_clear(dt, events)
            return events

        if self.piece is None:
            return events

        self._apply_action(Action(action), events)

        # A hard drop locks the piece and spawns the next one immediately, so
        # there is nothing left this step for gravity to act on.
        if self.game_over or self.piece is None:
            return events

        self._apply_gravity(dt, events)
        if self.game_over or self.piece is None:
            return events

        self._apply_lock_delay(dt, events)
        return events

    # -- actions ----------------------------------------------------------

    def _apply_action(self, action: Action, events: list[GameEvent]) -> None:
        piece = self.piece
        assert piece is not None

        if action is Action.NOOP:
            return

        if action in (Action.LEFT, Action.RIGHT):
            dx = -1 if action is Action.LEFT else 1
            if try_move(piece, self.board, dx, 0):
                self._last_move_was_rotation = False
                self._last_kick_index = -1
                self._on_successful_move()
                events.append(GameEvent(EventType.PIECE_MOVE, dx))
            return

        if action is Action.SOFT_DROP:
            if try_move(piece, self.board, 0, 1):
                self.stats.score += SOFT_DROP_POINTS
                self._last_move_was_rotation = False
                self._last_kick_index = -1
                self._gravity_accum = 0.0
                self._note_descent()
                events.append(GameEvent(EventType.SOFT_DROP))
            return

        if action is Action.HARD_DROP:
            distance = drop_distance(piece, self.board)
            piece.y += distance
            self.stats.score += HARD_DROP_POINTS * distance
            # Dropping is a translation, so it cancels any pending T-spin.
            if distance:
                self._last_move_was_rotation = False
                self._last_kick_index = -1
            events.append(GameEvent(EventType.HARD_DROP, distance))
            self._lock_piece(events)
            return

        if action in (Action.ROTATE_CW, Action.ROTATE_CCW):
            direction = 1 if action is Action.ROTATE_CW else -1
            result = try_rotate(piece, self.board, direction)
            if result:
                self._last_move_was_rotation = True
                self._last_kick_index = result.kick_index
                self._on_successful_move()
                events.append(GameEvent(EventType.PIECE_ROTATE, direction))
            return

        if action is Action.HOLD:
            self._do_hold(events)

    def _note_descent(self) -> None:
        """Restore the move-reset budget on reaching a new lowest row.

        Guideline Extended Placement resets the fifteen-move counter whenever a
        piece falls below every row it has previously occupied. Without this, a
        piece that is adjusted on a ledge and then slides off into a well
        arrives at the bottom with its whole budget already spent, and locks
        with no chance to adjust — which punishes a perfectly ordinary
        maneuver, and bites hardest at high levels where the delay is short.
        """
        piece = self.piece
        if piece is None:
            return
        lowest = max(y for _, y in piece.cells())
        if lowest > self._lowest_row:
            self._lowest_row = lowest
            self._lock_resets = 0

    def _on_successful_move(self) -> None:
        """Grant a lock-delay reset for a move made while resting on the stack.

        Capped at :data:`MAX_LOCK_RESETS`; without the cap a piece can be
        rotated back and forth on top of the stack forever and the game never
        progresses.
        """
        piece = self.piece
        assert piece is not None

        # Checked first: an SRS kick can push a piece downward, and the budget
        # a descent grants must not be immediately eaten by the move that
        # caused it.
        self._note_descent()

        if is_grounded(piece, self.board) and self._lock_resets < MAX_LOCK_RESETS:
            self._lock_timer = 0.0
            self._lock_resets += 1

    def _do_hold(self, events: list[GameEvent]) -> None:
        if not self.config.allow_hold or self.hold_used:
            events.append(GameEvent(EventType.HOLD_DENIED))
            return

        assert self.piece is not None
        current = self.piece.type
        if self.hold is None:
            self.hold = current
            self._spawn(events)
        else:
            swapped = self.hold
            self.hold = current
            self._spawn(events, piece_type=swapped)

        self.hold_used = True
        events.append(GameEvent(EventType.PIECE_HOLD, int(current)))

    # -- gravity and locking ----------------------------------------------

    def _apply_gravity(self, dt: float, events: list[GameEvent]) -> None:
        piece = self.piece
        assert piece is not None

        seconds_per_row = gravity_seconds_per_row(self.stats.level)
        if seconds_per_row > 0.0:
            seconds_per_row /= max(1e-6, self.config.gravity_scale)

        if seconds_per_row <= 0.0:
            # True 20G: the piece is on the stack the instant it exists.
            distance = drop_distance(piece, self.board)
            if distance:
                piece.y += distance
                self._last_move_was_rotation = False
                self._last_kick_index = -1
                self._note_descent()
            return

        self._gravity_accum += dt
        while self._gravity_accum >= seconds_per_row:
            self._gravity_accum -= seconds_per_row
            if not try_move(piece, self.board, 0, 1):
                self._gravity_accum = 0.0
                break
            # Falling is a translation, so it cancels any pending T-spin.
            self._last_move_was_rotation = False
            self._last_kick_index = -1
            self._note_descent()

    def _apply_lock_delay(self, dt: float, events: list[GameEvent]) -> None:
        piece = self.piece
        assert piece is not None

        if not is_grounded(piece, self.board):
            self._lock_timer = 0.0
            return

        self._lock_timer += dt * 1000.0
        if self._lock_timer >= lock_delay_ms(self.stats.level):
            self._lock_piece(events)

    def _lock_piece(self, events: list[GameEvent]) -> None:
        piece = self.piece
        assert piece is not None

        tspin = detect_tspin(
            self.board, piece, self._last_move_was_rotation, self._last_kick_index
        )
        cells = piece.cells()

        # Lock out: the piece came to rest entirely inside the buffer, above the
        # visible playfield. This is a loss even though nothing overlapped.
        locked_out = all(y < VISIBLE_TOP for _, y in cells)

        self.board.lock(cells, piece.type)
        self.stats.pieces_placed += 1
        events.append(GameEvent(EventType.PIECE_LOCK, int(piece.type)))

        full = self.board.full_rows()
        # Empty the rows now, collapse them after the clear delay. Scoring still
        # happens immediately, so the player is paid at the moment of the lock.
        self.board.blank_rows(full)
        lines = len(full)
        perfect = lines > 0 and not any(self.board.rows)

        outcome = score_lock(
            lines=lines,
            tspin=tspin,
            level=self.stats.level,
            combo=self.stats.combo,
            b2b_active=self.stats.b2b,
            perfect_clear=perfect,
        )

        stats = self.stats
        stats.score += outcome.score
        stats.combo = outcome.combo
        stats.b2b = outcome.b2b

        if tspin is not TSpin.NONE:
            stats.tspins += 1
            events.append(GameEvent(EventType.TSPIN, int(tspin)))

        if lines:
            stats.lines += lines
            stats.line_clears[lines] += 1
            if lines == 4:
                stats.tetrises += 1
            events.append(GameEvent(EventType.LINE_CLEAR, lines, tuple(full)))
            if outcome.combo > 0:
                events.append(GameEvent(EventType.COMBO, outcome.combo))
            if outcome.perfect_clear:
                stats.perfect_clears += 1
                events.append(GameEvent(EventType.PERFECT_CLEAR))

            new_level = level_for_lines(stats.lines, self.config.start_level)
            if new_level > stats.level:
                stats.level = new_level
                events.append(GameEvent(EventType.LEVEL_UP, new_level))

        if locked_out:
            self._end_game(events)
            return

        self.hold_used = False

        if full:
            # Hold the board here: no active piece, rows sitting empty, until
            # the clear delay expires.
            self._pending_collapse = full
            self._clear_timer = 0.0
            self.piece = None
        else:
            self._spawn(events)

    # -- the line-clear pause ---------------------------------------------

    def _advance_clear(self, dt: float, events: list[GameEvent]) -> None:
        self._clear_timer += dt * 1000.0
        if self._clear_timer >= clear_delay_ms(self.stats.level):
            self._collapse(events)

    def _collapse(self, events: list[GameEvent]) -> None:
        rows = self._pending_collapse
        self._pending_collapse = []
        self._clear_timer = 0.0
        self.board.clear_rows(rows)
        self._spawn(events)

    def finish_clear(self) -> list[GameEvent]:
        """Complete a pending clear immediately, skipping the pause.

        For tools and tests that step with ``dt=0`` and would otherwise sit
        forever on a board with no active piece.
        """
        events: list[GameEvent] = []
        if self._pending_collapse:
            self._collapse(events)
        return events

    @property
    def clearing_rows(self) -> tuple[int, ...]:
        """Rows currently sitting empty, waiting to collapse."""
        return tuple(self._pending_collapse)

    @property
    def clear_progress(self) -> float:
        """How far through the clear pause we are, 0 to 1."""
        if not self._pending_collapse:
            return 0.0
        return min(1.0, self._clear_timer / clear_delay_ms(self.stats.level))

    @property
    def lock_progress(self) -> float:
        """How far through the lock delay the active piece is, 0 to 1.

        Drives the visual pulse that tells the player how long they have left —
        the most timing-sensitive moment in the game, and one with no feedback
        at all before this.
        """
        if self.piece is None or self._lock_timer <= 0.0:
            return 0.0
        return min(1.0, self._lock_timer / lock_delay_ms(self.stats.level))

    # -- spawning ---------------------------------------------------------

    def _spawn(
        self,
        events: list[GameEvent] | None = None,
        piece_type: PieceType | None = None,
    ) -> list[GameEvent]:
        if events is None:
            events = []

        self.piece = Piece.spawn(piece_type if piece_type is not None else self.bag.next())
        self._gravity_accum = 0.0
        self._lock_timer = 0.0
        self._lock_resets = 0
        self._last_move_was_rotation = False
        self._last_kick_index = -1
        # Where this piece started; descending past it refills the move budget.
        self._lowest_row = max(y for _, y in self.piece.cells())

        # Block out: the new piece has nowhere to appear.
        if self.board.collides(self.piece.cells()):
            events.append(GameEvent(EventType.PIECE_SPAWN, int(self.piece.type)))
            self._end_game(events)
            return events

        events.append(GameEvent(EventType.PIECE_SPAWN, int(self.piece.type)))
        return events

    def _end_game(self, events: list[GameEvent]) -> None:
        self.game_over = True
        events.append(GameEvent(EventType.GAME_OVER, self.stats.score))

    # -- read-only views for renderers and observations -------------------

    def ghost(self) -> tuple[tuple[int, int], ...]:
        if self.piece is None:
            return ()
        return ghost_cells(self.piece, self.board)

    def active_cells(self) -> tuple[tuple[int, int], ...]:
        return self.piece.cells() if self.piece else ()

    def preview(self, count: int | None = None) -> list[PieceType]:
        return self.bag.peek(count)

    def telemetry(self) -> dict[str, float | int | bool]:
        """The per-step ``info`` payload — requirement 4, made concrete.

        Everything an agent, a logger or a reward-shaping wrapper could want,
        computed once and handed over in one dict.
        """
        board = self.board
        stats = self.stats
        return {
            "score": stats.score,
            "lines": stats.lines,
            "level": stats.level,
            "combo": max(0, stats.combo),
            "b2b": stats.b2b,
            "pieces_placed": stats.pieces_placed,
            "tspins": stats.tspins,
            "tetrises": stats.tetrises,
            "perfect_clears": stats.perfect_clears,
            "holes": board.holes(),
            "bumpiness": board.bumpiness(),
            "aggregate_height": board.aggregate_height(),
            "max_height": board.max_height(),
            "game_over": self.game_over,
        }

    def to_ascii(self) -> str:
        return self.board.to_ascii(self.active_cells())
