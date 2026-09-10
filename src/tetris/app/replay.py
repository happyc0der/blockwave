"""Recording and replaying a session, exactly.

The engine is deterministic given a seed, a sequence of actions and a fixed
timestep, so a recording needs nothing more than those three things. That turns
"the game glitched" into a file someone can hand over, and turns a glitch into a
regression test — which is the only durable defence for requirement 5.

The recording is sparse: a 240 Hz loop is almost entirely idle ticks, so only
ticks that carried an action are stored. A ten-minute game is a few thousand
entries rather than a hundred and forty thousand.

:func:`apply_tick` is deliberately shared by the live game loop and by playback.
If the two had their own copies of "how a tick is applied", a replay could
diverge from the session it claims to reproduce, and the whole tool would be
worse than useless.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..core.constants import Action
from ..core.engine import EngineConfig, TetrisEngine
from ..core.events import GameEvent

FORMAT_VERSION = 1


def apply_tick(engine: TetrisEngine, actions: list[Action], dt: float) -> list[GameEvent]:
    """Advance the engine by one logic tick, returning everything it emitted.

    Time passes once per tick no matter how many actions arrived in it,
    otherwise a burst of auto-repeat would also accelerate gravity.
    """
    if not actions:
        return engine.step(Action.NOOP, dt)

    events: list[GameEvent] = []
    for index, action in enumerate(actions):
        events.extend(engine.step(action, dt if index == 0 else 0.0))
        if engine.game_over:
            break
    return events


def state_digest(engine: TetrisEngine) -> str:
    """A short fingerprint of everything a replay must reproduce."""
    stats = engine.stats
    payload = "|".join(
        str(part)
        for part in (
            stats.score,
            stats.lines,
            stats.level,
            stats.pieces_placed,
            stats.tspins,
            stats.tetrises,
            int(engine.game_over),
            ",".join(str(row) for row in engine.board.rows),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(slots=True)
class Replay:
    seed: int
    start_level: int
    dt: float
    ticks: int = 0
    #: ``(tick_index, action)`` for every non-idle action, in order.
    actions: list[tuple[int, int]] = field(default_factory=list)
    digest: str = ""
    score: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": FORMAT_VERSION,
                "seed": self.seed,
                "start_level": self.start_level,
                "dt": self.dt,
                "ticks": self.ticks,
                "actions": self.actions,
                "digest": self.digest,
                "score": self.score,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Replay:
        data = json.loads(text)
        version = data.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported replay version {version!r}")
        return cls(
            seed=int(data["seed"]),
            start_level=int(data["start_level"]),
            dt=float(data["dt"]),
            ticks=int(data["ticks"]),
            actions=[(int(t), int(a)) for t, a in data["actions"]],
            digest=str(data.get("digest", "")),
            score=int(data.get("score", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Replay:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class Recorder:
    """Collects a replay as the game is played."""

    def __init__(self, seed: int, start_level: int, dt: float) -> None:
        self.replay = Replay(seed=seed, start_level=start_level, dt=dt)

    def tick(self, actions: list[Action]) -> None:
        """Record one logic tick. Call once per tick, even when idle."""
        index = self.replay.ticks
        for action in actions:
            if action is not Action.NOOP:
                self.replay.actions.append((index, int(action)))
        self.replay.ticks = index + 1

    def finish(self, engine: TetrisEngine) -> Replay:
        self.replay.digest = state_digest(engine)
        self.replay.score = engine.stats.score
        return self.replay


def playback(replay: Replay) -> TetrisEngine:
    """Re-run a replay and return the resulting engine."""
    engine = TetrisEngine(
        EngineConfig(seed=replay.seed, start_level=replay.start_level)
    )

    by_tick: dict[int, list[Action]] = {}
    for index, action in replay.actions:
        by_tick.setdefault(index, []).append(Action(action))

    for index in range(replay.ticks):
        apply_tick(engine, by_tick.get(index, []), replay.dt)

    return engine


def verify(replay: Replay) -> tuple[bool, TetrisEngine]:
    """Replay and check it reproduces the recorded final state."""
    engine = playback(replay)
    if not replay.digest:
        return True, engine
    return state_digest(engine) == replay.digest, engine
