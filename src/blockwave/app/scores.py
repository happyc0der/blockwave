"""Persistent personal best.

Deliberately defensive: a high score is a nice-to-have, and nothing here is
allowed to take the game down. Every read and write is wrapped, so a missing,
corrupt, unreadable or unwritable file degrades to an in-memory score and play
carries on.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.engine import Stats

#: Kept in the user's home directory rather than the repo, so an installed copy
#: works and nothing lands in git.
DEFAULT_PATH = Path.home() / ".blockwave-highscore.json"


@dataclass(slots=True)
class HighScores:
    best: int = 0
    best_lines: int = 0
    best_level: int = 1
    games: int = 0


def load(path: Path | None = None) -> HighScores:
    """Read the saved scores, falling back to a fresh record on any problem."""
    path = path or DEFAULT_PATH
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return HighScores()
        return HighScores(
            best=int(data.get("best", 0)),
            best_lines=int(data.get("best_lines", 0)),
            best_level=int(data.get("best_level", 1)),
            games=int(data.get("games", 0)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Missing, unreadable, malformed, or holding the wrong types.
        return HighScores()


def save(scores: HighScores, path: Path | None = None) -> bool:
    """Write the scores. Returns whether it worked; never raises.

    Writes to a temporary file and replaces the target, so an interrupted save
    cannot leave a half-written file behind for the next run to choke on.
    """
    path = path or DEFAULT_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(asdict(scores), handle, indent=2)
            os.replace(handle.name, path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
        return True
    except (OSError, ValueError, TypeError):
        return False


def submit(scores: HighScores, stats: Stats) -> bool:
    """Fold a finished game into ``scores``. Returns True on a new record.

    Bests are tracked per-category, so a long game that scored poorly still
    improves ``best_lines``. Only beating the score counts as "a new record".
    """
    scores.games += 1
    scores.best_lines = max(scores.best_lines, stats.lines)
    scores.best_level = max(scores.best_level, stats.level)

    if stats.score > scores.best:
        scores.best = stats.score
        return True
    return False
