"""Game performance — the yardstick the agent never sees.

The only module in this package permitted to read score, lines or level (the
firewall's static test exempts it). Nothing here returns anything the learner
consumes. It measures; it does not steer.

Lines are tracked as the sum of positive increments of each env's per-game line
count, because a soft reset zeroes that count — a naive "lines at top-out"
would read the *new* game's zero.
"""

from __future__ import annotations

import numpy as np


class GameTracker:
    def __init__(self, n_envs: int) -> None:
        self._prev_lines = np.zeros(n_envs)
        self._prev_pieces = np.zeros(n_envs)
        self.reset_window()

    def reset_window(self) -> None:
        self.lines = 0.0
        self.pieces = 0.0
        self.top_outs = 0
        self.steps = 0

    def update(self, infos: list[dict]) -> None:
        for i, info in enumerate(infos):
            lines, pieces = info["lines"], info["pieces"]
            self.lines += max(0.0, lines - self._prev_lines[i])
            self.pieces += max(0.0, pieces - self._prev_pieces[i])
            self._prev_lines[i], self._prev_pieces[i] = lines, pieces
            self.top_outs += int(info["top_out"])
            self.steps += 1

    def summary(self) -> dict[str, float]:
        return {
            "lines_per_piece": self.lines / max(self.pieces, 1.0),
            "lines_per_1k_steps": 1000.0 * self.lines / max(self.steps, 1),
            "pieces_per_1k_steps": 1000.0 * self.pieces / max(self.steps, 1),
            "top_outs_per_1k_steps": 1000.0 * self.top_outs / max(self.steps, 1),
            # Per piece, not per step. A per-step rate falls whenever the agent
            # merely plays slower, which is not the same as dying less.
            "top_outs_per_piece": self.top_outs / max(self.pieces, 1.0),
        }
