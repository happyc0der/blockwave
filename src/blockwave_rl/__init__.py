"""An agent that learns BLOCKWAVE from pixels, with no reward from the game.

The game's score, line count and level never reach the agent — not in its
observations, not in its reward, not in how its checkpoints are chosen. It
optimizes a principle ("prefer states resembling those you have seen") and
derives the reward function itself from its own experience.

This package depends on `blockwave`; `blockwave` never depends on this.
"""
