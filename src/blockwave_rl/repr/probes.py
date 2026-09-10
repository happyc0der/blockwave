"""Linear probes: does the representation encode the board? OFFLINE DIAGNOSTIC ONLY.

A frozen encoder's latents are regressed onto board properties computed from
the true occupancy grid. The targets come from engine state, which is exactly
why this module must never feed the policy or the reward — it exists to answer
one question before any RL is attempted: does the latent space actually
represent the things a reward about board quality would have to be about?

If a *linear* readout cannot recover stack height, the latents do not encode
it, and no reward built on them can rank boards by it.
"""

from __future__ import annotations

import numpy as np

#: Thresholds a representation must clear before RL begins.
THRESHOLDS = {"max_height": 0.80, "holes": 0.60, "bumpiness": 0.50}


def column_heights(board: np.ndarray) -> np.ndarray:
    """Per-column stack height for a (rows, cols) 0/1 grid, row 0 at the top."""
    rows = board.shape[0]
    filled = board.astype(bool)
    first = np.where(filled.any(axis=0), filled.argmax(axis=0), rows)
    return rows - first


def features(board: np.ndarray) -> dict[str, float]:
    heights = column_heights(board)
    filled = board.astype(bool)
    rows = board.shape[0]
    holes = 0
    for col in range(board.shape[1]):
        top = rows - heights[col]
        holes += int((~filled[top:, col]).sum())
    return {
        "max_height": float(heights.max()),
        "holes": float(holes),
        "bumpiness": float(np.abs(np.diff(heights)).sum()),
    }


def r_squared(latents: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Held-out R^2 of an ordinary least-squares linear readout."""
    X = np.hstack([latents, np.ones((len(latents), 1))])
    coef, *_ = np.linalg.lstsq(X[train], target[train], rcond=None)
    pred = X[test] @ coef
    resid = ((target[test] - pred) ** 2).sum()
    total = ((target[test] - target[test].mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else 0.0


def evaluate(latents: np.ndarray, boards: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    table = [features(b) for b in boards]
    return {
        name: r_squared(latents, np.array([row[name] for row in table]), train, test)
        for name in THRESHOLDS
    }
