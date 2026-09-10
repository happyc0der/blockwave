"""Pytest fixtures. Board-building helpers live in ``helpers.py``."""

from __future__ import annotations

import pytest

from tetris.core.board import Board


@pytest.fixture
def empty_board() -> Board:
    return Board()
