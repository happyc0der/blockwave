"""Pytest configuration marker for ``tests/``.

There are no shared fixtures. The one that lived here, ``empty_board``, was
defined twice (here and in ``helpers.py``) and used by nothing, and both copies
were removed in the 2026-09-15 sweep. Board-building helpers are plain
functions in ``helpers.py``, imported directly by the tests that need them.

The file is kept only because a ``conftest.py`` is the conventional place a
reader looks first; removing it would change nothing that runs.
"""
