"""Persistent high scores.

The theme of these tests is that a high score is a nice-to-have: no state of the
save file may ever stop the game from starting.
"""

from __future__ import annotations

import json

import pytest

from blockwave.app import scores as scores_store
from blockwave.app.scores import HighScores
from blockwave.core.engine import Stats


def make_stats(score: int, lines: int = 0, level: int = 1) -> Stats:
    return Stats(score=score, lines=lines, level=level)


# -- submit ---------------------------------------------------------------


def test_first_game_sets_a_record():
    scores = HighScores()
    assert scores_store.submit(scores, make_stats(1000)) is True
    assert scores.best == 1000
    assert scores.games == 1


def test_lower_score_is_not_a_record():
    scores = HighScores(best=5000)
    assert scores_store.submit(scores, make_stats(1000)) is False
    assert scores.best == 5000


def test_equalling_the_best_is_not_a_record():
    scores = HighScores(best=5000)
    assert scores_store.submit(scores, make_stats(5000)) is False
    assert scores.best == 5000


def test_bests_are_tracked_per_category():
    # A long, low-scoring game should still improve the lines record.
    scores = HighScores(best=99_999, best_lines=5, best_level=2)
    assert scores_store.submit(scores, make_stats(100, lines=80, level=9)) is False
    assert scores.best == 99_999
    assert scores.best_lines == 80
    assert scores.best_level == 9


def test_games_counter_increments_regardless():
    scores = HighScores(best=10_000)
    for _ in range(3):
        scores_store.submit(scores, make_stats(1))
    assert scores.games == 3


# -- persistence ----------------------------------------------------------


def test_round_trip(tmp_path):
    path = tmp_path / "scores.json"
    original = HighScores(best=1234, best_lines=56, best_level=7, games=8)

    assert scores_store.save(original, path) is True
    assert scores_store.load(path) == original


def test_missing_file_yields_defaults(tmp_path):
    assert scores_store.load(tmp_path / "nope.json") == HighScores()


@pytest.mark.parametrize(
    "content",
    [
        "",                       # empty
        "not json at all",        # garbage
        "[1, 2, 3]",              # valid json, wrong shape
        '{"best": "lots"}',       # right shape, wrong type
        '{"best": null}',
    ],
    ids=["empty", "garbage", "wrong-shape", "wrong-type", "null"],
)
def test_corrupt_file_yields_defaults_without_raising(tmp_path, content):
    path = tmp_path / "scores.json"
    path.write_text(content, encoding="utf-8")
    assert scores_store.load(path) == HighScores()


def test_save_to_an_unwritable_location_reports_failure(tmp_path):
    # A directory where the file should be: unwritable, but must not raise.
    path = tmp_path / "scores.json"
    path.mkdir()
    assert scores_store.save(HighScores(best=1), path) is False


def test_save_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "scores.json"
    scores_store.save(HighScores(best=1), path)
    assert [p.name for p in tmp_path.iterdir()] == ["scores.json"]


def test_save_replaces_atomically(tmp_path):
    # An existing good file must survive being overwritten with new content.
    path = tmp_path / "scores.json"
    scores_store.save(HighScores(best=1), path)
    scores_store.save(HighScores(best=2), path)

    assert json.loads(path.read_text())["best"] == 2
    assert scores_store.load(path).best == 2
