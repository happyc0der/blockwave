"""The reward firewall: score, lines and level never reach the agent.

This is the guarantee the whole project rests on. "The agent was never given a
reward" means nothing if the score is printed on the pixels it learns from, so
these tests attack the observation directly rather than trusting the crop.

They exist because the first version of the crop leaked. At the agent's 4 px
cell size the HUD was laid out for 30 px and nothing clipped it: in
BOARD_PLUS_PREVIEW the LINES value was drawn below its masked panel in plain
view, and in BOARD_ONLY label fragments spilled into the playfield. Separately,
the combo counter was drawn inside the playfield at every cell size.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blockwave.core.constants import Action  # noqa: E402
from blockwave_rl.env.base import BlockwaveEnv, EnvConfig, ObsMode  # noqa: E402
from blockwave_rl.env.crops import Variant  # noqa: E402

VARIANTS = list(Variant)
CELL_SIZES = [3, 4, 6, 8]

#: Every piece of game-progress state that is drawn anywhere on screen. Each is
#: varied independently; the observation must not move for any of them.
PROGRESS_FIELDS = {
    "score": [0, 7, 1_234, 9_999_999],
    "lines": [0, 3, 88, 9_999],
    "level": [1, 9, 20, 99],
    "combo": [-1, 0, 1, 7, 42],
    "b2b": [False, True],
}


def settled_env(variant: Variant, cell_px: int = 4) -> BlockwaveEnv:
    """An env with a non-trivial board, so leaks cannot hide in empty space."""
    env = BlockwaveEnv(EnvConfig(variant=variant, cell_px=cell_px, gravity_scale=8.0))
    env.reset(seed=11)
    for action in [Action.LEFT] * 3 + [Action.HARD_DROP] + [Action.RIGHT] * 3 + [Action.HARD_DROP]:
        env.step(int(action))
    return env


def observe_with(env: BlockwaveEnv, **stats) -> np.ndarray:
    for name, value in stats.items():
        setattr(env.engine.stats, name, value)
    return env._pixels()


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.value for v in VARIANTS])
@pytest.mark.parametrize("cell_px", CELL_SIZES)
@pytest.mark.parametrize("field", sorted(PROGRESS_FIELDS))
def test_progress_state_never_changes_the_observation(variant, cell_px, field):
    """Vary one piece of progress state; every observation must be identical."""
    env = settled_env(variant, cell_px)
    frames = [observe_with(env, **{field: value}) for value in PROGRESS_FIELDS[field]]
    for value, frame in zip(PROGRESS_FIELDS[field][1:], frames[1:]):
        changed = int((frame != frames[0]).sum())
        assert changed == 0, (
            f"{variant.value} @ {cell_px}px: setting {field}={value!r} changed "
            f"{changed} observed pixels — progress state is leaking to the agent"
        )


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.value for v in VARIANTS])
def test_all_progress_state_at_once(variant):
    env = settled_env(variant)
    base = observe_with(env, score=0, lines=0, level=1, combo=-1, b2b=False)
    loud = observe_with(env, score=9_999_999, lines=9_999, level=99, combo=42, b2b=True)
    assert np.array_equal(base, loud)


def test_the_environment_emits_no_reward():
    """There is no reward here to leak. The intrinsic reward is computed elsewhere."""
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0))
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(3_000):
        _, reward, *_ = env.step(int(rng.integers(env.n_actions)))
        assert reward == 0.0


def test_score_lives_only_in_info():
    env = BlockwaveEnv(EnvConfig(gravity_scale=8.0))
    obs, info = env.reset(seed=3)
    out = env.step(int(Action.HARD_DROP))
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 0.0 and out[2] is False
    assert "score" in out[4] and "lines" in out[4], "evaluation needs these in info"


def test_board_state_mode_is_occupancy_only():
    """The feasibility track sees cells, not numbers."""
    env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.BOARD_STATE, gravity_scale=8.0))
    obs, _ = env.reset(seed=3)
    assert obs.dtype == np.uint8
    assert set(np.unique(obs)) <= {0, 1}
    env.engine.stats.score = 9_999_999
    assert np.array_equal(obs, env.occupancy())


def test_training_code_never_reads_info():
    """Static check: nothing under blockwave_rl reads score or lines out of info."""
    import pathlib

    import blockwave_rl

    root = pathlib.Path(blockwave_rl.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in {"base.py", "evaluate.py"}:
            continue  # the env writes info; the evaluator is allowed to read it
        text = path.read_text()
        for key in ("'score'", '"score"', "'lines'", '"lines"', "'level'", '"level"'):
            if key in text:
                offenders.append(f"{path.relative_to(root)} mentions {key}")
    assert not offenders, offenders
