"""The vector env's measurement must not depend on how it is parallelised.

`--workers` is a throughput knob. It used to also decide which games were
played: each worker seeded its envs from its own index, so halving the worker
count handed the evaluator a different sample and moved the numbers. That is a
measurement bug rather than a performance one, and it is the sort that reads as
a real effect when two runs are compared at different settings.
"""

from __future__ import annotations

import numpy as np

from blockwave_rl.world.pixel_vector import PixelVecEnv


class _NullReward:
    """Stands in for `FrameReward`: the seeding question is about the envs."""

    def death_charge(self, gamma: float) -> float:
        return -1.0

    def encode(self, frames: np.ndarray) -> np.ndarray:
        return frames.reshape(len(frames), -1).astype(np.float32)

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        return np.zeros(len(latents), dtype=np.float32)


def _rollout(n_envs: int, n_workers: int, steps: int, seed: int) -> np.ndarray:
    vec = PixelVecEnv(n_envs, n_workers, reward_fn=_NullReward(), seed=seed, frame_stack=2)
    try:
        vec.reset()
        rng = np.random.default_rng(0)
        frames = []
        for _ in range(steps):
            batch = vec.step(rng.integers(0, 8, size=n_envs))
            frames.append(batch.frames.copy())
        return np.stack(frames)
    finally:
        vec.close()


def test_worker_count_does_not_change_what_is_measured():
    """Same envs, same seed, same actions -> same frames, at any parallelism."""
    few = _rollout(8, 2, steps=12, seed=1234)
    many = _rollout(8, 4, steps=12, seed=1234)
    assert np.array_equal(few, many), (
        "worker count changed the observations, so it changed which games were "
        "played and any cross-run comparison at different --workers is invalid"
    )


def test_env_seeds_follow_global_index():
    """Env i is the same game whether it is worker 0's third or worker 2's first."""
    split_two = _rollout(4, 2, steps=8, seed=99)
    split_four = _rollout(4, 4, steps=8, seed=99)
    for index in range(4):
        assert np.array_equal(split_two[:, index], split_four[:, index]), (
            f"env {index} diverged between layouts, so its seed still depends on "
            "which worker happened to host it"
        )
