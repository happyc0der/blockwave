"""Many pixel environments across processes, with the reward computed centrally.

The split matters for throughput. Workers do what is cheap and parallel —
stepping the engine, rendering a frame, reading the two events off it — and send
back one 88x88 frame per step. The main process stacks frames for the policy and,
only on the steps where a piece landed, encodes those frames and computes the
reward for all of them in one batch on the accelerator.

That keeps the encoder and the world model in one place, out of the workers, and
runs them over a batch rather than one state at a time. Rewards arrive on
placements, so the model runs on a few percent of steps.

No worker touches engine state: `info` carries the game's own numbers for
`evaluate.GameTracker` and nothing else reads it.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PixelBatch:
    frames: np.ndarray     # (N, stack, H, W) uint8 — what the policy sees
    reward: np.ndarray     # (N,) intrinsic, nonzero only where a piece landed
    locked: np.ndarray     # (N,) bool — the per-placement discount clock
    top_out: np.ndarray    # (N,) bool
    info: list[dict]       # evaluation only


def _worker(conn, count: int, seed: int, config: dict) -> None:
    from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
    from ..env.crops import Variant
    from .events import FrameEvents

    envs, detectors, previous = [], [], []
    for index in range(count):
        env = BlockwaveEnv(
            EnvConfig(
                obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW,
                cell_px=config["cell_px"], agent_hz=config["agent_hz"],
                gravity_scale=config["gravity_scale"], frame_stack=1,
            )
        )
        env.reset(seed=seed + index)
        events = FrameEvents(env.layout, env.crop)
        events.calibrate(env._pixels())
        envs.append(env)
        detectors.append(events)
        previous.append(env._pixels())

    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send([env._pixels() for env in envs])
            elif command == "step":
                out = []
                for index, (env, events, action) in enumerate(zip(envs, detectors, payload)):
                    *_, info = env.step(int(action))
                    frame = env._pixels()
                    placed, ended = events.observe(previous[index], frame, int(action))
                    previous[index] = frame
                    out.append((frame, placed, ended, info))
                conn.send(out)
            elif command == "close":
                break
    finally:
        conn.close()


class PixelVecEnv:
    """Vectorised pixel envs. Rewards are supplied by ``reward_fn`` in batch."""

    def __init__(
        self,
        n_envs: int,
        n_workers: int,
        *,
        reward_fn,
        gamma: float = 0.99,
        frame_stack: int = 4,
        seed: int = 0,
        agent_hz: float = 5.0,
        gravity_scale: float = 4.0,
        cell_px: int = 4,
    ) -> None:
        if n_envs % n_workers:
            raise ValueError("n_envs must divide evenly across workers")
        self.n_envs = n_envs
        self.frame_stack = frame_stack
        self.reward_fn = reward_fn
        self.death_charge = reward_fn.death_charge(gamma)
        self._per = n_envs // n_workers
        config = {"cell_px": cell_px, "agent_hz": agent_hz, "gravity_scale": gravity_scale}
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker, args=(child, self._per, seed + w * 10_000, config), daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._stacks: list[deque] = []

    def reset(self) -> np.ndarray:
        for conn in self._conns:
            conn.send(("reset", None))
        frames = [f for conn in self._conns for f in conn.recv()]
        self._stacks = [deque([f] * self.frame_stack, maxlen=self.frame_stack) for f in frames]
        return self._observe()

    def _observe(self) -> np.ndarray:
        return np.stack([np.stack(tuple(stack)) for stack in self._stacks])

    def step(self, actions: np.ndarray) -> PixelBatch:
        for w, conn in enumerate(self._conns):
            conn.send(("step", actions[w * self._per : (w + 1) * self._per].tolist()))
        rows = [r for conn in self._conns for r in conn.recv()]

        frames = [row[0] for row in rows]
        placed = np.array([row[1] for row in rows], dtype=bool)
        ended = np.array([row[2] for row in rows], dtype=bool)
        for stack, frame in zip(self._stacks, frames):
            stack.append(frame)

        reward = np.zeros(self.n_envs, dtype=np.float32)
        # A top-out is charged for the whole dead game; the board on screen is
        # already a fresh one, so there is nothing there to score.
        reward[ended] = self.death_charge
        scored = placed & ~ended
        if scored.any():
            reward[scored] = self.reward_fn(
                self.reward_fn.encode(np.stack([f for f, keep in zip(frames, scored) if keep]))
            )
        return PixelBatch(
            frames=self._observe(),
            reward=reward,
            locked=placed | ended,
            top_out=ended,
            info=[row[3] for row in rows],
        )

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
