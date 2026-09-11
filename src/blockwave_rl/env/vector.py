"""Run many environments across processes.

Empowerment is ~95% of an environment step's cost, and it is pure Python, so
the GIL serializes it. Separate processes are the only way to use more than one
core. Each worker owns several environments and steps them as a batch, which
amortizes the pipe round-trip over many envs rather than paying it per env.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Batch:
    grid: np.ndarray        # (N, 2, 24, 10)
    queue: np.ndarray       # (N, Q)
    reward: np.ndarray      # (N,)
    locked: np.ndarray      # (N,) bool — the per-placement discount clock
    top_out: np.ndarray     # (N,) bool
    info: list[dict]        # evaluation only


def _worker(conn, count: int, horizon: int, seed: int) -> None:
    from ..reward.empowerment_env import EmpowermentEnv

    envs = [EmpowermentEnv(horizon=horizon) for _ in range(count)]
    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                obs = [env.reset(seed=seed + i) for i, env in enumerate(envs)]
                conn.send([(g, q) for g, q in obs])
            elif command == "step":
                results = [env.step(int(a)) for env, a in zip(envs, payload)]
                conn.send([(r.grid, r.queue, r.reward, r.locked, r.top_out, r.info) for r in results])
            elif command == "close":
                break
    finally:
        conn.close()


class ProcessVecEnv:
    def __init__(self, n_envs: int, n_workers: int, horizon: int = 2, seed: int = 0) -> None:
        if n_envs % n_workers:
            raise ValueError("n_envs must divide evenly across workers")
        self.n_envs = n_envs
        per = n_envs // n_workers
        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, per, horizon, seed + w * 10_000), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._per = per

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for conn in self._conns:
            conn.send(("reset", None))
        obs = [o for conn in self._conns for o in conn.recv()]
        return np.stack([g for g, _ in obs]), np.stack([q for _, q in obs])

    def step(self, actions: np.ndarray) -> Batch:
        for w, conn in enumerate(self._conns):
            conn.send(("step", actions[w * self._per : (w + 1) * self._per].tolist()))
        rows = [r for conn in self._conns for r in conn.recv()]
        return Batch(
            grid=np.stack([r[0] for r in rows]),
            queue=np.stack([r[1] for r in rows]),
            reward=np.array([r[2] for r in rows], dtype=np.float32),
            locked=np.array([r[3] for r in rows], dtype=bool),
            top_out=np.array([r[4] for r in rows], dtype=bool),
            info=[r[5] for r in rows],
        )

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
