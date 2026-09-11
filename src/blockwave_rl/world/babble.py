"""Trying key-press programs to find out what they do.

The board-state reward enumerated futures with the simulator. A pixel agent has
to learn them, so it babbles: for each piece it picks one of the macros at
random, runs it blind, and records what the screen looked like before and after.

Nothing here reads the engine. The piece is known to have resolved because the
NEXT panel shifted, and the game is known to have ended because the playfield
emptied — both from `events.py`, both validated against the engine's own
accounting in `tests/rl/test_world_events.py`.

A macro that ends a game is kept with a flag rather than dropped. "This program
kills me from here" is exactly the outcome a reward about keeping control has to
predict, and after a soft reset the screen shows a fresh board, which is the
opposite of what happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from ..env.crops import Variant
from .events import FrameEvents
from .macros import MACROS, N_MACROS

#: A piece cannot outlive this many decisions, so a macro that somehow never
#: resolves cannot hang the collection.
MAX_STEPS_PER_PIECE = 200


@dataclass(slots=True)
class Babble:
    """What each try produced, in latent space."""

    before: np.ndarray   # (N, D) the screen when the piece entered play
    macro: np.ndarray    # (N,) which program was run
    after: np.ndarray    # (N, D) the screen once it resolved
    died: np.ndarray     # (N,) bool: the program ended the game

    def __len__(self) -> int:
        return len(self.macro)

    def split(self, fraction: float = 0.9, seed: int = 0) -> tuple[Babble, Babble]:
        order = np.random.default_rng(seed).permutation(len(self))
        cut = int(fraction * len(order))
        return self[order[:cut]], self[order[cut:]]

    def __getitem__(self, index) -> Babble:
        return Babble(self.before[index], self.macro[index], self.after[index], self.died[index])


def collect(
    vae,
    n_pieces: int,
    *,
    seed: int = 0,
    agent_hz: float = 5.0,
    gravity_scale: float = 4.0,
    cell_px: int = 4,
    chunk: int = 1024,
    progress: int = 0,
) -> Babble:
    """Run ``n_pieces`` random macros and return the transitions they produced."""
    env = BlockwaveEnv(
        EnvConfig(
            obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=cell_px,
            agent_hz=agent_hz, gravity_scale=gravity_scale, frame_stack=1,
        )
    )
    env.reset(seed=seed)
    events = FrameEvents(env.layout, env.crop)
    events.calibrate(env._pixels())
    rng = np.random.default_rng(seed)

    pending: list[np.ndarray] = []      # frames waiting to be encoded, in pairs
    macros: list[int] = []
    deaths: list[bool] = []
    latents: list[np.ndarray] = []

    def flush() -> None:
        if pending:
            latents.append(vae.embed(np.stack(pending)))
            pending.clear()

    frame = env._pixels()
    while len(macros) < n_pieces:
        macro = int(rng.integers(N_MACROS))
        actions = MACROS[macro].actions
        before = frame
        died = False
        resolved = False
        for step in range(MAX_STEPS_PER_PIECE):
            action = actions[step] if step < len(actions) else int(actions[-1])
            env.step(action)
            current = env._pixels()
            placed, ended = events.observe(frame, current, action)
            frame = current
            died |= ended
            if placed or ended:
                resolved = True
                break
        if not resolved:
            continue
        pending.extend((before, frame))
        macros.append(macro)
        deaths.append(died)
        if len(pending) >= chunk:
            flush()
        if progress and len(macros) % progress == 0:
            print(f"  {len(macros):,}/{n_pieces:,} macros tried", flush=True)
    flush()

    stacked = np.concatenate(latents)
    return Babble(
        before=stacked[0::2].astype(np.float32),
        macro=np.array(macros, dtype=np.int64),
        after=stacked[1::2].astype(np.float32),
        died=np.array(deaths, dtype=bool),
    )
