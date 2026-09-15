"""Watch the pixel agent play: live in a window, or recorded to a video.

    python -m blockwave_rl.world.watch_pixel pretrained/pixel/policy.pt --live
    python -m blockwave_rl.world.watch_pixel pretrained/pixel/policy.pt          # writes an MP4

The agent acts exactly as in training — the same crop, the same frame stack, the
same sampled policy — while the picture is drawn at human size from the same
engine state. What the agent sees is the small grey 88x88 crop with the HUD cut
out; what you see is that game with the HUD in. The score on screen is for you.
It never reaches the agent.

Which game: games from the held-out evaluation seed, in order, always. Choosing
the best-looking one would be choosing by score. The recording stops at the end
of the first game; the live view keeps playing until you close the window.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blockwave.render.compositor import Compositor
from blockwave.render.layout import HUMAN_CELL_PX, Layout

from ..env.base import BlockwaveEnv, EnvConfig, N_ACTIONS, ObsMode
from ..env.crops import Variant
from ..evaluate import EVAL_SEED
from .events import FrameEvents

HOLD_END_S = 1.5


@dataclass
class Session:
    """Everything a viewer needs, built once: the agent's env and the human view of it."""

    env: BlockwaveEnv
    policy: object
    events: FrameEvents
    compositor: Compositor
    stack: deque
    seen: np.ndarray
    hz: float

    def decide(self) -> tuple[int, bool]:
        """One agent decision. Returns (action, game ended)."""
        import torch

        with torch.no_grad():
            observation = torch.as_tensor(np.stack(tuple(self.stack))[None])
            action, _, _ = self.policy.act(observation)
        action = int(action)
        self.env.step(action)
        current = self.env._pixels()
        _, ended = self.events.observe(self.seen, current, action)
        self.seen = current
        self.stack.append(current)
        return action, ended

    def frame(self) -> np.ndarray:
        return self.compositor.render(self.env.engine)


def open_session(checkpoint: Path, *, profile: str, seed: int) -> Session:
    import torch

    from ..agents.nets import PixelPolicy

    config = json.loads((checkpoint.parent / "config.json").read_text())
    hz = float(config.get("agent_hz", 5.0))
    stack_size = int(config.get("frame_stack", 4))
    env = BlockwaveEnv(
        EnvConfig(
            obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
            agent_hz=hz, gravity_scale=float(config.get("gravity", 4.0)), frame_stack=1,
        )
    )
    env.reset(seed=seed)
    events = FrameEvents(env.layout, env.crop)
    events.calibrate(env._pixels())

    policy = PixelPolicy((stack_size, 88, 88), N_ACTIONS)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()
    torch.manual_seed(seed)

    seen = env._pixels()
    return Session(
        env=env, policy=policy, events=events,
        compositor=Compositor(Layout(HUMAN_CELL_PX), profile),
        stack=deque([seen] * stack_size, maxlen=stack_size), seen=seen, hz=hz,
    )


def record(checkpoint: Path, out: Path, *, speed: float, profile: str, seed: int, max_decisions: int) -> int:
    """Write one complete game to ``out``. Returns the number of decisions."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write video")
    session = open_session(checkpoint, profile=profile, seed=seed)
    frame = session.frame()
    height, width = frame.shape[:2]
    fps = session.hz * speed
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    encoder.stdin.write(frame.tobytes())
    decisions = 0
    try:
        for _ in range(max_decisions):
            decisions += 1
            _, ended = session.decide()
            if ended:
                break
            frame = session.frame()
            encoder.stdin.write(frame.tobytes())
        for _ in range(int(HOLD_END_S * fps)):
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
    return decisions


def play_live(checkpoint: Path, *, speed: float, profile: str, seed: int, max_decisions: int) -> int:
    """Play in the game's own window, paced to real time. Returns decisions made.

    One decision is one engine tick, so presenting once per decision shows every
    state the game passes through. At 5 Hz that is a deliberately slow picture:
    it is the rate the agent actually plays at. ``speed`` scales it.

    Esc, Q or closing the window stops it. A top-out holds the last frame
    briefly, then the next game starts from the same held-out seed sequence.
    """
    import pygame

    from blockwave.render.display import Display

    session = open_session(checkpoint, profile=profile, seed=seed)
    frame = session.frame()
    height, width = frame.shape[:2]
    display = Display((width, height), title="BLOCKWAVE - the agent is playing")
    clock = pygame.time.Clock()
    rate = session.hz * speed

    decisions = games = 0
    pieces = lines = 0.0
    last_pieces = last_lines = 0.0
    try:
        display.present(frame)
        while decisions < max_decisions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    return decisions
            decisions += 1
            _, ended = session.decide()
            # Per-game tallies from the evaluator's own counters, as positive
            # increments: a top-out soft-resets the engine, so its stats vanish.
            stats = session.env.engine.stats
            pieces += max(0.0, stats.pieces_placed - last_pieces)
            lines += max(0.0, stats.lines - last_lines)
            last_pieces, last_lines = stats.pieces_placed, stats.lines
            if ended:
                games += 1
                print(f"game {games}: {int(pieces)} pieces, {int(lines)} lines, {decisions} decisions so far", flush=True)
                pieces = lines = 0.0
                last_pieces = last_lines = 0.0
                for _ in range(int(HOLD_END_S * rate)):
                    clock.tick(rate)
                continue
            display.present(session.frame())
            clock.tick(rate)
    finally:
        display.close()
    return decisions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--live", action="store_true", help="play in a window instead of writing a video")
    p.add_argument("--out", type=Path, default=None, help="video path (default: next to the checkpoint)")
    p.add_argument("--speed", type=float, default=None,
                   help="multiple of real time (default: 1 live, 2 for video)")
    p.add_argument("--profile", default="arcade")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--max-decisions", type=int, default=20_000)
    args = p.parse_args()

    if args.live:
        speed = 1.0 if args.speed is None else args.speed
        n = play_live(args.checkpoint, speed=speed, profile=args.profile, seed=args.seed, max_decisions=args.max_decisions)
        print(f"{n} decisions")
        return
    speed = 2.0 if args.speed is None else args.speed
    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_game.mp4")
    n = record(args.checkpoint, out, speed=speed, profile=args.profile, seed=args.seed, max_decisions=args.max_decisions)
    print(f"{n} decisions -> {out}")


if __name__ == "__main__":
    main()
