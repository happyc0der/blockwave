"""Watch the pixel agent play: one game, rendered for people.

    python -m blockwave_rl.world.watch_pixel runs/pixel_ppo_seed0/ckpt_01600.pt

The agent acts exactly as in training — the same crop, the same frame stack, the
same sampled policy — while the video is drawn at human size from the same
engine state. What the agent sees is the small grey 88x88 crop; what the video
shows is that game.

Which game: the first complete one from the held-out evaluation seed, always.
Choosing the best-looking game would be choosing by score.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

from blockwave.render.compositor import Compositor
from blockwave.render.layout import HUMAN_CELL_PX, Layout

from ..env.base import BlockwaveEnv, EnvConfig, N_ACTIONS, ObsMode
from ..env.crops import Variant
from ..evaluate import EVAL_SEED
from .events import FrameEvents

HOLD_END_S = 1.5


def record(checkpoint: Path, out: Path, *, speed: float, profile: str, seed: int, max_decisions: int) -> int:
    import torch

    from ..agents.nets import PixelPolicy

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write video")
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
    stack = deque([seen] * stack_size, maxlen=stack_size)
    compositor = Compositor(Layout(HUMAN_CELL_PX), profile)
    frame = compositor.render(env.engine)
    height, width = frame.shape[:2]
    fps = hz * speed
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
            with torch.no_grad():
                observation = torch.as_tensor(np.stack(tuple(stack))[None])
                action, _, _ = policy.act(observation)
            env.step(int(action))
            current = env._pixels()
            _, ended = events.observe(seen, current, int(action))
            seen = current
            stack.append(current)
            if ended:
                break
            frame = compositor.render(env.engine)
            encoder.stdin.write(frame.tobytes())
        for _ in range(int(HOLD_END_S * fps)):
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
    return decisions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--profile", default="arcade")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--max-decisions", type=int, default=20_000)
    args = p.parse_args()
    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_game.mp4")
    print(f"{record(args.checkpoint, out, speed=args.speed, profile=args.profile, seed=args.seed, max_decisions=args.max_decisions)} decisions -> {out}")


if __name__ == "__main__":
    main()
