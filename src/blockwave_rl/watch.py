"""Watch a trained agent play: one game, rendered to a video for people.

The agent acts exactly as it did in training: the same environment, the same
decision rate and fall speed (read from the run's config), the same sampled
policy. Each decision is drawn once through the game's own compositor at human
size, so the video shows the dynamics the agent actually lived in. It shows the
full HUD. The agent never saw any of it; it acts on board state.

Which game: the first complete game from the held-out evaluation seed, always.
Choosing the best-looking game would be choosing by score.

    python -m blockwave_rl.watch runs/abs5hz_long_seed0/ckpt_04000.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from blockwave.render.compositor import Compositor
from blockwave.render.layout import HUMAN_CELL_PX, Layout

from .evaluate import EVAL_SEED, checkpoint_actor
from .reward.empowerment_env import EmpowermentEnv

#: Seconds the last frame is held, so the end of the game is visible.
HOLD_END_S = 1.5


def record(checkpoint: Path, out: Path, *, speed: float, profile: str, seed: int, max_decisions: int) -> int:
    """Play one game and write it to ``out``. Returns the number of decisions."""
    import torch

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write video")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    hz = config.get("agent_hz", 20.0)
    env = EmpowermentEnv(
        horizon=config["horizon"], agent_hz=hz, gravity_scale=config.get("gravity", 4.0),
    )
    act = checkpoint_actor(checkpoint, "cpu", config["horizon"])
    torch.manual_seed(seed)
    grid, queue = env.reset(seed=seed)

    compositor = Compositor(Layout(HUMAN_CELL_PX), profile)
    frame = compositor.render(env.env.engine)
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
            step = env.step(int(act(grid[None], queue[None])[0]))
            # A top-out soft-resets inside step(), so the board on screen would
            # already be a fresh one: stop on the last frame of the game.
            if step.top_out:
                break
            frame = compositor.render(env.env.engine)
            encoder.stdin.write(frame.tobytes())
            grid, queue = step.grid, step.queue
        for _ in range(int(HOLD_END_S * fps)):
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
    return decisions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: next to the checkpoint")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed relative to real time")
    p.add_argument("--profile", default="arcade", help="visual profile; the agent never saw pixels")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--max-decisions", type=int, default=20_000)
    args = p.parse_args()
    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_game.mp4")
    decisions = record(
        args.checkpoint, out, speed=args.speed, profile=args.profile, seed=args.seed, max_decisions=args.max_decisions,
    )
    print(f"{decisions} decisions -> {out}")


if __name__ == "__main__":
    main()
