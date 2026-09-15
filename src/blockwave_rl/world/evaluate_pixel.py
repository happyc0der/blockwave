"""Evaluate pixel checkpoints the same way the board-state ones were evaluated.

    python -m blockwave_rl.world.evaluate_pixel runs/pixel_ppo_seed0 --reward runs/pixel

Held-out seeds, complete games only, every checkpoint on the schedule, and no
checkpoint chosen by what it scores. The measurement is `evaluate.GameTracker`,
unchanged, so pixel and board-state numbers are directly comparable.

The baselines are measured here too, because they belong to the dynamics rather
than to the observation: a policy that never hard-drops plays the same game
whether it is reading pixels or occupancy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluate import BASELINES, EVAL_SEED, GameTracker, _ratio, baseline_actor
from .pixel_vector import PixelVecEnv
from .reward import FrameReward


def play(
    act,
    reward_fn: FrameReward,
    *,
    steps_per_env: int,
    envs: int,
    workers: int,
    seed: int = EVAL_SEED,
    agent_hz: float = 5.0,
    gravity_scale: float = 4.0,
    frame_stack: int = 4,
    gamma: float = 0.99,
) -> dict[str, float]:
    vec = PixelVecEnv(
        envs, workers, reward_fn=reward_fn, gamma=gamma, frame_stack=frame_stack,
        seed=seed, agent_hz=agent_hz, gravity_scale=gravity_scale,
    )
    try:
        frames = vec.reset()
        tracker = GameTracker(envs)
        for _ in range(steps_per_env):
            batch = vec.step(act(frames))
            tracker.update(batch.info, batch.reward, batch.locked)
            frames = batch.frames
    finally:
        vec.close()
    window = {**tracker.summary(), "intrinsic_per_placement": _ratio(tracker.env_intrinsic, tracker.env_placements)}
    return {
        **tracker.complete_games(),
        "pieces_per_1k_steps": window["pieces_per_1k_steps"],
        **{f"window_{k}": v for k, v in window.items()},
        "steps": tracker.steps,
    }


def checkpoint_actor(path: Path, device: str, obs_shape):
    import torch

    from ..agents.nets import PixelPolicy

    policy = PixelPolicy(obs_shape, 8).to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()

    def act(frames: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action, _, _ = policy.act(torch.as_tensor(frames, device=device))
        return action.cpu().numpy()

    return act


def _row(name: str, r: dict[str, float]) -> str:
    # Complete games are measured between an env's first and last top-out, so a
    # window too short for two deaths yields nothing to measure. A good agent
    # needs a longer one than a bad agent: say so rather than printing nan.
    if not r["games"]:
        return (
            f"{name:>26s}  no complete games in this window -- raise --steps-per-env "
            f"(default 20000; {r['steps']:,} env-steps total here)"
        )
    return (
        f"{name:>26s}  lines/pc {r['lines_per_piece']:.4f} ±{r['lines_per_piece_se']:.4f}  "
        f"deaths/pc {r['top_outs_per_piece']:.4f}  pieces/game {r['pieces_per_game']:5.1f}  "
        f"intrinsic/pl {r['intrinsic_per_placement']:+.3f}  games {r['games']:.0f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("targets", nargs="+", help="run directories, or a baseline name: " + ", ".join(BASELINES))
    p.add_argument("--reward", default="runs/pixel")
    p.add_argument("--steps-per-env", type=int, default=20_000)
    p.add_argument("--envs", type=int, default=48)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--last", type=int, default=0,
                   help="evaluate only the final N checkpoints (0 = all)")
    p.add_argument("--append", action="store_true",
                   help="append to eval.jsonl instead of replacing it")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1)
    # The policy samples its actions rather than taking the argmax, so without a
    # fixed torch seed two evaluations of the same checkpoint play different
    # games and the published number cannot be reproduced. Seeded from --seed so
    # the env seeds and the action draws move together.
    torch.manual_seed(args.seed)
    reward_fn = FrameReward.load(args.reward, device=device)

    for target in args.targets:
        if target in BASELINES:
            actor = baseline_actor(target, args.seed)
            result = play(
                lambda frames: actor(frames, None), reward_fn,
                steps_per_env=args.steps_per_env, envs=args.envs, workers=args.workers, seed=args.seed,
            )
            print(_row(f"{target} (pixel dynamics)", result), flush=True)
            out = Path("runs") / "baselines"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{target}_pixel.json").write_text(json.dumps(result, indent=2))
            continue

        run = Path(target)
        config = json.loads((run / "config.json").read_text())
        # `ckpt_*.pt` is what training writes; `policy.pt` is what a shipped
        # agent is called, so a pretrained directory evaluates like a run.
        checkpoints = sorted(run.glob("ckpt_*.pt")) or sorted(run.glob("policy.pt"))
        checkpoints = checkpoints[args.every - 1 :: args.every] if args.every > 1 else checkpoints
        if args.last:
            checkpoints = checkpoints[-args.last :]
        stack = int(config.get("frame_stack", 4))
        with (run / "eval.jsonl").open("a" if args.append else "w") as log:
            for checkpoint in checkpoints:
                result = play(
                    checkpoint_actor(checkpoint, device, (stack, 88, 88)), reward_fn,
                    steps_per_env=args.steps_per_env, envs=args.envs, workers=args.workers,
                    seed=args.seed, agent_hz=config.get("agent_hz", 5.0),
                    gravity_scale=config.get("gravity", 4.0), frame_stack=stack,
                )
                result["checkpoint"] = checkpoint.name
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(_row(f"{run.name}/{checkpoint.stem}", result), flush=True)


if __name__ == "__main__":
    main()
