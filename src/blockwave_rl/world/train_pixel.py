"""Train PPO on the pixel reward: the deliverable.

    python -m blockwave_rl.world.train_pixel --out runs/pixel_ppo --reward runs/pixel

The agent sees a stack of cropped frames and presses one key per decision. Its
reward is computed from those same frames, through the encoder and world model
it fitted to its own experience — no engine state anywhere in the path. Game
score is logged by `evaluate.GameTracker` as a yardstick and enters nothing:
not the reward, not a gradient, not the choice of checkpoint.

The learner is the board-state track's, unchanged: PPO with the discount clock
ticking per placement rather than per step, and a top-out charged for the whole
dead game. Everything learned there about how to train this reward carries over;
what is new is that the reward no longer borrows the simulator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..agents.nets import PixelPolicy
from ..agents.ppo import PPOConfig, RunningStd, gae, update
from ..evaluate import GameTracker
from ..train import lr_scale
from .pixel_vector import PixelVecEnv
from .reward import FrameReward


def _device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    args.device = _device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reward_fn = FrameReward.load(args.reward, device=args.device)
    envs = PixelVecEnv(
        args.envs, args.workers, reward_fn=reward_fn, gamma=args.gamma, frame_stack=args.frame_stack,
        seed=args.seed, agent_hz=args.agent_hz, gravity_scale=args.gravity,
    )
    frames = envs.reset()
    policy = PixelPolicy(frames.shape[1:], 8).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    config = PPOConfig(gamma=args.gamma, lam=args.lam, lam_step=args.lam_step, entropy=args.entropy, lr=args.lr)
    scale = RunningStd(args.envs, args.gamma)
    tracker = GameTracker(args.envs)
    log = (out / "log.jsonl").open("w")

    T, N = args.rollout, args.envs
    total_updates = args.steps // (T * N)
    started = time.perf_counter()

    for step in range(1, total_updates + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale(step, total_updates, args.lr_decay_start)

        buf = {k: [] for k in ("frames", "actions", "logp", "values", "rewards", "locked")}
        intrinsic, placements, deaths, alive = 0.0, 0, 0, 0.0
        for _ in range(T):
            observation = torch.as_tensor(frames, device=args.device)
            with torch.no_grad():
                action, logp, value = policy.act(observation)
            batch = envs.step(action.cpu().numpy())
            tracker.update(batch.info)   # evaluation only

            scale.update(batch.reward, batch.locked)
            buf["frames"].append(frames)
            buf["actions"].append(action.cpu().numpy())
            buf["logp"].append(logp.cpu().numpy())
            buf["values"].append(value.cpu().numpy())
            buf["rewards"].append(batch.reward / scale.std)
            buf["locked"].append(batch.locked)
            intrinsic += float(batch.reward[batch.locked].sum())
            placements += int(batch.locked.sum())
            deaths += int(batch.top_out.sum())
            alive += float(batch.reward[batch.locked & ~batch.top_out].sum())
            frames = batch.frames

        with torch.no_grad():
            _, next_value = policy(torch.as_tensor(frames, device=args.device))
        arr = {k: np.asarray(v) for k, v in buf.items()}
        adv, ret = gae(
            arr["rewards"], arr["values"], next_value.cpu().numpy(), arr["locked"],
            args.gamma, args.lam, args.lam_step,
        )

        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), device=args.device)  # noqa: E731
        stats = update(
            policy, optimizer,
            {
                "frames": flat(arr["frames"]), "actions": flat(arr["actions"]).long(),
                "logp": flat(arr["logp"]), "adv": flat(adv), "ret": flat(ret),
            },
            config,
            inputs=("frames",),
        )

        record = {
            "update": step,
            "env_steps": step * T * N,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "intrinsic_per_placement": intrinsic / max(placements, 1),
            "alive_per_placement": alive / max(placements - deaths, 1),
            "deaths_per_1k_steps": 1000.0 * deaths / (T * N),
            **{f"loss_{k}": v for k, v in stats.items()},
            **{f"game_{k}": v for k, v in tracker.summary().items()},
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        tracker.reset_window()

        if step % args.save_every == 0 or step == total_updates:
            torch.save(policy.state_dict(), out / f"ckpt_{step:05d}.pt")
        if step % args.print_every == 0 or step == 1:
            print(
                f"upd {step:5d}/{total_updates}  steps {record['env_steps']:>10,}  "
                f"alive/pl {record['alive_per_placement']:+.3f}  "
                f"deaths/pc {record['game_top_outs_per_piece']:.4f}  "
                f"lines/pc {record['game_lines_per_piece']:.4f}  "
                f"ent {stats['entropy']:.2f}  {record['elapsed_s']:.0f}s",
                flush=True,
            )

    log.close()
    envs.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="runs/pixel_ppo")
    p.add_argument("--reward", default="runs/pixel", help="where vae.pt and world.pt live")
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--envs", type=int, default=48)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lam-step", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-decay-start", type=float, default=0.0)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--agent-hz", type=float, default=5.0)
    p.add_argument("--gravity", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=25)
    train(p.parse_args())


if __name__ == "__main__":
    main()
