"""Train PPO on the board-state empowerment track.

The learner sees the board, the active piece and the queue, and is rewarded
only by queue-corrected empowerment. Game performance is logged through
`evaluate.GameTracker` purely as a yardstick: it never enters a gradient, and
checkpoints are saved on a fixed schedule, never selected by it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agents.nets import BoardPolicy
from .agents.ppo import PPOConfig, RunningStd, gae, update
from .env.base import N_ACTIONS
from .env.vector import ProcessVecEnv
from .evaluate import GameTracker


def _device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def lr_scale(step: int, total: int, decay_start: float = 0.0) -> float:
    """Learning-rate multiplier for update ``step`` (1-based) of ``total``.

    Held at 1 until ``decay_start`` of the run has passed, then linear to zero.
    ``decay_start=0`` is plain linear decay from the first update, which is what
    every run before this option used. The 10M and 50M runs both flattened
    exactly as their rate approached zero, so a run that asks where learning
    saturates has to hold the rate up first.
    """
    frac = (step - 1) / total
    if frac < decay_start:
        return 1.0
    return 1.0 - (frac - decay_start) / (1.0 - decay_start)


def train(args: argparse.Namespace) -> None:
    # One thread here: the workers already occupy the cores during collection,
    # and on MPS the update does not need CPU threads anyway.
    torch.set_num_threads(1)
    args.device = _device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = ProcessVecEnv(
        args.envs, args.workers, horizon=args.horizon, seed=args.seed, death=args.death, gamma=args.gamma,
        agent_hz=args.agent_hz, gravity_scale=args.gravity,
    )
    grid, queue = envs.reset()
    policy = BoardPolicy(grid.shape[1:], queue.shape[1], N_ACTIONS).to(args.device)
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

        buf = {k: [] for k in ("grid", "queue", "actions", "logp", "values", "rewards", "locked")}
        intrinsic, placements, deaths, alive = 0.0, 0, 0, 0.0
        for _ in range(T):
            g = torch.as_tensor(grid, device=args.device)
            q = torch.as_tensor(queue, device=args.device)
            with torch.no_grad():
                action, logp, value = policy.act(g, q)
            batch = envs.step(action.cpu().numpy())
            tracker.update(batch.info)   # evaluation only

            scale.update(batch.reward, batch.locked)
            buf["grid"].append(grid)
            buf["queue"].append(queue)
            buf["actions"].append(action.cpu().numpy())
            buf["logp"].append(logp.cpu().numpy())
            buf["values"].append(value.cpu().numpy())
            buf["rewards"].append(batch.reward / scale.std)
            buf["locked"].append(batch.locked)
            intrinsic += float(batch.reward[batch.locked].sum())
            placements += int(batch.locked.sum())
            deaths += int(batch.top_out.sum())
            alive += float(batch.reward[batch.locked & ~batch.top_out].sum())
            grid, queue = batch.grid, batch.queue

        with torch.no_grad():
            _, next_value = policy(torch.as_tensor(grid, device=args.device), torch.as_tensor(queue, device=args.device))
        arr = {k: np.asarray(v) for k, v in buf.items()}
        adv, ret = gae(
            arr["rewards"], arr["values"], next_value.cpu().numpy(), arr["locked"],
            args.gamma, args.lam, args.lam_step,
        )

        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), device=args.device)
        stats = update(
            policy, optimizer,
            {
                "grid": flat(arr["grid"]), "queue": flat(arr["queue"]),
                "actions": flat(arr["actions"]).long(), "logp": flat(arr["logp"]),
                "adv": flat(adv), "ret": flat(ret),
            },
            config,
        )

        record = {
            "update": step,
            "env_steps": step * T * N,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "intrinsic_per_placement": intrinsic / max(placements, 1),
            # Excludes top-outs, so it compares across death accountings.
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
                f"upd {step:4d}/{total_updates}  steps {record['env_steps']:>9,}  "
                f"intrinsic/pl {record['intrinsic_per_placement']:+.3f}  "
                f"alive/pl {record['alive_per_placement']:+.3f}  "
                f"deaths/pc {record['game_top_outs_per_piece']:.4f}  "
                f"lines/pc {record['game_lines_per_piece']:.4f}  "
                f"ent {stats['entropy']:.2f}  {record['elapsed_s']:.0f}s",
                flush=True,
            )

    log.close()
    envs.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/emp_ppo")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--envs", type=int, default=96)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lam-step", type=float, default=1.0)
    p.add_argument("--death", choices=("absorbing", "one_step"), default="absorbing")
    # Decisions per second and fall speed. Neither gives the agent information;
    # together they set how many decisions a piece lives, i.e. the length of the
    # credit-assignment chain.
    p.add_argument("--agent-hz", type=float, default=20.0)
    p.add_argument("--gravity", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-decay-start", type=float, default=0.0,
                   help="fraction of the run to hold the learning rate before decaying linearly to zero")
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    # MPS: the PPO update is ~26x faster than on CPU (0.48s vs 12.9s). Not
    # bit-deterministic, unlike the environment, so runs are reproducible only
    # approximately.
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--print-every", type=int, default=10)
    train(p.parse_args())


if __name__ == "__main__":
    main()
