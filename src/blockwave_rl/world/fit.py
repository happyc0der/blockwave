"""Fit the world model and check the reward it produces, before training on it.

    python -m blockwave_rl.world.fit --out runs/pixel

Babble, fit, calibrate, then validate. The validation is the point: the
board-state track can compute this same reward *exactly*, with the simulator, so
the pixel estimate can be checked against ground truth on the same boards before
a single step of RL is spent on it.

That comparison reads engine state and is therefore an offline diagnostic, in
the same category as the representation probes: it never reaches the agent, the
reward or a gradient. It answers one question — does replacing the simulator
with a learned model preserve the ordering the reward is supposed to express?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from ..env.crops import Variant
from ..repr.pixel_gate import drift_policy
from ..reward.empowerment import HorizonEmpowermentReward
from ..reward.rollout import EXPLOITS, heuristic
from .babble import collect
from .empowerment import PixelEmpowerment
from .events import FrameEvents
from .model import report, train


def probe_states(n: int, *, seed: int, agent_hz: float, horizon: int) -> tuple[np.ndarray, list, list]:
    """Frames at placement moments, with the board and queue behind each.

    The board and queue are for the exact reward only, which is the yardstick
    here and nothing else.
    """
    env = BlockwaveEnv(
        EnvConfig(
            obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
            agent_hz=agent_hz, gravity_scale=4.0, frame_stack=1,
        )
    )
    env.reset(seed=seed)
    events = FrameEvents(env.layout, env.crop)
    events.calibrate(env._pixels())
    rng = np.random.default_rng(seed)
    policies = [EXPLOITS["random"], drift_policy(), heuristic()]
    frames, boards, queues = [], [], []
    frame = env._pixels()
    step = 0
    while len(frames) < n:
        policy = policies[(len(frames) // 32) % len(policies)]
        action = policy(env.engine, rng, step)
        env.step(action)
        step += 1
        current = env._pixels()
        placed, ended = events.observe(frame, current, action)
        frame = current
        if not placed or ended:
            continue
        engine = env.engine
        queue = engine.preview(horizon) if engine.piece is None else (
            [engine.piece.type] + engine.preview(horizon - 1)
        )
        frames.append(current)
        boards.append(list(engine.board.rows))
        queues.append(queue[:horizon])
    return np.stack(frames), boards, queues


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("runs/pixel"))
    p.add_argument("--vae", type=Path, default=None,
                   help="where to read vae.pt from (default: --out), so a re-fit can reuse "
                        "an encoder without writing back into its directory")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing world.pt in --out")
    p.add_argument("--pieces", type=int, default=60_000, help="macros to try while babbling")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--branch", type=int, default=12)
    p.add_argument("--probe-states", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent-hz", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import torch

    from ..repr.pixel_vae import PixelVAE

    # A re-fit once wrote a new kernel width straight back over `runs/pixel/world.pt`,
    # which was the reward a shipped agent had been trained against. The agent in
    # `pretrained/` then carried a reward it had never seen. Refuse by default.
    existing = args.out / "world.pt"
    if existing.exists() and not args.force:
        raise SystemExit(
            f"{existing} already exists. A trained agent may depend on it: overwriting it in "
            f"place makes that agent's reward unreproducible.\n"
            f"Write to a new --out (and point --vae at the old one to reuse its encoder), "
            f"or pass --force if you really mean to replace it."
        )

    vae_dir = args.vae or args.out
    args.out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(vae_dir / "vae.pt", map_location="cpu")
    vae = PixelVAE(blob["latent_dim"], blob["size"])
    vae.load_state_dict(blob["state_dict"])
    vae.eval()

    babble_path = args.out / "babble.npz"
    if babble_path.exists():
        stored = np.load(babble_path)
        from .babble import Babble

        data = Babble(stored["before"], stored["macro"], stored["after"], stored["died"])
        print(f"babble: reusing {babble_path} ({len(data):,} macros tried)", flush=True)
    else:
        print(f"babbling {args.pieces:,} macros", flush=True)
        data = collect(vae, args.pieces, seed=args.seed, agent_hz=args.agent_hz, progress=args.pieces // 6)
        np.savez_compressed(
            babble_path, before=data.before, macro=data.macro, after=data.after, died=data.died,
        )
    fit_set, held_out = data.split(0.9, seed=args.seed)
    print(f"  {len(fit_set):,} to fit, {len(held_out):,} held out; {data.died.mean():.1%} ended the game", flush=True)

    model = train(fit_set, epochs=args.epochs, seed=args.seed, device=args.device, log=True)
    quality = report(model, held_out, device=args.device)
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in quality.items()), flush=True)

    # The reward, calibrated on the agent's own states.
    frames, boards, queues = probe_states(
        args.probe_states, seed=args.seed + 7, agent_hz=args.agent_hz, horizon=args.horizon,
    )
    latents = vae.embed(frames)
    empty_env = BlockwaveEnv(
        EnvConfig(obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
                  agent_hz=args.agent_hz, gravity_scale=4.0, frame_stack=1)
    )
    empty_env.reset(seed=args.seed)
    empty_latent = vae.embed(empty_env._pixels()[None])

    reward = PixelEmpowerment(
        model, horizon=args.horizon, branch=args.branch, device=args.device, seed=args.seed,
    )
    # Two futures are the same outcome when the model cannot resolve them apart,
    # and the model's *mean* error is the wrong measure of that: a tail of badly
    # predicted states inflates it, and at that width the count collapses 528
    # predicted futures into fewer than two on a typical board. Ablated at 30M
    # steps, one variable at a time: choosing the width by how much the reward
    # varies across visited states reached 0.0635 lines per piece where the mean
    # error's width reached ~0.053.
    fitted = float(np.sqrt(quality["mse"] * data.before.shape[1]))
    width = reward.choose_width(latents[:256], empty_latent, width=fitted)
    print(f"  kernel width {fitted:.3f} fitted -> {width:.3f} chosen", flush=True)

    # The yardstick: the same reward, counted exactly with the simulator.
    exact_fn = HorizonEmpowermentReward(args.horizon)
    exact = np.array([exact_fn(board, queue) for board, queue in zip(boards, queues)])
    pixel = reward(latents)

    agreement = {
        "spearman": round(spearman(pixel, exact), 4),
        "pearson": round(float(np.corrcoef(pixel, exact)[0, 1]), 4),
        "kernel_width": round(reward.width, 4),
        "baseline": round(reward.baseline, 4),
        "pixel_mean": round(float(pixel.mean()), 4),
        "pixel_std": round(float(pixel.std()), 4),
        "exact_mean": round(float(exact.mean()), 4),
        "exact_std": round(float(exact.std()), 4),
        "states": int(len(exact)),
    }
    torch.save({"state_dict": model.state_dict(), "latent_dim": int(data.before.shape[1]),
                "width": reward.width, "baseline": reward.baseline,
                "horizon": args.horizon, "branch": args.branch}, args.out / "world.pt")
    (args.out / "world.json").write_text(json.dumps({"model": quality, "agreement": agreement}, indent=2))
    print("  agreement with the exact reward: " + "  ".join(f"{k} {v}" for k, v in agreement.items()), flush=True)


if __name__ == "__main__":
    main()
