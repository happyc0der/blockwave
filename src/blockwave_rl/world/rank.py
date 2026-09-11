"""Does the pixel reward rank competent play above every exploit?

    python -m blockwave_rl.world.rank --out runs/pixel

The plan's Stage 3, applied to the pixel reward: no training run launches until
a scripted competent player out-scores every hand-built exploit under the
reward, each policy's reward computed exactly as it would be during training —
from frames, through the agent's own learned model, with no engine state
anywhere in the path.

This is the test that matters more than agreement with the exact reward. The
exact reward saturates: most boards have full headroom and score zero, so a
rank correlation across states is mostly measuring ties. What the reward has to
get right is the ordering between *policies*, and that is what this measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
from ..env.crops import Variant
from ..repr.pixel_gate import drift_policy
from ..reward.rollout import EXPLOITS, heuristic
from .empowerment import PixelEmpowerment
from .events import FrameEvents
from .model import MacroModel


def score(policy, reward: PixelEmpowerment, vae, *, placements: int, seed: int, agent_hz: float,
          max_steps: int = 400_000) -> np.ndarray:
    """The reward at every placement of one policy's play."""
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
    frame = env._pixels()
    pending: list[np.ndarray] = []
    values: list[float] = []
    step = 0
    while len(values) + len(pending) < placements and step < max_steps:
        action = policy(env.engine, rng, step)
        env.step(action)
        step += 1
        current = env._pixels()
        placed, ended = events.observe(frame, current, action)
        frame = current
        if placed and not ended:
            pending.append(current)
        if len(pending) >= 256:
            values.extend(reward(vae.embed(np.stack(pending))).tolist())
            pending.clear()
    if pending:
        values.extend(reward(vae.embed(np.stack(pending))).tolist())
    return np.array(values)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("runs/pixel"))
    p.add_argument("--placements", type=int, default=600)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--agent-hz", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import torch

    from ..repr.pixel_vae import PixelVAE

    blob = torch.load(args.out / "vae.pt", map_location="cpu")
    vae = PixelVAE(blob["latent_dim"], blob["size"])
    vae.load_state_dict(blob["state_dict"])
    vae.eval()

    world = torch.load(args.out / "world.pt", map_location="cpu")
    model = MacroModel(world["latent_dim"])
    model.load_state_dict(world["state_dict"])
    reward = PixelEmpowerment(
        model, horizon=world["horizon"], width=world["width"], branch=world["branch"], device=args.device,
    )
    reward._baseline = world["baseline"]

    policies = {"competent": heuristic(), "drift": drift_policy(), **EXPLOITS}
    results: dict[str, dict] = {}
    for name, policy in policies.items():
        pooled = [
            score(policy, reward, vae, placements=args.placements, seed=100 + s, agent_hz=args.agent_hz)
            for s in range(args.seeds)
        ]
        values = np.concatenate([v for v in pooled if len(v)])
        if not len(values):
            continue
        results[name] = {
            "mean": float(values.mean()),
            "sem": float(values.std(ddof=1) / np.sqrt(len(values))),
            "placements": int(len(values)),
        }
        print(f"  {name:<15} {results[name]['mean']:+.4f} ± {results[name]['sem']:.4f}"
              f"  ({results[name]['placements']} placements)", flush=True)

    reference = results["competent"]
    verdict = {}
    for name, row in results.items():
        if name == "competent":
            continue
        gap = reference["mean"] - row["mean"]
        z = gap / np.sqrt(reference["sem"] ** 2 + row["sem"] ** 2)
        verdict[name] = {"margin": round(gap, 4), "z": round(float(z), 2), "beaten": bool(z > 2.5)}
        print(f"  competent over {name:<15} margin {gap:+.4f}  z {z:+.1f}  "
              f"{'ok' if z > 2.5 else 'NOT BEATEN'}", flush=True)

    passed = all(row["beaten"] for row in verdict.values())
    (args.out / "rank.json").write_text(json.dumps({"scores": results, "verdict": verdict, "passed": passed}, indent=2))
    print("RANKING PASSED" if passed else "RANKING FAILED", flush=True)


if __name__ == "__main__":
    main()
