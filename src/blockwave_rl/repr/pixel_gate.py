"""Stage 2 for pixels: build a corpus, fit an encoder, and gate it on probes.

    python -m blockwave_rl.repr.pixel_gate --out runs/pixel

Nothing downstream may start until this passes. The thresholds in `probes.py`
were set before any pixel result existed, and they are the plan's: a *linear*
readout of the frozen latents must recover max height (R^2 >= 0.80), holes
(0.60) and bumpiness (0.50) on held-out frames. The probe targets come from
engine state, which is exactly why this is an offline diagnostic and never a
reward: it answers one question, before RL, about whether the latent space
represents what a reward about board quality would have to be about.

The corpus is a deliberate mixture, because a representation gap is
indistinguishable from an objective's opinion. It covers what the agent will
actually visit: random play, the drift policy PPO discovers first, competent
play, and near-empty boards after a reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..reward.rollout import EXPLOITS, heuristic
from . import probes
from .corpus import build_frames


def drift_policy():
    """Uniform over every key except HARD_DROP: what PPO learns first."""
    choices = np.array([0, 1, 2, 3, 5, 6, 7])

    def policy(engine, rng, t):
        return int(rng.choice(choices))

    return policy


def sources(total: int) -> dict:
    return {
        "random": (EXPLOITS["random"], int(0.30 * total), False),
        "drift": (drift_policy(), int(0.20 * total), False),
        "competent": (heuristic(), int(0.30 * total), False),
        "early": (drift_policy(), int(0.20 * total), True),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("runs/pixel"))
    p.add_argument("--frames", type=int, default=60_000)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--agent-hz", type=float, default=5.0)
    args = p.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    corpus_path = args.out / "corpus.npz"
    if corpus_path.exists():
        blob = np.load(corpus_path)
        frames, boards, is_lock, source = (blob[k] for k in ("frames", "boards", "is_lock", "source"))
        print(f"corpus: reusing {corpus_path} ({len(frames):,} frames)", flush=True)
    else:
        names = list(sources(args.frames))
        frames, boards, is_lock, source = build_frames(
            sources(args.frames), seed=args.seed, agent_hz=args.agent_hz,
        )
        np.savez_compressed(
            corpus_path, frames=frames, boards=boards, is_lock=is_lock, source=source,
        )
        counts = ", ".join(f"{n} {int((source == i).sum()):,}" for i, n in enumerate(names))
        print(f"corpus: {len(frames):,} frames ({counts}); {int(is_lock.sum()):,} on lock events", flush=True)

    from .pixel_vae import reconstruction_error, train

    print(f"training the encoder on {device}", flush=True)
    model = train(
        frames, latent_dim=args.latent_dim, epochs=args.epochs, beta=args.beta,
        seed=args.seed, device=device, log=True,
    )
    torch.save({"state_dict": model.state_dict(), "latent_dim": args.latent_dim,
                "size": int(frames.shape[-1])}, args.out / "vae.pt")

    # Probes read the settled board, so they use the frames taken at lock events.
    latents = model.embed(frames)
    lock = np.flatnonzero(is_lock)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(lock)
    split = int(0.8 * len(shuffled))
    train_idx, test_idx = shuffled[:split], shuffled[split:]
    index = {int(i): k for k, i in enumerate(lock)}
    results = probes.evaluate(
        latents[lock], boards[lock],
        np.array([index[int(i)] for i in train_idx]),
        np.array([index[int(i)] for i in test_idx]),
    )

    per_source = {
        f"recon_{int(i)}": round(reconstruction_error(model, frames[source == i][:2000]), 4)
        for i in np.unique(source)
    }
    passed = all(results[k] >= probes.THRESHOLDS[k] for k in probes.THRESHOLDS)
    report = {
        "probes": {k: round(v, 4) for k, v in results.items()},
        "thresholds": probes.THRESHOLDS,
        "passed": passed,
        "frames": int(len(frames)),
        "lock_frames": int(is_lock.sum()),
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "beta": args.beta,
        "seed": args.seed,
        **per_source,
    }
    (args.out / "probes.json").write_text(json.dumps(report, indent=2))

    for name, threshold in probes.THRESHOLDS.items():
        mark = "PASS" if results[name] >= threshold else "FAIL"
        print(f"  {name:<11} R^2 {results[name]:.3f}  (threshold {threshold:.2f})  {mark}")
    print("  reconstruction error per source:", per_source)
    print("GATE PASSED" if passed else "GATE FAILED", flush=True)


if __name__ == "__main__":
    main()
