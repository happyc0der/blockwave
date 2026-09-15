"""Sharpen the reward: train the encoder to keep what the count depends on.

    python -m blockwave_rl.world.sharpen --out runs/pixel_sharp

The 150M run found the ceiling, and it was not compute. The reward counts
futures the model can tell apart, and the first model told which of 44 programs
ran only 19% of the time. Distinguishing a disaster from a reasonable placement
is inside that resolution; refining good play into better play is not.

The first encoder was trained to reconstruct frames, so it kept whatever makes a
picture look right — not necessarily what separates one placement from another.
Four cells out of two hundred can be the whole difference, and a reconstruction
loss barely notices them.

So the encoder is trained here with three jobs at once:

* **reconstruct** the frame, which keeps it honest and keeps the probes meaningful;
* **inverse model** — say which program ran, given the frames before and after.
  This is the one that matters: it cannot be done without representing exactly
  the differences between programs' outcomes, which is what the count needs.
  Curiosity-driven work uses the same trick for the same reason;
* **forward model** — predict the outcome of a program, which is the model the
  reward uses anyway, now learning against features shaped to support it. With
  ``--forward-loss retrieval`` it is trained on the thing the reward actually
  needs: from one state, the true outcome must land nearer the prediction for the
  program that ran than for any of the other 43. Squared error alone does not ask
  for that, and a model can have a fine average error while its 44 predictions
  sit in a huddle — which is what a first attempt at sharpening produced.

What it saves are the same two artifacts as before, in the same formats, so the
reward, the gates, training and evaluation all read them unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..repr import probes
from ..repr.pixel_vae import PixelVAE, as_input
from .babble import Babble, collect_to_disk
from .macros import N_MACROS
from .model import MacroModel, report


class Inverse(nn.Module):
    """Which program turned this screen into that one?"""

    def __init__(self, latent_dim: int, hidden: int = 256, n_macros: int = N_MACROS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_macros),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([before, after], dim=1))


def train_jointly(
    before: np.ndarray,
    after: np.ndarray,
    macro: np.ndarray,
    died: np.ndarray,
    *,
    latent_dim: int = 64,
    epochs: int = 8,
    batch: int = 256,
    lr: float = 1e-3,
    beta: float = 0.1,
    # Weighted as curiosity work weights them, and for the same reason: the
    # forward loss is minimised by collapsing every latent to one point, so it
    # must not dominate. Reconstruction and the inverse model both push back.
    inverse_weight: float = 1.0,
    forward_weight: float = 0.2,
    forward_loss: str = "retrieval",
    seed: int = 0,
    device: str = "cpu",
    log: bool = True,
) -> tuple[PixelVAE, MacroModel, Inverse]:
    torch.manual_seed(seed)
    size = before.shape[-1]
    encoder = PixelVAE(latent_dim, size=size).to(device)
    forward = MacroModel(latent_dim).to(device)
    inverse = Inverse(latent_dim).to(device)
    parameters = list(encoder.parameters()) + list(forward.parameters()) + list(inverse.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    generator = torch.Generator().manual_seed(seed)

    macro_t = torch.as_tensor(macro, device=device)
    died_t = torch.as_tensor(died.astype(np.float32), device=device)

    for epoch in range(epochs):
        order = torch.randperm(len(macro), generator=generator).numpy()
        totals = np.zeros(4)
        seen = 0
        for start in range(0, len(order), batch):
            index = np.sort(order[start : start + batch])   # sorted: memmap reads stay sequential
            x0 = as_input(before[index]).to(device)
            x1 = as_input(after[index]).to(device)
            m = macro_t[index]
            d = died_t[index]

            logits0, mu0, logvar0 = encoder(x0)
            logits1, mu1, logvar1 = encoder(x1)
            recon = sum(
                nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(index)
                for logits, x in ((logits0, x0), (logits1, x1))
            )
            kl = sum(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(index)
                for mu, logvar in ((mu0, logvar0), (mu1, logvar1))
            )

            identified = nn.functional.cross_entropy(inverse(mu0, mu1), m)

            predicted, death_logit = forward(mu0, m)
            alive = 1.0 - d
            # The target is the encoder's own latent, detached: the forward model
            # chases the representation, it does not get to flatten it.
            error = ((predicted - mu1.detach()) ** 2).mean(dim=1)
            fit = (error * alive).sum() / alive.sum().clamp(min=1.0)
            if forward_loss == "retrieval":
                # Every program's prediction from this same state, scored by how
                # close it lands to what actually happened. Being right on
                # average is not enough: the right program has to win.
                # Built here rather than through `outcomes`, which runs under
                # no_grad: this one has to carry gradients.
                every = torch.arange(forward.n_macros, device=device)
                candidates, _ = forward(
                    mu0.repeat_interleave(forward.n_macros, dim=0), every.repeat(len(index)),
                )
                candidates = candidates.view(len(index), forward.n_macros, -1)
                gap = ((candidates - mu1.detach().unsqueeze(1)) ** 2).mean(dim=2)
                # Scaled by the typical gap, which is the only scale available:
                # with a fixed temperature these logits sit within a fraction of
                # each other, the softmax is flat, and the gradient vanishes —
                # measured, it stayed at chance (ln 44) for two whole epochs.
                temperature = gap.mean().detach().clamp(min=1e-6)
                picked = nn.functional.cross_entropy(-gap / temperature, m, reduction="none")
                fit = fit + (picked * alive).sum() / alive.sum().clamp(min=1.0)
            fatal = nn.functional.binary_cross_entropy_with_logits(death_logit, d)

            loss = recon + beta * kl + inverse_weight * identified + forward_weight * (fit + fatal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([float(recon.detach()), float(identified.detach()), float(fit.detach()), float(fatal.detach())]) * len(index)
            seen += len(index)
        if log:
            r, i, f, dth = totals / seen
            print(f"  epoch {epoch + 1}/{epochs}  recon {r:.1f}  identify {i:.3f}  forward {f:.4f}  fatal {dth:.3f}", flush=True)

    # Back to the CPU: everything downstream (the report, the probes, the
    # calibration) works there, and the artifacts are loaded per device anyway.
    return encoder.eval().cpu(), forward.eval().cpu(), inverse.eval().cpu()


def probe_gate(encoder, *, seed: int, n: int = 3000) -> dict[str, float]:
    """Re-run the representation gate: do the sharper latents still encode the board?

    Reconstruction is no longer the only job the encoder has, so this is not a
    formality — features shaped to separate programs could in principle drop what
    a reward about board quality needs. The boards come from a fresh rollout
    rather than from the babble store, which keeps only pixels — which is why
    this takes no frames: it has to generate states whose true board it knows.
    """
    from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
    from ..env.crops import Variant
    from ..repr.pixel_gate import drift_policy
    from .events import FrameEvents

    env = BlockwaveEnv(
        EnvConfig(obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
                  agent_hz=5.0, gravity_scale=4.0, frame_stack=1)
    )
    env.reset(seed=seed + 99)
    events = FrameEvents(env.layout, env.crop)
    events.calibrate(env._pixels())
    policy = drift_policy()
    rng = np.random.default_rng(seed)
    frames, boards = [], []
    frame = env._pixels()
    step = 0
    while len(frames) < n and step < 400_000:
        action = policy(env.engine, rng, step)
        env.step(action)
        step += 1
        current = env._pixels()
        placed, ended = events.observe(frame, current, action)
        frame = current
        if placed and not ended:
            frames.append(current)
            boards.append(env.occupancy().copy())   # diagnostic only
    latents = encoder.embed(np.stack(frames))
    order = np.random.default_rng(seed).permutation(len(latents))
    cut = int(0.8 * len(order))
    return {k: round(v, 4) for k, v in probes.evaluate(latents, np.stack(boards), order[:cut], order[cut:]).items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("runs/pixel_sharp"))
    p.add_argument("--pieces", type=int, default=120_000)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--forward-weight", type=float, default=0.2)
    p.add_argument("--forward-loss", choices=("mse", "retrieval"), default="retrieval")
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--branch", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent-hz", type=float, default=5.0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    store = args.out / "babble"
    if (store / "after.npy").exists():
        before = np.load(store / "before.npy", mmap_mode="r")
        after = np.load(store / "after.npy", mmap_mode="r")
        macro = np.load(store / "macro.npy")
        died = np.load(store / "died.npy")
        print(f"babble: reusing {store} ({len(macro):,} macros tried)", flush=True)
    else:
        print(f"babbling {args.pieces:,} macros, frames kept", flush=True)
        before, after, macro, died = collect_to_disk(
            store, args.pieces, seed=args.seed, agent_hz=args.agent_hz, progress=args.pieces // 6,
        )

    cut = int(0.9 * len(macro))
    print(f"  {cut:,} to fit, {len(macro) - cut:,} held out; {died.mean():.1%} ended the game", flush=True)
    encoder, forward, inverse = train_jointly(
        before[:cut], after[:cut], macro[:cut], died[:cut],
        latent_dim=args.latent_dim, epochs=args.epochs, seed=args.seed, device=device,
        forward_weight=args.forward_weight, forward_loss=args.forward_loss,
    )

    # Held-out quality, measured exactly as the first model's was.
    held = Babble(
        encoder.embed(np.asarray(before[cut:])), macro[cut:],
        encoder.embed(np.asarray(after[cut:])), died[cut:],
    )
    quality = report(forward, held, device="cpu")
    with torch.no_grad():
        identified = inverse(
            torch.as_tensor(held.before), torch.as_tensor(held.after)
        ).argmax(dim=1).numpy()
    quality["inverse_top1"] = float((identified == held.macro).mean())
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in quality.items()), flush=True)

    # The reward is relative to an empty board, so measure that board's value
    # with the new model, and re-run the representation gate on the new latents.
    from ..env.base import BlockwaveEnv, EnvConfig, ObsMode
    from ..env.crops import Variant
    from .empowerment import PixelEmpowerment

    env = BlockwaveEnv(
        EnvConfig(obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
                  agent_hz=args.agent_hz, gravity_scale=4.0, frame_stack=1)
    )
    env.reset(seed=args.seed)
    width = float(np.sqrt(quality["mse"] * args.latent_dim))
    reward = PixelEmpowerment(forward, horizon=args.horizon, branch=args.branch, device="cpu", seed=args.seed)
    empty_latent = encoder.embed(env._pixels()[None])
    visited = encoder.embed(np.asarray(after[cut : cut + 400]))
    width = reward.choose_width(visited, empty_latent, width=width)

    gate = probe_gate(encoder, seed=args.seed)

    torch.save(
        {"state_dict": encoder.state_dict(), "latent_dim": args.latent_dim, "size": int(before.shape[-1])},
        args.out / "vae.pt",
    )
    torch.save(
        {"state_dict": forward.state_dict(), "latent_dim": args.latent_dim, "width": width,
         "baseline": reward.baseline, "horizon": args.horizon, "branch": args.branch},
        args.out / "world.pt",
    )
    (args.out / "world.json").write_text(
        json.dumps({"model": quality, "width": width, "baseline": reward.baseline, "probes": gate}, indent=2)
    )
    for name, value in gate.items():
        mark = "PASS" if value >= probes.THRESHOLDS[name] else "FAIL"
        print(f"  probe {name:<11} R^2 {value:.3f}  (threshold {probes.THRESHOLDS[name]:.2f})  {mark}", flush=True)
    print(f"  kernel width {width:.3f} (was 1.945)   baseline {reward.baseline:.3f}", flush=True)


if __name__ == "__main__":
    main()
