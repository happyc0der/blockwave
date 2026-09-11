"""A convolutional VAE over rendered frames — the pixel agent's representation.

The board-state track read the true occupancy grid. A pixel agent may not: it
sees what a player sees, so everything downstream — the reward, and the model
the reward needs — is built on these latents.

The encoder is judged before any of that, by `probes.py`: if a *linear* readout
of the latents cannot recover stack height, holes and bumpiness, then the latent
space does not represent the things a reward about board quality must be about,
and no amount of RL fixes that. The corpus matters as much as the architecture,
which is why `corpus.build_frames` mixes policies deliberately — an encoder
trained only on one policy's boards resolves only that policy's boards.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Frames are uint8 grayscale; the network works in [0, 1].
SCALE = 255.0


class PixelVAE(nn.Module):
    """88x88 grayscale in, `latent_dim` out. Strided conv down, transposed up."""

    def __init__(self, latent_dim: int = 32, size: int = 88) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),     # 44
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),    # 22
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),   # 11
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # 6
        )
        with torch.no_grad():
            self._shape = self.encoder(torch.zeros(1, 1, size, size)).shape[1:]
        flat = int(np.prod(self._shape))
        self.mu = nn.Linear(flat, latent_dim)
        self.logvar = nn.Linear(flat, latent_dim)
        self.project = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1), nn.ReLU(),            # 11
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),             # 22
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),              # 44
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),                          # 88
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.project(z).view(-1, *self._shape))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def embed(self, frames: np.ndarray, batch: int = 512) -> np.ndarray:
        """Deterministic latents (posterior means) — what the reward will see."""
        device = next(self.parameters()).device
        out = []
        for start in range(0, len(frames), batch):
            x = as_input(frames[start : start + batch]).to(device)
            out.append(self.encode(x)[0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.latent_dim), dtype=np.float32)


def as_input(frames: np.ndarray) -> torch.Tensor:
    """(N, H, W) uint8 -> (N, 1, H, W) float in [0, 1]."""
    x = torch.as_tensor(np.asarray(frames), dtype=torch.float32) / SCALE
    return x.unsqueeze(1) if x.dim() == 3 else x


def train(
    frames: np.ndarray,
    *,
    latent_dim: int = 32,
    epochs: int = 12,
    beta: float = 1.0,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log: bool = False,
) -> PixelVAE:
    torch.manual_seed(seed)
    model = PixelVAE(latent_dim, size=frames.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(frames), generator=gen).numpy()
        total, seen = 0.0, 0
        for start in range(0, len(frames), batch):
            x = as_input(frames[order[start : start + batch]]).to(device)
            logits, mu, logvar = model(x)
            # Continuous targets in [0, 1]: cross-entropy is the usual choice,
            # and it weights the bright cells that carry the board's shape.
            recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        if log:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / seen:.1f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def reconstruction_error(model: PixelVAE, frames: np.ndarray, batch: int = 512) -> float:
    """Mean absolute per-pixel error of the reconstruction, in [0, 1] units."""
    device = next(model.parameters()).device
    total, seen = 0.0, 0
    for start in range(0, len(frames), batch):
        x = as_input(frames[start : start + batch]).to(device)
        recon = torch.sigmoid(model.decode(model.encode(x)[0]))
        total += float((recon - x).abs().sum())
        seen += x.numel()
    return total / max(seen, 1)
