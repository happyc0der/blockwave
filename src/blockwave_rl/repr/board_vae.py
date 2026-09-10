"""A VAE over the 20x10 occupancy grid.

A Gaussian over its latents is a density that does **not** factorize over
cells. That is the point: the per-cell Bernoulli scored empty cells
independently, so a policy that left six columns permanently empty got that
likelihood for free. A joint model has to account for the board as a whole.

Same density family as the pixel track (Gaussian over VAE latents), which makes
this a truer feasibility gate than the Bernoulli it replaces.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BoardVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 256, cells: int = 200) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(cells, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, latent_dim)
        self.logvar = nn.Linear(hidden // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, cells),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def embed(self, boards: np.ndarray) -> np.ndarray:
        """Deterministic latent (the posterior mean) — what the reward sees."""
        x = torch.as_tensor(np.asarray(boards, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        mu, _ = self.encode(x.to(device))
        return mu.cpu().numpy()


def train(
    states: np.ndarray,
    *,
    latent_dim: int = 16,
    epochs: int = 30,
    beta: float = 0.5,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> BoardVAE:
    torch.manual_seed(seed)
    model = BoardVAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.as_tensor(states.reshape(len(states), -1).astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=gen)
        for start in range(0, len(data), batch):
            x = data[order[start : start + batch]].to(device)
            logits, mu, logvar = model(x)
            recon = nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model.cpu()


@torch.no_grad()
def reconstruction_error(model: BoardVAE, states: np.ndarray) -> float:
    """Mean per-cell error of thresholded reconstructions."""
    x = torch.as_tensor(states.reshape(len(states), -1).astype(np.float32))
    mu, _ = model.encode(x)
    recon = (torch.sigmoid(model.decoder(mu)) > 0.5).float()
    return float((recon != x).float().mean())
