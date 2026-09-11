"""Predicting what a key-press program does, in latent space.

Given the screen as the piece enters play and one of the macros, predict the
screen once the piece has resolved — and whether running it ends the game. This
is the simulator's replacement: the board-state reward enumerated futures by
asking the engine, and this asks a model the agent fitted to its own experience.

It predicts a *change* rather than an absolute latent. Most of a frame survives
a placement untouched, so predicting the difference starts from the right answer
and has only the piece's effect left to learn.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .babble import Babble
from .macros import N_MACROS


class MacroModel(nn.Module):
    def __init__(self, latent_dim: int, n_macros: int = N_MACROS, hidden: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_macros = n_macros
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + n_macros, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.delta = nn.Linear(hidden, latent_dim)
        self.death = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, macro: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = nn.functional.one_hot(macro, self.n_macros).to(z.dtype)
        h = self.trunk(torch.cat([z, onehot], dim=1))
        return z + self.delta(h), self.death(h).squeeze(-1)

    @torch.no_grad()
    def outcomes(self, z: torch.Tensor, macros: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Every macro applied to every state: (states, survival probability).

        ``z`` is (N, D); the result is (N, M, D) and (N, M).
        """
        macros = torch.arange(self.n_macros, device=z.device) if macros is None else macros
        n, m = len(z), len(macros)
        flat_z = z.repeat_interleave(m, dim=0)
        flat_m = macros.repeat(n)
        nxt, death = self(flat_z, flat_m)
        return nxt.view(n, m, -1), (1.0 - torch.sigmoid(death)).view(n, m)


def train(
    data: Babble,
    *,
    hidden: int = 256,
    epochs: int = 20,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log: bool = False,
) -> MacroModel:
    torch.manual_seed(seed)
    model = MacroModel(data.before.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    before = torch.as_tensor(data.before, device=device)
    after = torch.as_tensor(data.after, device=device)
    macro = torch.as_tensor(data.macro, device=device)
    died = torch.as_tensor(data.died.astype(np.float32), device=device)
    alive = 1.0 - died
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(macro), generator=gen).to(device)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            predicted, death_logit = model(before[index], macro[index])
            # A run that ended the game has no successor screen to predict: what
            # the frame shows then is a fresh board, which is the opposite of
            # what happened. Those rows train the death head only.
            weight = alive[index]
            error = ((predicted - after[index]) ** 2).mean(dim=1)
            fit = (error * weight).sum() / weight.sum().clamp(min=1.0)
            fatal = nn.functional.binary_cross_entropy_with_logits(death_logit, died[index])
            loss = fit + fatal
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        if log:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / seen:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def report(model: MacroModel, data: Babble, device: str = "cpu") -> dict[str, float]:
    """Held-out quality, against the baselines worth beating.

    ``skill`` compares the model's error to predicting no change at all: 1.0 is
    perfect, 0.0 is no better than assuming the screen stays as it is. Beating
    the mean-change baseline is the weaker bar — that one ignores the board.
    """
    before = torch.as_tensor(data.before, device=device)
    after = torch.as_tensor(data.after, device=device)
    macro = torch.as_tensor(data.macro, device=device)
    alive = ~data.died
    predicted, death_logit = model(before, macro)

    live = torch.as_tensor(alive, device=device)
    def mse(x: torch.Tensor) -> float:
        return float((((x - after) ** 2).mean(dim=1) * live).sum() / live.sum().clamp(min=1.0))

    still = mse(before)
    mean_shift = mse(before + (after - before)[live].mean(dim=0))
    model_error = mse(predicted)
    # Can the model tell its own programs apart? For each held-out try, rank
    # all programs by how close their predicted outcome is to what actually
    # happened. A model whose error swamps the difference between programs
    # ranks at chance, and a reward that counts distinct outcomes through it is
    # blind however good its average error looks.
    predicted_all, _ = model.outcomes(before)
    gap = (predicted_all - after.unsqueeze(1)).pow(2).mean(dim=2)
    order = gap.argsort(dim=1)
    rank = (order == macro.unsqueeze(1)).float().argmax(dim=1)[torch.as_tensor(alive, device=device)]

    death_prob = torch.sigmoid(death_logit).cpu().numpy()
    predicted_death = death_prob > 0.5
    true_death = data.died
    return {
        "mse": model_error,
        "mse_no_change": still,
        "mse_mean_change": mean_shift,
        "skill": 1.0 - model_error / still if still > 0 else float("nan"),
        "macro_top1": float((rank == 0).float().mean()),
        "macro_rank": float(rank.float().mean()),
        "macro_chance_rank": (model.n_macros - 1) / 2,
        "death_rate": float(true_death.mean()),
        "death_recall": float(predicted_death[true_death].mean()) if true_death.any() else float("nan"),
        "death_precision": float(true_death[predicted_death].mean()) if predicted_death.any() else float("nan"),
    }
