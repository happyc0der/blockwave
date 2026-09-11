"""Policy and value network for the board-state track."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class BoardPolicy(nn.Module):
    """A small conv trunk over the (2, 24, 10) grid, joined by the piece queue."""

    def __init__(self, grid_shape: tuple[int, int, int], queue_size: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        channels, height, width = grid_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, channels, height, width)).numel()
        self.trunk = nn.Sequential(nn.Linear(flat + queue_size, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        # Small initial logits: start near-uniform so early exploration is broad.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, grid: torch.Tensor, queue: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(grid).flatten(1)
        h = self.trunk(torch.cat([h, queue], dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, grid, queue):
        logits, value = self(grid, queue)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, grid, queue, action):
        logits, value = self(grid, queue)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


class PixelPolicy(nn.Module):
    """The pixel agent: stacked frames in, one key out.

    The usual three-layer convolutional stack for 84x84-scale frames, which is
    what this is (88x88). Frames arrive as uint8 and are scaled here rather than
    in the environment, so what crosses between processes stays a byte per pixel.
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden: int = 512) -> None:
        super().__init__()
        stack, height, width = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, stack, height, width)).numel()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.conv(frames.float() / 255.0))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, frames):
        logits, value = self(frames)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, frames, action):
        logits, value = self(frames)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
