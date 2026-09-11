"""PPO with per-placement discounting.

PPO rather than DQN because the reward is computed at collection time and
consumed immediately; an off-policy replay buffer would carry stale rewards.

**The discount clock ticks per placement, not per agent step.** Reward arrives
only when a piece locks, so both gamma and lambda are applied only on steps
where a placement happened (``locked``), and are 1.0 in between. Two
consequences, both intended:

* Stalling is neutral. With a per-tick discount and non-positive rewards,
  dragging a piece out would *postpone* its negative reward and look better.
* Credit is shared fairly within a piece. The lock's TD error propagates back
  undiminished to every action that shaped the piece — the rotation three ticks
  earlier matters as much as the final drop. A per-tick lambda would give early
  actions exponentially less credit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class PPOConfig:
    gamma: float = 0.99          # per placement
    lam: float = 0.95            # per placement
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 0.5


def gae(rewards, values, next_value, locked, gamma, lam):
    """Generalized advantage estimation with a per-step, placement-gated discount.

    All arrays are (T, N). ``locked[t]`` says a placement happened during step t,
    so the discount and trace decay apply when bootstrapping past it.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    g = np.where(locked, gamma, 1.0).astype(np.float32)
    l = np.where(locked, lam, 1.0).astype(np.float32)
    for t in reversed(range(T)):
        nxt = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + g[t] * nxt - values[t]
        last = delta + g[t] * l[t] * last
        adv[t] = last
    return adv, adv + values


class RunningStd:
    """Running variance of the per-placement discounted return, for reward scaling."""

    def __init__(self, n: int, gamma: float) -> None:
        self.ret = np.zeros(n)
        self.gamma = gamma
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0

    def update(self, rewards: np.ndarray, locked: np.ndarray) -> None:
        self.ret = self.ret * np.where(locked, self.gamma, 1.0) + rewards
        batch_mean, batch_var, n = self.ret.mean(), self.ret.var(), len(self.ret)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)


def update(policy: nn.Module, optimizer, batch: dict, config: PPOConfig) -> dict:
    n = batch["actions"].shape[0]
    idx = np.arange(n)
    size = n // config.minibatches
    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    steps = 0
    for _ in range(config.epochs):
        np.random.shuffle(idx)
        for start in range(0, n, size):
            mb = idx[start : start + size]
            logp, entropy, value = policy.evaluate(batch["grid"][mb], batch["queue"][mb], batch["actions"][mb])
            ratio = torch.exp(logp - batch["logp"][mb])
            adv = batch["adv"][mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv)
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (value - batch["ret"][mb]).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad)
            optimizer.step()
            stats["policy"] += policy_loss.item()
            stats["value"] += value_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["clip_frac"] += ((ratio - 1).abs() > config.clip).float().mean().item()
            steps += 1
    return {k: v / steps for k, v in stats.items()}
