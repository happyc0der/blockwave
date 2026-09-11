"""The pixel reward as one object: frames in, intrinsic reward out.

Everything the agent's reward depends on is here — the encoder it learned to see
with, the model it learned its own keys with, and the count of distinguishable
futures built on both. Nothing in this path touches engine state, and the same
object is used for training and for evaluating a checkpoint, so the two cannot
drift apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .empowerment import PixelEmpowerment
from .model import MacroModel


class FrameReward:
    def __init__(self, vae, empowerment: PixelEmpowerment) -> None:
        self.vae = vae
        self.empowerment = empowerment

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> FrameReward:
        """Load the encoder and world model produced by `repr.pixel_gate` and `world.fit`."""
        import torch

        from ..repr.pixel_vae import PixelVAE

        directory = Path(directory)
        blob = torch.load(directory / "vae.pt", map_location=device)
        vae = PixelVAE(blob["latent_dim"], blob["size"])
        vae.load_state_dict(blob["state_dict"])
        vae.to(device).eval()

        world = torch.load(directory / "world.pt", map_location=device)
        model = MacroModel(world["latent_dim"])
        model.load_state_dict(world["state_dict"])
        empowerment = PixelEmpowerment(
            model, horizon=world["horizon"], width=world["width"],
            branch=world["branch"], device=device,
        )
        empowerment._baseline = world["baseline"]
        return cls(vae, empowerment)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        return self.vae.embed(frames)

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        return self.empowerment(latents)

    def death_charge(self, gamma: float) -> float:
        return self.empowerment.death_charge(gamma)
