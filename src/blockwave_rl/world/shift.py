"""Is the world model still accurate on the states a trained agent reaches?

    python -m blockwave_rl.world.shift runs/pixel_ppo_seed0/ckpt_01627.pt 3000

The model is fitted once, while babbling — near-random play. A trained agent
visits flatter, lower boards, and if the model is worse there then the reward
degrades exactly where the agent is going, silently. This lets the agent play,
hijacks every other piece with a random macro, and scores the frozen model on
those transitions. Compare with the held-out numbers in `world.json`.

Board occupancy is read for one line of context about how full those boards
are; like the probes, it is a diagnostic and reaches nothing.
"""
import sys
from collections import deque
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)

from blockwave_rl.env.base import BlockwaveEnv, EnvConfig, ObsMode
from blockwave_rl.env.crops import Variant
from blockwave_rl.agents.nets import PixelPolicy
from blockwave_rl.world.babble import Babble
from blockwave_rl.world.events import FrameEvents
from blockwave_rl.world.macros import MACROS, N_MACROS
from blockwave_rl.world.model import MacroModel, report
from blockwave_rl.repr.pixel_vae import PixelVAE

ckpt, n_pairs = Path(sys.argv[1]), int(sys.argv[2])
blob = torch.load("runs/pixel/vae.pt", map_location="cpu")
vae = PixelVAE(blob["latent_dim"], blob["size"]); vae.load_state_dict(blob["state_dict"]); vae.eval()
world = torch.load("runs/pixel/world.pt", map_location="cpu")
model = MacroModel(world["latent_dim"]); model.load_state_dict(world["state_dict"]); model.eval()
policy = PixelPolicy((4, 88, 88), 8); policy.load_state_dict(torch.load(ckpt, map_location="cpu")); policy.eval()

env = BlockwaveEnv(EnvConfig(obs_mode=ObsMode.PIXELS, variant=Variant.BOARD_PLUS_PREVIEW, cell_px=4,
                             agent_hz=5.0, gravity_scale=4.0, frame_stack=1))
env.reset(seed=4242)
events = FrameEvents(env.layout, env.crop); events.calibrate(env._pixels())
rng = np.random.default_rng(0); torch.manual_seed(0)
frame = env._pixels(); stack = deque([frame] * 4, maxlen=4)
before, macros, after, died = [], [], [], []
heights = []

def agent_action():
    with torch.no_grad():
        a, _, _ = policy.act(torch.as_tensor(np.stack(tuple(stack))[None]))
    return int(a)

steps = 0
while len(macros) < n_pairs and steps < 800_000:
    # Let the agent play a piece, then hijack the next one with a random macro.
    for _ in range(200):
        a = agent_action(); env.step(a); steps += 1
        cur = env._pixels(); placed, ended = events.observe(frame, cur, a); frame = cur; stack.append(cur)
        if placed or ended: break
    start = frame
    heights.append(int(env.occupancy().sum()))          # diagnostic only
    m = int(rng.integers(N_MACROS)); dead = False
    for step in range(200):
        acts = MACROS[m].actions
        a = acts[step] if step < len(acts) else int(acts[-1])
        env.step(a); steps += 1
        cur = env._pixels(); placed, ended = events.observe(frame, cur, a); frame = cur; stack.append(cur)
        dead |= ended
        if placed or ended: break
    before.append(start); after.append(frame); macros.append(m); died.append(dead)

data = Babble(vae.embed(np.stack(before)), np.array(macros), vae.embed(np.stack(after)), np.array(died))
q = report(model, data)
print(f"agent states ({len(data)} tries, mean occupancy {np.mean(heights):.0f} cells):")
print("  " + "  ".join(f"{k} {v:.3f}" for k, v in q.items()))
