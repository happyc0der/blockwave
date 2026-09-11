"""Game performance — the yardstick the agent never sees.

The only module in this package permitted to read score, lines or level (the
firewall's static test exempts it). Nothing here returns anything the learner
consumes. It measures; it does not steer.

Lines are tracked as the sum of positive increments of each env's per-game line
count, because a soft reset zeroes that count — a naive "lines at top-out"
would read the *new* game's zero.

Run as a module to evaluate a finished run after the fact::

    python -m blockwave_rl.evaluate runs/emp_seed0 random drift

Every checkpoint in a run directory is evaluated, in schedule order, on held-out
seeds. The output is a curve. Nothing here picks a "best" checkpoint, and
nothing should: selecting by score would leak the score into the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

#: Baselines every result is compared against.
#:
#: ``drift`` is uniform over every action except HARD_DROP. It exists because
#: the first training run found it: a policy that merely stops hard-dropping
#: lets each piece drift under gravity while random moves spread it across the
#: board, which halves deaths per piece against uniform random play. Beating
#: `random` is therefore not evidence of learning where pieces go; beating
#: `drift` is the bar.
BASELINES: dict[str, list[int]] = {
    "random": [0, 1, 2, 3, 4, 5, 6, 7],
    "drift": [0, 1, 2, 3, 5, 6, 7],
}

#: Far from every training seed (training workers use seed + w * 10_000 + i).
EVAL_SEED = 1_000_000


#: Per-env quantities a complete-games measurement sums.
_FIELDS = ("lines", "pieces", "top_outs", "intrinsic", "placements", "alive_intrinsic", "alive_placements")


class GameTracker:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self._prev_lines = np.zeros(n_envs)
        self._prev_pieces = np.zeros(n_envs)
        self.reset_window()

    def reset_window(self) -> None:
        self.env_lines = np.zeros(self.n_envs)
        self.env_pieces = np.zeros(self.n_envs)
        self.env_top_outs = np.zeros(self.n_envs)
        self.env_intrinsic = np.zeros(self.n_envs)
        self.env_placements = np.zeros(self.n_envs)
        # Placements that did not end the game: comparable across death accountings.
        self.env_alive_intrinsic = np.zeros(self.n_envs)
        self.env_alive_placements = np.zeros(self.n_envs)
        self.steps = 0
        # Running totals at each env's first and latest top-out in the window.
        self._first = np.full((len(_FIELDS), self.n_envs), np.nan)
        self._last = np.full((len(_FIELDS), self.n_envs), np.nan)

    @property
    def lines(self) -> float:
        return float(self.env_lines.sum())

    @property
    def pieces(self) -> float:
        return float(self.env_pieces.sum())

    @property
    def top_outs(self) -> int:
        return int(self.env_top_outs.sum())

    def update(self, infos: list[dict], rewards: np.ndarray | None = None, locked: np.ndarray | None = None) -> None:
        """Record one vector step. ``rewards``/``locked`` add intrinsic totals."""
        for i, info in enumerate(infos):
            lines, pieces = info["lines"], info["pieces"]
            self.env_lines[i] += max(0.0, lines - self._prev_lines[i])
            self.env_pieces[i] += max(0.0, pieces - self._prev_pieces[i])
            self._prev_lines[i], self._prev_pieces[i] = lines, pieces
            self.env_top_outs[i] += int(info["top_out"])
            if rewards is not None and locked is not None and locked[i]:
                self.env_intrinsic[i] += float(rewards[i])
                self.env_placements[i] += 1
                if not info["top_out"]:
                    self.env_alive_intrinsic[i] += float(rewards[i])
                    self.env_alive_placements[i] += 1
            if info["top_out"]:
                # After the increments: the death belongs to the game it ended.
                snap = [getattr(self, f"env_{f}")[i] for f in _FIELDS]
                if np.isnan(self._first[0, i]):
                    self._first[:, i] = snap
                self._last[:, i] = snap
            self.steps += 1

    def summary(self) -> dict[str, float]:
        return {
            "lines_per_piece": self.lines / max(self.pieces, 1.0),
            "lines_per_1k_steps": 1000.0 * self.lines / max(self.steps, 1),
            "pieces_per_1k_steps": 1000.0 * self.pieces / max(self.steps, 1),
            "top_outs_per_1k_steps": 1000.0 * self.top_outs / max(self.steps, 1),
            # Per piece, not per step. A per-step rate falls whenever the agent
            # merely plays slower, which is not the same as dying less.
            "top_outs_per_piece": self.top_outs / max(self.pieces, 1.0),
        }

    def standard_errors(self) -> dict[str, float]:
        """Standard errors of the per-piece rates, treating envs as independent.

        Each env's games are correlated with each other, so pieces are not the
        independent unit — envs are. This is the usual ratio-estimator error.
        """
        return {
            "lines_per_piece": _ratio_se(self.env_lines, self.env_pieces),
            "top_outs_per_piece": _ratio_se(self.env_top_outs, self.env_pieces),
        }

    def complete_games(self) -> dict[str, float]:
        """Rates over complete games only: each env's first to last top-out.

        A top-out restarts from an empty board, so games are independent and
        identically distributed. A fixed window instead cuts a game at each
        edge, and those cuts do not average out when game lengths are regular:
        the envs stay in phase. Measured on the drift baseline, a window after
        10k steps of burn-in and one after 20k disagreed by ~3 standard errors
        in deaths per piece. Complete games have no edges.

        Envs with fewer than two top-outs in the window contribute nothing; a
        policy that rarely dies needs a longer window, and ``games`` says so.
        """
        span = np.nan_to_num(self._last - self._first)
        t = dict(zip(_FIELDS, span))
        games = t["top_outs"]
        return {
            "games": float(games.sum()),
            "min_games_per_env": float(games.min()),
            "lines_per_piece": _ratio(t["lines"], t["pieces"]),
            "lines_per_piece_se": _ratio_se(t["lines"], t["pieces"]),
            "top_outs_per_piece": _ratio(games, t["pieces"]),
            "top_outs_per_piece_se": _ratio_se(games, t["pieces"]),
            "intrinsic_per_placement": _ratio(t["intrinsic"], t["placements"]),
            "intrinsic_per_placement_se": _ratio_se(t["intrinsic"], t["placements"]),
            "alive_per_placement": _ratio(t["alive_intrinsic"], t["alive_placements"]),
            "alive_per_placement_se": _ratio_se(t["alive_intrinsic"], t["alive_placements"]),
            "pieces_per_game": _ratio(t["pieces"], games),
            "lines_per_game": _ratio(t["lines"], games),
        }


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    return float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")


def _ratio_se(num: np.ndarray, den: np.ndarray) -> float:
    """Ratio-estimator standard error of sum(num)/sum(den), units independent."""
    n = len(num)
    if n < 2 or den.sum() == 0:
        return float("nan")
    resid = num - _ratio(num, den) * den
    return float(np.sqrt(resid.var(ddof=1) / n) / den.mean())


# -- after-the-fact evaluation ---------------------------------------------


def play(
    act: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    steps_per_env: int,
    envs: int,
    workers: int,
    burn_in: int = 0,
    horizon: int = 2,
    seed: int = EVAL_SEED,
    death: str = "absorbing",
    gamma: float = 0.99,
    agent_hz: float = 20.0,
    gravity_scale: float = 4.0,
) -> dict[str, float]:
    """Play ``act`` and measure it over complete games (see `complete_games`).

    The first version measured a fixed window from a fresh start, which
    over-weights the easy early game: it put the drift baseline at -0.225
    intrinsic per placement against about -0.33 in continuous play. Burn-in
    fixed most of that; complete games fix the rest. Window rates are still
    returned under ``window_*`` for policies that rarely die.
    """
    from .env.vector import ProcessVecEnv

    vec = ProcessVecEnv(
        envs, workers, horizon=horizon, seed=seed, death=death, gamma=gamma,
        agent_hz=agent_hz, gravity_scale=gravity_scale,
    )
    try:
        grid, queue = vec.reset()
        tracker = GameTracker(envs)
        for t in range(burn_in + steps_per_env):
            if t == burn_in:
                tracker.reset_window()
            batch = vec.step(act(grid, queue))
            tracker.update(batch.info, batch.reward, batch.locked)
            grid, queue = batch.grid, batch.queue
    finally:
        vec.close()
    window = {**tracker.summary(), "intrinsic_per_placement": _ratio(tracker.env_intrinsic, tracker.env_placements)}
    return {
        **tracker.complete_games(),
        "pieces_per_1k_steps": window["pieces_per_1k_steps"],
        **{f"window_{k}": v for k, v in window.items()},
        "steps": tracker.steps,
    }


def baseline_actor(name: str, seed: int = EVAL_SEED) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    choices = np.array(BASELINES[name])
    rng = np.random.default_rng(seed)
    return lambda grid, queue: rng.choice(choices, size=len(grid))


def checkpoint_actor(path: Path, device: str, horizon: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The policy exactly as it acts in training: sampled, not argmax.

    Argmax is a different policy — and for a frame-level agent a degenerate one,
    since a state whose most likely action is NOOP would then stall forever.
    """
    import torch

    from .agents.nets import BoardPolicy
    from .reward.empowerment_env import N_PIECES

    policy = BoardPolicy((2, 24, 10), horizon * N_PIECES, 8).to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()

    def act(grid: np.ndarray, queue: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action, _, _ = policy.act(torch.as_tensor(grid, device=device), torch.as_tensor(queue, device=device))
        return action.cpu().numpy()

    return act


def _row(name: str, r: dict[str, float]) -> str:
    return (
        f"{name:>18s}  lines/pc {r['lines_per_piece']:.4f} ±{r['lines_per_piece_se']:.4f}  "
        f"deaths/pc {r['top_outs_per_piece']:.4f} ±{r['top_outs_per_piece_se']:.4f}  "
        f"alive/pl {r['alive_per_placement']:+.3f} ±{r['alive_per_placement_se']:.3f}  "
        f"pieces/game {r['pieces_per_game']:5.1f}  games {r['games']:.0f} (min/env {r['min_games_per_env']:.0f})  "
        f"pieces/1k {r['pieces_per_1k_steps']:5.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("targets", nargs="+", help="run directories, or a baseline name: " + ", ".join(BASELINES))
    p.add_argument("--steps-per-env", type=int, default=30_000, help="measured steps per env")
    p.add_argument("--burn-in", type=int, default=0, help="unmeasured steps per env first")
    p.add_argument("--envs", type=int, default=96)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--death", choices=("absorbing", "one_step"), default="absorbing",
                   help="death accounting for the reported intrinsic reward; the same for every row")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--agent-hz", type=float, default=20.0, help="baselines only; runs use their own config")
    p.add_argument("--gravity", type=float, default=4.0, help="baselines only; runs use their own config")
    p.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint (schedule order)")
    args = p.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1)
    common = dict(
        steps_per_env=args.steps_per_env, burn_in=args.burn_in,
        envs=args.envs, workers=args.workers, seed=args.seed, death=args.death, gamma=args.gamma,
    )

    for target in args.targets:
        if target in BASELINES:
            dynamics = dict(agent_hz=args.agent_hz, gravity_scale=args.gravity)
            result = play(baseline_actor(target, args.seed), **common, **dynamics)
            print(_row(f"{target} @{args.agent_hz:g}Hz", result), flush=True)
            out = Path("runs") / "baselines"
            out.mkdir(parents=True, exist_ok=True)
            suffix = "" if (args.agent_hz, args.gravity) == (20.0, 4.0) else f"_{args.agent_hz:g}hz_g{args.gravity:g}"
            (out / f"{target}{suffix}.json").write_text(json.dumps({**result, **common, **dynamics}, indent=2))
            continue
        run = Path(target)
        config = json.loads((run / "config.json").read_text())
        horizon = config["horizon"]
        # The policy is evaluated in the dynamics it was trained in.
        dynamics = dict(agent_hz=config.get("agent_hz", 20.0), gravity_scale=config.get("gravity", 4.0))
        ckpts = sorted(run.glob("ckpt_*.pt"))
        ckpts = ckpts[args.every - 1 :: args.every] if args.every > 1 else ckpts
        with (run / "eval.jsonl").open("w") as log:
            for ckpt in ckpts:
                result = play(checkpoint_actor(ckpt, device, horizon), horizon=horizon, **common, **dynamics)
                result["checkpoint"] = ckpt.name
                log.write(json.dumps({**result, **common, **dynamics}) + "\n")
                log.flush()
                print(_row(f"{run.name}/{ckpt.stem}", result), flush=True)


if __name__ == "__main__":
    main()
