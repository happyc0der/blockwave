"""Compare training runs without overstating what the numbers support.

    python -m blockwave_rl.compare runs/pixel_ppo_150m_width runs/pixel_ppo_50m_width
    python -m blockwave_rl.compare --markdown runs/pixel_ppo_*

This exists because of a mistake. Every pixel run was reported with the standard
error `evaluate.py` prints, and those comparisons were written up as findings.
That standard error is the spread across *games played by one trained network* —
it says how precisely that network was measured. It says nothing about how much
the training run varies, and when a second seed of an identical config finally
ran, the two differed by 42% of their mean: a seed-to-seed spread 19x the
within-run error. Every effect previously reported was a fraction of it.

So the rule here is mechanical rather than advisory:

  * runs are grouped by config, seed excluded, so seeds of one setting collect
    into one row;
  * a group of one seed gets no uncertainty and no sigma, only a note that it
    cannot support a comparison;
  * a difference between groups is quoted only when both sides have >= 2 seeds,
    and it is always shown next to the smallest difference the seed count could
    actually resolve.

Nothing goes into README.md that did not come out of this tool.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

#: Config keys that do not define a configuration: where it was written, which
#: seed it drew, and how it was parallelised or logged.
IGNORED = {"out", "seed", "device", "workers", "save_every", "print_every"}


@dataclass
class Run:
    path: Path
    seed: int
    config: dict
    checkpoint: str
    lines_per_piece: float
    within_run_se: float
    pieces_per_game: float
    deaths_per_piece: float

    @property
    def key(self) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in self.config.items() if k not in IGNORED))


def load(path: Path) -> Run | None:
    """Read a run's final evaluated checkpoint, or None if it was never evaluated."""
    evaluations = path / "eval.jsonl"
    config_file = path / "config.json"
    if not evaluations.exists() or not config_file.exists():
        return None
    rows = [json.loads(line) for line in evaluations.open() if line.strip()]
    if not rows:
        return None
    config = json.loads(config_file.read_text())
    last = rows[-1]
    return Run(
        path=path,
        seed=int(config.get("seed", 0)),
        config=config,
        checkpoint=last["checkpoint"],
        lines_per_piece=last["lines_per_piece"],
        within_run_se=last["lines_per_piece_se"],
        pieces_per_game=last["pieces_per_game"],
        deaths_per_piece=last["top_outs_per_piece"],
    )


@dataclass
class Group:
    """Runs that differ only by seed."""

    runs: list[Run]

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def values(self) -> list[float]:
        return [r.lines_per_piece for r in self.runs]

    @property
    def mean(self) -> float:
        return sum(self.values) / self.n

    @property
    def seed_sd(self) -> float | None:
        """Sample SD across seeds. None at n=1, where it does not exist."""
        if self.n < 2:
            return None
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (self.n - 1))

    @property
    def se_of_mean(self) -> float | None:
        sd = self.seed_sd
        return None if sd is None else sd / math.sqrt(self.n)

    @property
    def resolvable(self) -> float | None:
        """Smallest difference two groups this size could separate at 2 sigma."""
        sd = self.seed_sd
        return None if sd is None else 2.0 * sd * math.sqrt(2.0 / self.n)

    def label(self) -> str:
        c = self.runs[0].config
        reward = Path(str(c.get("reward", "?"))).name
        return f"{int(c.get('steps', 0)) / 1e6:.0f}M steps, reward={reward}"


def group(runs: list[Run]) -> list[Group]:
    buckets: dict[tuple, list[Run]] = {}
    for run in runs:
        buckets.setdefault(run.key, []).append(run)
    groups = [Group(sorted(v, key=lambda r: r.seed)) for v in buckets.values()]
    return sorted(groups, key=lambda g: int(g.runs[0].config.get("steps", 0)))


def describe(g: Group) -> list[str]:
    out = [f"{g.label()}   [{g.n} seed{'s' if g.n != 1 else ''}]"]
    for run in g.runs:
        out.append(
            f"    seed {run.seed}  {run.checkpoint:16s} {run.lines_per_piece:.4f} lines/pc"
            f"   {run.deaths_per_piece:.4f} deaths/pc   {run.pieces_per_game:5.1f} pieces/game"
            f"   (within-run SE {run.within_run_se:.4f})"
        )
    if g.n < 2:
        out.append(
            "    n=1 -- no uncertainty estimate. The within-run SE above measures this one"
        )
        out.append(
            "    network, not the training process, and cannot support a comparison."
        )
    else:
        out.append(
            f"    mean {g.mean:.4f}   seed SD {g.seed_sd:.4f}   SE(mean) {g.se_of_mean:.4f}"
        )
        out.append(f"    resolvable at 2 sigma with n={g.n}: differences >= {g.resolvable:.4f}")
        if g.n == 2:
            out.append("    (n=2 gives a 1-dof SD estimate: treat it as an order of magnitude)")
    return out


def contrast(a: Group, b: Group, assume_sd: float | None = None) -> list[str]:
    """State a difference only when both sides can carry one."""
    head = f"{a.label()}  vs  {b.label()}"
    diff = a.mean - b.mean
    if a.n < 2 or b.n < 2:
        thin = [g.label() for g in (a, b) if g.n < 2]
        lines = [
            head,
            f"    difference in means: {diff:+.4f}",
            "    NO COMPARISON -- single seed: " + "; ".join(thin),
            "    A lone run carries no uncertainty, so this difference cannot be",
            "    told apart from run-to-run variation.",
        ]
        if assume_sd:
            # Borrowing an SD measured elsewhere is weaker than measuring it here:
            # it assumes the arms vary like whatever group it came from. It is
            # still the difference between "cannot say" and "cannot say anything",
            # and gross effects -- the agent learning at all -- clear it easily.
            ratio = abs(diff) / (assume_sd * math.sqrt(2))
            reads = "would survive it" if ratio >= 2 else "would not survive it"
            lines.append(
                f"    against a BORROWED seed SD of {assume_sd:.4f}: {ratio:.1f} sigma, {reads}"
            )
        return lines
    pooled = math.sqrt((a.seed_sd**2) / a.n + (b.seed_sd**2) / b.n)
    sigma = diff / pooled if pooled else float("inf")
    verdict = "distinguishable" if abs(sigma) >= 2 else "NOT distinguishable from noise"
    return [
        head,
        f"    difference in means: {diff:+.4f}   ({sigma:+.2f} sigma)   {verdict}",
        f"    seed SDs: {a.seed_sd:.4f} / {b.seed_sd:.4f}",
    ]


def markdown(groups: list[Group]) -> str:
    lines = [
        "| configuration | seeds | lines per piece | pieces per game |",
        "|---|---|---|---|",
    ]
    for g in groups:
        if g.n < 2:
            value = f"{g.mean:.4f} _(1 seed, no error bar)_"
        else:
            lo, hi = min(g.values), max(g.values)
            value = f"{g.mean:.4f} _(n={g.n}, {lo:.4f}–{hi:.4f})_"
        pieces = sum(r.pieces_per_game for r in g.runs) / g.n
        lines.append(f"| {g.label()} | {g.n} | {value} | {pieces:.1f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--markdown", action="store_true", help="emit the README table")
    p.add_argument("--assume-sd", type=float, default=None,
                   help="seed SD measured elsewhere, to weigh single-seed gaps against")
    args = p.parse_args()

    loaded = [r for r in (load(path) for path in args.runs) if r is not None]
    missing = [str(path) for path in args.runs if load(path) is None]
    if not loaded:
        raise SystemExit("no evaluated runs found (need eval.jsonl and config.json)")
    groups = group(loaded)

    if args.markdown:
        print(markdown(groups))
        return

    for g in groups:
        print("\n".join(describe(g)))
        print()
    if len(groups) > 1:
        print("-- contrasts " + "-" * 50)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                print("\n".join(contrast(groups[j], groups[i], args.assume_sd)))
                print()
    if missing:
        print(f"not evaluated, skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
