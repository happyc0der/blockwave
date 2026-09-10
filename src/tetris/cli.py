"""Command line entry point: ``tetris <command>``."""

from __future__ import annotations

import argparse


def _play(args: argparse.Namespace) -> int:
    from .app.main import Game

    Game(
        seed=args.seed,
        start_level=args.level,
        profile=args.profile,
        cell_px=args.cell,
        audio=not args.mute,
        music=not (args.mute or args.no_music),
        record_to=args.record,
    ).run()
    return 0


def _shot(args: argparse.Namespace) -> int:
    """Render a still frame to a PNG, without opening a window."""
    import random

    import numpy as np
    import pygame

    from .core.constants import Action
    from .core.engine import EngineConfig, TetrisEngine
    from .render.compositor import Compositor
    from .render.layout import Layout

    engine = TetrisEngine(EngineConfig(seed=args.seed or 0, start_level=args.level))
    rng = random.Random(args.seed or 0)
    for _ in range(args.pieces):
        # Aim each piece at a column rather than walking randomly, so the stack
        # spreads out the way a real game's does instead of towering up in the
        # middle and topping out after a dozen pieces.
        for _ in range(rng.randrange(3)):
            engine.step(Action.ROTATE_CW, 0.0)
        target = rng.randrange(10)
        for _ in range(10):
            if engine.piece is None:
                break
            current = min(x for x, _ in engine.piece.cells())
            if current == target:
                break
            engine.step(Action.LEFT if current > target else Action.RIGHT, 0.0)
        engine.step(Action.HARD_DROP, 0.0)
        # Stepping with dt=0 never advances the line-clear pause, so skip it
        # explicitly rather than stalling on a board with no active piece.
        engine.finish_clear()
        if engine.game_over:
            engine.reset()

    frame = Compositor(Layout(args.cell), args.profile).render(engine)
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    pygame.image.save(surface, args.out)
    print(f"wrote {args.out}  {frame.shape[1]}x{frame.shape[0]}")
    return 0


def _gen_assets(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .audio.generate import generate

    directory = Path(args.out) if args.out else None
    print("generating sounds...")
    files = generate(directory, music=not args.no_music)
    total = sum(path.stat().st_size for path in files)
    print(f"\nwrote {len(files)} files, {total / 1024:.0f} KB, to {files[0].parent}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Re-run a recorded session and report whether it reproduces exactly."""
    from pathlib import Path

    from .app.replay import Replay, verify

    replay = Replay.load(Path(args.file))
    matched, engine = verify(replay)
    stats = engine.stats

    print(f"seed {replay.seed}  level {replay.start_level}  {replay.ticks:,} ticks")
    print(f"  recorded score {replay.score:,}")
    print(f"  replayed score {stats.score:,}  lines {stats.lines}  pieces {stats.pieces_placed}")
    print(f"  {'reproduced exactly' if matched else 'DIVERGED from the recording'}")
    return 0 if matched else 1


def _bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmarks

    run_benchmarks(steps=args.steps)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tetris", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="play the game")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--level", type=int, default=1, help="starting level")
    play.add_argument("--profile", default="arcade", choices=["flat", "arcade", "arcade_max"])
    play.add_argument("--cell", type=int, default=30, help="pixels per cell")
    play.add_argument("--mute", action="store_true", help="no sound at all")
    play.add_argument("--no-music", action="store_true", help="sound effects only")
    play.add_argument("--record", default=None, metavar="FILE", help="write a replay of each game")
    play.set_defaults(func=_play)

    shot = sub.add_parser("shot", help="render a still frame to a PNG")
    shot.add_argument("out")
    shot.add_argument("--seed", type=int, default=0)
    shot.add_argument("--level", type=int, default=1)
    shot.add_argument("--pieces", type=int, default=12)
    shot.add_argument("--profile", default="arcade", choices=["flat", "arcade", "arcade_max"])
    shot.add_argument("--cell", type=int, default=30)
    shot.set_defaults(func=_shot)

    gen = sub.add_parser("gen-assets", help="synthesize the sound effects and music")
    gen.add_argument("--out", default=None, help="target directory (default: assets/sfx)")
    gen.add_argument("--no-music", action="store_true", help="effects only")
    gen.set_defaults(func=_gen_assets)

    rep = sub.add_parser("replay", help="re-run a recorded session and verify it")
    rep.add_argument("file")
    rep.set_defaults(func=_replay)

    bench = sub.add_parser("bench", help="measure steps per second")
    bench.add_argument("--steps", type=int, default=20_000)
    bench.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
