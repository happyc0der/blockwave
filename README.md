# TETRIS // MIAMI

A guideline Tetris simulator with a dark 90s Miami Vice arcade presentation,
built as the foundation for a later vision-based RL project.

Everything is generated: the graphics are painted in numpy, the font is a
procedural bitmap atlas, and every sound and music track is synthesized from
oscillators at build time. No image, font or audio file is downloaded or
bundled.

```bash
uv run tetris play
```

---

## Quick start

```bash
uv sync --extra dev          # install
uv run tetris gen-assets     # synthesize sounds and music (once)
uv run tetris play           # play
```

`gen-assets` writes ~3.7 MB of `.wav` into `assets/sfx/`. The game runs without
them — it just stays silent.

## Controls

| Key | Action |
|---|---|
| ← / → | move |
| ↓ | soft drop |
| **Space** | hard drop |
| ↑ or X | rotate clockwise |
| Z or Ctrl | rotate anticlockwise |
| C or Shift | hold |
| Esc or P | pause |
| Enter | start / restart / resume |
| Q | quit to title |

Auto-shift is DAS 133 ms, ARR 10 ms, both tunable in `app/input.py`. The DAS
charge deliberately survives a piece locking, so holding a direction slides each
new piece immediately instead of stalling for a fresh delay.

## Commands

```bash
uv run tetris play  [--level N] [--profile P] [--cell N] [--mute] [--no-music] [--record FILE]
uv run tetris shot  OUT.png [--pieces N] [--seed N] [--profile P] [--cell N]
uv run tetris gen-assets [--out DIR] [--no-music]
uv run tetris replay FILE
uv run tetris bench [--steps N]
```

**Visual profiles** — `flat` (neon blocks on a dark ground, completely still),
`arcade` (the default: scanlines, bloom, vignette, sun, grid floor and screen
shake) and `arcade_max` (everything, plus harder shake, chromatic aberration and
particle bursts on line clears).

Every field on a profile has to visibly change the frame — `tests/test_profiles.py`
renders each one on and off and asserts the output differs, so a setting that
does nothing cannot ship.

**`--cell`** sets the pixel size of one board cell; the whole scene is laid out
in cell units, so this scales the window.

## Rules

Modern guideline: SRS rotation with both wall-kick tables, a 7-bag randomizer,
hold, ghost piece, a 5-piece preview, lock delay with capped move resets,
T-spins by the three-corner rule, back-to-back and combos.

Difficulty ramps every 10 lines. Gravity follows the guideline curve — 1.0 s per
row at level 1, becoming instantaneous (20G) at level 18 — after which the lock
delay keeps shrinking, from 500 ms down to a 150 ms floor. The pause after a
line clear tightens on the same schedule, and the music steps through four
tempo bands as the level rises.

## Replays

```bash
uv run tetris play --record run.json
uv run tetris replay run.json
```

The engine is deterministic given a seed, a sequence of actions and a fixed
timestep, so a replay reproduces a session exactly — verified against a digest
of the final board and stats. That turns "the game glitched" into a file, and a
glitch into a regression test.

Recordings are sparse (idle ticks are not stored), so a full game is a few
kilobytes.

## Driving the engine directly

The engine is pure Python and numpy with no pygame import, so it runs headless
at ~330k steps/sec:

```python
from tetris.core.engine import TetrisEngine, EngineConfig
from tetris.core.constants import Action

engine = TetrisEngine(EngineConfig(seed=42, start_level=1))

events = engine.step(Action.HARD_DROP, dt=1 / 60)
print([e.type.name for e in events])
print(engine.telemetry())     # score, lines, level, holes, bumpiness, heights...
print(engine.to_ascii())      # the board as text
```

`step(action, dt)` takes time explicitly rather than reading a clock, so a
fixed-timestep game loop, a replay and a headless script all drive it
identically. Every step returns the events it produced; audio and visual effects
subscribe to that stream and never read engine state directly.

## Layout

```
src/tetris/
├── core/         pure logic — no pygame anywhere
│   ├── board.py      bitboard rows + a parallel colour grid
│   ├── constants.py  piece shapes, SRS kick tables, difficulty curve
│   ├── engine.py     step(action, dt), the single source of truth
│   ├── events.py     the event stream
│   ├── piece.py      SRS rotation and kicks
│   ├── randomizer.py 7-bag
│   └── rules.py      gravity, lock delay, scoring, T-spins
├── render/       one numpy compositor, shared by every consumer
│   ├── compositor.py the scene, painted in layers
│   ├── effects.py    bloom, scanlines, vignette, chromatic aberration
│   ├── font.py       procedural 5x7 bitmap font
│   ├── layout.py     scene geometry in cell units
│   ├── palette.py    the Miami palette — the only colour literals
│   ├── profiles.py   flat / arcade / arcade_max
│   └── display.py    the pygame window — the only module that opens one
├── audio/
│   ├── synth.py      oscillators, envelopes, the sound set, the music
│   ├── bank.py       loading, channel priority, playback
│   └── generate.py   render everything to assets/sfx
└── app/          the playable game
    ├── main.py       240 Hz fixed-timestep loop, scenes, panels
    ├── input.py      DAS/ARR and key bindings
    ├── particles.py  line-clear sparks
    ├── replay.py     recording and exact playback
    └── scores.py     persistent personal best
```

Three ideas hold this together:

**The engine knows nothing about rendering.** `core/` imports no pygame, opens
no window and touches no audio device, which is what makes it fast headless and
testable without a display.

**One renderer.** `Compositor.render()` produces every frame anyone sees, at
whatever cell size is asked for. Decoration is confined to the background layer
(painted over by the playfield well) or to global post-process passes, so it can
never obscure a cell.

**Effects subscribe to events.** Sound, screen shake and particles all consume
the engine's `GameEvent` stream. None of them reach into engine internals.

## Tests

```bash
uv run pytest
```

292 tests. The interesting ones are not the happy paths:

- **`test_srs.py`** — kick tables checked structurally (a transition and its
  reverse must negate, candidate for candidate), plus the I-piece floor kick and
  a rotation that must be refused.
- **`test_engine.py`** — the classic ways a Tetris implementation breaks: lock
  resets that let a piece float forever, hold used twice, both top-out
  conditions, and a ghost that disagrees with the drop. Plus a 5-seed fuzz run
  auditing board invariants after every one of 10,000 random actions.
- **`test_clear_delay.py`** — the clear pause, and the invariant that the board
  never holds a full uncollapsed row.
- **`test_audio.py`** — signal hygiene: no clipping, no DC offset, no click at a
  loop seam. Both real audio bugs in this project were found here rather than by
  listening.
- **`test_overlay.py`** — every panel line's rect must be disjoint from every
  other, across cell sizes and nine-digit scores.
- **`test_replay.py`** — a replay must reproduce its session byte for byte.
- **`test_profiles.py`** — no profile field may be dead config. Three separate
  bugs in this project were a setting that was declared, documented and never
  read by anything; this makes that impossible to ship.

```bash
uv run tetris bench
```

Guards the throughput floors: the engine runs ~330k steps/sec against a 50k
floor, and every visual profile clears 120 fps at 30 px cells.

## Requirements

Python 3.12, `numpy`, `pygame-ce`, `gymnasium`. Managed with `uv`.

## Notes

`render/randomize.py` and the compositor's `include_hud` flag are dormant. They
belong to the later RL phase and nothing in the game uses them.

`_archive_propaganda_filter/` holds an unrelated Chrome extension that occupied
this directory before the project started. It is kept only so nothing was
deleted.
