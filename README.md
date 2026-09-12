# BLOCKWAVE

A guideline-compatible falling-block arcade game with a dark 90s Miami neon
presentation. Written in Python, rendered in numpy, and **entirely generated** —
there is not a single image, font or audio file in this repository. The
graphics are painted per-frame into arrays, the typeface is a procedural bitmap
atlas, and every sound effect and music track is synthesized from oscillators at
build time.

```bash
uv sync --extra dev        # install
uv run blockwave gen-assets  # synthesize sounds and music (once)
uv run blockwave play        # play
```

---

## Contents

- [Controls](#controls) · [Commands](#commands)
- [The ruleset](#the-ruleset) · [Difficulty](#difficulty)
- [Architecture](#architecture) · [Rendering](#rendering) · [Audio](#audio)
- [Replays](#replays) · [Driving the engine](#driving-the-engine-directly)
- [The learning agent](#the-learning-agent)
- [Tests](#tests) · [Requirements](#requirements) · [License](#license)

---

## Controls

| Key | Action |
|---|---|
| ← / → | Move |
| ↓ | Soft drop |
| **Space** | Hard drop |
| ↑ or X | Rotate clockwise |
| Z or Ctrl | Rotate anticlockwise |
| C or Shift | Hold |
| Esc or P | Pause |
| Enter | Start / restart / resume |
| Q | Quit to title |

Auto-shift defaults to **DAS 133 ms, ARR 10 ms**, both tunable in
`app/input.py`. The DAS charge deliberately survives a piece locking, so holding
a direction slides each new piece immediately rather than stalling for a fresh
delay on every spawn — the single most common reason a falling-block game feels
sluggish.

## Commands

```bash
blockwave play  [--level N] [--profile P] [--cell N] [--mute] [--no-music] [--record FILE]
blockwave shot  OUT.png [--pieces N] [--seed N] [--profile P] [--cell N]
blockwave gen-assets [--out DIR] [--no-music]
blockwave replay FILE
blockwave bench [--steps N]
```

`--cell` sets the pixel size of one board cell. The whole scene is laid out in
cell units, so this is the only knob needed to scale the window.

**Visual profiles** (`--profile`):

| Profile | What it adds |
|---|---|
| `flat` | Neon blocks on a dark ground. No post-processing, completely still. |
| `arcade` *(default)* | Scanlines, bloom, vignette, the sun and grid floor, screen shake. |
| `arcade_max` | All of the above, harder shake, chromatic aberration, particle bursts. |

Every field on a profile has to visibly change the frame — `tests/test_profiles.py`
renders each one on and off and asserts the output differs, so a setting that
does nothing cannot ship.

---

## The ruleset

Modern guideline behaviour throughout.

**Rotation — SRS with both kick tables.** Each piece has four explicit rotation
states rather than a matrix rotated at runtime. When a rotation collides, five
candidate offsets are tried in order and the first that fits wins; if all five
collide the rotation is refused. The I piece has its own kick table, separate
from the one J/L/S/T/Z share — conflating them is the most common SRS bug, and
the tests check both tables structurally as well as by example.

**Piece selection — 7-bag.** Every permutation of the seven pieces is dealt
before any repeats, which bounds the worst-case drought at 12 pieces. That bound
is what keeps a lost game a bad decision rather than bad luck.

**Hold** — once per piece, with the preview dimmed while it is spent.
**Ghost piece** — computed from the same function the hard drop uses, so it can
never disagree with where the piece actually lands.
**Preview** — five pieces deep.

**Lock delay — Extended Placement.** A landed piece stays movable for the lock
delay, and each successful move or rotation resets that window, capped at **15
resets** so a piece cannot be juggled on the stack forever. The counter is
restored whenever the piece falls below every row it has previously occupied, so
adjusting on a ledge and then dropping into a well does not leave you stranded
with no budget at the bottom.

**T-spins — the three-corner rule.** A T-spin requires a T piece, a rotation as
the last action, and at least three of the four corners of its bounding box
occupied (walls and floor count). It scores as a *full* spin when both front
corners are filled, or when the rotation only fitted via the final kick
candidate — that last case is what makes T-spin triples score correctly.

**Line clears pause the board.** Cleared rows blank immediately, the board holds
for the clear delay while they wipe out from the centre, then the stack
collapses and the next piece arrives. Scoring happens at the lock, so you are
paid at the moment of the placement.

### Scoring

| Clear | Base | With T-spin |
|---|---|---|
| Single | 100 | 800 |
| Double | 300 | 1200 |
| Triple | 500 | 1600 |
| Quad (four lines) | 800 | — |
| No lines | — | 400 (mini: 100) |

All multiplied by the current level. **Back-to-back** ×1.5 for consecutive
"difficult" clears (a quad, or any T-spin that clears lines) — a placement that
clears nothing leaves the chain intact; only an easy clear breaks it.
**Combo** adds 50 × combo × level. **Perfect clear** pays 800 / 1200 / 1800 /
2000 on top. Soft drop scores 1 per cell, hard drop 2.

## Difficulty

The level rises every **10 lines**. Three curves tighten together:

| Level | Gravity (s/row) | Lock delay | Clear delay | Time per piece |
|---|---|---|---|---|
| 1 | 1.000 | 500 ms | 400 ms | 18.50 s |
| 5 | 0.355 | 437 ms | 354 ms | 6.83 s |
| 10 | 0.064 | 358 ms | 296 ms | 1.51 s |
| 15 | 0.007 | 279 ms | 238 ms | 0.41 s |
| 18 | instant (20G) | 232 ms | 203 ms | 0.23 s |
| 20 | instant (20G) | 200 ms | 180 ms | 0.20 s |
| 25 | instant (20G) | 150 ms | 150 ms | 0.15 s |

Gravity follows the guideline curve, `(0.8 - (level-1) × 0.007) ^ (level-1)`,
until it crosses an instant-drop threshold at **level 18** — from there a piece
is on the stack the moment it spawns, and the lock delay is the only thing still
ramping. The curve was measured rather than assumed: the steepest single-level
drop in time-per-piece is to 73%, a consistent geometric ramp with no cliff, and
the transition into 20G costs only 21% of the budget.

Worth knowing, because it surprises people: at level 1 the lock delay is 2.7% of
a piece's life; at level 15 it is **68.7%**, and past 18 it is all of it. The
mechanic never changes — its share of the game does. Past level 25 the curves sit
at their floor and it becomes a score run.

The music steps through four tempo bands (116 → 165 bpm) as the level rises.

---

## Architecture

```
src/blockwave/
├── core/         pure logic — imports no pygame anywhere
│   ├── board.py      bitboard rows + a parallel colour grid
│   ├── constants.py  piece shapes, SRS kick tables, the difficulty curve
│   ├── engine.py     step(action, dt) — the single source of truth
│   ├── events.py     the event stream
│   ├── piece.py      SRS rotation and kicks
│   ├── randomizer.py 7-bag
│   └── rules.py      gravity, delays, scoring, T-spins
├── render/       one numpy compositor
│   ├── compositor.py the scene, painted in layers
│   ├── effects.py    bloom, scanlines, vignette, chromatic aberration
│   ├── font.py       procedural 5×7 bitmap font
│   ├── layout.py     scene geometry, in cell units
│   ├── palette.py    the Miami palette — the only colour literals in the repo
│   ├── profiles.py   flat / arcade / arcade_max
│   └── display.py    the pygame window — the only module that opens one
├── audio/
│   ├── synth.py      oscillators, envelopes, the sound set, the music
│   ├── bank.py       loading, channel priority, playback
│   └── generate.py   render everything to assets/sfx
└── app/          the playable game
    ├── main.py       240 Hz fixed-timestep loop, scenes, overlay panels
    ├── input.py      DAS/ARR and key bindings
    ├── particles.py  line-clear sparks
    ├── replay.py     recording and exact playback
    └── scores.py     persistent personal best
```

Three ideas hold it together.

**The engine knows nothing about presentation.** `core/` imports no pygame,
opens no window and touches no audio device. It runs headless at ~300k
steps/sec, which is what makes the fuzz tests and the benchmark possible.

**Time is passed in, never read.** `step(action, dt)` takes the timestep
explicitly, so the game loop, a replay and a headless script all drive the engine
identically — and a replay is guaranteed to reproduce the session it recorded.

**Effects subscribe to events.** Every `step` returns the events it produced.
Sound, screen shake, particles and the on-screen toasts all consume that stream;
none of them reach into engine state.

### The game loop

Logic advances in fixed **240 Hz** slices on an accumulator, decoupled from
rendering, which happens at whatever the display refreshes at. The game feels
identical on a 60 Hz laptop panel and a 144 Hz monitor, and DAS timings stay
honest on both. A frame delta is clamped so that a stall — dragging the window,
waking from sleep — cannot fast-forward the game through several pieces.

## Rendering

The whole scene is described **once in cell units** and scaled by a single
`cell_px`, so a 30 px arcade window and a 12 px one run identical layout code.

A frame is composited entirely in numpy; pygame only appears at the moment the
finished array is put on screen. The background, panels and well never change,
so they are rendered once at startup and cached — a frame costs one array copy,
the cell paint, and the post-process pass. The board is painted by turning the
colour grid into pixels with two `np.repeat` calls rather than a per-cell loop.

Layers are ordered so that **decoration can never obscure a cell**:

1. **Background** — gradient, sun, perspective grid. Painted first, then the
   playfield well is laid *over* it, so background art is physically incapable of
   reaching a cell.
2. **Chrome** — panels, borders, labels. All outside the playfield rect.
3. **Cells** — ghost, locked stack, active piece. The information layer.
4. **Post-process** — bloom, scanlines, vignette, shake. Global passes that
   modulate the whole frame uniformly.

Particles obey the same rule by construction: they are drawn only *outside* the
playfield rect, so a spark can never land on a block.

Bloom runs at quarter resolution on a strided view. It is a low-frequency effect
by nature, and blurring at full resolution cost 10 ms of a 14 ms frame — enough
to leave the default profile barely above 60 fps.

## Audio

Everything is synthesized in numpy: pulse waves with a duty cycle, saw, triangle
and white noise, shaped by ADSR envelopes and geometric pitch sweeps. Twenty
effects and five music loops, about 3.7 MB, written to `assets/sfx` by
`blockwave gen-assets`. No downloads, no licence to track, and every timbre is a
number someone can change.

`bank.py` enforces **channel priority**. Auto-repeat fires a move blip every
10 ms, and a naive bank lets those eat every mixer channel so the line-clear
stinger — the one sound that carries information — is the one that gets dropped.
Anything reporting a state change outranks piece chatter.

Audio degrades to silence rather than failing: no device, no generated files, or
a mixer that will not open all leave the game running. `--mute` and `--no-music`
are there too.

## Replays

```bash
blockwave play --record run.json
blockwave replay run.json
```

The engine is deterministic given a seed, an action sequence and a fixed
timestep, so a replay reproduces a session exactly — checked against a digest of
the final board and stats. That turns "the game glitched" into a file, and a
glitch into a regression test.

Recordings are sparse: a 240 Hz loop is almost entirely idle ticks, so only
ticks that carried an action are stored and a full game is a couple of kilobytes.
Playback and the live game share the same `apply_tick` function — if each had its
own idea of how a tick is applied, a replay could quietly diverge from the
session it claims to reproduce.

## Driving the engine directly

```python
from blockwave.core.engine import Engine, EngineConfig
from blockwave.core.constants import Action

engine = Engine(EngineConfig(seed=42, start_level=1))

events = engine.step(Action.HARD_DROP, dt=1 / 60)
print([e.type.name for e in events])
print(engine.telemetry())   # score, lines, level, holes, bumpiness, heights...
print(engine.to_ascii())    # the board as text
```

`telemetry()` returns score, lines, level, combo, back-to-back, pieces placed,
T-spins, quads, perfect clears, holes, bumpiness, aggregate and max height.

## The learning agent

`src/blockwave_rl/` is a research package: an agent that learns to play from a
reward it computes for itself, and **never sees the score**. Score, lines and
level never reach its observations, its reward, its gradients or the choice of
which checkpoint to report. Only the evaluator reads them, and a static test
fails if any other module does. The package depends on the game, never the
reverse, and `torch` is an optional extra, so installing the game does not pull
it in.

```bash
uv sync --extra dev --extra rl
uv run python -m blockwave_rl.train --out runs/demo --agent-hz 5 --steps 10000000   # ~15-20 min on an M4 Pro
uv run python -m blockwave_rl.evaluate runs/demo drift random --agent-hz 5 --every 8
uv run python -m blockwave_rl.watch runs/demo/ckpt_00813.pt   # one game as video; needs ffmpeg
```

The pixel track has its own chain — representation, world model, training —
each stage gated before the next begins:

```bash
uv run python -m blockwave_rl.repr.pixel_gate --out runs/pixel    # encoder + probe thresholds
uv run python -m blockwave_rl.world.fit --out runs/pixel          # learn what the keys do
uv run python -m blockwave_rl.world.rank --out runs/pixel         # must beat every exploit
uv run python -m blockwave_rl.world.train_pixel --out runs/pixel_ppo --reward runs/pixel
```

The reward is **empowerment**: how many distinct futures the agent's key
presses can still reach. The agent presses real keys, one per decision.

There are two tracks. The **board-state** track reads true occupancy and counts
futures exactly with the simulator — a feasibility check for the reward. After
150M steps it clears 0.16 lines per piece and survives ~70 pieces per game,
still improving when the run ended.

The **pixel** track is the deliverable: the agent sees only the screen, so it
cannot borrow the simulator either. It learns what its own key presses do by
trying them, and counts futures through that model. At 50M steps it clears 0.090
lines per piece and survives 50.6 pieces per game — within a few percent of the
privileged board-state agent at the same budget, from frames alone.

A scripted heuristic clears 0.39 and never tops out; the strongest trivial
baseline clears 0.002 and survives 20.

[RESEARCH.md](RESEARCH.md) records every approach tried, including the ones
that failed and why.

## Tests

```bash
uv run pytest       # 254 test functions, 432 cases after parametrisation (129 for the agent)
uv run blockwave bench
```

The interesting ones are not the happy paths:

| File | What it defends against |
|---|---|
| `test_srs.py` | Kick-table transcription errors — every transition and its reverse must negate, candidate for candidate. Plus the I floor kick and a rotation that must be refused. |
| `test_engine.py` | The classic failures: lock resets that let a piece float forever, hold used twice, both top-out conditions, a ghost that disagrees with the drop. Plus a 5-seed fuzz run auditing board invariants after every one of 10,000 random actions. |
| `test_clear_delay.py` | The clear pause, and the invariant that the board never holds a full uncollapsed row. |
| `test_audio.py` | Signal hygiene — no clipping, no DC offset, no click at a loop seam. Both real audio bugs in this project were found here rather than by listening. |
| `test_profiles.py` | Dead config. Three separate bugs here were a setting declared, documented and never read by anything. |
| `test_overlay.py` | Overlapping panel text, across cell sizes and nine-digit scores. |
| `test_replay.py` | A replay must reproduce its session byte for byte. |
| `test_perf.py` | Throughput floors, so performance cannot rot silently. |
| `rl/test_firewall.py` | Score leaking to the agent. Boards that differ only in score must give byte-identical observations and rewards, and no agent module may read score, lines or level. |
| `rl/test_adversarial.py` | Reward hacking, checked before any training. Exploit policies that beat a reward are recorded as strict xfails rather than hidden. |
| `rl/test_empowerment.py` | The reward's properties, including a flaw a learner found (dying fast paid) and the fix: one more placement alive is never worth less than dying now. |

`bench` measures engine steps/sec and per-profile frame rates against those
floors.

## Requirements

Python 3.12+, `numpy`, `pygame-ce`. Managed with [uv](https://docs.astral.sh/uv/);
`requirements.txt` and `requirements-dev.txt` are provided for plain pip. The
agent additionally needs `torch` (the `rl` extra), and `ffmpeg` to write videos.

```bash
pip install -r requirements-dev.txt && pip install -e .
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

## Notes

BLOCKWAVE implements the publicly documented "guideline" behaviours common to
modern falling-block games. It is an independent project, not affiliated with or
endorsed by The Tetris Company, and uses no assets, name or branding from any
commercial game.

[DESIGN.md](DESIGN.md) records the design decisions and the bugs that changed
them.
