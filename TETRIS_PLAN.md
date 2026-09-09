# Tetris Simulator + Visual RL Agent — Plan

## Context

The goal is a two-step project: **(1)** a solid, bug-free Tetris simulator with a dark 90s Miami Vice arcade presentation, and **(2)** an RL agent that learns to play it from **visual** information only.

The premise for this session was that `Basic RL/` held old code for a dead Tetris simulator that needed auditing and repair. **It does not.** The audit found:

- `Basic RL/` contains a Chrome extension named **"Propaganda Filter for X"** — 9 files (`manifest.json`, `content.js`, `popup.{html,css,js}`, `background.js`, `generate-icons.html`, `README.md`, `ICONS_README.md`). Entirely unrelated to Tetris or RL.
- No git repository, no hidden files, no stashed work.
- No audio assets anywhere under `TOFIN/`; no file matching `*tetris*` anywhere under `~/Desktop`.
- Toolchain: `uv` and miniconda present. System Python is 3.14 with **no** numpy/pygame/gymnasium/torch installed.

Nothing to salvage or repair. **This is a greenfield build.** The old extension files get archived, not deleted.

### Decisions locked in with the user

| Decision | Choice |
|---|---|
| Stack | Python + `pygame-ce`, Gymnasium-style env API |
| Ruleset | Modern guideline (SRS + wall kicks, 7-bag, hold, ghost, lock delay) |
| Audio | Chiptune SFX + BGM synthesized procedurally in numpy, written to `.wav` |
| Old files | Move to `Basic RL/_archive_propaganda_filter/` |
| Visuals | **One** renderer, dark UI, Miami Vice palette and fonts, graded effects budget |

### Why this stack

Requirement 4 ("information can be fed in easily") and step 2 (learn from pixels) are the binding constraints. A Python engine puts the environment and the torch agent in one process: pixel observations are a numpy slice away, headless training runs fast with no display, and there is no browser bridge to keep in sync. `pygame-ce` (not stock `pygame`) is the maintained fork with better blit performance and current Python support.

---

## The visual-fidelity problem, and how this plan resolves it

The agent must be robust to the distortions that naturally occur in the real presentation, but training must stay fast. These pull in opposite directions, and an earlier draft of this plan resolved it badly — by giving the agent a separate sterile renderer. That is wrong: it trains the agent on a distribution it never actually plays in, and any robustness it appears to have is untested.

**The correct resolution is one renderer, one visual language, and three levers.**

### Lever 1 — structural clutter control (the real fix)

Clutter is controlled by *where decoration is allowed to live*, not by turning the art off:

> **Playfield legibility invariant:** decoration renders in the background layer (behind locked cells) or in a global post-process pass (scanlines, vignette, bloom, grade). **Nothing ever occludes an occupied cell**, and playfield contrast never drops below a fixed threshold.

This is enforced by a test, not by discipline (see M2). It means the arcade can look loud while the *information* stays clean — which is exactly the middle ground of "not too much visual clutter". Concretely: particle bursts are clipped to outside the playfield rect or drawn underneath locked cells; the animated sun and grid floor sit behind the board at reduced luminance; the board itself always renders high-contrast neon-on-near-black.

### Lever 2 — `VisualProfile`, a graded effects budget

One code path, one dial:

| Profile | Contents | Use |
|---|---|---|
| `flat` | Neon blocks on dark ground. No post-process. | Fast warmup / debugging only |
| `arcade` | Static scanlines, subtle bloom, dark gradient + dimmed sun/grid, no shake, no particles | **Default for both human play and training** |
| `arcade_max` | Everything: shake, particles, chromatic aberration, animated background | Showing off, and the hardest robustness eval |

The human and the agent both default to `arcade`. They are looking at the same thing.

### Lever 3 — domain randomization (this is what buys robustness)

Applied per-episode at train time, as cheap numpy ops in the post-process pass: brightness/contrast jitter, hue rotation *within* the Miami palette, scanline phase and intensity jitter, ±2 px board offset, bloom variance, mild sensor noise. Screen shake is deliberately included — it is free translation-robustness training, as long as the board stays fully in frame.

Ramped on a curriculum: near-zero early (fast initial learning), rising as training progresses (robustness). This is the honest answer to "account for natural distortion *and* learn fast" — you get both by ordering them in time, not by splitting the renderer.

### Why this is still fast

Two mechanics do the work:

1. **The compositor is numpy; pygame is only the display path.** Board cells are painted by array slicing; scanlines are a cached 1-D multiply; vignette a cached 2-D multiply; bloom a separable box blur of the bright mask. No pygame surface round-trip in the training loop, and no mixer or window init headless.
2. **Resolution independence.** All layout is in normalized units. The human window renders at 1080p; the agent's frame renders the *same scene with the same effects* at 168×168 native. Identical visuals, ~40× fewer pixels — the agent never pays for pixels it downsamples away.

Every static layer (background gradient, sun, grid, scanline mask, vignette, per-color glow sprites) is pre-rendered once at init and cached.

---

## Design principles

**1. The engine knows nothing about pygame.** `core/` is pure Python + numpy — importable, steppable, and testable with no display, no audio device, no window. This is what makes headless training fast and CI possible.

**2. One renderer, one truth.** Human and agent see the same scene through the same code, differing only in resolution and profile. See above.

**3. Logic time is decoupled from render time.** A fixed-timestep accumulator runs logic at 240 Hz; rendering happens at display refresh (vsync). Game feel is identical at 60, 120, or 144 Hz and DAS/ARR timings stay honest. Requirement 1 ("high tick rate") is a high *logic* tick rate, not merely a high frame rate.

**4. Effects subscribe to events, never to state.** The engine emits a `GameEvent` stream; audio and effects consume it. Neither ever reaches into engine internals.

---

## Visual identity — dark UI, Miami Vice

Single source of truth in `render/palette.py`. Everything is dark; neon is the only bright thing on screen.

| Role | Color |
|---|---|
| Ground / void | `#0D0221` near-black indigo |
| Panel fill | `#1A0B2E` |
| Panel stroke | neon, 1–2 px, always glowing |
| Primary | `#FF1E8E` hot magenta |
| Secondary | `#00F0FF` cyan |
| Tertiary | `#B026FF` electric purple |
| Accent | `#FF6B35` sunset orange |
| Warning / danger | `#FF2E63` |

Tetromino colors are drawn from this palette (not the standard Tetris colors) so the whole board reads as one piece of art, while staying maximally separable in hue for a CNN — deliberately checked as part of the legibility test.

**Fonts.** A **procedurally generated bitmap font atlas**, built at asset-generation time alongside the audio: chunky pixel glyphs, magenta→cyan vertical gradient fill, cyan glow outline, optional chrome bevel for headings. This is self-contained (no font licensing), pixel-crisp at any integer scale, matches the era exactly, and is resolution-independent like the rest of the renderer. A system-monospace chain is the fallback only if the atlas is unavailable.

---

## Target structure

```
Basic RL/
├── TETRIS_PLAN.md                   # living plan doc (copy of this file)
├── README.md                        # new: how to run, controls, env API
├── pyproject.toml                   # uv-managed, Python 3.12
├── .python-version
├── _archive_propaganda_filter/      # the 9 old extension files, moved
├── assets/
│   ├── sfx/                         # generated .wav files (not committed)
│   └── fonts/                       # generated bitmap font atlas
├── src/tetris/
│   ├── core/                        # PURE LOGIC — no pygame import anywhere
│   │   ├── constants.py             # piece shapes, SRS kick tables, gravity table
│   │   ├── board.py                 # bitboard rows, collision, line clear
│   │   ├── piece.py                 # active piece + rotation state
│   │   ├── randomizer.py            # 7-bag, seeded
│   │   ├── rules.py                 # scoring, t-spin, level curve, gravity
│   │   └── engine.py                # TetrisEngine.step() — single source of truth
│   ├── env/
│   │   ├── tetris_env.py            # Gymnasium env wrapping TetrisEngine
│   │   ├── observations.py          # rgb | grid | features encoders
│   │   └── wrappers.py              # frame stack, grayscale, resize, reward shaping
│   ├── render/
│   │   ├── palette.py               # Miami Vice palette — single source of color
│   │   ├── profiles.py              # VisualProfile: flat | arcade | arcade_max
│   │   ├── layout.py                # normalized, resolution-independent layout
│   │   ├── compositor.py            # numpy scene compositor (shared by both paths)
│   │   ├── layers.py                # background, board, HUD, foreground layers
│   │   ├── effects.py               # scanlines, bloom, vignette, shake, particles
│   │   ├── randomize.py             # domain randomization (train-time)
│   │   ├── font.py                  # procedural bitmap font atlas + text drawing
│   │   └── display.py               # pygame window/vsync — display path only
│   ├── audio/
│   │   ├── synth.py                 # numpy oscillators + ADSR -> .wav
│   │   ├── bank.py                  # SoundBank: load, mix, channel priority
│   │   └── events.py                # GameEvent -> sound mapping
│   ├── app/
│   │   ├── input.py                 # DAS/ARR, remappable keybinds
│   │   ├── scenes.py                # menu / game / pause / game-over
│   │   └── main.py                  # human-play entry point
│   └── cli.py                       # play | bench | gen-assets | replay
└── tests/
    ├── test_board.py
    ├── test_srs.py                  # kick-table conformance
    ├── test_rules.py                # scoring, t-spin, level curve
    ├── test_engine.py               # lock delay, hold, top-out
    ├── test_env.py                  # Gym API conformance + determinism
    ├── test_legibility.py           # THE playfield legibility invariant
    └── test_perf.py                 # steps/sec floor (guards requirement 1)
```

---

## Milestones

### M0 — Scaffolding
1. Create `Basic RL/_archive_propaganda_filter/` and move the 9 extension files into it. Nothing deleted.
2. Copy this plan to `Basic RL/TETRIS_PLAN.md` as the living doc; refine it there as work proceeds.
3. `uv init` + `uv venv --python 3.12` (3.12 has the most reliable wheels for `pygame-ce` + `torch`; the system 3.14 does not).
4. `pyproject.toml` deps: `numpy`, `pygame-ce`, `gymnasium`; dev extras `pytest`, `pytest-benchmark`. Torch deferred to step 2.
5. `git init` — the absence of version control is why the "old code" turned out to be unrecoverable.

**Done when:** `uv run python -c "import pygame; import gymnasium"` succeeds.

### M1 — Pure engine + test suite

The heart of the project. `core/`, zero pygame imports.

**`board.py` — bitboard representation.** The board is a `list[int]` of 20 rows plus buffer rows above, each row a 10-bit integer. Collision is a bitwise AND; a full row is `row == 0b1111111111`; line clear is a list filter. Faster than a 2-D array and much harder to get subtly wrong.

**`constants.py` — the data that must be exactly right.** Piece spawn shapes and orientations, the **JLSTZ kick table** and the **separate I-piece kick table** (O does not kick), and the gravity table.

**`rules.py`**
- Gravity: `seconds_per_row = (0.8 - (level - 1) * 0.007) ** (level - 1)` for levels 1–15; level 15+ is effectively 20G.
- Level up every 10 lines cleared — requirement 3, the difficulty curve.
- Lock delay 500 ms at level 1, shrinking to 200 ms by level 20, floored at 150 ms. This, not gravity alone, is what makes late levels genuinely hard.
- Scoring: single/double/triple/tetris = 100/300/500/800 × level; T-spin single/double/triple = 800/1200/1600; T-spin mini 100; back-to-back × 1.5; combo 50 × combo × level; soft drop 1/cell, hard drop 2/cell; perfect-clear bonus.
- T-spin by the 3-corner rule, with the mini distinction from front-corner occupancy and the SRS kick index used.

**`engine.py` — `TetrisEngine.step(action, dt)`.** Actions: `NOOP, LEFT, RIGHT, SOFT_DROP, HARD_DROP, ROT_CW, ROT_CCW, HOLD`. Returns an `EngineState` plus a list of `GameEvent`s.

**Requirement 5 ("plays nicely without bugs") is discharged here**, by testing precisely what classically breaks in Tetris implementations:
- SRS kicks: all 8 transitions × both tables, including the T-spin-triple kick and I-piece floor kicks.
- Lock delay **move reset**, capped at 15 resets — otherwise a piece can be held aloft forever.
- Hold usable exactly once per piece.
- Both top-out conditions: block out (spawn overlaps) and lock out (piece locks entirely above the ceiling).
- DAS charge preserved across piece spawn.
- Line clear interacting with lock in the same tick.
- Ghost-piece position identical to hard-drop landing position, always.

**Done when:** `pytest` green, and an ASCII debug renderer plays a full game to top-out via scripted actions.

### M2 — Renderer + Gymnasium env

Built together, because the legibility invariant is a property of both.

**Renderer.** `layout.py` defines everything in normalized units. `compositor.py` composites layers in numpy at a requested resolution under a requested `VisualProfile`. `display.py` is the only module that touches a pygame window. `font.py` generates and draws the bitmap font atlas.

**`test_legibility.py` — the invariant, mechanized.** Render a known board state at every profile, every randomization seed, and both resolutions; then run a trivial per-cell dominant-color classifier over the playfield region and assert it recovers the board exactly, with contrast above threshold. **If an effect breaks this test, the effect is wrong — not the test.** This is what makes "low visual clutter" an enforced property instead of an aesthetic opinion, and it is the single most important guard in the project.

**Env — requirement 4, the "feed information in easily" layer:**

```python
env = TetrisEnv(obs_type="rgb", profile="arcade", randomize=0.0,
                render_mode=None, seed=0, start_level=1)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- `obs_type="rgb"` — 84×84×3 (configurable) frames from the shared compositor. **Default for step 2.**
- `obs_type="grid"` — 20×10 binary occupancy + piece one-hots. A cheap non-visual baseline; invaluable for proving the learning code works before blaming the vision stack.
- `obs_type="features"` — holes, bumpiness, aggregate height, max height. For sanity-checking reward shaping only.
- `include_hud` — whether next/hold panels are in frame. Default on for next + hold, off for score text (digits are near-useless to a CNN and cost pixels).

`info` carries full telemetry each step: `lines_cleared, level, score, combo, b2b, tspin, holes, bumpiness, aggregate_height, piece_count`.

`wrappers.py` supplies frame-stacking, grayscale, and resize as composable wrappers. Frame stacking stays **out** of the env, so the env holds no hidden temporal state and stays trivially seedable.

**Determinism is a hard requirement:** same seed + same action sequence + `randomize=0` ⇒ byte-identical observations. With `randomize>0`, the randomization draws from the episode seed, so runs stay reproducible. Tested in `test_env.py`.

**Perf targets** (asserted in `test_perf.py`, so requirement 1 can't silently regress):
- engine-only `step()`: ≥ 50k steps/sec
- env, `obs_type="grid"`: ≥ 20k steps/sec
- env, `obs_type="rgb"`, `profile="arcade"` @168×168: ≥ 2k steps/sec
- env, `obs_type="rgb"`, `profile="flat"` @168×168: ≥ 8k steps/sec
- human mode: locked 60 fps render, 240 Hz logic

If `arcade` misses 2k/sec, the fix is profiling the compositor (likely the bloom blur) — not falling back to a separate sterile renderer.

### M3 — Playable arcade game

**`app/input.py` — requirement 2, classic controls.** Defaults, all remappable:

| Key | Action |
|---|---|
| ←/→ | move (DAS 133 ms, ARR 10 ms, both tunable) |
| ↓ | soft drop |
| Space | hard drop |
| ↑ / X | rotate CW |
| Z / Ctrl | rotate CCW |
| C / Shift | hold |
| Esc / P | pause |

**`app/scenes.py`** — attract/title → menu → game → pause → game over → high scores, as a small scene stack. Dark panels, neon strokes, gradient bitmap type throughout.

**Requirement 6 — the presentation.** Sunset-with-horizontal-slits sun behind an animated perspective grid floor, both dimmed and strictly behind the board. Neon glow on tetromino edges from pre-rendered per-color sprites. Cached scanline and vignette masks. Screen shake on hard drop and tetris; particle bursts and a chromatic-aberration flash on line clear — all clipped so they never occlude a locked cell. Every effect is individually toggleable and lives in the profile table, so the human and agent views stay in sync by construction.

### M4 — Procedural chiptune audio

**`audio/synth.py`** — numpy oscillators (square with duty cycle, saw, triangle, noise), ADSR envelopes, pitch sweeps, written to `assets/sfx/*.wav`. Generated once via `tetris gen-assets` (same command builds the font atlas), then cached. No downloads, no licensing questions, fully tunable.

Sound set: `menu_move`, `menu_select`, `menu_back`, `piece_move`, `piece_rotate`, `piece_lock`, `piece_hold`, `hard_drop`, `line_clear_1/2/3`, `tetris`, `t_spin`, `level_up`, `game_over`, `pause`. Plus a looping synthwave BGM — bassline + arpeggio from a small step sequencer, tempo scaling with level so the music tightens as difficulty rises.

**`audio/bank.py`** — channel allocation and priority so DAS-repeated move blips don't starve line-clear stingers. Missing files degrade to silent no-ops, never a crash. Headless mode initialises no mixer at all.

### M5 — Polish and hardening
- **Replay recording**: every run logs `(seed, action, dt)`. A replay reproduces a session exactly — turning "the game glitched" into a reproducible test case, which is the real defence for requirement 5.
- `--record` flag dumping frames to GIF/MP4.
- Difficulty-curve tuning pass by playtesting levels 1–20.
- `README.md`: controls, config, env API, and the visual-profile/randomization knobs.

### M6 — Step 2: the RL agent (sketched; planned properly in its own session)

M1–M5 are built so this drops in cleanly:
- Frame-stacked (4×) 84×84 grayscale observations, small CNN encoder.
- DQN family (Double + Dueling + PER) as baseline; PPO as the alternative.
- Reward: `lines_cleared ** 2` scaled, small survival bonus, top-out penalty. Optional shaping on holes/bumpiness delta **behind a flag** — it speeds learning a lot but is no longer purely visual learning, so it must be explicit opt-in, never a default.
- **Two curricula, both already exposed by M2's env config:** difficulty (`start_level`, `gravity_scale`) and visual (`randomize` 0 → 1, `profile` `arcade` → `arcade_max`). Ramp visual randomization only after the agent is reliably clearing lines — that ordering is what delivers fast learning *and* distortion robustness.
- Validate the training loop on `obs_type="grid"` first. If the agent can't learn from a clean 20×10 grid, the problem is the agent, not the vision stack — that separation saves days of misdirected debugging.
- **Robustness eval:** train at `randomize=0.5`, evaluate at `arcade_max` with `randomize=1.0`. The gap between those two scores is the honest measure of whether the distortion training worked.

---

## Verification

**Per-milestone gates:**
- M1: `uv run pytest tests/test_board.py tests/test_srs.py tests/test_rules.py tests/test_engine.py` green. The SRS kick tests matter most.
- M2: `test_legibility.py` green across all profiles × randomization seeds × resolutions. `test_env.py` green — Gym API conformance and determinism. `uv run tetris bench` meets the steps/sec floors.
- M3: `uv run tetris play` — I play it directly, drive it via the screenshot tooling, and confirm by screenshot that board, HUD, next/hold, ghost piece, and the dark neon presentation all render correctly. Manual pass: DAS feel, wall kicks against a wall, T-spin, tetris clear, pause/resume, top-out → game over → restart.
- M4: `uv run tetris gen-assets` writes every wav and the font atlas; play a session and confirm each event fires its sound with no channel starvation and no crackle.
- M5: record a replay, play it back, assert final state hashes match.

**End-to-end:** a scripted 10,000-step random-action headless run with no exception and no state-invariant violation (board never holds a full uncleaned row, piece never overlaps locked cells, score never decreases), then the same seed replayed in human mode producing an identical final score.

**Visual sanity, done honestly:** dump a contact sheet of agent observations at each profile and randomization level. If I can't read the board from the 84×84 frames by eye, neither can a CNN — and that's a bug in the renderer, caught before a single training run is wasted on it.

---

## Open item

**BGM scope.** The procedural sequencer is the most speculative piece of M4. If it doesn't sound good quickly, ship SFX-only and revisit — the SFX carry most of the arcade feel, and a bad loop is worse than none.
