# Finishing the Tetris simulator

> Living plan doc. Steps 0 and 1 are **done** — see the status note at the
> bottom for what changed and what is next.

## Context

`Basic RL/` originally held an unrelated Chrome extension ("Propaganda Filter for
X"), now archived in `_archive_propaganda_filter/`. Everything below it is new
work: a guideline Tetris engine, a Miami Vice renderer, and a playable game.

**Scope of this phase — the game, and nothing else.** No RL agent, no Gym
environment, no observation design, no reward shaping, no domain randomization.
The agent comes *after* the simulator is finished and needs nothing from us until
then. Two modules written earlier (`render/randomize.py`, and the `include_hud`
switch on the compositor) are dormant — they are harmless, they are not built on
here, and they are not part of this plan's work or its definition of done.

**Done and committed (M0–M3):**

- `core/` — guideline engine, pure Python + numpy, zero pygame. SRS with both
  kick tables, 7-bag, hold, ghost, lock delay with capped move resets, T-spins,
  back-to-back, combos. 315k steps/sec, 129 tests green.
- `render/` — one numpy compositor, dark Miami palette, `flat` / `arcade` /
  `arcade_max` profiles, procedural 5×7 bitmap font.
- `app/` — classic controls with DAS 133 ms / ARR 10 ms, 240 Hz fixed-timestep
  loop decoupled from display refresh.

**What prompted this round.** Playing it surfaced a visible bug, and an audit
surfaced a shipped one:

1. **The game-over score is drawn on top of the `GAME OVER` text.** Two separate
   pieces of code each draw a game-over message and neither knows about the
   other — `render/compositor.py:373` centres a banner on the **board** (plate
   spans y 358–423 at `cell=30`); `app/main.py` `_present` centres `SCORE n` on
   the **frame** (y 405–426). They overlap by 18 px, and the score passes
   `dim=False` so it also fights the lit board behind it.
2. **`tetris bench` crashes** — `cli.py:60` imports `tetris.bench`, never written.
3. `tests/test_perf.py` was promised and never written, so the tick-rate
   requirement is unguarded.

### Against the original seven requirements

| # | Requirement | State |
|---|---|---|
| 1 | High tick rate | done (240 Hz logic) — needs a perf guard |
| 2 | Classic controls | done |
| 3 | Progressively harder | implemented — needs a real playtest pass |
| 4 | Easy to feed information in | `TetrisEngine.step(action, dt)` + `telemetry()` |
| 5 | Plays nicely, no bugs | 129 tests; game-over bug open |
| 6 | 90s Miami Vice arcade UI | looks right; **no animation yet** |
| 7 | Sound effects | **nothing exists** |

The two real gaps to a finished game are **sound** and **motion** — line clears
currently blink out instantly, which is the biggest thing standing between this
and feeling like an arcade cabinet.

---

## Step 0 — the game-over panel

### Take the banner out of the compositor

Delete the `if engine.game_over:` branch and `_draw_banner` from
`render/compositor.py` (lines 372–395); trim line 30 to
`from .font import draw_text, draw_text_centered` (`GLYPH_W` and `text_size`
become unused there — `_stroke` stays, used at 208 and 243).

Fixing it here rather than nudging coordinates: title, paused and game over are
all scene presentation, which already lives in the app layer — this one piece had
escaped. It also means the compositor draws only the game itself, never a message
over it.

### `fit_scale` in `render/font.py`

```python
def fit_scale(text: str, max_width: int, preferred: int, tracking: int = 1) -> int:
    """Largest scale <= preferred whose rendered width fits max_width."""
```

Steps down from `preferred` using the existing `text_size`. Every panel line goes
through it, so nothing can overflow at any `cell_px` or score magnitude.

### One measured panel drawer in `app/main.py`

Replace `_overlay`. The collision happened because both draws hard-coded their own
centres; laying lines out from a running cursor over measured heights makes
overlap structurally impossible.

```python
@dataclass(frozen=True, slots=True)
class PanelLine:
    text: str
    size: str          # "hero" | "title" | "body" | "label"
    color: RGB
    gap_before: float = 0.0    # in cell units
```

`_draw_panel(frame, lines, dim=0.30)` resolves each `size` to a scale
(`hero = max(3, cell // 5)` down to `label = max(1, cell // 14)`), runs it through
`fit_scale`, measures the block, dims the frame, fills a plate **centred on
`layout.board`** (not the frame — that is what kept the old banner off the side
panels), strokes it neon, then draws top-down from a cursor.

Game over, with the score as the largest element on screen:

```
GAME OVER               title,  magenta
SCORE                   label,  dim        (gap above)
128,450                 HERO,   cyan
NEW RECORD              body,   magenta    (only when beaten)
BEST 204,900            body,   dim
LINES 42   LEVEL 5      body,   dim
ENTER RESTART  Q QUIT   label,  dim        (gap above)
```

Title and Paused reuse the drawer. Thousands separators throughout.

### `app/scores.py` — new

```python
@dataclass
class HighScores:
    best: int = 0
    best_lines: int = 0
    best_level: int = 1
    games: int = 0

def load(path: Path | None = None) -> HighScores
def save(scores: HighScores, path: Path | None = None) -> None
def submit(scores: HighScores, stats: Stats) -> bool   # True on a new record
```

Stored at `~/.tetris-rl-highscore.json`, not in the repo. **Every read and write
wrapped in try/except** — a missing, corrupt or unwritable file degrades to an
in-memory score and the game carries on. Wire into `Game.__init__` (load) and the
`EventType.GAME_OVER` branch of `_react` (submit, stash `self._new_record`, save).

### Tests

`tests/test_scores.py` — round-trip via `tmp_path`; missing file yields defaults;
**corrupt file yields defaults without raising**; `submit` returns True only on a
genuine beat.

`tests/test_overlay.py` — the mechanized version of the bug just found, so it
cannot return: build the game-over line list across several `cell_px` values and
scores up to nine digits, then assert **every line's rect is disjoint from every
other** and all sit inside the plate, which sits inside the board.

---

## Step 1 — loose ends

- **`src/tetris/bench.py`** — the missing module, so `tetris bench` works.
  `run_benchmarks(steps)` times engine stepping and full-frame rendering at each
  profile, printing measured rates against targets.
- **`tests/test_perf.py`** — guard requirement 1: engine ≥ 50k steps/sec
  (currently 315k) and a full `arcade` frame at human scale comfortably inside a
  60 Hz budget. Generous thresholds — this catches an order-of-magnitude
  regression, not noise.
- **Dead code** — the unused `argv` parameter at `app/main.py:236`; the redundant
  `engine.stats.level = self._start_level` in `_new_game` (`reset` already applies
  it from config).
- **`InputState.on_piece_locked()`** is deliberately a no-op — DAS charge survives
  a lock by *not* being reset. Add the test that asserts the behaviour its name
  claims, so the emptiness is verified rather than merely commented.
- **Reconcile plan with reality** — `app/scenes.py` and `render/layers.py` were
  folded into `main.py` and `compositor.py`; that was right at this size. Update
  `TETRIS_PLAN.md` to match rather than creating empty modules.

---

## Step 2 — game feel

The largest gap between this and an arcade cabinet. Lines currently vanish
between one frame and the next.

### Line-clear delay in the engine

Guideline Tetris pauses ~400 ms on a clear, and that pause is what the animation
lives in. This is a real gameplay change, so it goes in the engine rather than
being faked by the renderer.

At lock: award score and count lines immediately (as now), **blank the cleared
rows but do not collapse them**, and record `_pending_collapse: list[int]` with a
`_clear_timer`. No piece is active during the delay. When the timer expires the
stack collapses and the next piece spawns.

Blanking rather than deferring the clear matters: it keeps `Board.check_invariants`
honest — no full row ever sits uncollapsed — and it is what the player sees
anyway (the row empties, then the stack falls).

`step()` must advance the clear timer while `piece is None`, which is a change
from the current early return. `GameEvent` gains an optional
`rows: tuple[int, ...]` payload so `LINE_CLEAR` can say *which* rows went, which
the renderer needs. `clear_delay_ms` joins `rules.py` beside the gravity and
lock-delay curves, shrinking slightly at high level so late play stays tense.

Existing tests that hard-drop and immediately assert `stats.lines` keep passing,
because scoring still happens at lock; a handful that assume the next piece exists
immediately after a clearing drop will need a timer advance, which is the correct
new behaviour.

### Renderer and app

- **Clear flash** — cleared rows blaze white then fall away over the delay.
- **Collapse** — the stack above eases down rather than snapping.
- **Lock-delay pulse** — the active piece brightens as its lock timer runs out.
  Currently there is no feedback at all for the most timing-sensitive moment in
  the game.
- **Particles** on clears and **screen shake** already exist for `arcade_max`;
  wire particles to the clear rows and keep them clipped so they never cover a
  cell.
- **Level-up flourish** — a brief banner and a palette pulse, so requirement 3 is
  something you feel rather than read off a number.

---

## Step 3 — sound

Everything synthesized in numpy — oscillators (square with duty cycle, saw,
triangle, noise), ADSR envelopes, pitch sweeps — written once to
`assets/sfx/*.wav` by `tetris gen-assets` and cached. No downloads, no licence to
track, fully tunable.

Set: `menu_move`, `menu_select`, `menu_back`, `piece_move`, `piece_rotate`,
`piece_lock`, `piece_hold`, `hard_drop`, `line_clear_1/2/3`, `tetris`, `t_spin`,
`level_up`, `game_over`, `pause`. Plus a looping synthwave bassline and arpeggio
whose tempo tightens as the level rises.

`audio/bank.py` handles channel priority so DAS-repeated move blips cannot starve
a line-clear stinger, degrades to silent no-ops on a missing file, and initialises
no mixer at all when there is no audio device. It subscribes to the existing
`GameEvent` stream — no new plumbing, and the clear-delay work in step 2 gives the
line-clear stinger room to land.

*Risk:* the BGM sequencer is the speculative part. If it does not sound good
quickly, ship SFX-only and revisit — a bad loop is worse than silence.

---

## Step 4 — finishing

- **Difficulty playtest** — actually play levels 1–20 and tune the gravity and
  lock-delay curves. Requirement 3 cannot be verified by a unit test.
- **Attract / title screen** — currently text over a live board; give it a proper
  layout with the high score and a demo stack.
- **Replay recording** — log `(seed, action, dt)`. A replay reproduces a session
  exactly, turning "it glitched" into a reproducible test case, which is the real
  long-term defence for requirement 5.
- **`README.md`** — how to run, controls, profiles, and how to drive the engine
  programmatically (requirement 4).

### Definition of done

Sound on every meaningful event; line clears animate; the difficulty curve has
been played to level 20 and tuned; `pytest` green; `tetris play`, `tetris shot`
and `tetris bench` all work; README written. At that point the simulator is
finished and the agent phase can begin from a stable base.

---

## Verification

- `uv run pytest tests/` — 129 existing tests green, plus new score, overlay,
  perf, clear-delay and input tests.
- `uv run tetris bench` — runs and prints numbers instead of crashing.
- `uv run tetris shot /tmp/gameover.png --pieces 40 --seed 4`, then read the PNG:
  `GAME OVER`, the hero score and the stat lines cleanly separated, score visibly
  the largest element, plate inside the board.
- The same panel at `--cell 12` and `--cell 40` and with a forced nine-digit
  score, confirming `fit_scale` holds at both extremes.
- Headless (`SDL_VIDEODRIVER=dummy`): play to a game over, confirm the panel
  renders and the record is written; restart and confirm `BEST` shows it.
- Delete, then corrupt, `~/.tetris-rl-highscore.json` — the game starts normally
  both times.
- Frame-by-frame dump of a line clear, read back as images, to confirm the flash
  and collapse actually read as motion rather than a stutter.
- `uv run tetris play` — real games, to a genuine top-out, at low and high level.


---

## Status — steps 0 and 1 complete

**Step 0, the game-over panel.** The `GAME OVER` banner is gone from the
compositor, which now draws the game and never a message over it. `app/main.py`
lays every overlay line out from a running cursor over measured heights, so
lines cannot collide; the score is the `hero` size and dominates the panel.
`render/font.fit_scale` shrinks any line that would overflow, and the plate is
clamped to the board. High scores persist via `app/scores.py`, written
atomically and degrading to an in-memory score on any file problem.

Two bugs surfaced while checking the render:

- the font had **no comma glyph**, so every score rendered as `128?450`. Added
  `,`, `'` and `%`.
- the side stats panel did not use `fit_scale`, so adding thousands separators
  pushed a seven-figure score past the panel edge. It now shrinks to fit.

**Step 1, loose ends.** `src/tetris/bench.py` was missing entirely, so
`tetris bench` crashed on every invocation — now written and working. Running it
immediately found a real performance problem: the default `arcade` profile
rendered at **67 fps**, barely above 60 and dropping frames on a high-refresh
display. Profiling showed **bloom was 10 ms of a 14 ms frame**, blurring at full
resolution. Bloom is low-frequency by nature, so it now works on a strided view
at 1/4 resolution: **67 → 145 fps**, with a mean channel difference of 0.67 and
no visible change.

Also: `tests/test_perf.py` guards those floors, `tests/test_input.py` covers DAS
and ARR (including the DAS-charge-survives-a-lock behaviour that
`on_piece_locked` exists to name), and the dead `argv` parameter and redundant
level assignment are gone.

Test count: **129 → 193**.

### Structure, as actually built

`app/scenes.py` and `render/layers.py` from the original plan were folded into
`app/main.py` and `render/compositor.py`. At this size that was the right call —
scenes are a four-value enum and a line list, and splitting the compositor's
layers across modules would have meant passing the frame around for no gain.
Panels now added: `app/scores.py`, `src/tetris/bench.py`.

### Next: step 2, game feel

Line-clear delay in the engine, then the clear flash, collapse animation,
lock-delay pulse and level-up flourish. Then step 3, sound.
