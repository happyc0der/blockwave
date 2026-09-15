# Design notes

Why BLOCKWAVE is built the way it is, and the bugs that changed it. The
[README](README.md) covers what it does; this covers the decisions.

---

## The engine imports no pygame

`core/` is pure Python and numpy. It opens no window, touches no audio device,
and reads no clock.

That constraint pays for itself three times over. The engine runs headless at
~300k steps/sec, which makes fuzz testing cheap enough to run 10,000 random
actions per seed in the normal test suite. Every rule is testable without a
display, so CI needs no graphics stack. And because `step(action, dt)` takes its
timestep as an argument rather than reading a clock, the game loop, a replay and
a headless script drive the engine through *identical* code paths — which is
what makes exact replay reproduction possible at all.

## The board is stored twice

A bitboard (`list[int]`, one integer per row) for collision, and a parallel
`uint8` colour grid for rendering.

Collision is the hot path and it becomes a bitwise AND; a full row is an equality
test against `0b1111111111`. The colour grid is only touched when a piece locks
or lines clear, so it costs nothing per step.

The obvious risk is the two drifting apart, so `Board.check_invariants()` asserts
they agree and the test suite calls it after every mutation — including after
each of the 10,000 steps in the fuzz runs.

## One renderer, and decoration that cannot occlude

There is a single compositor. A visual profile does not select a different
renderer; it only decides how much decoration this frame may spend.

What keeps the board readable is not restraint but **layer order**. The
background is painted first and the playfield well is laid *over* it, so
background art is physically incapable of reaching a cell. Chrome lives outside
the playfield rect. Post-process passes are global and modulate the whole frame
uniformly. Particles are drawn only outside the playfield rect.

Every one of those is a structural guarantee rather than a tuning exercise. It
means the arcade look can be as loud as it likes without ever costing legibility.

## Line clears pause the board

Originally a cleared line vanished between one frame and the next. That is the
single biggest thing separating a falling-block game from an arcade cabinet, and
it cannot be fixed in the renderer: by the time the renderer sees the board, the
stack has already collapsed and there is nothing left to animate in the right
place.

So the pause lives in the engine. At lock, score is awarded and the cleared rows
are **blanked but not collapsed**; a timer runs; then the stack falls and the
next piece spawns. Blanking rather than deferring matters — it keeps
`check_invariants` honest, since the board is never sitting on a full uncollapsed
row, and it is what the player sees anyway.

`LINE_CLEAR` gained a `rows` payload, because a count cannot tell the renderer
what to animate.

## Effects subscribe to events

Every `step` returns the events it produced. Sound, screen shake, particles and
the on-screen toasts all consume that stream. None of them read engine state.

This is why adding audio required no new plumbing at all — `bank.py` maps events
to sounds and that was the entire integration.

---

## Bugs that changed the design

### Two things drew the same message

The game-over score was rendered on top of the "GAME OVER" text. The cause was
structural: the compositor drew a banner centred on the *board* while the app
drew the score centred on the *frame*, and neither knew the other existed.

Moving coordinates would have papered over it. The real fix was that scene
presentation belongs to the app layer — title and paused already lived there, and
this one piece had escaped. The compositor now draws the game and never a message
over it, and one panel drawer lays every line out from a running cursor over
measured heights, so overlap is impossible regardless of text length or cell
size.

`test_overlay.py` asserts every line's rect is disjoint from every other, across
four cell sizes and scores up to nine digits.

### Dead config, three times

A field gets added to `VisualProfile`, documented, given per-profile values — and
the code that should read it never lands. Nothing fails, because a setting that
does nothing looks exactly like a setting that is switched off.

It happened with `particles` (declared, unimplemented), then with `shake`
(declared, never read — every profile shook identically, including the one named
`flat`), and then again when particles were re-implemented gated on `shake`,
which itself did nothing.

`test_profiles.py` closes the class: for every field, render a frame with it off
and one with it on and assert the frames differ. A field that changes no pixel is
dead by definition. A companion test asserts the field list covers the dataclass,
so a newly added field that nobody wired up fails immediately.

### Two audio bugs found by arithmetic, not by ear

The sound is synthesized, and it was written by someone who cannot hear the
output. So the tests check what arithmetic can:

- **DC offset on 11 of 20 sounds.** A pulse wave at 22% duty sits at +1 for 22%
  of each cycle and -1 for the rest — a mean of -0.56. That thumps at the start
  and end of every sound and eats headroom. Fixed by centring the oscillator for
  any duty cycle.
- **A click at the music loop seam.** The percussive envelope had a genuinely
  instant attack, so the loop jumped 0.389 between its last and first sample —
  once per bar cycle, for the whole session. A 1 ms ramp cut it to 0.011.

`test_audio.py` now checks every sound for NaNs, clipping, DC offset and a
non-zero start, and every music track for a continuous loop seam.

### The move-reset budget never recovered

Guideline Extended Placement restores the fifteen-move counter when a piece falls
below every row it has occupied. This implementation only reset it on spawn.

The consequence was subtle and unfair: adjust a piece on a ledge, exhaust the
counter, then slide it off into a well, and it arrived at the bottom with nothing
left — locking immediately with no chance to reposition. A perfectly ordinary
maneuver, punished.

### Bloom was 71% of a frame

`blockwave bench` was written as routine cleanup and immediately found that the
default profile rendered at 67 fps — above 60, so nothing *looked* broken, but
with no headroom and dropping frames on any high-refresh display.

Bloom was 10 ms of a 14 ms frame, blurring at full resolution. It is a
low-frequency effect by nature, so that work was wasted; it now runs on a strided
view at quarter resolution. 67 → 145 fps, mean channel difference 0.67, no
visible change.

---

### The vignette was five times slower than the scanlines

Both are one multiply of the frame by a cached mask, yet `arcade_max` sat at
119.7 fps against a 120 floor and the profiler put 0.88 ms of each frame in
`vignette` alone — five times the identical operation in `scanlines`. The
difference was the mask's shape. The scanline mask is `(H, 1, 1)`; the vignette
mask was `(H, W, 1)`, and broadcasting a stride-0 axis of length 3 through
numpy's inner loop is a slow path. Storing the same values as a contiguous
`(H, W, 3)` array — 4 MB more, built once per compositor — brought the multiply
to 0.16 ms. The output is byte-identical; the frame is 0.7 ms cheaper; the
heaviest profile went from a coin-flip at the floor to 1.23× above it.

Two things did *not* work, and are recorded so nobody tries them again: fusing
the scanline and vignette masks into one multiply saves only 0.16 ms, because
scanlines were never the expensive one; and rewriting bloom's upsample as a
reshaped broadcast-add, though numerically exact, was *slower* than two
`np.repeat` calls. Bloom's remaining 2 ms is the upsample itself.

## Things deliberately not done

**No 180° rotation.** Not part of the guideline, and it would have needed a kick
table with no published reference to check against.

**No garbage / versus mode.** Single player only.

**The difficulty curve was measured, not tuned by feel.** The numbers show no
cliff — the steepest single-level drop in time-per-piece is to 73%, and the
transition into 20G at level 18 costs only 21%. Whether it *feels* right is a
separate question that no test can settle.

**Screen shake is full strength on `arcade`.** When the profile field was found
to be dead, the honest options were to disable shake on the default profile or to
declare the strength it already had. The latter keeps the game feeling as it
always did and makes the config true.
