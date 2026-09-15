# blockwave_rl — research log

The goal: an agent that learns to play from what it sees, sends real key
presses, and derives its own reward. It is never given the game's reward.
[README](README.md) covers the game and [DESIGN](DESIGN.md) its engineering;
this file records what the agent side has tried, what worked, and what did not,
failures included.

## Rules every result here obeys

- **Score, lines and level never reach the agent**: not its observations, its
  reward, its gradients, its checkpoint selection, or the choice between
  learner settings. `evaluate.py` is the only module that reads them, and a
  static test fails if any other module mentions them.
- **A top-out is a soft reset.** The board clears, `terminated` stays False,
  and the learner bootstraps through it, so dying buys no fresh start.
- **Checkpoints are saved on a fixed schedule** and all of them are evaluated
  after the fact. Learner variants are compared on the intrinsic objective
  (the thing they optimize), and game score is reported beside it, never used
  to choose.
- **The board-state track reads true occupancy.** It is a feasibility check of
  the reward, not the pixel-only deliverable, and is labelled as such.
- **A single seed cannot support a comparison**, and the standard error
  `evaluate.py` prints is not an error bar on one. It is the spread across games
  played by *one trained network*; the spread across training runs is far
  larger. Comparisons are quoted only at n >= 2 per arm, through
  `python -m blockwave_rl.compare`, which refuses to print a sigma at n=1.
  **This rule was added late, after §7 showed that several results below had
  broken it.** Sections 3-6 are left as they were written, with the claims §7
  withdraws marked in place, because how a wrong conclusion survived matters as
  much as the correction.

## 1. SMiRL: failed on a full-width board

[SMiRL](https://arxiv.org/abs/1912.05510) rewards the agent for seeing *likely*
states under a density fit to its own history. On the paper's 4-wide board with
trominoes that meant clearing lines. On a 10-wide board it does not.

- **The paper's per-cell Bernoulli density rewards concentration.** A policy
  that never moves a piece stacks a narrow tower and leaves six columns
  permanently empty. A factorized density predicts those cells perfectly, so
  they cost nothing: the edge columns cost the tower −0.006 against the
  heuristic's −1.362, outweighing the tower's worse centre (−1.808 vs −0.830).
  The reward-maximizing policy is to do nothing. A 4-wide board has no spare
  columns to exploit. Recorded as strict xfails in `tests/rl/test_adversarial.py`.
- **A learned density makes the ranking arbitrary.** A Gaussian over a board
  VAE's latents (probes pass: max height R² 0.877, holes 0.671, bumpiness 0.808)
  ranks competent play against the concentration exploits with a sign that
  flips with the VAE's corpus size and training length: margins from +0.43 to
  −0.61 across ordinary configurations. Pinned by a deterministic test. (A first
  check varied only the VAE seed and wrongly concluded the objective was robust;
  the sweep retracted that.)
- **Frame-level control breaks its timing.** A per-tick reward on the locked
  board measures how often the board changes, so slow play earns more ticks of
  an unchanging board.

## 2. Empowerment: the reward in use

Reward the agent for **control over its future**: how many distinct outcomes
its key presses can still reach. Within one piece's life the game is
deterministic, so channel capacity reduces exactly to log(number of reachable
outcomes), with no variational estimator. The real simulator serves as a
perfect world model, so this tests the principle; a pixel agent would need a
learned one.

- **One step ahead it is blind.** An empty board, a low stack and a 16-high
  tower offer the same placement count; options vanish only at death.
- **Two or three steps ahead it sees danger coming**, conditioned on the
  visible queue. Three steps: empty 9.886, 16-high tower 9.812, 18-high 9.306,
  a tower at the spawn rows 9.041.
- **Queue correction.** An I piece has 17 placements and a T 34 whatever the
  board, which swamps the signal. The reward subtracts the empty board's
  empowerment for the same queue. That term depends only on the queue, which
  the agent cannot influence, so it shifts every policy equally and changes no
  ranking. It cut competent play's per-placement std from 0.632 to 0.002.
- **Game over is zero control**: no futures remain, so raw empowerment is
  log(1 + 0) = 0. This is what the principle says a dead board is worth, not
  an injected penalty.
- **It ranks competent play above all six exploits** (random, all-NOOP,
  hard-drop spam, oscillation, rotate spam, HOLD spam): horizon 2, pooled over
  five seeds, z 12.5–15.6 after queue correction.

The result is a loss-of-control signal: zero while the board has headroom,
negative as options shrink, and strongly negative at death.

## 3. Stage 1: PPO on the board-state track

**Setup.** Frame-level key presses (the 8-key `Action` set, one per step, 20
decisions per second, gravity ×4 at a fixed level 1). The policy sees locked
cells and the active piece over all 24 rows, plus the two-piece queue its reward
is conditioned on; a small conv net feeds actor and critic heads. PPO, 96 envs ×
128 steps, γ = 0.99 and λ = 0.95 per *placement* (so stalling is neutral), lr
2.5e-4 decaying linearly to zero, entropy 0.01, 10M steps: about 14 minutes on
an M4 Pro. `python -m blockwave_rl.train`, then `python -m blockwave_rl.evaluate`.

### What training found

**1. The cheapest improvement is to stop hard-dropping.** Within 600k steps
HARD_DROP fell from 12.5% of actions to 0.3%. Each piece then drifts for ~80
steps under gravity while random moves spread it across the board, instead of
landing near the spawn columns, and deaths per piece fall by more than half. A
random policy that never hard-drops does the same, so it is now a named
baseline, `drift`, and it is the real bar. Beating uniform random proves
nothing.

**2. Only one-step credit was learned.** Hard drop is the one action whose
effect (a lock) lands on the very next step. Every positioning move shares its
piece's single outcome with ~80 mostly random moves. Entropy stayed near 1.9
of a possible 1.95, and PPO's clip fraction stayed well under the ~0.1 of a
run that is visibly learning.

**3. Better credit exposed a reward flaw.** Decaying the trace per tick inside
a piece (`--lam-step 0.95`) let PPO optimize its reward better, and it did so
by hurrying doomed games to their end: 1.8 placements in danger before each
death, against drift's 5.9. With game over charged as zero control for *one*
placement, dying fast is cheaper than lingering, because the soft reset hands
back full control. That is the fresh-start bonus the soft reset was supposed to
deny, returning through the value function.

**4. Absorbing death fixes it by construction.** Charge zero control for every
placement the dead game would have had: −E(empty, q) − γ·b̄/(1−γ), where b̄ is
the mean baseline over the 7-bag's exact queue windows. This is the smallest
charge under which one more placement alive is never worse than dying now, on
every board and queue. The episode still does not terminate, and `test_empowerment.py`
pins both the flaw and the fix.

**5. The evaluation was biased twice before it was right.** Measuring from a
fresh start over a short window over-weighted the easy early game: drift read
−0.225 intrinsic per placement against about −0.33 in continuous play. Burn-in
fixed most of that, but drift's games are regular, so the envs stayed in phase
and a fixed window cut them unevenly. A top-out restarts from an empty board,
so games are independent and identically distributed, and evaluation now counts
complete games only.

### Results, one seed per configuration

Held-out seeds, complete games (~500 or more per row), every policy scored
under absorbing death. Checkpoints 400 and 800 of 813 are on the pre-declared
schedule. Standard errors are across envs.

| policy | lines/piece | deaths/piece | pieces/game | objective/placement |
|---|---|---|---|---|
| random | 0.0000 | 0.0822 | 12.2 | −46.21 ± 0.05 |
| drift | 0.0064 ± 0.0005 | 0.0327 | 30.6 | −19.38 ± 0.06 |
| λ per placement, one-step death, 400 | 0.0136 ± 0.0009 | 0.0307 | 32.6 | −18.24 ± 0.07 |
| λ per placement, one-step death, 800 | **0.0151 ± 0.0008** | **0.0307** | **32.6** | **−18.23 ± 0.07** |
| λ per tick, one-step death, 400 | 0.0044 ± 0.0005 | 0.0460 | 21.7 | −26.77 ± 0.14 |
| λ per tick, one-step death, 800 | 0.0060 ± 0.0007 | 0.0413 | 24.2 | −24.10 ± 0.08 |
| λ per tick, absorbing death, 400 | 0.0092 ± 0.0007 | 0.0317 | 31.6 | −18.80 ± 0.06 |
| λ per tick, absorbing death, 800 | 0.0060 ± 0.0007 | 0.0330 | 30.3 | −19.61 ± 0.08 |

For scale, the scripted heuristic clears 0.397 lines per piece and never tops
out.

The plain learner is the best of the three, and significantly better than drift
on every measure: 2.4× the lines, two more pieces per game, and a higher
objective. Its reward had the one-step flaw, but that learner never found the
exploit. Per-tick credit did not help once the exploit was removed. These are
single seeds, and a one-seed ranking of learners is weak evidence.

**6. The credit chain was too long all along.** The plan's Stage 4 asked for a
configuration where a piece lives 15–40 decisions and the heuristic can still
position pieces. It was checked against random play, which hard-drops every ~8
steps and so never lets a piece live long. A policy that does not hard-drop
(the first thing PPO learns) lives 88 decisions per piece at 20 Hz:

| decisions/second (gravity ×4) | drift: decisions per piece | heuristic: lines/piece |
|---|---|---|
| 20 | 88 | 0.396 |
| 10 | 47 | 0.393 |
| 5 | 24 | 0.393, never tops out |

5 Hz is the first rate inside the window with the heuristic unaffected. That
means a 3.6× shorter credit chain and 3.6× more placements per unit of compute,
and neither knob tells the agent anything.

### The declared Stage 1 check

Chosen by the plan's Stage 4 criterion, before any 5 Hz result existed: 5 Hz,
gravity ×4, absorbing death, λ per placement, all else as above. Seeds 0, 1 and
2; checkpoints 400 and 800 on the schedule; baselines re-measured at 5 Hz. The
20 Hz config with absorbing death also gets seed 0, paired with the first run.

| policy | lines/piece | deaths/piece | pieces/game | objective/placement | games |
|---|---|---|---|---|---|
| random, 5 Hz | 0.0000 | 0.0822 | 12.2 | −46.19 ± 0.05 | 20152 |
| drift, 5 Hz | 0.0016 ± 0.0001 | 0.0506 | 19.8 | −29.33 ± 0.07 | 3754 |
| seed 0, ckpt 400 | 0.0177 ± 0.0006 | 0.0291 | 34.3 | −17.38 ± 0.04 | 1687 |
| seed 0, ckpt 800 | 0.0206 ± 0.0006 | 0.0281 | 35.6 | −16.83 ± 0.04 | 1794 |
| seed 1, ckpt 400 | 0.0201 ± 0.0005 | 0.0287 | 34.9 | −17.11 ± 0.04 | 1886 |
| seed 1, ckpt 800 | 0.0207 ± 0.0005 | 0.0282 | 35.5 | −16.87 ± 0.04 | 1778 |
| seed 2, ckpt 400 | 0.0191 ± 0.0006 | 0.0290 | 34.5 | −17.28 ± 0.04 | 1699 |
| seed 2, ckpt 800 | 0.0205 ± 0.0005 | 0.0281 | 35.5 | −16.83 ± 0.03 | 1783 |

And the paired 20 Hz run, against the 20 Hz baselines in the earlier table:

| policy | lines/piece | deaths/piece | pieces/game | objective/placement | games |
|---|---|---|---|---|---|
| 20 Hz, absorbing, seed 0, ckpt 400 | 0.0117 ± 0.0008 | 0.0345 | 29.0 | −20.46 ± 0.12 | 636 |
| 20 Hz, absorbing, seed 0, ckpt 800 | 0.0262 ± 0.0010 | 0.0274 | 36.5 | −16.36 ± 0.07 | 568 |

### Verdict: Stage 1 passes, weakly

- **It passes as written, and against the stronger bar.** On every seed and at
  both scheduled checkpoints, the declared configuration beats random *and*
  drift on every measure, by dozens of standard errors: 13× drift's line rate
  and 1.8× its game length. The seeds agree to within 0.0002 lines per piece.
  Nothing here was chosen by score.
- **Absorbing death helped the plain learner.** Paired on seed 0 at 20 Hz, it
  reached 0.0262 lines per piece and 36.5 pieces per game, against 0.0151 and
  32.6 for the one-step accounting. One pair is one pair, but the direction
  matches the training curves at every window.
- **The shorter credit chain did not change the outcome.** At equal env steps,
  5 Hz and 20 Hz agents end in the same place, about 36 pieces per game. At 5 Hz
  that took 2.7× more placements. Chain length was not the binding constraint.
- **It is far from competent.** 0.02 lines per piece is 5% of the heuristic's
  0.39, and the agent still tops out every ~36 pieces where the heuristic never
  does. The objective has plenty of headroom: the heuristic scores ~0 per live
  placement and never pays the death charge. Every run was still improving when
  its learning rate reached zero at 10M steps.

The reward learns something real, repeatably, without ever seeing the score.
Whether it can learn *Tetris* is still open. That needs either a much longer run
or a diagnostic that separates the learner from the objective.

## 4. Longer training: a slow learner, not a ceiling

Same declared configuration, 50M steps instead of 10M. The only change is run
length: the same linear learning-rate decay, stretched, so it decays 5× more
slowly. Held-out, complete games, checkpoints on the fixed schedule
(`runs/abs5hz_long_seed*`):

| checkpoint | env steps | lines/piece, seeds 0 / 1 / 2 | mean | pieces/game, seeds 0 / 1 / 2 | mean |
|---|---|---|---|---|---|
| 1000 | 12.3M | 0.0360 / 0.0306 / 0.0322 | 0.0329 | 39.9 / 38.3 / 39.1 | 39.1 |
| 2000 | 24.6M | 0.0621 / 0.0529 / 0.0516 | 0.0555 | 45.3 / 43.6 / 42.9 | 43.9 |
| 3000 | 36.9M | 0.0844 / 0.0707 / 0.0654 | 0.0735 | 50.2 / 47.2 / 46.0 | 47.8 |
| 4000 | 49.2M | 0.0923 / 0.0812 / 0.0769 | **0.0834** | 51.7 / 49.5 / 48.5 | **49.9** |

Standard errors within each row are ≤ 0.0012 lines per piece over 1,350–2,000
games; the spread between seeds (sd 0.008 at checkpoint 4000) is the larger
uncertainty. Deaths per piece at checkpoint 4000: 0.0194 / 0.0202 / 0.0206.

Every seed improves at every scheduled checkpoint. At 50M the mean is 4.0× the
10M runs' line rate (0.0206) and 21% of the heuristic's 0.393. The curves rose
almost linearly while the learning rate was high and flattened only as it
approached zero, as the 10M runs did. The ceiling at 10M was mostly the
schedule, not the objective. Where it does saturate is still unmeasured.

To watch it: `python -m blockwave_rl.watch runs/abs5hz_long_seed0/ckpt_04000.pt`
renders the first complete game on the held-out seed through the game's own
compositor. The game is chosen by that rule, never by how well it went. That
game cleared 2 lines against an average of ~4.7. The agent spreads pieces
across the full width and sometimes finishes a line, but it leaves holes and
builds an uneven stack, which fits 23% of the heuristic.

### What it learned, and why that is slow

A behaviour profile over 1,500 placements per policy. It is evaluator-side:
it reads the engine's hole count, which the agent never sees.

| policy (5 Hz) | decisions/piece | locked by hard drop | holes per placement |
|---|---|---|---|
| drift | 23.7 | 0% | +2.70 |
| 10M run, final | 27.1 | 0% | +1.57 |
| 50M run, ckpt 1000 | 24.8 | 0% | +1.17 |
| 50M run, ckpt 2000 | 24.0 | 0% | +0.86 |
| 50M run, ckpt 4000 | 24.6 | 0% | +0.74 |
| scripted heuristic | 5.1 | 99% | +0.03 |

- **It never hard-drops, at any checkpoint.** Every piece falls the whole way
  while the agent steers it. Under per-placement discounting, time is free, so
  hard drop has no reward value. Using the whole fall to adjust is rational.
- **What improved is placement quality.** Holes per placement fell steadily,
  from drift's 2.70 to 0.74. That is the whole of the gain, and still ~20× the
  heuristic.
- **Why that is slow.** Empowerment is blind to holes on a low board: a hole
  removes none of the next pieces' reachable placements, so it costs nothing
  when it is made. Its price arrives many placements later, when the covered
  rows keep the stack high near the top. It reaches the placement that caused
  it only through the value function's long chain back from danger and death.
  A practical horizon (2–3 pieces) does not change this. A hole makes the stack
  taller, so a long enough lookahead would see it reach the top sooner, but on a
  low board that means tens of pieces ahead, far beyond what exact counting can
  afford.


## 5. Where is the ceiling? Not at 150M steps

The 10M and 50M runs both flattened exactly as their learning rate reached
zero, so neither measured saturation. This run holds the rate for the first 80%
of 150M steps and decays it over the last 20% (`--lr-decay-start 0.8`), same
configuration otherwise, seed 0, ~4.5 hours. Held-out, complete games:

| checkpoint | env steps | lines/piece | gain | deaths/piece | pieces/game | games |
|---|---|---|---|---|---|---|
| 2000 | 24.6M | 0.0642 ± 0.0009 | | 0.0216 | 46.2 | 1590 |
| 4000 | 49.2M | 0.0967 ± 0.0009 | +0.0325 | 0.0191 | 52.3 | 1398 |
| 6000 | 73.7M | 0.1154 ± 0.0011 | +0.0188 | 0.0176 | 56.7 | 1311 |
| 8000 | 98.3M | 0.1342 ± 0.0014 | +0.0188 | 0.0164 | 60.9 | 1159 |
| 10000 | 122.9M | 0.1416 ± 0.0013 | +0.0073 | 0.0159 | 62.8 | 1111 |
| 12000 | 147.5M | **0.1629 ± 0.0013** | +0.0213 | **0.0144** | **69.6** | 1065 |

**No ceiling.** Gains slowed but never stopped, and the largest single gain of
the second half came last, during the decay. 0.163 lines per piece is 41% of
the heuristic's 0.393, and double the 50M result.

**The schedule really was the limit at 50M.** At the same 49.2M steps this run
scores 0.0967 with its rate still at full, against 0.0923 for the 50M run whose
rate had just annealed to zero — so annealing bought that run nothing, and the
flattening was the schedule ending, not learning saturating.

A final game: `runs/abs5hz_150m_seed0/ckpt_12207_game.mp4`, 1,674 decisions and
13 lines, against 2 lines for the 50M agent's game. Still the first complete
game on the held-out seed, never the best one.

What this does not answer is where it does saturate, or whether the seed spread
(±10% at 50M) holds at this length: this is one seed.

## 6. The pixel stage

The board-state track reads true occupancy. The deliverable reads the screen,
which changes one thing fundamentally: **empowerment needs a model of the game,
and a pixel agent may not borrow the simulator's.** Counting reachable futures
exactly, as sections 2-5 did, is knowledge of the rules. So the agent learns
what its keys do, and counts futures through that.

### The crop is forced

BOARD_PLUS_PREVIEW, not BOARD_ONLY. One piece of lookahead is blind — on any
board with headroom every placement is still reachable, which is finding 1 of
section 2 — and seeing two pieces ahead means the next-piece panel has to be in
frame. The strictest crop cannot carry this reward. Both variants were built in
Stage 0 for exactly this choice.

### Representation (plan Stage 2), thresholds set in advance

A conv VAE over 88x88 grayscale frames, corpus mixed over random, drift,
competent and near-empty play. Linear probes on the frozen 32-d latents,
held out:

| probe | R² | threshold |
|---|---|---|
| max height | 0.926 | 0.80 |
| holes | 0.833 | 0.60 |
| bumpiness | 0.703 | 0.50 |

Reconstruction error is 0.8-1.7% per pixel and even across the four sources, so
no policy's states are badly represented — the gap that made the SMiRL ranking
arbitrary rather than merely wrong.

### Two events, read off the screen

The reward needs to know when a piece lands and when a game ends; the
board-state track took both from `info`. Both are visible, and both detectors
are validated against the engine over five policies. Four false starts, each a
real property of the game rather than a tuning failure:

* A line clear blanks its rows before collapsing them — a large *relative* drop
  in filled area, which read as a game-over. Only a top-out empties the board
  outright, so the test is absolute.
* The playfield is never blank: it has a border, and a piece with its ghost is
  on screen a frame after any reset. Hence a calibrated baseline.
* Previews are miniatures, ~12 pixels a piece, and the hold preview dims to 45%
  while spent. The brightness cut that works for the playfield is blind to a
  held piece.
* Only the *first* hold of a game draws from the queue; later holds swap with
  the piece already held and shift nothing. Counting that first one as a
  placement would pay a policy a reward tick for pressing one key.

### Learning what the keys do

The agent babbles: for each piece it runs one of 44 blind key-press programs
(rotate r, step k, drop) and records the screen before and after. 60,000 tries,
4.3% of them fatal. A small network predicts each program's effect in latent
space and whether it ends the game:

| held out | value | baseline |
|---|---|---|
| skill vs "nothing changes" | 0.546 | 0.0 |
| identifies which program ran, top-1 | 0.192 | 0.023 (chance) |
| mean rank of the true program | 4.6 of 44 | 21.5 (chance) |
| fatal programs caught | 0.504 | — |

That last pair matters more than the average error. A model whose error swamps
the difference between programs cannot support a count of *distinct* futures
however good its mean error looks, so "can it tell its own keys apart" is
measured directly.

### The reward

Effective number of distinguishable futures under a Gaussian kernel, weighted
by each future's chance of not ending the game, log-scaled, minus the same
quantity on an empty board. The kernel width is the model's own held-out error:
two futures are the same when the model cannot resolve them apart. That is not
a free knob — it tightens as the model improves. The second piece is expanded
over a fixed sample of programs, fixed rather than re-drawn per call, because a
reward that scores one state differently twice is noise in the training signal.

### Does it agree with the reward it replaces?

The board-state track can compute this same quantity *exactly*, so the pixel
estimate can be checked against ground truth before any RL is spent on it. Two
comparisons, and they say different things.

**Per state, weakly.** Over 400 states from mixed play: Spearman 0.31, Pearson
0.44. Some of that is unavoidable — the exact reward saturates, scoring zero on
every board with headroom, so a rank correlation across states is largely
measuring ties.

**Between policies, decisively.** The plan's Stage 3 gate, run with the reward
computed exactly as it would be in training — from frames, through the agent's
own model, no engine state in the path:

| policy | reward per placement |
|---|---|
| competent | **−0.4367 ± 0.0032** |
| all_noop | −0.6287 |
| hard_drop_spam | −0.6286 |
| drift | −0.6655 |
| oscillate | −0.6708 |
| hold_spam | −0.6673 |
| rotate_spam | −0.6626 |
| random | −0.6830 |

Competent play beats every exploit at z 23-30. The exact reward separates the
same policies by about 1% (z 12-15 after the queue correction), because it
saturates where this one does not.

**Caveat, stated rather than buried.** Part of that wider separation may be the
model being less certain on cluttered boards rather than the agent genuinely
having less control there. The two are hard to tell apart from the outside, and
a reward that pays for legibility rather than for control is a different
objective wearing the same name. What can be said is that it orders policies
correctly and does so from pixels alone.

Next: PPO on this reward, the same learner and the same protocol as the
board-state track, with game score reported and never used.

### Training on it

PPO on the pixel reward, same learner and protocol as the board-state track:
5 Hz, 10M steps, 48 envs, seed 0, the learning rate held for 80% of the run.
58 minutes at ~2,500 steps/sec. Held out, complete games, the fixed checkpoint
schedule:

| checkpoint | env steps | lines/piece | deaths/piece | pieces/game | games |
|---|---|---|---|---|---|
| 400 | 2.5M | 0.0124 ± 0.0006 | 0.0324 | 30.9 | 1035 |
| 800 | 4.9M | 0.0087 ± 0.0005 | 0.0354 | 28.3 | 1044 |
| 1200 | 7.4M | 0.0164 ± 0.0008 | 0.0304 | 32.9 | 895 |
| 1600 | 9.8M | **0.0239 ± 0.0008** | **0.0280** | **35.7** | 933 |
| drift, measured through pixels | | 0.0018 ± 0.0003 | 0.0508 | 19.7 | 1880 |

**It matches the board-state agent.** At the same 10M steps that agent — reading
true occupancy, with the reward counted exactly by the simulator — reached
0.0206 lines per piece and 35.5 pieces per game across three seeds. This one
reaches 0.0239 and 35.7 from frames alone, through a model of its own keys.
Thirteen times the drift baseline's line rate, and 1.8x its game length.

Two honest qualifications. The schedules differ — the board-state 10M runs
decayed their learning rate from the first update, this one held it to 80% —
so the fair reading is "the same ballpark at the same budget", not "better".
And this is one seed; the board-state seeds spread by about ±10% at this length.

The dip at checkpoint 800 is real rather than measurement noise: the training
curve flattened over the same stretch, and the held-out numbers followed it
down and back up. Progress here is not monotonic.

A game: `runs/pixel_ppo_seed0/ckpt_01627_game.mp4` — 1,062 decisions, 2 lines,
the first complete game on the held-out seed as always. The agent sees the small
88x88 grey crop; the video is the same game drawn at human size.

### Does the model go stale as the agent improves?

The world model is fitted once, while babbling — near-random play. A trained
agent visits different boards, and a model that is worse there would degrade the
reward exactly where the agent is heading, without anything failing. Measured on
3,000 transitions taken from the 10M agent's own play (`world.shift`):

| | babbling, held out | the agent's states |
|---|---|---|
| skill vs "nothing changes" | 0.546 | 0.550 |
| identifies which program ran | 0.192 | 0.188 |
| mean rank of the true program | 4.61 of 44 | 4.56 of 44 |
| fatal programs caught | 0.504 | 0.564 |

No drift worth acting on at this level of play. Worth re-checking at higher
skill, where boards stay flatter for longer than anything in the corpus.

### 50M steps from pixels

Same configuration, five times the length: 8,138 updates, 5.2 hours, seed 0.
Held out, complete games, the fixed schedule:

| checkpoint | env steps | lines/piece | deaths/piece | pieces/game | games |
|---|---|---|---|---|---|
| 2000 | 12.3M | 0.0237 ± 0.0008 | 0.0272 | 36.7 | 915 |
| 4000 | 24.6M | 0.0500 ± 0.0012 | 0.0236 | 42.4 | 691 |
| 6000 | 36.9M | 0.0559 ± 0.0012 | 0.0232 | 43.2 | 794 |
| 8000 | 49.2M | **0.0901 ± 0.0013** | **0.0198** | **50.6** | 649 |

Against the 10M pixel run (0.0239, 35.7): 3.8x the line rate and fifteen more
pieces per game. Against the drift baseline: fifty times the line rate and two
and a half times the game length. It is 23% of the scripted heuristic.

**Compared with the privileged track, carefully.** The board-state agent's three
50M seeds averaged 0.0834 and 49.9 at the same step count, which this nominally
beats — but those runs decayed their learning rate from the first update while
this one held it to 80%, and schedule was worth a lot on that track. The fair
comparison is the board-state 150M run, which used this schedule shape: at
49.2M it read 0.0967 and 52.3. So the pixel agent is about 7% behind the
privileged agent on lines and 3% behind on survival, at equal steps.

That is the result worth stating plainly: **learning from the screen, with a
reward derived from a model of its own key presses, costs a few percent against
reading the true board with a reward computed exactly by the simulator.**

**The model did not go stale.** Re-measured on 3,000 transitions from this
agent's own play, at a level far above anything in the babbling corpus: skill
0.552, true-program rank 4.31 of 44, fatal programs caught 0.521 — the same
numbers as on the corpus it was fitted to. The reward held its accuracy where
the agent went.

Still one seed, and the board-state seeds spread about 10% at this length.

### 150M steps from pixels: the reward, not the compute, is the ceiling

> **Withdrawn by §7.** Both the section title and the "three times the compute
> bought 6%" conclusion rest on comparing two single runs. The seed spread
> measured in §7 is roughly five times that 6%, so this comparison cannot
> distinguish a compute effect from two draws of the same configuration.

Same configuration again, three times the length: 24,414 updates, 15 hours.

| checkpoint | env steps | lines/piece | deaths/piece | pieces/game | games |
|---|---|---|---|---|---|
| 6000 | 36.9M | 0.0559 ± 0.0011 | 0.0233 | 43.0 | 801 |
| 12000 | 73.7M | 0.0531 ± 0.0011 | 0.0242 | 41.2 | 763 |
| 18000 | 110.6M | 0.0685 ± 0.0011 | 0.0228 | 43.9 | 671 |
| 24000 | 147.5M | **0.0959 ± 0.0020** | **0.0194** | **51.6** | 601 |

**Three times the compute bought 6%.** The 50M run reached 0.0901 and 50.6 in a
fifth of the time. Compare the privileged track over the same stretch: it went
from 0.0834 at 50M to 0.163 at 150M, nearly doubling. The gap between the two
tracks was 7% at 50M and is 41% at 150M.

**The flat stretch was the learning rate; the ceiling is not.** From 50M to
120M the curve sat at 0.063-0.067 while entropy fell and the clip fraction rose
— the policy kept moving without improving. Annealing recovered it, 0.0685 at
110M to 0.0959 at 147M, a 40% gain from the schedule alone. So the plateau was
not a hard limit, but what the anneal recovered only matched what a 50M run
reaches five times sooner.

**What that points at.** The reward counts futures its model can tell apart, and
that model has 0.55 prediction skill and identifies which of 44 programs ran 19%
of the time. Early on, telling a disastrous placement from a reasonable one is
well within that resolution. Refining good play into better play is not: the
distinctions get finer than the model can make, so the reward stops
discriminating before the policy stops being improvable. The board-state track,
counting exactly, has no such limit.

The fix this implies is a sharper model or representation — more babbling data,
features trained to separate the outcomes of different programs (an inverse
model, as curiosity work uses), a larger latent — and not more PPO steps. On the
evidence here, beyond ~50M steps at this model quality the compute is wasted.

### Sharpening it made it worse, and the trial confounded two changes

Trained on the sharpened reward — the inverse-shaped encoder *and* the
re-calibrated kernel width — against the original, same learner, same seed:

| env steps | original reward | sharpened + re-calibrated |
|---|---|---|
| ≤5M | 0.0114 | 0.0095 |
| ≤10M | 0.0177 | 0.0169 |
| ≤15M | 0.0250 | 0.0179 |
| ≤20M | 0.0301 | 0.0178 |
| ≤25M | 0.0466 | 0.0201 |
| ≤30M | 0.0546 | **0.0221** |

Stopped at 31M: less than half the line rate, with no sign of closing. Reverted
to the original reward, which is what `pretrained/` ships.

So a reward can look better on every static measure — identifies its own
programs more often, catches more fatal ones, probes higher, resolves 43 futures
where the old one resolved 2 — and still teach less. Resolution is evidently not
the whole story, and the plausible cost is variance: a finer measure also counts
fine distinctions the model gets wrong, which is signal and noise together.

**The trial changed two things at once**, encoder and width, so it cannot say
which one hurt. That is a flaw in the experiment rather than in the result: the
honest conclusion is only that this pair is worse than the original, not that
sharpening or re-calibration is individually bad. Separating them is one run
each, and worth doing before anyone builds on either.

One artifact bug found while reverting, and worth recording because it was
already public: re-calibration had been applied *in place* to `world.pt`, so the
shipped reward briefly carried a width the shipped agent had never been trained
with. Anyone evaluating it would have got intrinsic numbers on a different scale
than the published ones. Restored from the values recorded in `world.json`.

### Separating the two variables, one run each

> **Withdrawn by §7.** Isolating the two variables was the right design, but
> "one run each" is exactly the flaw §7 exposes: with one seed per arm there is
> no uncertainty to compare the arms against. The conclusion that the encoder
> hurt and the width helped is not supported by this evidence. The experiment
> would need ~169 seeds per arm to resolve an effect of the size claimed.

The failed trial changed the encoder and the kernel width together, so it could
not say which hurt. Two runs at 30M steps, one variable each, same learner, same
seed, same schedule — and the reference is the original pair.

Held out, complete games, at the final checkpoint (29.5M steps):

| variant | encoder | kernel width | lines/piece | pieces/game |
|---|---|---|---|---|
| **width only** | original 32-d | re-calibrated 0.486 | **0.0635 ± 0.0013** | **45.1** |
| reference | original 32-d | fitted 1.945 | ~0.053 (interpolated) | ~43 |
| encoder only | sharpened 64-d | fitted 2.532 | 0.0295 ± 0.0010 | 38.4 |
| both (stopped at 31M) | sharpened 64-d | re-calibrated 0.633 | ~0.022 | ~36 |

The training curves, at the last window before either run began annealing:

| ≤20M steps | lines/piece |
|---|---|
| width only | 0.0366 |
| reference | 0.0301 |
| encoder only | 0.0224 |
| both | 0.0178 |

**It was the encoder, and the width was the good half.** Re-calibration helps:
0.0635 at 29.5M against the reference's ~0.053, reaching in 30M steps what the
original reward needed nearly 37M for. Sharpening the encoder hurts about as
much: 0.0295, and between 9.8M and 19.7M it did not improve at all (0.0210 to
0.0210) while the width-only run doubled (0.0204 to 0.0405).

That inverts the conclusion the combined run suggested, which had been recorded
here as "the sharpened reward taught less" without knowing which half was
responsible. One run each was the way to find out, and worth the six hours.

Why the sharper encoder should be worse is not settled. Its latents separate the
outcomes of different programs better — that is what it was trained for — and the
probes still pass. A plausible reading is that the inverse-model objective spends
capacity on distinctions that predict *which key was pressed* rather than what
the board became, and the reward only cares about the latter. That is a
hypothesis, not a finding.

`world.fit` now chooses the width by the spread criterion rather than the mean
error, since that is the half that helps.

### The flagship, re-trained with the better-calibrated reward

> **Withdrawn by §7.** The +5% reported here, and the claim that this reaches at
> 50M what the original needed 150M for, are single-seed comparisons. §7 measures
> the seed spread at roughly six times that +5%. The re-trained agent is a
> perfectly good agent; the reason given for preferring it does not hold.

Same configuration as the 50M flagship, changing only the kernel width — the
half the ablation showed helps. Held out, complete games:

| checkpoint | env steps | lines/piece | pieces/game |
|---|---|---|---|
| 800 | 4.9M | 0.0142 ± 0.0008 | 32.1 |
| 1600 | 9.8M | 0.0200 ± 0.0006 | 35.3 |
| 3200 | 19.7M | 0.0405 ± 0.0011 | 39.8 |
| 4800 | 29.5M | 0.0670 ± 0.0014 | 45.4 |
| 6400 | 39.3M | 0.0541 ± 0.0013 | 42.6 |
| 8000 | 49.2M | **0.0948 ± 0.0016** | **51.3** |

Against the original flagship's 0.0901 ± 0.0013 and 50.6: about 5% better on
lines, roughly two standard errors, and 1% on survival. A small win at the end
of the run.

**The larger effect is on the way there.** At matched steps mid-run the gap was
22% at 20M and 18% at 35M, and it closed as both runs annealed. So the
re-calibration buys speed rather than a higher ceiling — which is still worth
having, because it reaches at 50M what the original reward needed 150M for.

Checkpoint 6400 reads 0.0541 against 0.0682 before it and 0.0747 after. Single
checkpoints wobble; the training windows over that stretch were flat rather than
falling, which is why the schedule evaluates several and the table shows them
all rather than the best.

`pretrained/` now ships this agent with the reward it was trained against, so
the two stay matched.

## 7. A second seed, and the retraction of §§5–6

Every pixel comparison above ran one seed per configuration. The plan called for
three. The shortcut seemed defensible because `evaluate.py` prints a standard
error of about ±0.0016 on 600-odd complete games, and the effects being reported
were two to three times that. A second seed of the 150M re-calibrated flagship —
same config, same reward, same schedule, `--seed 1` instead of `--seed 0` — was
run to firm up the number.

| seed | lines/piece | deaths/piece | pieces/game | games |
|---|---|---|---|---|
| 0 | 0.0991 ± 0.0017 | 0.0191 | 52.5 | 600 |
| 1 | **0.0647 ± 0.0014** | 0.0217 | 46.1 | 660 |

The two differ by 0.0344, **42% of their mean**. The seed-to-seed SD is 0.0243,
about **19× the within-run standard error** that every earlier comparison had
been quoted against.

(These are the final, fully-corrected measurements. The seed gap was first seen
at 0.0965 / 0.0650 before the two evaluator defects below were fixed; correcting
them moved both numbers slightly and the gap not at all.)

### Why the error bar was the wrong one

`evaluate.py`'s standard error is computed across *envs within one evaluation*.
It answers "how precisely did we measure this network?" — and the answer, ±0.0016,
is correct. The question every comparison actually asked was "would training this
configuration again land somewhere else?", and nothing in that number addresses
it. Two sources of variance were being conflated, and the smaller one was
standing in for the larger.

Against the measured 0.0243, the effects previously reported — and what they
became once the evaluator was fixed and every run re-measured at its true final
checkpoint:

| claim | as reported | re-measured | fraction of seed SD |
|---|---|---|---|
| re-calibrated reward beats original at 50M | +0.0047 | **+0.0016** | 0.07 |
| 150M re-calibrated beats 150M original | +0.0020 | **+0.0001** | 0.00 |
| 150M beats 50M, re-calibrated | +0.0031 | −0.0122 | (sign reversed) |

None is distinguishable from drawing the same configuration twice. The second
row is the cleanest statement of the whole episode: on matched seeds the two
rewards score **0.0990** and **0.0991**. A 5% advantage was reported for one of
them.

### Two smaller problems found on the way

Fixing this exposed a measurement bug. `PixelVecEnv` seeded each worker from its
own index, so `--workers` decided *which games were played*: changing the
parallelism silently changed the number. Seeding now follows the global env index
(`tests/rl/test_pixel_vector.py` asserts the two layouts agree byte-for-byte).

Re-measuring every final checkpoint under the corrected seeding — and at the true
final checkpoint rather than whichever one an `--every 8` schedule happened to
land on — moved the numbers by 0.003–0.004 each. That is the same size as all
three effects in the table above. The 50M re-calibration gap, reported as
"+0.0047, 2.28σ", became **+0.0013** under nothing more than corrected
bookkeeping, before seed variance was considered at all.

Old curves are kept as `runs/*/eval_pre_seedfix.jsonl`; they are internally
consistent but do not reproduce against current code.

**The evaluator was also not reproducible at all.** The policy samples its
actions rather than taking the argmax, and neither evaluator called
`torch.manual_seed`, so every evaluation played a different sequence of games.
Evaluating one byte-identical checkpoint twice gave 0.0953 and 0.0949 — a wobble
of about a quarter of the within-run SE, small enough to have gone unnoticed
indefinitely, and fatal to "check the numbers for yourself". Both evaluators now
seed from `--seed`, and re-running the same command twice reproduces exactly.

Having both an unseeded and a seeded measurement of five checkpoints gives two
independent draws of each, which pins down how noisy an evaluation actually is:

| run | draw A | draw B | implied SD | reported SE |
|---|---|---|---|---|
| 10M | 0.0212 | 0.0238 | 0.0018 | 0.0008 |
| 30M encoder-only | 0.0301 | 0.0310 | 0.0006 | 0.0010 |
| 30M width-only | 0.0633 | 0.0609 | 0.0017 | 0.0012 |
| 50M original | 0.0940 | 0.0925 | 0.0011 | 0.0016 |
| 50M re-calibrated | 0.0953 | 0.0941 | 0.0008 | 0.0019 |
| | | **mean** | **0.0012** | **0.0013** |

**The within-run standard error was accurate.** It estimates evaluation noise to
within 6%, and that was never the problem. The problem is that it answers "how
precisely was this network measured?" when every comparison in §§5–6 was asking
"would training this configuration again land somewhere else?" — a quantity 17×
larger that nothing in the pipeline measured at all.

That also right-sizes the two smaller defects. Unseeded sampling was a
*reproducibility* failure, not an accuracy one: the noise it introduced was
already inside the quoted error bar, but a published number could not be checked.
The worker-index seeding was a *validity* failure: it made the measurement depend
on a flag that should not have touched it. Neither one moved any conclusion. The
seed count did.

### What this costs, and what it does not

Withdrawn: every comparison between reward variants and between compute budgets
in §§5–6, marked in place above. The sharpening ablation's conclusion goes with
them — the design was right, the seed count was not.

Not withdrawn: the deliverable. Both seeds clear the drift baseline (0.0018) by
36× and 54×, and clear a random policy by more still. An agent seeing only an
88×88 crop, pressing one key per decision, learning what those keys do by trying
them and deriving its own reward, plays Tetris far better than any trivial
policy. That gap is two orders of magnitude larger than the seed spread and was
never in question.

Also not withdrawn: the curve *shape*, which both seeds share. A long flat
plateau, then most of the measurable gain arriving after the learning-rate anneal
begins at 80%. Seeds diverge by update ~3,600 and never reconverge.

### Why this is not being resolved with more runs

To separate the 0.0047 re-calibration effect at 2σ:

```
seeds per arm = 2 · (2σ/δ)²  with σ = 0.0243
    δ = 0.0047  ->  214 seeds per arm
    δ = 0.0100  ->   47
    δ = 0.0200  ->   12
```

At 5 hours per 50M run, 214 seeds is six weeks of continuous compute per arm.
Three seeds per arm — the plan's requirement — resolves nothing below ~0.040,
which is nearly half the mean. **These questions are not open pending
more compute; they are unanswerable at this noise level**, and saying so is the
result.

The one follow-up with real expected value is the variance itself. Seeds diverge
early and every run depends on the anneal, which points at the learning-rate
schedule. A drop from σ = 0.024 to under 0.010 would be visible in 3 seeds, where
a mean shift of the size under discussion is not.

### What changed in the repository

- `python -m blockwave_rl.compare` groups runs by config, refuses to print a
  sigma at n=1, and emits the README table. Nothing enters README.md otherwise.
- A rule at the top of this file: single seeds do not support comparisons.
- `scripts/launch` refuses to start a run on battery, after a 150M run lost 100
  minutes to clamshell sleep (macOS `sleep` is 0 on AC but 1 on battery).
- `world/fit.py` refuses to overwrite an existing `world.pt`; doing so in place
  once shipped an agent with a reward it had never trained against.
- Both evaluators seed `torch`, so a published number reproduces exactly.
- `PixelVecEnv` seeds by global env index, asserted byte-for-byte across worker
  layouts in `tests/rl/test_pixel_vector.py`.
- `tests/test_perf.py` takes the best of three render timings: contention only
  ever makes a renderer look slower, so the mean was biased by whatever else was
  running and `arcade_max` failed at 117 against its 120 floor during training.
- A pre-commit hook runs `tests/rl` (`scripts/install-hooks`), after a commit
  once landed with a failing test because the check was chained off a `tail`.
