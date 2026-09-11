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

| seed 0 | lines/piece | deaths/piece | pieces/game | games |
|---|---|---|---|---|
| ckpt 1000 (12.3M steps) | 0.0360 ± 0.0007 | 0.0251 | 39.9 | 1736 |
| ckpt 2000 (24.6M) | 0.0621 ± 0.0007 | 0.0221 | 45.3 | 1581 |
| ckpt 3000 (36.9M) | 0.0844 ± 0.0010 | 0.0199 | 50.2 | 1347 |
| ckpt 4000 (49.2M) | **0.0923 ± 0.0012** | **0.0194** | **51.7** | 1352 |

That's 4.5× the 10M runs' line rate (0.0206) and about 23% of the heuristic's
0.393. The training curve rose almost linearly while the learning rate was
high and flattened only as it approached zero, as the 10M runs did. The
ceiling at 10M was mostly the schedule, not the objective.

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

Seeds 1 and 2: in progress.
