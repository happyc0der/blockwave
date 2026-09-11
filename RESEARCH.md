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
