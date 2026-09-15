# Status

Working state for the RL track. Updated at the end of each phase, so a session
that starts cold does not have to reconstruct it from git log and run
directories.

_Last updated: 2026-09-15._

## Where the project is

The pixel agent is **finished and shipped**. It sees only an 88x88 crop of the
playfield and preview, presses one key per decision at 5 Hz, and learns from a
reward it fitted to its own experience. The game's score, lines and level reach
`evaluate.py` and nothing else, enforced by `tests/rl/test_firewall.py`.

Both seeds of the flagship configuration clear the drift baseline by a wide
margin (36x and 55x). That is the deliverable and it holds.

## The finding that reframed the results

A second seed of an identical 150M config scored **0.0647** against seed 0's
**0.0991** — a spread of 42% of the mean, and a seed-to-seed SD about **19x**
the within-run standard error that every earlier comparison had been quoted
with. On matched seeds the original and re-calibrated rewards score 0.0990 and
0.0991; a 5% advantage had been reported for the latter.

Consequences, all now reflected in README.md and RESEARCH.md:

* Every single-seed comparison between reward variants and compute budgets is
  **uninterpretable**. The effects were 0.09-0.22x the seed SD.
* The "recalibrated reward reaches at 50M what the original needed 150M for"
  claim has been **retracted**.
* Resolving the re-calibration effect at 2 sigma would need hundreds of seeds
  per arm -- it re-measured to +0.0016, 0.07x the seed SD. That question is
  closed as unanswerable at this noise level, not pending.

Chasing it turned up two more defects in the evaluator. Neither moved a
conclusion — measured against paired draws, the within-run SE turned out to be an
accurate estimate of evaluation noise (0.0012 implied vs 0.0013 reported). They
were failures of reproducibility and validity rather than of accuracy:

| defect | kind | status |
|---|---|---|
| `torch` action sampling unseeded, so no evaluation reproduced | reproducibility | fixed; both evaluators seed from `--seed` |
| `PixelVecEnv` seeded per worker, so `--workers` chose which games were played | validity | fixed; seeds follow global env index |
| training seed never measured at all | **the real one** | irreducible at ~0.022; why comparisons need n >= 2 |

## Rules that came out of it

1. **Nothing enters README.md except through `python -m blockwave_rl.compare`.**
   It refuses to print a sigma at n=1 and refuses cross-group comparisons unless
   both sides have >= 2 seeds.
2. `evaluate.py`'s standard error is *within-run*: it measures one trained
   network, not the training process. It is not an error bar on a comparison.
3. State the question, the seed count, and what would falsify it **before**
   launching a run.
4. Before quoting a number, check it reproduces: run the same command twice and
   confirm the output is identical.

## Tooling

| command | what it is for |
|---|---|
| `scripts/launch <run-dir> -m module ...` | start a long run; refuses to start on battery, holds caffeinate against idle sleep, writes a pidfile |
| `scripts/after <run-dir> <final-ckpt> -- <cmd>` | run something after a run, only if the final checkpoint exists |
| `scripts/install-hooks` | pre-commit hook: `ruff check src tests`, then `pytest tests/rl` |
| `uv run ruff check src tests` | the correctness gate (`[tool.ruff]` in pyproject); every ignored rule carries its reason |
| `python -m blockwave_rl.compare runs/...` | grouped results with honest uncertainty; `--markdown` emits the README table |

macOS sleeps on **battery** even when disabled on AC (`pmset -g custom` shows
`sleep 0` for AC, `sleep 1` for battery). A 150M run lost 100 minutes to
'Clamshell Sleep' this way. `scripts/launch` refuses to start on battery.
Nothing can prevent lid-close sleep without an external display.

## Open, if anyone picks it up again

* **Seed instability is the only follow-up with real expected value.** Seeds
  diverge by update ~3,600 and never reconverge, and every run gets most of its
  measurable gain after the LR anneal begins at 80%. A lower base LR or an
  earlier anneal might cut the variance, and a *variance* drop (0.024 -> under
  0.010) is visible in 3 seeds where a mean shift is not. ~15 h.
* Absolute headroom is real but belongs to a different reward: 0.08 lines/piece
  against a 0.40 theoretical maximum. Empowerment rewards keeping options open;
  clearing lines is a side effect, not the objective.
* Full checkpoint curves in `runs/*/eval_pre_seedfix.jsonl` were measured before
  worker-count-independent seeding landed, so they are internally consistent but
  do not reproduce exactly against current code. The finals in `eval.jsonl` do.
