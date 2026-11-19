# RL Minesweeper

Five players on the same mine-grid puzzle, compared on the same 10,000 boards:

| Player | What it is |
| --- | --- |
| `random` | clicks any hidden cell |
| `solver` | exact mine probabilities by enumeration; the reference |
| `supervised` | CNN trained on the true mine positions (not RL) |
| `dqn` | double dueling DQN with action masking |
| `ppo` | PPO with action masking and GAE |

The three learners share one fully convolutional ResNet (`msrl/models.py`), so a
network trained on 9x9 also runs on larger boards.

## Run it

```sh
uv sync
uv run pytest                                   # env and solver tests
uv run python -m msrl.bench                     # throughput on this machine

uv run python -m msrl.evaluate --agent solver   # baselines -> results/<board>/
uv run python -m msrl.evaluate --agent random

uv run python -m msrl.supervised                # each writes runs/<name>/ and results/<board>/<name>.json
uv run python -m msrl.dqn
uv run python -m msrl.ppo

uv run python -m msrl.export_web                # ONNX models + results.json for the site
cd web && pnpm install && pnpm dev              # the showcase site
```

Every trainer takes `--board beginner|intermediate|expert`, `--env-steps`, `--name`
and `--init <checkpoint>` (fine-tune from another board size).

## Rules of the environment

- The only action is revealing a cell; flags are not needed to win.
- Mines are placed on the first click so the clicked cell and its neighbours are
  empty, which always opens a region.
- Reward: a share of 1.0 for each newly opened cell, +1 for a win, -1 for a mine.

## How results are judged

`msrl/evaluate.py` plays 10,000 fixed boards per player. Each lost game is
classified by the exact solver as a `blunder` (a provably safe cell existed),
a `bad_guess` (had to guess, chose worse odds than the best) or `unlucky`
(took the best odds and lost).

## Layout

```
msrl/        environment, solver, network, trainers, evaluation, export
tests/       pytest suite
results/     committed evaluation summaries and training curves
runs/        checkpoints and logs (ignored)
web/         Vite + React site; game logic and solver ported to TypeScript
```
