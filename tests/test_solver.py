from itertools import combinations

import numpy as np

from msrl.env import HIDDEN, BoardSpec, MinesweeperEnv, neighbor_sum, random_valid_actions
from msrl.evaluate import RandomAgent, evaluate
from msrl.solver import SolverAgent, analyze


def brute_force(obs: np.ndarray, n_mines: int) -> np.ndarray:
    h, w = obs.shape
    hidden = np.flatnonzero(obs.ravel() == HIDDEN)
    revealed = obs != HIDDEN
    tally = np.zeros(h * w)
    total = 0
    for combo in combinations(hidden, n_mines):
        mines = np.zeros(h * w, dtype=bool)
        mines[list(combo)] = True
        counts = neighbor_sum(mines.reshape(1, h, w))[0]
        if (counts[revealed] == obs[revealed]).all():
            tally += mines
            total += 1
    prob = np.full(h * w, np.nan)
    prob[hidden] = tally[hidden] / total
    return prob.reshape(h, w)


def test_matches_brute_force():
    spec = BoardSpec(5, 5, 5)
    rng = np.random.default_rng(0)
    env = MinesweeperEnv(spec, 40, seed=1, auto_reset=False)
    checked = 0
    for _ in range(4):
        env.step(random_valid_actions(env.action_mask(), rng))
        obs = env.obs()
        for i in np.flatnonzero(~env.finished):
            prob, exact = analyze(obs[i], spec.mines)
            assert exact
            np.testing.assert_allclose(prob, brute_force(obs[i], spec.mines), atol=1e-12)
            checked += 1
    assert checked > 30


def test_probabilities_sum_to_mine_count():
    spec = BoardSpec(9, 9, 10)
    env = MinesweeperEnv(spec, 50, seed=2, auto_reset=False)
    env.step(np.full(50, 40))
    for o in env.obs()[~env.finished]:
        prob, _ = analyze(o, spec.mines)
        assert abs(np.nansum(prob) - spec.mines) < 1e-9


def test_solver_beats_random():
    spec = BoardSpec(9, 9, 10)
    solver = evaluate(SolverAgent(spec), spec, games=300)
    rand = evaluate(RandomAgent(), spec, games=300)
    assert solver["win_rate"] > 0.8 > rand["win_rate"]
    assert solver["losses"]["blunder"] == 0 and solver["losses"]["bad_guess"] == 0
