from collections import deque

import numpy as np
import pytest

from msrl.env import HIDDEN, PRESETS, BoardSpec, MinesweeperEnv, random_valid_actions


class SlowBoard:
    """Plain single-board reference implementation."""

    def __init__(self, spec: BoardSpec, scores: np.ndarray):
        self.spec, self.scores = spec, scores
        self.mines = None
        self.revealed = np.zeros((spec.h, spec.w), dtype=bool)
        self.done = self.won = False

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < self.spec.h and 0 <= cc < self.spec.w:
                    yield rr, cc

    def count(self, r, c):
        return sum(self.mines[rr, cc] for rr, cc in self._neighbors(r, c))

    def click(self, r, c):
        spec = self.spec
        if self.mines is None:
            cand = [
                (self.scores[i, j], i * spec.w + j)
                for i in range(spec.h)
                for j in range(spec.w)
                if abs(i - r) > 1 or abs(j - c) > 1
            ]
            cand.sort()
            self.mines = np.zeros((spec.h, spec.w), dtype=bool)
            for _, flat in cand[: spec.mines]:
                self.mines[flat // spec.w, flat % spec.w] = True
        if self.mines[r, c]:
            self.done = True
            return
        queue = deque([(r, c)])
        self.revealed[r, c] = True
        while queue:
            i, j = queue.popleft()
            if self.count(i, j) == 0:
                for n in self._neighbors(i, j):
                    if not self.revealed[n]:
                        self.revealed[n] = True
                        queue.append(n)
        if self.revealed.sum() == spec.safe:
            self.done = self.won = True

    def obs(self):
        out = np.full((self.spec.h, self.spec.w), HIDDEN, dtype=np.uint8)
        for i, j in zip(*np.nonzero(self.revealed)):
            out[i, j] = self.count(i, j)
        return out


@pytest.mark.parametrize("name", ["beginner", "intermediate", "expert"])
def test_matches_reference(name):
    spec = PRESETS[name]
    n = 64
    rng = np.random.default_rng(0)
    env = MinesweeperEnv(spec, n, seed=1, auto_reset=False)
    slow = [SlowBoard(spec, env.scores[i]) for i in range(n)]
    for _ in range(spec.cells):
        if env.finished.all():
            break
        actions = random_valid_actions(env.action_mask(), rng)
        was_live = ~env.finished
        obs, _, done, info = env.step(actions)
        for i in np.flatnonzero(was_live):
            slow[i].click(*divmod(int(actions[i]), spec.w))
            assert slow[i].done == done[i]
            assert slow[i].won == info["won"][i]
            assert (slow[i].obs() == obs[i]).all()
    assert env.finished.all()
    assert [s.won for s in slow] == env.won.tolist()


def test_first_click_opens_and_is_safe():
    spec = PRESETS["beginner"]
    env = MinesweeperEnv(spec, 2000, seed=3, auto_reset=False)
    rng = np.random.default_rng(4)
    actions = random_valid_actions(env.action_mask(), rng)
    obs, reward, done, info = env.step(actions)
    assert not info["hit"].any()
    r, c = np.divmod(actions, spec.w)
    assert (obs[np.arange(env.n), r, c] == 0).all()
    assert (info["revealed"] >= 4).all()  # at least a corner opening
    assert (env.mines.reshape(env.n, -1).sum(1) == spec.mines).all()
    assert not (env.mines & env.revealed).any()


def test_rewards_and_mask():
    spec = PRESETS["beginner"]
    env = MinesweeperEnv(spec, 500, seed=5, auto_reset=False, progress_reward=1.0)
    rng = np.random.default_rng(6)
    total = np.zeros(env.n)
    while not env.finished.all():
        mask = env.action_mask()
        assert (mask.sum(1) == spec.cells - env.n_revealed).all()
        _, reward, _, info = env.step(random_valid_actions(mask, rng))
        total += reward
    # a win collects all progress (1.0) plus the win bonus
    assert np.allclose(total[env.won], 2.0, atol=1e-5)
    assert (total[~env.won] < 0).all() or (total[~env.won] <= 0.0 + 1e-6).all()


def test_revealed_action_rejected():
    env = MinesweeperEnv(PRESETS["beginner"], 4, seed=7, auto_reset=False)
    env.step(np.full(4, 40))
    with pytest.raises(ValueError):
        env.step(np.full(4, 40))


def test_auto_reset_gives_fresh_board():
    spec = PRESETS["beginner"]
    env = MinesweeperEnv(spec, 256, seed=8)
    rng = np.random.default_rng(9)
    seen_done = 0
    for _ in range(200):
        obs, _, done, _ = env.step(random_valid_actions(env.action_mask(), rng))
        assert (obs[done] == HIDDEN).all()
        seen_done += done.sum()
    assert seen_done > 0
