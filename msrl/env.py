"""Batched Minesweeper environment.

N boards are stepped at once with NumPy. The only action is "reveal a cell"
(flat index row * W + col). Mines are placed on the first click so that the
clicked cell and its 8 neighbours are mine-free, which guarantees an opening.

A board is a deterministic function of its `scores` field and the first click:
the M lowest-scoring cells outside the first-click zone become mines. That
makes evaluation reproducible (every agent gets the same score fields) and
lets the TypeScript port be checked against this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIDDEN = 9  # observation value of an unrevealed cell; 0-8 are neighbour counts
N_PLANES = 10


@dataclass(frozen=True)
class BoardSpec:
    h: int
    w: int
    mines: int

    def __post_init__(self):
        if self.cells - 9 < self.mines:
            raise ValueError("too many mines for a guaranteed first-click opening")

    @property
    def cells(self) -> int:
        return self.h * self.w

    @property
    def safe(self) -> int:
        return self.cells - self.mines


PRESETS = {
    "beginner": BoardSpec(9, 9, 10),
    "intermediate": BoardSpec(16, 16, 40),
    "expert": BoardSpec(16, 30, 99),
}


def _shifts(x: np.ndarray):
    """Yield the 8 neighbour-shifted views of a (N, H, W) array, zero padded."""
    h, w = x.shape[1:]
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                yield p[:, dy : dy + h, dx : dx + w]


def neighbor_sum(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape, dtype=np.int8)
    for s in _shifts(x):
        out += s
    return out


def dilate(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for s in _shifts(x):
        out |= s
    return out


def place_mines(scores: np.ndarray, r: np.ndarray, c: np.ndarray, n_mines: int) -> np.ndarray:
    """Mines for boards with score fields `scores` (K, H, W) and first clicks (r, c)."""
    k, h, w = scores.shape
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    zone = (np.abs(rows - r[:, None, None]) <= 1) & (np.abs(cols - c[:, None, None]) <= 1)
    s = np.where(zone, np.inf, scores).reshape(k, -1)
    idx = np.argpartition(s, n_mines - 1, axis=1)[:, :n_mines]
    mines = np.zeros((k, h * w), dtype=bool)
    np.put_along_axis(mines, idx, True, axis=1)
    return mines.reshape(k, h, w)


class MinesweeperEnv:
    """Batch of independent boards.

    auto_reset=True  : a finished board is replaced by a fresh one in the same
                       step; the returned observation is the fresh board.
    auto_reset=False : finished boards freeze; actions on them are ignored.
    """

    def __init__(
        self,
        spec: BoardSpec,
        n: int,
        seed: int | None = None,
        progress_reward: float = 1.0,
        auto_reset: bool = True,
        scores: np.ndarray | None = None,
    ):
        self.spec = spec
        self.n = n
        self.progress_reward = progress_reward
        self.auto_reset = auto_reset
        self.rng = np.random.default_rng(seed)
        shape = (n, spec.h, spec.w)
        if scores is None:
            scores = self.rng.random(shape, dtype=np.float32)
        elif scores.shape != shape:
            raise ValueError(f"scores must have shape {shape}")
        self.scores = scores.copy()
        self.mines = np.zeros(shape, dtype=bool)
        self.counts = np.zeros(shape, dtype=np.int8)
        self.revealed = np.zeros(shape, dtype=bool)
        self.started = np.zeros(n, dtype=bool)
        self.finished = np.zeros(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)
        self.n_revealed = np.zeros(n, dtype=np.int64)
        self._idx = np.arange(n)

    # ---- views -----------------------------------------------------------
    def obs(self) -> np.ndarray:
        """(N, H, W) uint8: 0-8 for revealed cells, HIDDEN otherwise."""
        return np.where(self.revealed, self.counts, HIDDEN).astype(np.uint8)

    def action_mask(self) -> np.ndarray:
        """(N, H*W) bool: True where the cell can still be revealed."""
        return ~self.revealed.reshape(self.n, -1)

    # ---- dynamics --------------------------------------------------------
    def step(self, actions: np.ndarray):
        spec, idx = self.spec, self._idx
        actions = np.asarray(actions)
        r, c = np.divmod(actions, spec.w)
        live = ~self.finished
        if self.revealed[idx, r, c][live].any():
            raise ValueError("action on an already revealed cell")

        fresh = np.flatnonzero(live & ~self.started)
        if fresh.size:
            m = place_mines(self.scores[fresh], r[fresh], c[fresh], spec.mines)
            self.mines[fresh] = m
            self.counts[fresh] = neighbor_sum(m)
            self.started[fresh] = True

        hit = live & self.mines[idx, r, c]
        ok = live & ~hit
        self.revealed[idx[ok], r[ok], c[ok]] = True
        flood = np.flatnonzero(ok & (self.counts[idx, r, c] == 0))
        if flood.size:
            self._flood(flood, r[flood], c[flood])

        n_rev = self.revealed.reshape(self.n, -1).sum(1)
        new = n_rev - self.n_revealed
        self.n_revealed = n_rev
        won = ok & (n_rev == spec.safe)
        reward = (self.progress_reward * new / spec.safe).astype(np.float32)
        reward[won] += 1.0
        reward[hit] = -1.0
        done = hit | won
        info = {"won": won, "hit": hit, "revealed": n_rev.copy()}

        if self.auto_reset:
            self._reset(np.flatnonzero(done))
        else:
            self.finished |= done
            self.won |= won
        return self.obs(), reward, done, info

    def _flood(self, k: np.ndarray, r: np.ndarray, c: np.ndarray):
        rev = self.revealed[k]
        zero = self.counts[k] == 0
        frontier = np.zeros_like(rev)
        frontier[np.arange(k.size), r, c] = True
        while True:
            grow = dilate(frontier & zero) & ~rev
            if not grow.any():
                break
            rev |= grow
            frontier = grow
        self.revealed[k] = rev

    def _reset(self, k: np.ndarray):
        if not k.size:
            return
        self.scores[k] = self.rng.random((k.size, self.spec.h, self.spec.w), dtype=np.float32)
        self.mines[k] = False
        self.counts[k] = 0
        self.revealed[k] = False
        self.started[k] = False
        self.n_revealed[k] = 0


def random_valid_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random True index per row of a (N, A) bool mask."""
    return (rng.random(mask.shape, dtype=np.float32) * mask).argmax(1)
