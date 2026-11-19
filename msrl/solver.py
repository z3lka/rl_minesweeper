"""Exact mine-probability solver.

For one observed board, `analyze` returns P(mine) for every hidden cell under
a uniform prior over all mine layouts consistent with the visible numbers and
the total mine count. Frontier cells are split into independent components,
each component is enumerated by backtracking, and components are recombined
through the shared mine budget.

Used as (a) the hand-written reference agent and (b) the judge that decides
whether a learned agent's fatal move was a blunder or a forced guess.
"""

from __future__ import annotations

from math import comb

import numpy as np

from .env import HIDDEN, BoardSpec


class _Overflow(Exception):
    pass


def _enumerate(n_vars: int, cons: list[tuple[list[int], int]], budget: int):
    """All 0/1 assignments satisfying `cons`, tallied by number of mines.

    Returns {k: (n_solutions, [times var v is a mine])}.
    """
    var_cons = [[] for _ in range(n_vars)]
    for ci, (vs, _) in enumerate(cons):
        for v in vs:
            var_cons[v].append(ci)
    need = [cnt for _, cnt in cons]
    free = [len(vs) for vs, _ in cons]
    assign = [0] * n_vars
    sols: dict[int, list] = {}
    nodes = 0

    def rec(i: int, k: int):
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise _Overflow
        if i == n_vars:
            entry = sols.get(k)
            if entry is None:
                entry = sols[k] = [0, [0] * n_vars]
            entry[0] += 1
            tally = entry[1]
            for v in range(n_vars):
                if assign[v]:
                    tally[v] += 1
            return
        cs = var_cons[i]
        for val in (0, 1):
            ok = True
            for c in cs:
                free[c] -= 1
                need[c] -= val
            for c in cs:
                if need[c] < 0 or need[c] > free[c]:
                    ok = False
                    break
            if ok:
                assign[i] = val
                rec(i + 1, k + val)
            for c in cs:
                free[c] += 1
                need[c] += val
        assign[i] = 0

    rec(0, 0)
    return sols


def _conv(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, x in a.items():
        for j, y in b.items():
            out[i + j] = out.get(i + j, 0) + x * y
    return out


def analyze(obs: np.ndarray, n_mines: int, budget: int = 300_000):
    """Return (prob, exact). prob is (H, W) float, NaN on revealed cells."""
    h, w = obs.shape
    hidden = obs == HIDDEN
    prob = np.full((h, w), np.nan)
    if hidden.all():
        prob[:] = 0.0  # the first click is safe by construction
        return prob, True

    # constraints: (hidden neighbours of a revealed number) must contain `count` mines
    var_id: dict[int, int] = {}
    cons_map: dict[tuple[int, ...], int] = {}
    for r, c in zip(*np.nonzero(~hidden)):
        vs = []
        for rr in range(max(r - 1, 0), min(r + 2, h)):
            for cc in range(max(c - 1, 0), min(c + 2, w)):
                if hidden[rr, cc]:
                    vs.append(var_id.setdefault(rr * w + cc, len(var_id)))
        if vs:
            cons_map[tuple(sorted(vs))] = int(obs[r, c])
    cons = list(cons_map.items())
    n_vars = len(var_id)
    flat_of = [0] * n_vars
    for flat, v in var_id.items():
        flat_of[v] = flat
    interior = int(hidden.sum()) - n_vars

    # connected components, variables ordered so constraints complete early
    var_cons = [[] for _ in range(n_vars)]
    for ci, (vs, _) in enumerate(cons):
        for v in vs:
            var_cons[v].append(ci)
    seen_v = [False] * n_vars
    seen_c = [False] * len(cons)
    comps = []
    for start in range(n_vars):
        if seen_v[start]:
            continue
        order, comp_cons, queue = [], [], [start]
        seen_v[start] = True
        while queue:
            v = queue.pop(0)
            order.append(v)
            for ci in var_cons[v]:
                if not seen_c[ci]:
                    seen_c[ci] = True
                    comp_cons.append(ci)
                    for u in cons[ci][0]:
                        if not seen_v[u]:
                            seen_v[u] = True
                            queue.append(u)
        comps.append((order, comp_cons))

    try:
        tallies = []
        for order, comp_cons in comps:
            local = {v: i for i, v in enumerate(order)}
            lc = [([local[v] for v in cons[ci][0]], cons[ci][1]) for ci in comp_cons]
            tallies.append(_enumerate(len(order), lc, budget))
    except _Overflow:
        return _estimate(prob, cons, flat_of, hidden, n_mines, interior), False

    def ways_rest(m: int, cells: int) -> int:
        return comb(cells, m) if 0 <= m <= cells else 0

    counts = [{k: e[0] for k, e in t.items()} for t in tallies]
    total = {0: 1}
    for cnt in counts:
        total = _conv(total, cnt)
    z = sum(n * ways_rest(n_mines - k, interior) for k, n in total.items())
    if z == 0:
        raise ValueError("observation is inconsistent with the mine count")

    for ci, (order, _) in enumerate(comps):
        others = {0: 1}
        for cj, cnt in enumerate(counts):
            if cj != ci:
                others = _conv(others, cnt)
        num = [0] * len(order)
        for k, (_, tally) in tallies[ci].items():
            weight = sum(n * ways_rest(n_mines - k - j, interior) for j, n in others.items())
            if weight:
                for i, t in enumerate(tally):
                    num[i] += t * weight
        for i, v in enumerate(order):
            prob.flat[flat_of[v]] = num[i] / z

    if interior:
        num = sum(n * ways_rest(n_mines - k - 1, interior - 1) for k, n in total.items())
        frontier = np.zeros(h * w, dtype=bool)
        frontier[flat_of] = True
        prob[hidden & ~frontier.reshape(h, w)] = num / z
    return prob, True


def _estimate(prob, cons, flat_of, hidden, n_mines, interior):
    """Cheap fallback when enumeration is too large: worst local density."""
    h, w = prob.shape
    est = np.zeros(len(flat_of))
    for vs, cnt in cons:
        for v in vs:
            est[v] = max(est[v], cnt / len(vs))
    frontier = np.zeros(h * w, dtype=bool)
    frontier[flat_of] = True
    prob.flat[flat_of] = est
    if interior:
        prob[hidden & ~frontier.reshape(h, w)] = min(max((n_mines - est.sum()) / interior, 0.0), 1.0)
    return prob


class SolverAgent:
    """Clicks every provably safe cell; otherwise guesses the least likely mine."""

    name = "solver"

    def __init__(self, spec: BoardSpec, first_click: int | None = None):
        self.spec = spec
        self.first_click = first_click if first_click is not None else (spec.h // 2) * spec.w + spec.w // 2
        self.queues: dict[int, list[int]] = {}

    def act(self, obs: np.ndarray, mask: np.ndarray, live: np.ndarray) -> np.ndarray:
        actions = np.zeros(len(obs), dtype=np.int64)
        for i in np.flatnonzero(live):
            queue = [a for a in self.queues.get(i, ()) if mask[i, a]]
            if not queue:
                if mask[i].all():
                    queue = [self.first_click]
                else:
                    p = analyze(obs[i], self.spec.mines)[0].ravel()
                    safe = np.flatnonzero(p == 0)
                    queue = safe.tolist() if safe.size else [int(np.nanargmin(p))]
            actions[i] = queue.pop()
            self.queues[i] = queue
        return actions
