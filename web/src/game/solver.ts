// TypeScript port of msrl/solver.py: exact P(mine) for every hidden cell.
// Counts are kept as doubles; they stay far below 1e308 on the boards used
// here, and a probability of exactly 0 remains exactly 0.

import { HIDDEN, type Spec } from "./board";

type Tally = Map<number, { n: number; per: number[] }>;

function enumerate(nVars: number, cons: { vars: number[]; count: number }[], budget: number): Tally | null {
  const varCons: number[][] = Array.from({ length: nVars }, () => []);
  cons.forEach((c, ci) => c.vars.forEach((v) => varCons[v].push(ci)));
  const need = cons.map((c) => c.count);
  const free = cons.map((c) => c.vars.length);
  const assign = new Uint8Array(nVars);
  const sols: Tally = new Map();
  let nodes = 0;
  let overflow = false;

  const rec = (i: number, k: number) => {
    if (overflow) return;
    if (++nodes > budget) {
      overflow = true;
      return;
    }
    if (i === nVars) {
      let e = sols.get(k);
      if (!e) sols.set(k, (e = { n: 0, per: new Array(nVars).fill(0) }));
      e.n++;
      for (let v = 0; v < nVars; v++) if (assign[v]) e.per[v]++;
      return;
    }
    const cs = varCons[i];
    for (let val = 0; val <= 1; val++) {
      let ok = true;
      for (const c of cs) {
        free[c]--;
        need[c] -= val;
      }
      for (const c of cs)
        if (need[c] < 0 || need[c] > free[c]) {
          ok = false;
          break;
        }
      if (ok) {
        assign[i] = val;
        rec(i + 1, k + val);
      }
      for (const c of cs) {
        free[c]++;
        need[c] += val;
      }
    }
    assign[i] = 0;
  };
  rec(0, 0);
  return overflow ? null : sols;
}

function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let out = 1;
  for (let i = 1; i <= k; i++) out = (out * (n - k + i)) / i;
  return Math.round(out);
}

function conv(a: Map<number, number>, b: Map<number, number>): Map<number, number> {
  const out = new Map<number, number>();
  for (const [i, x] of a) for (const [j, y] of b) out.set(i + j, (out.get(i + j) ?? 0) + x * y);
  return out;
}

/** P(mine) per cell; NaN on revealed cells. Returns null if enumeration is too large. */
export function analyze(obs: ArrayLike<number>, spec: Spec, budget = 300_000): Float64Array | null {
  const { h, w, mines } = spec;
  const prob = new Float64Array(h * w).fill(NaN);
  let nHidden = 0;
  for (let i = 0; i < h * w; i++) if (obs[i] === HIDDEN) nHidden++;
  if (nHidden === h * w) return prob.fill(0);

  const varId = new Map<number, number>();
  const consMap = new Map<string, { vars: number[]; count: number }>();
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      if (obs[r * w + c] === HIDDEN) continue;
      const vs: number[] = [];
      for (let rr = Math.max(r - 1, 0); rr <= Math.min(r + 1, h - 1); rr++)
        for (let cc = Math.max(c - 1, 0); cc <= Math.min(c + 1, w - 1); cc++) {
          const flat = rr * w + cc;
          if (obs[flat] !== HIDDEN) continue;
          if (!varId.has(flat)) varId.set(flat, varId.size);
          vs.push(varId.get(flat)!);
        }
      if (vs.length) {
        vs.sort((x, y) => x - y);
        consMap.set(vs.join(","), { vars: vs, count: obs[r * w + c] });
      }
    }
  const cons = [...consMap.values()];
  const nVars = varId.size;
  const flatOf = new Array<number>(nVars);
  for (const [flat, v] of varId) flatOf[v] = flat;
  const interior = nHidden - nVars;

  const varCons: number[][] = Array.from({ length: nVars }, () => []);
  cons.forEach((c, ci) => c.vars.forEach((v) => varCons[v].push(ci)));
  const seenV = new Uint8Array(nVars);
  const seenC = new Uint8Array(cons.length);
  const comps: { order: number[]; tally: Tally }[] = [];
  for (let start = 0; start < nVars; start++) {
    if (seenV[start]) continue;
    const order: number[] = [];
    const compCons: number[] = [];
    const queue = [start];
    seenV[start] = 1;
    while (queue.length) {
      const v = queue.shift()!;
      order.push(v);
      for (const ci of varCons[v]) {
        if (seenC[ci]) continue;
        seenC[ci] = 1;
        compCons.push(ci);
        for (const u of cons[ci].vars)
          if (!seenV[u]) {
            seenV[u] = 1;
            queue.push(u);
          }
      }
    }
    const local = new Map(order.map((v, i) => [v, i]));
    const tally = enumerate(
      order.length,
      compCons.map((ci) => ({ vars: cons[ci].vars.map((v) => local.get(v)!), count: cons[ci].count })),
      budget,
    );
    if (!tally) return null;
    comps.push({ order, tally });
  }

  const counts = comps.map((c) => new Map([...c.tally].map(([k, e]) => [k, e.n])));
  let total = new Map([[0, 1]]);
  for (const c of counts) total = conv(total, c);
  let z = 0;
  for (const [k, n] of total) z += n * comb(interior, mines - k);
  if (z === 0) return null;

  comps.forEach((comp, ci) => {
    let others = new Map([[0, 1]]);
    counts.forEach((c, cj) => {
      if (cj !== ci) others = conv(others, c);
    });
    const num = new Array(comp.order.length).fill(0);
    for (const [k, e] of comp.tally) {
      let weight = 0;
      for (const [j, n] of others) weight += n * comb(interior, mines - k - j);
      if (weight) e.per.forEach((t, i) => (num[i] += t * weight));
    }
    comp.order.forEach((v, i) => (prob[flatOf[v]] = num[i] / z));
  });

  if (interior) {
    let num = 0;
    for (const [k, n] of total) num += n * comb(interior - 1, mines - k - 1);
    for (let i = 0; i < h * w; i++) if (obs[i] === HIDDEN && !varId.has(i)) prob[i] = num / z;
  }
  return prob;
}
