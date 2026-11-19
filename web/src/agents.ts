// The five players, as they run in the browser.
// Networks are the ONNX exports of the PyTorch checkpoints; the solver and
// the random clicker are plain TypeScript.

import * as ort from "onnxruntime-web/wasm";
import { Game, HIDDEN } from "./game/board";
import { analyze } from "./game/solver";

ort.env.wasm.numThreads = 1;

export type AgentId = "supervised" | "dqn" | "ppo" | "solver" | "random";

export const AGENTS: { id: AgentId; name: string; kind: string; about: string; heat: string }[] = [
  {
    id: "supervised",
    name: "Supervised",
    kind: "Neural network, trained on the answer key",
    about:
      "For every hidden cell it predicts whether a mine is underneath, having been shown the true mines during training. It plays by opening the cell it believes safest. This is not reinforcement learning; it is the bar reinforcement learning has to reach.",
    heat: "Blue is the predicted chance that a cell is safe.",
  },
  {
    id: "dqn",
    name: "DQN",
    kind: "Neural network, trained from reward",
    about:
      "Deep Q-learning. It is never told where the mines are. It only receives a share of one point for each cell it opens, one more for a win and minus one for a mine, and from that learns what every possible click is worth.",
    heat: "Blue is the value it expects from a click, stretched across the hidden cells.",
  },
  {
    id: "ppo",
    name: "PPO",
    kind: "Neural network, trained from reward",
    about:
      "Proximal policy optimisation. Same rewards as DQN, but instead of valuing clicks it learns a probability of making each one, and nudges those probabilities toward moves that turned out better than expected.",
    heat: "Blue is its preference for a click, stretched across the hidden cells.",
  },
  {
    id: "solver",
    name: "Exact solver",
    kind: "Hand-written, no learning",
    about:
      "Counts every mine layout that agrees with the visible numbers and opens the cell that holds a mine in the fewest of them. It never makes a mistake of logic, so every game it loses is a guess it was forced to take.",
    heat: "Blue is the exact chance that a cell is safe.",
  },
  {
    id: "random",
    name: "Random",
    kind: "No strategy",
    about: "Opens any hidden cell. It is the floor that every other player has to clear, and it almost never wins.",
    heat: "No preference: every hidden cell looks the same to it.",
  },
];

export interface Thought {
  heat: Float32Array; // 0..1 per cell, NaN where revealed
  pick: number;
  /** The player's own figure for the chosen cell: chance safe, expected reward or share of preference. */
  figure: number;
}

const sessions = new Map<string, Promise<ort.InferenceSession>>();

function session(url: string) {
  let s = sessions.get(url);
  if (!s) {
    s = ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    sessions.set(url, s);
  }
  return s;
}

function argBest(values: ArrayLike<number>, sign: 1 | -1): number {
  let best = -1;
  for (let i = 0; i < values.length; i++) {
    if (Number.isNaN(values[i])) continue;
    if (best < 0 || sign * values[i] > sign * values[best]) best = i;
  }
  return best;
}

export async function think(id: AgentId, game: Game, modelUrl?: string): Promise<Thought> {
  const { h, w } = game.spec;
  const obs = game.obs();
  const heat = new Float32Array(h * w).fill(NaN);
  const center = Math.floor(h / 2) * w + Math.floor(w / 2);

  if (id === "random") {
    const hidden: number[] = [];
    obs.forEach((v, i) => {
      if (v === HIDDEN) {
        heat[i] = 0.35;
        hidden.push(i);
      }
    });
    return { heat, pick: hidden[Math.floor(Math.random() * hidden.length)], figure: NaN };
  }

  if (id === "solver") {
    const p = analyze(obs, game.spec);
    if (!p) return think("random", game);
    p.forEach((v, i) => {
      if (!Number.isNaN(v)) heat[i] = 1 - v;
    });
    const chosen = game.started ? argBest(p, -1) : center;
    return { heat, pick: chosen, figure: 1 - p[chosen] };
  }

  if (!modelUrl) throw new Error(`no model for ${id}`);
  const sess = await session(modelUrl);
  const out = await sess.run({ obs: new ort.Tensor("uint8", obs, [1, h, w]) });
  const cells = out.cells.data as Float32Array;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < cells.length; i++) {
    if (obs[i] !== HIDDEN) continue;
    lo = Math.min(lo, cells[i]);
    hi = Math.max(hi, cells[i]);
  }
  const raw = new Float32Array(h * w).fill(NaN);
  for (let i = 0; i < cells.length; i++) {
    if (obs[i] !== HIDDEN) continue;
    raw[i] = cells[i];
    heat[i] = id === "supervised" ? 1 / (1 + Math.exp(-cells[i])) : hi > lo ? (cells[i] - lo) / (hi - lo) : 0.5;
  }
  const pick = id === "supervised" && !game.started ? center : argBest(raw, 1);
  return { heat, pick, figure: view(id, cells, obs)[pick] };
}

/* ---------------------------------------------------------- explanations */
// "Why this cell?" is answered by occlusion: cover one nearby number at a
// time, ask the player again, and see how far its opinion of the cell moves.

export interface Why {
  target: number;
  value: number; // the player's figure for the target (see `view`)
  second: number | null; // its figure for the best other hidden cell
  truth: number | null; // exact chance that the target is a mine
  cells: number[]; // the numbers that were covered, one at a time
  without: number[]; // the target's figure with that number covered (NaN if unknown)
}

/** Raw per-cell output for each of `n` boards; null where the solver gave up. */
async function rawBatch(id: AgentId, game: Game, batch: Uint8Array, n: number, modelUrl?: string): Promise<(Float32Array | null)[]> {
  const { h, w } = game.spec;
  const size = h * w;
  if (id === "solver") {
    const rows: (Float32Array | null)[] = [];
    for (let k = 0; k < n; k++) {
      const p = analyze(batch.subarray(k * size, (k + 1) * size), game.spec, 60_000);
      rows.push(p ? Float32Array.from(p, (v) => 1 - v) : null);
    }
    return rows;
  }
  if (!modelUrl) throw new Error(`no model for ${id}`);
  const sess = await session(modelUrl);
  const out = await sess.run({ obs: new ort.Tensor("uint8", batch, [n, h, w]) });
  const data = out.cells.data as Float32Array;
  return Array.from({ length: n }, (_, k) => data.subarray(k * size, (k + 1) * size));
}

/** Turn raw output into the figure shown to the reader, for cells hidden in `obs`. */
function view(id: AgentId, raw: Float32Array, obs: Uint8Array): Float32Array {
  const v = new Float32Array(raw.length).fill(NaN);
  if (id === "ppo") {
    let top = -Infinity;
    for (let i = 0; i < raw.length; i++) if (obs[i] === HIDDEN) top = Math.max(top, raw[i]);
    let sum = 0;
    for (let i = 0; i < raw.length; i++) if (obs[i] === HIDDEN) sum += v[i] = Math.exp(raw[i] - top);
    for (let i = 0; i < raw.length; i++) v[i] /= sum;
    return v;
  }
  for (let i = 0; i < raw.length; i++) {
    if (obs[i] !== HIDDEN) continue;
    v[i] = id === "supervised" ? 1 / (1 + Math.exp(-raw[i])) : raw[i];
  }
  return v;
}

export async function explain(id: AgentId, game: Game, target: number, modelUrl?: string): Promise<Why | null> {
  if (id === "random" || game.done) return null;
  const { h, w } = game.spec;
  const size = h * w;
  const obs = game.obs();
  const reach = w > 9 ? 3 : 4;
  const tr = Math.floor(target / w);
  const tc = target % w;
  const cells: number[] = [];
  for (let i = 0; i < size; i++) {
    if (obs[i] === HIDDEN || obs[i] === 0) continue;
    if (Math.abs(Math.floor(i / w) - tr) <= reach && Math.abs((i % w) - tc) <= reach) cells.push(i);
  }
  const batch = new Uint8Array((cells.length + 1) * size);
  for (let k = 0; k <= cells.length; k++) batch.set(obs, k * size);
  cells.forEach((cell, k) => (batch[(k + 1) * size + cell] = HIDDEN));

  const raws = await rawBatch(id, game, batch, cells.length + 1, modelUrl);
  if (!raws[0]) return null;
  const base = view(id, raws[0], obs);
  let second: number | null = null;
  for (let i = 0; i < size; i++) {
    if (i === target || Number.isNaN(base[i])) continue;
    if (second === null || base[i] > second) second = base[i];
  }
  const without = cells.map((_, k) => {
    const raw = raws[k + 1];
    return raw ? view(id, raw, obs)[target] : NaN;
  });
  const p = game.started ? analyze(obs, game.spec) : null;
  return { target, value: base[target], second, truth: p ? p[target] : null, cells, without };
}
