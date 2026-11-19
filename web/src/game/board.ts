// TypeScript port of msrl/env.py for a single board. Checked against the
// Python implementation by board.test.ts using fixtures.json.

export const HIDDEN = 9;

export interface Spec {
  h: number;
  w: number;
  mines: number;
}

export const PRESETS: Record<string, Spec> = {
  beginner: { h: 9, w: 9, mines: 10 },
  intermediate: { h: 16, w: 16, mines: 40 },
};

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Game {
  readonly spec: Spec;
  readonly scores: ArrayLike<number>;
  mines: Uint8Array | null = null;
  counts: Uint8Array;
  revealed: Uint8Array;
  nRevealed = 0;
  done = false;
  won = false;
  hit = -1;
  moves = 0;

  constructor(spec: Spec, scores: ArrayLike<number>) {
    this.spec = spec;
    this.scores = scores;
    this.counts = new Uint8Array(spec.h * spec.w);
    this.revealed = new Uint8Array(spec.h * spec.w);
  }

  static random(spec: Spec, rng: () => number = Math.random): Game {
    const scores = new Float32Array(spec.h * spec.w);
    for (let i = 0; i < scores.length; i++) scores[i] = rng();
    return new Game(spec, scores);
  }

  get started(): boolean {
    return this.mines !== null;
  }

  neighbors(a: number): number[] {
    const { h, w } = this.spec;
    const r = Math.floor(a / w);
    const c = a % w;
    const out: number[] = [];
    for (let rr = Math.max(r - 1, 0); rr <= Math.min(r + 1, h - 1); rr++)
      for (let cc = Math.max(c - 1, 0); cc <= Math.min(c + 1, w - 1); cc++)
        if (rr !== r || cc !== c) out.push(rr * w + cc);
    return out;
  }

  private place(first: number) {
    const { h, w, mines } = this.spec;
    const zone = new Set([first, ...this.neighbors(first)]);
    const order: number[] = [];
    for (let i = 0; i < h * w; i++) if (!zone.has(i)) order.push(i);
    order.sort((x, y) => this.scores[x] - this.scores[y]);
    this.mines = new Uint8Array(h * w);
    for (const i of order.slice(0, mines)) this.mines[i] = 1;
    for (let i = 0; i < h * w; i++) {
      let n = 0;
      for (const j of this.neighbors(i)) n += this.mines[j];
      this.counts[i] = n;
    }
  }

  /** Reveal cell `a`. Returns the cells that became visible. */
  click(a: number): number[] {
    if (this.done || this.revealed[a]) return [];
    if (!this.mines) this.place(a);
    this.moves++;
    if (this.mines![a]) {
      this.done = true;
      this.hit = a;
      return [];
    }
    const opened = [a];
    this.revealed[a] = 1;
    for (let q = 0; q < opened.length; q++) {
      const cur = opened[q];
      if (this.counts[cur] !== 0) continue;
      for (const n of this.neighbors(cur)) {
        if (!this.revealed[n]) {
          this.revealed[n] = 1;
          opened.push(n);
        }
      }
    }
    this.nRevealed += opened.length;
    if (this.nRevealed === this.spec.h * this.spec.w - this.spec.mines) {
      this.done = true;
      this.won = true;
    }
    return opened;
  }

  obs(): Uint8Array {
    const out = new Uint8Array(this.revealed.length);
    for (let i = 0; i < out.length; i++) out[i] = this.revealed[i] ? this.counts[i] : HIDDEN;
    return out;
  }
}
