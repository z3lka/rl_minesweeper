import { expect, test } from "vitest";
import { Game } from "./board";
import fixtures from "./fixtures.json";
import { analyze } from "./solver";

interface Fixture {
  h: number;
  w: number;
  mines: number;
  scores: number[];
  steps: { action: number; obs: number[]; done: boolean; won: boolean }[];
}

test("TypeScript board matches the Python environment", () => {
  const games = fixtures as Fixture[];
  expect(games.length).toBeGreaterThan(5);
  for (const fx of games) {
    const game = new Game(fx, fx.scores);
    for (const step of fx.steps) {
      const before = game.obs();
      game.click(step.action);
      expect(game.done).toBe(step.done);
      expect(game.won).toBe(step.won);
      // on a mine hit the Python obs is unchanged, as here
      expect(Array.from(game.obs())).toEqual(step.done && !step.won ? Array.from(before) : step.obs);
    }
  }
});

test("solver probabilities are consistent with the real mines", () => {
  let checked = 0;
  let skipped = 0;
  for (const fx of fixtures as Fixture[]) {
    const game = new Game(fx, fx.scores);
    for (const step of fx.steps.slice(0, -1)) {
      game.click(step.action);
      const p = analyze(game.obs(), fx);
      if (!p) {
        skipped++; // enumeration over budget, same fallback as Python
        continue;
      }
      checked++;
      let sum = 0;
      p.forEach((v, i) => {
        if (Number.isNaN(v)) return;
        sum += v;
        if (v === 0) expect(game.mines![i]).toBe(0);
        if (v === 1) expect(game.mines![i]).toBe(1);
      });
      expect(sum).toBeCloseTo(fx.mines, 6);
    }
  }
  expect(checked).toBeGreaterThan(skipped * 5);
});
