import type { AgentId } from "./agents";

export interface CurvePoint {
  env_steps: number;
  wall: number;
  win_rate: number;
  cleared: number;
}

export interface Result {
  run: string;
  win_rate: number;
  cleared: number;
  mean_moves: number;
  games: number;
  losses: { blunder: number; bad_guess: number; unlucky: number; unknown: number };
  train_seconds?: number;
  env_steps?: number;
  params?: number;
  model?: string;
  curve?: CurvePoint[];
}

export type Results = Record<string, Result[]>;

export function find(results: Results | null, board: string, id: AgentId): Result | undefined {
  return results?.[board]?.find((r) => r.run === id || r.run === `${id}_${board}`);
}

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;

export function duration(seconds: number): string {
  const m = seconds / 60;
  if (m < 1) return `${Math.round(seconds)} seconds`;
  if (m < 90) return `${Math.round(m)} minutes`;
  return `${(m / 60).toFixed(1)} hours`;
}

export function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)} million`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} thousand`;
  return String(n);
}
