"""Shared training plumbing: CLI flags, metric logging, periodic and final evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .env import PRESETS
from .evaluate import NetAgent, evaluate, load_net, save_checkpoint
from .models import BoardNet, pick_device

DEV_SEED = 7  # periodic evaluation during training; the final report uses EVAL_SEED


def base_parser(algo: str, env_steps: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="beginner", choices=PRESETS)
    ap.add_argument("--name", default=None, help=f"run name (default {algo}_<board>)")
    ap.add_argument("--env-steps", type=float, default=env_steps)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=float, default=500_000)
    ap.add_argument("--eval-games", type=int, default=2000)
    return ap


def build_net(args, device, value_head: bool) -> BoardNet:
    if args.init:
        net, _ = load_net(args.init, device)
        if value_head and net.value_head is None:
            raise ValueError("--init checkpoint has no value head")
        return net
    return BoardNet(args.channels, args.blocks, value_head).to(device)


class Run:
    def __init__(self, algo: str, args, first_click: int | None = None):
        self.algo, self.args = algo, args
        self.name = args.name or f"{algo}_{args.board}"
        self.spec = PRESETS[args.board]
        self.device = pick_device(args.device)
        self.first_click = first_click
        self.dir = Path("runs") / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        self.metrics = open(self.dir / "metrics.jsonl", "w")
        self.curve: list[dict] = []
        self.best = -1.0
        self.next_eval = 0.0
        self.t0 = time.perf_counter()
        torch.manual_seed(args.seed)
        print(f"[{self.name}] device={self.device} board={args.board}", flush=True)

    def _agent(self, net):
        return NetAgent(net, self.device, name=self.name, first_click=self.first_click)

    def maybe_eval(self, net, env_steps: int, force: bool = False, **extra):
        if not force and env_steps < self.next_eval:
            return
        # dense early on (learning is fastest there), then every eval_every steps
        self.next_eval = min(env_steps * 1.5 + 20_000, env_steps + self.args.eval_every)
        r = evaluate(self._agent(net), self.spec, self.args.eval_games, seed=DEV_SEED, judge=False)
        net.train()
        row = {
            "env_steps": int(env_steps),
            "wall": round(time.perf_counter() - self.t0, 1),
            "win_rate": r["win_rate"],
            "cleared": round(r["cleared"], 4),
            **{k: round(float(v), 5) for k, v in extra.items()},
        }
        self.curve.append(row)
        self.metrics.write(json.dumps(row) + "\n")
        self.metrics.flush()
        print(row, flush=True)
        meta = dict(algo=self.algo, board=self.args.board, first_click=self.first_click, env_steps=int(env_steps))
        save_checkpoint(self.dir / "last.pt", net, **meta)
        if r["win_rate"] > self.best:
            self.best = r["win_rate"]
            save_checkpoint(self.dir / "best.pt", net, **meta)

    def finish(self, net, env_steps: int):
        self.maybe_eval(net, env_steps, force=True)
        train_seconds = time.perf_counter() - self.t0
        best, meta = load_net(self.dir / "best.pt", self.device)
        result = evaluate(self._agent(best), self.spec)
        result.update(
            algo=self.algo,
            train_seconds=round(train_seconds, 1),
            env_steps=int(env_steps),
            best_at_env_steps=meta["env_steps"],
            params=sum(p.numel() for p in best.parameters()),
            device=str(self.device),
            config=vars(self.args),
            curve=self.curve,
        )
        out = Path("results") / self.args.board / f"{self.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        summary = {k: result[k] for k in ("win_rate", "cleared", "losses", "train_seconds")}
        print(f"[{self.name}] final {summary} -> {out}", flush=True)
