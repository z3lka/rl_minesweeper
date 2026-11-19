"""Fixed-seed evaluation shared by every agent.

    uv run python -m msrl.evaluate --agent solver --board beginner
    uv run python -m msrl.evaluate --agent runs/dqn_beginner/best.pt --board beginner

All agents face the same score fields (see env.py), so results are comparable
game by game. Each lost game is classified by the exact solver:

    blunder    a provably safe cell existed, the agent clicked a mine anyway
    bad_guess  no safe cell existed, but the agent's cell was riskier than the best
    unlucky    the agent took (one of) the least risky cells and still lost
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .env import PRESETS, BoardSpec, MinesweeperEnv, random_valid_actions
from .models import BoardNet, pick_device
from .solver import SolverAgent, analyze

EVAL_SEED = 20261119


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, live):
        return random_valid_actions(mask, self.rng)


class NetAgent:
    """Greedy agent: reveal the valid cell with the highest network output."""

    def __init__(self, net: BoardNet, device, name: str = "net", first_click: int | None = None):
        self.net, self.device, self.name, self.first_click = net.eval(), device, name, first_click

    @torch.no_grad()
    def scores(self, obs: np.ndarray) -> torch.Tensor:
        return self.net(torch.from_numpy(obs).to(self.device))[0]

    def act(self, obs, mask, live):
        q = self.scores(obs).masked_fill(~torch.from_numpy(mask).to(self.device), -torch.inf)
        actions = q.argmax(1).cpu().numpy()
        if self.first_click is not None:
            actions[mask.all(1)] = self.first_click
        return actions


def save_checkpoint(path, net: BoardNet, **meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": net.config, "state_dict": net.state_dict(), "meta": meta}, path)


def load_net(path, device="cpu") -> tuple[BoardNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = BoardNet(**ckpt["config"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net, ckpt.get("meta", {})


def evaluate(agent, spec: BoardSpec, games: int = 10_000, seed: int = EVAL_SEED, judge: bool = True) -> dict:
    scores = np.random.default_rng(seed).random((games, spec.h, spec.w), dtype=np.float32)
    env = MinesweeperEnv(spec, games, auto_reset=False, scores=scores)
    moves = np.zeros(games, dtype=np.int64)
    verdicts = {"blunder": 0, "bad_guess": 0, "unlucky": 0, "unknown": 0}
    t = time.perf_counter()
    while not env.finished.all():
        live = ~env.finished
        obs = env.obs()
        actions = agent.act(obs, env.action_mask(), live)
        _, _, _, info = env.step(actions)
        moves += live
        if judge:
            for i in np.flatnonzero(info["hit"]):
                p, exact = analyze(obs[i], spec.mines)
                p = p.ravel()
                if not exact:
                    verdicts["unknown"] += 1
                elif (p == 0).any():
                    verdicts["blunder"] += 1
                elif p[actions[i]] > np.nanmin(p) + 1e-9:
                    verdicts["bad_guess"] += 1
                else:
                    verdicts["unlucky"] += 1
    out = {
        "agent": getattr(agent, "name", type(agent).__name__),
        "board": {"h": spec.h, "w": spec.w, "mines": spec.mines},
        "games": games,
        "seed": seed,
        "win_rate": float(env.won.mean()),
        "cleared": float((env.n_revealed / spec.safe).mean()),
        "mean_moves": float(moves.mean()),
        "eval_seconds": round(time.perf_counter() - t, 2),
    }
    if judge:
        out["losses"] = {k: v / games for k, v in verdicts.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="random | solver | path to checkpoint")
    ap.add_argument("--board", default="beginner", choices=PRESETS)
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--first-click", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="write JSON here (default results/<board>/<name>.json)")
    args = ap.parse_args()

    spec = PRESETS[args.board]
    if args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "solver":
        agent = SolverAgent(spec, args.first_click)
    else:
        device = pick_device(args.device)
        net, meta = load_net(args.agent, device)
        first = args.first_click if args.first_click is not None else meta.get("first_click")
        agent = NetAgent(net, device, name=meta.get("algo", "net"), first_click=first)
    if args.name:
        agent.name = args.name

    result = evaluate(agent, spec, args.games)
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else Path("results") / args.board / f"{agent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
