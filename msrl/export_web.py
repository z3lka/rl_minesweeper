"""Export everything the website needs.

    uv run python -m msrl.export_web

Writes into web/public/:
    models/<board>/<run>.onnx   board-size agnostic ONNX graphs (checked against PyTorch)
    data/results.json           every results/<board>/*.json merged into one file
and web/src/game/fixtures.json, seeded games used to test the TypeScript game port.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .env import PRESETS, MinesweeperEnv, random_valid_actions
from .evaluate import load_net

WEB = Path("web")


class CellsOnly(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs)[0]


def export_model(ckpt: Path, out: Path) -> float:
    import onnxruntime as ort

    net, _ = load_net(ckpt)
    model = CellsOnly(net).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros((1, 9, 9), dtype=torch.uint8),
        str(out),
        input_names=["obs"],
        output_names=["cells"],
        dynamic_axes={"obs": {0: "n", 1: "h", 2: "w"}, "cells": {0: "n", 1: "hw"}},
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(out))
    worst = 0.0
    for h, w in ((9, 9), (16, 16), (16, 30)):
        obs = np.random.default_rng(h * w).integers(0, 10, (3, h, w), dtype=np.uint8)
        got = sess.run(None, {"obs": obs})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(obs)).numpy()
        worst = max(worst, float(np.abs(got - want).max()))
    if worst > 1e-3:
        raise RuntimeError(f"{out}: ONNX differs from PyTorch by {worst}")
    return worst


def fixtures(games_per_board: int = 6) -> list[dict]:
    out = []
    for name in ("beginner", "intermediate"):
        spec = PRESETS[name]
        env = MinesweeperEnv(spec, games_per_board, seed=99, auto_reset=False)
        rng = np.random.default_rng(100)
        games = [
            {"h": spec.h, "w": spec.w, "mines": spec.mines, "scores": env.scores[i].ravel().tolist(), "steps": []}
            for i in range(games_per_board)
        ]
        while not env.finished.all():
            live = ~env.finished
            actions = random_valid_actions(env.action_mask(), rng)
            # steer away from mines most of the time so fixtures cover long games
            safe = random_valid_actions(env.action_mask() & ~env.mines.reshape(env.n, -1), rng)
            actions = np.where(env.started & (rng.random(env.n) < 0.9), safe, actions)
            obs, _, done, info = env.step(actions)
            for i in np.flatnonzero(live):
                games[i]["steps"].append(
                    {"action": int(actions[i]), "obs": obs[i].ravel().tolist(), "done": bool(done[i]), "won": bool(info["won"][i])}
                )
        out += games
    return out


def main():
    results = {}
    for board_dir in sorted(Path("results").iterdir()):
        rows = []
        for f in sorted(board_dir.glob("*.json")):
            r = json.loads(f.read_text())
            r["run"] = f.stem
            r.pop("config", None)
            ckpt = Path("runs") / f.stem / "best.pt"
            if ckpt.exists():
                rel = Path("models") / board_dir.name / f"{f.stem}.onnx"
                err = export_model(ckpt, WEB / "public" / rel)
                r["model"] = rel.as_posix()
                print(f"exported {rel} (max abs diff vs torch {err:.1e})")
            rows.append(r)
        results[board_dir.name] = rows
    data = WEB / "public" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "results.json").write_text(json.dumps(results))
    fx = WEB / "src" / "game" / "fixtures.json"
    fx.parent.mkdir(parents=True, exist_ok=True)
    fx.write_text(json.dumps(fixtures()))
    print(f"wrote {data / 'results.json'} and {fx}")


if __name__ == "__main__":
    main()
