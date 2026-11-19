"""Throughput benchmark: env steps/s and network samples/s on CPU vs MPS.

    uv run python -m msrl.bench
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .env import PRESETS, MinesweeperEnv, random_valid_actions
from .models import BoardNet


def bench_env(name: str, n: int, steps: int = 200) -> float:
    env = MinesweeperEnv(PRESETS[name], n, seed=0)
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(steps):
        env.step(random_valid_actions(env.action_mask(), rng))
    return n * steps / (time.perf_counter() - t)


def bench_net(name: str, device: str, batch: int, iters: int = 8):
    spec = PRESETS[name]
    dev = torch.device(device)
    net = BoardNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    obs = torch.randint(0, 10, (batch, spec.h, spec.w), dtype=torch.uint8, device=dev)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def run(train: bool):
        for i in range(iters + 2):
            if i == 2:
                sync()
                t = time.perf_counter()
            if train:
                loss = net(obs)[0].square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    net(obs)[0].argmax(1).cpu()
        sync()
        return batch * iters / (time.perf_counter() - t)

    return run(False), run(True)


def main():
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    print("\nenv steps/s (random policy)")
    for name in PRESETS:
        for n in (256, 1024, 4096):
            print(f"  {name:13s} n={n:5d}  {bench_env(name, n):12,.0f}")
    print("\nnetwork samples/s (6x64 ResNet): inference / train step")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for name in ("beginner", "intermediate"):
        for device in devices:
            for batch in (256, 1024) if device == "cpu" else (256, 1024, 4096):
                inf, tr = bench_net(name, device, batch)
                print(f"  {name:13s} {device:4s} batch={batch:5d}  {inf:12,.0f} / {tr:12,.0f}")


if __name__ == "__main__":
    main()
