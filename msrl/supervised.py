"""Supervised baseline: predict which hidden cells are safe.

Not reinforcement learning. The simulator knows where the mines are, so every
visited board gives a full grid of labels; the agent plays by clicking the
cell it believes safest. States come from the model's own play (plus a few
random safe moves so games reach the late stage even early in training).

    uv run python -m msrl.supervised --board beginner
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, base_parser, build_net
from .env import HIDDEN, MinesweeperEnv, random_valid_actions


def main():
    ap = base_parser("supervised", env_steps=15e6)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--explore", type=float, default=0.1, help="share of random safe moves")
    args = ap.parse_args()

    spec_center = None
    run = Run("supervised", args)
    spec = run.spec
    run.first_click = spec_center = (spec.h // 2) * spec.w + spec.w // 2
    dev = run.device
    net = build_net(args, dev, value_head=False)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    n = args.envs
    total = int(args.env_steps) // n
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(i / total, 1))))
    env = MinesweeperEnv(spec, n, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    obs = env.obs()
    loss_avg = 0.0

    for step in range(total):
        run.maybe_eval(net, step * n, loss=loss_avg)
        o = torch.from_numpy(obs).to(dev)
        hidden = (o == HIDDEN).flatten(1)
        mines = env.mines.reshape(n, -1)
        target = torch.from_numpy(~mines).to(dev).float()
        weight = hidden & torch.from_numpy(env.started).to(dev)[:, None]
        logits = net(o)[0]
        loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        loss_avg = loss.item() if step == 0 else 0.99 * loss_avg + 0.01 * loss.item()

        greedy = logits.detach().masked_fill(~hidden, -torch.inf).argmax(1).cpu().numpy()
        safe = random_valid_actions(env.action_mask() & ~mines, rng)
        use_safe = (rng.random(n) < args.explore) | ~env.started
        obs, _, _, _ = env.step(np.where(use_safe, safe, greedy))

    run.finish(net, total * n)


if __name__ == "__main__":
    main()
