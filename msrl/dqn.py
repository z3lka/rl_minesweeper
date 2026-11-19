"""Double dueling DQN with action masking.

The agent only sees rewards: +1 win, -1 mine, and (optionally) a share of 1.0
for every newly revealed cell. Invalid actions (revealed cells) are masked in
both action selection and the bootstrap target.

    uv run python -m msrl.dqn --board beginner
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, base_parser, build_net
from .env import HIDDEN, MinesweeperEnv, random_valid_actions


class Replay:
    def __init__(self, capacity: int, h: int, w: int):
        self.obs = np.zeros((capacity, h, w), dtype=np.uint8)
        self.nxt = np.zeros((capacity, h, w), dtype=np.uint8)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, obs, act, rew, nxt, done):
        idx = (self.pos + np.arange(len(act))) % self.capacity
        self.obs[idx], self.nxt[idx], self.act[idx], self.rew[idx], self.done[idx] = obs, nxt, act, rew, done
        self.pos = (self.pos + len(act)) % self.capacity
        self.size = min(self.size + len(act), self.capacity)

    def sample(self, batch: int, rng, device):
        i = rng.integers(0, self.size, batch)
        return tuple(torch.from_numpy(x[i]).to(device) for x in (self.obs, self.act, self.rew, self.nxt, self.done))


def q_values(net, obs):
    adv, value = net(obs)
    return value[:, None] + adv - adv.mean(1, keepdim=True)


def main():
    ap = base_parser("dqn", env_steps=15e6)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--updates", type=int, default=1, help="gradient updates per env batch step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--progress-reward", type=float, default=1.0)
    ap.add_argument("--buffer", type=float, default=1e6)
    ap.add_argument("--warmup", type=float, default=50_000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--eps-frac", type=float, default=0.2, help="fraction of training over which epsilon decays")
    args = ap.parse_args()

    run = Run("dqn", args)
    spec, dev, n = run.spec, run.device, args.envs
    net = build_net(args, dev, value_head=True)
    target = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    env = MinesweeperEnv(spec, n, seed=args.seed, progress_reward=args.progress_reward)
    rng = np.random.default_rng(args.seed + 1)
    buf = Replay(int(args.buffer), spec.h, spec.w)
    total = int(args.env_steps) // n
    obs = env.obs()
    loss_avg, q_avg = 0.0, 0.0

    for step in range(total):
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * step / (args.eps_frac * total))
        run.maybe_eval(net, step * n, loss=loss_avg, q=q_avg, eps=eps)

        with torch.no_grad():
            o = torch.from_numpy(obs).to(dev)
            greedy = q_values(net, o).masked_fill((o != HIDDEN).flatten(1), -torch.inf).argmax(1).cpu().numpy()
        explore = rng.random(n) < eps
        actions = np.where(explore, random_valid_actions(env.action_mask(), rng), greedy)
        nxt, rew, done, _ = env.step(actions)
        buf.add(obs, actions, rew, nxt, done)
        obs = nxt

        if buf.size < args.warmup:
            continue
        for _ in range(args.updates):
            o, a, r, o2, d = buf.sample(args.batch, rng, dev)
            with torch.no_grad():
                invalid = (o2 != HIDDEN).flatten(1)
                a2 = q_values(net, o2).masked_fill(invalid, -torch.inf).argmax(1, keepdim=True)
                y = r + args.gamma * (1 - d) * q_values(target, o2).gather(1, a2).squeeze(1)
            q = q_values(net, o).gather(1, a[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                for p, tp in zip(net.parameters(), target.parameters()):
                    tp.lerp_(p, args.tau)
        if step % 50 == 0:
            loss_avg = 0.9 * loss_avg + 0.1 * loss.item()
            q_avg = 0.9 * q_avg + 0.1 * q.mean().item()

    run.finish(net, total * n)


if __name__ == "__main__":
    main()
