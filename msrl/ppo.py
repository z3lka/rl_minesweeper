"""PPO with action masking and GAE.

Same rewards and network body as DQN; the per-cell output is a policy logit
and a pooled head estimates the state value.

    uv run python -m msrl.ppo --board beginner
"""

from __future__ import annotations

import numpy as np
import torch

from .common import Run, base_parser, build_net
from .env import HIDDEN, MinesweeperEnv


def masked_log_probs(logits, obs):
    """Log-probabilities over valid cells and the policy entropy."""
    valid = (obs == HIDDEN).flatten(1)
    logp = torch.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=1)
    safe_logp = torch.where(valid, logp, torch.zeros_like(logp))
    entropy = -(safe_logp.exp() * safe_logp * valid).sum(1)
    return logp, entropy


def main():
    ap = base_parser("ppo", env_steps=20e6)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--progress-reward", type=float, default=1.0)
    args = ap.parse_args()

    run = Run("ppo", args)
    spec, dev, n, T = run.spec, run.device, args.envs, args.rollout
    net = build_net(args, dev, value_head=True)
    opt = torch.optim.Adam(net.parameters(), args.lr, eps=1e-5)
    env = MinesweeperEnv(spec, n, seed=args.seed, progress_reward=args.progress_reward)
    iters = int(args.env_steps) // (n * T)
    obs = torch.from_numpy(env.obs()).to(dev)

    b_obs = torch.zeros((T, n, spec.h, spec.w), dtype=torch.uint8, device=dev)
    b_act = torch.zeros((T, n), dtype=torch.long, device=dev)
    b_logp = torch.zeros((T, n), device=dev)
    b_val = torch.zeros((T, n), device=dev)
    b_rew = torch.zeros((T, n), device=dev)
    b_done = torch.zeros((T, n), device=dev)
    stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

    for it in range(iters):
        run.maybe_eval(net, it * n * T, **stats)
        opt.param_groups[0]["lr"] = args.lr * (1 - it / iters)

        with torch.no_grad():
            for t in range(T):
                logits, value = net(obs)
                logp, _ = masked_log_probs(logits, obs)
                action = torch.multinomial(logp.exp(), 1).squeeze(1)
                b_obs[t], b_act[t], b_val[t] = obs, action, value
                b_logp[t] = logp.gather(1, action[:, None]).squeeze(1)
                nxt, rew, done, _ = env.step(action.cpu().numpy())
                b_rew[t] = torch.from_numpy(rew).to(dev)
                b_done[t] = torch.from_numpy(done.astype(np.float32)).to(dev)
                obs = torch.from_numpy(nxt).to(dev)
            next_value = net(obs)[1]
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                alive = 1 - b_done[t]
                delta = b_rew[t] + args.gamma * next_value * alive - b_val[t]
                adv[t] = last = delta + args.gamma * args.lam * alive * last
                next_value = b_val[t]
            ret = adv + b_val

        f_obs, f_act, f_logp = b_obs.flatten(0, 1), b_act.flatten(), b_logp.flatten()
        f_adv, f_ret = adv.flatten(), ret.flatten()
        for _ in range(args.epochs):
            perm = torch.randperm(n * T, device=dev)
            for i in range(0, n * T, args.minibatch):
                mb = perm[i : i + args.minibatch]
                logits, value = net(f_obs[mb])
                logp, entropy = masked_log_probs(logits, f_obs[mb])
                ratio = (logp.gather(1, f_act[mb, None]).squeeze(1) - f_logp[mb]).exp()
                a = f_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                loss_pi = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                loss_v = (value - f_ret[mb]).square().mean()
                loss = loss_pi + args.value_coef * loss_v - args.entropy * entropy.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        stats = {"loss_pi": loss_pi.item(), "loss_v": loss_v.item(), "entropy": entropy.mean().item()}

    run.finish(net, iters * n * T)


if __name__ == "__main__":
    main()
