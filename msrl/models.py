"""Shared network: a fully convolutional ResNet over the board.

Input is the uint8 observation (N, H, W); the one-hot encoding happens inside
the model so replay buffers and the browser only ever handle raw boards.
Output is one number per cell, plus an optional scalar from a pooled head
(state value for PPO, V(s) for dueling DQN). No layer depends on board size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import N_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class BoardNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6, value_head: bool = False):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks, "value_head": value_head}
        self.stem = nn.Conv2d(N_PLANES, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.cell_head = nn.Conv2d(channels, 1, 1)
        self.value_head = (
            nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU(), nn.Linear(channels, 1))
            if value_head
            else None
        )

    def forward(self, obs: torch.Tensor):
        """obs: (N, H, W) integer board. Returns (cells (N, H*W), value (N,) or None)."""
        x = F.one_hot(obs.long(), N_PLANES).permute(0, 3, 1, 2).float()
        x = self.blocks(F.relu(self.stem(x)))
        cells = self.cell_head(x).flatten(1)
        value = None
        if self.value_head is not None:
            pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
            value = self.value_head(pooled).squeeze(1)
        return cells, value


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
