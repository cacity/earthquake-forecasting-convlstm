from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


@dataclass(frozen=True)
class ConvLSTMConfig:
    input_channels: int
    hidden_channels: int = 64
    num_layers: int = 1
    kernel_size: int = 3


class ConvLSTM(nn.Module):
    def __init__(self, cfg: ConvLSTMConfig):
        super().__init__()
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be >=1")
        cells = []
        in_ch = cfg.input_channels
        for _ in range(cfg.num_layers):
            cells.append(ConvLSTMCell(in_ch, cfg.hidden_channels, cfg.kernel_size))
            in_ch = cfg.hidden_channels
        self.cells = nn.ModuleList(cells)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected x [B,L,C,H,W]")
        b, l, c, h, w = x.shape
        device = x.device
        hs = [
            torch.zeros(b, self.cfg.hidden_channels, h, w, device=device, dtype=x.dtype)
            for _ in range(self.cfg.num_layers)
        ]
        cs = [
            torch.zeros(b, self.cfg.hidden_channels, h, w, device=device, dtype=x.dtype)
            for _ in range(self.cfg.num_layers)
        ]

        for t in range(l):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                hs[layer], cs[layer] = cell(inp, hs[layer], cs[layer])
                inp = hs[layer]
        return hs[-1]


class ConvLSTMForecaster(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64, num_layers: int = 1):
        super().__init__()
        self.backbone = ConvLSTM(
            ConvLSTMConfig(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
        )
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_last = self.backbone(x)
        return self.head(h_last)

