# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SmallCNNMel(nn.Module):
    """
    DP-friendly CNN for mel input.
    Input: [B, n_mels, T]
    """
    def __init__(self, n_mels: int, max_time: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, max_time)
            z = self.conv(dummy)
            flat = z.flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.conv(x)
        z = z.flatten(1)
        return self.head(z)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_model(cfg, kind: str, X_sample, num_classes: int):
    if kind == "cnn_mel":
        return SmallCNNMel(n_mels=cfg.n_mels, max_time=cfg.max_time, num_classes=num_classes)
    if kind == "linear_head":
        in_dim = int(X_sample.shape[1])
        return LinearHead(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model kind: {kind}")