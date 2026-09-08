"""Global-temperature CTKD (Li et al., AAAI 2023), adapted to speech KD.

Equations 8, 11, 12 and official global-T defaults:
https://github.com/zhengli97/CTKD/tree/56112892d5aca069bd56bb057feee9f7ff9e4141
Unlike the historical linear-temperature probe, this optimizes temperature
adversarially. The student optimizer/loss balance stay matched to this project.
"""
import math

import torch
from torch import nn


def curriculum_magnitude(epoch, ramp_epochs=10):
    return (1 - math.cos(math.pi * min(max(epoch, 0), ramp_epochs) / ramp_epochs)) / 2


class _ReverseTemperatureGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, magnitude):
        ctx.magnitude = magnitude
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.magnitude * gradient, None


class GlobalTemperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.ones(1))

    def forward(self, epoch):
        latent = _ReverseTemperatureGradient.apply(self.raw, curriculum_magnitude(epoch))
        return 1 + 20 * torch.sigmoid(latent)
