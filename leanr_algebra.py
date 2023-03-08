#!/usr/bin/env python3
import torch
import torch.nn as nn

from plot import plot_predictions


class LeanrAlgebra(nn.Module):
    def __init__(self):
        super().__init
        delf.a = nn.Parameter(torch.randn(1, dtype=tensor.float), requires_grad=True)
        delf.b = nn.Parameter(torch.randn(1, dtype=tensor.float), requires_grad=True)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b
