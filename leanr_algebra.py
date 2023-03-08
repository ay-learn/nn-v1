import torch
import torch.nn as nn


class LeanrAlgebra(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b

