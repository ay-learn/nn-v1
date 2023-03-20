import torch
import torch.nn as nn


class LeanrAlgebraV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.reshape(-1, 1)
        return self.linear_layer(x)


torch.manual_seed(42)
model = LeanrAlgebraV2()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# print(model.state_dict())
