#!/usr/bin/env python3
import os

import torch
import torch.nn as nn

from leanr_algebra import LeanrAlgebra

X = torch.arange(0, 1, 0.02)
y = 0.7 * X + 0.3

split = int(0.8 * len(X[:]))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

torch.manual_seed(42)
module_0 = LeanrAlgebra()


# Training + Testing
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=module_0.parameters(), lr=0.01)

epochs = 10_000
for epoch in range(epochs):
    # Training
    torch.manual_seed(42)
    module_0.train()
    optimizer.zero_grad()
    loss_fn(module_0(X_train), y_train).backward()
    optimizer.step()
    # Testing

torch.manual_seed(42)
with torch.inference_mode():
    y_pred_test = module_0(X_test)


# plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred_test)
print(module_0.state_dict())
os._exit(0)
