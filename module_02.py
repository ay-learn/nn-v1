#!/usr/bin/env python3
import os

import torch

from leanr_algebra import loss_fn
from leanr_algebra import module_0
from leanr_algebra import optimizer

X = torch.arange(0, 1, 0.02)
y = 0.7 * X + 0.3

split = int(0.8 * len(X[:]))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


loss_train = []
loss_test = []
epoc_count = []

# epochs = int(1000_000 * 0.01)
epochs = 100

for epoch in range(epochs):
    module_0.train()
    optimizer.zero_grad()
    loss = loss_fn(module_0(X_train), y_train).backward()
    optimizer.step()
    # Test
    with torch.inference_mode():
        y_pred_test = module_0(X_test)
    if epoch % 10 == 10:
        loss_train.append(loss.detach().numpy())

torch.manual_seed(42)
with torch.inference_mode():
    y_pred_test = module_0(X_test)


# plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred_test)
print(module_0.state_dict())
os._exit(0)
