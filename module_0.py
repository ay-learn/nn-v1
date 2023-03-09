#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

from leanr_algebra import LeanrAlgebra
from plot import plot_predictions

# import torch.nn as nn


X = torch.arange(0, 1, 0.02)
y = 0.7 * X + 0.3

split = int(0.8 * len(X[:]))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

torch.manual_seed(42)
module_0 = LeanrAlgebra()
# with torch.inference_mode():
#     y_pred = module_0(X_test)
# print(X, y_train)
# # print(module_0)


loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=module_0.parameters(), lr=0.01)

epochs = int(1000_000 * 0.01)
for epoch in range(epochs):
    torch.manual_seed(42)
    module_0.train()
    optimizer.zero_grad()
    loss_fn(module_0(X_train), y_train).backward()
    optimizer.step()

torch.manual_seed(42)
with torch.inference_mode():
    y_pred_test = module_0(X_test)


plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred_test)
os._exit(0)
