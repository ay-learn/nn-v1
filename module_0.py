#!/usr/bin/env python3
import torch
import torch.nn as nn

from leanr_algebra import LeanrAlgebra
from plot import plot_predictions

a = 0.7
b = 0.6
X = torch.arange(0, 1, 0.01)
y = a * X + b
split = int(0.8 * len(X[:]))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[split:], y[:split]


module_0 = LeanrAlgebra()
# print(X, X_train)
# print(module_0)
with torch.inference_mode():
    y_pred = module_0(X_train)


plot_predictions(X_train, y_train, X_test, y_test, y_pred)
