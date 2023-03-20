#!/usr/bin/env python3
import torch


def get_data_02():
    X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
    y = 0.7 * X + 0.3

    split = int(0.8 * len(X[:]))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, y_train, X_test, y_test
