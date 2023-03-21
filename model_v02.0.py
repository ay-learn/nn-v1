#!/usr/bin/env python3
import torch

from leanr_algebraV2 import loss_fn
from leanr_algebraV2 import model
from leanr_algebraV2 import optimizer
from plot import *
from save import model_path

# from plot import plot_model_loss
# from plot import plot_predictions

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = 0.7 * X + 0.3

split = int(0.8 * len(X[:]))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


loss_train = []
loss_test = []
epoch_count = []

# epochs = int(1000_000 * 0.01)
epochs = 100

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    y_pred = model(X_train)
    loss_value = loss_fn(y_pred, y_train)
    loss_value.backward()

    optimizer.step()
    # Test
    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test)

        loss_test_value = loss_fn(y_pred_test, y_test)
    # Diff
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_train.append(loss_value.detach().numpy())
        loss_test.append(loss_test_value.detach().numpy())


plot_model_v01_loss(epoch_count, loss_train, loss_test)


# Save Model
PATH_MODEL = model_path("v02.0_pytorch.pth")
torch.save(model.state_dict(), PATH_MODEL)
