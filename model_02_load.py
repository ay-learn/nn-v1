#!/usr/bin/env python3
import torch

from data import get_data_02
from plot import *
from save import model_path

X_train, y_train, X_test, y_test = get_data_02()
PATH_MODEL = model_path("02_pytorch.pth")

from leanr_algebraV2 import LeanrAlgebraV2

model_eval = LeanrAlgebraV2()
model_eval.load_state_dict(torch.load(PATH_MODEL))
model_eval.eval()
with torch.inference_mode():
    y2 = model_eval(X_test)

plot_predictions(X_train, y_train, X_test, y_test, predictions=y2)
