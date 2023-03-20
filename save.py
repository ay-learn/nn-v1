# save model using Path lib
from pathlib import Path


def model_path(name):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    return MODEL_SAVE_PATH
