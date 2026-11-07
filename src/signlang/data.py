"""Loading Sign Language MNIST and handling its labels.

Pixels are scaled by 1/255 only (decision D-5). Labels in the CSV are 0-25
with 9 (J) and 25 (Z) absent; they are remapped to 0-23 in letter order.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "sign_mnist_train.csv"
TEST_CSV = RAW_DIR / "sign_mnist_test.csv"

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# Raw CSV label (0-25, alphabet position) for each of the 24 classes.
RAW_LABELS = np.array([ord(c) - ord("A") for c in LETTERS])
_RAW_TO_NEW = {raw: new for new, raw in enumerate(RAW_LABELS)}


def load_csv(path):
    """Return (X, y_raw): X float32 (n, 784) in [0, 1], y_raw int labels 0-25."""
    df = pd.read_csv(path)
    y_raw = df["label"].to_numpy(dtype=np.int64)
    X = df.drop(columns="label").to_numpy(dtype=np.float32) / 255.0
    return X, y_raw


def remap_labels(y_raw):
    """Map the 24 present raw labels to 0-23 in letter order."""
    y_raw = np.asarray(y_raw)
    unknown = set(np.unique(y_raw)) - set(_RAW_TO_NEW)
    if unknown:
        raise ValueError(f"unexpected raw labels: {sorted(unknown)}")
    return np.array([_RAW_TO_NEW[v] for v in y_raw], dtype=np.int64)


def label_to_letter(y):
    """Letter (str) for a 0-23 label, or an array of letters for an array."""
    if np.ndim(y) == 0:
        return LETTERS[int(y)]
    return np.array([LETTERS[int(v)] for v in y])


def load_train():
    """Training set as (X, y) with y in 0-23."""
    X, y_raw = load_csv(TRAIN_CSV)
    return X, remap_labels(y_raw)


def load_test():
    """Official test set as (X, y) with y in 0-23."""
    X, y_raw = load_csv(TEST_CSV)
    return X, remap_labels(y_raw)


def to_images(X):
    """(n, 784) -> (n, 28, 28). Also accepts a single 784 vector."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(28, 28)
    return X.reshape(-1, 28, 28)
