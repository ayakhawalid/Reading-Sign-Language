"""Metrics shared by all stages."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import LETTERS

N_CLASSES = len(LETTERS)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=np.arange(N_CLASSES), zero_division=0))


def confusion(y_true, y_pred):
    """24x24 count matrix; rows are true letters, columns predicted letters."""
    return confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))


def top_confused_pairs(cm, n=5):
    """The n letter pairs with the most confusions in both directions.

    Returns a list of (letter_a, letter_b, count_ab, count_ba), where count_ab
    is the number of true-a items predicted as b, and a is the direction with
    the larger count. Sorted by count_ab + count_ba, descending.
    """
    cm = np.asarray(cm)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[1]):
            ab, ba = int(cm[i, j]), int(cm[j, i])
            if ab + ba == 0:
                continue
            if ba > ab:
                pairs.append((LETTERS[j], LETTERS[i], ba, ab))
            else:
                pairs.append((LETTERS[i], LETTERS[j], ab, ba))
    pairs.sort(key=lambda p: (-(p[2] + p[3]), p[0], p[1]))
    return pairs[:n]


def summary(y_true, y_pred):
    return {"accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "n": int(len(y_true))}
