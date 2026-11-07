"""One seed for the whole project."""

import os
import random

import numpy as np

SEED = 42


def set_seed(seed=SEED):
    """Seed `random`, `numpy` and, if installed, `torch`. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    return seed
