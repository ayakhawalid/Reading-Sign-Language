"""Evaluation-only perturbations: integer shifts (D-6) and Gaussian noise (D-7).

Never use these to build training data.
"""

import numpy as np


def _as_images(X):
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28), True
    if X.ndim == 3 and X.shape[1:] == (28, 28):
        return X, False
    raise ValueError(f"expected (n, 784) or (n, 28, 28), got {X.shape}")


def shift(images, dx, dy):
    """Shift by integer pixels; dx > 0 moves right, dy > 0 moves down.

    The image is padded by replicating its edge pixels and then cropped back
    to 28x28, so no black bar is introduced. Returns the input's shape.
    """
    imgs, flat = _as_images(images)
    dx, dy = int(dx), int(dy)
    p = max(abs(dx), abs(dy))
    if p == 0:
        out = imgs.copy()
    else:
        padded = np.pad(imgs, ((0, 0), (p, p), (p, p)), mode="edge")
        r0, c0 = p - dy, p - dx
        out = padded[:, r0:r0 + 28, c0:c0 + 28]
    return out.reshape(-1, 784) if flat else out


def add_gaussian_noise(X, sigma, rng):
    """Add N(0, sigma^2) noise to [0, 1]-scaled images and clip to [0, 1]."""
    X = np.asarray(X)
    noisy = X + rng.normal(0.0, sigma, size=X.shape)
    return np.clip(noisy, 0.0, 1.0).astype(X.dtype)
