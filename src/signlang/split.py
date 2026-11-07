"""Near-duplicate grouping and the one train/validation split of the project.

The training CSV holds a little over a thousand photos, each expanded by
Kaggle into about 20 altered copies (brightness/contrast, small rotation,
resampling filters, pixelation). A random split would put copies of the
same photo on both sides. We recover the groups of copies from the pixels
and split by group (decision D-4; method as in TEXTBOOK deviation C-2).

Distance between two images: 1 - Pearson correlation of their pixels,
i.e. cosine distance after per-image standardisation. It ignores
brightness and contrast changes, which are part of Kaggle's augmentation.
This standardisation is used only to find groups; models still see x/255.
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from .data import LETTERS, ROOT

SPLIT_PATH = ROOT / "data" / "splits" / "split_v1.npz"
THRESHOLD_GRID = np.logspace(-3, -1, 21)   # 0.001 ... 0.1, 10 steps per decade


def standardise_rows(X):
    """Each image to zero mean and unit standard deviation (float64)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    return X / np.clip(X.std(axis=1, keepdims=True), 1e-8, None)


def neighbour_edges(X, y, n_neighbors=30):
    """Same-label k-nearest-neighbour edges under correlation distance.

    Returns (rows, cols, dist), one entry per (image, neighbour) pair. The
    image itself is excluded from its own neighbour list.
    """
    F = standardise_rows(X)
    rows, cols, dist = [], [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(n_neighbors, len(idx) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        d, j = nn.fit(F[idx]).kneighbors()
        rows.append(np.repeat(idx, k))
        cols.append(idx[j].ravel())
        dist.append(np.clip(d.ravel(), 0.0, None))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(dist)


def components(rows, cols, dist, threshold, n):
    """Connected components of the graph of edges with dist < threshold."""
    keep = dist < threshold
    graph = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def duplicate_groups(X, y, n_neighbors=30, threshold=None, grid=THRESHOLD_GRID):
    """Recover groups of altered copies of the same photo.

    If `threshold` is None it is chosen where the number of groups is least
    sensitive to the threshold: over a log-spaced grid, the step with the
    smallest relative change in group count; the threshold is the geometric
    midpoint of that step.

    Returns a dict with keys group_id, distances (all edge distances),
    threshold, grid, n_groups_grid.
    """
    n = len(y)
    rows, cols, dist = neighbour_edges(X, y, n_neighbors)
    n_groups = np.array([components(rows, cols, dist, t, n).max() + 1 for t in grid])
    if threshold is None:
        rel_change = np.abs(np.diff(np.log(n_groups)))
        i = int(np.argmin(rel_change))
        threshold = float(np.sqrt(grid[i] * grid[i + 1]))
    group_id = components(rows, cols, dist, threshold, n)
    return {"group_id": group_id, "distances": dist, "threshold": float(threshold),
            "grid": np.asarray(grid), "n_groups_grid": n_groups}


def group_stats(group_id):
    sizes = np.bincount(group_id)
    return {
        "n_groups": int(len(sizes)),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_mean": float(sizes.mean()),
        "size_max": int(sizes.max()),
        "size_quantiles": {str(q): float(np.quantile(sizes, q))
                           for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "frac_images_in_singletons": float(np.mean(sizes[group_id] == 1)),
    }


def make_split(group_id, y, val_frac=0.2, random_state=42, min_val_per_class=100,
               max_tries=10):
    """Group-aware 80/20 split. Retries with seed+1 if a letter has too few
    validation images. Returns (train_idx, val_idx, seed_used)."""
    for seed in range(random_state, random_state + max_tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(gss.split(np.zeros(len(y)), y, groups=group_id))
        if np.bincount(y[val_idx], minlength=len(LETTERS)).min() >= min_val_per_class:
            return np.sort(train_idx), np.sort(val_idx), seed
    raise RuntimeError(f"no seed in {max_tries} tries gave >= {min_val_per_class} "
                       "validation images per letter")


def make_random_split(y, val_frac=0.2, random_state=42):
    """Plain stratified random 80/20 split (Stage 1 diagnostic only)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
    return np.sort(train_idx), np.sort(val_idx)


def nearest_train_distance(X, y, train_idx, val_idx):
    """For each validation image, correlation distance to the closest
    training image of the same letter (a leakage check)."""
    F = standardise_rows(X)
    out = np.empty(len(val_idx))
    for c in np.unique(y):
        tr = train_idx[y[train_idx] == c]
        m = y[val_idx] == c
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(F[tr])
        out[m] = np.clip(nn.kneighbors(F[val_idx[m]])[0][:, 0], 0.0, None)
    return out


def class_counts(y, idx):
    counts = np.bincount(y[idx], minlength=len(LETTERS))
    return {letter: int(k) for letter, k in zip(LETTERS, counts)}


def save_split(path=SPLIT_PATH, stats=None, **arrays):
    """Save index arrays to `path` (.npz) and `stats` to a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if stats is not None:
        path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_split(path=SPLIT_PATH):
    """Dict with train_idx, val_idx, group_id, threshold, rand_train_idx, rand_val_idx."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
