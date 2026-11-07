"""Plotting helpers shared by all stages. Every function returns a Figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import LETTERS

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 24 distinguishable colours: tab20 followed by the first 4 of tab20b.
PALETTE = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors[:4])


def savefig(fig, path):
    """Save as PDF (tight bbox, no timestamp so reruns give identical files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    return path


def _to_img(x):
    x = np.asarray(x)
    return x.reshape(28, 28) if x.size == 784 else x


def image_grid(images, titles=None, ncols=10, cmap="gray", row_labels=None,
               panel_size=1.0, figsize=None, vmin=0.0, vmax=1.0):
    """Images in a grid of `ncols` columns, optional titles and row labels."""
    images = list(images)
    nrows = int(np.ceil(len(images) / ncols))
    if figsize is None:
        figsize = (panel_size * ncols, panel_size * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if k >= len(images):
            continue
        ax.imshow(_to_img(images[k]), cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            ax.set_title(titles[k], fontsize=8)
    if row_labels is not None:
        for r, lab in enumerate(row_labels):
            axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                                  fontsize=12, labelpad=8)
    fig.tight_layout()
    return fig


def weight_images(W, letters, mean_images=None, figsize=(10, 4.5)):
    """Weight vectors as 28x28 images, RdBu_r with symmetric limits per panel.

    If `mean_images` is given, a second row shows them in gray.
    """
    nrows = 1 if mean_images is None else 2
    fig, axes = plt.subplots(nrows, len(letters), figsize=figsize, squeeze=False)
    for c, (w, letter) in enumerate(zip(W, letters)):
        lim = float(np.max(np.abs(w)))
        im = axes[0, c].imshow(_to_img(w), cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[0, c].set_title(letter)
        fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.04)
        if mean_images is not None:
            axes[1, c].imshow(_to_img(mean_images[c]), cmap="gray", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def confusion_heatmap(cm, letters=LETTERS, normalize=False, ax=None, title=None):
    """Heatmap of a confusion matrix; `normalize=True` divides each row by its sum."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(letters)), letters, fontsize=7)
    ax.set_yticks(range(len(letters)), letters, fontsize=7)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def curve(x, ys, labels, xlabel="", ylabel="", logx=False, ax=None, marker="o"):
    """One line per entry of `ys`, all against `x`."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for y, lab in zip(ys, labels):
        ax.plot(x, y, marker=marker, label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if any(labels):
        ax.legend()
    return fig


def scatter_2d(Z, y, letters=LETTERS, ax=None, legend=True, s=4, alpha=0.6):
    """2D scatter coloured by letter, with one shared legend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    for c, letter in enumerate(letters):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=s, alpha=alpha, color=PALETTE[c],
                   label=letter, rasterized=True)
    if legend:
        ax.legend(markerscale=3, ncol=2, fontsize=7, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False)
    return fig
