"""Headless, closed-figure output; no algorithm depends on Matplotlib."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .pipeline import EdgeResult


def save_panels(original: np.ndarray, result: EdgeResult, path: str | Path) -> None:
    """Save the six requested stages with fixed image scales and labeled heatmaps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    panels = [("Original", original, "gray"), ("Grayscale [0,1]", result.gray, "gray"),
              ("Gaussian smoothed [0,1]", result.smoothed, "gray"),
              ("Gradient magnitude", result.magnitude, "magma"),
              ("Non-maximum suppression", result.suppressed, "magma"),
              ("Final edges", result.edges, "gray")]
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        if image.dtype == np.uint8:
            vmin, vmax = 0, 255
        elif cmap == "gray":
            vmin, vmax = 0, 1
        else:
            vmin, vmax = 0, max(float(result.magnitude.max()), 1e-12)
        artist = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        if cmap == "magma":
            fig.colorbar(artist, ax=ax, shrink=0.7)
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_edges(edges: np.ndarray, path: str | Path) -> None:
    """Lossless boolean edge map encoded as 0/255 PNG."""
    Image.fromarray(edges.astype(np.uint8)*255).save(path)


def save_threshold_comparison(rows: list[tuple[str, np.ndarray, np.ndarray, list[np.ndarray]]],
                              path: str | Path) -> None:
    """Compare two scenes against independent truth across three threshold pairs."""
    fig, axes = plt.subplots(len(rows), 5, figsize=(13, 3*len(rows)), squeeze=False)
    names = ["Original", "Ground truth", "Sensitive: .02/.05", "Balanced: .04/.10", "Strict: .08/.20"]
    for row_axes, (name, image, truth, detections) in zip(axes, rows):
        for ax, title, data in zip(row_axes, names, [image, truth, *detections]):
            ax.imshow(data, cmap="gray", interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        row_axes[0].set_ylabel(name.replace("_", " "))
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
