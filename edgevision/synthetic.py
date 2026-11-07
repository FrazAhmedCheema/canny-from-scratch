"""Small deterministic scenes; truth comes from analytic masks, never detection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image
from pathlib import Path
from .filters import positive_int


@dataclass(frozen=True)
class Scene:
    name: str
    image: np.ndarray
    truth: np.ndarray
    description: str


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """4-neighbor inner foreground perimeter, with outside-image background."""
    a = np.pad(mask, 1, constant_values=False)
    interior = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
    return mask & ~interior


def make_scenes(size: int = 128, seed: int = 1729) -> list[Scene]:
    """Return RGB/gray uint8 scenes. Noise is seeded; truth describes clean shapes.

    Truth labels are pixel-center conventions, not unique continuous contours.
    Use exact and radius-1 scores together; neither is an official dataset score.
    """
    size = positive_int(size, "size")
    if size < 32:
        raise ValueError("scene size must be at least 32")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    rect = (x >= size//4) & (x < 3*size//4) & (y >= size//4) & (y < 3*size//4)
    disk = (x-size/2)**2 + (y-size/2)**2 <= (size/4)**2
    diagonal = x >= y
    step = x >= size//2
    step_truth = np.zeros((size, size), bool)
    step_truth[:, size//2-1] = True  # Dark side of the interface for this scene.
    clean = (rect * 0.8).astype(float)
    noisy = np.clip(clean + rng.normal(0, 0.12, clean.shape), 0, 1)
    # Two rectangles of different contrast expose absolute-threshold tradeoffs.
    left = (x >= size//8) & (x < 3*size//8) & (y >= size//4) & (y < 3*size//4)
    right = (x >= 5*size//8) & (x < 7*size//8) & (y >= size//4) & (y < 3*size//4)
    mixed = np.clip(0.85*left + 0.25*right + rng.normal(0, 0.035, x.shape), 0, 1)
    def u8(a):
        return np.rint(a*255).astype(np.uint8)
    rgb = np.stack((0.9*rect, 0.5*rect, 0.2*rect), axis=-1)
    return [
        Scene("vertical_step", u8(step), step_truth, "Single vertical intensity step; truth on dark side."),
        Scene("rectangle_rgb", u8(rgb), inner_boundary(rect), "RGB rectangle; truth is its inner perimeter."),
        Scene("disk", u8(0.85*disk), inner_boundary(disk), "Rasterized disk; inner 4-neighbor perimeter."),
        Scene("diagonal", u8(0.8*diagonal), inner_boundary(diagonal), "45-degree half-plane; border excluded in evaluation."),
        Scene("noisy_rectangle", u8(noisy), inner_boundary(rect), "Gaussian noise sigma=0.12 added to rectangle; clean-mask truth."),
        Scene("mixed_contrast", u8(mixed), inner_boundary(left | right), "High/low contrast shapes with Gaussian noise sigma=0.035."),
    ]


def save_scenes(scenes: list[Scene], directory: str | Path) -> None:
    """Write compact lossless image and ground-truth PNG pairs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        Image.fromarray(scene.image).save(directory / f"{scene.name}.png")
        Image.fromarray(scene.truth.astype(np.uint8)*255).save(directory / f"{scene.name}_truth.png")
