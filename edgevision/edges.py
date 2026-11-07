"""Interpolated thinning, threshold classification, and explicit graph traversal."""
from __future__ import annotations
from collections import deque
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .filters import as_2d, real_scalar


def _bilinear(a: NDArray[np.float64], y: NDArray, x: NDArray) -> NDArray[np.float64]:
    """Sample at in-bounds continuous coordinates with bilinear interpolation."""
    y = np.clip(y, 0, a.shape[0] - 1)
    x = np.clip(x, 0, a.shape[1] - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, a.shape[0] - 1), np.minimum(x0 + 1, a.shape[1] - 1)
    wy, wx = y - y0, x - x0
    return ((1-wy) * ((1-wx)*a[y0, x0] + wx*a[y0, x1])
            + wy * ((1-wx)*a[y1, x0] + wx*a[y1, x1]))


def non_maximum_suppression(magnitude: ArrayLike, orientation: ArrayLike) -> NDArray[np.float64]:
    """Keep local maxima along the continuous gradient direction; O(HW).

    Canonical angle modulo pi makes the tie rule invariant to contrast
    reversal. Retain m>=forward and m>backward, m>0. This picks one side of
    a two-pixel plateau instead of keeping both. Long flat ridges can still
    create gaps; this is a deliberate local convention, not subpixel fitting.
    The outer pixel ring is always zero; images smaller than 3x3 have no edges.
    """
    m, theta = as_2d(magnitude, "magnitude"), as_2d(orientation, "orientation")
    if m.shape != theta.shape or np.any(m < 0):
        raise ValueError("matching shapes and nonnegative magnitude required")
    out = np.zeros_like(m)
    if min(m.shape) < 3:
        return out
    angle = np.mod(theta[1:-1, 1:-1], np.pi)
    dy, dx = np.sin(angle), np.cos(angle)
    # Remove roundoff-sized direction components on cardinal axes.
    dx = np.where(np.abs(dx) < 1e-14, 0.0, dx)
    dy = np.where(np.abs(dy) < 1e-14, 0.0, dy)
    y, x = np.mgrid[1:m.shape[0]-1, 1:m.shape[1]-1]
    forward = _bilinear(m, y + dy, x + dx)
    backward = _bilinear(m, y - dy, x - dx)
    center = m[1:-1, 1:-1]
    keep = (center > 0) & (center >= forward) & (center > backward)
    out[1:-1, 1:-1] = np.where(keep, center, 0.0)
    return out


def double_threshold(magnitude: ArrayLike, low: float, high: float,
                     *, mode: str = "relative") -> NDArray[np.uint8]:
    """Labels 0=rejected, 1=weak, 2=strong; inclusive threshold endpoints.

    relative: 0<=low<=high<=1, fractions of this image's NMS maximum.
    absolute: 0<=low<=high in Sobel magnitude units. Only POSITIVE values
    are eligible, so thresholds of zero never turn a blank image into edges.
    """
    m = as_2d(magnitude, "magnitude")
    low, high = real_scalar(low, "low"), real_scalar(high, "high")
    if np.any(m < 0) or low < 0 or high < low:
        raise ValueError("require nonnegative magnitude and 0 <= low <= high")
    if mode not in ("relative", "absolute"):
        raise ValueError("mode must be relative or absolute")
    if mode == "relative":
        if high > 1:
            raise ValueError("relative thresholds cannot exceed 1")
        low, high = low * float(m.max()), high * float(m.max())
    labels = np.zeros(m.shape, dtype=np.uint8)
    eligible = m > 0
    labels[eligible & (m >= low)] = 1
    labels[eligible & (m >= high)] = 2
    return labels


def hysteresis(labels: ArrayLike, connectivity: int = 8) -> NDArray[np.bool_]:
    """BFS reachability from all strong pixels through weak/strong pixels.

    A pixel is marked before enqueueing, so it is visited at most once.
    O(HW) time/storage for fixed 4/8 connectivity; no recursive stack risk.
    Does not mutate labels, and never treats opposite borders as adjacent.
    """
    a = np.asarray(labels)
    if a.ndim != 2 or min(a.shape, default=0) < 1 or a.dtype.kind not in "bui":
        raise ValueError("labels must be a nonempty integer 2D array")
    if not np.isin(a, [0, 1, 2]).all():
        raise ValueError("labels must be 0, 1 or 2")
    if isinstance(connectivity, (bool, np.bool_)) or not isinstance(connectivity, (int, np.integer)) or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    edges = a == 2
    queue = deque(map(tuple, np.argwhere(edges)))
    h, w = a.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and a[ny, nx] != 0 and not edges[ny, nx]:
                edges[ny, nx] = True
                queue.append((ny, nx))
    return edges
