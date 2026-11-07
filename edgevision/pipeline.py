"""Configurable pipeline retaining inspectable intermediate arrays and timings."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .filters import (BOUNDARIES, positive_int, real_scalar, to_grayscale,
                      smooth_image, sobel_gradients, magnitude_orientation)
from .edges import non_maximum_suppression, double_threshold, hysteresis


T = TypeVar("T")


@dataclass(frozen=True)
class EdgeConfig:
    """Threshold units and boundary behavior are explicit, never auto-guessed."""
    sigma: float = 1.2
    kernel_size: int = 7
    low: float = 0.1
    high: float = 0.25
    threshold_mode: str = "relative"
    boundary: str = "reflect"
    connectivity: int = 8

    def __post_init__(self) -> None:
        positive_int(self.kernel_size, "kernel_size", odd=True)
        if real_scalar(self.sigma, "sigma") <= 0:
            raise ValueError("sigma must be positive")
        low, high = real_scalar(self.low, "low"), real_scalar(self.high, "high")
        if low < 0 or high < low or (self.threshold_mode == "relative" and high > 1):
            raise ValueError("invalid threshold interval")
        if self.threshold_mode not in ("relative", "absolute") or self.boundary not in BOUNDARIES:
            raise ValueError("unknown threshold mode or boundary")
        if isinstance(self.connectivity, (bool, np.bool_)) or not isinstance(self.connectivity, (int, np.integer)) or self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass
class EdgeResult:
    """Float64 intermediates; uint8 labels; boolean final edges. Times in seconds."""
    gray: NDArray
    smoothed: NDArray
    gx: NDArray
    gy: NDArray
    magnitude: NDArray
    orientation: NDArray
    suppressed: NDArray
    labels: NDArray
    edges: NDArray
    timings: dict[str, float] = field(default_factory=dict)


def detect_edges(image: ArrayLike, config: EdgeConfig | None = None) -> EdgeResult:
    """Execute each stage once. Caller input is never modified.

    Timing covers validation and allocation within each stage, but excludes
    file I/O, figure rendering, and external evaluation metrics.
    """
    cfg = EdgeConfig() if config is None else config
    if not isinstance(cfg, EdgeConfig):
        raise ValueError("config must be an EdgeConfig")
    timings: dict[str, float] = {}

    def stage(name: str, fn: Callable[[], T]) -> T:
        start = perf_counter()
        value = fn()
        timings[name] = perf_counter() - start
        return value

    gray = stage("grayscale", lambda: to_grayscale(image))
    smoothed = stage("gaussian", lambda: smooth_image(gray, cfg.kernel_size, cfg.sigma, cfg.boundary))
    gx, gy = stage("sobel", lambda: sobel_gradients(smoothed, cfg.boundary))
    mag, theta = stage("magnitude_orientation", lambda: magnitude_orientation(gx, gy))
    suppressed = stage("non_maximum_suppression", lambda: non_maximum_suppression(mag, theta))
    labels = stage("double_threshold", lambda: double_threshold(suppressed, cfg.low, cfg.high, mode=cfg.threshold_mode))
    edges = stage("hysteresis", lambda: hysteresis(labels, cfg.connectivity))
    return EdgeResult(gray, smoothed, gx, gy, mag, theta, suppressed, labels, edges, timings)
