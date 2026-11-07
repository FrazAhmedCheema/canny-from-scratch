"""Validated image conversion and true convolution; NumPy only."""
from __future__ import annotations
from numbers import Integral, Real
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatImage = NDArray[np.float64]
BOUNDARIES = ("reflect", "edge", "constant")


def real_scalar(value: object, name: str) -> float:
    """Reject booleans, non-real scalars, and non-finite parameters."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def positive_int(value: object, name: str, *, odd: bool = False) -> int:
    """Validate an integer without silently truncating fractional values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1 or (odd and value % 2 == 0):
        raise ValueError(f"{name} must be positive" + (" and odd" if odd else ""))
    return value


def as_2d(values: ArrayLike, name: str = "image") -> FloatImage:
    """Return a finite, nonempty, real float64 matrix (possibly a view)."""
    a = np.asarray(values)
    if a.ndim != 2 or min(a.shape, default=0) < 1:
        raise ValueError(f"{name} must be a nonempty 2D array")
    if a.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="raise", invalid="raise"):
        a = a.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite values")
    return a


def to_grayscale(image: ArrayLike) -> FloatImage:
    """Convert HxW or HxWx3 to [0,1] float64 Rec.709 luma.

    uint8/uint16 are scaled by the dtype maximum; bool is 0/1. Floating
    input must already be in [0,1]. No gamma decoding or alpha compositing.
    """
    a = np.asarray(image)
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
        raise ValueError("image must have shape (H,W) or (H,W,3)")
    if min(a.shape[:2], default=0) < 1:
        raise ValueError("image cannot be empty")
    if a.dtype == np.uint8 or a.dtype == np.uint16:
        a = a.astype(np.float64) / np.iinfo(a.dtype).max
    elif a.dtype.kind in "bf":
        a = a.astype(np.float64)
    else:
        raise ValueError("use uint8, uint16, bool, or floating [0,1] images")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("floating image values must be finite and in [0,1]")
    return a.copy() if a.ndim == 2 else a @ np.array([0.2126, 0.7152, 0.0722])


def convolve2d(image: ArrayLike, kernel: ArrayLike, *, boundary: str = "reflect",
               cval: float = 0.0) -> FloatImage:
    """True same-shape convolution with an odd rectangular kernel.

    The kernel is flipped on both axes (unlike correlation). 'reflect' uses
    NumPy whole-sample reflection, excluding the boundary pixel; singleton
    axes repeat their only value. 'edge' replicates endpoints; 'constant'
    extends by cval. Sliding windows are a view, not an H*W*K copy.
    Time O(H*W*Kh*Kw); storage O((H+Kh)*(W+Kw)+H*W+Kh*Kw),
    excluding the logical window view. For fixed small kernels this is O(HW).
    """
    a, k = as_2d(image), as_2d(kernel, "kernel")
    if any(d % 2 == 0 for d in k.shape):
        raise ValueError("kernel dimensions must be odd")
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}")
    cval = real_scalar(cval, "cval")
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    kwargs = {"constant_values": cval} if boundary == "constant" else {}
    padded = np.pad(a, ((ph, ph), (pw, pw)), mode=boundary, **kwargs)
    windows = np.lib.stride_tricks.sliding_window_view(padded, k.shape)
    try:
        with np.errstate(over="raise", invalid="raise"):
            out = np.einsum("ijkl,kl->ij", windows, k[::-1, ::-1], optimize=False)
    except FloatingPointError as exc:
        raise ValueError("convolution overflow; rescale the input/kernel") from exc
    if not np.isfinite(out).all():
        raise ValueError("convolution produced non-finite values")
    return out


def gaussian_kernel(size: int = 5, sigma: float = 1.2) -> FloatImage:
    """Normalized sampled isotropic Gaussian, centered on the middle pixel."""
    size = positive_int(size, "size", odd=True)
    sigma = real_scalar(sigma, "sigma")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = np.arange(size, dtype=np.float64) - size // 2
    # Divide before squaring; an underflowed tail legitimately becomes zero.
    with np.errstate(over="ignore", under="ignore"):
        g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()  # Center is exactly one before normalization, even at tiny sigma.
    return np.outer(g, g)


def smooth_image(image: ArrayLike, size: int = 5, sigma: float = 1.2,
                 boundary: str = "reflect") -> FloatImage:
    """Gaussian smoothing via the explicit 2D convolution implementation."""
    return convolve2d(image, gaussian_kernel(size, sigma), boundary=boundary)


def sobel_gradients(image: ArrayLike, boundary: str = "reflect") -> tuple[FloatImage, FloatImage]:
    """Normalized Sobel derivatives: x points right, y points down.

    Kernels are written for TRUE convolution. Division by 8 gives unit
    response to an interior unit ramp. Positive downward intensity gives gy>0.
    """
    kx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]) / 8.0
    return (convolve2d(image, kx, boundary=boundary),
            convolve2d(image, kx.T, boundary=boundary))


def magnitude_orientation(gx: ArrayLike, gy: ArrayLike) -> tuple[FloatImage, FloatImage]:
    """Euclidean magnitude and atan2(gy,gx) radians in [-pi,pi].

    Orientation is set to zero at zero magnitude; hypot avoids avoidable
    overflow of gx**2+gy**2. A truly unrepresentable result is rejected.
    """
    x, y = as_2d(gx, "gx"), as_2d(gy, "gy")
    if x.shape != y.shape:
        raise ValueError("gradient shapes must match")
    with np.errstate(over="ignore"):
        mag = np.hypot(x, y)
    if not np.isfinite(mag).all():
        raise ValueError("gradient magnitude overflow")
    theta = np.arctan2(y, x)
    return mag, np.where(mag > 0, theta, 0.0)
