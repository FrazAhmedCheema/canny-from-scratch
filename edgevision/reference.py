"""OPTIONAL SciPy validation only. Not imported by the core pipeline."""
from __future__ import annotations
import numpy as np
from .filters import convolve2d, gaussian_kernel, sobel_gradients
from .edges import hysteresis


def compare_reference(seed: int = 1729) -> dict:
    """Compare same numerical contracts, not claim Canny-map equivalence."""
    try:
        import scipy
        from scipy import ndimage
    except ImportError:
        return {"status": "unavailable", "reason": "SciPy not installed; optional reference only"}
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(25, 31))
    k = rng.normal(size=(5, 3))
    errors = {}
    for ours, theirs in [("reflect", "mirror"), ("edge", "nearest"), ("constant", "constant")]:
        actual = convolve2d(a, k, boundary=ours, cval=0.25)
        expected = ndimage.convolve(a, k, mode=theirs, cval=0.25)
        errors[ours] = float(np.max(np.abs(actual-expected)))
    # Same finite sampled kernel, NOT scipy.gaussian_filter's default truncation.
    g = gaussian_kernel(7, 1.2)
    errors["sampled_gaussian"] = float(np.max(np.abs(convolve2d(a, g)-ndimage.convolve(a, g, mode="mirror"))))
    gx, gy = sobel_gradients(a)
    errors["sobel_x"] = float(np.max(np.abs(gx-ndimage.sobel(a, axis=1, mode="mirror")/8)))
    errors["sobel_y"] = float(np.max(np.abs(gy-ndimage.sobel(a, axis=0, mode="mirror")/8)))
    labels = rng.choice([0, 1, 2], size=(25, 31), p=[0.6, 0.35, 0.05]).astype(np.uint8)
    bfs_equal = {}
    for conn in (4, 8):
        structure = ndimage.generate_binary_structure(2, 1 if conn == 4 else 2)
        expected = ndimage.binary_propagation(labels == 2, structure=structure, mask=labels != 0)
        bfs_equal[str(conn)] = bool(np.array_equal(hysteresis(labels, conn), expected))
    passed = max(errors.values()) < 1e-12 and all(bfs_equal.values())
    return {"status": "passed" if passed else "failed", "scipy_version": scipy.__version__,
            "max_absolute_errors": errors, "hysteresis_equal": bfs_equal,
            "scope": "Convolution, sampled Gaussian, Sobel, graph reachability; no full Canny equivalence claim"}
