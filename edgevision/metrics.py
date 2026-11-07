"""Exact and tolerance-aware, one-to-one edge-pixel evaluation; NumPy + stdlib."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class EdgeScores:
    """TP is a maximum-cardinality one-to-one matching, not dilated overlap."""
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def _mask(a: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.bool_ or a.ndim != 2 or min(a.shape, default=0) < 1:
        raise ValueError(f"{name} must be a nonempty 2D boolean map")
    return a


def edge_scores(predicted: ArrayLike, truth: ArrayLike, *, tolerance: int = 0,
                valid: ArrayLike | None = None) -> EdgeScores:
    """Match edge pixels within a Euclidean radius in integer pixels.

    Exact mode is O(HW). For radius r>0, build sparse candidate edges in
    O(P*r²), then BFS augmenting paths to a MAXIMUM matching, O(P*E) worst
    case, O(HW+P+G+E) space. P/G are predicted/truth counts, E candidates.
    A thick predicted edge cannot earn multiple TPs for one truth pixel.
    Radius is limited to 5 to prevent accidental dense matching. If both maps
    are empty all scores are 1; if only one is empty all scores are 0.
    """
    p, g = _mask(predicted, "predicted"), _mask(truth, "truth")
    if p.shape != g.shape:
        raise ValueError("edge map shapes must match")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, (int, np.integer)) or not 0 <= tolerance <= 5:
        raise ValueError("tolerance must be an integer in [0,5]")
    if valid is not None:
        v = _mask(valid, "valid")
        if v.shape != p.shape:
            raise ValueError("valid mask shape must match")
        p, g = p & v, g & v
    pc, gc = int(p.sum()), int(g.sum())
    if tolerance == 0:
        tp = int((p & g).sum())
    elif not pc or not gc:
        tp = 0
    else:
        gt_index = np.full(g.shape, -1, dtype=np.int64)
        gt_index[g] = np.arange(gc)
        offsets = [(dy, dx) for dy in range(-tolerance, tolerance+1)
                   for dx in range(-tolerance, tolerance+1) if dy*dy+dx*dx <= tolerance*tolerance]
        offsets.sort(key=lambda d: (d[0]*d[0]+d[1]*d[1], d))
        adjacency: list[list[int]] = []
        h, w = p.shape
        for y, x in np.argwhere(p):
            adjacency.append([int(gt_index[y+dy, x+dx]) for dy, dx in offsets
                              if 0 <= y+dy < h and 0 <= x+dx < w and gt_index[y+dy, x+dx] >= 0])
        match_p, match_g = [-1]*pc, [-1]*gc
        tp = 0
        for start in range(pc):
            q, visited, parent = deque([start]), {start}, {}
            found = -1
            while q and found == -1:
                u = q.popleft()
                for v in adjacency[u]:
                    if v in parent:
                        continue
                    parent[v] = u
                    if match_g[v] == -1:
                        found = v
                        break
                    following = match_g[v]
                    if following not in visited:
                        visited.add(following)
                        q.append(following)
            if found != -1:
                tp += 1
                while found != -1:
                    u = parent[found]
                    previous = match_p[u]
                    match_p[u], match_g[found] = found, u
                    found = previous
    fp, fn = pc-tp, gc-tp
    if pc == gc == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp/pc if pc else 0.0
        recall = tp/gc if gc else 0.0
        f1 = 2*tp/(pc+gc)
    return EdgeScores(tp, fp, fn, precision, recall, f1)
