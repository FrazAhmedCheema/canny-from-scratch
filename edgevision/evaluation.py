"""Deterministic evaluation and genuine repeated wall-clock measurements."""
from __future__ import annotations
import csv
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import numpy as np
from .pipeline import EdgeConfig, detect_edges
from .synthetic import make_scenes
from .metrics import edge_scores
from .visualization import save_panels, save_edges, save_threshold_comparison
from .reference import compare_reference
from .filters import positive_int


def evaluate(output: str | Path, *, size: int = 128, seed: int = 1729,
             repeats: int = 5, reference: bool = False) -> dict:
    """Write metrics, timings, panels, and an environment manifest.

    Threshold sweeps reuse the same seeded scenes. Accuracy scores exclude
    kernel_radius+2 pixels on every border, for both prediction and truth.
    Benchmarks use one unmeasured warm-up and medians of repeated full calls.
    They are descriptive measurements, not throughput guarantees.
    """
    repeats = positive_int(repeats, "repeats")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    scenes = make_scenes(size, seed)
    settings = {"sensitive": EdgeConfig(low=0.02, high=0.05, threshold_mode="absolute"),
                "balanced": EdgeConfig(low=0.04, high=0.10, threshold_mode="absolute"),
                "strict": EdgeConfig(low=0.08, high=0.20, threshold_mode="absolute")}
    rows, benches = [], []
    comparison_rows = []
    for scene in scenes:
        scene_detections = []
        for setting, config in settings.items():
            result = detect_edges(scene.image, config)
            scene_detections.append(result.edges)
            valid = np.zeros(result.edges.shape, bool)
            margin = config.kernel_size//2 + 2
            valid[margin:-margin, margin:-margin] = True
            for tolerance in (0, 1):
                score = edge_scores(result.edges, scene.truth, tolerance=tolerance, valid=valid)
                rows.append({"scene": scene.name, "setting": setting, "low": config.low,
                             "high": config.high, "mode": config.threshold_mode,
                             "tolerance_px": tolerance, **score.to_dict()})
            if setting == "balanced":
                save_panels(scene.image, result, out / f"{scene.name}_stages.png")
                save_edges(result.edges, out / f"{scene.name}_edges.png")
                times, totals = [], []
                # The already computed result is the warm-up; it is not timed here.
                for _ in range(repeats):
                    start = perf_counter()
                    measured = detect_edges(scene.image, config)
                    totals.append(perf_counter()-start)
                    times.append(measured.timings)
                    assert np.array_equal(measured.edges, result.edges)
                benches.append({"scene": scene.name, "shape": list(scene.image.shape),
                                "repeats": repeats, "total_median_ms": 1000*float(np.median(totals)),
                                "total_min_ms": 1000*min(totals), "total_max_ms": 1000*max(totals),
                                "stage_median_ms": {k: 1000*float(np.median([t[k] for t in times])) for k in times[0]}})
        if scene.name in ("noisy_rectangle", "mixed_contrast"):
            comparison_rows.append((scene.name, scene.image, scene.truth, scene_detections))
    save_threshold_comparison(comparison_rows, out / "threshold_comparison.png")
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"seed": seed, "size": size, "settings": {k: asdict(v) for k, v in settings.items()},
                "evaluation_border": 5, "matching": "exact and radius-1 one-to-one maximum-cardinality",
                "versions": {p: version(p) for p in ["numpy", "matplotlib", "Pillow"]},
                "python": platform.python_version(), "platform": platform.system(),
                "machine": platform.machine(), "processor": platform.processor() or "not exposed",
                "benchmark": benches, "reference": compare_reference(seed) if reference else {"status": "not requested"}}
    (out / "report.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Useful summary without pretending macro averages are pooled pixel metrics.
    summary = ["# Measured evaluation results", "", "AI-generated study implementation; synthetic benchmarks only.", "",
               "| Setting | Macro precision (r=1) | Macro recall (r=1) | Macro F1 (r=1) |", "|---|---:|---:|---:|"]
    for setting in settings:
        selected = [r for r in rows if r["setting"] == setting and r["tolerance_px"] == 1]
        values = [np.mean([r[k] for r in selected]) for k in ["precision", "recall", "f1"]]
        summary.append(f"| {setting} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    summary += ["", f"Seed {seed}; {size}×{size}; {repeats} timed repetitions after one warm-up.",
                "Macro values average per-scene scores, not pixel-pooled counts. Exact scores are in metrics.csv.", "",
                "| Scene | Pipeline median (ms) | Min (ms) | Max (ms) |", "|---|---:|---:|---:|"]
    for b in benches:
        summary.append(f"| {b['scene']} | {b['total_median_ms']:.3f} | {b['total_min_ms']:.3f} | {b['total_max_ms']:.3f} |")
    summary += ["", "Stage medians, versions, boundary/threshold settings and optional reference results: report.json.",
                "Timings exclude I/O, matching, and rendering; sums of stage medians need not equal median total.",
                "These small synthetic scenes are not evidence of real-world generalization or competitive performance."]
    (out / "SUMMARY.md").write_text("\n".join(summary)+"\n")
    return manifest
