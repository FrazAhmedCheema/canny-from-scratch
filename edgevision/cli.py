"""Small CLI: synthetic samples, one-image detection, and evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .pipeline import EdgeConfig, detect_edges
from .synthetic import make_scenes, save_scenes
from .visualization import save_panels, save_edges
from .evaluation import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NumPy Canny-style edge detection")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = sub.add_parser("samples", help="Generate deterministic small image/ground-truth PNGs")
    samples.add_argument("--output", type=Path, default=Path("samples"))
    samples.add_argument("--size", type=int, default=128)
    samples.add_argument("--seed", type=int, default=1729)
    demo = sub.add_parser("detect", help="Detect edges in one image")
    demo.add_argument("image", type=Path)
    demo.add_argument("--output", type=Path, default=Path("demo-output"))
    demo.add_argument("--sigma", type=float, default=1.2)
    demo.add_argument("--kernel-size", type=int, default=7)
    demo.add_argument("--low", type=float, default=0.1)
    demo.add_argument("--high", type=float, default=0.25)
    demo.add_argument("--threshold-mode", choices=["absolute", "relative"], default="relative")
    demo.add_argument("--boundary", choices=["reflect", "edge", "constant"], default="reflect")
    ev = sub.add_parser("evaluate", help="Threshold sweep, timings, optional trusted reference")
    ev.add_argument("--output", type=Path, default=Path("results"))
    ev.add_argument("--size", type=int, default=128)
    ev.add_argument("--seed", type=int, default=1729)
    ev.add_argument("--repeats", type=int, default=5)
    ev.add_argument("--reference", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "samples":
            save_scenes(make_scenes(args.size, args.seed), args.output)
        elif args.command == "detect":
            with Image.open(args.image) as im:
                image = np.asarray(im.convert("RGB"))
            cfg = EdgeConfig(sigma=args.sigma, kernel_size=args.kernel_size, low=args.low,
                             high=args.high, threshold_mode=args.threshold_mode, boundary=args.boundary)
            result = detect_edges(image, cfg)
            args.output.mkdir(parents=True, exist_ok=True)
            save_panels(image, result, args.output / "stages.png")
            save_edges(result.edges, args.output / "edges.png")
            (args.output / "timings.json").write_text(json.dumps(result.timings, indent=2)+"\n")
        else:
            evaluate(args.output, size=args.size, seed=args.seed, repeats=args.repeats, reference=args.reference)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved {args.command} outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
