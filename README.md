# Computer Vision from Scratch — Canny-Style Edge Detection and Algorithm Evaluation

An inspectable NumPy implementation of Gaussian smoothing, Sobel gradients, interpolated non-maximum suppression, double thresholding, and BFS edge tracking. It includes synthetic ground truth, one-to-one edge matching, threshold sweeps, stage timings, and automated tests.

**Authorship:** this is an AI-generated implementation for review and study. It is not represented as independently authored by the applicant, and it is not an official NHL Stenden assignment solution or an admissions endorsement.

## Install

Python 3.12 or later (verified with Python 3.12.14). From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows activation: `.venv\Scripts\activate`. `requirements.txt` pins the four directly used runtime/test libraries to the versions used for the delivered results. SciPy is **not** a core dependency. To reproduce the optional trusted-reference checks:

```bash
python -m pip install scipy==1.17.0
```

## Test and run

```bash
python -m pytest -q
python -m edgevision --help
python -m edgevision samples --output samples --size 128 --seed 1729
python -m edgevision detect samples/noisy_rectangle.png --output demo-output
python -m edgevision evaluate --output results --size 128 --seed 1729 --repeats 5 --reference
```

`edgevision` is also installed as a console command. Running without SciPy skips one optional reference test; `--reference` records that the reference is unavailable rather than substituting it into the algorithm. Evaluation overwrites results in its output directory. Random image generation is seeded; measured timings naturally vary.

Detection parameters can be set explicitly:

```bash
python -m edgevision detect samples/mixed_contrast.png --output threshold-demo --sigma 1.2 --kernel-size 7 --low 0.04 --high 0.10 --threshold-mode absolute --boundary reflect
```

`relative` thresholds (the detect default) are fractions of the image's suppressed-gradient maximum, not fractions of its intensity. `absolute` thresholds use normalized Sobel units. CLI image loading converts to RGB; alpha is discarded, not composited. All computation runs locally without datasets or network access.

## Results included

- `samples/`: six 128×128 deterministic scenes and their binary ground-truth maps.
- `results/*_stages.png`: original, grayscale, smoothed image, gradient magnitude, suppressed magnitude, and final edges.
- `results/*_edges.png`: lossless binary detections for the balanced setting.
- `results/threshold_comparison.png`: noisy and mixed-contrast scenes, truth, and all three settings.
- [Measured summary](results/SUMMARY.md), [all precision/recall/F1 values](results/metrics.csv), and [stage timings and environment](results/report.json).
- [Verification record](docs/VERIFICATION.md): actual tests, installation checks, and reference-check scope.

Read exact pixel metrics and radius-1 metrics together. The latter use **maximum-cardinality one-to-one matching**, so a thick predicted line cannot claim several true positives for one annotated pixel. These are controlled synthetic experiments, not a natural-image benchmark or evidence of generalization. Thresholds were chosen for demonstration on these scenes; there is no held-out tuning claim.

## Architecture

| Module | Responsibility |
|---|---|
| `filters.py` | Input conversion, true convolution, Gaussian kernel, Sobel, magnitude/orientation |
| `edges.py` | Bilinear non-maximum suppression, threshold labels, BFS hysteresis |
| `pipeline.py` | Validated immutable configuration and inspectable intermediate results |
| `metrics.py` | Exact scores and sparse bipartite maximum matching |
| `synthetic.py` | Seeded images and ground truth derived from analytic masks |
| `evaluation.py` | Threshold sweeps, repeated stage/total timing, machine-readable reports |
| `reference.py` | Optional SciPy comparison, separate from the core |
| `visualization.py`, `cli.py` | Saved figures and command-line interface |

See [TECHNICAL.md](docs/TECHNICAL.md) for mathematics, boundary/tie conventions, complexity, and limitations, and [INTERVIEW.md](docs/INTERVIEW.md) for study questions with answers.
