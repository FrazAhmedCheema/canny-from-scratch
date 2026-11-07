# Measured evaluation results

AI-generated study implementation; synthetic benchmarks only.

| Setting | Macro precision (r=1) | Macro recall (r=1) | Macro F1 (r=1) |
|---|---:|---:|---:|
| sensitive | 0.8858 | 0.9996 | 0.9276 |
| balanced | 0.8849 | 0.9167 | 0.8723 |
| strict | 0.7182 | 0.7500 | 0.7057 |

Seed 1729; 128×128; 5 timed repetitions after one warm-up.
Macro values average per-scene scores, not pixel-pooled counts. Exact scores are in metrics.csv.

| Scene | Pipeline median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| vertical_step | 3.952 | 3.671 | 5.157 |
| rectangle_rgb | 4.269 | 4.171 | 4.697 |
| disk | 4.049 | 4.000 | 4.400 |
| diagonal | 4.628 | 4.132 | 4.968 |
| noisy_rectangle | 5.073 | 4.807 | 5.518 |
| mixed_contrast | 6.385 | 6.329 | 6.731 |

Stage medians, versions, boundary/threshold settings and optional reference results: report.json.
Timings exclude I/O, matching, and rendering; sums of stage medians need not equal median total.
These small synthetic scenes are not evidence of real-world generalization or competitive performance.
