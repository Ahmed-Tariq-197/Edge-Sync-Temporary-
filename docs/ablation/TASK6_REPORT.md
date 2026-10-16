# Task 6 Report: MobileNetV2 as the Re-ID Backbone (Ablation Experiment 6)

Status: Skeleton. Replace every `<fill>` after the Jetson runs. Do not edit the decision rules.

## 1. Summary

- Hypothesis (H6): Inference becomes faster, but false cross-robot matches may increase.
- H6a (inference faster): `<supported | not supported>`
- H6b (false matches increase): `<supported (significant) | directional only | not supported>`
- Rank-1 change: `<fill>` pp. End-to-end p50 speedup: `<fill>`x.
- Recommendation: `<fill>`

## 2. Setup

| Item | Value |
|---|---|
| Repository commit | `<fill>` |
| Device / power mode | Jetson Orin Nano 8GB, `nvpmodel` mode `<fill>`, `jetson_clocks` `<on/off>` |
| JetPack / TensorRT / CUDA | `<fill>` / `<fill>` / `<fill>` |
| Precision | FP16 (both engines) |
| Input | 256x128, ImageNet normalisation |
| Threshold | 0.70 |
| Robot split | Robot A cameras `<fill>`, Robot B cameras `<fill>` |
| Evaluation data | `<Market-1501 | custom dataset>`; queries `<fill>`, gallery `<fill>` |
| Latency protocol | Warm-up `<fill>`, measured calls `<fill>` |

## 3. Models

| Arm | Network | Weights source and training data | Training recipe | Seed | Engine size (MiB) |
|---|---|---|---|---|---|
| Baseline | OSNet x1.0 | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Candidate | MobileNetV2 + 1280->512 head | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## 4. Results

### 4.1 Inference speed

| Stage | OSNet p50 (ms) | MobileNetV2 p50 (ms) | Speedup | OSNet p99 (ms) | MobileNetV2 p99 (ms) | MobileNetV2 FPS |
|---|---|---|---|---|---|---|
| end_to_end | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| engine_only | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

Quality Gate 5 (<= 18 ms): OSNet `<pass/fail>`, MobileNetV2 `<pass/fail>`.

### 4.2 Accuracy

| Metric | OSNet | MobileNetV2 | Delta | McNemar p (Holm-adjusted) |
|---|---|---|---|---|
| Rank-1 | `<fill>` | `<fill>` | `<fill>` pp | `<fill>` |
| Rank-5 | `<fill>` | `<fill>` | `<fill>` pp | n/a |
| mAP | `<fill>` | `<fill>` | `<fill>` pp | n/a |

### 4.3 Cross-robot false matches

| Metric | OSNet | MobileNetV2 | Delta | McNemar p (Holm-adjusted) |
|---|---|---|---|---|
| False-handoff rate (95% CI) | `<fill>` | `<fill>` | `<fill>` pp | `<fill>` |
| Pair false-match rate at 0.70 | `<fill>` | `<fill>` | n/a | n/a |
| FMR at matched TAR 0.90 | `<fill>` | `<fill>` | n/a | n/a |
| EER | `<fill>` | `<fill>` | n/a | n/a |
| Correct-handoff rate | `<fill>` | `<fill>` | `<fill>` pp | n/a |

### 4.4 Thermal and resources

Peak temperature `<fill>`, throttling events `<fill>`, peak memory `<fill>` MiB (`tegrastats.log`).

## 5. Discussion

1. Speed gain versus accuracy cost: `<fill>`
2. Is the false-match change a discrimination loss or a threshold-scale shift? Compare the rows
   "FMR at matched TAR" and "Pair false-match rate at 0.70" in 4.3: `<fill>`
3. Impact on the handoff timing budget (identity stage share of the 886 ms window): `<fill>`
4. Impact on the multi-cue verifier (spatial gating compensation): `<fill>`

## 6. Threats to validity (observed)

`<fill: confirm or amend each item from the proposal, section 9>`

## 7. Conclusion and recommendation

`<fill>`

## 8. Reproduction

```
tools/ablation/task6/run_task6_benchmark.sh
```

Artifacts: `docs/ablation/task6_results/{osnet,mobilenetv2}.json`, `comparison.json`,
`comparison.md`, `tegrastats.log`. Commit hash: `<fill>`.
