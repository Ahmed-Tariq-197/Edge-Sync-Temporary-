# Task 6 Proposal: MobileNetV2 as the Re-ID Backbone (Ablation Experiment 6)

Status: Proposed. Owner: Tarik. Scope: `edgesync_identity` and ablation tooling only.

## 1. Objective

Replace OSNet with MobileNetV2 as the Re-ID feature extractor and measure the trade-off between
Rank-1 accuracy and inference speed on the Jetson Orin Nano (TensorRT 8.6).

- Metric to observe: Rank-1 accuracy drop vs. inference speed gain.
- Hypothesis (H6): Inference becomes faster, but false cross-robot matches may increase.

## 2. Background

The identity service (C-07) serves `IdentityVerify.srv` with 512-dim L2-normalised embeddings and a
cosine threshold of 0.70. Extractors implement `BaseIdentityExtractor`, so a backbone swap needs no
change to the verifier, the smoother, or `edgesync_msgs`.

MobileNetV2 natively produces a 1280-dim pooled feature. To keep the 512-dim contract, the
candidate adds a trained `Linear(1280, 512)` + `BatchNorm1d(512)` embedding head. This head is part
of the experimental arm and is trained on the same source data as the baseline.

## 3. Design

| Concern | Decision |
|---|---|
| Interface | `MobileNetV2TrtExtractor(BaseIdentityExtractor)`, same 512-dim L2-normalised output |
| Selection | `reid_backbone` ROS parameter (`osnet` or `mobilenetv2`) resolved by `ExtractorFactory` |
| Runtime | One shared `TrtEngineRunner` (TensorRT 8.6, static shape, FP16) used by both arms |
| Preprocessing | One shared `ReidPreprocessor`: 256x128, ImageNet mean/std, NCHW float32 |
| Configuration | `ablation_task6_osnet.yaml` and `ablation_task6_mobilenetv2.yaml`, identical except the backbone |
| Launch | `task6_identity_ablation.launch.py backbone:=<arm> engine_path:=<path>` |
| Measurement | `edgesync_identity.ablation` (latency, Rank-k, mAP, cross-robot metrics, comparison) |
| Messages | `edgesync_msgs` is not modified |

Prerequisite fix: the existing `OsnetTrtExtractor._execute_trt` returned zeros and never ran the
engine. It now delegates to the shared runner (separate commit), so the baseline is a real OSNet.

## 4. Variables

- Independent variable: backbone (OSNet x1.0 vs. MobileNetV2 + 512-dim head).
- Controlled: input resolution, preprocessing, FP16 precision, TensorRT version, power mode and
  clocks (`nvpmodel` mode 0, `jetson_clocks`), cosine threshold 0.70, spatial gating, evaluation
  data, query/gallery split, camera-to-robot split, measurement iterations.
- Dependent: Rank-1, mAP, latency (p50/p95/p99), throughput, engine size, cross-robot false-match
  metrics.

## 5. Data and training protocol

- Training source: Market-1501 `bounding_box_train` for the MobileNetV2 head and fine-tuning
  (`tools/ablation/task6/train_mobilenetv2_reid.py`). Training subjects only; test subjects are
  never seen (Experimental Design v1.1, section 5).
- OSNet weights: record the exact source and training data of the OSNet checkpoint in the report.
- Evaluation: Market-1501 `query` and `bounding_box_test` as the public proxy. When the custom
  Mobile UGV Target Handoff Dataset is ready, convert it to the same file-name convention
  (`<pid>_c<cam>s<seq>_<frame>.jpg`) and rerun with no code change; splits are by subject.
- Robot split: cameras 1-3 form Robot A, cameras 4-6 form Robot B. Each query is matched only
  against the other robot's gallery, mirroring a cross-robot handoff.

## 6. Metrics

| Metric | Definition |
|---|---|
| Rank-1, Rank-5, Rank-10, mAP | Market-1501 protocol: junk (-1) ignored; same identity and same camera ignored |
| Latency | Per-call wall time after 50 warm-up calls over 1000 calls; p50, p95, p99, mean, std |
| Stages | `end_to_end` (preprocess + inference + normalise) and `engine_only` (TensorRT call) |
| Throughput | 1000 / mean latency (ms) |
| Engine size | Serialized engine file size (MiB) |
| Pair false-match rate | Impostor cross-robot pairs with similarity >= 0.70 / all impostor pairs |
| False-handoff rate | Queries whose best cross-robot candidate is a different identity with similarity >= 0.70 |
| Matched-TAR FMR | False-match rate at the threshold where the true-accept rate reaches 0.90 |
| EER | Equal error rate over cross-robot pairs |

Matched-TAR FMR and EER separate a real discrimination loss from a similarity-scale shift, because a
fixed threshold of 0.70 is not calibrated per backbone.

## 7. Statistical plan (fixed before data collection)

- Paired exact McNemar test on per-query Rank-1 correctness and on per-query false-handoff flags.
- Wilson 95% intervals on the false-handoff rate.
- Significance level 0.05 with Holm correction across the two primary paired tests.
- Latency is reported as percentiles; no distributional assumptions are made.

## 8. Decision rules

- H6a, inference faster: candidate `end_to_end` p50 < baseline `end_to_end` p50.
- H6b, false matches increase: candidate false-handoff rate > baseline and McNemar p < 0.05
  (Holm-adjusted). Higher but not significant is reported as directional only.
- Rank-1 drop is reported as an effect size in percentage points; this experiment sets no pass/fail
  bound on it.
- Quality Gate 5 (OSNet-class latency <= 18 ms on Orin Nano) is reported for both arms.
- If MobileNetV2 is ever proposed for the production stack, the system-level criteria of
  Experimental Design section 8 (identity IDF1 within 5 percentage points) must be re-verified.

## 9. Threats to validity

1. Training-recipe mismatch: OSNet weights may come from a different recipe than the MobileNetV2 head.
2. Domain gap: Market-1501 is a proxy for the indoor UGV scenes.
3. Threshold calibration: 0.70 was chosen for OSNet; the matched-TAR analysis addresses this.
4. FP16 numerics may affect the two networks differently; report FP16 for both.
5. Thermal and clock state: lock clocks, log `tegrastats`, discard runs with throttling.
6. One training seed: report the seed; extend to multiple seeds if time permits.
7. Camera split is a modelling choice; the split is recorded in every result file.

## 10. Procedure

1. Workstation: install `tools/ablation/task6/requirements.txt`; run `train_mobilenetv2_reid.py`
   then `export_mobilenetv2_reid.py` to produce an ONNX file.
2. Jetson: `tools/ablation/task6/build_engine.sh <onnx> <engine>` for both models (FP16).
3. Jetson: build the workspace, source it, set `OSNET_ENGINE`, `MOBILENETV2_ENGINE`, `QUERY_DIR`,
   `GALLERY_DIR`, then run `tools/ablation/task6/run_task6_benchmark.sh`.
4. Copy `comparison.md` and the JSON files into `TASK6_REPORT.md`.

## 11. Deliverables

Source: `edgesync_identity/{runtime,factory,ablation}`, `models/mobilenetv2_trt_extractor.py`,
shared constants/preprocessor/math, node parameter `reid_backbone`.
Config and launch: two arm YAMLs and `task6_identity_ablation.launch.py` in `edgesync_bringup`.
Tests: metrics, factory, extractor, benchmark/compare, node selection, launch.
Tools: training, export, engine build, benchmark runner. Docs: this proposal and `TASK6_REPORT.md`.
