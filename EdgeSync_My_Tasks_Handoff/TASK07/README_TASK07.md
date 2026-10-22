# Task 7 — TransReID Re-Identification (identity extraction for Exp-13)

Owner: Task 7 (me). Everything in this folder was **copied** from the current project tree at
`C:/Semester 7/EdgeSync-main (3)Last/EdgeSync-main (3)Last/EdgeSync-main`; nothing was moved,
renamed or rewritten, and the originals are untouched.

## 1. What Task 7 delivers

* **TransReID extraction path** — `edgesync_identity/edgesync_identity/models/transreid_backbone.py`,
  `transreid_extractor.py`, `transreid_trt_extractor.py`, `extractor_factory.py` plus the input adapter
  `core/reid_input_adapter.py` and the shared contract `interfaces/base_identity_extractor.py`
  (valid crop → unit-norm float32[512]; invalid/empty crop → `np.zeros(512, float32)`).
* **A 768 → 512 SVD projection** fitted once from measured embeddings, shipped as `.npz` assets
  with recorded digests; the 512-D output is what the Exp-13 contract consumes.
* **SIE (single-image embedding) camera conditioning** inside TransReID, which is the lever the
  results report measures.
* **Evaluation validity machinery** — `scripts/evaluate_offline.py` now declares its population
  explicitly, `scripts/analyze_task7_statistics.py` computes CMC/mAP (canonical
  `run_real_pipeline.compute_cmc_map`, validated against torchreid 0.2.5), Wilson intervals,
  bootstrap mAP, exact McNemar with Holm–Bonferroni, and FNMR/FMR/EER with a matched-TAR read.
* **The ReidSignature producer side** — `nodes/identity_node.py` can publish
  `edgesync_msgs/msg/ReidSignature.msg` on `/edgesync/identity/reid_signature`, gated by the
  `publish_reid_signature` parameter (default `False`, identity_node.py:101), and can take its
  crop from a `sensor_msgs/Image` topic named by `reid_target_crop_topic` (default empty, :103).
* **OSNet baseline files are NOT mine.** `osnet_backbone.py`, `osnet_torch_extractor.py`,
  `osnet_trt_extractor.py` and `base_identity_extractor.py` are classified BASELINE_PROTECTED (4
  rows in this task carry that classification) and were read, not touched — charter §3B freezes them.

## 2. Where these files live in the full project

`source_snapshot/` preserves the original relative path verbatim (93 files);
tests, configs and assets are additionally mirrored under `Implementation Phase/edgesync_ws/src`-named subfolders for reading
convenience, and every mirror is recorded in `SOURCE_ARCHIVE_INVENTORY.json` as a second entry in
the same row's `copies[]` list with its own digest. The authoritative copy is always
`source_snapshot/<original relative path>`.

Top-level layout of this folder:

| Subfolder | Files | Content |
|---|---:|---|
| `source_snapshot/` | 93 | Every Task 7 project file at its original relative path (authoritative copy) |
| `tests/` | 18 | Readable mirror of the Task 7 test files (17 identity + 2 bringup, deduplicated by path) |
| `configs/` | 2 | exp7_transreid.yaml and the identity package.xml |
| `assets/` | 17 | Python model package + the 512-D SVD projection .npz files (checkpoint referenced only) |
| `reports/` | 42 | Proposal REV01/REV02, results REV01/REV02, forensic audit, preflight, provenance, readiness |
| `experiment_results/` | 27 | Recorded accuracy rows, statistics, population, provenance, integrity, ONNX and TRT-blocked captures |
| `manifests/` | 5 | ES-EXP07 final deliverable manifest, integration change manifest, handoff checklist, traceability matrix |

## 3. Created, modified, unchanged (measured, not assumed)

Against the 2026-10-16 archive (`C:/Semester 7/EdgeSync-main (3)Last.zip` — an **older** version,
used only for change identification, never as a copy source):

* **created after it** (22 rows) — the whole Exp-7 documentation tree, the ReidSignature
  producer/consumer wiring tests, the statistics and population machinery.
* **modified after it** (4 rows):

| File (original relative path) | Earliest known pre-change version |
|---|---|
| `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/models/extractor_factory.py` | edgesync_es07_backup/modified_20261019_exec digest `6ffe1d7841f0cf1f…` |
| `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/models/transreid_extractor.py` | edgesync_es07_backup/modified_20261019_exec digest `6e88686eec02e750…` |
| `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/nodes/identity_node.py` | pre_edit_backup digest `5b0ecb4c9260d5ce…` |
| `Implementation Phase/edgesync_ws/src/edgesync_identity/scripts/evaluate_offline.py` | edgesync_es07_backup/modified_20261019_exec digest `d04f5c85fb39b09e…` |

* **byte-identical to it** (34 rows).
* **not comparable** (81 rows: documents,
  out-of-repo evidence and dependency-completion copies that the archive does not contain).

Where the historical version genuinely is not available anywhere, the row says so instead of
guessing: `change_status` stays `NOT COMPARED …` and no before/after claim is made.

## 4. Dependencies that are not Task 7 work

Unchanged files Task 7 needs to import and run are in `../SHARED_DEPENDENCIES/` (messages,
build metadata, baseline nodes, perception/acquisition producers). Additional in-package
reference copies were added to make this slice executable: 7 rows classified
`SHARED_PROJECT_DEPENDENCY`, each with `reference_copy: true` in the inventory and the
authoritative original path recorded in `purpose`.

## 5. Tests — which were actually executed

Freshly executed **in this packaging run, against the copies in this folder** (Tier U):

```
command : python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors
cwd     : C:\Users\PC\edgesync_handoff_20261021\EdgeSync_My_Tasks_Handoff\TASK07\source_snapshot\Implementation Phase\edgesync_ws\src\edgesync_identity
result  : 360 passed, 15 skipped, 1 error during collection, exit code 1
```

Collection error: `tests/test_identity_service.py` — ModuleNotFoundError: No module named 'rclpy'.
Skips (15), with the reason recorded for each group:

* 8 — test_gate1_scaffolding.py: "TransReID assets not available" - tests/fixtures/ (synthetic_transreid.pth, synthetic_svd_768_to_512.npz) excluded by the size policy
* 4 — test_experiment7_forensic_fixes.py (3) + test_transreid_extractor_guardrails.py (1): transformer_120.pth / projection not present in the copy
* 1 — test_experiment7_forensic_fixes.py: Market-1501 dataset directory not present
* 2 — test_extractor_factory_registry.py and test_transreid_trt_extractor.py: ES07_HEAVY gate, skips on the project tree too

Per-file counts for the Task 7 files (each executed separately, `pytest -p no:cacheprovider`):

| Test file | Result | Exit | Evidence |
|---|---|---:|---|
| `test_gate1_scaffolding.py` | 24 passed, 8 skipped in 0.36s | 0 | `perfile_test_gate1_scaffolding.py.log` |
| `test_identity_node_transreid_integration.py` | 20 passed, 18 warnings in 0.31s | 0 | `perfile_test_identity_node_transreid_integration.py.log` |
| `test_transreid_extractor_guardrails.py` | 17 passed, 1 skipped in 2.27s | 0 | `perfile_test_transreid_extractor_guardrails.py.log` |
| `test_extractor_factory_registry.py` | 22 passed, 1 skipped in 0.50s | 0 | `perfile_test_extractor_factory_registry.py.log` |
| `test_evaluate_offline_population.py` | 28 passed in 0.40s | 0 | `perfile_test_evaluate_offline_population.py.log` |
| `test_exp7_offline_exp_id.py` | 24 passed, 9 warnings in 12.47s | 0 | `perfile_test_exp7_offline_exp_id.py.log` |
| `test_analyze_task7_statistics.py` | 68 passed in 5.58s | 0 | `perfile_test_analyze_task7_statistics.py.log` |

Inherited (historical) runs are kept in `../VALIDATION/inherited_evidence/` and are **not**
mixed into the fresh numbers.

## 6. Experiment outputs and the measured numbers

`experiment_results/` now holds 27 files: the in-repo recorded
rows under `results/experiment7/`, and the statistics/provenance/integrity/ONNX artefacts that
live in the out-of-repo run directory `C:/Users/PC/edgesync_es07_runs/` (origin recorded per row
in `SOURCE_ARCHIVE_INVENTORY.json`).

The headline accuracy result, quoted with its exact evaluation population
(`reports/ES-EXP07-TASK7-REPORT-01-REV02.md` §4.3; population file `experiment_results/statistics/task7_population.json`):

Population `directory`: 3,368 valid queries of 3,368 (0 queries without a gallery match),
750/750 query identities evaluated, 14.741 genuine and 15,909.1 candidate gallery rows per valid
query, gallery n = 15,913, max rank 50, canonical protocol `run_real_pipeline.compute_cmc_map`.

| Metric (TransReID arm) | `512_sie` | `512_agnostic` | Δ | Paired test |
|---|---|---|---|---|
| Rank-1 % | **93.3789** (Wilson 92.489–94.170) | 88.7470 (87.635–89.770) | +4.6318 pp | exact McNemar, Holm p = 4.361e-41 (5/161 discordant of 3,368) — REJECT H0 |
| Rank-10 % | 98.8717 | 97.8622 | +1.0095 pp | not in the declared family |
| mAP % | **85.1252** (bootstrap 84.42–85.85) | 77.3240 (76.49–78.18) | +7.8012 pp | descriptive, not in the declared family |

Reproduction: `results/experiment7/gate3_real_results_v3_reproduce.json` returns rank1
93.37885985748218 on the same counts as the recorded row — identical to the last digit, mAP
Δ 2.38e-10 (float32 accumulation order). The accuracy half of the experiment is therefore
`REPRODUCED`, not merely `COMPUTED`.

Operating-point finding, same population: at the shipped τ = 0.70, FNMR is 48.15–60.74 % and FMR
does not move detectably; each arm's EER sits at τ ≈ 0.306–0.367 and TAR 0.90 needs τ ≈ 0.455–0.508.
The genuine-pair pool is n = 49,649 per arm; the SIE arm's mean genuine similarity 0.6839 is
**below** the shipped gate while its median 0.7057 is barely above it.

What is **not** established: the recorded OSNet reference pair (`verification_threshold_audit.json`)
declares no population, pair count, camera rule or split, so it can be compared in sign but never
cell-for-cell against the measured arms; and no valid OSNet arm was measurable on this host.

## 7. Assets — copied vs referenced (brief §4)

Copied into `assets/`: 17 files (the Python model package and the 512-D
projection `.npz` files, each with its own digest row).

Referenced, **not** duplicated, because they are large or are the dataset itself. Exact original
path, byte size and SHA-256:

| Path | Bytes | SHA-256 |
|---|---:|---|
| `Implementation Phase/edgesync_ws/src/edgesync_bringup/data/Market-1501-v15.09.15` | directory, 68,036 files | (not hashed) |
| `Implementation Phase/edgesync_ws/src/edgesync_bringup/models/transreid/transformer_120.pth` | 414,724,314 | `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5` |
| `Implementation Phase/edgesync_ws/src/edgesync_identity/tests/fixtures/synthetic_transreid.pth` | 414,718,880 | `2e518f28bb5ee5019329f0a96647300fd90a47bae218cb8313bfc0960180658c` |
| `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_512_agnostic.npz` | 37,094,828 | `cb4c8b8654ff410a12bf379ce14e01bd000efba20408c2ff63dccd6f14e671b0` |
| `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_512_sie.npz` | 37,095,723 | `d2d50a8ed50fbb54b46b0229241280e8bc9f38c40c53c2a87c1239503f9ef144` |
| `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_768_agnostic.npz` | 55,193,927 | `459cfbb31236d558636813d59484e626dfc7e631af752e2596599a439d226ac2` |
| `C:/Users/PC/edgesync_es07_runs/onnx_trt09/transreid_feature_agnostic.onnx` | 343,380,183 | `318b6857585d5d7c4882a42156a7080c11be37f84a53957b05fb1b0a9aa82b22` |
| `C:/Users/PC/edgesync_es07_runs/onnx_trt09/transreid_feature_sie.onnx` | 343,399,114 | `2cd8ccc774c5933771e5b6b653d593de3862faddfd59b471b9cecccf4bda4362` |

No projection was refitted, no checkpoint was replaced by a placeholder, and no provenance claim
is made beyond what a digest supports. Dataset authenticity is not protocol validity: the
Market-1501 tree being present and correct does not by itself validate the evaluation protocol.

## 8. Runtime requirements to actually run this slice

* Python 3.10, `numpy`, `torch`, `cv2`, `scipy`, `pytest` (all present on this host — that is how
  §5 could execute).
* For the node tests only: a stub `rclpy` and message classes generated from the on-disk `.msg`.
  Real `rclpy` is **absent**, so the whole-package identity run stops with one collection error.
* For the ONNX/TensorRT paths: `onnxruntime` was available; `tensorrt` is **absent** and there is
  no Jetson, so those arms were never executed here.
* To run the suite from the copy: set `PYTHONPATH` to this folder's
  `source_snapshot/Implementation Phase/edgesync_ws/src/edgesync_identity` and invoke pytest inside that package directory
  (the workspace must be tested per package; a single root-level run mis-resolves the packages).

## 9. Evidence-backed vs unmeasured

| Statement | Status |
|---|---|
| TransReID SIE arm gives Rank-1 93.3789 / mAP 85.1252 on the declared population | **MEASURED and reproduced** |
| SIE beats the agnostic arm on Rank-1 and genuine-accept at τ | **MEASURED, Holm-significant** |
| τ = 0.70 is mis-set for every TransReID arm measured | **MEASURED** (FNMR/FMR/EER, matched-TAR) |
| Accuracy increases versus OSNet (H7a) | **NOT VERIFIED BY MEASUREMENT** — §3B freeze, no valid arm here |
| Any latency or memory number applies to Jetson Orin Nano | **NOT MEASURED** — no Jetson, no TensorRT |
| ReidSignature is produced by a running node in a real ROS 2 graph | **NOT MEASURED** — Tier U/N only; no producer of the crop input exists |
| `benchmark_latency_memory.json` is a Jetson figure | **NO** — a local-host proxy, labelled as such |

See `TASK07_STATUS_AND_BLOCKERS.md` for the open items and what each one needs.