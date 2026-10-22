# EdgeSync — My Tasks Handoff (Task 7 TransReID Re-ID, Task 13 CBAA/CBBA Handoff)

Prepared 2026-10-21 for the team leader. Everything here was **copied**, never moved; the
project tree was opened read-only and no project file was created, renamed, modified or
deleted by this packaging run.

## 1. Source of truth

| | |
|---|---|
| Packaged from (authoritative) | `C:/Semester 7/EdgeSync-main (3)Last/EdgeSync-main (3)Last/EdgeSync-main` |
| Inspected as a git repository? | No — `False`. Change history therefore comes from measured digests, not `git`. |
| Old archive used as a copy source? | **No.** `C:/Semester 7/EdgeSync-main (3)Last.zip` |
| Role of that archive | Change-identification evidence only. It is an **older version that predates the latest Task 7 and Task 13 work**. |

Proof that the current tree, not the ZIP, carries the work (measured by hashing the ZIP's
own stored entry bytes):

* The ZIP contains **no** `docs/experiments/` tree, no `negotiation/` package directory, no
  `cbba*` file, no `exp13*` file, no `ReidSignature.msg` and no `ConsensusBelief` definition.
  The packaged tree has all of them (16 Task 13 files and
  22 Task 7 files are simply absent from the archive).
* 8 packaged files are
  byte-different from their 2026-10-16 archive version, i.e. they were modified after it.

Three independent provenance sources were used so that no change claim rests on the ZIP alone:
the archive entry digests, 5 out-of-repo backup sets that hold
actual earlier versions (65 packaged files matched to a pre-change
version — see `pre_change` / `pre_change_verdict` in `SOURCE_ARCHIVE_INVENTORY.json`), and the
in-repo before→after change manifests copied verbatim into `TASK07/manifests/` and
`TASK13/manifests/`.

## 2. What is in this folder

| Folder | Files | What it is |
|---|---:|---|
| `TASK07/` | 206 | Task 7 — TransReID re-identification: source slice, tests, configs, model assets, reports, experiment evidence |
| `TASK13/` | 162 | Task 13 — CBAA (CBBA L=1) decentralised target handoff: source slice, tests, config, launch, benchmark results |
| `SHARED_DEPENDENCIES/` | 48 | Unchanged project files both tasks depend on (messages, baseline nodes, build metadata) |
| `INTEGRATION/` | 5 | Dependency map across Exp 7 → Exp 13 and the open integration blockers |
| `VALIDATION/` | 43 | Tests executed **fresh against the copies** plus the inherited historical evidence, kept separate |
| root | — | `README.md`, `MASTER_FILE_INDEX.csv/.md`, `SOURCE_ARCHIVE_INVENTORY.json`, `FINAL_PACKAGE_MANIFEST.json` |

Package total at the moment this README was written: **468 files, 61,668,125 bytes**
(no `__pycache__`, no `.pyc`, no build output). The authoritative delivered totals — which also cover
`MASTER_FILE_INDEX.csv`, `MASTER_FILE_INDEX.md` and `FINAL_PACKAGE_MANIFEST.json`, written after this
file — are in `FINAL_PACKAGE_MANIFEST.json` → `output_directory_structure`.

## 3. Inventory at a glance (measured)

327 inventory rows: TASK07 141, TASK13 97,
SHARED 47, VALIDATION 40,
INTEGRATION 2.

| Ownership classification (brief §6) | Rows |
|---|---:|
| MY_TASK_IMPLEMENTATION | 50 |
| MY_TASK_TEST_OR_ARTIFACT | 158 |
| SHARED_PROJECT_DEPENDENCY | 65 |
| BASELINE_PROTECTED | 51 |
| OTHER_TEAM_OWNED | 3 |

| Change status (measured against the 2026-10-16 archive) | Rows |
|---|---:|
| UNCHANGED since the archive | 86 |
| MODIFIED after the archive | 8 |
| ABSENT from the archive (created afterwards) | 40 |
| Not comparable (documents / dependency completion / out-of-repo evidence) | 164 |

## 4. Verification levels used throughout (brief §5/§8)

| Level | Meaning in this package |
|---|---|
| **Tier U** | Unit / model simulation in one process. No ROS graph, no network, no robot. |
| **Tier N** | Node-level tests under a **stubbed** `rclpy`, with message classes generated from the on-disk `.msg` files. This is **not** ROS 2 integration. |
| **Tier R** | Real ROS 2 / DDS graph. **NOT ACHIEVED — blocked, see below.** |
| **Tier H / J** | Jetson / TensorRT / physical robot. **NOT MEASURED — no such hardware on this host.** |

## 5. Deliberately not duplicated (brief §4)

Large assets are referenced with their exact original path, size and SHA-256 rather than
copied, to avoid duplicating the dataset:

* `Implementation Phase/edgesync_ws/src/edgesync_bringup/data/Market-1501-v15.09.15` — directory, 68,036 files
* `Implementation Phase/edgesync_ws/src/edgesync_bringup/models/transreid/transformer_120.pth` — 414,724,314 bytes, SHA-256 `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5`
* `Implementation Phase/edgesync_ws/src/edgesync_identity/tests/fixtures/synthetic_transreid.pth` — 414,718,880 bytes, SHA-256 `2e518f28bb5ee5019329f0a96647300fd90a47bae218cb8313bfc0960180658c`
* `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_512_agnostic.npz` — 37,094,828 bytes, SHA-256 `cb4c8b8654ff410a12bf379ce14e01bd000efba20408c2ff63dccd6f14e671b0`
* `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_512_sie.npz` — 37,095,723 bytes, SHA-256 `d2d50a8ed50fbb54b46b0229241280e8bc9f38c40c53c2a87c1239503f9ef144`
* `C:/Users/PC/edgesync_es07_runs/embeddings_full_directory/task7_transreid_768_agnostic.npz` — 55,193,927 bytes, SHA-256 `459cfbb31236d558636813d59484e626dfc7e631af752e2596599a439d226ac2`
* `C:/Users/PC/edgesync_es07_runs/onnx_trt09/transreid_feature_agnostic.onnx` — 343,380,183 bytes, SHA-256 `318b6857585d5d7c4882a42156a7080c11be37f84a53957b05fb1b0a9aa82b22`
* `C:/Users/PC/edgesync_es07_runs/onnx_trt09/transreid_feature_sie.onnx` — 343,399,114 bytes, SHA-256 `2cd8ccc774c5933771e5b6b653d593de3862faddfd59b471b9cecccf4bda4362`

Because those assets were not copied, 2 tests in `edgesync_bringup` fail **in the copy only**,
purely on asset-existence assertions, and 12 identity tests skip. That is a packaging
consequence, not a code defect; it is spelled out in `VALIDATION/VALIDATION_SUMMARY.md`.

## 6. What this package does **not** claim

* No ROS 2, DDS, TensorRT, Jetson or physical-robot validation. None was possible on this host
  (no `rclpy`, no generated messages, no `launch`, no ROS 2 CLI). See `VALIDATION/fresh_evidence/FRESH_RUN_RECORD.json`.
* No test is reported as passed because a test file exists; only freshly executed suites are
  reported as executed, and inherited results are labelled inherited.
* No projection was refitted, no model retrained, no result regenerated, no unrelated defect
  fixed, and no baseline-owned or other-team-owned file was changed.
* Conflicting historical reports are preserved as historical evidence, not rewritten.

## 7. The decisions the leader still has to make

1. **No image-crop producer exists.** Nothing in the workspace publishes a `sensor_msgs/Image`
   target crop, so Exp 7's signature publisher has no input and the real handoff is BLOCKED.
2. **`ConsensusBelief.msg` has never been compiled** — Tier R is unreachable until the workspace
   is built with ROS 2 present.
3. **CBAA does not stay live under two fault conditions** — `dead_participant` stalls 60/60 and
   `one_silent_candidate` ends undecided 60/60 in the Exp-13 arm. See `TASK13/README_TASK13.md`.
4. **Task 7's threshold τ = 0.70 is mis-set for every TransReID arm measured** (EER sits near
   τ ≈ 0.31–0.37); τ is an owner calibration decision and was not changed.
5. **The OSNet accuracy axis was never measurable here** — charter §3B freezes OSNet, so H7a is
   recorded as NOT VERIFIED BY MEASUREMENT.

## 8. How to verify the package itself

`FINAL_PACKAGE_MANIFEST.json` lists every delivered file with its SHA-256 and byte size;
`VALIDATION/HASH_AND_INTEGRITY_REPORT.md` states the copy-time verification outcome
(453 of
453 copy digests match their source) and how to
re-run it. The ZIP is described in §9 of the final report, not claimed here.

## 9. Start reading here

1. `TASK07/README_TASK07.md` and `TASK07/TASK07_STATUS_AND_BLOCKERS.md`
2. `TASK13/README_TASK13.md` and `TASK13/TASK13_STATUS_AND_BLOCKERS.md`
3. `INTEGRATION/TASK07_TASK13_DEPENDENCY_MAP.md` and `INTEGRATION/OPEN_INTEGRATION_BLOCKERS.md`
4. `VALIDATION/VALIDATION_SUMMARY.md`
5. `MASTER_FILE_INDEX.csv` for the machine-readable file-by-file view
