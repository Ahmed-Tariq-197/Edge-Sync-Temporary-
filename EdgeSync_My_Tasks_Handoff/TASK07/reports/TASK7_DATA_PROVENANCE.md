# TASK7_DATA_PROVENANCE — ES-EXP07 Task 7 evaluation data

**Experiment:** ES-EXP07 (TransReID implementation, deployment, benchmarking, deliverables)
**Document:** data provenance for every input behind the Task 7 numbers
**Written:** 2026-10-19 · **Host:** Windows 10.0.26300, Python 3.10.11, torch 2.5.1+cu121, cuDNN 90100, RTX 3050 Laptop 4.00 GiB
**Every count in this document was measured on this host in this session. No standard dataset count is assumed.**
Labels used below: `COMPUTED` (measured here from the artifact named), `VERIFIED-BY-HASH`, `NOT RECORDED` (unavailable — stated, not guessed), `NOT APPLICABLE`.

---

## D-01 Dataset identity and measured contents

| Field | Value | Evidence |
|---|---|---|
| Dataset | Market-1501 (Person Re-identification Meets Image Search, Zheng et al., Technical Report 2015) | `readme.txt` (2,229 B) shipped in the dataset root — `COMPUTED` |
| Version label on disk | `Market-1501-v15.09.15` | directory name — `COMPUTED` |
| Absolute root | `C:\Semester 7\EdgeSync-main (3)Last\EdgeSync-main (3)Last\EdgeSync-main\Implementation Phase\edgesync_ws\src\edgesync_bringup\data\Market-1501-v15.09.15` | resolved at run time and written into the manifest — `COMPUTED` |
| Total `.jpg` in tree | **61,295** | recursive count; equals the pre-modification capture `__DATASET_JPG_COUNT__\|61295` in `es07_baseline_20261019_exec.txt` — `COMPUTED`, unchanged |
| `bounding_box_train/` | **12,936** `.jpg`, **751** identities, ID numbers 2–1500 | `COMPUTED` |
| `bounding_box_test/` | **19,732** `.jpg`, **752** distinct pid values = 750 real identities + `0` (2,798 distractors) + `-1` (3,819 junk) | `COMPUTED` |
| `query/` | **3,368** `.jpg`, **750** identities, ID numbers 1–1501 | `COMPUTED` |
| Cameras present | test `{1,2,3,4,5,6}`, query `{1,2,3,4,5,6}` | parsed from `_c<N>s<k>_` in the filenames — `COMPUTED` |
| Non-image files inside the image dirs | 4 `Thumbs.db` (8,192 / 7,680 / 14,336 / 484,352 B) | `COMPUTED`; excluded by the `*.jpg` glob and by the strict filename grammar, so they never reached an extractor |
| Auxiliary annotation dirs | `gt_bbox/` (25,260 files), `gt_query/` (6,736 `.mat`) | `COMPUTED`; **not used** by this task |

**Cross-check against the dataset's own readme** (`COMPUTED`, both directions): the readme states 1501 annotated individuals, 751 for training, 750 for testing, 3,368 query images, 19,732 test images, 12,936 training images. Every one of those six numbers equals the value measured from the files on disk. This is what "verify the actual split instead of assuming standard counts" means here: the counts were measured first, then compared to an independent in-tree source rather than to memory.

## D-02 How the identity numbers are actually laid out (this is the finding that invalidated the old band filter)

Market-1501 IDs are **globally unique across the 1501 people and are *not* contiguous per split**. Measured:

- training identities present in `bounding_box_train/`: **751**, ID numbers spanning **2–1500**; **355 of them carry ID ≥ 752**.
- test identities present in `bounding_box_test/`: **750**; **355 of them carry ID ≤ 751**.
- `train ∩ test(real)` = **0** and `train ∩ query` = **0** — the ID *number* spaces of the two splits are disjoint, and `query` ⊂ `test` identities with `test_real − query = ∅`.
- ID numbers 1–651 are **not** all present in the train split: 310 of those numbers do not exist there (e.g. 1, 3, 4, 5, 6, 8, 9, 13 …).

Consequences recorded for the audit (task #20), each with the measurement that produces it:

1. The project documentation's split scheme — "Pretraining IDs 0001–0751, SVD fitting 0001–0650, calibration 0651–0751, test 0752–1501" (`fit_svd_projection.py:17-21`) — **does not describe this dataset**. Only the folder split does. Under that scheme, 355 of the 751 checkpoint-training identities fall inside the documented "test" band.
2. The old `evaluate_offline.py` filter (`id_min=752, id_max=1501` applied inside the test/query folders) evaluated **6,226 / 19,732** gallery files and **1,774 / 3,368** query files across **395 of 750** test identities — i.e. after the junk/distractor rows are accounted for, 6,226 of the 15,913 valid gallery rows (**39.13 %**), silently dropping 9,687 gallery and 1,594 query observations. Because no `pid ≤ 0` survives a `752 ≤ id` band, the distractor-rejection metric could only ever print `0.00 % (0/0)`. Reproduced by re-running the filter on the folder contents today — `COMPUTED`.
3. **No identity leakage into the Task 7 numbers.** The disjointness in the bullet list above (train ∩ query = 0, train ∩ test = 0) is the measurement that supports this, and it is a property of the data, not of this task's code.

## D-03 Evaluation population actually used, and its reconciliation

Population name: **`directory`** — the option added by TR-07 in `scripts/evaluate_offline.py`; its rule text is carried verbatim into every artifact.

| Rule | Definition (as executed) |
|---|---|
| gallery | every `*.jpg` in `bounding_box_test/` whose parsed `pid ≥ 0`; the `0000` distractor rows are **kept** as gallery negatives; `-1` junk rows are **excluded** |
| query | every `*.jpg` in `query/` whose parsed `pid > 0` |
| filename grammar | `^(-?\d+)_c(\d)s(\d)_.*\.jpg$` — a name that does not match is excluded as `unparsable_filename` instead of silently defaulting to `(-1, -1)` |

Measured reconciliation, written by `generate_task7_embeddings.py` into `task7_embeddings_manifest.json` and re-read by the statistics consumer:

```
gallery: files_scanned 19,732 = kept 15,913 + excluded{ junk_identity_-1: 3,819 }   identities 751   distractors_pid_0 2,798   id range [0, 1501]
query:   files_scanned  3,368 = kept  3,368 + excluded{ }                            identities 750                                    id range [1, 1501]
```

`files_scanned == kept + Σ excluded` holds in both rows — the statistics consumer asserts exactly this identity (`scan_accounting`) and aborts if it fails.

Rejected alternative, kept available behind `--population id_band_752_1501` with its measured counts printed: 6,226 / 15,913 gallery and 1,774 / 3,368 query rows, `distractors_pid_0 = 0` → distractor rejection becomes `NOT APPLICABLE`. It is **not** used for any Task 7 number.

## D-04 Embedding artifacts (the direct input to every Task 7 metric)

Directory: `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory\` — **outside** the project tree. Generated 2026-10-19T14:47:29Z, elapsed 956.4 s, host block as in the header above. `VERIFIED-BY-HASH` at generation and re-verified by the consumer before a single statistic was computed (`integrity.match: true` for all three files).

| Arm | File | Bytes | SHA-256 (full) | Dim | Rows |
|---|---|---|---|---|---|
| `transreid_768_agnostic` | `task7_transreid_768_agnostic.npz` | 55,193,927 | `459cfbb31236d558636813d59484e626dfc7e631af752e2596599a439d226ac2` | 768 | 15,913 gallery + 3,368 query |
| `transreid_512_agnostic` | `task7_transreid_512_agnostic.npz` | 37,094,828 | `cb4c8b8654ff410a12bf379ce14e01bd000efba20408c2ff63dccd6f14e671b0` | 512 | same |
| `transreid_512_sie` | `task7_transreid_512_sie.npz` | 37,095,723 | `d2d50a8ed50fbb54b46b0229241280e8bc9f38c40c53c2a87c1239503f9ef144` | 512 | same |
| manifest | `task7_embeddings_manifest.json` | 7,352 | `283c8c782e66869a429a6db4d63cbaa3f19d938fc68a418f2a997ba98dfe990a` | — | — |
| per-arm exclusions | `*_exclusions.json` × 3 | 37 each | empty exclusion lists | — | 0 excluded rows in every arm |

Properties measured at generation and re-checked by the consumer (`COMPUTED`):

- Array inventory of every arm file: `gallery_features (15913, dim) float32`, `query_features (3368, dim) float32`, `gallery_paths (15913) <U92`, `query_paths (3368) <U80`, `gallery_pids`/`query_pids (… ) int32`, `gallery_camids`/`query_camids int32`, `gallery_latencies_ms`/`query_latencies_ms float64` — so each row carries its own source path, pid and camid, and no metric depends on an unlabeled index.
- **19,281 real embeddings** (15,913 + 3,368) per arm; per-arm `gallery_excluded = query_excluded = 0`, `extractor_failure_counts {}`, `last_failure null` — so no row is a sentinel standing in for a failed inference (charter §5).
- unit-norm max deviation **1.7881e-07** for both 512-D arms, `any_zero_row false`; the wire contract `float32[512]`, unit norm, `EMBEDDING_DIM = 512` is satisfied by both. The **768-D arm is deliberately not unit-norm** — it is the raw post-BNNeck feature, measured max `|‖x‖ − 1| = 37.398` — which is exactly why it is labelled a diagnostic arm and never the deployed wire.
- **parity cross-check ran on every image** (`every_nth: 1`, gallery 15,913 + query 3,368), tolerance 2e-05, result `PASS` — the production `extract_embedding()` output equalled the independently re-derived centre → SVD → L² contract for all 19,281 images; a violation aborts the run.
- SIE arm receives `cam_label = camid − 1` per the deployed node convention; the 768-D arm is the pre-projection BNNeck feature and is labelled `diagnostic, not the deployed wire`.
- `latency_note` in the manifest: the per-image `latency_ms` bookkeeping is unpaced on a 4 GB laptop GPU with no warm-up/repeat protocol and **must not** be reported as Jetson Orin Nano latency (charter §7). `NOT MEASURED` for the target.
- OSNet: the factory attempt in the same run is recorded as `FAILED — recorded, not repaired (charter §3B): RuntimeError: operator torchvision::nms does not exist`, with `effect_on_comparison` = no OSNet embeddings ⇒ no paired per-query outcomes ⇒ a TransReID-vs-OSNet paired test is `NOT APPLICABLE` on this host. No OSNet file was modified.

## D-05 Statistics inputs, thresholds and the artifacts written

- Threshold **τ = 0.70** read at run time from `Implementation Phase\edgesync_ws\src\edgesync_bringup\config\algorithm_thresholds.yaml`, key `algorithm_thresholds.ros__parameters.reid_similarity_threshold`; output carries `matches_expected_0.70: true`. A config that does not carry 0.70 makes the consumer exit non-zero (regression-tested). τ was **not** retuned (frozen supervisor decision).
- Acceptance rule inherited from `core/identity_verifier.py:101` (rejects when `sim < τ`) ⇒ **accept iff `score >= τ`**, applied on float32 scores with `norm < 1e-6 → 0.0` and clip to `[-1, 1]`; cross-checked against `IdentityVerifier.compute_cosine_similarity` on 200 real pairs (seed 11): max abs difference **1.788e-07** (`COMPUTED`).
- Metric source: canonical `run_real_pipeline.compute_cmc_map` reached through `evaluate_offline.compute_cmc_map` (the shipped implementation, validated against torchreid 0.2.5). The consumer's own per-query pass is asserted equal to it before any number is reported; measured max-abs difference **exactly 0.0** for all three arms and both camera partitions (`COMPUTED`).
- Pair population written to `task7_population.json` and embedded in `task7_statistics.json`: **49,649** genuine pairs, **53,532,220** impostor pairs, **13,115** same-pid/same-camera duplicate pairs excluded, **0** junk gallery rows in-array, **0** queries without any impostor candidate. Symmetric duplicates are excluded, so no pair is double-counted.
- Canonical pair rules as executed: a genuine pair is (query, gallery row) with equal pid; same-pid **and** same-camera rows are dropped as duplicates before scoring; queries with no surviving gallery match are invalid (there are none here: 3,368/3,368 valid).
- Artifacts: `C:\Users\PC\edgesync_es07_runs\statistics\task7_statistics.json`, `task7_population.json`, `task7_statistics.md`; logs `analyze_arm1_512agnostic.log` (exit 0, 30.7 s) and `analyze_all_arms.log` (exit 0, 123.3 s). Each log's first line is the `CMD:` record of its exact invocation.

## D-06 What was deliberately NOT done to this data

- No dataset file was created, deleted, moved, renamed or re-encoded. Read-only access; the `.jpg` total re-measured today equals the pre-modification capture (`COMPUTED`).
- No split was changed; `directory` reproduces the population the recorded arm used (rule text + provenance string travel with every artifact).
- No model, projection or threshold was re-fit or retuned to improve a Task 7 number. The 768→512 projection was consumed exactly as shipped (`9626bea5…`, see `TASK7_WEIGHTS_PROVENANCE.md`).
- No test-set-driven selection: hyper-parameters, arm list, pair rules and the 3-hypothesis McNemar family were fixed before the analysis run; nothing was chosen by looking at a metric.
- No failed inference was replaced by a zero/random/synthetic embedding; every arm reports 0 exclusions and 0 failure counts, which is a measured property rather than an assumption.
- Camera-partition results are a **viewpoint-split proxy only** — no robot, no handoff event, no handoff label exists in this project. Charter §8 forbids inferring cross-robot handoff success or false-handoff rates from these numbers, and none is claimed.

## D-07 Provenance gaps (stated, not papered over)

| Gap | Status |
|---|---|
| The dataset's acquisition channel and an official release checksum | `NOT RECORDED` — the tree contains no download record or signature for `Market-1501-v15.09.15`. What is verifiable is internal consistency: the readme's six published counts equal the measured ones (D-01). External authenticity of the images cannot be established from this host. |
| Per-file digests for 61,295 images | `NOT MEASURED` (never captured, and hashing 61k files was not part of any gate). Coverage provided instead: the pre/post `.jpg` count, the scanned/kept/excluded reconciliation per split, and 19,281 embeddings each tied to a parsed `(pid, camid)`. |
| Which run produced each projection `.npz`, and with which `--id-min/--id-max` | `NOT RECORDED` — the `.npz` files carry no command, date, sample-count or identity-range metadata. See `TASK7_WEIGHTS_PROVENANCE.md` W-04 for what *is* measurable from their linear algebra. |
| Where the OSNet arm can actually run | `BLOCKED on this host` — `torchvision::nms` missing; §3B forbids repairing it. Re-run on a host where `torchreid` imports. |

## D-08 Reproducing the measurements in this document

Every count above is emitted by one read-only script:

```
python C:\Users\PC\edgesync_es07_runs\provenance_measure.py     # exit 0, ~8 s
```

It walks the dataset folders, hashes and opens all six projection `.npz` files, reads the two checkpoint
inventories, re-hashes the three embedding `.npz` files against the manifest, and writes
`C:\Users\PC\edgesync_es07_runs\provenance\task7_provenance_measurements.json`. It creates or modifies
nothing inside the project tree. The manifest digest it computes (`283c8c78…`, 7,352 B) equals the one
the statistics consumer recorded in `task7_statistics.json → inputs.manifest.sha256`, so the two
independent code paths agree on the same input.

