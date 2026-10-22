# Task 7 Proposal: TransReID as the Re-ID Backbone (Ablation Experiment 7)

Status: Executed as far as this host allows — **REV02**. Owner: Abanob. Scope: `edgesync_identity` and ablation tooling only. `edgesync_msgs` is not modified.

**REV02 supersedes `ES-EXP07-TASK7-PROPOSAL-01-REV01.md`** (21,351 B, sha256-16 `845aeb70c5fdbd0f`), which stays on disk **byte-for-byte unmodified** — re-verified at the moment this REV02 was written, and duplicated at `C:\Users\PC\edgesync_es07_backup\modified_20261019_exec\ES-EXP07-TASK7-PROPOSAL-01-REV01.md` with the same digest. REV02 changes no design rule and no decision rule; it records, against REV01, what was actually executed, what was measured, and which gates remain open. Each edit is tagged **[REV02]**; untagged text is identical to REV01. No REV01 `[REV01]` tag was removed, and no Appendix A number was altered.

> **[REV02] provenance correction that REV01 could not verify.** REV01's supersession line says v01 is "preserved byte-for-byte in `es07_audit/backup/`". Measured on this tree: **no `es07_audit/` directory exists anywhere in the project**, and `ES-EXP07-TASK7-PROPOSAL-01.md`, `ES-EXP07-TASK7-REPORT-01.md`, `ES-EXP07-STRUCTURAL-REPORT-01.md`, `ES-EXP07-TASK7-SUMMARY-01.md` and `JETSON-RUN-SHEET-PREP-01.md` are **all absent** (checked by name and by recursive `find`). The digests `aa9f5f7dbf9d9127` (proposal v01) and `926a907b2449e7a9` (report v01) therefore cannot be re-verified against a file on disk; they are carried forward as *asserted*, not measured. This is a finding for `TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`, not something REV02 can repair.

**Gate (REV01 text, retained):** this REV01 is *preparation*. Nothing here may be executed, and no row may be filled with a device claim, until owner decisions **G.1–G.7** land (structural report section G).

**[REV02] Gate status as executed.** The third master charter superseded the preparation-only stance for everything that does **not** need the device or an owner-set budget. Status per gate, each claim marked:

| Gate | REV01 meaning | Status at REV02 | Evidence |
|---|---|---|---|
| G.1 | identity-stage share of `T_txn = 886 ms` (and the packaging fixes listed as run-sheet `G-d`/`G-f`) | **OPEN** — never fixed by the owner | `config/timing_budgets.yaml:11 dt_id_ms = 120` is an on-disk candidate, not a decision. H7b therefore **cannot be decided** |
| G.2 | projection must be hash-pinned, not name-picked | **SATISFIED by measurement, not by owner reply** | deployed `svd_projection_768_to_512_real_v2.npz`, 1,466,587 B, sha256 `9626bea5526df5b0a9d5b4dccec1c552224b5a6da504a0ab3a69f6bbfec1dbfc`, recorded in the embeddings manifest and re-verified by the statistics consumer |
| G.3 | edge memory envelope for the Re-ID engine | **OPEN** — no envelope exists | `config/hw_params.yaml:36 unified_memory_limit_mb = 8192` is still read by no node. H7c therefore **cannot be decided** |
| G.5 | camera id for SIE, given `IdentityVerify.srv` is frozen | **DECIDED by the owner (logged as PF-10c)** | "add a per-robot `reid_camera_id` ROS parameter"; implemented, `edgesync_msgs` untouched |
| G.6 | a repository commit to key results against | **OPEN / NOT OBTAINABLE** | not a git repository; `git init`/commit are prohibited by charter §2 and the owner handles Git. Results carry `NO-COMMIT` |
| G.7 | permission to touch the Jetson | **DENIED FOR THIS SESSION** | owner reply: "Not reachable now — I'll connect it myself later" → every device row is `BLOCKED — Pending Jetson Orin Nano hardware access` |
| — | ONNX dependencies | **AUTHORIZED** | `pip install --user onnx onnxruntime`; both 1.23.2, measured |

**Charter (as assigned):** *Experiment 7 (TransReID): Replace OSNet with TransReID. Metric to observe: Memory footprint on Edge and Inference Latency vs. Accuracy. Hypothesis: While accuracy increases, the 200+ ms inference latency might break the critical TTO handoff budget, proving OSNet is the optimal trade-off.*

**Why this proposal exists.** The prior audit (`ES-EXP07-TASK7-SUMMARY-01.md`) established that the charter's primary metric has **never** been measured on edge hardware, that its latency premise (200+ ms) was never observed on any harness, that the offline latency evidence on disk contradicts itself in sign, and that τ = 0.70 FNMR has three live definitions depending on an undeclared pair population. This design closes those four gaps before any number is collected.

---

## 1. Objective

Replace OSNet with TransReID ViT-B/16 as the Re-ID feature extractor and measure, **on the Jetson Orin Nano (FP16)**, the three-axis trade-off the charter names: accuracy, inference latency, and memory footprint.

**[REV01]** "TensorRT 8.6" is **UNVERIFIED** — it is an assertion with no source on disk. `tensorrt` is imported by 2 files but declared in no `package.xml` (`edgesync_identity/package.xml:10-14`), **no `.engine`, `.plan`, `.onnx` or `.trt` file exists anywhere in the workspace**, and `ws/build|install|log` are absent, i.e. the workspace has never been `colcon`-built. The JetPack / TensorRT / CUDA triplet must be read off the device at run time and recorded, not assumed here.

**[REV02] Status of that paragraph — three of its four claims are now obsolete, one still holds.**
- `.onnx`: **no longer true.** Two ONNX graphs exist, produced by `scripts/export_transreid_onnx.py` and byte-reproducible across two independent runs: `transreid_feature_agnostic.onnx` 343,380,183 B sha256 `318b6857585d5d7c4882a42156a7080c11be37f84a53957b05fb1b0a9aa82b22` and `transreid_feature_sie.onnx` 343,399,114 B sha256 `2cd8ccc774c5933771e5b6b653d593de3862faddfd59b471b9cecccf4bda4362` (COMPUTED; both `onnx.checker` PASS, opset 17, IR v8, **779 nodes agnostic / 783 SIE**, 154 / 155 initializers, 0 external-data tensors, 0 input-contract violations, and an off-contract 128×64 input is **refused by the graph** with `InvalidArgument`). **Neither is a deployment**: charter §6 steps 6–10 (engine build, engine run, on-device parity) were not performed.
- `.engine` / `.plan` / `.trt`: **still true — zero such files exist anywhere.** `tensorrt` and `pycuda` are not importable on this Windows host and are still declared in no `package.xml`; `build_transreid_trt_engine.py` is delivered and unit-tested against a stub, never executed against a real TensorRT.
- `ws/build|install|log`: **still true** — the workspace has never been `colcon`-built, and `colcon` is not installed here.
- JetPack / TensorRT / CUDA triplet: `NOT MEASURED — Pending Jetson Orin Nano hardware access`. This host is Windows 10.0.26300, Python 3.10.11, torch 2.5.1+cu121, cuDNN 90100, RTX 3050 Laptop 4 GiB.

Hypothesis (H7): accuracy increases, but edge inference latency and edge memory footprint exceed the handoff budget / 8 GB envelope, proving OSNet is the optimal trade-off.

## 2. Background

The identity service (C-07) serves `IdentityVerify.srv` with 512-dim L2-normalised embeddings and a cosine threshold of 0.70. Extractors implement `BaseIdentityExtractor`, so a backbone swap changes neither the verifier, the smoother, nor `edgesync_msgs`.

TransReID emits 768-D post-BN-Neck CLS features. To keep the 512-dim contract, the candidate arm uses a **centered truncated-SVD 768→512 projection fitted on the training partition only** (R₅₁₂ = 0.98641 already measured; must be re-fitted and re-versioned, never re-fitted on evaluation features).

**[REV02] The projection was NOT re-fitted, by design.** Charter §5 forbids silently refitting it or changing its fitting population, and the guard is measurable rather than rhetorical: the deployed `.npz` stores `projection`, `mean`, `singular_values`, `total_variance_ratio`, while the current `scripts/fit_svd_projection.py` additionally writes `explained_variance_ratio` — a key **absent** from the shipped file. The deployed matrix therefore came from an earlier version of that script and re-running it is *not* a proven route to `9626bea5…`. Task 7 consumed the shipped file hash-pinned. Three further measured corrections (full detail in `TASK7_WEIGHTS_PROVENANCE.md` §W-02):
- **The fitting band in the script's docstring is wrong.** `fit_svd_projection.py:17-21` documents "SVD Fitting: IDs 0001–0650"; measured against the files that band captures **341 of the 751 training identities / 6,241 images** (the "calibration" band 651–751 captures 55 identities / 1,180 images). No test leakage results — that population lies wholly inside `bounding_box_train`, which is identity-disjoint from test and query — but the documented number is not the real one.
- **R₅₁₂ = 0.9864118 is in-sample**, measured on features from the same checkpoint, on the split that checkpoint was trained on. It is a property of that fit, not an out-of-sample compression claim.
- **Names do not identify content.** `svd_projection_768_to_512_real_v2.npz` ≡ `_real_v3_reproduce.npz` (byte-identical), and `svd_projection_768_to_512_real.npz` ≡ the un-suffixed `svd_projection_768_to_512.npz` (`dc2b9072850c299d`), which differs from the deployed `real_v2` by up to 0.2518 per element. Selecting an asset by name can silently pick the wrong matrix.

**Prerequisite fixes (each its own commit, none may be silently folded into the experiment):**

| ID | Defect on disk today | Consequence if unfixed |
|---|---|---|
| P1 | `OsnetTrtExtractor._execute_trt` returns `np.zeros(512)` and never touches the execution context — verified at `models/osnet_trt_extractor.py:72-75` | the baseline arm produces zero embeddings, i.e. not a real OSNet |
| P2 | No TransReID TensorRT extractor exists; `transreid_trt_extractor.py` is neither on disk nor imported anywhere (AST scan: 0 unresolved relative imports). No engine artifact of any format exists. `edgesync_bringup/setup.py:11-16` installs only `config/` and `launch/`, so `models/` and `data/` never reach an install space | the candidate arm cannot run on the edge at all, and even a correct engine path in `models/` would not be installed **[REV01]** |
| P3 | The contract is `extract_embedding(self, image_crop)` (`interfaces/base_identity_extractor.py:27`) and the node calls it with no camera id (`nodes/identity_node.py:266-272`), so `TransReIdExtractor`'s optional `cam_label` (`models/transreid_extractor.py:170`) is never supplied in the ROS path | TransReID loses the camera embedding. Measured, **native-768 arm only**: 88.64 → 89.91 Rank-1 (`gate3_real_results_reconciled.json` → `/corrected_ablation_results/transreid_native_post_bn_768_no_cam/rank1` and `..._with_cam_sie/rank1`). The 512-D SVD arm (93.379, `gate3_real_results_v3_reproduce.json`) must not be quoted against this row — it is a different **representation**, and **[REV02] measured to be the same population**: both artifacts carry `n_query 3,368` / `n_gallery 15,913`, so the 88.64 → 89.91 → 93.379 spread is arm-vs-arm on one evaluation set, not set-vs-set **[REV01]** |
| P4 | TransReID needs `[0.5]³` normalisation, **not** ImageNet stats | a shared ImageNet-stat preprocessor silently handicaps the candidate; this was the root cause of a 22-point Rank-1 error |
| P5 | `evaluate_offline.py:65` filters gallery/query to `id_min=752, id_max=1501` | drops 355 of the 750 official test identities; and because 355 *training* identities also lie in 752–1501, it is **not** a leakage guard |

**[REV02] Execution status of P1–P5.** Each row is a measured statement about this host, not a plan.

| ID | Status | What was actually done / found |
|---|---|---|
| P1 | **NOT FIXED — FROZEN BY CHARTER (§3B), and worse than REV01 recorded** | OSNet was inspected only. Two distinct failures were measured: (a) the `osnet_trt` path returns `np.zeros(512)`; (b) the `osnet_torch` path cannot even import on this host — `osnet_torch_extractor.py:64` `from torchreid.models import build_model` raises `RuntimeError: operator torchvision::nms does not exist` (torchvision build without that op). Its fallback `_build_osnet_standalone()` is a **placeholder**: instantiated in memory it holds **8 entries / 42,945 elements (42,816 trainable + 129 buffers)**, shares **0 of 567** key names with `osnet_x1_0_market1501.pth`, and loads with `missing_keys = 7 / unexpected_keys = 567` under `strict=False` — i.e. it consumes none of the checkpoint. Consequence for Task 7: **there is no valid OSNet arm on this host**, so the paired OSNet-vs-TransReID test is `NOT APPLICABLE` and H7a is **undecidable here**. Recorded, not repaired |
| P2 | **FIXED for the extractor/factory; engine build still blocked** | `models/transreid_trt_extractor.py` created (`f090f477b7896531`) and dispatched by `models/extractor_factory.py` (`e539e6696d3286bd`, §3C shared change: TransReID arm only, four frozen-arm regressions asserted unchanged); `scripts/build_transreid_trt_engine.py` (`45621317687f3356`) and `scripts/benchmark_transreid_jetson.py` (`d7f4b0236ddb5696`) created. **No `.engine` exists and none was built**: TensorRT is absent here and G.7 was denied. The class is unit-tested with a stubbed `tensorrt`/`pycuda`, which is explicitly *not* real inference (§9) |
| P3 | **CLOSED by owner decision G.5 / PF-10c** | Implemented as a **per-robot ROS parameter**, `IdentityVerify.srv` untouched: `identity_node.py:76 declare_parameter('reid_camera_id', 0)` → `core/reid_input_adapter.py:81-102 resolve_camera_label()` → `cam_label = reid_camera_id − 1`, passed to `extract_embedding()` only when the extractor accepts it; `algorithm_thresholds.yaml:54` ships `reid_camera_id: 0` (0 = camera-agnostic, 1–6 = apply SIE). Measured effect on the deployed 512-D arm (see Appendix B): SIE **+4.632 pp Rank-1** (88.747 → 93.379) and **+7.801 pp mAP** (77.324 → 85.125); exact McNemar on per-query Rank-1, Holm-adjusted p = 4.361e-41 → REJECT |
| P4 | **CLOSED and tested** | `[0.5]³` is applied by the TransReID paths; the TRT arm's numpy preprocessing was asserted **bit-identical** to the real torch preprocessing of `TransReIdExtractor` under `ES07_HEAVY=1` (test row 10). Per-arm normalisation is recorded in the embeddings manifest |
| P5 | **CLOSED; REV01's own numbers for this row are wrong in direction** | `evaluate_offline.py` (`e341653663cea613`) now takes an explicit `--population` and writes the population into every output. Measured facts that supersede the row above: Market-1501 identity numbers are **globally unique and non-contiguous per split**, so *no* ID band can express a split — `bounding_box_test` spans ids 0–1501 (751 unique identities incl. distractor `0`), `query` spans 1–1501 (750). The band filter is not merely lossy; it is not a partition. Task 7 used `population = directory` (§5) |

P3 requires an interface decision (add a camera id to the extractor contract) and is **escalated to the supervisor, not decided here**. If it is refused, the candidate arm runs without SIE and that must be stated in the report title, not buried. **[REV02] This paragraph is discharged: it was decided (G.5, PF-10c) and both arms — with SIE and camera-agnostic — were measured, so no title caveat is required.**

**[REV01] P3 alone does not open the deployed path.** Fixing the Python contract is not sufficient: `edgesync_msgs/srv/IdentityVerify.srv` carries **no camera, frame or sensor-id field** anywhere, and `.msg`/`.srv` files are frozen by owner instruction. So even with `cam_label` in the signature, the ROS service request has nowhere to put the camera id. The camera value would have to arrive from a side channel (parameter, topic, or a "query is always camera X" assumption), each of which changes the semantics of the deployed verifier. This is decision **G.5**, and it is a bigger decision than P3 as originally written. See also arm **A2** in `JETSON-RUN-SHEET-PREP-01-REV01.md` §4 — the no-cam arm is the only configuration reachable by the code as it stands.

**[REV02] G.5 resolved, and its semantics recorded exactly as the [REV01] paragraph predicted.** The camera id arrives from a **side channel**: a per-robot launch parameter. That is correct only under the deployment assumption *"one identity node per robot, and every crop that node receives came from that robot's camera"* — which is how `identity_node.py` works, since the service request has no per-query camera field. Consequences that must travel into any report: (a) `reid_camera_id` is a **robot-configuration** fact, not a measurement of the incoming frame; a mis-set parameter silently changes the embedding, so the node logs the resolved label and the reason on startup; (b) the SIE arm is reachable in ROS only when each robot's launch sets its own id — the parameter ships `0`, so the default path stays camera-agnostic; (c) `edgesync_msgs` was **not** modified (`git`-free check by digest; see Appendix C).

## 3. Design

| Concern | Decision |
|---|---|
| Interface | `TransreidTrtExtractor(BaseIdentityExtractor)`, same 512-dim L2-normalised output; OSNet keeps its own extractor |
| Selection | **[REV01]** the parameter that exists on disk is **`reid_model_type`** (`nodes/identity_node.py:57-60`, `config/algorithm_thresholds.yaml:34`, resolved by `models/extractor_factory.py:38-89` with values `osnet_mock` \| `osnet_trt` \| `osnet_torch` \| `transreid`). `reid_backbone` does **not** exist and is not declared by any node. Two further facts bind this row: (a) `launch/robot_bringup.launch.py:79-95` passes a **dict after the YAML** in `parameters=[...]`, so the launch dict silently overrides `algorithm_thresholds.yaml:34`; (b) `algorithm_thresholds.yaml:34` ships `osnet_mock`, whose extractor returns a pseudo-embedding from crop channel means (`models/osnet_trt_extractor.py:89-90`). Any ablation must select via the launch argument (`launch/robot_bringup.launch.py:22-29`) or via a documented override, never by editing the YAML alone |
| Runtime | **[REV01]** planned shared `TrtEngineRunner` — this class **does not exist on disk** and no engine artifact exists either; it is a deliverable of Part 3, not a precondition available today. Until then the only arms that can actually run are `osnet_mock` (mock), `osnet_torch` and `transreid` via PyTorch (`extractor_factory.py:38-89`), i.e. **the ablation as written is a laptop experiment until P1/P2 are closed** |
| Preprocessing | One shared `ReidPreprocessor` for geometry (256×128, NCHW float32) and **per-arm normalisation constants**, which are logged into every result file (`transreid = [0.5]³`, `osnet = ImageNet`) |
| Feature head | TransReID: 768-D post-BN-Neck → centered SVD 512; projection applied **inside the engine graph** (folded as a linear layer) or after it, recorded as `projection_stage`, identical across all trials |
| Camera embedding | SIE `xishu = 1.0` with a per-query camera id if P3 is approved; otherwise `sie_applied = false` recorded in every file |
| Configuration | **[REV01]** `ablation_task7_osnet.yaml`, `ablation_task7_transreid.yaml` under `edgesync_bringup/config/` — identical except model type, weights, projection and normalisation. They must use the on-disk key names (`reid_model_type`, `reid_checkpoint_path`, `reid_projection_path`, `reid_engine_path`), not new ones, and must state that `config/` is installed by `setup.py:11-16` while `models/` and `results/` are **not** |
| Launch | **[REV01]** `task7_identity_ablation.launch.py model_type:=<arm> engine_path:=<path>` — arg names mirror the existing `robot_bringup.launch.py:22-29` args, and the launch **dict must not re-override** the arm's own YAML (see Selection row) |
| Measurement | `edgesync_identity.ablation` — latency, Rank-k, mAP, memory, cross-robot metrics, comparison |
| Messages | `edgesync_msgs` is **not** modified |
| Artifacts | versioned filenames only; no baseline JSON or baseline `.npz` is overwritten |

**[REV02] What was built against that table, row by row.** The table above is the design of record and was **not** edited; this block states the as-built facts.

| Concern | As built (measured) |
|---|---|
| Interface | Delivered. `TransreidTrtExtractor` emits float32[512], unit-norm, returns the `zeros(512)` sentinel on failure and never raises, per the frozen contract in `interfaces/base_identity_extractor.py`. 34 tests + 1 gated, and under `ES07_HEAVY=1` 35 pass with the preprocessing cross-check |
| Selection | Unchanged mechanism, extended value list: `transreid_trt` added to `reid_model_type` in the factory, the node, and the YAML comment. No `reid_backbone` was introduced. The launch-dict override hazard (a) remains and was **not** "fixed" — changing which of the two parameter sources wins would alter OSNet and unrelated behaviour, which §3B/§3C forbid; instead the launch path now derives its defaults from `edgesync_bringup/reid_launch_defaults.py` (`6a8f85dc5748c9bb`) so an arm selected in the YAML is not silently replaced |
| Runtime | `TrtEngineRunner` was **not** created; the engine call lives inside `transreid_trt_extractor.py`. **`return np.zeros(512)` is not used as a fallback anywhere in the new code**: a missing/incompatible engine raises at construction and the arm is `BLOCKED`, not faked. The ablation therefore remains a laptop experiment exactly as [REV01] predicted, because P1 (OSNet) is unfrozen-by-charter and no engine exists |
| Preprocessing | Shared geometry + per-arm constants, asserted bit-identical (P4 row above) and recorded in the manifest for all 19,281 extracted images |
| Feature head | **`projection_stage = "post_graph_npz"`.** The REV01 alternative "inside the engine graph (folded)" was **not** taken: the exported graph ends at the post-BNNeck 768-D CLS feature and the SVD 768→512 map is applied by numpy after inference. Reason: the projection is a fitted artefact with its own provenance (§2) and folding it into an ONNX graph would produce a matrix with no hash lineage. Identical for every trial and every arm; recorded in the manifest and re-read by the statistics consumer |
| Camera embedding | `sie_applied` is now a **per-arm measured fact, not a flag**: three arms were extracted and evaluated — `transreid_768_agnostic`, `transreid_512_agnostic`, `transreid_512_sie` — so both sides of the G.5 decision are quantified (Appendix B) |
| Configuration | `ablation_task7_osnet.yaml` / `ablation_task7_transreid.yaml` were **NOT created**, and `config/` still contains no ablation YAML. Deliberate: creating an OSNet arm config implies an OSNet arm that cannot run (§3B). The TransReID arm is reachable today through `algorithm_thresholds.yaml` (`3576df5a194e491b`) + the launch defaults module. `setup.py:11-16` was **not** changed, so `models/`, `data/` and `results/` are still not installed |
| Launch | `task7_identity_ablation.launch.py` was **NOT created**. `launch/robot_bringup.launch.py` (`64749d07e4efb8f5`) was extended with the TransReID parameter set instead. Launch-level verification is **BLOCKED on this host**: the team's own `tests/test_robot_bringup_launch.py` fails at `from launch import LaunchDescription → ImportError`, so no launch file in this workspace can be executed or parsed here |
| Measurement | Delivered as `scripts/analyze_task7_statistics.py` (`6c0cb21fd30c3975`, 61,798 B) + `tests/test_analyze_task7_statistics.py` (`4ac045e75caf7515`, 41,031 B, 68 tests). It is fail-closed by construction: SHA-256 integrity gate on every input `.npz` against the manifest, τ read from the YAML (it **refuses** to run if τ ≠ 0.70), manifest-vs-array agreement gate, and an assertion that the per-query aggregates equal the canonical `run_real_pipeline.compute_cmc_map` result before any number is printed |
| Messages | Confirmed unchanged: `edgesync_msgs` appears in no changed-file row and `IdentityVerify.srv` was never opened for write |
| Artifacts | Confirmed: no baseline JSON or `.npz` was overwritten. Every Task 7 output is a new versioned file outside the repo (`C:\Users\PC\edgesync_es07_runs\…`) or a new document at the project root |

## 4. Variables

**Independent:** backbone (OSNet x1.0 vs TransReID ViT-B/16 + SVD 512). **[REV02] Only the TransReID side was varied and measured.** The three arms actually evaluated are a *within-backbone* family (768-D raw / 512-D projected camera-agnostic / 512-D projected with SIE); the backbone axis is `NOT MEASURED` on this host for the P1 reasons above.

**Controlled:** input resolution, geometry preprocessing, FP16, TensorRT version, power mode and clocks (`nvpmodel` mode 0, `jetson_clocks`), threshold 0.70, spatial gating, evaluation data, query/gallery definition, camera-to-robot split, iteration counts, engine build flags, `projection_stage`. **[REV02] Of these, FP16 / TensorRT version / power mode / clocks / engine build flags were never exercised** (no device, no engine). Threshold (0.70, read from YAML and asserted), evaluation data, query/gallery definition (population `directory`), camera split and `projection_stage` were controlled and recorded in every output file.

**Dependent:** Rank-1/5/10, mAP, latency (p50/p95/p99), throughput, **engine size (MiB)**, **resident memory delta (MiB)**, peak memory during the run, cross-robot false-match metrics. **[REV02] Accuracy and false-match metrics: COMPUTED (Appendix B). Latency, throughput, engine size and every memory figure: `NOT MEASURED — Pending Jetson Orin Nano hardware access`.**

## 5. Data and training protocol

- **Dataset:** Market-1501 v15.09.15, official release. Provenance pinned: `Market-1501-v15.09.15.zip`, 152,733,771 B, sha256 `416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96`; proven byte-identical to the in-repo copy (member-name sets identical, 200/200 sampled bytes identical in each of the five folders). Research-only licence per the shipped `readme.txt`.
- **Split rule:** partition membership is defined by **folder**, never by identity number. Measured on disk: `bounding_box_train` 12,936 images / 751 identities (ids 0002–1500), `bounding_box_test` 19,732 / 750 identities (0001–1501) + 3,819 junk (`-1`) + 2,798 distractors (`0000`), `query` 3,368 / 750. train ∩ test = **0** identities; query ≡ test identity set.
- **Evaluation set (pre-registered, identical to the Gate-3 baseline so results stay comparable):** `n_query = 3,368`; gallery = `bounding_box_test` minus `pid -1` junk = **15,913**, with `pid 0000` distractors **retained** and reported as a separate ablation row if the owner prefers their removal.
- **Training source:** the SVD projection (and any fine-tuning or head training) uses the **training partition only**; test subjects are never seen. No threshold, projection rank, or hyper-parameter is tuned on the evaluation set.
- **Baseline weights:** record the exact URL, download date, sha256 and documented training data of `osnet_x1_0_market1501.pth` and `transformer_120.pth` in the report; identity-level exposure to the evaluation partition is argued from folder membership and must be stated as such.
- **Custom data:** when the Mobile UGV Target Handoff Dataset is ready, convert it to the same `<pid>_c<cam>s<seq>_<frame>.jpg` convention and re-run with no code change; splits by subject.
- **Robot split:** cameras 1–3 = Robot A, cameras 4–6 = Robot B; each query is matched only against the **other** robot's gallery, mirroring a cross-robot handoff. The split is written into every result file.

**[REV02] Execution of this protocol, with two corrections.**

1. **The evaluation set actually used matches the third bullet exactly, and was asserted from the arrays rather than assumed:** gallery scanned 19,732 → kept 15,913, excluded 3,819 junk (`pid -1`), **2,798 distractors retained**, 751 unique gallery identities (750 real + `pid 0`), ids 0–1501; query 3,368 kept / 0 excluded, 750 identities, ids 1–1501; 0 junk rows and 0 zero-norm rows in either array; every query had at least one impostor candidate. This supersedes the second bullet's "750 identities" reading of the test folder only when the distractor identity is counted — the honest figure for the retained gallery array is **751 unique pid values including `0`**. No standard dataset count was assumed anywhere; every number above came from the walk in this run.
2. **`population = directory` is the only valid label; `id_band` is invalid as a *rule*, and — measured — it was not the population of the recorded accuracy artifacts either.** `evaluate_offline.py` now emits both and records which was used. REV01 and this revision's earlier draft both assumed the recorded Gate-3 numbers were band-filtered; reading the artifacts themselves corrects that: `gate3_real_results.json`, `gate3_real_results_v2_corrected.json`, `gate3_real_results_v3_reproduce.json` and `gate3_real_results_reconciled.json` all carry **`n_query = 3,368`, `n_gallery = 15,913`** — the same directory population this experiment used. The `id_min=752 / id_max=1501` rule is therefore **not** the provenance of the recorded accuracy rows; it survives only as an opt-in, self-labelling population (the rule is registered at `evaluate_offline.py:84` and applied at `:155`; `DEFAULT_POPULATION` is `'directory'` at `:99` and is used at `:112,473`; the chosen name is written into the manifest at `:175` and stored in every output at `:572`) that would drop 355 of 750 test identities if selected — REV01's P5 wording ("`evaluate_offline.py:65` filters …") describes the pre-fix file and overstates its current reach. Two consequences, both favourable to evidence quality and one unfavourable to a paired test:
   - today's `transreid_512_sie` arm is a **REPRODUCTION** of the recorded TransReID row, not a new number on a different set (see Appendix B note);
   - a **descriptive** comparison against the recorded OSNet aggregate is arithmetically legitimate (same query/gallery counts), but the OSNet arm still cannot run on this host and no per-query OSNet outcomes exist in any artifact, so the **paired** test remains `NOT APPLICABLE` (charter §8).
3. **Embedding extraction was performed on the corrected code paths, not on a mock:** 19,281 rows (3,368 query + 15,913 gallery) × three arms, `transformer_120.pth` sha256 `843055293e160da0…`, GPU PyTorch (FP32), `[0.5]³` normalisation, 256×128, `projection_stage = post_graph_npz`, elapsed 956.4 s, manifest sha256 `283c8c782e66869a…` (7,352 B). Per-image contract checks passed for all rows: 512-D, finite, unit-norm to max |‖x‖−1| = **1.7881e-07**; the 768-D array is deliberately *not* unit-norm (max deviation **37.398**) because it is raw post-BNNeck, and cosine is taken after normalisation. Full detail: `TASK7_DATA_PROVENANCE.md` §D-04.
4. **Custom data / Mobile UGV handoff dataset: does not exist.** No robot, handoff event, trajectory or handoff label exists in this project, so §5's cross-robot handoff protocol is `NOT MEASURED and NOT INFERABLE` — the camera-partition figures in Appendix B are labelled **PROXY** and are not handoff measurements (charter §8).
5. **[REV02] Configuration agreement, measured.** `gate3_real_results_v3_reproduce.json/pipeline_config` records `transreid_pixel_mean/std = [0.5]³`, `transreid_sie_applied = true`, `transreid_sie_xishu = 1.0`, `transreid_neck_feat = after (post-BNNeck)`, `image_size_hw = [256,128]`, `svd_variance_retained = 0.9864118099212646`, `device = cuda` — identical on every point to the arm configuration this experiment extracted. That independent agreement, plus the digit-for-digit reproduction in Appendix B, is the strongest pipeline-correctness evidence available without the device.

## 6. Metrics

| Metric | Definition |
|---|---|
| Rank-1/5/10, mAP | Market-1501 protocol: junk (`-1`) ignored; same identity **and** same camera excluded from a query's positives |
| Latency | **[REV01]** Per-call wall time after **50 warm-up calls over 200 timed calls**, **3 independent trials**, batch 1, `torch.cuda.Event` timing — i.e. the harness already documented in `scripts/run_real_pipeline.py::step4_latency` and locked in `JETSON-RUN-SHEET-PREP-01-REV01.md` §2. v01 said 1,000 calls here while every artifact on disk used 200; p50, p95, p99, mean, std and trial spread reported |
| Stages | `end_to_end` (preprocess + inference + projection + normalise) and `engine_only` (TensorRT call) |
| Throughput | 1,000 / mean latency (ms) |
| Engine size | Serialized `.engine` file size (MiB) |
| **Resident memory Δ** | **[REV01]** Orin Nano has **unified memory**, so `nvidia-smi` is not a footprint read. Use `tegrastats` MEM fields plus the process's own counters: `VmRSS` from `/proc/<pid>/status`, `MemAvailable` from `/proc/meminfo`, and `torch.cuda.max_memory_allocated()` — read immediately before engine load and after warm-up; Δ = after − before in MiB, median of 3 trials, raw per-trial values reported. Reported separately from `peak during run`. Laptop `peak_runtime_gpu_memory_mb` is **not** accepted as an edge figure |
| **Declared pair population** | Genuine pairs for FNMR = same-identity pairs whose gallery match lies in the **other robot's cameras**; impostor pairs = different-identity pairs across the two robots. Both counts are written into the result file. **Any FNMR quoted without its population is invalid** |
| Pair false-match rate | Impostor cross-robot pairs with similarity ≥ 0.70 / all impostor pairs |
| False-handoff rate | Queries whose best cross-robot candidate is a different identity with similarity ≥ 0.70 |
| FNMR at 0.70 | Share of declared genuine pairs scoring below 0.70 — the deployed operating point |
| Matched-TAR FMR | FMR at the threshold where true-accept rate = 0.90 |
| EER | Equal error rate over the declared cross-robot pairs |

Matched-TAR FMR and EER separate a real discrimination loss from a similarity-scale shift, because 0.70 is not calibrated per backbone. Reference: on the sampled population already measured, TransReID has the **better** separation (FMR 0.014 % vs 0.114 %, EER 1.678 % vs 2.052 %) and the **worse** 0.70 operating point (FNMR 38.51–48.66 % vs 18.70 %) — a calibration effect, which is exactly the distinction this table is designed to make measurable.

**[REV02] As computed.**

- **Latency, Stages, Throughput, Engine size, Resident memory Δ, Peak: `NOT MEASURED — Pending Jetson Orin Nano hardware access` (G.7 denied).** The harness for them is delivered as `scripts/benchmark_transreid_jetson.py` and remains unexecuted; no laptop number is entered in their place (§8 prohibition).
- **Declared pair population — the honest correction.** The headline FNMR/FMR/EER table uses the **cross-camera, duplicate-excluded** pool over the whole official test population: genuine = same-identity pairs whose gallery row is in a *different camera* (same-identity **and** same-camera rows are the same physical appearance captured twice, excluded from both pools; junk excluded; distractors `pid 0` retained as impostor candidates, because a false accept against a distractor is exactly what the threshold exists to prevent). Measured counts: **49,649 genuine / 53,532,220 impostor / 13,115 duplicate pairs excluded / 0 junk rows / 0 queries without an impostor candidate**. This is **not** REV01's literal "other robot's cameras" pool; the robot-split version was computed separately as the camera-partition block in Appendix B and is labelled PROXY. Both are reported so the reader can see which population each figure belongs to, which is the rule this row exists to enforce.
- **τ = 0.70 was verified to be the τ actually applied**, not assumed: read live from `algorithm_thresholds.yaml` key `algorithm_thresholds.ros__parameters.reid_similarity_threshold`, asserted `== 0.70`, and the acceptance rule replicated from production (`core/identity_verifier.py:101` rejects when `sim < τ`, so **accept iff `sim ≥ τ`**). Similarities are computed with the production matrix path and cross-checked against `IdentityVerifier.compute_cosine_similarity` on 200 sampled pairs: max abs difference **1.7881e-07** (float32 vs float64 reduction order). Locked boundary case: `float32(0.7)` is `0.699999988`, so a score of exactly float64 0.70 is accepted while the float32 rounding is what the node compares against — this is a test, not a discussion.
- **EER / τ@TAR solvers respect the discreteness of the score pool.** An early version used continuous bisection and reported an EER for a threshold that accepts no actual pair; it now solves over the observed score set with a bounded bracket search. The cost was measured, not assumed: a naive full enumeration of distinct scores (53.5 M impostor values per arm) timed out and that run was stopped at ~4 min; its output directory is kept only as `statistics_smoke/` scratch and is **explicitly not results**. With the bounded solver the full three-arm analysis completes in **123.3 s** (768-agnostic 46.3 s · 512-agnostic 39.0 s · 512-sie 38.0 s) and a single arm in 30.7 s, with identical numbers for the shared arm to the last digit. The test suite carries a 2 M-score cost guard (< 20 s) so this cannot silently regress again.
- **Scope of every verification figure: appearance gate only.** The Mahalanobis spatial gate (`identity_verifier.py:114`, cutoff 9.21) is `NOT MEASURED` — no trajectory or covariance data exists offline.

## 7. Statistical plan (fixed before data collection)

- Paired exact McNemar on per-query Rank-1 correctness and on per-query false-handoff flags.
- Wilson 95 % intervals on the false-handoff rate and on FNMR.
- α = 0.05 with Holm correction across the two primary paired tests.
- Latency reported as percentiles; no distributional assumption. A run is discarded and repeated if `tegrastats` shows throttling or if trial spread exceeds 5 %.
- Memory Δ reported as median of 3 trials with the raw per-trial values.
- Accuracy effect sizes are reported in percentage points with their CIs; **no pass/fail bound is set on them.**

**[REV02] How this plan was executed, and the one place it could not be.**

- **The family was declared before analysis and is *within-TransReID*, because no OSNet arm exists on this host (P1).** Three pre-registered hypotheses, each tested on 3,368 paired query units: **H1** per-query Rank-1 hit, **H2** per-query genuine-accept at τ, **H3** per-query false-accept at τ — all `512_agnostic` vs `512_sie`. Holm–Boferoni monotone step-down over a family of **3** (not 2), α = 0.05. Results in Appendix B.
- **OSNet-vs-TransReID paired test: `NOT APPLICABLE`.** Charter §8 permits a paired test only where valid paired query-level outcomes exist for *both* arms, and §3B forbids altering OSNet to make the comparison possible. Recorded as `RuntimeError: operator torchvision::nms does not exist`; no substitute, no imputation, no re-derived baseline.
- **Wilson 95 %** on Rank-1 and on every rate; **percentile bootstrap (seed 20261019)** on mAP; exact McNemar via `math.comb`; percentile reporting with no distributional assumption; latency/memory rows unexecuted, so their discard rules stand unused.
- **Canonical-metric agreement is asserted, not assumed:** the consumer recomputes CMC/mAP per query and asserts equality with `run_real_pipeline.compute_cmc_map` (validated against torchreid 0.2.5) before reporting; measured difference 0.0 on all three arms.

## 8. Decision rules (do not edit after data collection)

- **H7a — accuracy increases:** candidate mAP > baseline mAP. Rank-1, Rank-5, Rank-10 are reported individually, not folded in. Reference measurement on the laptop already shows the two can disagree: mAP +1.556 pp while **Rank-1 −0.950 pp** (`gate3_real_results_v3_reproduce.json`).
- **H7b — latency breaks the handoff budget:** candidate `end_to_end` p95 exceeds the identity stage's share of `T_txn = 886 ms` (**[REV01]** full path: `System Design Phase/4_Math_Protocol_ROS2_v2.3.html:261`, verified to contain `T_txn = 886 ms | T_claim = 200 ms | T_ack = 100 ms | Negotiation Timeout = 80 ms`; machine-readable copy `edgesync_bringup/config/timing_budgets.yaml:8,11,15,18,19`). The identity share is fixed by the owner **before** collection. The charter's "200 ms" is not a budget documented anywhere in the design set and is not used as a gate. Quality Gate 5 (OSNet-class ≤ 18 ms on Orin Nano) is reported for both arms.
- **H7c — memory is prohibitive:** candidate resident Δ plus the concurrent perception stack exceeds the envelope the owner fixes before collection for the 8 GB device. Unset here deliberately; `UNVERIFIED` until measured.
- **"OSNet is the optimal trade-off" may be concluded only if OSNet wins at least two of the three axes** (accuracy, latency, memory) with the latency and memory axes decided by **edge** measurements. Otherwise the verdict is *trade-off unresolved*.
- **Prohibition carried from the audit:** `gate3_real_results_reconciled.json` currently issues a production recommendation ("Reject TransReID … inferior edge efficiency") on the same page that records `jetson_orin_nano_status: NOT MEASURED`. No report in this experiment may repeat that pattern; a device-dependent conclusion requires device data.

**[REV02] Application of the rules — the text above is unchanged, as required.**

| Rule | Verdict on the evidence available | Why |
|---|---|---|
| H7a | **`NOT VERIFIED BY MEASUREMENT HERE` — descriptively supported against a recorded baseline whose provenance is unverified** | The rule is "candidate mAP > baseline mAP". The candidate side is measured (85.1252, `transreid_512_sie`). The baseline cannot be produced on this host (P1: `torchvision::nms` import failure / placeholder backbone consuming 0 of 567 keys / `osnet_trt` `zeros(512)` / `osnet_mock` channel means), so the only baseline available is the recorded aggregate **94.3290 Rank-1 / 83.5692 mAP** — which the artifact's own `n_query = 3,368`, `n_gallery = 15,913` confirm is **the same population** as today's measurement. Read against it: mAP **+1.5560 pp** (candidate higher ⇒ the literal rule is satisfied) while **Rank-1 −0.9501 pp** (candidate lower), exactly the mAP/Rank-1 disagreement §8 warns about. That is **not** a verification: the OSNet figure was produced by an extractor not named in the artifact and is not reproducible here, and §8's paired-test route is closed because no per-query OSNet outcomes exist anywhere on disk. So the honest verdict is "supported arithmetically against a record, unverified by measurement" |
| H7b | **UNVERIFIED — `NOT MEASURED`** | No engine, no device, and the identity-stage share of `T_txn` (G.1) was never fixed. Quality Gate 5 cannot be applied to either arm |
| H7c | **UNVERIFIED — `NOT MEASURED`** | No resident Δ (G.7 denied) and no envelope (G.3 open) |
| "OSNet is the optimal trade-off" | **May NOT be concluded. Verdict: `TRADE-OFF UNRESOLVED`** | Two of the three axes have no edge measurement and the third has no baseline. This is the only conclusion the rules permit here |
| Prohibition compliance | **Held** | Neither `ES-EXP07-TASK7-REPORT-01-REV02.md` nor this document issues a production recommendation; every device-dependent cell says `NOT MEASURED` and names the gate that unblocks it |

What *can* be stated from measured data, without touching any rule: within TransReID, on the official test population, the SIE camera embedding is worth **+4.632 pp Rank-1 / +7.801 pp mAP** over the camera-agnostic 512-D arm (Holm p = 4.361e-41, REJECT) and **+7.601 pp genuine-accept at τ = 0.70** (Holm p = 2.427e-72, REJECT), while the false-accept difference is **not** distinguishable from zero (Δ −0.24 pp, Holm p = 0.5728). And the calibration effect REV01 anticipated is real and now measured: all three arms have **better separation than the recorded OSNet reference** (EER 1.52–2.01 % vs 2.052 %) yet **worse operation at the shipped τ** (FNMR 48.15–60.74 % vs 18.70 %) — each EER sits at τ ≈ 0.31–0.37, i.e. roughly **0.33 below** the deployed 0.70. On this evidence the 0.70 gate, not TransReID's discrimination, is the binding constraint. §8 forbids this report from changing τ; it is stated as the owner's calibration decision to make.

## 9. Threats to validity

1. **ViT FP16 numerics on TensorRT** — LayerNorm/attention precision may degrade TransReID more than OSNet; report FP16 for both and keep an FP32 accuracy reference row off-device.
2. **Engine export fidelity** — SIE, the BN-Neck and the SVD projection must survive ONNX export; the exported graph is hash-recorded and its output is compared against the PyTorch path on ≥ 200 fixed images (max abs deviation reported).
3. **Training-recipe mismatch** — OSNet weights and the TransReID checkpoint come from different recipes; recorded, not equalised.
4. **Domain gap** — Market-1501 is a proxy for indoor UGV scenes.
5. **Threshold calibration** — 0.70 was chosen for OSNet; the matched-TAR row addresses it.
6. **Thermal/clock state** — clocks locked, `tegrastats` logged, throttled runs discarded.
7. **Single seed** — seed reported; multi-seed only if time permits.
8. **Camera split is a modelling choice** — recorded in every result file.
9. **Memory metric ambiguity** — FP32 parameter bytes, checkpoint bytes, FP16 engine bytes and on-device resident Δ are four different numbers (already measured: 384.405 vs 8.276 MB = 46.4×; 395.512 vs 9.918 MB = 39.9×; resident Δ unknown). Every claim names which one it uses.

**[REV02] Observed status, plus two new threats this experiment created.**

| # | Threat | Status |
|---|---|---|
| 1 | FP16 numerics | **OPEN / NOT MEASURED** — everything measured here is FP32 PyTorch; no FP16 row exists for either arm |
| 2 | Export fidelity | **PARTIALLY MET.** Graph hash-recorded, checker PASS, dynamic batch exercised at 1 and 2 (output (1,768)/(2,768), width observed 768). Torch↔ONNX parity: worst max-abs 768-D **1.2398e-05** (agnostic) / **1.2875e-05** (SIE), worst post-contract 512-D **6.091e-07** / **6.855e-07**, cosine ≥ 0.999999999995, tolerance 0.002, cosine floor 0.9999, verdict PASS. **Deviation from this row: parity ran on `parity_n = 8` fixed images, not the pre-registered ≥ 200.** The ≥ 200-image comparison must be re-run against the *engine*, which does not exist yet |
| 3 | Recipe mismatch | Recorded, unchanged (§W-01: `transformer_120.pth` is an external artefact — no URL, download date or training-data statement exists on disk anywhere; the checkpoint's own 211 tensors and five `(751,768)` classifier heads are the only evidence of a 751-identity training set) |
| 4 | Domain gap | Unchanged and unmeasurable here: no UGV imagery exists in the project |
| 5 | Threshold calibration | **Now measured** (§8): the τ-vs-EER gap is ≈ 0.33 for every arm |
| 6 | Thermal/clock state | **NOT APPLICABLE offline**; required on the device |
| 7 | Single seed | Seed 20261019 used for the bootstrap only; no stochastic model fitting was performed. Multi-seed never exercised |
| 8 | Camera split | Recorded in every output; the split is used as a proxy and labelled as such |
| 9 | Memory ambiguity | Four quantities were listed and only some are re-measurable today. **Reconciled by measurement:** Appendix A's "checkpoint 395.512 MB" is the file (414,724,314 B = **395.51 MiB**) and "FP32 parameter memory 384.405 MB" is 100,769,512 trainable elements × 4 B = 403,078,048 B = **384.41 MiB** — both are **MiB values printed under an "MB" label**, and their 11.11 MiB difference is the five `(751,768)` aux classifier heads plus 7,685 buffer elements (11.03 MiB of tensors) plus ~0.08 MiB of container overhead (`TASK7_WEIGHTS_PROVENANCE.md` §W-01). So the two Appendix A figures are **not** a contradiction — they are two definitions of "memory", exactly what this threat warns about. **New quantity measured:** the ONNX graphs hold 343,249,920 B (agnostic) / 343,268,352 B (SIE) of float32 initialisers = **327.35 / 327.37 MiB**, i.e. 57.06 MiB *less* than the trainable-parameter figure; that gap was **not decomposed in this run and is recorded as an open item**. **Still unknown:** FP16 engine size and on-device resident Δ (no engine, no device). **Not re-measurable here at all:** the OSNet side of the original row (`torchvision::nms`) |
| **10 (new, then corrected)** | **Population label — the assumed discontinuity does not exist; the real hazard is the opposite** | REV01 and this revision's earlier draft assumed every recorded number on disk was `id_band`-filtered. **Measured and falsified:** all four Gate-3 accuracy artifacts carry `n_query = 3,368` / `n_gallery = 15,913`, i.e. the same `directory` population, and today's `transreid_512_sie` **reproduces** `gate3_real_results_v3_reproduce.json → accuracy/transreid` (rank1/5/10 identical to the last digit; mAP Δ 2.38e-10). Because the populations *agree*, a recorded cell now *looks* differencable when its arm, extractor or pair-pool definition does not — which is the trap. Mitigation: population, arm and pair-pool are stated in every table header here and in the report; differences are taken only between same-definition cells. **Measured state of the `id_band` rule after the §5-TR-07/B-12 fix:** it is no longer a default — `DEFAULT_POPULATION` is `'directory'` (`evaluate_offline.py:99`, used at `:112,473`), and the band rule survives only as an opt-in, self-labelling choice (`id_band_752_1501`, registered at `:84`, applied at `:155`, and echoed into every output as the manifest's `population` field at `:175` → `:572`). The residual hazard is therefore not a silent default but a *recorded* one: any pre-fix artifact produced with the band still carries the same field names as a directory-population artifact, so the label must be read, not assumed |
| **11 (new)** | **Same-checkpoint evaluation** | The candidate arm is evaluated on the official test split using a checkpoint whose training exposure is documented only by folder algebra and its own head shapes; no independent re-training was permitted (§3B/§5). Identity-level exposure is *argued*, never measured |

## 10. Procedure

1. Workstation: `tools/ablation/task7/requirements.txt`; export TransReID + projection to ONNX with `export_transreid_reid.py`; run the PyTorch-vs-ONNX parity check; re-fit the projection on the training partition only and write `svd_projection_768_to_512_task7_v1.npz` with its R₅₁₂ and sha256. **[REV01]** name the *input* projection by hash, not by filename: the inventory found six byte-identical duplicate groups among the SVD assets (`svd_projection_768_to_512_real.npz` ≡ the un-suffixed file; `_real_v2.npz` ≡ `_real_v3_reproduce.npz`; three names → only two distinct contents). The re-fit seed asset is `svd_projection_768_to_512_real_v2.npz`, sha256-16 `9626bea5526df5b0` (pinned in the run sheet §3).
2. Jetson: `tools/ablation/task7/build_engine.sh <onnx> <engine>` for both arms (FP16, batch 1). **[REV01]** blocked by P1/P2 and by the missing `tensorrt`/`pycuda` declarations; nothing on disk can build an engine today.
3. Jetson: build the workspace (`colcon build`, never done on this tree — `ws/build|install|log` absent), source it, and set the arm through **ROS parameters that the node actually declares** — `reid_model_type`, `reid_engine_path`, `reid_checkpoint_path`, `reid_projection_path` (`nodes/identity_node.py:57-62,70-75`). **[REV01]** v01 said `OSNET_ENGINE`/`TRANSREID_ENGINE`/`reid_backbone`: no env variable is read by any node and neither `reid_backbone` exists.
4. Copy `comparison.md` and the JSON files into `ES-EXP07-TASK7-REPORT-01-REV01.md`, into new versioned result files only.

**[REV02] Procedure as actually followed.**

1. **Partially executed, deliberately deviating in two named ways.** (a) `tools/ablation/task7/requirements.txt` was not created; dependencies were installed user-level on the owner's authorization (PF-10a) and pinned in `TASK7_WEIGHTS_PROVENANCE.md` §W-06. (b) The exporter exists as `scripts/export_transreid_onnx.py` (`65c74dce235718d2`), not `export_transreid_reid.py`, and the graph ends at 768-D — the projection is **not** folded in, so it stays hash-verifiable (`projection_stage = post_graph_npz`). (c) **The projection was NOT re-fitted and `svd_projection_768_to_512_task7_v1.npz` was NOT created**: charter §5 forbids a silent refit, §W-02 shows the current script cannot reproduce the deployed key set, and its fitting band is wrong (§2). The deployed matrix was consumed hash-pinned. Parity check executed; PASS at 8 images (see §9 row 2 for the shortfall).
2. **BLOCKED — Pending Jetson Orin Nano hardware access.** Not even reachable from this host: `tensorrt`/`pycuda` absent, G.7 denied. `scripts/build_transreid_trt_engine.py` is delivered, compiles, and is unit-tested against a stub. `build_engine.sh` was not created because a shell script cannot run the TensorRT parser on a machine that has no TensorRT.
3. **BLOCKED — Pending Jetson Orin Nano hardware access.** `colcon` absent, `ws/build|install|log` still missing, and launch parsing fails on this host (`test_robot_bringup_launch.py` → `ImportError: LaunchDescription`). The parameter names in this step are confirmed correct and extended in situ: `reid_model_type` accepts `transreid_trt`, and `reid_camera_id` (G.5) was added alongside the four existing asset paths.
4. **Executed with a stronger source than `comparison.md`.** REV02's numbers were produced by `scripts/analyze_task7_statistics.py`, which fail-closes on hash/manifest/τ mismatch, and copied into `ES-EXP07-TASK7-REPORT-01-REV02.md`. All artifacts are new versioned files; no baseline JSON or `.npz` was overwritten.

Additional steps this experiment required and REV01 did not anticipate — each delivered with its own tests, recorded in `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §4: explicit population selection in `evaluate_offline.py` (P5), the input-adapter/camera-label path (P3/G.5), full-batch embedding extraction on the corrected paths (19,281 rows × 3 arms, `integrity.match = true` on all three), and the statistics consumer plus its 68-test suite.

## 11. Deliverables

Source: `edgesync_identity/{runtime,factory,ablation}`, `models/transreid_trt_extractor.py`, shared constants/preprocessor/math, node parameter **[REV01]** `reid_model_type` (existing, extended) — not a new `reid_backbone` parameter. Config and launch: two arm YAMLs and `task7_identity_ablation.launch.py` in `edgesync_bringup`. Tests: metrics, factory, extractor, parity, benchmark/compare, node selection, launch. Tools: export, engine build, benchmark runner. Docs: this proposal, `ES-EXP07-TASK7-REPORT-01.md`, `TASK7_DATA_PROVENANCE.md`, `TASK7_WEIGHTS_PROVENANCE.md`.

**[REV02] Deliverable ledger.** The list above is the design of record; this is what exists on disk with digests. Full 27-row table with pre-edit backups: `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §4.

| Deliverable | Status | Identifier |
|---|---|---|
| `models/transreid_trt_extractor.py` | DELIVERED, stub-tested only | `f090f477b7896531` |
| `models/extractor_factory.py` (TransReID dispatch, §3C shared) | DELIVERED, OSNet arms regression-asserted unchanged | `e539e6696d3286bd` |
| `core/reid_input_adapter.py` + node `reid_camera_id` | DELIVERED | `14ac3d4358ee2d6e`, node `2d4890746f3c8d29` |
| `scripts/export_transreid_onnx.py` | DELIVERED + EXECUTED twice, byte-reproducible | `65c74dce235718d2` |
| `scripts/build_transreid_trt_engine.py` | DELIVERED, **UNEXECUTED** (no TensorRT) | `45621317687f3356` |
| `scripts/benchmark_transreid_jetson.py` | DELIVERED, **UNEXECUTED** (no device) | `d7f4b0236ddb5696` |
| `scripts/analyze_task7_statistics.py` (+ its 68 tests) | DELIVERED + EXECUTED on real embeddings | `6c0cb21fd30c3975` / `4ac045e75caf7515` |
| Tests: metrics, factory, extractor, parity, node selection | DELIVERED, 303 passed / 2 skipped, exit 0 | §5 rows 21–24 |
| Tests: **launch** | **BLOCKED on this host** — `launch` package not importable | §5 row 5/14 |
| Tests: **benchmark/compare** | **NOT MEASURED** — needs the device | — |
| Config: `ablation_task7_osnet.yaml`, `ablation_task7_transreid.yaml` | **NOT CREATED** (rationale §3, as-built table) | — |
| Launch: `task7_identity_ablation.launch.py` | **NOT CREATED**; `robot_bringup.launch.py` extended instead | `64749d07e4efb8f5` |
| Docs: this proposal (REV02), `ES-EXP07-TASK7-REPORT-01-REV02.md` | DELIVERED | see Appendix C |
| Docs: `TASK7_DATA_PROVENANCE.md`, `TASK7_WEIGHTS_PROVENANCE.md` | DELIVERED | `0ab868cbc84a3bc9`, `acb1f73e865ccd4b` |
| Docs: `ES-EXP07-TRANSREID-PREFLIGHT-REPORT.md`, `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` | DELIVERED, maintained | `928cbf74d7a512a5`, recomputed in the final manifest |
| Docs: `ES-EXP07-TASK7-PROPOSAL-01.md` / `ES-EXP07-TASK7-REPORT-01.md` (v01) | **ABSENT ON DISK** — see the provenance correction in the header | asserted `aa9f5f7dbf9d9127` / `926a907b2449e7a9` |

---

### Appendix A — Offline reference values (laptop, RTX 3050; **not** edge results, do not quote as such)

| Quantity | OSNet | TransReID | Source |
|---|---|---|---|
| Rank-1 / mAP (%) | 94.329 / 83.569 | 93.379 / 85.125 | `gate3_real_results_v3_reproduce.json` |
| Documented-harness median latency (ms) | 18.88 / 19.74 / 20.71 | 13.43 / 13.59 / 13.84 | `gate3_real_results.json`, `phase9`, `phase8` |
| Second-harness median latency (ms) | 28.169 | 58.718 | `benchmark_latency_memory.json` — **generator not on disk, method UNVERIFIED** |
| FP32 parameter memory (MB) | 8.276 | 384.405 | `benchmark_latency_memory.json` |
| Checkpoint on disk (MB) | 9.918 | 395.512 | same |
| FNMR @ 0.70 (%) | 18.70 | 38.51 – 48.66 | `verification_threshold_audit.json` |
| EER (%) @ EER τ | 2.052 @ 0.555 | 1.678 @ 0.330 (centered SVD) | same |
| Host test suite | 56 passed, 0 failed (`rclpy` module not collectable) | | fresh run 2026-10-16 |

The two latency rows **disagree in sign** on the same GPU; resolving that with one documented harness is a precondition of H7b, not a result of it.

**[REV02] Fence around this table — corrected after reading the artifacts.** Every number above is retained **unchanged**; this proposal does not overwrite history. Two assumptions in REV01 (and in this revision's earlier draft) are falsified by measurement, and one survives:

- **Falsified:** "this table was computed on the `id_band` population." All four Gate-3 accuracy artifacts — `gate3_real_results.json`, `_v2_corrected`, `_v3_reproduce`, `_reconciled` — record **`n_query = 3,368` / `n_gallery = 15,913`**, the same `directory` population as Appendix B. Today's `transreid_512_sie` therefore **reproduces** the recorded TransReID row: rank1 93.37885985748218, rank5 97.95130641330167, rank10 98.87173396674585 identical to the last digit; mAP 85.12520406508766 now vs 85.12520406484964 recorded, Δ 2.38e-10. That is the strongest pipeline-correctness evidence available without hardware, and it should be read as such rather than as a coincidence.
- **Falsified, and more sharply:** the recorded 71.1698 → 93.379 "correction" was **not** a population change. `gate3_real_results.json` (71.1698) and `_v2_corrected` / `_v3_reproduce` (93.379) carry identical `n_query`/`n_gallery`, so the 22.21 pp gap lies entirely in feature extraction and projection, not in which images were scored. `gate3_real_results_reconciled.json:19` names that cause as `"Corrupted by Centered Truncated-SVD mean subtraction and omitted JPM"` — while the 93.379 row it replaced **is** the centred projection (`svd_projection_768_to_512_real_v2.npz`, ‖μ‖ = 10.372914), so the two labels contradict each other and neither artifact states which preprocessing produced 71.1698. Open item for the consistency audit.
- **Survives, for a different reason:** the OSNet column stays `UNVERIFIABLE HERE, NOT REPRODUCED`, and not because of the population — because the extractor that would produce it cannot run on this host (P1: `torchvision::nms` import failure, placeholder backbone consuming 0 of 567 checkpoint keys, `zeros(512)` TRT stub). §3B forbids repairing it.

Consequently differences against Appendix A are permitted **only** between cells of the same arm, same representation and same declared pair-pool definition. That excludes every verification cell in this table: `verification_threshold_audit.json` records four model rows and **no population, pair count, camera rule or split metadata at all**, so its FNMR/FMR/EER cannot be reconciled with §4.4 by construction. The two TransReID memory rows were re-derived from today's measurements and are **consistent with each other** once their definitions are named (384.405 MiB = 100,769,512 trainable elements × 4 B; 395.512 MiB = checkpoint file bytes) — see threat §9.9. The latency rows remain unreconciled between themselves. Details: `TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`.

### Appendix B — Task 7 measured results, this host **[REV02]**

Population `directory`; gallery 15,913 rows / 751 unique pids incl. 2,798 distractors; query 3,368 rows / 750 pids; 0 junk; τ = 0.70 read from YAML; accept iff `sim ≥ τ`; appearance gate only; `projection_stage = post_graph_npz`; FP32 PyTorch on Windows (not FP16, not TensorRT, not edge). Source `C:\Users\PC\edgesync_es07_runs\statistics\task7_statistics.{json,md}`, generated 2026-10-19T16:06:52Z from embeddings generated 14:47:29Z; `integrity.match = true` for all three `.npz`; canonical-vs-per-query difference 0.0. All cells COMPUTED unless marked.

Retrieval (market-1501 CMC/mAP, 3,368 valid queries, 0 invalid):

| Arm | dim | Role | Rank-1 % (Wilson 95 %) | Rank-5 % | Rank-10 % | mAP % (bootstrap 95 %) | Distractor rejection |
|---|---|---|---|---|---|---|---|
| `transreid_768_agnostic` | 768 | descriptive only — not a contract output | 88.7470 (87.635–89.770) | 96.0214 | 97.9216 | 77.33 (76.49–78.20) | 94.71 % (2650/2798) |
| `transreid_512_agnostic` | 512 | deployed arm | 88.7470 (87.635–89.770) | 96.1401 | 97.8622 | 77.32 (76.49–78.18) | 96.21 % (2692/2798) |
| `transreid_512_sie` | 512 | deployed arm, SIE | **93.3789 (92.489–94.170)** | 97.9513 | 98.8717 | **85.13 (84.42–85.85)** | 96.00 % (2686/2798) |

Coverage: 750/750 query identities evaluated, 14.741 genuine candidates and 15,909.1 candidates per valid query. Rank-band spread for the camera-agnostic 512-D arm: 2,989 / 249 / 58 / 59 / 13 queries in rank 1 / 2–5 / 6–10 / 11–50 / >50.

**Reproduction note (the `transreid_512_sie` row is `COMPUTED` *and* `REPRODUCED`).** Rank-1 93.37885985748218, Rank-5 97.95130641330167, Rank-10 98.87173396674585 are **digit-for-digit identical** to `gate3_real_results_v3_reproduce.json → accuracy/transreid` (and to `_v2_corrected`), which records the same `n_query 3,368` / `n_gallery 15,913`; mAP 85.12520406508766 here vs 85.12520406484964 there, Δ 2.38e-10. `_pipeline_config` in that file agrees with this run on `[0.5]³`, SIE `xishu 1.0`, post-BNNeck, `image_size_hw [256,128]` and `svd_variance_retained 0.9864118099212646`. This is a reproduction of the extraction + retrieval path on the published test population — not a new measurement of the hypothesis, and not a check on the OSNet row in the same file, which no runnable arm on this host can produce.

Verification at the shipped threshold, pair pool 49,649 genuine / 53,532,220 impostor / 13,115 duplicates excluded:

| Arm | FNMR @ 0.70 (95 % CI) | FMR @ 0.70 | EER (τ at EER) | FMR at TAR 0.90 (τ) | per-query genuine accept | per-query false accept |
|---|---|---|---|---|---|---|
| `transreid_768_agnostic` | 54.6637 % (54.23–55.10) | 0.00789 % | 2.0095 % @ 0.36706 | 0.2975 % @ 0.49699 | 84.86 % | 26.48 % |
| `transreid_512_agnostic` | 60.7384 % (60.31–61.17) | 0.00521 % | 1.9718 % @ 0.30581 | 0.2810 % @ 0.45528 | 81.09 % | 19.92 % |
| `transreid_512_sie` | **48.1540 % (47.71–48.59)** | 0.00585 % | **1.5242 % @ 0.32755** | **0.1377 % @ 0.50845** | **88.69 %** | 20.16 % |

Paired inference, declared family of 3, Holm-corrected, 3,368 paired units, `512_agnostic` vs `512_sie`:

| ID | Per-query unit | Rates | Δ | Discordant (b / c) | exact McNemar p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H1 | Rank-1 hit | 0.8875 → 0.9338 | −0.046318 | 5 / 161 | 2.18039e-41 | **4.36078e-41** | REJECT H0 |
| H2 | genuine accept at τ | 0.8109 → 0.8869 | −0.076010 | 3 / 259 | 8.09013e-73 | **2.42704e-72** | REJECT H0 |
| H3 | false accept at τ | 0.1992 → 0.2016 | −0.002375 | 73 / 81 | 0.572846 | 0.572846 | FAIL TO REJECT |

OSNet vs TransReID paired test: **NOT APPLICABLE** — `RuntimeError: operator torchvision::nms does not exist`, recorded and not repaired (§3B).

Camera-partition **PROXY** (one dataset, viewpoint change between camera groups; no robot, no handoff event, no handoff label — *not* a cross-robot handoff measurement, charter §8):

| Arm | direction | valid / total queries | gallery rows | Rank-1 % | mAP % | FNMR @ τ | FMR @ τ |
|---|---|---|---|---|---|---|---|
| 768 agnostic | cams 1-2-3 → 4-5-6 | 1843 / 1915 | 7,752 | 90.02 | 81.17 | 54.20 % | 0.0069 % |
| 768 agnostic | cams 4-5-6 → 1-2-3 | 1449 / 1453 | 8,161 | 91.10 | 83.03 | 53.38 % | 0.0065 % |
| 512 agnostic | cams 1-2-3 → 4-5-6 | 1843 / 1915 | 7,752 | 89.91 | 81.28 | 60.28 % | 0.0046 % |
| 512 agnostic | cams 4-5-6 → 1-2-3 | 1449 / 1453 | 8,161 | 91.30 | 82.78 | 59.12 % | 0.0043 % |
| 512 SIE | cams 1-2-3 → 4-5-6 | 1843 / 1915 | 7,752 | 93.76 | 87.74 | 48.22 % | 0.0056 % |
| 512 SIE | cams 4-5-6 → 1-2-3 | 1449 / 1453 | 8,161 | 94.20 | 87.60 | 48.04 % | 0.0051 % |

`NOT MEASURED` in this appendix, and not inferred from anything above: end-to-end and engine-only latency, p50/p95/p99, throughput, FP16 accuracy, engine size, resident memory Δ, peak memory, the Mahalanobis spatial gate, cross-robot handoff success, and every H7b/H7c quantity.

### Appendix C — Artifact digests pinned by this revision **[REV02]**

| Artifact | Bytes | sha256-16 |
|---|---|---|
| `ES-EXP07-TASK7-PROPOSAL-01-REV01.md` (unmodified) | 21,351 | `845aeb70c5fdbd0f` |
| `ES-EXP07-TASK7-REPORT-01-REV01.md` (unmodified) | 10,123 | `f5951ac3511f9e39` |
| `JETSON-RUN-SHEET-PREP-01-REV01.md` (unmodified) | 28,413 | `1b65fb6ee5d071cf` |
| `transformer_120.pth` | 414,724,314 | `843055293e160da0` (full: `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5`) |
| `svd_projection_768_to_512_real_v2.npz` (deployed) | 1,466,587 | `9626bea5526df5b0` (full: `…a9d5b4dccec1c552224b5a6da504a0ab3a69f6bbfec1dbfc`) |
| `transreid_feature_agnostic.onnx` | 343,380,183 | `318b6857585d5d7c` |
| `transreid_feature_sie.onnx` | 343,399,114 | `2cd8ccc774c59337` |
| embeddings manifest | 7,352 | `283c8c782e66869a` |
| `task7_transreid_768_agnostic.npz` | 55,193,927 | `459cfbb31236d558` |
| `task7_transreid_512_agnostic.npz` | 37,094,828 | `cb4c8b8654ff410a` |
| `task7_transreid_512_sie.npz` | 37,095,723 | `d2d50a8ed50fbb54` |
| `statistics/task7_statistics.json` (consumer output, source of Appendix B) | 63,412 | `76d88dfc5e5c20c4` |
| `statistics/task7_statistics.md` | 5,298 | `a96e01ee40d724c0` |
| `statistics/task7_population.json` | 7,676 | `931744b17e59a139` |
| `results/experiment7/gate3_real_results.json` (read-only, population check) | 1,557 | counts `n_query 3,368` / `n_gallery 15,913`, `accuracy/transreid.rank1 71.16983372921615` |
| `results/experiment7/gate3_real_results_v2_corrected.json` (read-only) | 3,247 | same counts, `rank1 93.37885985748218` |
| `results/experiment7/gate3_real_results_v3_reproduce.json` (read-only) | 1,667 | same counts, `rank1 93.37885985748218` — **the row reproduced by `transreid_512_sie`** |
| `results/experiment7/verification_threshold_audit.json` (read-only) | — | four model rows, `tau 0.7`, **no population or pair-count metadata** |

`.engine` / `.plan` / `.trt`: **none exist**. `osnet_x1_0_market1501.pth`: present, **not** used to produce any Task 7 figure; see §W-05 for why it cannot be loaded by the code on this host.
