# Task 7 Report: TransReID as the Re-ID Backbone (Ablation Experiment 7)

Status: Partially executed — offline axes **COMPUTED**, every edge axis **NOT MEASURED — Pending Jetson Orin Nano hardware access.** Paired proposal: `ES-EXP07-TASK7-PROPOSAL-01-REV02.md`. Owner: Abanob.

**REV02 supersedes `ES-EXP07-TASK7-REPORT-01-REV01.md`** (10,123 B, sha256-16 `f5951ac3511f9e39`), which remains on disk **byte-for-byte unmodified** (re-verified when this file was written; duplicate in `C:\Users\PC\edgesync_es07_backup\modified_20261019_exec\`, same digest). REV01 was a skeleton whose every cell was `<fill>`; REV02 fills exactly the cells that this host can measure and leaves the device cells unfilled with their blocker named. **The decision rules (§0, §8 of the proposal) were not edited**, as REV01 required. Each change is tagged **[REV02]**.

> **[REV02] Deviation disclosed before any number appears.** REV01 pre-registered an *OSNet-vs-TransReID* comparison. That arm could not run: `osnet_torch_extractor.py:64` raises `RuntimeError: operator torchvision::nms does not exist`, its fallback backbone is a placeholder sharing 0 of 567 checkpoint keys, and `osnet_trt` returns `np.zeros(512)`. Charter §3B freezes OSNet and §8 forbids a paired test without outcomes for both arms, so **no OSNet figure in this report is measured, imputed, or quoted as a result**, and the pre-registered paired family was replaced by a *within-TransReID* family (H1–H3, §0/§4.4). This deviation reduces the experiment to one half of the charter's trade-off claim; the other half needs the device and a runnable baseline.

---

## 0. Pre-registration (must be signed **before** any data is collected)

| Item | Fixed value |
|---|---|
| Gallery / query definition | `n_query = 3,368`, gallery = `bounding_box_test` minus `pid -1` = 15,913, distractors (`pid 0000`) `<retained>` OR `<removed>` |
| Split rule | folder membership only; numeric pid ranges are not used |
| Genuine / impostor pair population | same-identity pairs in the other robot's cameras / different-identity cross-robot pairs; counts: genuine `<fill>`, impostor `<fill>` |
| Identity stage share of `T_txn = 886 ms` | `<fill>` ms — on-disk candidates, **not a device result**: `config/timing_budgets.yaml:11 dt_id_ms = 120`, `:15 t_txn_ms = 886`, `:19 t_claim_ms = 200`, `:18 t_ack_ms = 100`; owner decision G.1 **[REV01]** |
| Edge memory envelope for the Re-ID engine | `<fill>` MiB — `config/hw_params.yaml:36 unified_memory_limit_mb = 8192` is the only envelope on disk and no node reads it (all 24 keys unconsumed); owner decision G.3 **[REV01]** |
| Latency harness (single, documented) | **pre-registered, no longer `<fill>`:** 50 warm-up / 200 timed / 3 trials / batch 1 / `torch.cuda.Event`, idle gate ≤ 5 % utilisation — `run_real_pipeline.py::step4_latency`, locked in run sheet §2. The `benchmark_latency_memory.json` harness is **retired** (generator not on disk, method UNVERIFIED) **[REV01]** |
| SIE / `cam_label` available in the runtime path? | `<yes (P3 approved)` OR `<no — candidate arm runs without SIE>` |

**[REV02] As actually fixed before analysis.** Nothing in the table above was altered after collection; τ = 0.70 stayed.

| Item | Status | Value used / blocker |
|---|---|---|
| Gallery / query definition | **FILLED, asserted from the arrays** | `population = directory`: query 3,368 rows / 750 identities; gallery 15,913 kept of 19,732 scanned, 3,819 junk excluded, **distractors RETAINED** (2,798 rows, `pid 0`), 751 unique gallery pids. 0 junk rows and 0 zero-norm rows in either evaluated array **[REV02]** |
| Split rule | **HONORED** | folder membership only. Measured and recorded: identity numbers are globally unique and non-contiguous per split, so `id_band` filtering cannot express a partition. **[REV02] correction:** the on-disk Gate-3 accuracy artifacts were **not** produced by that filter — all four record `n_query 3,368` / `n_gallery 15,913`, the same `directory` population as this report. `id_band` is a live defect in `evaluate_offline.py`'s default path, not the provenance of a recorded number |
| Genuine / impostor pair population | **FILLED, with a declared redefinition** | genuine **49,649**, impostor **53,532,220**, duplicates excluded 13,115, junk 0, queries without an impostor candidate 0. Pool = same-identity pairs in a *different camera* vs different-identity pairs, over the whole official test population — **not** literally "other robot's cameras", because no robot assignment exists in Market-1501. The camera-group (1-2-3 ↔ 4-5-6) version is computed separately in §4.5 and labelled PROXY. Any FNMR here is quoted with this population beside it **[REV02]** |
| Identity stage share of `T_txn = 886 ms` | **UNSET — decision G.1 never landed** | Consequence: H7b is **undecidable**, and no latency cell in this report may be compared to a budget. `dt_id_ms = 120` remains a candidate on disk, not a gate |
| Edge memory envelope | **UNSET — decision G.3 never landed** | Consequence: H7c is **undecidable**. `unified_memory_limit_mb = 8192` is read by no node |
| Latency harness | **FIXED, NOT EXECUTED** | The harness is unchanged and is coded in `scripts/benchmark_transreid_jetson.py`; it requires a TensorRT engine that does not exist and a device that was not granted (G.7 denied: "Not reachable now — I'll connect it myself later"). `NOT MEASURED — Pending Jetson Orin Nano hardware access` |
| SIE / `cam_label` available in the runtime path? | **YES — G.5 approved (PF-10c) and implemented** | Per-robot ROS parameter `reid_camera_id` (`identity_node.py:76` → `core/reid_input_adapter.py:81-102` → `cam_label = reid_camera_id − 1`), ships `0` = camera-agnostic; `edgesync_msgs/srv/IdentityVerify.srv` untouched. Both sides measured: `transreid_512_sie` and `transreid_512_agnostic` |
| **[REV02] Paired family (added, because the registered one was unreachable)** | DECLARED BEFORE ANALYSIS | H1 per-query Rank-1 hit, H2 per-query genuine-accept at τ, H3 per-query false-accept at τ, each `512_agnostic` vs `512_sie` on 3,368 paired units; Holm–Boferoni over a family of 3, α = 0.05 |

## 1. Summary

- Hypothesis (H7): accuracy increases, but edge latency and edge memory break the handoff budget, proving OSNet is the optimal trade-off.
- H7a (accuracy increases — mAP): `<supported | not supported>`; Rank-1 `<fill>` pp, mAP `<fill>` pp. **[REV02] `NOT VERIFIED BY MEASUREMENT HERE`.** The candidate arm is `REPRODUCED` (§4.3): today's `transreid_512_sie` returns the recorded TransReID row exactly (93.37885985748218 / 85.12520406508766 on 3,368 × 15,913). Differenced against the recorded OSNet row in the same file that is **mAP +1.5560 pp but Rank-1 −0.9501 pp**, so "accuracy increases" is not uniform even within that record. No valid OSNet arm exists on this host (§3B frozen; P1), so the axis was never testable here — which is not the same as "not supported".
- H7b (edge latency exceeds the identity-stage budget): `<supported | not supported>` — decided only by Jetson `end_to_end` p95. **[REV02] `UNVERIFIED — NOT MEASURED` (no engine, no device, and G.1 left the budget unset).**
- H7c (edge memory prohibitive): `<supported | not supported>` — decided only by Jetson resident Δ. **[REV02] `UNVERIFIED — NOT MEASURED` (G.3 and G.7 both open).**
- End-to-end p50 ratio (TransReID / OSNet): `<fill>`×. Resident memory ratio: `<fill>`×. **[REV02] Both `NOT MEASURED`; no laptop ratio is substituted (REV01 §4.1 rule retained).**
- Recommendation (OSNet optimal / TransReID optimal / trade-off unresolved): `<fill>` — may not be "OSNet optimal" unless OSNet wins ≥ 2 of 3 axes with the latency and memory axes measured **on device**. **[REV02] `TRADE-OFF UNRESOLVED`. Zero of three axes are decided on device; one axis (accuracy) has a candidate arm but no baseline. This report issues no production recommendation.**

**[REV02] What the measured half does say.** Within TransReID, on the official test population at the shipped τ: the SIE camera embedding is the largest single lever found — **Rank-1 +4.632 pp (88.747 → 93.379)** and **mAP +7.801 pp (77.324 → 85.125)**, exact McNemar Holm p = 4.361e-41; genuine-accept at τ rises **+7.60 pp (81.09 → 88.69)**, Holm p = 2.427e-72; false-accept does not move detectably (Δ −0.24 pp, Holm p = 0.573). Discrimination is *better* than the recorded OSNet reference on every arm (EER 1.52–2.01 % vs the recorded 2.052 %) while the operating point at τ = 0.70 is *far worse* (FNMR 48.15–60.74 % vs the recorded 18.70 %) — the EER sits at τ ≈ 0.31–0.37. That recorded pair comes from `verification_threshold_audit.json`, which declares **no population, pair count, camera rule or split**, so these two directions are agreement in sign, never a difference in cells. **On this evidence the binding constraint is the threshold, not the backbone.** τ was not changed by this report; it is an owner calibration decision.

## 2. Setup

| Item | Value |
|---|---|
| Repository commit | `<fill>` — **currently unfillable:** `git rev-parse --show-toplevel` resolves to `C:/Users/PC` (the home directory), not this project, and the tree has no commit of its own. A commit hash is not obtainable until a repository is initialised at the authoritative root and the first commit made — prerequisite **[REV01]** added as G.6; until then write `NO-COMMIT (see G.6)`, never a guessed hash |
| Device / power mode | Jetson Orin Nano 8GB, `nvpmodel` mode `<fill>`, `jetson_clocks` `<on/off>` — `hw_params.yaml:35` says `jetson_nvpmodel_mode: 0 # 15W 6-core`, but that is a **config comment no node reads**; the mode must be read back with `sudo nvpmodel -q` on the device, not copied from the YAML **[REV01]** |
| JetPack / TensorRT / CUDA | `<fill>` / `<fill>` / `<fill>` |
| Precision | FP16 (both engines) |
| Input | 256×128; normalisation **per arm**: TransReID `<0.5,0.5,0.5 \| other>` = `<fill>`, OSNet `<fill>` |
| Projection stage | `<in-engine \| post-engine>`; R₅₁₂ `<fill>`; `.npz` sha256 `<fill>` |
| Threshold | 0.70 (unchanged) |
| Robot split | Robot A cameras `<fill>`, Robot B cameras `<fill>` |
| Evaluation data | `<Market-1501 \| custom>`; queries `<fill>`, gallery `<fill>`; dataset sha256 `<fill>` |
| Latency protocol | Warm-up 50, measured calls 200, trials 3 (run sheet §2) — trial spread `<fill>` % must be reported, and a trial whose spread exceeds 5 % is discarded and repeated **[REV01]** |
| Throttling check | `<fill>` |

**[REV02] As executed on this host (the rows above are the device plan and are unchanged):**

| Item | Value | Label |
|---|---|---|
| Repository commit | `NO-COMMIT (G.6 open; charter §2 prohibits `git init`/commit — the owner handles Git) | NOT APPLICABLE |
| Device / power mode | **NOT MEASURED — Pending Jetson Orin Nano hardware access** (G.7 denied). Nothing was read off a device; `nvpmodel`/`jetson_clocks` were not run | NOT MEASURED |
| JetPack / TensorRT / CUDA | `NOT MEASURED` on the target. This host: Windows 10.0.26300, Python 3.10.11, `torch 2.5.1+cu121`, cuDNN 90100, CUDA available, RTX 3050 Laptop 4 GiB, `onnx 1.23.2` / `onnxruntime 1.23.2` (CPU EP, user-level install authorized PF-10a), `numpy 1.26.4`; `tensorrt`, `pycuda`, `colcon` **absent** | COMPUTED (host), NOT MEASURED (target) |
| Precision | **FP32 PyTorch for every measured figure.** The FP16/TensorRT precision row is `NOT MEASURED — Pending Jetson Orin Nano hardware access` | deviation, disclosed |
| Input | 256×128 NCHW float32; TransReID normalisation `mean = std = [0.5, 0.5, 0.5]` after `/255`, asserted **bit-identical** between the torch path and the new TRT arm's numpy path under `ES07_HEAVY=1`; recorded per arm in the embeddings manifest. OSNet normalisation: `NOT APPLICABLE` (arm cannot run) | COMPUTED / NOT APPLICABLE |
| Projection stage | `post_graph_npz` — the exported graph ends at the post-BNNeck 768-D CLS feature; the centered SVD is applied by numpy after inference, identical for every arm and trial. R₅₁₂ = **0.9864118** (in-sample). `.npz` = `svd_projection_768_to_512_real_v2.npz`, 1,466,587 B, sha256 `9626bea5526df5b0a9d5b4dccec1c552224b5a6da504a0ab3a69f6bbfec1dbfc`, **not re-fitted** (charter §5) | COMPUTED |
| Threshold | τ = **0.70**, read live from `algorithm_thresholds.yaml` key `algorithm_thresholds.ros__parameters.reid_similarity_threshold`, asserted `== 0.70`; acceptance rule replicated from `core/identity_verifier.py:101` (**reject if `sim < τ`**, i.e. accept iff `sim ≥ τ`); similarities produced by the production cosine matrix and cross-checked against `IdentityVerifier.compute_cosine_similarity` on 200 sampled pairs, max abs difference **1.7881e-07**. Never tuned, never re-selected | COMPUTED |
| Robot split | Cameras 1–3 → group A, 4–6 → group B, both directions computed and reported as **PROXY ONLY** (§4.5). No robot exists, so this is a viewpoint split on one dataset, not a handoff configuration | PROXY |
| Evaluation data | Market-1501 v15.09.15 official, `population = directory`; queries **3,368**, gallery **15,913** (3,819 junk removed, 2,798 distractors retained); zip 152,733,771 B, sha256 `416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96` | COMPUTED |
| Latency protocol | **NOT EXECUTED.** No timed call was made on any target; no laptop latency is entered in §4.1 | NOT MEASURED |
| Throttling check | NOT APPLICABLE (no device run) | NOT MEASURED |

**[REV02] Extraction provenance for every figure below:** 19,281 rows (3,368 query + 15,913 gallery) × 3 arms, produced 2026-10-19T14:47:29Z by the corrected code paths (real `transformer_120.pth`, GPU FP32), elapsed 956.4 s, manifest 7,352 B sha256 `283c8c782e66869a…`; analysis generated 2026-10-19T16:06:52Z. The statistics consumer re-hashes each `.npz` against that manifest and **refuses to report** on mismatch — `integrity.match = true` for all three arms. Contract checks per image: 512-D, finite, unit-norm to max `|‖x‖ − 1| = 1.7881e-07`; the 768-D array is deliberately non-unit (max deviation 37.398) because it is raw post-BNNeck. No zero-sentinel row survived into any evaluated array.

## 3. Models

| Arm | Network | Weights source (URL, date) | Checkpoint sha256 | Documented training data | Params | Engine size (MiB) |
|---|---|---|---|---|---|---|
| Baseline | OSNet x1.0 native 512-D | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Candidate | TransReID ViT-B/16 + SIE + centered SVD 768→512 — **[REV01]** this row is only reachable if G.5/P3 is approved; otherwise the arm is the deployed no-SIE configuration (run sheet arm **A2**) and the header must read "TransReID **without SIE**", per proposal §2 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

**[REV02] Filled, with two rows that cannot be filled honestly.**

| Arm | Network | Weights source (URL, date) | Checkpoint sha256 | Documented training data | Params | Engine size (MiB) |
|---|---|---|---|---|---|---|
| Baseline | OSNet x1.0 native 512-D | **NOT AVAILABLE** — no URL or download date is recorded for `osnet_x1_0_market1501.pth` anywhere on disk | file present; **NOT USED to produce any figure in this report** | **NOT MEASURED — arm cannot execute here.** `osnet_torch` import raises `RuntimeError: operator torchvision::nms does not exist`; its fallback is a placeholder holding 8 entries / 42,945 elements (42,816 trainable + 129 buffers) that shares **0 of 567** checkpoint keys and loads with 7 missing / 567 unexpected under `strict=False`; `osnet_trt._execute_trt` returns `np.zeros(512)`; `osnet_mock` returns a channel-mean pseudo-embedding. Charter §3B: recorded, **not repaired** | n/a | **n/a — no OSNet engine exists** |
| Candidate (SIE approved) | TransReID ViT-B/16 + SIE + centered SVD 768→512 = **`transreid_512_sie`** | **NOT DOCUMENTED** — `transformer_120.pth` ships with no URL/date/training statement; the only evidence of its training set is internal (211 tensors, five `(751,768)` classifier heads ⇒ a 751-identity label space). Stated as asserted, not sourced | `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5` (414,724,314 B) | Argued from folder membership only: official `bounding_box_train` is 751 identities / 12,936 images, identity-disjoint from test and query (measured). **No independent proof the checkpoint saw only that split exists** | 211 checkpoint tensors / 103,661,037 elements; **100,769,512 trainable** after dropping `classifier.weight` (loads with `missing_keys = []`, 4 aux heads unexpected, coverage 100.0 %) | **NOT MEASURED — Pending Jetson Orin Nano hardware access.** No `.engine`/`.plan` exists. The nearest artefact is the serialized FP32 graph on disk: 343,399,114 B = **327.49 MiB** (SIE variant) — *not* an engine figure and not a device footprint |
| Candidate (no SIE) | Same backbone, camera-agnostic: **`transreid_512_agnostic`** | identical | identical | identical | identical | same |
| Reference (not a contract arm) | `transreid_768_agnostic` — raw post-BNNeck 768-D | identical | identical | identical | identical | same |

**G.5/P3 was approved, so the REV01 title caveat does not apply**: the SIE arm is reachable (via `reid_camera_id`, §0) *and* the no-SIE arm was measured alongside it, so the comparison is internal to this report rather than deferred.

PyTorch-vs-ONNX parity: max absolute embedding deviation `<fill>` over `<fill>` fixed images. **[REV02] Filled, with the shortfall stated: over `parity_n = 8` fixed images (the script's default), not the pre-registered ≥ 200 — worst 768-D max-abs 1.2398e-05 (agnostic) / 1.2875e-05 (SIE); worst post-contract 512-D 6.091e-07 / 6.855e-07; min cosine 0.9999999999955; tolerance 0.002, cosine floor 0.9999, verdict PASS for both graphs.** Both graphs re-exported byte-identically on a second run (sha256 `318b6857585d5d7c` agnostic / `2cd8ccc774c59337` SIE), `onnx.checker` PASS, opset 17, IR v8, 779 nodes (agnostic) / 783 (SIE), 154 / 155 initializers, dynamic batch exercised at 1 and 2 with observed width 768, and an off-contract 128×64 input refused by the graph (`InvalidArgument`). **This is ONNX parity, not engine parity**: charter §6 steps 6–10 (engine build, engine inference, engine-vs-PyTorch on ≥ 200 images) are unexecuted, so no deployment claim is made and none should be inferred from an `.onnx` existing.

## 4. Results

### 4.1 Inference latency (Jetson, `end_to_end` and `engine_only`)

| Stage | OSNet p50 (ms) | TransReID p50 (ms) | Ratio | OSNet p99 (ms) | TransReID p99 (ms) | TransReID FPS |
|---|---|---|---|---|---|---|
| end_to_end | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| engine_only | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

- Quality Gate 5 (≤ 18 ms): OSNet `<pass/fail>`, TransReID `<pass/fail>`.
- TransReID `end_to_end` p95 vs identity-stage budget (`<fill>` ms): `<within | exceeded>` by `<fill>` ms.
- Charter's "200+ ms" claim, tested against device data: `<observed | not observed>`.
- Laptop reference values are **not** reported here and may not be substituted.

**[REV02] Every cell above is `NOT MEASURED — Pending Jetson Orin Nano hardware access.`** Nothing was filled: no engine exists, G.7 was denied, and the harness (`scripts/benchmark_transreid_jetson.py`, 50 warm-up / 200 timed / 3 trials / batch 1 / `torch.cuda.Event`) is delivered but unexecuted. The four bullets under the table are equally unanswered — Quality Gate 5 cannot be applied to either arm, the p95 row has no budget because **G.1 was never set**, and the charter's "200+ ms" premise remains **untested on any device**: it has never been observed on any harness on this tree, which is why REV01 refused to use it as a gate. The last bullet is honored: the laptop figures on disk (18.88/19.74/20.71 vs 13.43/13.59/13.84 ms, and the second harness's 28.169 vs 58.718 ms) are recorded only in proposal Appendix A, are **retired as evidence**, and are not entered here — they disagree in sign on the same GPU.

### 4.2 Memory footprint on Edge

| Measure | OSNet | TransReID | Ratio | Method |
|---|---|---|---|---|
| Engine file on disk (MiB) | `<fill>` | `<fill>` | `<fill>` | file size |
| Resident device memory, before load (MiB) | `<fill>` | `<fill>` | n/a | `tegrastats` MEM fields + `VmRSS` + `MemAvailable` (unified memory). `nvpmodel` is a power-mode tool and reads no memory — v01 named it as a memory method **[REV01]** |
| Resident device memory, after warm-up (MiB) | `<fill>` | `<fill>` | n/a | same |
| **Resident Δ (MiB)** | `<fill>` | `<fill>` | `<fill>` | after − before, median of `<fill>` trials |
| Peak during run (MiB) | `<fill>` | `<fill>` | `<fill>` | `tegrastats.log` |
| Headroom against the `<fill>` MiB envelope | `<fill>` | `<fill>` | n/a | derived |

`peak_runtime_gpu_memory_mb` from the laptop JSON is excluded: it measured 410.727 vs 413.260 MB (OSNet higher), which is CUDA-context dominated and non-discriminating.

**[REV02] All six rows: `NOT MEASURED — Pending Jetson Orin Nano hardware access.`** The method column stands as written and is implemented in the delivered benchmark script. The envelope row is additionally blocked by **G.3 being unset**, so even a measured Δ would have no pass/fail bound. OSNet is `NOT APPLICABLE` in every row: the arm cannot run, so there is no process to measure.

What *is* measurable about memory, with each figure naming which quantity it is (threat §9.9):

| Quantity | Value | Definition |
|---|---|---|
| Checkpoint on disk | 414,724,314 B = **395.51 MiB** (= 414.72 MB decimal) | file bytes of `transformer_120.pth`. Appendix A calls this "395.512 MB" — the number is the **MiB** value wearing an MB label, which is itself part of threat §9.9 |
| Trainable parameters | 100,769,512 elements × 4 B = 403,078,048 B = **384.41 MiB** (= 403.08 MB decimal) | FP32 parameters after dropping `classifier.weight`; reconciles `103,661,037 − 5×576,768 − 5×1,537 = 100,769,512`. Appendix A's "FP32 parameter memory 384.405 MB" is this same MiB figure — so the two Appendix A rows are **consistent**, and their 11.11 MiB difference is the five `(751,768)` aux classifier heads plus 7,685 buffer elements (11.03 MiB) and ~0.08 MiB of container overhead |
| ONNX graph on disk | 343,380,183 B = **327.47 MiB** (agnostic) / 343,399,114 B = **327.49 MiB** (SIE) | serialized FP32 graph, file bytes |
| float32 initializers inside the graph | 343,249,920 / 343,268,352 B = **327.35 / 327.37 MiB** | weight bytes actually carried by the graph; 57.06 MiB below the trainable figure, **a gap not decomposed in this run — open item** |
| Emitted embedding arrays | 55,193,927 B = 52.64 MiB (768-D) / 37,094,828 B = 35.38 MiB / 37,095,723 B = 35.38 MiB (512-D) | 19,281 rows × dim × 4 B, the artifact every §4.3–§4.5 number derives from |
| **Engine size, resident Δ, peak during run, headroom** | **UNKNOWN** | require a TensorRT engine and the device |

The laptop `peak_runtime_gpu_memory_mb` exclusion is retained verbatim and applied: no CUDA-context figure appears anywhere in this report as a footprint claim.

### 4.3 Accuracy (Market-1501 protocol)

| Metric | OSNet | TransReID | Delta | McNemar p (Holm-adjusted) |
|---|---|---|---|---|
| Rank-1 | `<fill>` | `<fill>` | `<fill>` pp | `<fill>` |
| Rank-5 | `<fill>` | `<fill>` | `<fill>` pp | n/a |
| Rank-10 | `<fill>` | `<fill>` | `<fill>` pp | n/a |
| mAP | `<fill>` | `<fill>` | `<fill>` pp | n/a |

Offline reference (`gate3_real_results_v3_reproduce.json`, laptop, 2026-10-15): OSNet 94.329 / 83.569, TransReID 93.379 / 85.125 on 3,368 × 15,913. **[REV02] The TransReID half of this row is reproduced by §4.3 below — see the note after that table for what that does and does not license.**

**[REV02] Filled for the arms that exist; OSNet column withheld, with reason. Labels (charter §8): all three TransReID rows are `COMPUTED` from this run's embeddings, and `transreid_512_sie` is additionally `REPRODUCED` against `gate3_real_results_v3_reproduce.json`; the OSNet column is `NOT MEASURED`; every latency, memory and device cell is `NOT MEASURED`.**

Population `directory`, 3,368 valid queries of 3,368 (0 queries without a gallery match), 750/750 query identities evaluated, 14.741 genuine and 15,909.1 candidate gallery rows per valid query, max rank 50. Canonical protocol: `run_real_pipeline.compute_cmc_map` (validated against torchreid 0.2.5); the consumer recomputes per query and asserts equality — measured difference **0.0** on all three arms.

| Metric | OSNet | `transreid_512_sie` | `transreid_512_agnostic` | `transreid_768_agnostic` | Sie vs agnostic Δ | McNemar p (Holm) |
|---|---|---|---|---|---|---|
| Rank-1 % | **NOT MEASURED — arm cannot run here (§3)** | **93.3789** (Wilson 92.489–94.170) | 88.7470 (87.635–89.770) | 88.7470 (87.635–89.770) | **+4.6318 pp** | **4.361e-41 — REJECT H0** (5 / 161 discordant of 3,368) |
| Rank-5 % | NOT MEASURED | 97.9513 | 96.1401 | 96.0214 | +1.8112 pp | not in the declared family |
| Rank-10 % | NOT MEASURED | 98.8717 | 97.8622 | 97.9216 | +1.0095 pp | not in the declared family |
| mAP % | NOT MEASURED | **85.1252** (bootstrap 84.42–85.85) | 77.3240 (76.49–78.18) | 77.3274 (76.49–78.20) | **+7.8012 pp** | not in the declared family (descriptive) |
| Distractor rejection (`pid 0`, 2,798 rows) | NOT MEASURED | 96.00 % (2,686) | 96.21 % (2,692) | 94.71 % (2,650) | −0.21 pp | descriptive |

Rank-band spread, `transreid_512_agnostic`: 2,989 rank-1 / 249 rank 2–5 / 58 rank 6–10 / 59 rank 11–50 / 13 beyond 50. Per-query-camera Rank-1 breakdown is in `task7_statistics.json` under `by_query_camera` for each arm (camera 3 is consistently the easiest, 94.4–96.4 %).

**[REV02] The reference row above IS reproduced — and that is the strongest pipeline evidence in this report.** `gate3_real_results_v3_reproduce.json` records `n_query 3,368` / `n_gallery 15,913`, the same `directory` population as §4.3, so this is not a coincidence of four decimals: today's `transreid_512_sie` returns **rank1 93.37885985748218, rank5 97.95130641330167, rank10 98.87173396674585 — identical to the last digit** of the recorded row, with mAP 85.12520406508766 now vs 85.12520406484964 recorded (Δ 2.38e-10, float32 accumulation order). `gate3_real_results_v2_corrected.json` carries the same 93.379 row on the same counts. The reproduction covers the extraction path (checkpoint `843055293e160da0…`, projection `9626bea5…`, `[0.5]³`, post-BNNeck, SIE `xishu = 1.0`) and the CMC/mAP protocol, and it is why the accuracy half of this experiment can be called `REPRODUCED` rather than only `COMPUTED`.

What the reproduction does **not** buy: a paired test against the recorded OSNet row. `accuracy/osnet` in those files is a number attached to an extractor that cannot run on this host (§3B frozen; P1), its per-query outcomes are not on disk, and charter §8 permits a paired test only where valid paired query-level predictions exist for **both** arms. So `NOT APPLICABLE` still stands — for the baseline's sake, not the population's. Details: `TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`.

### 4.4 Cross-robot false matches, at the declared operating point

Pair population: genuine `<fill>`, impostor `<fill>` (must match §0).

**[REV02] Filled and re-declared (the label matters):** the pool is **cross-camera within the official test population**, not "other robot's cameras" — genuine **49,649**, impostor **53,532,220**, duplicate (same identity *and* same camera) pairs excluded **13,115**, junk rows **0**, queries with no impostor candidate **0**. §0 carries the same numbers, so the two sections agree. The robot-split version is §4.5 and is labelled PROXY. Scope of every row: **appearance gate only**; the Mahalanobis spatial gate is `NOT MEASURED` (no trajectory or covariance data exists offline).

| Metric | OSNet | TransReID | Delta | McNemar p (Holm-adjusted) |
|---|---|---|---|---|
| FNMR at 0.70 (95 % CI) | `<fill>` | `<fill>` | `<fill>` pp | n/a |
| False-handoff rate (95 % CI) | `<fill>` | `<fill>` | `<fill>` pp | `<fill>` |
| Pair false-match rate at 0.70 | `<fill>` | `<fill>` | n/a | n/a |
| FMR at matched TAR 0.90 | `<fill>` | `<fill>` | n/a | n/a |
| EER (τ) | `<fill>` | `<fill>` | n/a | n/a |
| Correct-handoff rate | `<fill>` | `<fill>` | `<fill>` pp | n/a |

**[REV02] Filled for the TransReID arms. OSNet: `NOT MEASURED — arm cannot run here`, and its recorded FNMR (18.70 %) is not entered as a comparator because `verification_threshold_audit.json` declares no pair population, count, camera rule or split, so that cell cannot be placed on this table's scale — it is not that the populations are known to differ; it is that one side of the comparison is undeclared.**

| Metric | `transreid_512_sie` | `transreid_512_agnostic` | `transreid_768_agnostic` | Δ (sie − agnostic) | McNemar p (Holm) |
|---|---|---|---|---|---|
| FNMR at τ = 0.70 (Wilson 95 %) | **48.1540 %** (47.71–48.59) | 60.7384 % (60.31–61.17) | 54.6637 % (54.23–55.10) | **−12.5844 pp** | n/a (pair-level, not in the family) |
| TAR at τ (= 1 − FNMR) | 51.8460 % | 39.2616 % | 45.3363 % | +12.5844 pp | n/a |
| Per-query genuine-accept rate | **88.69 %** | 81.09 % | 84.86 % | **+7.60 pp** | **2.427e-72 — REJECT H0** (3 / 259 discordant) |
| Pair false-match rate at 0.70 (FMR) | 0.00585 % (Wilson 0.00565–0.00606) | 0.00521 % (0.00502–0.00540) | 0.00789 % (0.00766–0.00813) | +0.00064 pp | n/a |
| Per-query false-accept rate | 20.16 % | 19.92 % | 26.48 % | +0.24 pp | **0.5728 — FAIL TO REJECT H0** (73 / 81 discordant) |
| FMR at matched TAR 0.90 (τ required) | **0.1377 % @ τ 0.50845** | 0.2810 % @ 0.45528 | 0.2975 % @ 0.49699 | −0.1433 pp | n/a |
| EER (τ at EER) | **1.5242 % @ 0.32755** | 1.9718 % @ 0.30581 | 2.0095 % @ 0.36706 | −0.4476 pp | n/a |
| Correct-handoff rate | **NOT MEASURED and NOT INFERABLE** — no handoff event, robot or label exists; a cross-camera retrieval success is not a handoff success (§8) | same | same | — | — |

Genuine-pair score distribution (n = 49,649 in each arm): `transreid_512_sie` mean **0.6839**, std 0.1330, min −0.1979, max 0.9677, quartiles 0.6133 / **median 0.7057** / 0.7791, p05 0.4402, p95 0.8565; `transreid_512_agnostic` mean **0.6447**, std 0.1422, quartiles 0.5612 / **median 0.6639** / 0.7476. **The mean of the SIE arm sits below the shipped τ while its median sits barely above it** — which is exactly why FNMR reaches ~48 % although Rank-1 is 93.4 %: the genuine mass is broad and centred almost on the gate. That is the calibration finding of this experiment: **τ = 0.70 is mis-set for every TransReID arm measured, by roughly 0.33 in similarity** — each arm's EER lies at τ ≈ 0.306–0.367, and reaching TAR 0.90 requires τ ≈ 0.455–0.508.

The matched-TAR row is what separates the two explanations REV01 set this table up to distinguish, and it also shows why the recorded cells cannot settle the comparison. At a *fixed* TAR of 0.90 the TransReID arms yield FMR 0.138–0.298 % — but **no OSNet FMR at matched TAR exists anywhere on this tree**, so the only recorded comparison is `verification_threshold_audit.json` (FMR 0.114 % OSNet vs 0.014 % TransReID, FNMR 18.70 % vs 38.51–48.66 %, EER 2.052 % @ 0.555 vs 1.678 % @ 0.330). **[REV02] What that file actually declares, measured by reading it: four model rows, `tau 0.7`, and per-model genuine/impostor mean and std — and no population, pair count, camera rule, split or tooling metadata of any kind.** Its scores are therefore not differencable against §4.4, and the reason is *undeclared definition*, not a known population change; REV01's assumption that it was band-filtered is not supportable from the artifact. Two of its cells do, however, corroborate this table in sign: its centred-SVD-with-camera genuine mean 0.6821 against today's measured 0.6839 on the SIE arm (Δ 0.0018 — near, but not an equality, because the pair pools are defined differently), and its TransReID EER 1.678 % @ τ 0.330 against today's 1.5242 % @ τ 0.32755. Both point the same way as the measured table: **all three arms have EER 1.52–2.01 %, at or below the recorded OSNet EER of 2.052 %, while every one of them fails far more genuine pairs at τ = 0.70 than the recorded 18.70 %.** Discrimination is not the loss; the operating point is.

REV01's mandatory-population note is satisfied on every FNMR quoted here, and the fourth FNMR on disk (93.98 / 97.62 % for an all-pairs population) is **not** used anywhere in this report.

### 4.5 Camera-partition proxy **[REV02] — new section, labelled so it cannot be mistaken for §4.4**

One dataset, two camera groups, no robot, no handoff event, no trajectory, no handoff label. It measures whether appearance identity survives a viewpoint change — the property a handoff *depends on* — and nothing more. Charter §8 forbids inferring cross-robot handoff success or false-handoff rates from retrieval metrics, so **no number below is a handoff measurement.** 3,368 queries partition to 1,915 (cams 1–3) and 1,453 (cams 4–6); galleries 7,752 and 8,161 rows; 72 and 4 queries have no same-identity row in the other group.

| Arm | direction | valid / total | Rank-1 % (Wilson) | mAP % | FNMR @ τ | FMR @ τ |
|---|---|---|---|---|---|---|
| 768 agnostic | 1-2-3 → 4-5-6 | 1843 / 1915 | 90.02 (88.56–91.30) | 81.17 | 54.20 % | 0.0069 % |
| 768 agnostic | 4-5-6 → 1-2-3 | 1449 / 1453 | 91.10 (89.52–92.46) | 83.03 | 53.38 % | 0.0065 % |
| 512 agnostic | 1-2-3 → 4-5-6 | 1843 / 1915 | 89.91 (88.45–91.20) | 81.28 | 60.28 % | 0.0046 % |
| 512 agnostic | 4-5-6 → 1-2-3 | 1449 / 1453 | 91.30 (89.74–92.65) | 82.78 | 59.12 % | 0.0043 % |
| 512 SIE | 1-2-3 → 4-5-6 | 1843 / 1915 | **93.76** (92.56–94.78) | **87.74** | **48.22 %** | 0.0056 % |
| 512 SIE | 4-5-6 → 1-2-3 | 1449 / 1453 | **94.20** (92.88–95.29) | 87.60 | **48.04 %** | 0.0051 % |

Both directions are reported because they are not symmetric (a 1.1–1.4 pp gap), and a single direction would have hidden that. The SIE arm wins every cell in both directions, which is the expected result if the camera embedding is doing what it claims — and a warning about deployment: `reid_camera_id` is a *robot-configuration* fact, not a measurement of the incoming frame, so a mis-set parameter puts the wrong camera embedding into every embedding the node produces.

## 5. Discussion

1. Speed gain versus accuracy cost: `<fill>`
2. Is the false-match change a discrimination loss or a similarity-scale shift? Compare "FMR at matched TAR" with "Pair false-match rate at 0.70" in §4.4: `<fill>`
3. Memory cost against the 8 GB envelope with the perception stack resident: `<fill>`
4. Impact on the handoff timing budget (identity stage share of the 886 ms `T_txn`): `<fill>`
5. Impact on the multi-cue verifier (spatial gating compensation): `<fill>`
6. If the τ = 0.70 gate is the binding constraint rather than discrimination, state the per-backbone calibration decision the owner must make (threshold is **not** changed by this report): `<fill>`

**[REV02] Answered only where evidence exists.**

1. **`NOT MEASURED` — the question is unanswerable here.** There is no latency for either arm and no valid OSNet accuracy. What can be said: the accuracy axis was never TransReID's problem to begin with on this evidence (85.13 mAP for the SIE arm, and its FMR at matched TAR is the best of anything measured), so the charter's framing — "while accuracy increases, latency might break the budget" — has its first half unsupported-but-promising and its second half entirely untested.
2. **Answered, and it is a similarity-scale shift, not a discrimination loss.** At a *fixed* operating point the impostor rate barely moves between arms (0.00521 % → 0.00585 %, per-query false-accept Δ +0.24 pp with Holm p = 0.573 — statistically indistinguishable) while the *acceptance* side moves massively (FNMR −12.58 pp, genuine-accept +7.60 pp, Holm p = 2.4e-72). Read the other way: at matched TAR 0.90, SIE needs τ ≈ 0.508 to accept 90 % of genuine pairs and then yields FMR 0.138 %, whereas the camera-agnostic arm needs τ ≈ 0.455 and yields 0.281 %. **Every arm's EER sits near τ ≈ 0.31–0.37, roughly 0.33 below the shipped 0.70.** A threshold chosen for a different backbone's similarity distribution is what these numbers show, not a weaker model.
3. **`NOT MEASURED`.** No resident Δ, no engine size, and no envelope (G.3 unset), so the 8 GB question is open. The only honest statement available: FP32 artifacts are large and mutually consistent (395.51 MiB checkpoint, 384.40 MiB of FP32 parameters, 327.5 MiB serialized ONNX), and an FP16 engine is expected to be materially smaller — but *expected* is not measured, and on Orin Nano's unified memory the engine bytes are only part of the resident footprint. The parameter-vs-engine ratio threat (46.4× / 39.9×) remains open.
4. **`NOT MEASURED`, and additionally undefined** — G.1 never assigned the identity stage a share of the 886 ms `T_txn`, so there is currently no number to compare a p95 against. Resolving H7b needs the device **and** that decision; either one alone leaves it open. `dt_id_ms = 120` remains an unconsumed config value.
5. **Partially answerable.** The verifier is multi-cue: `identity_verifier.py:114` applies a Mahalanobis spatial gate (cutoff 9.21) before the cosine gate can matter. All §4.4/§4.5 figures are **appearance-only**, so the deployed rejection rate is not the FNMR reported here — in principle a correct spatial gate compensates part of a mis-set appearance threshold, and in principle TransReID's lower impostor rate at matched TAR helps more when both gates are in series. Neither direction is measurable offline: no trajectory, covariance or handoff-label data exists in this project. Recorded as the reason the FNMR numbers above must not be read as deployed false-reject rates.
6. **Stated, and it is the actionable finding.** The owner must decide whether τ stays 0.70 for all backbones or is calibrated per backbone. This report **did not change τ**, did not tune it, and did not select any threshold as an operating point; every figure is at the shipped 0.70, and EER/τ@TAR are reported *diagnostically only*. If per-backbone calibration is authorized, the measured candidates on this population are τ ≈ 0.508 (SIE) / 0.455 (agnostic) for TAR 0.90 — **refit on a calibration partition, never on this test partition**, and the 0.70 result must remain the headline either way. Note also that OSNet's own calibration was never verified on this tree, so "0.70 is OSNet's threshold" is itself an unverified premise.

## 6. Threats to validity (observed)

`<fill: confirm or amend each item from the proposal, section 9>`

**[REV02] Confirmed, amended, or escalated — one line per proposal §9 item.**

1. **ViT FP16 numerics on TensorRT — CONFIRMED AS AN OPEN THREAT.** Nothing measured here is FP16; all three arms are FP32 PyTorch. The FP16 accuracy row and the OSNet-vs-TransReID FP16 comparison do not exist.
2. **Engine export fidelity — PARTLY MET, ONE REGISTERED REQUIREMENT MISSED.** Graph hashes recorded, checker PASS, 779 / 783 nodes (agnostic / SIE), opset 17, 0 external-data tensors, 0 input-contract violations, dynamic batch verified at 1 and 2, off-contract resolution refused. Torch↔ONNX parity PASS with worst 768-D deviation 1.24e-05 and worst post-contract 512-D deviation 6.9e-07 — **but on 8 images, not the ≥ 200 pre-registered**, and there is **no engine at all**, so §6 step 6–10 fidelity is untested. No deployment claim is made from the existence of an `.onnx`.
3. **Training-recipe mismatch — CONFIRMED AND ASYMMETRIC.** TransReID's checkpoint has no provenance document at all (no URL, date, or training statement; its 751-identity label space is inferred from head shapes), and the OSNet baseline cannot run, so the mismatch cannot even be characterized. Recorded, not equalized.
4. **Domain gap — CONFIRMED, UNMEASURABLE.** Market-1501 only; no UGV imagery, no handoff dataset in the project.
5. **Threshold calibration — ESCALATED FROM THREAT TO FINDING.** See §5.2/§5.6: every arm's EER lies ~0.33 below the shipped 0.70, and the matched-TAR row shows discrimination is not the loss. The 0.70 gate is the binding constraint on this evidence.
6. **Thermal/clock state — NOT APPLICABLE OFFLINE; REQUIRED BEFORE §4.1/§4.2.** The discard rules (throttling, >5 % trial spread) were never exercised because no timing run happened.
7. **Single seed — CONFIRMED, NARROW.** Seed 20261019 for the bootstrap and seed 11 for the 200-pair similarity cross-check, both recorded. No stochastic fitting, no fine-tuning, no threshold search was performed, so multi-seed would only matter for the (unexecuted) training arm.
8. **Camera split as a modelling choice — CONFIRMED, AND DOUBLED.** Written into every artifact; computed in **both** directions precisely because a single direction would have hidden the 1.1–1.4 pp asymmetry (§4.5).
9. **Memory metric ambiguity — CONFIRMED, AND THE TWO APPENDIX A FIGURES RECONCILED.** §4.2's table names each quantity with its units. `384.405` and `395.512` are both **MiB values printed under an "MB" label**: 384.41 MiB = 100,769,512 trainable elements × 4 B, 395.51 MiB = checkpoint file bytes, and their 11.11 MiB difference is the five `(751,768)` aux classifier heads plus 7,685 buffer elements (11.03 MiB of tensors) and ~0.08 MiB of container overhead. So the recorded pair was never a contradiction — it was two definitions of "memory", which is the threat. One gap **is** still undecomposed: the ONNX graphs carry 327.35 / 327.37 MiB of float32 initializers, 57.06 MiB below the trainable-parameter figure — recorded as an open item, and it bounds an engine size that remains unmeasured. The OSNet side of the original row (46.4× / 39.9×) cannot be re-derived at all here.
10. **[REV02 new, then reversed by measurement] The population hazard is the opposite of the one assumed.** REV01 and this report's earlier draft treated `directory` vs `id_band` as a discontinuity between this experiment and everything on disk. Measured: all four Gate-3 accuracy artifacts record `n_query 3,368` / `n_gallery 15,913`, so §4.3's TransReID row is a **reproduction**, not a new number. The live hazard is that agreement makes recorded cells *look* comparable when their arm, extractor or pair-pool definition is not — and the statistics consumer itself carries that error in writing: `task7_statistics.json → not_measured.statistical_comparison_against_recorded_gate3_numbers` reasons `NOT APPLICABLE — the recorded run used the invalid id_band population (TR-07), so its per-query outcomes are not the same population as this one`. The conclusion (no paired test against the record) is still correct, but its **stated reason is false**; the true reason is that no per-query outcomes for a runnable OSNet arm exist. Recorded as a documented defect of the consumer rather than patched, because editing that script would require a §1 backup and a re-run of its 68 tests; the correction belongs here and in the consistency audit. Source of the string: [analyze_task7_statistics.py:1140-1142](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_identity/scripts/analyze_task7_statistics.py#L1140-L1142).
11. **[REV02 new] No baseline arm at all.** Half the charter's claim ("proving OSNet is the optimal trade-off") has no comparator on this host. This is a scope limit of the experiment as executed, and it is the single largest threat to any recommendation.
12. **[REV02 new] The candidate arms share one checkpoint.** The three arms differ only in post-processing (dim, SIE), so the paired tests measure a projection/camera-embedding effect, not a backbone effect.

## 7. Conclusion and recommendation

`<fill>`

**[REV02] Conclusion.**

- **H7a: `NOT VERIFIED BY MEASUREMENT HERE`.** Needs a real OSNet arm; the only candidates here are a mock, a `zeros(512)` stub, and a placeholder backbone that consumes 0 of 567 checkpoint keys. §3B forbids repairing OSNet, so this was recorded, not fixed. What *was* established is a **reproduction**: the candidate arm returns the recorded TransReID accuracy exactly on the recorded population (§4.3), and against that same record the candidate is mAP **+1.5560 pp** / Rank-1 **−0.9501 pp** — a trade inside TransReID's own arms (SIE vs agnostic: Rank-1 +4.632 pp, mAP +7.801 pp, Holm p = 4.361e-41) is measured, while the backbone-vs-backbone axis is not.
- **H7b: UNVERIFIED — `NOT MEASURED`.** No engine, no device (G.7 denied), and no identity-stage budget (G.1 unset).
- **H7c: UNVERIFIED — `NOT MEASURED`.** No resident Δ (G.7), no envelope (G.3).
- **Recommendation: `TRADE-OFF UNRESOLVED`.** Not "OSNet optimal" — the rule requires OSNet to win ≥ 2 of 3 axes with latency *and* memory decided on device, and zero device axes were decided. Not "TransReID optimal" either: its accuracy is measured but its edge cost is entirely unmeasured. This report issues **no production recommendation**, which is the pattern REV01 explicitly prohibited repeating (`gate3_real_results_reconciled.json` recommending rejection while recording `jetson_orin_nano_status: NOT MEASURED`).
- **Supporting finding, stated without a recommendation attached:** within TransReID on the official test population, SIE + the shipped 512-D projection is the stronger configuration (+4.632 pp Rank-1, +7.801 pp mAP, +7.60 pp genuine-accept at τ, all Holm-significant; false-accept unchanged), and all arms discriminate better than the recorded reference while operating far worse at τ = 0.70. That points at threshold calibration, not at backbone choice.

**Explicit statement required by §7: which hypotheses were decided with device data.** **None.** H7a was decided by no data at all (baseline unavailable); H7b and H7c have **no device data** and remain `UNVERIFIED — NOT MEASURED`. Every device-dependent claim that would be needed to conclude the charter's hypothesis is absent from this report by design; §8's prohibition on off-device edge claims is honored by leaving those cells empty rather than filling them with laptop numbers.

**What unblocks the rest of this experiment, in order:**
1. **G.1** — the identity stage's share of `T_txn = 886 ms` (owner decision; no measurement needed).
2. **G.3** — the Re-ID resident-memory envelope on the 8 GB device, with the perception stack's resident cost stated alongside (owner decision).
3. Jetson Orin Nano access (**G.7**): read off and record `nvpmodel -q`, `jetson_clocks` state, and the actual JetPack / TensorRT / CUDA triplet.
4. Build the two FP16 engines from the hash-pinned ONNX graphs (`scripts/build_transreid_trt_engine.py`) and run engine-vs-PyTorch parity on ≥ 200 fixed images (the pre-registered count this host missed).
5. Run `scripts/benchmark_transreid_jetson.py` (50/200/3, batch 1, `torch.cuda.Event`, `tegrastats` + `VmRSS` + `MemAvailable`) and fill §4.1/§4.2 with device numbers; discard any throttled trial or trial with > 5 % spread.
6. A runnable OSNet baseline **supplied by its owner** (a host where `torchreid` imports, or a real OSNet TensorRT engine) — then, and only then, the registered OSNet-vs-TransReID paired test.
7. `reid_camera_id` set per robot on the device, with the resolved label and reason verified in the node's startup log.

## 8. Reproduction

```
tools/ablation/task7/run_task7_benchmark.sh
```

Artifacts: **[REV01]** `edgesync_bringup/results/experiment7/task7/{osnet,transreid}.json`, `comparison.json`, `comparison.md`, `tegrastats.log`, `memory_trials.json` — the results tree that already exists on disk is `edgesync_bringup/results/experiment7/`; v01's `docs/ablation/task7_results/` was a second location and would have split the evidence. Note `edgesync_bringup/setup.py:11-16` installs `config/` and `launch/` only, so `results/` is never in the install space — read it from the source tree. Commit hash `<fill>`. Result JSONs are written under new versioned names; no baseline artifact is overwritten.

**[REV02] The command above was NOT run and cannot be — it is the device path.** What was actually executed, in order, with exit statuses recorded in `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §5:

```
# from Implementation Phase/edgesync_ws/src/edgesync_identity

# 1. ONNX export + graph validation + Torch<->ONNX parity (charter §6 steps 1-5)
python scripts/export_transreid_onnx.py \
  --checkpoint ../edgesync_bringup/models/transreid/transformer_120.pth \
  --projection ../edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz \
  --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 \
  --out-dir C:/Users/PC/edgesync_es07_runs/onnx_trt09_rerun \
  --parity-n 8 --reference-device cpu
# exit 0; 2 graphs, both checker PASS, byte-identical to the first run; parity verdict PASS (8 images)

# 2. Embeddings for all three arms on the corrected code paths (19,281 rows each)
python scripts/generate_task7_embeddings.py \
  --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 \
  --transreid-checkpoint ../edgesync_bringup/models/transreid/transformer_120.pth \
  --transreid-projection ../edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz \
  --output-dir C:\Users\PC\edgesync_es07_runs\embeddings_full_directory \
  --population directory
# exit 0, 956.4 s -> task7_transreid_{768_agnostic,512_agnostic,512_sie}.npz + manifest
#   (7,352 B, sha256 283c8c782e66869a...); 0 exclusions and 0 extractor failures per arm;
#   the OSNet factory attempt in the same run is recorded as FAILED, not repaired (§3B)

# 3. Statistics (population, pairs, Wilson, bootstrap, McNemar+Holm, EER/tau@TAR, proxy)
python scripts/analyze_task7_statistics.py \
  --embeddings-dir C:/Users/PC/edgesync_es07_runs/embeddings_full_directory \
  --threshold-config ../edgesync_bringup/config/algorithm_thresholds.yaml \
  --out-dir C:/Users/PC/edgesync_es07_runs/statistics
# exit 0, 123.3 s (768-agnostic 46.3 s · 512-agnostic 39.0 s · 512-sie 38.0 s); every arm
#   integrity.match true, canonical-vs-per-query difference 0.0. The consumer re-hashes each .npz
#   against the manifest, requires manifest/array agreement, and REFUSES to run if tau != 0.70
# single-arm variant used first, for reference: --arms transreid_512_agnostic --skip-camera-partition
#   (exit 0, 30.7 s, identical numbers to the last digit)

# 4. Tests
python -m pytest tests/ --ignore=tests/test_identity_service.py -p no:cacheprovider -q
# exit 0: 303 passed, 2 skipped in 17.69 s
python -m pytest tests/test_analyze_task7_statistics.py -p no:cacheprovider -q        # 68 passed
ES07_HEAVY=1 python -m pytest tests/test_transreid_extractor_guardrails.py \
                           tests/test_transreid_trt_extractor.py -p no:cacheprovider -q
# exit 0: real-checkpoint load (missing_keys [], 100,769,512 params) + bit-identical preprocessing

# 5. NOT RUN — blocked on hardware
python scripts/build_transreid_trt_engine.py --onnx <graph.onnx> --arm <agnostic|sie> --out <e.engine>
# exit 1 BY DESIGN on this host: writes a build report showing tensorrt/pycuda NOT IMPORTABLE and
#   no /etc/nv_tegra_release, /proc/device-tree/model or /etc/nvidia-l4t-version marker. No engine.
python scripts/benchmark_transreid_jetson.py --engine <e.engine> --arm <arm> \
  --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 --out-dir <dir>
# exit 1 BY DESIGN: writes the protocol block plus power/memory capture and the verdict
#   "BLOCKED — Pending Jetson Orin Nano access ... a laptop benchmark is not a Jetson benchmark",
#   with ZERO latency numbers recorded
```

**[REV02] One laptop latency figure exists and is deliberately excluded.** The embeddings manifest carries a `latency_mean_ms` per arm, measured forward-only on this laptop GPU during extraction. §4.1's own rule bars substituting it, and it is also not the registered harness (no warm-up/repeats, `end_to_end` stages not separated), so it appears in no results cell here; it stays visible in the manifest for anyone who needs to know it was produced and set aside.

Commit hash: `NO-COMMIT` (G.6; charter §2 prohibits repository initialization — the owner handles Git). Result JSONs are new files outside the repository source tree under `C:\Users\PC\edgesync_es07_runs\`; `edgesync_bringup/results/experiment7/` contains only the pre-existing artifacts, none of which was overwritten, and its `task7/` subdirectory does not exist because nothing was measured on the device. Digests of every artifact this report cites: proposal REV02 Appendix C.
