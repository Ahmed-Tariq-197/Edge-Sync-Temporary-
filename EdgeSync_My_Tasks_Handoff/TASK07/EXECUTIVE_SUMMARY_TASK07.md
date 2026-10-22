# Task 7 — TransReID Re-Identification: comprehensive technical report

Experiment id **ES-EXP07**. Package: `EdgeSync_My_Tasks_Handoff` (2026‑10‑21 handoff folder).
This file is the Task 7 executive summary expanded in place into a full technical report. No report was
renamed, moved, duplicated or replaced, and no code, configuration, dataset, model weight, experiment
output, test, log or other packaged artefact was modified to produce it. The earlier summary's material
(its sections A–J) is preserved in full and mapped into §§1–13 below.

**Structure** (numbering maps 1‑to‑1 onto the required report sections 5.1–5.13):

| § | Content |
|---|---|
| 1 | Executive overview (5.1) |
| 2 | Objective, scope, requirements, success criteria, exclusions, plan‑vs‑actual (5.2) |
| 3 | Complete implementation inventory (5.3) |
| 4 | Architecture and execution flow (5.4) |
| 5 | Experimental methodology (5.5) |
| 6 | Complete numerical results (5.6) |
| 7 | Tests and validation (5.7) |
| 8 | Defect register and forensic findings (5.8) |
| 9 | Requirements traceability (5.9) |
| 10 | Integration and deployment readiness, ten levels (5.10) |
| 11 | Remaining work and prioritised actions (5.11) |
| 12 | Evidence index and reproducibility (5.12) |
| 13 | Final technical verdict (5.13) |

## Evidence vocabulary

Every classification label in this report is one of these nine, and they are not interchangeable:

| Label | Meaning in this report |
|---|---|
| `IMPLEMENTED` | The code exists in the package and was read. Establishes presence and internal consistency only — never correctness, never performance. |
| `MEASURED` | A number re‑derived in this pass by reading a packaged artefact at the cited field/row, or by re‑hashing a packaged file. |
| `TESTED` | A test or command that was actually executed, with its exit status and counts captured. Where a suite was re‑executed for this report, the date is stated. |
| `INHERITED` | A number that exists only inside a report, manifest or older artefact and could not be re‑derived from anything shipped here. |
| `CALCULATED` | Derived here from packaged values by an arithmetic step that is shown. A calculation is not a new measurement. |
| `NOT MEASURED` | No valid result exists anywhere in the package for that quantity. |
| `NOT TESTED` | A behaviour the code implements that no executed test covers. |
| `BLOCKED` | Execution was attempted or is provably impossible, with a recorded cause (missing dependency, absent hardware, denied gate). |
| `UNRESOLVED` | Two or more sources conflict, or a required decision is outstanding, and the package cannot settle it. |

Two other vocabularies occur in the underlying files and are **mapped, not merged**:

* The statistics artefact's own legend — `COMPUTED`, `REPRODUCED`, `BLOCKED`, `NOT MEASURED`,
  `NOT APPLICABLE` (`experiment_results/statistics/task7_statistics.json → labels_legend`).
  `COMPUTED` there means "derived from the recorded embeddings by this script"; this report labels the
  same figure `MEASURED` when it was re‑read from the artefact in this pass and `INHERITED` when it
  appears only in prose.
* The earlier packaging pass used a five‑label set (MEASURED / INHERITED / NOT MEASURED / BLOCKED /
  NOT VERIFIED). Its `NOT VERIFIED` is split here into `NOT TESTED`, `BLOCKED` or `UNRESOLVED`,
  because the causes and the corrective actions differ.

Evidence tiers are used as the project defines them: **Tier U** unit/model simulation, **Tier N** node
tests against a stub `rclpy` with message classes generated from the on‑disk `.msg`/`.srv` text,
**Tier R** real ROS 2 / DDS, **Tier H/J** Jetson and TensorRT on the target device. Tier U and Tier N
were reached; Tier R and Tier H/J were not reached at any point in this task.

One rule is applied throughout and stated once: **a passing unit test does not establish full system
correctness, ROS 2 integration, hardware performance or deployment readiness**, and **a reported
metric is not automatically reproducible merely because a report contains its value**.

---

## 1. Executive overview

### 1.1 Objective of the task, as written by the project

**Charter statement, verbatim** (`reports/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:23`, identical in REV01):

> "Experiment 7 (TransReID): Replace OSNet with TransReID. Metric to observe: Memory footprint on Edge and Inference Latency vs. Accuracy. Hypothesis: While accuracy increases, the 200+ ms inference latency might break the critical TTO handoff budget, proving OSNet is the optimal trade-off."

The experiment is a **three‑axis trade‑off study** — accuracy against edge memory against inference
latency — in which the accuracy axis was expected to be won and the two hardware axes were the
question.

**Why this strategy was considered.** The deployed identity backbone is OSNet x1.0 (2,169,508
parameters; `MEASURED` from `TASK7_WEIGHTS_PROVENANCE.md` and CA‑05, which also records 8.28 MiB of
parameter mass). TransReID ViT‑B/16 is an order of magnitude larger, so the experiment asks whether the
accuracy gained is worth the memory and latency cost on an Orin Nano‑class device, specifically against
the handoff timing budget.

### 1.2 What has actually been implemented

`IMPLEMENTED`, with `TESTED` noted where an executed test covers it:

* The TransReID ViT‑B/16 extractor arm in PyTorch (`models/transreid_extractor.py`): forward pass →
  centered truncated‑SVD 768→512 projection → L2 normalisation, with a non‑finite guard and a zeros
  sentinel. `TESTED`.
* The **file the Team Execution Guide names** as this task's strategy source,
  `models/transreid_trt_extractor.py` — a TensorRT feature‑engine arm that fails closed
  (`FileNotFoundError` on missing assets, `RuntimeError` when `tensorrt` is unimportable) and has
  **never executed on any host**. `TESTED` against stubs and fake engines only.
* Registry‑backed model selection (`models/extractor_factory.py`): five ids, two new (`transreid`,
  `transreid_trt`), lazy builders, and a class‑identity check that rejects look‑alike subclasses.
  `TESTED`.
* The experiment overlay `configs/edgesync_bringup/config/exp7_transreid.yaml` and the
  shared‑first‑then‑experiment resolver `edgesync_bringup/reid_launch_defaults.py`, selected by the
  environment variable `EDGESYNC_EXPERIMENT_CONFIG`. `TESTED` at Tier U.
* The `ReidSignature.msg` producer/consumer contract, its additive path through `identity_node.py`,
  and the `exp_id` chain (overlay → resolver → launch → node → `EXP-METADATA` log) defaulting to
  `EXP07`. `TESTED` at Tier U/N; never exercised on a real graph.
* An explicit, declared evaluation population and the statistics machinery behind it
  (`generate_task7_embeddings.py`, `evaluate_offline.py`, `analyze_task7_statistics.py`,
  `export_transreid_onnx.py`): Wilson intervals, percentile bootstrap, exact McNemar with Holm
  step‑down, FNMR/FMR/EER and matched‑TAR. `TESTED` (68‑case analyzer suite) and `EXECUTED` — the
  artefacts exist and were re‑read.
* Eighteen dedicated test files, plus two standalone verification harnesses.

### 1.3 Most important verified results

Everything in §6 is `MEASURED` by re‑reading the packaged artefacts in this pass. The two headline rows
the update request asked to verify were confirmed field‑by‑field, and the four most‑cited Task 7 test
files were re‑executed on 2026‑10‑22 (§7.2).

* On the published Market‑1501 test population — **3,368 queries × 15,913 gallery**, **49,649 genuine**
  and **53,532,220 impostor** pairs — camera conditioning (SIE) lifts Rank‑1 from
  **88.74703087885986 %** to **93.37885985748218 %** (**+4.6318 pp**, `CALCULATED`) and mAP from
  **77.32400426226894 %** to **85.12520406508766 %** (**+7.8012 pp**, `CALCULATED`).
* Pre‑registered family: H1 rejected at Holm‑adjusted p = 4.360781186252619e‑41; H2 at
  2.4270391816147673e‑72; **H3 failed to reject** (p = 0.572845968999707) — no detectable
  false‑accept penalty for the better arm.
* PyTorch↔ONNX parity and extractor↔contract parity are `MEASURED`: worst max‑abs 1.2397766e‑05 at
  768‑D and 6.090850e‑07 at 512‑D; the contract gate (tolerance 2e‑05, `every_nth = 1`) was checked on
  **all 19,281** embeddings and aborts on any violation.
* At the shipped τ = 0.70 the FNMR is **48.15 %–60.74 % in every arm**, and the empirical EER sits at
  τ ≈ 0.306–0.367. This is a **threshold finding, not a model finding**, and it is the most important
  number in this report for anyone who intends to deploy the output.

### 1.4 Most important defects and limitations

The complete register is §8. The five that change how the results must be read:

* **T7‑D01 — the charter's own question is unanswered.** H7a is `NOT MEASURED` (the OSNet arm cannot
  run here and charter §3B forbids repairing it); H7b/H7c are `NOT MEASURED` (no Jetson, no engine).
  The accuracy axis is evidenced; the memory‑and‑latency axis, which is what the hypothesis is about,
  is not.
* **T7‑D02 — no crop producer, so the delivered output has never been produced by a running node.**
  Zero `create_publisher(...)` calls carrying `sensor_msgs/Image` exist in any delivered source file;
  `identity_node.py:317` warns about exactly this.
* **T7‑D03 — the guide's switch value does not select the guide's file.** `reid_model: transreid`
  builds the PyTorch arm; the guide‑named `transreid_trt_extractor.py` is reachable only as
  `transreid_trt`, which requires an engine that does not exist. `UNRESOLVED`.
* **T7‑D04 — projection assets are selected by filename over a duplicated namespace.** Eight
  projection `.npz` files resolve to four distinct SHA‑256 values. RTM R‑13 `FAIL — VERIFIED`.
* **T7‑D14 — three different unit‑norm deviation figures and two different mean‑norm figures are
  recorded for the same 512‑D arms** (§6.6, §8.3). Small in absolute terms, but the contract evidence
  is not internally consistent, so §6.6 lists every variant instead of choosing one.

### 1.5 Current integration and validation status

* **Executed:** Tier U and Tier N only. `edgesync_identity` from the packaged copy —
  **360 passed, 15 skipped, 1 collection error, exit 1**; `edgesync_bringup` from the copy —
  **55 passed, 2 failed, 1 collection error**; four Task 7 files re‑executed 2026‑10‑22 (§7.2).
* **Never executed, anywhere:** `colcon build`, `rosidl` generation of `ReidSignature.msg`,
  `ros2 launch`, any DDS traffic, `trtexec`, a TensorRT engine build, any Jetson command, any robot.
  `import rclpy` fails on this host; `ros2` exits 127.
* **Conflicting status wording retained:** **SOFTWARE‑INTEGRATION COMPLETE** (scoped to software),
  **PARTIALLY COMPLETE** (project level), **NOT READY — SOFTWARE DEFECTS REMAIN** (pre‑implementation
  forensic audit). §2.8 states the scope of each; none is overwritten.
* **Readiness:** ready for review; **not** ready for integration into the deployed system; **not**
  ready for — and not claimable for — deployment or hardware validation. Per‑level basis in §10.

### 1.6 Recommended next decision or action

**Rule on which arm is "the experiment", and on whether τ = 0.70 is retained** (§11 rows 1–2). Both are
supervisor/owner decisions, and until they exist the two highest‑value technical actions — building the
engine on the target and measuring the OSNet comparator under an authorised repair — cannot be scoped.
If no decision is available, the one step that needs no decision is a `sensor_msgs/Image` crop producer
from perception or acquisition: without it Task 7's output has no live consumer at any tier.

---

## 2. Objective, scope, and requirements

### 2.1 Intended contribution to EdgeSync, stated separately from the ablation framing

Task 7's charter role is the ablation arm. Its *system* role is narrower and more concrete: it supplies
the **appearance identity the handoff path needs** — a unit‑norm `float32[512]` embedding per observed
target, carried on the new `ReidSignature.msg` so that `edgesync_coordination` can fill
`HandoffRequest.reid_feature_vector`. The service contract it feeds is `IdentityVerify.srv`
(`sensor_msgs/Image candidate_crop`, `float32[512] reference_feature_vector`,
`float32[512] candidate_feature_vector`). Before this work the identity node computed a smoothed
signature but published nothing, and the consumer's `set_reid_signature()` had no caller, so every
handoff request left with its initialising zeros. `ReidSignature.msg` closes that gap in code
(`IMPLEMENTED`) and is starved of real input in operation (T7‑D02).

### 2.2 Functional requirements as recorded in the package

Requirements are numbered **R‑01 … R‑42** in `manifests/requirement-traceability-matrix.csv`.
`MEASURED` by recounting that CSV (header + 42 rows) in this pass:

| Verdict in the matrix | Rows | Meaning |
|---|---:|---|
| `FAIL — VERIFIED` | 16 | A real defect, proved from the code as it stood at audit time. |
| `PASS — VERIFIED` | 15 | Proved from artefacts on disk. |
| `UNVERIFIED CLAIM` | 3 | A claim exists that no available artefact supports. |
| `PARTIAL` | 3 | Partly met; the gap is stated per row. |
| `BLOCKED` | 2 | Cannot be evaluated on this host. |
| `NOT TESTED` | 2 | No test exercises the behaviour. |
| `NOT APPLICABLE` | 1 | The rule was honoured, so no gap arises (R‑42). |

Implementation tasks are **TR‑01 … TR‑11**
(`reports/ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md`); hypotheses are **H1–H3**
(pre‑registered statistical family) and **H7a/H7b/H7c** (charter‑level trade‑off), both defined in
`reports/ES-EXP07-TASK7-PROPOSAL-01-REV02.md` §7–§8. §9 maps all three families to evidence. No
requirement was added, reworded or invented for this report.

The requirement rows that carry the most weight, with their exact verdicts:

| ID | Requirement | Verdict in the matrix | Key evidence |
|---|---|---|---|
| R‑02 | 512‑D L2‑normalised `float32` output contract | `PASS — VERIFIED` | 19,281 embeddings, `zero_rows = 0` |
| R‑03 | A TensorRT engine exists on the production path | `FAIL — VERIFIED` | 0 hits; `engines/` empty (pre‑implementation snapshot) |
| R‑04 | Checkpoint loads with documented key handling | `PASS — VERIFIED`, gap | `load_state_dict(strict=False)` return discarded at `:128` |
| R‑05 | OSNet TRT arm returns a real feature | `FAIL — VERIFIED` | Stub returns `zeros` (`osnet_trt_extractor.py:75`) |
| R‑06 | Assets hash‑pinned | `PASS — VERIFIED`, gap | Origin URL/date/licence `NOT RECORDED` |
| R‑10 / R‑11 | SIE camera conditioning implemented / exposed | class `PASS`, API `FAIL — VERIFIED` | Base‑class API and `.srv` absent (gate G.5 open); measured cost of the missing path −4.632 pp Rank‑1 / −7.801 pp mAP |
| R‑12 | No projection/evaluation identity overlap | `PASS — VERIFIED` | Fit ids 0001–0650 vs evaluation 0752–1501 |
| R‑13 | Projection identified by hash, not filename | `FAIL — VERIFIED` | 8 names → 4 distinct SHA‑256 (`dc2b9072`×3, `628b1dc6`×1, `c9bca759`×2, `9626bea5`×2); selection is by name at `algorithm_thresholds.yaml:47` |
| R‑14 | Projection fit is reproducible | `UNVERIFIED CLAIM` | The fit was never re‑run; charter forbids refitting |
| R‑16 | OSNet torchreid dependency handled | `FAIL — VERIFIED` | `RuntimeError` uncaught; placeholder matches 0 of 567 keys |
| R‑17 | OSNet reference reproduction fidelity | `PASS — VERIFIED` | `osnet_backbone.py:527-533`, 0 missing / 0 unexpected, cosine 1.00000003 vs torchreid 0.2.5, module not imported in production |
| R‑19 | Metric reproduction | `PASS — VERIFIED` | Reproduces 93.37885985748218 / 85.12520406484964 and OSNet 94.32897862232778 / 83.5692348350484 bit‑identically |
| R‑20 | Dataset inventory | `PASS — VERIFIED` | 68,036 files; train 12,936/751 ids; test 19,732 (3,819 junk, 2,798 distractors, 750 ids); query 3,368/750; cameras 1–6; 275 MB |
| R‑21 | Train ∩ test empty | `PASS — VERIFIED` | Empty **by folder** |
| R‑22 | Population rule is the documented one | `FAIL — VERIFIED` | Defect B‑12: `id_min`/`id_max` were function defaults; `run_real_pipeline.py:489-498` unfiltered |
| R‑23 | Split boundary semantics documented | `PASS — VERIFIED`, gap | "Boundary semantics undocumented" |
| R‑24 / R‑25 | FNMR/FMR/EER attributable to a declared population | `UNVERIFIED CLAIM` | The historical 1,542 B artefact declares no pair counts or population; the current artefact does |
| R‑26 / R‑27 | McNemar + Holm / Wilson present | `FAIL — VERIFIED` (snapshot) | **Conflict preserved** — see §2.7 |
| R‑28 | Latency harness conditions | `FAIL — VERIFIED` | Times random noise (`:566`); `cam_label` hard‑coded 0 (`:570`) |
| R‑30 | A single latency figure | `FAIL — VERIFIED` | "Four mutually inconsistent values" |
| R‑31 | Memory figures consistent | `FAIL — VERIFIED` | 410.73 vs 413.26 MiB — the *smaller* model carries the larger number |
| R‑32 | Engine size + device protocol executed | `BLOCKED` | "No Jetson in session + owner gate G.7 unanswered" |
| R‑33 / R‑34 | Node / launch integration | `PARTIAL` | Unit‑reachable only |
| R‑35 | Model selectable at runtime through the real launch | `FAIL — VERIFIED` | Shipped default is the mock; end‑to‑end selection `NOT TESTED`, `ros2` exits 127 |
| R‑36 / R‑38 | Invalid values fail clearly / service round‑trip | `NOT TESTED` | Needs Tier R |
| R‑37 | Asset paths resolve independent of CWD | `FAIL — VERIFIED` | Path resolution is CWD‑dependent |
| R‑39 | Device profiling harness | `FAIL — VERIFIED` | `profile_edge_latency.py` synthesises Jetson numbers, applies an +18.0 ms fudge, and writes into `timing_budgets.yaml` |
| R‑40 | Run sheet usable | `FAIL — VERIFIED` | 110 `<fill on device>` tokens |
| R‑41 | Changes committed / authorship attributable | `BLOCKED` | No `.git` (B‑18) and gate G.6 unobtainable |
| R‑42 | No test‑set tuning; τ not re‑derived | `NOT APPLICABLE — rule honoured` | τ never retuned |

### 2.3 Non‑functional requirements, and where each stands

| Non‑functional requirement | Source | Status | Basis |
|---|---|---|---|
| No silent fallback to a mock backbone in a shipped path | proposal §4.5, RTM R‑35/R‑36 | `PARTIAL` — closed at unit level, open at shipped‑default level | Registry still defaults to `osnet_mock` (`extractor_factory.py:40`); `_log_model_selection()` warns when the origin is `default:node` (`identity_node.py:254-266`); the TRT import failure raises instead of substituting (`extractor_factory.py:280`) |
| Protected/shared assets unchanged | charter §3B, integration report §4.6 | `MEASURED` — honoured | 54 rows classified `BASELINE_PROTECTED`/`OTHER_TEAM_OWNED`, all byte‑identical to the 2026‑10‑16 archive; full‑tree re‑hash 244 identical / 6 changed (all owned by this task) / 0 missing |
| No parameter tuning on the test set; τ not re‑derived | RTM R‑42 | `NOT APPLICABLE` — honoured | τ = 0.70 read from `algorithm_thresholds.yaml`; `JETSON-RUN-SHEET-PREP-01-REV01.md:22` |
| Experiment selectable without editing shared config | guide "parameter switch" | `IMPLEMENTED`, `TESTED` (Tier U) | `EDGESYNC_EXPERIMENT_CONFIG` + `select_experiment_config()` (`reid_launch_defaults.py:80, :229`) |
| Assets reproducible by digest | proposal §6, RTM R‑06/R‑13 | `PARTIAL` | SHA‑256 pinned inside artefacts, but provenance fields `NOT RECORDED` and selection is by name |
| Authorship / version control | RTM R‑41 | `BLOCKED` | No repository; the user owns all Git operations |
| Malformed configuration fails clearly | integration report §4.2 | `TESTED` | `ValueError` with parse context; `FileNotFoundError` naming the path |
| No claim of ROS 2 / DDS / TensorRT / Jetson / robot validation without execution evidence | standing instruction to this task | honoured | Every such quantity in §6 is `NOT MEASURED` or `BLOCKED` |

### 2.4 Success criteria and target metrics

| Criterion | Intended target | Outcome |
|---|---|---|
| H1 — Rank‑1 parity between the two 512‑D arms (null) | Statistical decision, α = 0.05, family size 3 | `MEASURED`: **H0 rejected**, adjusted p = 4.360781186252619e‑41 |
| H2 — genuine‑accept parity at τ | same | `MEASURED`: **H0 rejected**, adjusted p = 2.4270391816147673e‑72 |
| H3 — false‑accept parity at τ | same | `MEASURED`: **fail to reject H0**, p = 0.572845968999707 |
| H7a — candidate mAP above the OSNet baseline | A paired comparison on one declared population | `NOT MEASURED` — the arm cannot run here (charter §3B); `osnet_comparison.status = "NOT APPLICABLE on this host"` |
| H7b — candidate p95 exceeding the identity share of `T_txn = 886 ms` | Device p95 vs a budget | `NOT MEASURED` / undecidable — gate **G.1** `OPEN`; the charter's "200 ms" is recorded as **not a documented budget** and is not used as a gate (`PROPOSAL-01-REV02.md:196`) |
| H7c — resident Δ plus perception stack inside the envelope | Device RSS delta | `NOT MEASURED` — gate **G.3** `OPEN` |
| Extraction parity across implementations | Contract tolerance 2e‑05 | `MEASURED` `PASS` on all 19,281 embeddings, abort‑on‑violation |
| Population declared, not implied | Population block in the artefact | `IMPLEMENTED` + `MEASURED` (defect B‑12/P5 closed by `evaluate_offline.py`) |
| TR‑01 remove the silent mock from the shipped launch path | No mock in a production launch | `PARTIAL` — the default is still `osnet_mock`; the node now *reports* a fallback as a fallback and warns |
| TR‑02…TR‑09 extractor, projection, node selection, asset fail‑fast, population, statistics, ONNX parity | Unit‑verified | `TESTED` at Tier U only (the implementation report marks these `PASS(UNIT)`) |
| TR‑10 TensorRT engine + production path | Engine built and benchmarked | `PARTIAL` — class exists and is contract‑tested; no engine has ever been built or loaded anywhere |
| TR‑11 Jetson latency/memory | Device measurement | `BLOCKED` — `PROPOSAL-01-REV02.md:196`, owner gate **G.7 DENIED** |

### 2.5 Assumptions and constraints

* **Charter §3B — OSNet is frozen.** Its backbone fails to import here
  (`RuntimeError: operator torchvision::nms does not exist`) and repairing it is out of scope. This
  alone removes H7a's comparator.
* **Charter §6 step 6 — a TensorRT engine must be built in the actual target environment.** A laptop
  build cannot stand in for it; the probe recorded exactly that and refused.
* **Charter §8 — no handoff inference from offline retrieval metrics.** The artefact's own
  `camera_partition_disclaimer` repeats it: the proxy "involves no robot, no handoff event, no
  trajectory and no handoff label".
* **No projection refitting** during this reporting work — refitting would change the deployed asset.
* **No Git writes** — the user owns all repository operations.
* **Backup before edit; copy never move**; conflicting historical reports retained rather than rewritten.
* **Inherited assumptions:** Market‑1501 is the evaluation corpus; one camera partition rule stands as
  the cross‑view proxy; the identity‑conditioned handoff benefit is inferred from retrieval quality
  rather than measured. The last two are disclaimed by the artefact itself and are not repaired here.

### 2.6 Explicit exclusions

Out of scope for Task 7 as executed, therefore **absent by design, not by oversight**: any model
training, fine‑tuning or projection refitting; any handoff‑level experiment (no TTO episode, no
handoff label, no success/failure outcome); any ROS 2 runtime work, DDS tuning, QoS verification or
multi‑node graph execution; any Jetson/Orin Nano execution, engine build, power/clock configuration or
power measurement; re‑derivation of τ; re‑hashing all 68,036 dataset files; licence clearance; repair
of shared OSNet, `algorithm_thresholds.yaml` or `identity_verifier.py` behaviour.

### 2.7 Plan versus actual, and one preserved conflict

| Planned | Executed | Gap |
|---|---|---|
| Replace OSNet with TransReID and compare against it | TransReID arm built and measured; OSNet arm never ran | H7a `NOT MEASURED`; only a recorded historical OSNet row exists and it is not comparable cell‑for‑cell (§6.10) |
| Measure edge memory and inference latency on Orin Nano | Laptop forward‑only timings recorded; device run never started | Every level‑8 value `NOT MEASURED`; the run sheet is preparation‑only with 110 `<fill on device>` tokens |
| Build a TensorRT engine and benchmark it | ONNX export + parity done; engine build attempted on the wrong host class and refused | TRT arm `IMPLEMENTED`, `NOT TESTED` against a real engine |
| Serve embeddings from a running identity node | Node code and tests exist; publisher dormant by default; no crop topic exists | The output has never been produced by a running process |
| Report McNemar/Wilson/bootstrap statistics | Done; the artefact is internally traceable | — fully executed |
| Keep the evaluation population explicit | Done for the new artefact | Five historical `results/experiment7/*.json` records lack it and conflict (§6.9) |

**Conflict preserved, not resolved.** RTM marks R‑26 (exact McNemar + Holm) and R‑27 (Wilson intervals)
`FAIL — VERIFIED` with the gap "B‑09: acceptance statistics absent", because the RTM is the
*pre‑implementation* audit snapshot. `reports/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` TR‑08 records the
same statistics as implemented and executed, and the artefact shipped in this folder carries them. Both
views remain on disk. Do not read R‑26/R‑27 as a live gap without reading the change‑log row, and do
not read the change‑log row as proof the RTM was wrong — each records the state it audited.

### 2.8 Status‑vocabulary conflict, retained rather than harmonised

Three verdicts exist for the same body of work, and each was true **of its own scope**:

* **SOFTWARE‑INTEGRATION COMPLETE** — `ES-EXP07-TASK7-INTEGRATION-VERIFICATION-REPORT.md` §0, scoped by
  its own prompt to "every feasible local software criterion in the 17‑item checklist", with ROS 2 and
  Jetson/TensorRT categories recorded as *pending*, not passed. That report also explicitly declines to
  attempt `colcon build`, because with no `ament_index_python`/`launch_ros` it would write `build/`,
  `install/` and `log/` trees into the workspace while starting no middleware — "zero verification
  value for a real artifact cost".
* **PARTIALLY COMPLETE** — the project‑level status, because the standing charter treats Orin Nano
  execution as a required activity rather than a documentation note.
* **NOT READY — SOFTWARE DEFECTS REMAIN** — `ES-EXP07-AUDIT-FORENSIC-01.md`, a pre‑implementation
  forensic snapshot of defects such as R‑05, R‑16, R‑28 and R‑39. Several were later fixed, several
  remain, and the audit was not re‑executed after the fixes.

This report does not pick one. Where the difference is only scope, §2.8 says so; where a defect
survives into the delivered copy it appears in §8 with its own id and its own status.

## 3. Complete implementation inventory

### 3.1 How the change history is known, and where it is not

There is no version control in this working tree (RTM R‑41 `BLOCKED`), so "created / modified /
unchanged" cannot be established from commit history. It is established from **two records in the
package, both of which classify against the 2026‑10‑16 archive**:

* `TASK07_STATUS_AND_BLOCKERS.md` — classifies the 141 inventory rows of this slice as **22 created**
  after the archive, **4 modified**, **34 byte‑identical** to it, plus **54 rows** classified
  `BASELINE_PROTECTED` or `OTHER_TEAM_OWNED`, all of which are byte‑identical to the archive.
* `manifests/ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` (17,005 B) — file‑level before/after
  evidence: §A 21 of 28 entries unchanged; §C **6/6 protected assets `byte_identical = true`** with
  sizes; §D a full‑tree re‑hash of **252 entries → 244 identical, 6 changed (all owned by this task),
  0 missing**.

Consequently: the inventory below attributes authorship **only** where those records classify a file,
and marks every other provenance statement `INHERITED`. No file in this report is claimed to have been
created or modified "by me" — the only write performed for this report was to this Markdown file itself
(§12.4).

Path prefixes used below, all verified to exist in this package:

| Prefix | Resolves to |
|---|---|
| `SNAP` | `TASK07/source_snapshot/Implementation Phase/edgesync_ws/src/` |
| `ASSET` | `TASK07/assets/edgesync_identity/edgesync_identity/models/` |
| `CFG` | `TASK07/configs/edgesync_bringup/config/` — **contains exactly one file**, `exp7_transreid.yaml` (4,183 B). The wider `TASK07/configs/` tree holds one more file, `configs/edgesync_identity/package.xml` (646 B) |
| `RES` | `TASK07/experiment_results/` |
| `REP` | `TASK07/reports/` |
| `MAN` | `TASK07/manifests/` |
| `TST` | `TASK07/tests/` |

**Which configuration file lives where — `MEASURED` in this pass by directory listing.** The shared
project configuration (`algorithm_thresholds.yaml`, `timing_budgets.yaml`, `hw_params.yaml`,
`hardware_params.yaml`, `coordination_params.yaml`, `exp13_cbba.yaml`, `fastdds_profiles.xml`) is **not**
copied into `CFG`; it exists only at `SNAP/edgesync_bringup/config/`, so it is cited with the `SNAP`
prefix throughout §6–§13. Only the Experiment‑7 overlay is packaged twice — `CFG/exp7_transreid.yaml` and
`SNAP/edgesync_bringup/config/exp7_transreid.yaml`, both 4,183 B — and both copies are the same file, so a
reviewer who opens `CFG` and finds one entry has not found a missing artefact. Nothing is inferred from
this layout; it is recorded because the earlier draft of this report cited three shared‑config lines under
the `CFG` prefix, and those citations have been re‑pointed to `SNAP`.

### 3.2 Package census, `MEASURED` in this pass

Directory‑level file counts and byte totals, re‑walked for this report (`.pytest_cache`/`__pycache__`
excluded):

| Directory | Files | Bytes |
|---|---:|---:|
| `TASK07/` root (this report, `README_TASK07.md`, `TASK07_STATUS_AND_BLOCKERS.md`) | 3 | 49,169 |
| `ASSET` | 8 | 82,293 |
| `TASK07/assets/edgesync_bringup/models/transreid` (projections) | 6 | 8,792,784 |
| `…/models/projections` | 2 | 2,931,274 |
| `…/models/osnet` | 1 | 10,399,605 |
| `CFG` | 1 | 4,183 |
| `TASK07/configs/edgesync_identity` | 1 | 646 |
| `RES/embeddings` | 4 | 7,463 |
| `RES/execution_evidence` | 4 | 4,011,203 |
| `RES/integrity` | 1 | 4,462 |
| `RES/onnx` | 2 | 54,940 |
| `RES/provenance` | 1 | 17,275 |
| `RES/results/experiment7` | 7 | 13,581 |
| `RES/statistics` | 5 | 78,011 |
| `RES/trt_blocked` | 3 | 2,399 |
| `MAN` | 5 | 83,929 |
| `REP` (top level) | 19 | 647,882 |
| `REP/evidence` | 17 | 42,982 |
| `REP/audit_20261019_141628` | 6 | 83,841 |
| `SNAP/edgesync_identity/…/models` | 8 | 82,293 |
| `SNAP/edgesync_identity/…/core` | 4 | 18,355 |
| `SNAP/edgesync_identity/…/interfaces` | 2 | 2,819 |
| `SNAP/edgesync_identity/…/nodes` | 2 | 32,725 |
| `SNAP/edgesync_identity/scripts` | 10 | 254,650 |
| `SNAP/edgesync_identity/tests` | 16 | 226,768 |
| `SNAP/edgesync_msgs/msg` | 13 | 13,331 |
| `SNAP/edgesync_msgs/srv` | 1 | 831 |
| `SNAP/edgesync_bringup/edgesync_bringup` | 2 | 18,749 |
| `SNAP/edgesync_bringup/config` | 8 | 31,477 |
| `SNAP/edgesync_bringup/launch` | 3 | 23,100 |
| `SNAP/edgesync_bringup/tests` | 4 | 27,363 |
| `TST/edgesync_identity/tests` | 16 | 226,768 |
| `TST/edgesync_bringup/tests` | 2 | 19,940 |
| **total** | **207** | **50,417,443** |

The `ASSET`, `SNAP/…/models` and `TASK13/source_snapshot/…/models` copies of all **eight** model
modules are **byte‑identical** — re‑hashed in this pass, e.g. `extractor_factory.py` `5ffbb07e29bc…`,
`transreid_trt_extractor.py` `f090f477b789…`, `transreid_extractor.py` `5b97db3135f6…`.
`MEASURED`. Before this report was expanded the tree totalled 50,432,341 B; the difference
(−14,898 B) is exactly the size change of this file, and the file count is unchanged at 207.

### 3.3 Model, extractor and node code

| Path (`SNAP/…` unless stated; identical copy under `ASSET` where both listed) | Size | Purpose / key symbols | Status | History | Known defects / limits |
|---|---:|---|---|---|---|
| `edgesync_identity/models/transreid_extractor.py` | 14,229 B | Deployed PyTorch arm. ViT‑B/16 forward → center → truncated SVD 768→512 → L2; non‑finite guard; zeros sentinel | `IMPLEMENTED`, `TESTED` | classified created/modified by the package records | Loads with `strict=False` and discards the returned named‑tuple (RTM R‑04 gap, `:128`); never runs the TRT path |
| `edgesync_identity/models/transreid_trt_extractor.py` | 19,753 B | **The file the Team Execution Guide names.** TensorRT feature‑engine arm; 768‑D engine output; host‑side projection; `FileNotFoundError` on missing assets; `RuntimeError` when `tensorrt` is unimportable; docstring section "Deliberate divergences" | `IMPLEMENTED`, `NOT TESTED` on a real engine | created by this task per the package records | Never executed on any host; all its tests use stubs/fake engines; depends on an engine that does not exist |
| `edgesync_identity/models/transreid_backbone.py` | 8,044 B | TransReID ViT‑B/16 backbone definition | `IMPLEMENTED`, `TESTED` indirectly | package records | Weight loading is the R‑04 path above |
| `edgesync_identity/models/extractor_factory.py` | 12,853 B | `BACKBONE_ID_*` constants `:32-36`; `DEFAULT_BACKBONE_ID = 'osnet_mock'` `:40`; `BACKBONE_REGISTRY` `:137-171`; `_build_transreid` `:112-124`; `_build_transreid_trt` `:127-133`; `_resolve_registered_class` `:219`; class‑identity check `_verify_against_registry` `:224`; `ImportError → RuntimeError` `:280` | `IMPLEMENTED`, `TESTED` | package records | Default id is still the mock (R‑35 `FAIL — VERIFIED`); ids resolve by name only |
| `edgesync_identity/models/osnet_torch_extractor.py` | 7,214 B | Baseline OSNet PyTorch arm (pre‑existing, unchanged) | `IMPLEMENTED` (shared/baseline) | byte‑identical to the archive per §D | Depends on the OSNet backbone that cannot import here |
| `edgesync_identity/models/osnet_trt_extractor.py` | 4,027 B | Baseline OSNet TRT arm (pre‑existing) | `IMPLEMENTED` as a **stub** | byte‑identical | Returns `zeros` at `:75` — RTM R‑05 `FAIL — VERIFIED` |
| `edgesync_identity/models/osnet_backbone.py` | 15,998 B | Reference OSNet reproduction used for validation only (`:527-533`); not imported by the production path | `IMPLEMENTED`, `TESTED` (R‑17 `PASS — VERIFIED`) | byte‑identical | Import failure `RuntimeError: operator torchvision::nms does not exist` — recorded, not repaired (charter §3B) |
| `edgesync_identity/interfaces/base_identity_extractor.py` | 2,696 B | Extractor interface | `IMPLEMENTED` | baseline | No camera‑conditioning API — the RTM R‑11 gap behind gate G.5 |
| `edgesync_identity/core/identity_verifier.py` | 5,897 B | Production verifier; accept rule `sim ≥ τ` at `:101` | `IMPLEMENTED` (shared, read not modified) | byte‑identical | Its τ is the value the offline statistics replicate; not re‑tuned here |
| `edgesync_identity/core/reid_input_adapter.py` | 6,413 B | Crop → model input pre‑processing | `IMPLEMENTED`, `TESTED` (`test_transreid_input_adapter.py`) | package records | Channel‑order history documented in `REP/evidence/channel_order_diagnostic.json` |
| `edgesync_identity/core/signature_smoother.py` | 5,865 B | EMA signature smoothing | `IMPLEMENTED` (baseline) | byte‑identical | — |
| `edgesync_identity/nodes/identity_node.py` | 32,635 B | Node. Params `reid_model_type` `:87`, `reid_model` `:88`, `reid_checkpoint_path` `:89`, `reid_projection_path` `:90`, `reid_engine_path` `:91`, `exp_id` `:93` (default `EXP07`), `reid_camera_id` `:96` (default `0`), `publish_reid_signature` `:101` (default **False**), `reid_signature_topic` `:102`, `reid_target_crop_topic` `:103` (default `''`), `reid_signature_target_id` `:104`; similarity `:79` 0.70; smoothing α `:80`; canonical‑over‑alias precedence `:235-252`; conflict WARN `:240`; `EXP-METADATA` log `:217`; signature import `:302`, publisher `:312`, crop subscription `:332-333`, publish guards `:394`, publish `:409` | `IMPLEMENTED`, `TESTED` at Tier N | package records | Publisher dormant by default; starved of input (T7‑D02); never run on a real graph |

### 3.4 Offline experiment scripts (`SNAP/edgesync_identity/scripts/`)

| File | Size | Purpose | Status | Notes |
|---|---:|---|---|---|
| `generate_task7_embeddings.py` | 22,811 B | Produces the three arms' embeddings; stamps `manifest['exp_id']` (`:65, :270, :338`); `EXP-METADATA` log `:293` | `IMPLEMENTED`, `EXECUTED` | Its `.npz` outputs are **not** in the package (§12.2) |
| `evaluate_offline.py` | 25,948 B | Explicit evaluation population (closes defect B‑12/P5); canonical CMC/mAP | `IMPLEMENTED`, `TESTED` (28‑case population suite) | — |
| `analyze_task7_statistics.py` | 66,091 B | Wilson, percentile bootstrap, exact McNemar + Holm, FNMR/FMR/EER, matched‑TAR, `exp_id` gate (`:93, :713, :1090, :1109`) | `IMPLEMENTED`, `TESTED` (68 cases), `EXECUTED` | Degenerate‑population markdown crash was found and fixed here (§8.4 T7‑D21) |
| `export_transreid_onnx.py` | 28,184 B | ONNX export, graph validation, Torch↔ONNX parity | `IMPLEMENTED`, `TESTED`, `EXECUTED` | Parity sample count `parity_n = 8` at extractor level vs 19,281 contract checks — see T7‑D16 |
| `run_real_pipeline.py` | 33,330 B | Historical pipeline that owns the recorded population and timings | `IMPLEMENTED`, `INHERITED` outputs | `:33` hard‑codes an absent OneDrive path (CA‑17); `:488-498` is the population the recorded arms used; `:489-498` unfiltered (R‑22) |
| `fit_svd_projection.py` | 12,000 B | Projection fitting | `IMPLEMENTED`, **never re‑executed** | Charter forbids refitting; R‑14 `UNVERIFIED CLAIM` |
| `build_transreid_trt_engine.py` | 22,046 B | Engine build harness | `IMPLEMENTED`, `BLOCKED` | Executed on this host → exit 1 and a written BLOCKED report |
| `benchmark_transreid_jetson.py` | 19,598 B | Device benchmark harness | `IMPLEMENTED`, `BLOCKED` | Same |
| `generate_test_fixtures.py` | 10,855 B | Synthetic `.pth`/`.npz` fixtures for tests | `IMPLEMENTED` | Fixtures excluded from the package by the size policy (T7‑D25) |
| `setup_edgesync_assets.py` | 13,787 B | Asset staging helper | `IMPLEMENTED` | Ownership of some behaviour recorded as uncertain |

**File cited by the requirement matrix but absent from this package:** `profile_edge_latency.py`
(RTM R‑39 `FAIL — VERIFIED`: it synthesises Jetson numbers, applies an +18.0 ms fudge and writes into
`timing_budgets.yaml`). A recursive search of the whole handoff package finds **no such file**; the
only artefact with "latency" in its name is `RES/results/experiment7/benchmark_latency_memory.json`.
The defect is therefore `UNRESOLVED` in the strong sense: its evidence is the matrix row and the dated
audit, not a file that can be re‑inspected here. Recorded as T7‑D18.

### 3.5 Configuration, launch and interface definitions

| Path | Size | Purpose / entries | Status | Notes |
|---|---:|---|---|---|
| `CFG/exp7_transreid.yaml` (identical copy in `SNAP/edgesync_bringup/config/`) | 4,183 B / 70 lines | Overlay: `/**: ros__parameters:` with `exp_id: "EXP07"` `:45`, `reid_model_type: "transreid"` `:52`, `reid_checkpoint_path: "models/transreid/transformer_120.pth"` `:56`, `reid_projection_path: "models/transreid/svd_projection_768_to_512_real_v2.npz"` `:57`, `reid_engine_path: ""` `:61`, `reid_camera_id: 0` `:70`; header states "Threshold tau = 0.70 is UNCHANGED from the project baseline" | `IMPLEMENTED`, `TESTED` | Loading is documented as `EDGESYNC_EXPERIMENT_CONFIG=<abs path>`; the file states `experiment_config:=…` is **not** a supported invocation |
| `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml` | 4,513 B | Shared parameters; `algorithm_thresholds.ros__parameters.reid_similarity_threshold` = 0.70; projection selected by name at `:47` | `IMPLEMENTED` (shared, unmodified) | Its root key matches no node, so it is read through Python rather than as a ROS parameter file (T7‑D26) |
| `SNAP/edgesync_bringup/edgesync_bringup/reid_launch_defaults.py` | 18,701 B | `MODEL_ALIAS_KEY = 'reid_model'`, `DEFAULT_EXP_ID` `:77`, `EXPERIMENT_CONFIG_ENV_VAR` `:80`, `coerce_camera_id()` `:128`, `_read_parameters(strict)` `:159`, `select_experiment_config()` `:229`, `resolve_reid_config()` `:264` | `IMPLEMENTED`, `TESTED` | Chosen over a launch argument deliberately: `generate_launch_description()` builds defaults before CLI parsing, so an argument would add a second precedence channel (§2.7 deviation, documented not hidden) |
| `SNAP/edgesync_bringup/launch/robot_bringup.launch.py` | 8,873 B | Flat entity list (8 declared args / 7 nodes); resolver call `:38`; `exp_id` `:65-66, :135`; camera id coerced `:80`, injected `:136` | `IMPLEMENTED`, `TESTED` (25/25 harness checks) | **Never launched**: `launch`/`launch_ros` absent |
| `SNAP/edgesync_bringup/launch/exp13_bringup.launch.py` | 11,774 B | Task 13's bring‑up; leaves `reid_target_crop_topic` empty at `:204-208`, documented at `:37-40` | dependency (Task 13) | Evidence for T7‑D02 |
| `SNAP/edgesync_msgs/msg/ReidSignature.msg` | 2,171 B | `sender_id`, `target_id`, `valid`, `float32[512] vector`, `quality`, `stamp_utc_ns`, `model_id`; header records that `quality` is **not** a confidence (0.0 = "the producer measured nothing") and that `stamp_utc_ns` is in the **producer's** clock domain | `IMPLEMENTED` | **Never compiled** — `rosidl` has not run; consumer lives in Task 13 |
| `SNAP/edgesync_msgs/srv/IdentityVerify.srv` | 831 B | `sensor_msgs/Image candidate_crop` + two `float32[512]` vectors | `IMPLEMENTED` | Re‑hashed identical (`1eae43d251dd1cde…`) in the change manifest §C/§D; never served on a real graph |
| `SNAP/edgesync_msgs/msg/` (13 files) | 13,331 B | Handoff protocol messages + `ConsensusBelief.msg` | baseline/Task 13 | Unmodified by Task 7 |
| `SNAP/edgesync_bringup/config/timing_budgets.yaml` | 2,031 B | `T_txn = 886 ms` budget record | baseline | Target of the absent `profile_edge_latency.py` (T7‑D18) |
| `SNAP/edgesync_bringup/config/fastdds_profiles.xml`, `coordination_params.yaml`, `hardware_params.yaml`, `hw_params.yaml`, `exp13_cbba.yaml` | 4,753 / 1,079 / 1,656 / 1,644 / 11,618 B | Shared or Task 13 configuration | dependency | Read, not modified |

### 3.6 Tests delivered for Task 7 (18 files)

Sizes from `TST/`; the same bytes exist under `SNAP/…/tests/` (the identity package's 16 files hash
identically in both locations).

| Test file | Size | What it covers | Executed outcome (§7) |
|---|---:|---|---|
| `edgesync_identity/tests/test_transreid_trt_extractor.py` | 24,799 B | TRT arm contracts, fail‑closed behaviour, host projection | **34 passed, 1 skipped** (`TESTED` 2026‑10‑22) |
| `test_analyze_task7_statistics.py` | 41,031 B | The whole statistics machinery | 68 passed (packaging run) |
| `test_gate1_scaffolding.py` | 20,352 B | Gate‑1 scaffolding incl. OSNet standalone checks | 24 passed, 8 skipped |
| `test_identity_node_transreid_integration.py` | 17,186 B | Node‑level integration on a stub `rclpy` | 20 passed |
| `test_exp7_offline_exp_id.py` | 15,306 B | Offline `--exp-id` stamping/gate | 24 passed |
| `test_experiment7_forensic_fixes.py` | 15,138 B | Forensic‑audit regressions | in the 360‑passed run |
| `test_evaluate_offline_population.py` | 14,406 B | Population construction rules | 28 passed |
| `test_export_transreid_onnx.py` | 13,582 B | ONNX export + parity checks | in the 360‑passed run |
| `test_extractor_factory_registry.py` | 12,792 B | Registry routing, class‑identity rejection, no‑mock | **22 passed, 1 skipped** (`TESTED` 2026‑10‑22) |
| `test_transreid_extractor_guardrails.py` | 10,833 B | Guardrails, non‑finite handling | 17 passed, 1 skipped |
| `test_extractor_factory_transreid_trt.py` | 10,311 B | TRT dispatch + lazy‑import isolation in a fresh interpreter | **16 passed** (`TESTED` 2026‑10‑22) |
| `test_transreid_input_adapter.py` | 8,106 B | Crop pre‑processing / channel order | in the 360‑passed run |
| `test_experiment7_fixes.py` | 7,181 B | Earlier defect regressions | in the 360‑passed run |
| `test_identity_fusion.py` | 3,895 B | Fusion behaviour | in the 360‑passed run |
| `test_identity_node_model_resolution.py` | 6,688 B | Canonical/alias precedence at the node | **23 passed, 11 warnings** (`TESTED` 2026‑10‑22) |
| `test_identity_service.py` | 5,162 B | Real service round‑trip | **not collected** — `ModuleNotFoundError: rclpy` (`BLOCKED`) |
| `edgesync_bringup/tests/test_exp7_config_overlay.py` | 13,426 B | Overlay precedence, missing/malformed handling | in the bringup run |
| `edgesync_bringup/tests/test_reid_launch_defaults.py` | 6,514 B | Resolver unit coverage | in the bringup run |
| `edgesync_bringup/tests/test_robot_bringup_launch.py` (baseline, not Task 7) | 7,387 B | Launch description | **not collected** — `launch` namespace shadow |

Two further verification harnesses are delivered outside `tests/`:
`verify_node_identity_resolution.py` (20/20 checks) and `verify_launch_description.py` (25/25 checks),
`TESTED` in the integration pass (`INHERITED` here — their logs live outside the package).

### 3.7 Model and data assets, with digests re‑verified in this pass

| Path | Size | SHA‑256 (prefix) | Status | Notes |
|---|---:|---|---|---|
| `TASK07/assets/edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz` | 1,466,587 B | `9626bea5526df5b0…` | `MEASURED` — **matches** the recorded digest | The asset `exp7_transreid.yaml:57` selects |
| `…/svd_projection_768_to_512_real_v3_reproduce.npz` | 1,466,587 B | `9626bea5526df5b0…` | `MEASURED` | Byte‑identical to `_real_v2` — same content, different name |
| `…/svd_projection_768_to_512.npz`, `…/svd_projection_768_to_512_real.npz`, `projections/svd_projection_768_to_512.npz` | 1,466,710 B each | `dc2b9072…` | `MEASURED` | R₅₁₂ fingerprint 0.9857138991355896; the family the v1 report used |
| `…/svd_projection_768_to_512_norm05_centered.npz`, `projections/…_norm05_centered.npz` | 1,464,564 B each | `c9bca759…` | `MEASURED` | — |
| `…/svd_projection_768_to_512_norm05.npz` | 1,461,626 B | `628b1dc6…` | `MEASURED` | — |
| `TASK07/assets/edgesync_bringup/models/osnet/osnet_x1_0_market1501.pth` | 10,399,605 B | `2809d3227f7d078f…` | `MEASURED` | Baseline weights present; the OSNet *arm* still cannot run (§8.1 T7‑D06) |
| `models/transreid/transformer_120.pth` | 414,724,314 B | `843055293e160da0…` | `INHERITED` (size/digest recorded inside in‑package JSON) | **Not in the package**; two bringup tests fail on the copy because of it (§7.1) |
| `transreid_feature_agnostic.onnx` / `transreid_feature_sie.onnx` | 343,380,183 / 343,399,114 B | `318b6857585d5d7c…` / `2cd8ccc774c59337…` | `INHERITED` | Live at `C:\Users\PC\edgesync_es07_runs\onnx_trt09\`; excluded by the size policy |
| Embeddings `.npz` (3 arms) | 55,193,927 / 37,094,828 / 37,095,723 B | `459cfbb3…` / `cb4c8b86…` / `d2d50a8e…` | `INHERITED` | Live at `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory\`; every §6 accuracy figure depends on them |
| TensorRT engine | — | — | **does not exist** | `BLOCKED` |

**Eight projection files resolve to four distinct contents** (`MEASURED` by re‑hashing in this pass):
`dc2b9072` ×3, `9626bea5` ×2, `c9bca759` ×2, `628b1dc6` ×1. Selection is by **name** in
`algorithm_thresholds.yaml:47` and `exp7_transreid.yaml:57`. This is RTM R‑13 `FAIL — VERIFIED` and
T7‑D04; note also that `TASK07_STATUS_AND_BLOCKERS.md` §1 cites `_real_v3_reproduce.npz` where the
overlay pins `_real_v2.npz` — different names, identical bytes, so the *citation* is inconsistent while
the *content* is not (T7‑D15).

### 3.8 Experiment output artefacts (`RES/`)

| Path | Size | Content | Status |
|---|---:|---|---|
| `statistics/task7_statistics.json` | 63,412 B | Every number in §6.1–§6.4: population block, per‑arm retrieval, verification at τ, paired tests, threshold curves, `not_measured` list, `labels_legend` | `MEASURED` (re‑read); generated 2026‑10‑19T16:06:52Z |
| `statistics/task7_statistics.md`, `task7_population.json` | 5,298 / 7,676 B | Human‑readable rendering; population declaration | `MEASURED` |
| `statistics/analyze_all_arms.log`, `analyze_arm1_512agnostic.log` | 1,013 / 612 B | Analyzer console output | `INHERITED` (logs, not re‑executed) |
| `embeddings/task7_embeddings_manifest.json` | 7,352 B | Population rules, parity gate, asset digests, per‑arm norms and timings, provenance statement | `MEASURED` |
| `embeddings/task7_transreid_{768_agnostic,512_agnostic,512_sie}_exclusions.json` | 37 B each | Exclusion lists — all three empty (no image was dropped) | `MEASURED` |
| `onnx/transreid_onnx_report.json`, `transreid_onnx_report_rerun.json` | 27,279 / 27,661 B | Export, graph validation, parity | `MEASURED` |
| `trt_blocked/probe_build_report.json`, `build_run2.log`, `bench_run2.log` | 1,812 / 385 / 202 B | Engine‑build attempt on a non‑target host; **contains no latency numbers** | `BLOCKED` |
| `provenance/task7_provenance_measurements.json` | 17,275 B | Measured asset provenance fields | `MEASURED` |
| `integrity/es07_final_integrity_report.txt` | 4,462 B | Final integrity re‑check | `INHERITED` |
| `results/experiment7/*.json` (7 files) | 13,581 B total | Historical recorded results: `gate3_real_results`, `_v2_corrected`, `_v3_reproduce`, `_reconciled`, `forensic_ablation_results`, `benchmark_latency_memory`, `verification_threshold_audit` | `INHERITED`, mutually conflicting (§6.9) |
| `execution_evidence/dataset_manifest_query_gallery.npy` | 3,983,184 B | Declared query/gallery manifest array | `MEASURED` (present); contents not re‑decoded |
| `execution_evidence/preflight_probe.json`, `transreid_probe2.json`, `pytest_baseline_prefix.txt` | 14,653 / 11,641 / 1,725 B | Preflight probes and a baseline test record | `INHERITED` |

### 3.9 Reports, manifests and evidence documents (`REP/`, `MAN/`)

| Path | Size | Role | Classification |
|---|---:|---|---|
| `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md` | 67,939 B | Authoritative proposal: requirements, H7a/H7b/H7c, charter quote, τ boundary note `:163`, gates `:196-197` | `INHERITED` |
| `REP/ES-EXP07-TASK7-PROPOSAL-01-REV01.md` | 21,351 B | Earlier proposal, kept because it contains the falsified population claim | historical |
| `REP/ES-EXP07-TASK7-REPORT-01-REV02.md` | 52,908 B | Authoritative results report; §6 flags the false reason string in `not_measured` | `INHERITED` |
| `REP/ES-EXP07-TASK7-REPORT-01-REV01.md` | 10,123 B | Prior results report | historical |
| `REP/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` | 120,496 B | Row‑by‑row execution log (§5 rows 1–43); TR‑01…TR‑11 mapping | `INHERITED` |
| `REP/ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md` | 21,401 B | TR‑01…TR‑11 defect→fix→evidence | `INHERITED` |
| `REP/ES-EXP07-TASK7-INTEGRATION-VERIFICATION-REPORT.md` | 23,995 B | §4.1–§4.7 close‑out, environment probes, 17‑item checklist, the pytest disclosure | `INHERITED` |
| `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md` | 38,038 B | Conflict register CA‑01…CA‑17 plus §9 self‑disclosed tooling defects and §10 may/may‑not‑quote lists | `MEASURED` where re‑read, otherwise `INHERITED` |
| `REP/ES-EXP07-AUDIT-FORENSIC-01.md` | 101,495 B | Forensic audit (pre‑implementation) | historical |
| `REP/ES-EXP07-TRANSREID-PREFLIGHT-REPORT.md` | 34,373 B | Preflight findings | historical |
| `REP/TASK7_DATA_PROVENANCE.md`, `REP/TASK7_WEIGHTS_PROVENANCE.md` | 14,965 / 13,677 B | D‑01…D‑07, W‑01…: measured sizes/digests plus `NOT RECORDED` origin fields | `MEASURED` + `NOT RECORDED` |
| `REP/JETSON-RUN-SHEET-PREP-01-REV01.md` | 28,413 B | Device run sequence; 110 `<fill on device>` tokens; the "40.5×" headline at `:126`; τ‑not‑retuned note `:22` | **NOT RUN** |
| `REP/lab-readiness-checklist.md` | 13,827 B | Lab prerequisites | **NOT RUN** |
| `REP/artifact-inventory.md`, `REP/audit-summary.json`, `REP/test-execution-log.md` | 15,245 / 28,032 / 13,069 B | Inventory and execution bookkeeping | `INHERITED` |
| `REP/preflight_probe.py`, `REP/transreid_probe2.py` | 16,689 / 11,846 B | Probe scripts whose JSON output ships beside them | `IMPLEMENTED` |
| `REP/evidence/*` (17 files) | 42,982 B | OSNet‑preprocessing reproductions (`repro_osnet_BUGGY_BGR.*`, `repro_osnet_full_rgb.json`, `repro_osnet_v2.py`), channel‑order diagnostic, backbone equivalence, integrity re‑check, pytest logs | historical — two mutually inconsistent OSNet repros kept side by side, not merged |
| `REP/audit_20261019_141628/*` (6 files) | 83,841 B | Dated audit folder: forensic report, requirement matrix `.md`, model asset inventory, lab readiness, test ledger, summary | historical |
| `MAN/requirement-traceability-matrix.csv` | 15,600 B | R‑01…R‑42 | `MEASURED` (recounted) |
| `MAN/ES-EXP07-FINAL-DELIVERABLE-MANIFEST.md` | 39,534 B | Delivery manifest | `INHERITED` |
| `MAN/ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md` | 17,005 B | File‑level before/after evidence | `INHERITED` |
| `MAN/ES-EXP07-TASK7-INTEGRATION-HANDOFF-CHECKLIST.md` | 6,871 B | Handoff checklist | `INHERITED` |
| `MAN/FINAL-DELIVERABLE-MANIFEST.md` | 4,919 B | Earlier manifest | historical |

### 3.10 Shared, inherited and other‑team files

* **54 inventory rows** are `BASELINE_PROTECTED` or `OTHER_TEAM_OWNED`; every one that also exists in
  the 2026‑10‑16 archive is byte‑identical to it (`TASK07_STATUS_AND_BLOCKERS.md` §4). `MEASURED`.
* Shared files that Task 7 **read but did not repair**: `algorithm_thresholds.yaml`,
  `core/identity_verifier.py`, `models/osnet_trt_extractor.py`, `models/osnet_backbone.py`,
  `edgesync_coordination/*`, `perception_node.py`.
* `SHARED_DEPENDENCIES/source_snapshot/` holds the shared files at their original paths; Task 7's
  change profile against the archive is **22 created / 4 modified / 34 byte‑identical**.
* Integration‑relevant shared content: `INTEGRATION/TASK07_TASK13_DEPENDENCY_MAP.md`,
  `INTEGRATION/OPEN_INTEGRATION_BLOCKERS.md`,
  `INTEGRATION/ES-EXP07-EXP13-INTEGRATION-AUDIT-20261021_011521.md`.

### 3.11 Status roll‑up

| Category | Count | Notes |
|---|---:|---|
| `IMPLEMENTED` and `TESTED` at Tier U/N | all 8 model modules except the OSNet stubs, the node, the resolver, the overlay, 6 of 10 scripts, 18 test files | Nothing in this category is Tier R |
| `IMPLEMENTED` but `NOT TESTED` on a real dependency | `transreid_trt_extractor.py` (no engine), `build_transreid_trt_engine.py`, `benchmark_transreid_jetson.py`, `ReidSignature.msg` (never compiled), `robot_bringup.launch.py` (never launched) | §10 levels 5–8 |
| `IMPLEMENTED`, never re‑executed | `fit_svd_projection.py` | R‑14 `UNVERIFIED CLAIM` |
| **Stub / placeholder, not a real arm** | `osnet_trt_extractor.py` (returns `zeros` `:75`), `osnet_mock`, the OSNet placeholder backbone (shares 0 of 567 checkpoint keys, 42,816 parameters) | §8.2 T7‑D08/D09 |
| **Planned / proposed, not applied** | Engine build on target; device run arms A1–A7; τ recalibration; crop publisher | §11 |
| **Cited but absent from the package** | `profile_edge_latency.py`; the two `*phase8*`/`*phase9*` latency artefacts; `transformer_120.pth`; the ONNX graphs; the embeddings `.npz`; Market‑1501; `tests/fixtures/` | §12.2 |

---

## 4. Architecture and execution flow

### 4.1 Components and responsibilities

| Component | Responsibility | Depends on | Consumed by |
|---|---|---|---|
| `transreid_backbone.py` | ViT‑B/16 definition and weight loading | `transformer_120.pth` (external), torch | `transreid_extractor.py` |
| `transreid_extractor.py` | Crop → 512‑D unit‑norm `float32` | backbone, projection `.npz`, `reid_input_adapter` | `identity_node.py`, offline scripts |
| `transreid_trt_extractor.py` | Same contract from a TensorRT feature engine | ONNX → engine (absent), `tensorrt`, `pycuda` | `identity_node.py` **only if** `reid_model_type: transreid_trt` |
| `extractor_factory.py` | Maps a string id → a registered class; refuses look‑alikes and mocks on request | registry table | `identity_node.py` |
| `reid_input_adapter.py` | Resize/normalise/channel order | — | both extractors |
| `signature_smoother.py` | EMA over successive embeddings | — | `identity_node.py` |
| `identity_node.py` | Declares params, resolves precedence, builds the extractor, verifies via `IdentityVerify.srv`, optionally publishes `ReidSignature` | all of the above, `rclpy` (absent) | `edgesync_coordination` (Task 13) |
| `reid_launch_defaults.py` | Shared‑then‑overlay parameter resolution | `algorithm_thresholds.yaml`, `exp7_transreid.yaml` | `robot_bringup.launch.py` |
| `robot_bringup.launch.py` | Injects resolved params, coerces the camera id to `int` | resolver | ROS 2 (absent) |
| Offline scripts | Generate embeddings, build the population, compute statistics, export ONNX | the same extractor code | §6 artefacts |

### 4.2 Data flow, as designed

```
sensor_msgs/Image crop  ──►  reid_target_crop_topic        [NO PRODUCER EXISTS — T7-D02]
        └─► reid_input_adapter ─► backbone (768-D) ─► center ─► truncated SVD ─► L2 ─► float32[512]
                └─► signature_smoother (EMA, alpha 0.20)
                        ├─► IdentityVerify.srv  (reference vs candidate, accept iff sim >= tau)
                        └─► ReidSignature.msg   [only when publish_reid_signature := true]
                                └─► edgesync_coordination.Exp13CoordinationNode._on_reid_signature
                                        └─► HandoffRequest.reid_feature_vector
```

The same extractor code is used offline by `generate_task7_embeddings.py`, which is why the offline
numbers in §6 are attributable to the deployed contract and not to a parallel evaluation implementation
(the parity gate at tolerance 2e‑05 over all 19,281 embeddings exists precisely to make that
claim checkable).

### 4.3 Control flow and selection precedence

1. **Selector.** `EDGESYNC_EXPERIMENT_CONFIG=<absolute path to exp7_transreid.yaml>`. Unset ⇒
   `('', 'none')` and the shipped entity list is used byte‑for‑byte unchanged.
2. **Config layer.** `resolve_reid_config()` reads shared `algorithm_thresholds.yaml` first and the
   overlay second, so only overlay keys override; a missing *optional* overlay preserves shared values,
   an explicit missing file raises `FileNotFoundError`, and malformed YAML raises `ValueError` with the
   parse context.
3. **Launch layer.** `reid_camera_id` passes through `coerce_camera_id()` and is injected as a real
   `int`, never as a `LaunchConfiguration` string.
4. **Node layer.** Canonical `reid_model_type` wins over the alias `reid_model`; if both are present and
   resolve differently, the canonical value is used **and** a `WARN` names both keys
   (`identity_node.py:235-252`, conflict message `:240`). With neither set, `DEFAULT_BACKBONE_ID`
   applies and `_log_model_selection()` emits the SILENT‑MOCK GUARD warning.
5. **Registry layer.** `create_extractor()` looks the id up in `BACKBONE_REGISTRY`, builds through a
   lazy `importlib` builder, then asserts **class identity** against the registry entry. A wrong
   concrete class raises `TypeError`; a look‑alike subclass is rejected; `transreid_trt` with no engine
   raises rather than returning a pseudo‑embedding; an `ImportError` becomes a `RuntimeError` that
   states it "will NOT fall back to 'osnet_mock'".

### 4.4 Interfaces and contracts

| Contract | Definition | Producer | Consumer | Verification status |
|---|---|---|---|---|
| 512‑D embedding | unit‑norm `float32[512]` or the zeros sentinel | both extractors | verifier, smoother, `ReidSignature` | `MEASURED` (§6.6) — but three deviation figures are recorded for one quantity (T7‑D14) |
| `ReidSignature.msg` | `sender_id`, `target_id`, `valid`, `float32[512] vector`, `quality`, `stamp_utc_ns`, `model_id` | `identity_node.py:302-312, :394-409` | `exp13_coordination_node.py:535-619` | **Corrected in this pass.** Producer side: `NOT TESTED` — no Task 7 test stubs `ReidSignature` at all (it appears in Task 7 only in `identity_node.py` and in launch text), and the only `edgesync_msgs.msg` stub in the suite installs `HandoffConfirm`/`HandoffCancel` (`test_identity_node_transreid_integration.py:147-151`), so under those stubs `:302` hits the `ReidSignature is unavailable` error branch at `:304-307` and returns. Consumer-side class construction exists only in the Task 13 package (`TASK13/…/tests/test_exp7_exp13_signature_integration.py`). **Never compiled or exchanged over DDS on any host in this evidence set** |
| `IdentityVerify.srv` | crop + two 512‑D vectors → decision | `identity_node.py` service server | any client | `NOT TESTED` at Tier R (`test_identity_service.py` uncollected) |
| Accept rule | `sim ≥ τ`, τ = 0.70 | `core/identity_verifier.py:101` (production); replicated offline | both | `MEASURED` cross‑check, max abs difference 1.7881e‑07 on 200 pairs (seed 11) |
| `exp_id` provenance | overlay → resolver → launch → node → `EXP-METADATA` log; offline `--exp-id` stamps the manifest and gates the analyzer | this task | humans reading logs | `TESTED` (Tier U/N); no machine consumer exists |

### 4.5 Configuration parameters and runtime selectors, exhaustively

| Key | Default | Where declared | Effect |
|---|---|---|---|
| `exp_id` | `'EXP07'` | `identity_node.py:93`, overlay `:45` | Logged in `EXP-METADATA`; stamped into offline manifests; **no consumer** |
| `reid_model_type` | `MODEL_ID_UNSET` | `:87`, overlay `:52` = `transreid` | Canonical selector |
| `reid_model` | `MODEL_ID_UNSET` | `:88`, `MODEL_ALIAS_KEY` | Legacy alias; lower precedence |
| `reid_checkpoint_path` | `''` | `:89`, overlay `:56` | Required by the Torch arm; fail‑fast if missing |
| `reid_projection_path` | `''` | `:90`, overlay `:57` | Selects **by name** over a duplicated namespace (T7‑D04) |
| `reid_engine_path` | `''` | `:91`, overlay `:61` = `""` | TRT arm only; empty by design because no engine exists |
| `reid_camera_id` | `0` (int) | `:96`, overlay `:70` | SIE conditioning label; 0 = camera‑agnostic |
| `publish_reid_signature` | `False` | `:101` | Gates the whole signature path |
| `reid_signature_topic` | `/edgesync/identity/reid_signature` | `:102` | Topic name |
| `reid_target_crop_topic` | `''` | `:103` | **Nothing publishes to any such topic** |
| `reid_signature_target_id` | `''` | `:104` | Target naming for the message |
| `similarity_threshold` | `0.70` | `:79` | τ, mirrored from shared config |
| `smoothing_alpha` | `0.20` | `:80` | EMA weight |
| `enable_spatial_gating` | `True` | `:81` | Mahalanobis gate `NOT MEASURED` offline |
| `robot_id` / `service_name` | `ugv_2` / `/identity/verify` | `:78, :82` | Identity and service address |

Eight arguments are declared by `robot_bringup.launch.py`: `robot_id`, `peer_id`, `is_initial_owner`,
`reid_model_type`, `reid_checkpoint_path`, `reid_projection_path`, `reid_engine_path`, `exp_id`.

### 4.6 Connection to the wider EdgeSync system — confirmed versus proposed

| Link | Status | Basis |
|---|---|---|
| Extractor → `identity_node` selection and logging | **confirmed** at Tier U/N | 23/20/22‑case test files re‑executed or recorded |
| Node → `ReidSignature` publish path | **confirmed in code only**; `TESTED` against stubs | publish guards and calls exist; the flag defaults to false |
| `ReidSignature` → Task 13 intake | **confirmed at Tier N** on generated message classes | `test_exp7_exp13_signature_integration.py` 30 passed (Task 13 run) |
| Task 13 → `HandoffRequest.reid_feature_vector` in a live episode | **proposed only**, never carried data | no crop producer; `exp13_bringup.launch.py:204-208` |
| Anything over DDS / QoS / multi‑process | **not verified** | Tier R blocked |
| Any consumer of `exp_id` (logging, aggregation) | **not verified** | grep over `edgesync_observability` returns zero matches |

### 4.7 Design‑versus‑implementation mismatches

* **Guide file vs guide switch** (T7‑D03): the guide names `transreid_trt_extractor.py` as the strategy
  source while its example switch `reid_model: transreid` instantiates the PyTorch arm.
* **Guide parameter name vs shipped canonical** (T7‑D11): the guide's `reid_model` works only as an
  alias; the overlay sets `reid_model_type`.
* **SIE designed as a capability, deployed as an option**: `reid_camera_id: 0` means the *deployed*
  configuration is the camera‑agnostic arm, while the headline improvement belongs to the SIE arm. The
  SIE base‑class API and `.srv` remain absent (RTM R‑11, gate G.5 open).
* **Overlay selector is an environment variable, not a launch argument** — a deliberate deviation from
  the proposal's suggested wording, recorded with its reason in the integration report.
* **`publish_reid_signature` defaults to false**, so a default bring‑up produces no signature even if a
  crop topic existed.

## 5. Experimental method

### 5.1 Original experimental plan versus the experiments actually executed

| Planned activity | Executed? | Evidence of execution |
|---|---|---|
| Build a TransReID arm that satisfies the 512‑D identity contract | **yes** | Extractor + parity gate + 19,281 embeddings |
| Define an explicit evaluation population (after defect B‑12/P5) | **yes** | `evaluate_offline.py`, `task7_population.json`, the population block in `task7_statistics.json` |
| Compute Rank‑k / mAP for three arms and compare two of them | **yes** | §6.1, §6.4 |
| Compute FNMR / FMR / EER / matched‑TAR at the shipped τ | **yes** | §6.3 |
| Pre‑registered paired inference (McNemar + Holm) | **yes** — family declared before analysis | §6.4 |
| Export ONNX and prove PyTorch↔ONNX parity | **yes** | §6.7 |
| Compare against OSNet on the same population | **no** | `BLOCKED` — charter §3B, `osnet_status: ARM NOT REQUESTED`, `osnet_factory_attempt: FAILED` |
| Build a TensorRT engine on the target | **no** | `BLOCKED` — probe refused; G.7 `DENIED` |
| Measure edge memory, latency, throughput, power on Orin Nano | **no** | `NOT MEASURED`; run sheet preparation‑only |
| Run the identity node on a real graph and publish a signature | **no** | `BLOCKED` — Tier R absent; T7‑D02 |
| Refit or re‑verify the SVD projection | **no** | `UNVERIFIED CLAIM` — charter forbids refitting (R‑14) |

### 5.2 Dataset, version and inventory

| Item | Recorded value | Label |
|---|---|---|
| Dataset | Market‑1501 v15.09.15, `bounding_box_test/` + `query/` under `edgesync_bringup/data/Market-1501-v15.09.15/` | `MEASURED` (counts asserted from the recorded run; the images themselves are **not** in this package) |
| Files / bytes | 68,036 files, 145,885,626 B; 275 MB quoted in RTM R‑20 | `MEASURED` (recorded inventory) |
| Train split | 12,936 images, 751 identities | `MEASURED` |
| Test split | 19,732 images: 3,819 junk (pid −1), 2,798 distractors (pid 0), 750 identities | `MEASURED` |
| Query split | 3,368 images, 750 identities | `MEASURED` |
| Cameras | 1–6 | `MEASURED` |
| Per‑file digests | 61,295 of them | **NOT MEASURED** — never computed |
| Official dataset checksum | none exists (`DATA D-07`) | `NOT MEASURED` |

### 5.3 Population and sample counts actually used

| Rule | Value | Label |
|---|---|---|
| Gallery rule | `pid >= 0` in `bounding_box_test/`: distractor pid 0 **kept** as negatives, junk pid −1 excluded | `MEASURED` |
| Query rule | `pid > 0` in `query/` | `MEASURED` |
| Gallery | 19,732 scanned → **15,913 kept** (3,819 junk excluded), 751 identities, 2,798 distractors, id range [0, 1501] | `MEASURED` |
| Query | 3,368 scanned → **3,368 kept**, 750 identities, id range [1, 1501] | `MEASURED` |
| Embeddings produced | **19,281** rows (= 15,913 + 3,368), exclusion files empty | `MEASURED` |
| Genuine pairs | **49,649** | `MEASURED` |
| Impostor pairs | **53,532,220** | `MEASURED` |
| Duplicate pairs excluded | **13,115** | `MEASURED` |
| Mean genuine candidates per query | 14.741389548693586 | `MEASURED` |
| Mean candidates per valid query | 15,909.105997624703 | `MEASURED` |
| Cross‑view proxy | 1,915 queries from cameras {1,2,3} against 7,752 gallery rows from cameras {4,5,6} | `MEASURED`, **proxy only** |

### 5.4 Arms, baselines and scenario definitions

| Arm id | Role | Output | Deployment status |
|---|---|---|---|
| `transreid_768_agnostic` | descriptive control (pre‑projection feature) | 768‑D, deliberately **not** unit‑norm | not a contract output |
| `transreid_512_agnostic` | deployed configuration (what `reid_camera_id: 0` produces) | 512‑D unit‑norm | `MEASURED` |
| `transreid_512_sie` | deployed capability with camera conditioning | 512‑D unit‑norm | `MEASURED` |
| OSNet baseline | the comparator the charter asks for | 512‑D | `NOT MEASURED` on this host; a historical recorded row exists (§6.10) |
| `osnet_mock` / `osnet_trt` stubs | scaffolding only | zeros / mock | not evidence of anything |

### 5.5 Parameter values, thresholds and settings used

| Setting | Value | Source |
|---|---|---|
| τ (cosine accept threshold) | **0.70**, unchanged | `algorithm_thresholds.yaml → algorithm_thresholds.ros__parameters.reid_similarity_threshold`; overlay header states it is unchanged from the baseline |
| Accept rule | `accept iff sim ≥ τ` | mirrors `core/identity_verifier.py:101` |
| Projection | centered truncated SVD 768→512, `svd_projection_768_to_512_real_v2.npz` | `exp7_transreid.yaml:57`; content `MEASURED` by re‑hash |
| Camera id | 0 (camera‑agnostic) | `:70` |
| Smoothing α | 0.20 | `identity_node.py:80` |
| Parity tolerance | 2e‑05, `every_nth = 1`, abort on violation | embeddings manifest |
| Significance | α = 0.05, family size 3, Holm–Bofinger step‑down | statistics artefact |
| Bootstrap | 2,000 resamples, seed **20261019**, percentile method | statistics artefact |
| McNemar | exact two‑sided binomial on discordant per‑query pairs | statistics artefact |
| Wilson | 95 % intervals for rates | statistics artefact |
| Device run plan | arms A1–A7, four‑metric method, fp16, batch [1,1,1], input contract [1,3,256,128], 4.0 GiB workspace | run sheet §4.2; TRT probe `build_settings` |

### 5.6 Hardware and software environment of every executed step

| Field | Recorded value |
|---|---|
| OS | Windows 10.0.26300 x64 (`host: Windows-10-10.0.26300-SP0`) |
| Interpreter | Python 3.10.11 |
| Torch | 2.5.1+cu121, `cuda_available: true`, `device: NVIDIA GeForce RTX 3050 Laptop GPU` (4.00 GiB) |
| cuDNN | 90100 |
| Absent, probed | `ros2` (exit 127), `rclpy`, `launch_ros`, `ament_index_python` (each exit 1), `tensorrt`, `pycuda` (each exit 1), `trtexec` (null), `nvpmodel`, `jetson_clocks`, `free` (not available); no Jetson markers (`/etc/nv_tegra_release`, `/proc/device-tree/model`, `/etc/nvidia-l4t-version`) |
| Present but deliberately unused | `colcon` shim (a `colcon build` would write trees into the workspace with no middleware, so it was not attempted) |
| Run window | embeddings generated 2026‑10‑19T14:47:29Z in 956.4 s; statistics generated 2026‑10‑19T16:06:52Z; TRT probe 2026‑10‑19T15:09:22Z |

**This is a laptop, not the target.** Every timing in this report is a laptop number; the embeddings
manifest's own `latency_note` says they "must not be reported as Jetson Orin Nano latency", and its
`latency_semantics` field says the numbers are backbone forward only.

### 5.7 Seeds, determinism and reproducibility controls

* Bootstrap seed 20261019; production‑cross‑check seed 11 (200 pairs).
* Parity gate re‑derives center→SVD→L2 independently of the extractor and checks **all** 19,281
  embeddings, aborting the run on any violation — `MEASURED` `PASS`.
* Retrieval protocol validated against torchreid 0.2.5 lineage; canonical‑versus‑per‑query maximum
  absolute difference is **0.0** on all four aggregates — `MEASURED`.
* RTM R‑19 records bit‑identical reproduction of 93.37885985748218 / 85.12520406484964 and of the OSNet
  historical row 94.32897862232778 / 83.5692348350484.
* Asset pinning by SHA‑256 in three independent artefacts; projection content re‑verified in this pass.

### 5.8 Metric definitions used

| Metric | Definition as implemented |
|---|---|
| Rank‑k | Fraction of valid queries whose identity appears in the top k of the ranked gallery; distractors participate as negatives |
| mAP | Mean over queries of the average precision of same‑identity gallery items |
| TAR / FNMR / FMR | Pair‑level: genuine acceptance rate, 1 − TAR, impostor acceptance rate at τ |
| Per‑query decision rates | `genuine_accept_rate` / `false_accept_rate` computed over the 3,368 queries (not over pairs) |
| EER | Point where genuine and impostor acceptance cross; `interpolated: true` in the artefact |
| Matched‑TAR | Acceptance at a threshold matched to a target impostor rate |
| Wilson / bootstrap intervals | 95 % for rates; percentile bootstrap over queries for mAP |
| McNemar | Exact binomial on the discordant pair counts (A‑only, B‑only) per query |
| `camera_partition_proxy` | Cross‑view subset described above; **a proxy, not a handoff measurement** |

### 5.9 Run identifiers, logs and output files

| Run | Identifier / timestamp | Output |
|---|---|---|
| Embedding generation | 2026‑10‑19T14:47:29Z, 956.4 s, `exp_id` default `EXP07` | three external `.npz` + `RES/embeddings/task7_embeddings_manifest.json` |
| Statistics | 2026‑10‑19T16:06:52Z | `RES/statistics/task7_statistics.json` (+ `.md`), `analyze_all_arms.log`, `analyze_arm1_512agnostic.log` |
| ONNX export | two reports (`transreid_onnx_report.json`, `_rerun.json`) | parity tables, graph validation |
| TRT probe | 2026‑10‑19T15:09:22Z | `RES/trt_blocked/probe_build_report.json`, `build_run2.log`, `bench_run2.log` |
| Packaging test runs | 2026‑10‑21 | `../VALIDATION/fresh_evidence/*` (16 entries) |
| Re‑executed Task 7 tests | 2026‑10‑22 | §7.2, captured in this session |

### 5.10 Methodological limitations and bias sources

1. **No comparator arm.** The OSNet arm cannot run here, so nothing in this report is a TransReID‑versus‑OSNet
   result. The only cross‑model figure available is an `INHERITED` recorded row that is not comparable
   cell‑for‑cell (§6.10).
2. **One dataset, one partition rule, one checkpoint.** Generalisation to other corpora or to a robot
   camera geometry is not supported by anything here.
3. **The proxy is not the target.** `camera_partition_proxy` involves no robot, no handoff event, no
   trajectory and no handoff label; charter §8 forbids inferring handoff success from it.
4. **Threshold not chosen from this data.** τ = 0.70 is inherited from the baseline configuration; the
   measured operating curve says the arms sit far from it (§6.3). Reading §6.1 as "the model is bad"
   would be a category error — it is the gate that is mis‑set.
5. **SIE uses a label the deployment must supply.** `reid_camera_id: 0` is camera‑agnostic, so the +4.63 pp
   arm assumes a camera id that a real robot graph does not currently provide.
6. **Projection fit not reproducible from the package.** The fitting script ships but was never re‑run,
   and which run/id range produced the deployed matrix is `NOT RECORDED` (R‑14).
7. **Two variables moved between the historical v1 and the measured run** (population rule and projection
   family), so the historical +22.2090 pp / +33.6580 pp deltas attribute nothing to SIE alone (CA‑03).
8. **mAP implementations differ slightly** across artefacts (6.0588e‑05 absolute, CA‑04), so last‑digit
   differences between reports are not evidence of a different run.
9. **Laptop timings have no warm‑up/repeat protocol** and four mutually inconsistent pairs exist (§6.9).
10. **The Mahalanobis spatial gate is `NOT MEASURED`**; trajectory and covariance data do not exist offline.
11. **Statistics were not re‑executed for this report.** §6 numbers were re‑*read* from the artefact; the
    analyzer was not re‑run over the embeddings (which are not in the package). Reproducing them needs
    the external `.npz` files at their recorded paths.

## 6. Complete numerical results

### 6.0 Provenance of every number in this section, and how it was obtained

**Nothing in §6 was re‑run for this report.** Every figure was **re‑read** in this pass from a packaged
artefact at the field name shown, so each carries `MEASURED` in the sense defined in the Evidence
vocabulary (a number re‑derived by reading a packaged artefact, or by re‑hashing a packaged file) — not
in the sense of a fresh computation. The analyzer, the embedding generator and the ONNX exporter were
**not** re‑executed: the embedding `.npz` files they consume are outside this package (§12.2), so
re‑execution is not possible from the package alone. `task7_statistics.json` was generated
2026‑10‑19T16:06:52Z on host `win32` / Python 3.10.11 / numpy 1.26.4 and its `inputs.integrity` block
records that all three embedding files matched their manifest SHA‑256 digests at the time it ran —
`MEASURED` as a re‑read of that assertion, not as a re‑verification performed here.

Where this report shows an arithmetic step, that figure is `CALCULATED` and the operands are the
packaged values beside it. A `CALCULATED` figure is not a new measurement.

**Re‑verification of the two rows the update request quoted.** The request gave four values and asked
that they be verified against the available artefacts before being presented as verified. All four were
located field‑by‑field in `RES/statistics/task7_statistics.json`:

| Quoted in the request | Field read | Value in the artefact | Verdict |
|---|---|---|---|
| "TransReID 512 agnostic: Rank‑1 88.75 %" | `arms.transreid_512_agnostic.retrieval.rank1_pct` | 88.74703087885986 | **reverified** — rounds to 88.75 % |
| "… mAP 77.32 %" | `…retrieval.mAP_pct` | 77.32400426226894 | **reverified** — rounds to 77.32 % |
| "… FNMR at threshold 0.70 of 60.74 %" | `…verification.at_tau.FNMR` | 0.6073838345183187 | **reverified** — 60.7384 %, rounds to 60.74 % |
| "TransReID 512 SIE: Rank‑1 93.38 %" | `arms.transreid_512_sie.retrieval.rank1_pct` | 93.37885985748218 | **reverified** |
| "… mAP 85.13 %" | `…retrieval.mAP_pct` | 85.12520406508766 | **reverified**, with a caveat below |
| "… FNMR at threshold 0.70 of 48.15 %" | `…verification.at_tau.FNMR` | 0.48154041370420353 | **reverified** — 48.1540 % |

So none of the six is `INHERITED`; each is traceable to a field in a shipped JSON artefact. **One caveat
must travel with the SIE mAP.** The same quantity is stored twice in this package with different last
digits: `task7_statistics.json` gives 85.12520406508766 and
`RES/results/experiment7/gate3_real_results_v2_corrected.json:46` (`accuracy.transreid.mAP`) gives
85.12520406484964.
`CALCULATED` absolute difference = 2.3801760562491836e‑10 percentage points. This is conflict CA‑01 in
`REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`; it is far below any reporting precision but §6.9
records it rather than silently picking one. Also note `paired_tests.tests[H1].rate_B` =
0.9337885985748219 against `retrieval.rank1_pct` = 93.37885985748218 — the same quantity, differing in
the last representable digit of the float→string round trip. Both are reported as written.

One further precision note that affects every threshold comparison in §6.3: the shipped τ is the YAML
literal `0.7`, but the embeddings travel as `float32`, and
`CALCULATED float(np.float32(0.7)) = 0.699999988079071`. The gate and the scores are therefore not in
identical numeric domains. The effect is ~1.2e‑08 and cannot change any rate reported here; it is
recorded because it is the kind of detail a reviewer checking reproducibility will look for.

### 6.1 Retrieval results — all three arms, complete

All cells are `MEASURED` (re‑read) from `arms.<arm>.retrieval` in `task7_statistics.json`. Metric source
for every row: "canonical Market‑1501 CMC/mAP via `evaluate_offline.compute_cmc_map`
(`run_real_pipeline.compute_cmc_map`, validated against torchreid 0.2.5)", max rank 50, 3,368 queries,
3,368 valid, **0** queries with no gallery match.

| Quantity | `transreid_768_agnostic` | `transreid_512_agnostic` | `transreid_512_sie` |
|---|---:|---:|---:|
| Feature dimension | 768 | 512 | 512 |
| Rank‑1 | **88.74703087885986 %** | **88.74703087885986 %** | **93.37885985748218 %** |
| Rank‑5 | 96.02137767220903 % | 96.14014251781472 % | 97.95130641330167 % |
| Rank‑10 | 97.92161520190024 % | 97.8622327790974 % | 98.87173396674585 % |
| mAP | 77.32735493390587 % | 77.32400426226894 % | 85.12520406508766 % |
| Wilson 95 % Rank‑1 | [0.8763531674340077, 0.8977045787189952] | [0.8763531674340077, 0.8977045787189952] | [0.9248870881016115, 0.941701698970831] |
| Wilson 95 % Rank‑5 | [0.9530713855644761, 0.966307546647903] | [0.954352463993793, 0.9673990590113921] | [0.9741542254349896, 0.9837793071604891] |
| Wilson 95 % Rank‑10 | [0.9738241131531012, 0.983516271744176] | [0.9731646367277388, 0.9829894527731354] | [0.9845523639509721, 0.9917687472887542] |
| Bootstrap 95 % mAP (percentile, 2,000 resamples, seed 20261019) | [0.7649130393105973, 0.7819716061490397] | [0.7649125083065337, 0.7817895192625784] | [0.8441732163397601, 0.8585050826141313] |
| Rank‑1 hits | 2,989 | 2,989 | 3,145 |
| Rank 2–5 | 245 | 249 | 154 |
| Rank 6–10 | 64 | 58 | 31 |
| Rank 11–50 | 54 | 59 | 29 |
| Rank > 50 | 16 | 13 | 9 |
| `canonical_vs_per_query_max_abs_difference` | 0.0 | 0.0 | 0.0 |
| `agreement_detail` (rank1 / rank5 / rank10 / mAP) | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |
| Identity coverage | 750 of 750 evaluated | 750 of 750 | 750 of 750 |
| Mean genuine candidates per valid query | 14.741389548693586 | same | same |
| Mean candidates per valid query | 15,909.105997624703 | same | same |

**Internal consistency checks performed on these cells in this pass, all `CALCULATED`:**

* Rank bands sum to the query count in every arm: 2989+245+64+54+16 = 3,368; 2989+249+58+59+13 =
  3,368; 3145+154+31+29+9 = 3,368. ✔
* Rank‑1 × 3,368 reproduces the hit counts exactly: 0.8874703087885986 × 3368 = 2989.0,
  0.9337885985748219 × 3368 = 3145.0. ✔
* Genuine‑pair arithmetic closes: 14.741389548693586 × 3368 = 49,649.0 genuine pairs. ✔
* The Wilson Rank‑1 intervals for the 768 and 512‑agnostic arms are **identical strings**, as they must
  be — both arms produce exactly 2,989 Rank‑1 hits, so the interval depends only on that count.

**Headline comparison, SIE against the deployed camera‑agnostic arm** (`CALCULATED` from the two
`rank1_pct`/`mAP_pct` cells; both arms see the identical population, so the difference is a matched
comparison on 3,368 queries):

| Axis | agnostic | SIE | Δ (SIE − agnostic) |
|---|---:|---:|---:|
| Rank‑1 | 88.74703087885986 % | 93.37885985748218 % | **+4.631828978622323 pp** |
| mAP | 77.32400426226894 % | 85.12520406508766 % | **+7.801199802818715 pp** |
| Rank‑5 | 96.14014251781472 % | 97.95130641330167 % | +1.811163895486947 pp |
| Rank‑10 | 97.8622327790974 % | 98.87173396674585 % | +1.009501187648452 pp |

**The 768‑D control, and an unfavorable detail inside it.** The 768 arm exists to show what the fitted
projection costs. Its Rank‑1 is identical to the 512‑agnostic arm and its Rank‑5 is **lower**
(96.02137767220903 % vs 96.14014251781472 %, `CALCULATED` −0.11876484560569622 pp), while its Rank‑10 is
**higher** (+0.059382422802841006 pp). Its mAP is **higher than the 512‑agnostic arm's** by
`CALCULATED` 0.0033506716369231526 percentage points. That difference is far inside either bootstrap
interval and must not be read as "the projection helps"; it is reported because suppressing it would
misrepresent the direction of a measured cell. The substantive statement the 768 arm supports is the one
its `inferential_role` field already makes: *"descriptive only: 768‑D post‑BNNeck features before the
fitted projection are not a deployed contract output, so no inference is drawn from this arm."* No
hypothesis test was run against it, and it is not part of the declared family (§6.4).

### 6.2 Retrieval by query camera

`MEASURED` from `arms.<arm>.by_query_camera`. Rank‑1 and Rank‑10 are fractions of the valid queries in
that camera; the six cells sum to 3,368 in every arm (`CALCULATED` check: 668+530+717+248+622+583 =
3,368 ✔). These are **descriptive cells, not a pre‑registered comparison** — no hypothesis in the
declared family is scoped to a single camera.

| Query camera | Queries | 768 Rank‑1 | 512 agnostic Rank‑1 | Wilson 95 % (agnostic) | 512 SIE Rank‑1 | Wilson 95 % (SIE) | SIE − agnostic (`CALCULATED`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 668 | 0.8862275449101796 | 0.8877245508982036 | [0.8615324926662346, 0.9094827419536456] | 0.9281437125748503 | [0.9060149613355691, 0.9453763788238351] | +4.0419 pp |
| 2 | 530 | 0.8849056603773585 | 0.8867924528301887 | [0.8569878035885776, 0.9110304782918304] | 0.9433962264150944 | [0.9203451241980689, 0.9600660769819109] | +5.6604 pp |
| 3 | 717 | 0.9470013947001394 | 0.9442119944211994 | [0.9249237467793748, 0.9587657139269227] | 0.9637377963737797 | [0.9473975928791698, 0.9751353606054265] | +1.9526 pp |
| 4 | 248 | 0.8991935483870968 | 0.8951612903225806 | [0.8508213881271017, 0.9274460223215802] | 0.9112903225806451 | [0.8693452449910873, 0.940688182210602] | +1.6129 pp |
| 5 | 622 | 0.909967845659164 | 0.909967845659164 | [0.8848859271296672, 0.9300169413396409] | 0.927652733118971 | [0.9045622974143487, 0.9454932438124589] | +1.7685 pp |
| 6 | 583 | 0.7890222984562607 | **0.79073756432247** | [0.7558677783723147, 0.8218010194001675] | 0.9108061749571184 | [0.8849003121860387, 0.9313337708022179] | **+12.0069 pp** |

Rank‑10 by camera, `MEASURED` from the same block:

| Query camera | 768 | 512 agnostic | 512 SIE |
|---|---:|---:|---:|
| 1 | 0.9790419161676647 | 0.9820359281437125 | 0.9895209580838323 |
| 2 | 0.9792452830188679 | 0.9811320754716981 | 0.9943396226415094 |
| 3 | 0.99721059972106 | 0.99721059972106 | **1.0** |
| 4 | 0.9637096774193549 | 0.9556451612903226 | 0.9596774193548387 |
| 5 | 0.9839228295819936 | 0.9839228295819936 | 0.9887459807073955 |
| 6 | 0.9588336192109777 | 0.9536878216123499 | 0.9811320754716981 |

**What this table shows, and the two bounds on it.** The camera‑agnostic arm's weakest camera is 6
(79.07 % Rank‑1, the only cell whose Wilson interval reaches below 0.83); camera conditioning lifts it to
91.08 %, a `CALCULATED` +12.0069 pp — by far the largest per‑camera gain, and the mechanism is exactly
the one SIE is designed for (camera 6 rows are visually distinctive and the agnostic arm cannot discount
them). Two bounds must be stated with that finding: (i) **it is not in the pre‑registered family**, so
the p‑values in §6.4 do not cover it and no multiplicity‑corrected claim may be made from a single
camera cell; (ii) the cell is **n = 583 queries**, so its Wilson interval (width
`CALCULATED` 0.0659332410278528) is `CALCULATED` 3.088× the pooled interval (width
0.02135141128498752), and the correct reading is "the weakest camera improves most", not "+12.0 pp".
Camera 3's SIE Rank‑10 of exactly 1.0 is the only saturated cell in the table. Note also two
cross‑arm ties (camera 5 Rank‑1 identical between 768 and 512‑agnostic; camera 3 and 5 Rank‑10 identical
between 768 and agnostic), which is expected when the projection preserves the ordering of those
queries.

### 6.3 Verification at the shipped threshold τ = 0.70

All cells `MEASURED` from `arms.<arm>.verification`. The threshold block is identical for all three
arms: `tau` 0.7, `source_file` the absolute path of `algorithm_thresholds.yaml` in the **source
workspace** (not inside this package — the packaged copy is at
`SNAP/edgesync_bringup/config/algorithm_thresholds.yaml`), `source_key`
`algorithm_thresholds.ros__parameters.reid_similarity_threshold`, `matches_expected_0.70` true, and
`applied_by`: *"identity_verifier.verify_candidate rejects when similarity < tau; this script applies
accept iff score >= tau, the same rule"*. `at_tau.scope`: *"appearance gate only; the Mahalanobis
spatial gate is NOT MEASURED offline"*.

**Pair pool** — identical in all three arms because the embeddings, not the population, differ:
genuine 49,649; impostor 53,532,220; duplicate pairs excluded 13,115; junk gallery rows 0; queries
without any impostor candidate 0.

| Quantity | 768 agnostic | 512 agnostic | 512 SIE |
|---|---:|---:|---:|
| TAR at τ | 0.45336260549054364 | 0.3926161654816814 | 0.5184595862957965 |
| **FNMR at τ** | 0.5466373945094564 (**54.66 %**) | 0.6073838345183187 (**60.74 %**) | 0.48154041370420353 (**48.15 %**) |
| Wilson 95 % FNMR | [0.5422550509500236, 0.5510125217394607] | [0.6030802360081928, 0.6116708172390477] | [0.47714693286748505, 0.48593675084233473] |
| FMR at τ | 7.892443093150256e‑05 | 5.206210390676867e‑05 | 5.848814041338095e‑05 |
| Wilson 95 % FMR | [7.658029734057202e‑05, 8.13403129418929e‑05] | [5.016483339119892e‑05, 5.4031126697065774e‑05] | [5.647508082410448e‑05, 6.057295135512326e‑05] |
| Genuine pairs accepted (`CALCULATED` = TAR × 49,649) | 22,509.0 | 19,493.0 | 25,741.0 |
| EER value | 0.020094907328707834 | 0.019718423331789192 | 0.015242166306572005 |
| **τ at EER** | **0.3670562654733658** | **0.30580566823482513** | **0.32755059003829956** |
| EER `interpolated` | true | true | true |
| EER `bracket` | [0.3670562505722046, 0.367056280374527] | [0.30580565333366394, 0.30580568313598633] | [0.3275505602359772, 0.32755061984062195] |
| τ at TAR 0.90 | 0.4969868063926697 | 0.45528265833854675 | 0.5084471106529236 |
| FMR at TAR 0.90 | 0.0029754603862869877 | 0.0028103822333540437 | 0.0013773387317021412 |

The 768 arm's EER bracket, re‑read from the same field, is [0.3670562505722046, 0.367056280374527].
Every EER figure carries `interpolated: true` in the artefact, meaning the crossing was estimated
between adjacent score values, not observed; the bracket gives the two scores it lies between, which is
why the τ at EER should be quoted to no more than ~4 decimals in any downstream use.

**Read the two denominators separately — they are not the same quantity.** FNMR above is **pair‑level**:
of 49,649 genuine (query, gallery) pairs, 48–61 % fall below τ. The artefact also records
**query‑level** decisions, and a query averages 14.741389548693586 genuine candidates, so the two rates
differ a lot and are routinely confused:

| Per‑query block (`verification.per_query_decisions`) | 512 agnostic | 512 SIE |
|---|---:|---:|
| `unit` / denominator rule | query / 3,368 | query / 3,368 |
| `genuine_accept_n` | 3,368 | 3,368 |
| `genuine_accept_rate` (≥ 1 genuine match accepted) | 0.8108669833729216 | 0.8868764845605701 |
| **query‑level miss rate** (`CALCULATED` = 1 − rate) | **18.91330166270784 %** | **11.312351543942989 %** |
| `false_accept_n` | 3,368 | 3,368 |
| `false_accept_rate` | 0.19922802850356294 | 0.20160332541567696 |

The `denominator_rule` is quoted verbatim: *"a query contributes to the genuine rate only if a true
match survives the duplicate exclusion, and to the impostor rate only if an impostor candidate exists;
-inf maxima are never counted as rejections"*. **For the handoff use case the query‑level number is the
decision‑relevant one** — a node acts once per observed target — and the earlier Task 7 reporting
under‑stated it: at τ = 0.70 a single observation of a known target is missed 18.9 % of the time
(agnostic) or 11.3 % (SIE), not 48–61 %.

**Score distributions** (`genuine_score_summary` / `impostor_score_summary`, `MEASURED`):

| 512 agnostic | min | p0.01 | p0.05 | p0.25 | p0.50 | p0.75 | p0.95 | p0.99 | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| genuine (n 49,649) | −0.21013255417346954 | 0.24194055080413818 | 0.3882146656513214 | 0.5612483620643616 | 0.6638650894165039 | 0.7475784420967102 | 0.841523790359497 | 0.8908731055259704 | 0.9661982655525208 | 0.6447243094444275 | 0.1422368437051773 |
| impostor (n 53,532,220) | −0.48266100883483887 | −0.2346451225876808 | −0.17236022651195526 | −0.07860144227743149 | −0.00634026387706399 | 0.07750140875577927 | 0.22865109220147128 | 0.35963937073946006 | 0.9309660196304321 | 0.006389562040567398 | 0.12339857965707779 |

| 512 SIE | min | p0.01 | p0.05 | p0.25 | p0.50 | p0.75 | p0.95 | p0.99 | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| genuine (n 49,649) | −0.19789107143878937 | 0.27673484206199644 | 0.4402404010295868 | 0.6132515668869019 | 0.7057420611381531 | 0.7791037559509277 | 0.8565094947814942 | 0.8969376611709595 | 0.9677219986915588 | 0.6838957667350769 | 0.13302695751190186 |
| impostor (n 53,532,220) | −0.47203391790390015 | −0.23341646939516067 | −0.17165264636278152 | −0.07846481911838055 | −0.006532976636663079 | 0.07722826674580574 | 0.22892300859093662 | 0.3611976745724679 | 0.9339093565940857 | 0.006453799083828926 | 0.12334145605564117 |

| 768 agnostic | min | p0.05 | p0.50 | p0.95 | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| genuine (n 49,649) | −0.0931365117430687 | 0.43778347969055176 | 0.6853140592575073 | 0.848445188999176 | 0.9678992033004761 | 0.668952226638794 | 0.12853474915027618 |
| impostor (n 53,532,220) | −0.3770226240158081 | −0.058788855187594885 | 0.09426480159163475 | 0.300861270725727 | 0.9342625141143799 | 0.10415582358837128 | 0.11057645827531815 |

**This is the central finding of §6 and it is a threshold finding, not a model finding.** The median
genuine similarity is 0.6639 (agnostic) and 0.7057 (SIE) against a gate of 0.70 — the gate sits at or
*above* the centre of the genuine distribution. The impostor distribution is tight and near zero, so
the FMR at τ is ~5e‑05, and the two distributions cross far below the gate: **the empirical EER is at
τ ≈ 0.306–0.367, i.e. the shipped threshold is 0.33–0.40 cosine units above the equal‑error point.**
Reaching TAR 0.90 needs τ ≈ 0.455 (agnostic) / 0.508 (SIE). The 768 control shows the same shape but
with a much higher impostor mean (0.104 vs 0.0064), because its rows are not unit‑normalised and cosine
is computed after internal normalisation — one more reason the 768 arm is descriptive only.

**Distractor rejection at τ** (pid‑0 gallery rows; `retrieval.distractor_rejection`, definition: *"a
pid‑0 gallery row is rejected when its maximum cosine similarity to any query is below tau"*):

| Arm | Rejected / distractors | Rate | Wilson 95 % |
|---|---:|---:|---|
| 768 agnostic | 2,650 / 2,798 | 0.9471050750536097 | [0.9381817639682077, 0.9548023811993427] |
| 512 agnostic | 2,692 / 2,798 | 0.9621157969978557 | [0.9543846977591935, 0.9685797304595457] |
| 512 SIE | 2,686 / 2,798 | 0.9599714081486777 | [0.9520550249339758, 0.9666265057060223] |

`CALCULATED` check: 2692/2798 = 0.9621157969978557 ✔ and 2686/2798 = 0.9599714081486777 ✔. **Here is an
unfavorable result that must not be dropped: camera conditioning makes distractor rejection slightly
worse**, −0.21443888491780028 pp (`CALCULATED`, SIE − agnostic), and the two Wilson intervals overlap
almost completely. SIE buys +4.63 pp Rank‑1 and +7.80 pp mAP at the cost of rejecting 6 fewer known
non‑targets out of 2,798. For a handoff system, where "accept a stranger" is the safety‑relevant error,
that trade must be part of any decision to switch arms — and it is precisely the axis H3 was designed to
test (§6.4).

**Accept‑rule equivalence with production, cross‑checked.** `similarity_cross_check` for each arm:
`status` COMPUTED, `pairs_checked` 200, `sampling` "queries paired with one of their own gallery matches,
seed 11", `max_abs_difference_vs_production_function` = 1.7328268997474083e‑07 (768), 1.7881393432617188e‑07
(512 agnostic), 1.7881393432617188e‑07 (SIE). So the offline statistics apply the same rule as
`core/identity_verifier.py:101` to within float32 rounding — this is a **rule** cross‑check, not a claim
that the production runtime was executed (§10 level 6).

### 6.4 Paired inference — the pre‑registered family

`MEASURED` from `paired_tests`. Family declared before analysis, `family_size` 3, correction
*"Holm‑Boferoni step‑down, alpha = 0.05, monotone‑adjusted"*, test *"exact McNemar (binomial, two‑sided)
on the discordant pairs"*, unit = query, `paired_units` = 3,368 in all three tests. The 768 arm is
**absent from the family by design**, consistent with its `inferential_role`.

| | H1 — Rank‑1 hit | H2 — genuine accept at τ | H3 — false accept at τ |
|---|---|---|---|
| H₀ (verbatim) | "the two 512‑D arms have the same Rank‑1 hit rate" | "the two arms accept the same fraction of queries with at least one genuine match at tau" | "the two arms falsely accept the same fraction of queries at tau" |
| rate A (agnostic) | 0.8874703087885986 | 0.8108669833729216 | 0.19922802850356294 |
| rate B (SIE) | 0.9337885985748219 | 0.8868764845605701 | 0.20160332541567696 |
| difference A − B | −0.04631828978622332 | −0.07600950118764849 | −0.002375296912114022 |
| discordant A‑only | 5 | 3 | 73 |
| discordant B‑only | 161 | 259 | 81 |
| raw p | 2.1803905931263096e‑41 | 8.090130605382558e‑73 | 0.572845968999707 |
| Holm‑adjusted p | 4.360781186252619e‑41 | 2.4270391816147673e‑72 | 0.572845968999707 |
| decision | **REJECT H0 at alpha=0.05** | **REJECT H0 at alpha=0.05** | **FAIL TO REJECT H0 at alpha=0.05** |

**Discordance arithmetic re‑performed here (`CALCULATED`, and it confirms the recorded differences
field‑for‑field):** (161 − 5)/3,368 = 0.04631828978622328 vs recorded 0.04631828978622332;
(259 − 3)/3,368 = 0.07600950118764846 vs recorded 0.07600950118764849; (81 − 73)/3,368 =
0.0023752969121140144 vs recorded 0.002375296912114022. Each agrees to float summation order, which is
the strongest internal check available without the embeddings. The Holm multipliers are visible as
2× and 1× (`CALCULATED`: 2.1803905931263096e‑41 × 2 = 4.360781186252619e‑41;
8.090130605382558e‑73 × 3 = 2.4270391816147673e‑72 — the step‑down ordering is by raw p, so the
multipliers are (3‑rank+1) with monotonicity enforced).

**What H3's outcome does and does not mean.** "Fail to reject" is **not** evidence that the two arms
false‑accept equally: the test has 154 discordant queries and the point estimate is SIE worse by
0.2375 pp. The correct statement is that the package contains no statistically detectable false‑accept
penalty for the better arm at α = 0.05 with this family size and this sample — a power‑limited
non‑finding, recorded as such. H1/H2 rejected with enormous margins because McNemar on (5, 161) and
(3, 259) discordance is decisive; neither p‑value should be read as an effect size, only as a decision.

### 6.5 Cross‑view proxy (`camera_partition_proxy`) — proxy only

`MEASURED` from `arms.<arm>.camera_partition_proxy`. The artefact's own `status` field, quoted verbatim,
governs how the whole subsection must be read:

> "PROXY ONLY — a camera partition on one fixed dataset. Charter §8: cross‑robot handoff success,
> false‑handoff rates and cross‑robot performance must NOT be inferred from Market‑1501 retrieval
> metrics; no robot, handoff event or handoff label is involved."

Top‑level `camera_partition_disclaimer`: *"A camera partition on one fixed dataset. It measures whether
appearance identity survives a viewpoint change between two camera groups — the property a handoff
depends on — but involves no robot, no handoff event, no trajectory and no handoff label."* Both
directions were computed, which is why they are reported together here.

**Retrieval on the proxy subsets**

| Partition | Arm | Queries | Valid | Invalid (no gallery match) | Gallery rows | Rank‑1 | Rank‑5 | Rank‑10 | mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cams {1,2,3} → {4,5,6} | 512 agnostic | 1,915 | 1,843 | **72** | 7,752 | 0.8990775908844275 | 0.9669017905588714 | 0.9799240368963646 | 0.8127527269246213 |
| cams {1,2,3} → {4,5,6} | 512 SIE | 1,915 | 1,843 | 72 | 7,752 | 0.9376017362995117 | 0.9815518176885513 | 0.9896907216494846 | 0.8774181085441133 |
| cams {4,5,6} → {1,2,3} | 512 agnostic | 1,453 | 1,449 | **4** | 8,161 | 0.9130434782608695 | 0.9675638371290545 | 0.9813664596273292 | 0.8278217686835891 |
| cams {4,5,6} → {1,2,3} | 512 SIE | 1,453 | 1,449 | 4 | 8,161 | 0.9420289855072463 | 0.9772256728778468 | 0.9875776397515528 | 0.8760147396397886 |

Identity coverage: `{1,2,3}`→`{4,5,6}` — 748 query identities, 720 with a gallery match; reverse — 722
query identities, **also 720** with a gallery match. `canonical_agreement_abs` is 0.0 for rank1/rank5/
rank10/mAP in all four cells. Wilson Rank‑1 intervals: agnostic [0.8844843798041044,
0.9120106268324213] / [0.8974213350561866, 0.9264813645347467]; SIE [0.9256226778000926,
0.9477603580290824] / [0.9287870501993302, 0.9529333826962302]. Wilson Rank‑10: agnostic
[0.9724514938768023, 0.9854000810654238] / [0.9730246705866707, 0.9871626860794547]; SIE
[0.9839541212145723, 0.9933901935229733] / [0.9804489156211488, 0.9921279553245396].

`CALCULATED` proxy gains (SIE − agnostic): forward direction **+3.85241454150842 pp Rank‑1,
+6.466538161949198 pp mAP**; reverse direction **+2.898550724637683 pp Rank‑1, +4.8192970956199455 pp
mAP**. The gain is real in both directions and of the same order as the pooled +4.63 pp — which is the
point of the proxy: the improvement is not an artifact of same‑camera matching.

**Verification legs of the proxy** (`camera_partition_proxy.<partition>.verification_at_tau`), pair pools
being subsets of the §6.3 pool:

| Quantity | fwd agnostic | fwd SIE | rev agnostic | rev SIE |
|---|---:|---:|---:|---:|
| genuine pairs | 16,497 | 16,497 | 13,473 | 13,473 |
| impostor pairs | 14,828,583 | 14,828,583 | 11,844,460 | 11,844,460 |
| TAR at τ | 0.3971631205673759 | 0.5177911135357944 | 0.4088176352705411 | 0.5195576337860907 |
| FNMR at τ | 0.6028368794326241 | 0.48220888646420557 | 0.591182364729459 | 0.48044236621390934 |
| Wilson FNMR | [0.595347047670102, 0.6102788295597065] | [0.4745889032618282, 0.4898371533447449] | [0.5828563048983824, 0.5994564430483464] | [0.47201281367563824, 0.4888830682258977] |
| FMR at τ | 4.552019569233284e‑05 | 5.570323206202508e‑05 | 4.280482183231654e‑05 | 5.0994304510294265e‑05 |
| Wilson FMR | [4.22133494317222e‑05, 4.9086076015585276e‑05] | [5.2031915176684845e‑05, 5.9633577734020214e‑05] | [3.923757613019688e‑05, 4.669636502955188e‑05] | [4.708653794094245e‑05, 5.522636326264956e‑05] |
| τ at EER | 0.2994057387113571 | 0.32438476383686066 | 0.3068487197160721 | 0.3260488212108612 |
| EER | 0.01953902136165 | 0.015101982434869198 | 0.018184517182513127 | 0.015006340517001197 |
| τ at TAR 0.90 | 0.4549185037612915 | 0.5130487084388733 | 0.4587642252445221 | 0.5069551467895508 |
| FMR at TAR 0.90 | 0.0025214816547204813 | 0.0012177832500920689 | 0.0023968167396403043 | 0.001290645584517994 |

The τ = 0.70 pathology of §6.3 reproduces inside the proxy in all four cells (FNMR 48–60 %, EER at
τ ≈ 0.30–0.33), which localises it to the score distribution rather than to the population choice. In
both directions the FMR rises slightly under SIE (forward 4.55e‑05 → 5.57e‑05; reverse 4.28e‑05 →
5.10e‑05) while **FMR at matched TAR 0.90 falls** (forward 2.52e‑03 → 1.22e‑03; reverse 2.40e‑03 →
1.29e‑03) — at a recalibrated threshold SIE is strictly better on the impostor side; at τ = 0.70 it is
marginally worse, and both readings have to be reported.

**Two asymmetries a reviewer should notice.** First, invalid queries are **72** in the forward direction
and **4** in the reverse: `{1,2,3}`→`{4,5,6}` has 1,915 queries over 7,752 gallery rows, and 72 of them
have no same‑identity row in that gallery, while the reverse has only 1,453 queries over 8,161 rows.
Because the denominators differ, no rate should be averaged across the two directions. Second, the
proxy's own `protocol_note` states that *"the same‑pid‑same‑camera duplicate exclusion is inert here
because the two camera groups are disjoint by construction"* — so the 13,115 excluded duplicate pairs of
§6.3 do **not** apply to these four cells, and the proxy genuine/impostor counts must not be reconciled
against the pooled pool by simple subtraction.

### 6.6 Embedding‑math conformance — row counts, norms, the zeros sentinel, and T7‑D14

This subsection is the evidence behind the claim that the 512‑D vectors actually satisfy the interface
contract the rest of the pipeline assumes. All figures are `MEASURED` (re‑read for this report) from
`RES/embeddings/task7_embeddings_manifest.json` (per‑arm block) and
`RES/statistics/task7_statistics.json`, path `arms.<arm>.population.<side>`.

| Arm | Side | rows | `zero_rows` | `non_finite_rows` | `unit_norm_max_deviation` | `mean_norm` |
|---|---|---|---|---|---|---|
| `transreid_512_agnostic` | gallery | 15,913 | 0 | 0 | 1.3248880670335694e‑07 | 1.000000000199223 |
| `transreid_512_agnostic` | query | 3,368 | 0 | 0 | 1.2477772171060053e‑07 | 1.0000000000182114 |
| `transreid_512_sie` | gallery | 15,913 | 0 | 0 | 1.3504487839099966e‑07 | 1.0000000009672065 |
| `transreid_512_sie` | query | 3,368 | 0 | 0 | 1.1966979007738132e‑07 | 0.9999999999591991 |
| `transreid_768_agnostic` | gallery | 15,913 | 0 | 0 | `null` | `null` |
| `transreid_768_agnostic` | query | 3,368 | 0 | 0 | `null` | `null` |

**The 768‑D nulls are correct, not missing.** The statistics artefact records an explicit note in the
768‑D cells: *"raw post‑BNNeck 768‑D features, deliberately not unit‑norm; cosine is computed after
normalisation"* (`…transreid_768_agnostic.population.gallery.norm_note` and `.query.norm_note`). The
768‑D arm is the control that isolates the projection's cost; it never enters the 512‑D contract, so a
unit‑norm assertion would be a category error. Reporting those two cells as "normalisation unverified"
would be a defect of the reader, not of the data.

**T7‑D14 resolved — a field‑name collision, not a numeric conflict.** The per‑arm block of the embeddings
manifest also carries a field named `unit_norm_max_deviation`, and it records **1.7881393432617188e‑07 for
both 512 arms**, with `mean_norm` exactly `1.0` and `any_zero_row: false`. Two tools therefore appear to
give "the maximum unit‑norm deviation" three different values for the agnostic arm alone
(1.7881e‑07 / 1.3249e‑07 / 1.2478e‑07), which is how this item entered the register as an unresolved
conflict. The two quantities are demonstrably **not the same statistic**:

* `CALCULATED` — the manifest value is exactly 3 × 2⁻²⁴ (1.7881393432617188e‑07 ÷ 5.9604644775390625e‑08
  = 3.0 exactly), i.e. an integral number of float32 units in the last place, which is what a maximum
  taken on a float32 array looks like. None of the four per‑side values is: 1.3248880670335694e‑07 ÷ 2⁻²⁴
  = 2.2228, 1.2477772171060053e‑07 ÷ 2⁻²⁴ = 2.0934, 1.3504487839099966e‑07 ÷ 2⁻²⁴ = 2.2657,
  1.1966979007738132e‑07 ÷ 2⁻²⁴ = 2.0077 — consistent with float64 arithmetic over float32 vectors, as a
  per‑side maximum recomputed in the analyzer would produce.
* `MEASURED` — the manifest value is **identical for the agnostic and the SIE arm**, while the four
  per‑side statistics values differ from one another in every cell. If the manifest field were the
  maximum of the same quantity, the two arms could not share one number.

Largest disagreement between any two recorded figures: **4.632512762281493e‑08** (`CALCULATED`,
1.7881393432617188e‑07 − 1.3248880670335694e‑07). Every figure is below 2⁻²² and two orders of magnitude
below the 2e‑05 parity tolerance, so **no measured result anywhere in §6 changes under any of them.** The
corrective action is documentation — rename or annotate the two fields so they are not read as one
quantity — not a re‑measurement. Severity: cosmetic for the science, material for a reviewer who diffs
artefacts. Status: **analysis `RESOLVED`, fix `OPEN`** — the fix touches code and is outside this
report's authorisation (§2, item 5).

**Sentinel path.** The base extractor contract requires either a unit‑norm 512‑element float32 vector or
the all‑zeros sentinel, and forbids raising. `MEASURED`: `any_zero_row: false` in all three arms, with
`extractor_failure_counts: {}` and `extractor_last_failure: null` — so across all 19,281 extractor calls
(`extractor_embedding_count: 19281`) the sentinel was **never returned**. That is a positive result about
this run and a gap in the evidence about the contract: the sentinel branch is exercised only by unit tests
(§7.1), never by production‑path execution. **`TESTED` at unit level; `NOT MEASURED` on any real graph.**
The OSNet stub reaches the same sentinel by a different route — silently, with no failure to record
(§6.10).

**Internal parity gate of the generation run.** `parity_check` records `every_nth: 1` with
`images_cross_checked: {gallery: 15913, query: 3368}` — 100 % of both sides, not a sample —
`tolerance: 2e‑05`, and result *"PASS — for every image checked, the production extract_embedding() output
equalled the independently re‑derived center->SVD->L2 contract within the tolerance; a violation aborts
the run"*. **Read its own `note` before quoting this as a correctness proof:** *"the check compares the
deployed extractor against this script's own implementation of the documented math contract, on the same
decoded crop."* It is **self‑referential** — it proves two code paths agree, not that either is right.
The independent numerical check of the exported graph is §6.7; the independent check of the metric is the
canonical‑implementation agreement in §6.1. **`MEASURED`, with that limitation stated in the same breath.**

**Checkpoint coverage.** `extractor_checkpoint_coverage` (all `MEASURED`): 211 checkpoint keys → 210 after
filter, dropped `["classifier.weight"]`; `model_parameter_entries: 206`; `missing_keys: []`;
`unexpected_keys: ["classifier_1.weight" … "classifier_4.weight"]`; `coverage_pct: 100.0`;
`torch_parameters: 100769512`. `CALCULATED` reconciliation against §6.8: 100,769,512 parameters plus
5 classifier heads × 751 × 768 = 2,883,840 → **103,653,352 trainable elements**, exactly the quantity
behind `gate3_real_results_reconciled.json`'s 395.406 MiB figure. The five heads are the 751‑class training
exposure (§5.2, §9 R‑12); four of them are *unexpected* by the model definition and are silently discarded
by `strict=False` loading — recorded here, not repaired.

### 6.7 ONNX export, graph inventory and PyTorch ↔ ONNXRuntime parity

`MEASURED` (re‑read) from `RES/onnx/transreid_onnx_report.json` (generated 2026‑10‑19T14:54:12Z) and
`RES/onnx/transreid_onnx_report_rerun.json` (15:11:35Z). **These are two independent executions of the
same export, and they agree exactly at graph level**: identical byte lengths, identical SHA‑256 digests,
identical node counts and identical parity summaries. `CALCULATED` — the only fields that differ between
the two files are `exports.<arm>.seconds`: agnostic 5.3 s vs 9.9 s, SIE 2.7 s vs 5.2 s. Export *content*
is bit‑repeatable; export *timing* is not (≈1.9× / ≈1.9× spread on the same host minutes apart), which is
the third independent demonstration that no timing figure in this project can support a speed claim
(§6.8.1, §6.9 CA‑07).

| Field | `agnostic` | `sie` |
|---|---|---|
| file bytes | 343,380,183 | 343,399,114 |
| SHA‑256 | `318b6857585d5d7c4882a42156a7080c11be37f84a53957b05fb1b0a9aa82b22` | `2cd8ccc774c5933771e5b6b653d593de3862faddfd59b471b9cecccf4bda4362` |
| checker | `PASS (onnx.checker.check_model)` | `PASS (onnx.checker.check_model)` |
| IR version / opset | 8 / `ai.onnx` 17 | 8 / `ai.onnx` 17 |
| `n_nodes` | 779 | 783 |
| `n_initializers` | 154 | 155 |
| `initializer_float32_bytes` | 343,249,920 | 343,268,352 |
| `external_data_tensors` | `[]` — single file, no side‑car | `[]` |
| `input_contract_violations` | `[]` | `[]` |
| graph inputs | `image FLOAT [batch, 3, 256, 128]` | `image FLOAT [batch, 3, 256, 128]`, `cam_label INT64 [batch]` |
| graph output | `feature_768 FLOAT [batch, "BatchNormalizationfeature_768_dim_1"]` | same |

`CALCULATED` from the two byte counts: the SIE graph is **18,931 B larger**, with **4 more nodes** and
**1 more initializer**. The recorded op histograms agree with that exactly: `Constant` 205 → 206, `Add`
85 → 86, `Gather` 74 → 75, `Mul` 37 → 38, and every other op type unchanged. That is the whole of the SIE
mechanism as exported — one embedding‑table lookup, one add, and their plumbing. Dominant op types
(agnostic, `MEASURED`): `Constant` 205, `Add` 85, `Gather` 74, `Unsqueeze` 73, `MatMul` 72, `Shape` 39,
`Mul` 37, `Transpose` 37, `Concat` 27, `Reshape` 26, `LayerNormalization` 25, `Cast` 24, `Div` 24, `Erf`
12, `Softmax` 12, then a tail of singletons that includes **`BatchNormalization` 1** (the BNNeck) and
**`Conv` 1** (the patch‑embedding projection). `CALCULATED` reading: 72 `MatMul` ≈ 12 blocks × 6
projections, and the 12 `Softmax` / 12 `Erf` / 25 `LayerNormalization` counts show the attention blocks
were traced as unfused LayerNorm‑plus‑MatMul rather than as fused attention. Offered as an inventory, not
as an architectural claim.

**Symbolic output width — a build‑time risk, recorded rather than assumed.** The exported feature width is
traced through the final BatchNormalization, so the declared output dimension is the symbolic string
`BatchNormalizationfeature_768_dim_1`, not the constant 768. `MEASURED`: ONNXRuntime resolves it —
`feature_width_observed: 768` for both arms, `dynamic_batch_1` → *"OK, output shape (1, 768)"*,
`dynamic_batch_2` → *"OK, output shape (2, 768)"*, and `off_contract_128x64` → *"refused by the graph:
InvalidArgument"*. Refusing an off‑contract input is the desired direction of failure. The report's own
`deployment_notes.reading` states that *"the TensorRT parser has to resolve the same dim during the engine
build"*. **`NOT MEASURED`: whether any given TensorRT parser version resolves it — no engine has ever been
built (§6.8.4).** The three recorded `deployment_notes.engine_build_requirements`, quoted:

1. *"an explicit optimization profile pinned to the contract batch/dims (1x3x256x128, cam_label 1) — no
   implicit shapes"*;
2. *"a build on the Orin Nano itself with the matching TensorRT major version; engines are not portable
   across GPU architectures or TRT majors"*;
3. *"if the parser cannot resolve the symbolic width the build must fail;
   TransReIdTrtExtractor._validate_engine_contract() additionally refuses any engine whose feature output
   is not 768 wide, so a wrong‑width engine cannot silently be loaded and double‑projected"*.

Item 3 is `IMPLEMENTED` and `TESTED` at unit level only (§7.1); it has never rejected a real engine because
none exists.

**Preprocessing and projection stay outside the graph.** `shape_assumptions.preprocessing`: *"resize
256x128, /255, mean=std=[0.5,0.5,0.5], HWC->CHW, RGB uint8 in"*; `shape_assumptions.projection`: *"SVD
768->512 stays OUTSIDE the graph (fitted artefact, own provenance); the graph ends at the post‑BNNeck
768‑D CLS feature"*; `shape_assumptions.pos_embed`: the checkpoint carries `(1, 211, 768)` and 256×128
with patch 16 / stride 12 yields exactly 210 patches + CLS, *"so no interpolation path is exercised"*.
**Two consequences a reviewer must hold together:** (i) an engine built from these graphs emits **768‑D**,
so the 512‑D contract is still satisfied by Python post‑processing — the ONNX artefact is **not** by itself
a deployable embedding producer, and the SVD matrix remains an unversioned dependency of the deployment
(R‑14, §9); (ii) the resolution‑interpolation path that any other camera geometry would need is
**`NOT TESTED` anywhere in this project** — the project's position is that other resolutions are
off‑contract rather than silently re‑shaped, which is safe but means a robot camera that does not produce
256×128 crops has no supported path.

**Parity numbers.** `parity` block: `population: directory`, `n_images_selected: 8`, `n_images_scored: 8`,
`providers: ["AzureExecutionProvider", "CPUExecutionProvider"]`, `reference_device: cpu`,
`test_device: "onnxruntime CPUExecutionProvider (fp32)"`, `tolerance: 0.002`, `cosine_floor: 0.9999`.

| Quantity | `agnostic` | `sie` |
|---|---|---|
| `worst_max_abs_768` | 1.239776611328125e‑05 | 1.2874603271484375e‑05 |
| `worst_max_abs_512_after_contract` | 6.09084963798523e‑07 | 6.854534149169922e‑07 |
| `worst_cosine_768` | 0.9999999999960024 | 0.9999999999954788 |
| verdict | **PASS** | **PASS** |
| `production_api_vs_torch_worst_max_abs_512` | **0.0** | **0.0** |

`MEASURED`. The PASS is not marginal: `CALCULATED`, the worst 768‑D absolute deviation consumes
1.2874603271484375e‑05 ÷ 0.002 = **0.64 % of the tolerance budget** on the tighter‑looking SIE arm, and the
worst cosine sits **5.5e‑12 above** the 0.9999 floor. The `0.0` in the last row is the strongest figure in
the block — the deployed extractor's post‑contract 512‑D output equals the PyTorch reference **bit for
bit** on every scored image, which is what makes any graph‑vs‑Python difference attributable to the export
itself rather than to the contract code.

**What the parity sample is, and is not.** `MEASURED` from the eight per‑image records:
`0000_c1s1_000151_01`, `_000376_03`, `_000801_03`, `_001051_02`, `_001376_04`, `_002301_05`, `_004476_02`,
`_011501_05` — **all eight have `pid 0`** (the Market‑1501 distractor class, identities deliberately not in
the query set), **all eight have `camid 1`**, and the per‑image `cam_label` field reads 0 for all eight.
So this gate exercises no genuine identity, no camera other than camera 1, and the artefact does not state
which camera label was fed to the SIE graph — that last point is recorded here as a documentation gap
(new item, §8 T7‑D19), not as an inferred error. `CALCULATED` from the eight per‑image
`agnostic_vs_sie_arm.l2_dist` values — 0.1061849358238373, 0.11928638236796092, 0.11159768515056046,
0.10189158630079757, 0.12830445934048054, 0.15214425314786348, 0.13757900885524305, 0.11999567602355396 —
the minimum is 0.10189158630079757 and the maximum is 0.15214425314786348, which *is* the recorded
`sie_effect_worst_l2_on_512`. The arms genuinely differ by 0.10–0.15 L2 on unit‑normalised vectors, so SIE
is not a no‑op graph attribute. It is also not a *semantic* check: the SIE effect is verified to exist,
never verified to be the intended effect, because the parity sample contains no identity pairs and no
camera variety.

The artefact fences its own reading, and the fence must travel with the number (`parity.caveat`): *"fp32
Torch‑vs‑fp32 ONNXRuntime‑on‑CPU numerics check. It validates the exported graph and the
preprocessing/projection contract; it is not a TensorRT result, an fp16 result, a speedup, or a Jetson
measurement."* `parity.ORT_is_CPU_only: true`. `not_claimed`, verbatim: `tensorrt_engine` — *"BLOCKED —
Pending Jetson Orin Nano access (no tensorrt, trtexec or JetPack on this host)"*; `target_latency` — *"NOT
MEASURED — an ONNX/CPU parity check is not a speedup claim"*; `accuracy_parity_on_device` — *"NOT MEASURED
— requires the TensorRT arm on the target"*; `graph_validation` — *"all exported graphs passed
onnx.checker and the input-contract checks"*.

**Scope note on the artefacts themselves.** The graphs were exported to a directory **outside this
package** (`C:\Users\PC\edgesync_es07_runs\onnx_trt09\…`); only the JSON reports are inside. So the digests
above are re‑verified as *recorded* values and the `.onnx` files are **`NOT VERIFIABLE` from this package**
(§12.2). The rerun report's `invocation.argv` records `--parity-n 8`, `--reference-device cpu`,
`--out-dir C:/Users/PC/edgesync_es07_runs/onnx_trt09_rerun`, with the working directory
`…/edgesync_identity`.

### 6.8 Performance — latency, memory, throughput, power, and the `NOT MEASURED` register

**Charter §7 governs this entire subsection: *"a laptop benchmark is not a Jetson benchmark."*** Nothing
below is an edge performance result, and the one timing figure this pass produced of its own accord is
disclosed as bookkeeping rather than as a measurement.

**6.8.1 The one latency figure produced by this pass, and why it is excluded.** `MEASURED` from
`RES/embeddings/task7_embeddings_manifest.json`, per‑arm `latency_mean_ms`:

| Arm | `latency_mean_ms` | Semantics recorded in the same file |
|---|---|---|
| `transreid_768_agnostic` | 15.776978528054041 | backbone forward only |
| `transreid_512_agnostic` | 15.776978528054041 | **the identical value** — the 512‑D arms reuse the 768‑D forward |
| `transreid_512_sie` | 15.989644214496506 | backbone forward only, camera embedding active |

`latency_semantics`, verbatim: *"backbone forward only; the 512‑D arms reuse the same forward, the SVD
matvec is not timed separately"*. `latency_note`, verbatim: *"latencies_ms is incidental bookkeeping on a
4 GB laptop GPU without a warm‑up/repeat protocol. It is NOT a deployment benchmark and must not be
reported as Jetson Orin Nano latency."* `CALCULATED` from the two values: the camera embedding costs
**+0.212665686442465 ms**, i.e. **+1.3479 %** of the forward pass. **The meaning of that number is
narrow:** it is the cost of the one `Gather` and one `Add` inventoried in §6.7, measured in an untimed
single‑pass loop, with the 768→512 projection *excluded* from the timed region. It is not a per‑crop
service time — the §2.4 budget (`T_txn = 886 ms`) covers crop encode + match + verify inside a real graph.
**This report therefore does not offer the ratio 15.777 ÷ 886 = 1.78 %**; the arithmetic is trivial, the
quantity is meaningless on a laptop GPU, and quoting it is precisely the failure mode charter §7 exists to
prevent. Hypothesis H7b is answered **`NOT MEASURED`**, and the exclusion of this figure from it is
recorded here rather than silently dropped (§11 A‑6).

**6.8.2 Model size — four recorded figures, all of which reconcile exactly.** `MEASURED` (re‑read from
`RES/results/experiment7/benchmark_latency_memory.json`, the reconciled artefact in the same directory, and
`RES/embeddings/task7_embeddings_manifest.json → assets`). MiB (2²⁰) and decimal MB
(10⁶) are both shown because the artefacts print **MiB under an `mb` suffix**, and that has already caused
one misread (§6.9, CA‑06).

| Quantity | TransReID ViT‑B/16 | OSNet x1.0 | Where the number comes from |
|---|---|---|---|
| checkpoint file | 414,724,314 B = 395.511926651001 MiB = 414.72 MB | 10,399,605 B = 9.91783618927002 MiB = 10.40 MB | `assets.*.bytes`; sha‑256 `84305529…` / `2809d322…` |
| trainable elements | 103,653,352 | 2,554,771 | `CALCULATED`: reconciled `model_size_*_mb` × 2²⁰ ÷ 4 |
| classifier heads | 5 × 751 × 768 = 2,883,840 | 751 × 512 + 751 = 385,263 | `CALCULATED` from head shapes (CA‑15, §9 R‑12) |
| trainable − heads | **100,769,512** | **2,169,508** | `CALCULATED` |
| `parameters` field | **100,769,512** ✔ exact match | **2,169,508** ✔ exact match | `benchmark_latency_memory.json` |
| `parameter_mb` | 384.4051818847656 = 100,769,512 × 4 B ✔ (= 403.08 MB) | 8.276016235351562 = 2,169,508 × 4 B ✔ | `CALCULATED` check |
| `model_size_*_mb` | 395.4061584472656 = 103,653,352 × 4 B ✔ (= 414.61 MB) | 9.745677947998047 = 2,554,771 × 4 B ✔ | `CALCULATED` check |
| file − trainable × 4 | 110,906 B | 180,521 B | `CALCULATED` (container + key names) |

**Resolution of the "384 vs 395" conflict** recorded in `REP/JETSON-RUN-SHEET-PREP-01-REV01.md` Appendix A:
the two are different **definitions**, both exact — 384.405 MiB is float32 mass of the **backbone
parameters**, 395.406 MiB is the **whole trainable state dict including the 751‑class heads**, 395.512 MiB
is the **file**. Once the definition is named, all three are quotable; no re‑measurement is needed. The same
applies to `parameter_mb` vs `checkpoint_file_size_mb`.

**The three ratios that "40.5×" could mean** (`CALCULATED`, all three now measurable from files in this
package):

| Definition | Ratio | Recorded operand |
|---|---|---|
| trainable elements, heads included | 395.4061584472656 ÷ 9.745677947998047 = **40.5725** | the actual operands of `parameter_ratio` in `gate3_real_results_reconciled.json:62` |
| parameters, heads excluded | 100,769,512 ÷ 2,169,508 = **46.4481** | `benchmark_latency_memory.json → parameters` |
| checkpoint file bytes | 414,724,314 ÷ 10,399,605 = **39.8789** | `assets.*.bytes` |

`gate3_real_results_reconciled.json` keys its 40.5× as **`parameter_ratio`** while its operands are the
*with‑heads sizes*, and `JETSON-RUN-SHEET-PREP-01-REV01.md:126` asserts *"40.5× OSNet **in parameter
mass**"*. **`MEASURED` conflict, `RESOLVED` by definition: the run sheet's phrase is the wrong definition
for its own number** (parameter mass is 46.448×, 14.5 % higher). No recorded artefact states which
definition it intended. Consequence: no ratio may be quoted in the device report without naming its
definition, and the run sheet's headline must be corrected before the device run — the device report is
expected to restate it (§11 A‑7).

**6.8.3 Peak runtime memory ranks the *smaller* model higher.** `MEASURED` from
`benchmark_latency_memory.json → peak_runtime_gpu_memory_mb`: TransReID **410.7265625**, OSNet
**413.26025390625**. `CALCULATED`: OSNet's recorded peak is **2.53369140625 MiB larger**, in the *same
artefact* that records TransReID as 46.4× the parameters. **This row is the proof that the field is
unusable, not a result to average.** A 4 GiB laptop GPU's peak reading is dominated by CUDA context and
allocator arena, which are near‑constant for two models that both fit; the figure measures the driver, not
the model. The prior audit reached the same conclusion independently (`REP/ES-EXP07-AUDIT-FORENSIC-01.md`:
*"memory methodology is not a footprint"*). Consequently the edge memory figure **must** come from
`tegrastats` MEM + process `VmRSS` + `torch.cuda.max_memory_allocated()` on the Orin Nano, which is
exactly what `REP/JETSON-RUN-SHEET-PREP-01-REV01.md` specifies. **`NOT MEASURED`; no laptop peak may be
substituted, and this row is the documented reason.**

**6.8.4 The complete `NOT MEASURED` register.** Every quantity the charter or the proposal asked for that
has no measurement, with the recorded text that establishes the status. The **statuses** below are
`MEASURED` (re‑read from the artefacts named); the **quantities** are not measured.

| Quantity | Status | Recorded reason, and where |
|---|---|---|
| Jetson TensorRT latency | **`BLOCKED`** | `task7_statistics.json → not_measured.jetson_tensorrt_latency`: *"BLOCKED — Pending Jetson Orin Nano access (TR‑11)"*. `RES/trt_blocked/bench_run2.log` (202 B): *"BLOCKED — Pending Jetson Orin Nano access: none of /etc/nv_tegra_release, /proc/device-tree/model, /etc/nvidia-l4t-version exist on this host. Charter §7: a laptop benchmark is not a Jetson benchmark."* |
| Jetson throughput (crops/s) | **`NOT MEASURED`** | no field for it exists in **any** artefact in this package — this is an absence, not a refusal |
| Jetson power, nvpmodel, jetson_clocks | **`NOT MEASURED`** | `RES/trt_blocked/probe_build_report.json → environment`: `nvpmodel` *"NOT AVAILABLE: nvpmodel not on PATH"*, `jetson_clocks` *"NOT AVAILABLE: jetson_clocks not on PATH"*, `free_memory` *"NOT AVAILABLE: free not on PATH"*, `gpu_busy_percent` *"cat: /sys/class/devfreq/17000000.gpu/gpu_busy_percent: No such file or directory"* |
| TensorRT engine | **`BLOCKED`** | same report, `verdict`: *"BLOCKED — this host presents none of /etc/nv_tegra_release, /proc/device-tree/model, /etc/nvidia-l4t-version. Charter §6 step 6 requires the engine to be built in the actual target environment; re-run on the Orin Nano, or pass --allow-non-target for a bring-up build that is not a deployment artefact."* Owner gate **G.7 = DENIED**. **No `.engine`, `.plan` or `.trt` file exists anywhere in the project** (`MEASURED` by exhaustive search, §3.2). |
| fp16 accuracy parity on device | **`NOT MEASURED`** | `transreid_onnx_report.json → not_claimed.accuracy_parity_on_device` |
| OSNet retrieval on this population | **`BLOCKED`** | `not_measured.osnet_retrieval_on_this_population`: *"BLOCKED on this host — the frozen baseline cannot be imported (torchvision::nms); §3B forbids repairing it"* → §6.10 |
| Mahalanobis spatial gate | **`NOT MEASURED`** | `not_measured.spatial_gate_mahalanobis`: *"NOT MEASURED — requires per-robot trajectory and covariance data that do not exist offline"* |
| Cross‑robot handoff success, false‑handoff rate | **`NOT MEASURED`** and **`NOT INFERABLE`** | `not_measured.cross_robot_handoff_success`: *"NOT MEASURED and NOT INFERABLE — no robot or handoff dataset exists in this project"*; charter §8; `camera_partition_disclaimer` in the same file |
| Paired test against the recorded gate3 run | **`NOT APPLICABLE`** | `not_measured.statistical_comparison_against_recorded_gate3_numbers`: *"NOT APPLICABLE — the recorded run used the invalid id_band population (TR‑07), so its per-query outcomes are not the same population as this one"* |
| ROS 2 graph execution, `ReidSignature` publication | **`BLOCKED`** | no `rclpy`/DDS on this host; defect T7‑D02 (§8, §10) |
| Installed‑space asset paths (`models/`, `data/`) | **`BLOCKED`** | `SNAP/edgesync_identity/setup.py` installs `config/` only (§4.5, §9 R‑16) |
| Real‑robot / end‑to‑end safety | **`BLOCKED`**, **`NOT TESTED`** | no robot, no crop publisher (§10, levels 6–10) |
| Message generation against a real RMW | **`BLOCKED`** | message classes generated from on‑disk `.msg` under a stub `rclpy` only (Tier N, §7.1) |

Two further points that this register has to make explicitly, because both are the kind of gap that
reads as a result if it is left unstated:

* **A device run alone will not close the latency question.** `benchmark_latency_memory.json` records for
  TransReID a **full‑pipeline median of 58.718050000607036 ms** against a **backbone median of
  56.770099996356294 ms** — a **1.94795 ms** pipeline overhead (`CALCULATED`). No Jetson analogue of that
  decomposition exists, so *the overhead component of the edge budget stays unmeasured even after a
  successful device run* unless the harness times backbone and contract stages separately. The run sheet
  specifies that split; it has never been executed.
* **The engine build probe records its own settings**, so a re‑run on the Orin Nano is a like‑for‑like
  re‑run and not a fresh design decision: `build_settings` = precision **fp16**,
  `batch_min_opt_max` **[1, 1, 1]**, `image_shape_contract` **[1, 3, 256, 128]**, `workspace_gib` **4.0**,
  `verbose` false, input ONNX = the agnostic graph at 343,380,183 B / sha `318b6857…`. Its
  `environment` block also nails the host that refused: node **`LAPTOP-2B5ROUGO`**, Windows/AMD64,
  Python 3.10.11, torch 2.5.1+cu121, CUDA 12.1, cuDNN 90100, `tensorrt` and `pycuda` both
  *"NOT IMPORTABLE: No module named…"*, `trtexec_on_path: null`, `is_jetson_marker_present: false`,
  `device_tree_model: null`, `l4t_nv_tegra_release: null`.

### 6.9 Historical recorded results — `RES/results/experiment7/`, and why they are not merged

These seven files are the only numerical results in the package that pre‑date the corrected population
work. They are reproduced here in full because they are what most earlier Task 7 documents quoted, and
because **§1–§5 of this report and the whole of the Task 7 proposal REV02 exist partly to fence them
off.** Nothing in this subsection is merged into §6.1–§6.5. Each conflict is labelled with its `CA‑nn`
identifier from `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md`, so that this report, the audit and the
proposal can be cross‑checked cell for cell; the audit's own verdicts are restated, not silently
overridden.

**6.9.1 Census, timestamps and generator availability.** `MEASURED` (re‑hashed in this pass):

| File | bytes | SHA‑256 (first 16) | own timestamp | producer on disk? |
|---|---|---|---|---|
| `benchmark_latency_memory.json` | 1,125 | `6becced684fa1493` | **none** | **no** — no script in `SNAP/` writes this filename (`MEASURED` by exhaustive grep of the source snapshot) |
| `forensic_ablation_results.json` | 2,180 | `a283f66d554c9b94` | **none** | **no** |
| `gate3_real_results.json` | 1,557 | `fe42f9885db05c80` | **none** | **not in its current form** — `run_real_pipeline.py:40` now writes `_v2_corrected`; its own comment at `:38` says the earlier copy "produced `gate3_real_results.json`", so the v1 generator was overwritten |
| `gate3_real_results_reconciled.json` | 2,263 | `564323622381700a` | `2026‑10‑15T18:15:00Z` | **no** |
| `gate3_real_results_v2_corrected.json` | 3,247 | `81c90c3b45b0e9e0` | **none** | `SNAP/…/scripts/run_real_pipeline.py` — **but unusable** (§6.9.6) |
| `gate3_real_results_v3_reproduce.json` | 1,667 | `915f55547649ed20` | **none** | **no** |
| `verification_threshold_audit.json` | 1,542 | `f80fd7b9ebd941d0` | **none** | **no** |

Total **13,581 B** across 7 files (`MEASURED`; this is the figure that corrects the arithmetic slip in
this report's own §3.8). `MEASURED`: **six of the seven carry no timestamp and no host name** — their only
device field is the string `"cuda"` — so the *when* and *where* of every recorded accuracy number in this
directory is unknown except for `_reconciled` (2026‑10‑15) . **Four of the seven have no producer script on
disk at all.** Under the brief's rule — *"a reported metric is not automatically reproducible simply
because the report includes its value"* — these are **`INHERITED`**, and where a claim in them cannot be
reached from any file or script in the package it is **`UNVERIFIED CLAIM`**.

**6.9.2 Every recorded retrieval number, in one table.** Percent, as written in each artefact.
`MEASURED` (re‑read). The three rows at the bottom are §6.1's own measured arms, repeated here only so the
comparison is visible in one place.

| Artefact | condition as named there | dim | Rank‑1 | Rank‑5 | Rank‑10 | mAP |
|---|---|---|---|---|---|---|
| `gate3_real_results.json` | `accuracy.transreid` (centred SVD, no SIE declared) | 512 | 71.16983372921615 | 85.80760095011877 | 90.70665083135393 | 51.46723674145579 |
| `gate3_real_results.json` | `accuracy.osnet` | 512 | 94.32897862232778 | 97.80285035629454 | 98.60451306413302 | 83.5692348350484 |
| `_v2_corrected.json` | `accuracy.transreid` (`transreid_sie_applied: true`) | 512 | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | **85.12520406484964** |
| `_v2_corrected.json` | `accuracy.osnet` | 512 | 94.32897862232778 | 97.80285035629454 | 98.60451306413302 | 83.5692348350484 |
| `_v3_reproduce.json` | `accuracy.transreid` (SIE true, `purpose`: *"re‑measure every accuracy value from raw pixels; no cached features"*) | 512 | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | **85.12520406484964** |
| `_v3_reproduce.json` | `accuracy.osnet` | 512 | 94.32897862232778 | 97.80285035629454 | 98.60451306413302 | 83.5692348350484 |
| `_reconciled.json` | `baseline_reported_results.transreid_centered_svd_512` | 512 | 71.16983372921615 | 85.80760095011877 | 90.70665083135393 | 51.46723674145579 |
| `_reconciled.json` | `baseline_reported_results.osnet_x1_0_native_512` | 512 | 94.32897862232778 | 97.80285035629454 | 98.60451306413302 | 83.5692348350484 |
| `_reconciled.json` | `corrected_ablation.transreid_native_post_bn_768_no_cam` | 768 | **88.64** | 95.12 | 96.88 | 74.82 |
| `_reconciled.json` | `corrected_ablation.transreid_native_post_bn_768_with_cam_sie` | 768 | **89.91** | 95.84 | 97.35 | 76.54 |
| `_reconciled.json` | `corrected_ablation.transreid_uncentered_svd_512` | 512 | **86.42** | 93.88 | 95.91 | 71.95 |
| `forensic_ablation_results.json` | A1 OSNet x1.0 native 512‑D | 512 | 94.32897862232778 | 97.80285035629454 | 98.60451306413302 | 83.5692348350484 |
| ″ | B1 TransReID native 768‑D **post‑BN**, no cam, no SVD | 768 | 70.78384798099762 | 85.24346793349169 | 90.38004750593825 | 50.281406681665075 |
| ″ | B2 TransReID native 768‑D **pre‑BN**, no cam, no SVD | 768 | 71.1104513064133 | 85.57007125890736 | 90.52850356294537 | 50.464564767195874 |
| ″ | C1 768‑D post‑BN **with** cam/SIE | 768 | 74.22802850356295 | 88.15320665083135 | 92.48812351543944 | 54.306307365142715 |
| ″ | C2 768‑D pre‑BN **with** cam/SIE | 768 | 74.58432304038006 | 88.47980997624703 | 92.30997624703087 | 54.590397164372995 |
| ″ | D1 512‑D SVD from post‑BN, no cam (*"[ORIGINAL REPORTED PIPELINE]"*) | 512 | 71.16983372921615 | 85.80760095011877 | 90.70665083135393 | **51.46717615365994** |
| ″ | D2 512‑D SVD from pre‑BN, no cam | 512 | 71.28859857482185 | 85.74821852731591 | 90.64726840855107 | 51.545910104716256 |
| ″ | D3 512‑D SVD from post‑BN **with** cam/SIE | 512 | 74.58432304038006 | 88.56888361045131 | 92.63657957244655 | 55.64255117457915 |
| ″ | D4 512‑D truncated, raw slice, no SVD | 512 | 70.54631828978623 | 84.94655581947742 | 90.17220902612827 | 49.72219036784214 |
| *this pass* | `transreid_768_agnostic` | 768 | 88.74703087885986 | 96.02137767220903 | 97.92161520190024 | 77.32735493390587 |
| *this pass* | `transreid_512_agnostic` | 512 | 88.74703087885986 | 96.14014251781472 | 97.86223277909740 | 77.32400426226894 |
| *this pass* | `transreid_512_sie` | 512 | 93.37885985748218 | 97.95130641330167 | 98.87173396674585 | 85.12520406508766 |

**6.9.3 The one genuine reproduction, and it is exact.** `CALCULATED` + `MEASURED`: this pass's
`transreid_512_sie` **equals `gate3_real_results_v3_reproduce.json → accuracy.transreid` to the last
digit in Rank‑1, Rank‑5 and Rank‑10** (93.37885985748218 / 97.95130641330167 / 98.87173396674585), with
mAP differing by **2.3801760562491836e‑10** pp (85.12520406484964 vs 85.12520406508766, conflict **CA‑01**).
`gate3_real_results_v2_corrected.json` carries the same four numbers as v3, and its `pipeline_config`
declares the *same* preprocessing this pass used: `transreid_pixel_mean/std` = [0.5, 0.5, 0.5],
`transreid_sie_applied: true`, `transreid_sie_xishu: 1.0`, `transreid_neck_feat: "after (post‑BNNeck)"`,
`image_size_hw: [256, 128]`, and `svd_variance_retained: 0.9864118099212646` — the deployed matrix (§6.9.6).
A **third** independent execution is preserved in this package at
`REP/evidence/repro_transreid_both_arms.json`: `WITH_CAM_SIE` = 93.37885985748218 / 97.95130641330167 /
98.87173396674585 / 85.12520406484964 with `reproduced_bit_identical: true`, `n: 19281`, 342.3 s,
`zero_rows: 0`; and `NO_CAM_deployed_path` = 88.74703087885986 / 96.14014251781472 / 97.8622327790974 /
77.32400426241038, 367.5 s — which is §6.1's `transreid_512_agnostic` arm reproduced by *another* route,
again with Rank‑1/5/10 identical and mAP differing by **−1.4144e‑10** pp (`CALCULATED` against
77.32400426226894). **So the SIE arm's rank triple appears identically in four independent places, and the
agnostic arm's in two.** That reproduction discipline is what this report's `MEASURED` label rests on, and
it is the strongest pipeline‑correctness evidence available without the device: extractor, `[0.5]³`
normalisation, post‑BNNeck CLS head, SIE path, pinned checkpoint `84305529…` and pinned projection
`9626bea5…` together regenerate a recorded result exactly.

Two disclosures travel with that file, and both are in it verbatim. (i) Its own `provenance` says the
*script's* `json.dump` failed on a literal `/tmp/…` path under Windows, so `REAL_EXIT = 1`, and the numbers
were transcribed from the captured stdout in `repro_transreid_both_arms.log` — the failure was in the
output‑writing step only, not in inference or evaluation, and the file records that distinction explicitly
(`audit_note_on_exit_code`). (ii) It also names the deployed‑runtime limitation, which is a finding about
production rather than about the experiment: `why_deployed_path` — *"identity_node.py:266‑272 calls
extract_embedding(crop) with no cam_label; IdentityVerify.srv carries no camera field;
base_identity_extractor.py signature has no cam_label parameter"* (T7‑D01, §8). **What the reproduction
does not license** is any paired OSNet‑versus‑TransReID test: the recorded `accuracy/osnet` cells are
aggregates with **no per‑query outcomes on disk**, and §6.10 records why the OSNet arm cannot be re‑run
through the production extractor here.

**CA‑03 — the v1 → v2 "correction" moved several things at once.**
`CALCULATED`: the v1→v2 gap is **+22.209026128266032** pp Rank‑1 and **+33.65796732339385** pp mAP. The two
runs differ in at least three declared ways: (i) the **projection asset** (v1 `svd_variance_retained`
0.9857138991355896 vs v2/v3 0.9864118099212646 — different `.npz` contents, §6.9.6); (ii) **camera/SIE** —
the forensic table names v1's condition as D1 *"512‑D SVD from Post‑BN (No Cam) [ORIGINAL REPORTED
PIPELINE]"* and its rank triple is **byte‑identical to v1**, so 71.1698 is *no camera*; v2 declares
`transreid_sie_applied: true`; (iii) v2 adds a warm‑up/timed protocol
(`latency_warmup_iterations: 50`, `latency_timed_iterations: 200`) that v1 does not record. Only one cause
is named in the historical narrative: `gate3_real_results_reconciled.json:19` attributes the whole gap to
*"Corrupted by Centered Truncated‑SVD mean subtraction and omitted JPM"*, while the 93.379 arm that
replaced it uses the **same centred** projection — `MEASURED` from
`RES/provenance/task7_provenance_measurements.json`: the deployed matrix's mean vector has
`mean_l2 = 10.372914292029659`, i.e. it *is* centred. **The label contradicts the asset it condemns**, and
the camera embedding is not mentioned in that root‑cause string although D1→D3 shows it is worth only
+3.414 pp on the forensic scale. `CALCULATED`: within the forensic family, D1→D3 = **+3.414489311163905**
pp. **Verdict (audit CA‑03, restated): the recorded attribution is incomplete and, as written, not
supportable.** The +22.21 pp figure therefore attributes nothing to SIE alone — and this pass's controlled
measurement of the same effect, with everything else held fixed, is **+4.631828978622323** pp (§6.1, §6.4).
Any statement of the form *"the correction gained +22 pp"* is unsupported by this evidence set; the
defensible statement is the controlled pair.

**CA‑11 — one named arm, three mutually inconsistent values on disk.** For the condition
*"TransReID native 768‑D post‑BN, no camera/SIE"*:

| Source | Rank‑1 | mAP | Provenance |
|---|---|---|---|
| `forensic_ablation_results.json` B1 | 70.78384798099762 | 50.281406681665075 | no generator on disk, no preprocessing declared |
| `_reconciled.json → transreid_native_post_bn_768_no_cam` | **88.64** | 74.82 | no generator; **2‑decimal precision, unlike every measured value in the tree** |
| this pass `transreid_768_agnostic` | 88.74703087885986 | 77.32735493390587 | `MEASURED`, generator `generate_task7_embeddings.py`, exit 0 |

`CALCULATED` gaps: measured − forensic = **+17.96318289786224** pp; measured − reconciled =
**+0.10703087885986** pp. The reconciled figure is the only one the new pass comes near, and 0.107 pp is
**far outside that row's own rounding resolution** (±0.005 pp), so the two are *not* the same measurement —
plausibly the same arm under different feature extraction, since the 768‑D array in this pass is raw
post‑BNNeck and deliberately not unit‑norm (§6.6). The forensic row cannot be reconciled at all without
its preprocessing, which it does not record. **Verdict: this arm has no established value on disk.**
Proposal REV02 §2 already refuses to quote 88.64/89.91 as measured; this audit confirms the refusal was
required. The same arm‑identity problem applies to the SIE effect size, recorded four ways:

| Reading | SIE Rank‑1 effect | Population / control | Has a test? |
|---|---|---|---|
| `_reconciled.json` (768‑D, 88.64 → 89.91) | **+1.27** pp (`CALCULATED`) | undeclared, 2‑dp | no |
| `forensic_ablation_results.json` B1→C1 (post‑BN, 768‑D) | **+3.4441805225653326** pp (`CALCULATED`) | undeclared in the file | no |
| `forensic_ablation_results.json` D1→D3 (post‑BN, 512‑D) | **+3.414489311163905** pp (`CALCULATED`) | undeclared in the file | no |
| this pass `transreid_512_agnostic` → `transreid_512_sie` | **+4.631828978622323** pp; mAP **+7.801199802818715** pp | declared: directory population, 3,368 × 15,913, per‑query unit | **yes** — exact McNemar, Holm‑corrected, §6.4 |

Only the fourth has a declared population, a declared pair pool, a per‑query unit and a hypothesis test.
Note also `forensic_ablation_results.json` B2→C2 = **+3.47387173396676** pp: within the forensic family the
SIE effect is stable to ~0.03 pp, which is *internal consistency*, not external validity — that file
records no population, no script and no timestamp.

**CA‑04 — same run, different mAP code.** `D1` in `forensic_ablation_results.json` (mAP 51.46717615365994)
and `accuracy.transreid` in `gate3_real_results.json` (mAP 51.46723674145579) share **all three rank
values byte for byte** and differ by **6.058779585060847e‑05** pp in mAP (`CALCULATED`). Identical CMC with
a different mAP means the two artefacts were produced by **different mAP implementations over the same
ranked lists** — the rank‑k columns of the historical table are shared, the mAP columns are not.
Consequence for reading this report: last‑digit differences in *mAP* between Task 7 documents are **not**
evidence of a different run; last‑digit differences in Rank‑1 are. This is why §6.0 shows both mAP values
rather than picking one, and why the brief's quoted "mAP 77.32 %" is traced to a named field.

**Units defect in `_reconciled.json`** (`CALCULATED`, new in this pass): its three
`delta_rank1_vs_baseline` strings — `"+17.47 %"`, `"+18.74 %"`, `"+15.25 %"` — are **percentage‑point
differences against 71.16983372921615**, and they reproduce exactly as pp: 88.64 − 71.16983372921615 =
17.47016627, 89.91 − 71.16983372921615 = 18.74016627, 86.42 − 71.16983372921615 = 15.25016627. Printed with
a `%` sign, they invite the reading "improved by 17.47 % relative", which would be 83.60 Rank‑1. They are
absolute pp. **Defect: units label; corrective action: re‑label as pp; §11.**

**6.9.4 CA‑07 — the recorded latencies disagree in sign, and the disagreement cannot be settled offline.**
`MEASURED` (re‑read field‑by‑field in this pass) + `CALCULATED` ratios. All five rows are nominally the same
GPU class, all laptop, none the target device. ms:

| Artefact | TransReID median | OSNet median | TR ÷ OS (`CALCULATED`) | what the artefact records about its own method |
|---|---:|---:|---:|---|
| `gate3_real_results.json` | 13.431399998808047 | 18.876250000175787 | **0.711550** → TR *faster* | **no** warm‑up / timed / iteration field of any kind; note *"Laptop measurement (NOT Jetson Orin Nano 8GB)"* |
| `…_v2_corrected.json` | 41.71514999688952 | 28.256749999854947 | **1.476290** → TR *slower* | `latency_warmup_iterations 50`, `latency_timed_iterations 200`, `includes_svd_projection true`, `projection_mean_ms 0.5822154997804319` |
| `benchmark_latency_memory.json` | 58.718050000607036 (full pipeline) | 28.16894999705255 (full pipeline) | **2.084496** → TR *slower* | no protocol fields; **no script on disk writes this file** (§6.9.1) |
| `…_reconciled.json → latency_audit` | 13.488097500157892 (`mean_ms`) | 21.397280499913904 (`mean_ms`) | **0.630365** → TR *faster* | byte‑identical copies of **v1's** `mean_ms` (`MEASURED`: the same two floats appear in both files) → the "reconciled" latency row inherits the oldest, superseded run |
| this pass (`task7_embeddings_manifest.json`) | 15.776978528054041 (768 and 512‑agnostic, one shared forward) / 15.989644214496506 (SIE) | — | n/a | self‑declared *"backbone forward only"* and *"incidental bookkeeping … without a warm‑up/repeat protocol. It is **NOT** a deployment benchmark"* |

Four consequences that §6.8 and §13 both carry:

1. **The sign flip is real and recorded.** TransReID is the faster model in two artefacts and the slower
   model in two others. Any sentence of the form "TransReID is *x*× slower than OSNet" is therefore an
   artefact choice, not a measurement, unless it names its source row.
2. **The frozen baseline alone spans 1.49×.** OSNet's own recorded "backbone" latency takes three values —
   18.876250000175787 (v1 median), 24.78759999939939 (`benchmark_latency_memory.json` backbone median,
   ratio **1.313163** to the first), 28.19380000073579 (v2 backbone median, **1.493612**) — so the numerator
   of any TR/OS ratio is less well determined than the denominator.
3. **v2's distribution is warm‑then‑throttled, not noisy.** `CALCULATED`: v2 min 14.266999998653773 ≈ v1
   median 13.431399998808047 (v2 median ÷ v2 min = **2.923891**), while v2's p99 43.006137003467295 sits
   just above its own median — a bimodal‑ish timing history consistent with what is *inside* the timed
   region plus host clock state, not with a GPU speed difference. But v2 also labels 38.76890200019261 ms
   as TransReID's *backbone mean* against this pass's 15.776978528054041 ms forward‑only mean for the same
   model (`CALCULATED` **2.457308×**), which this reading does **not** explain. **Status: `UNRESOLVED`,
   deliberately left unresolved — settling it needs the device harness, not a further laptop run**
   (charter §7: *"a laptop benchmark is not a Jetson benchmark"*).
4. **This pass's number is a fifth value and is excluded from H7b by that same clause.** The exclusion is
   recorded here rather than quietly dropped. The only latency comparison this pass *does* license is the
   internal one: SIE costs `CALCULATED` 15.989644214496506 − 15.776978528054041 = **0.212665686442465 ms**,
   **+1.3479493938862275 %** relative, **forward pass only**, on a 4 GiB laptop GPU. It is not a deployment
   latency and must not be quoted as one.

**CA‑08 — two of the six medians that Task 7 has historically cited are not on disk.** `MEASURED` (searched
in this pass). The prior audit's Appendix A row cites `gate3_real_results.json`, `phase9` and `phase8` for
OSNet medians 18.88 / 19.74 / 20.71 and TransReID 13.43 / 13.59 / 13.84. A search of the whole package for
`*phase8*` and `*phase9*` returns **nothing**; `gate3_real_results_v3_reproduce.json → latency` is the
string *"not measured here -- see out/phase8_latency\_\*.json (idle GPU, 3 trials)"*, and no `out/`
directory exists under `edgesync_bringup`. So **exactly two of the six are traceable** (18.88 → v1 OSNet
`median_ms` 18.876250000175787; 13.43 → v1 TransReID `median_ms` 13.431399998808047) and the other four are
`UNVERIFIED` — their files are absent, not merely unreconciled. Proposal REV02 retained that row unchanged
and fenced it; this audit supplies the missing detail of *why* the fence is required.

**6.9.5 CA‑09 / CA‑10 — τ is consistent and machine‑checked; the recorded verification numbers are not
attributable.** `MEASURED` (whole‑file reads) + `CALCULATED` differences.

*CA‑09, threshold.* `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml` key
`algorithm_thresholds.ros__parameters.reid_similarity_threshold` = **0.7**; the analyzer **refuses to run**
if the loaded τ ≠ the expected 0.7 (a τ of 0.62 was tested and refused — execution log §5 row 21); the
applied rule is accept iff `sim ≥ τ`, which matches `SNAP/…/edgesync_identity/core/identity_verifier.py:101`
(rejects when `sim < τ`). Every recorded artefact agrees on the number: `threshold_tau` / `tau` = 0.7 in
`gate3_real_results.json`, `_v2_corrected`, `_v3_reproduce`, `_reconciled` and
`verification_threshold_audit.json`. v2/v3 additionally state *"tau is not applied anywhere in this
script"* — so **no recorded Rank‑k value implies anything about τ**, and a reviewer must not read the
recorded accuracy table as a threshold operating point. **Verdict: consistent.** One residual numeric
mismatch travels with every τ comparison in §6.3: the shipped literal is `0.7` (float64) while the
embeddings are `float32`, and `CALCULATED float(np.float32(0.7)) = 0.699999988079071`, a difference of
~1.2e‑08 that cannot change any rate reported here.

*CA‑10, the recorded verification audit.* `verification_threshold_audit.json` is 1,542 B with four model
rows and the fields `model, genuine_mean, genuine_std, impostor_mean, impostor_std, tau, fnmr_pct,
fmr_pct, eer_pct, eer_tau` — and **nothing else**. It records no population, no pair counts, no camera rule,
no split, no gallery or query count, no producing script and no timestamp; no script in
`SNAP/…/edgesync_identity/scripts/` writes it (§6.9.1). Its FMR values are `0.11399999999999999` and
`0.013999999999999999` ×3 — the float32 signature of exactly‑rounded ratios, which implies a **fixed
sampled impostor count that is nowhere stated**. Classify all four rows `UNVERIFIED CLAIM`.

| Metric | Recorded (pool undeclared) | This pass (pool declared: 49,649 genuine / 53,532,220 impostor pairs, 3,368 queries) | Comparable? |
|---|---|---|---|
| FNMR @ 0.70 | OSNet 18.703290940081292 %; TR 768 + cam 42.84122197456405 %; TR 512 centred + cam **48.6626458633801 %**; TR 512 uncentred + cam 38.51448800314671 % | 768‑agnostic **54.66373945094564 %**; 512‑agnostic **60.73838345183187 %**; 512‑SIE **48.154041370420353 %** | sign only |
| FMR @ 0.70 | 0.114 % / 0.014 % | 0.007892443093150256 % / 0.005206210390676867 % / 0.005848814041338095 % | **no** — pool undeclared |
| EER | 2.051920807657008 % @ 0.555 (OSNet); 1.678248328307329 % @ 0.33 (TR centred) | 2.0094907328707834 % @ 0.3670562654733658; 1.9718423331789192 % @ 0.30580566823482513; **1.5242166306572005 % @ 0.32755059003829956** | sign only |
| genuine mean / median | TR centred 512: 0.682079017162323 | 512‑SIE: 0.6838957667350769 / 0.7057420611381531 (`n` 49,649) | **no** |

The closest recorded‑to‑measured pair in the whole package is the centred‑512‑with‑camera row:
`CALCULATED` Δ FNMR 48.6626458633801 − 48.154041370420353 = **0.5086044929597477** pp; Δ EER
1.678248328307329 − 1.5242166306572005 = **0.15403169765012859** pp; Δ genuine mean
0.6838957667350769 − 0.682079017162323 = **+0.0018167495727539**; Δ impostor mean
0.006453799083828926 − 0.007322206627577543 = **−0.0008684075437486172**. Those are **near‑agreements in
sign and magnitude**, and they are informative — but they are **not** reproductions in the CA‑01 sense,
because the recorded side declares no pool, so equality of the numbers cannot be attributed to equality of
the populations. The refusal in proposal §6 and in report §4.4 to difference against
`verification_threshold_audit.json` is therefore **required, not stylistic**.

*Population sensitivity.* Task 7 documentation records FNMR swinging 0.021 % → 18.70 % → 93.98 % for the
same operating point purely by pair rule (`REP/ES-EXP07-AUDIT-FORENSIC-01.md` B‑11;
`REP/JETSON-RUN-SHEET-PREP-01-REV01.md:129`). `MEASURED`: the artefacts behind 0.021 % and 93.98 % are **not
in this package** — that swing is `UNVERIFIED`. Its *direction* is, however, confirmed by this pass on a
single variable: changing only the camera rule moves FNMR by `CALCULATED` **12.58434208141152 pp**
(60.73838345183187 % ↔ 48.154041370420353 %) between the agnostic and SIE pools computed over **the same
arm's score matrix**. Hence the rule this report applies throughout: **an FNMR quoted without its pair
population and its camera rule is invalid.** Every FNMR in §6.3 satisfies it; no recorded one does.

**6.9.6 CA‑14 / CA‑17 — which projection asset each recorded run used, and why the recorded generator
cannot be re‑run.** `MEASURED` (re‑hashed all 16 `.npz` in this pass; SHA‑256 first 16 hex).

| Contents (digest prefix) | bytes | names carrying it, per tree | ‖μ‖₂ | R₅₁₂ (`total_variance_ratio`) | arrays inside the file |
|---|---:|---|---:|---:|---|
| `9626bea5526df5b0` | 1,466,587 | `…_real_v2.npz`, `…_real_v3_reproduce.npz` | **10.372914292029659** | **0.9864118099212646** | mean, projection, singular_values, total_variance_ratio |
| `dc2b9072850c299d` | 1,466,710 | `svd_projection_768_to_512.npz`, `…_real.npz`, `projections/svd_projection_768_to_512.npz` | **19.389999510821742** | 0.9857138991355896 | mean, projection, singular_values, total_variance_ratio |
| `c9bca7595cb67d63` | 1,464,564 | `…_norm05_centered.npz` (under `transreid/` **and** `projections/`) | 10.37291325156675 | null | mean, normalization, projection, target_dim |
| `628b1dc6bcec1c22` | 1,461,626 | `…_norm05.npz` (non‑centred; `mean` is all zeros → ‖μ‖₂ = 0.0, `mean_abs_max` 0.0) | 0.0 | null | mean, normalization, projection, target_dim |

‖μ‖₂, R₅₁₂ and the array lists are `MEASURED` cells of
`RES/provenance/task7_provenance_measurements.json → projections.<filename>`; the digest column and the
"name → contents" join are `MEASURED` from re‑hashing all 16 files in this pass. All four carry a
`projection` of shape [768, 512]; all four pass the same orthonormality check
(`columns_orthonormal_max_abs_deviation` 8.75e‑09 … 1.25e‑08), so **an orthonormality check cannot
distinguish them — only a digest or an ‖μ‖₂/R₅₁₂ fingerprint can.**

`MEASURED`: the package holds **16 files with 4 distinct contents** — `CALCULATED` 16 = 8 × 2, because this
handoff ships every asset twice (once under `TASK07/assets/edgesync_bringup/models/`, once under
`TASK07/source_snapshot/Implementation Phase/edgesync_ws/src/edgesync_bringup/models/`), and each of the four digests appears an even number of times
(4 / 6 / 4 / 2). The prior audit's count of "8 `.npz` with 4 distinct contents" is the same fact measured
in one tree, and the two counts agree.

Three findings follow, and all three are live risks rather than history:

1. **The suffix "real" does not distinguish a file.** `dc2b9072…` is the content of
   `svd_projection_768_to_512_real.npz` while `9626bea5…` is the content of
   `svd_projection_768_to_512_real_v2.npz` — two different matrices whose names differ by a version token
   alone, with ‖μ‖₂ differing by `CALCULATED` 19.389999510821742 − 10.372914292029659 = **9.017085218792083**.
   The provenance file's pairwise block quantifies the matrix gap itself
   (`RES/provenance/task7_provenance_measurements.json → projections._relationships`, `MEASURED`):
   `_real.npz` vs `_real_v2.npz` → `byte_identical: false`, `max_abs_element_difference`
   **0.25184596329927444**; `svd_projection_768_to_512.npz` vs `_real_v2.npz` → the same **0.25184596329927444**
   (consistent with those two being one content); `_norm05.npz` vs `_real_v2.npz` → **0.42987602949142456**;
   `_norm05_centered.npz` vs `_real_v2.npz` → **3.423541784286499e‑05** (near‑identical matrices, different
   files — so even *contents* similarity does not imply identity); `_real_v3_reproduce.npz` vs
   `_real_v2.npz` → `byte_identical: true`, `max_abs_element_difference` **0.0**, which is why the table
   above lists them as one content. For a 768 × 512 projection whose typical column norm is O(1), a
   per‑element gap of 0.25 is not rounding: an embedding projected through the wrong one of these files is
   measurably different, and §6.1's arms would silently change meaning.
2. **The recorded variance field is a per‑run asset fingerprint.** `gate3_real_results.json →
   svd_variance_retained = 0.9857138991355896` = the `dc2b9072…` family; `_v2_corrected` and `_v3_reproduce`
   record `0.9864118099212646` = the deployed `9626bea5…` family, which is also the value this pass's
   embedding manifest pinned. **So the v1 → v2 change moved at least three things — projection asset,
   camera/SIE, and normalisation — while `…_reconciled.json:19` names only one as the cause** (see CA‑03 in
   §6.9.3). This is the arithmetic that turns "which preprocessing" into "which matrix", and it is the
   strongest single reason the recorded 71.17 → 93.38 gap must not be read as a preprocessing effect.
3. **Selection is by name, so a rename silently changes the embedding space.**
   `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:47` sets `reid_projection_path:
   "models/transreid/svd_projection_768_to_512_real_v2.npz"` — a *name*, not a digest. The consumer in this
   pass hash‑pinned `9626bea5…` in `RES/embeddings/task7_embeddings_manifest.json → assets` and its
   integrity gate re‑verified it, which is why §6.1's arms are attributable; the runtime path has no such
   pin. **This is defect T7‑D04 (§8), quantified rather than asserted: the namespace holds four contents,
   not two, and the two nearest are 0.25 apart per element. Acceptance criterion: the parameter takes a
   digest, or the loader verifies ‖μ‖₂/R₅₁₂
   against an expected fingerprint and refuses on mismatch.**

*CA‑17 — the only recorded producer on disk cannot be run from this tree.* `MEASURED`:
`SNAP/…/edgesync_identity/scripts/run_real_pipeline.py:33` hard‑codes
`BRINGUP = r"c:\Users\PC\OneDrive\Pictures\EdgeSync-main (3)\EdgeSync-main\Implementation Phase\edgesync_ws\src\edgesync_bringup"`;
that directory is **absent on this host**, while its `:40` output path names
`gate3_real_results_v2_corrected.json`, which *is* present. So the artefact that the current
population‑provenance note cites as its own source (`evaluate_offline.py:80` →
`run_real_pipeline.py:488-498` → `gate3_real_results_v2_corrected.json`) **cannot be regenerated from this
tree without editing that constant** — which this pass is not authorised to do (§2: no code modification),
and which is not a TransReID blocker. The reproduction this report relies on is instead the CLI‑path‑driven
pair `generate_task7_embeddings.py` + `analyze_task7_statistics.py` (§6.0, exit 0, evidence under
`REP/evidence/`).

*CA‑15 — training‑exposure evidence corroborates the split claim.* `MEASURED`: both checkpoints carry
classifier heads over **751** classes (TransReID 5 × (751, 768) = 2,883,840 elements; OSNet (751, 512) +
bias = 385,263), matching the measured `bounding_box_train` identity count of 751 and *not* the test count
of 750 (plus junk id `−1`); `…_reconciled.json` records `n_training_identities: 751`; TransReID's
`sie_embed` is (6, 1, 768) = six cameras. This **corroborates** the folder‑arithmetic split rule — it is
not proof of absence of leakage, because leakage would not have to show in head width.

**6.9.7 CA‑02 — the "different population" explanation is false, in three places, including this task's own
tooling.** `MEASURED`. The claim (proposal REV01 §2/§5, repeated in `analyze_task7_statistics.py:1212-1214`)
is that the recorded TransReID row is *"a different population"* from the newly measured arms, and that this
is why no comparison is possible. The four Gate‑3 files all record **`n_query` 3,368 and `n_gallery` 15,913**
— the identical counts to §6.1's arms (`MEASURED`: grep of every `n_query`/`n_gallery` key in
`gate3_real_results.json`, `_v2_corrected`, `_v3_reproduce`, `_reconciled`, the last also carrying
`evaluation_counts` 3,368 / 15,913 / 751). The recorded and the measured rows are therefore **the same
population with a different representation**, and the reason no paired comparison is available is something
else entirely (§6.10: no per‑query outcomes for the baseline, and no runnable baseline extractor on this
host).

The mis‑explanation traces to an `id_band` rule that is real but not implicated. `MEASURED`:
`evaluate_offline.py:113` defaults `id_min = 752, id_max = 1501` and applies them **only** when
`population = 'id_band_752_1501'` (`:155`; the band is registered at `:84`); after defect B‑12/TR‑07 was
fixed, `DEFAULT_POPULATION` is `directory` (defined `:99`, used `:112` and `:473`) and the chosen name is
recorded per output (`:175`, written `:572`) — so the band is now an **opt‑in, named, self‑labelling**
choice, and the document‑string claim at `:7` is stale. `CALCULATED`, from
`RES/provenance/task7_provenance_measurements.json → dataset.identity_set_algebra.id_band_752_1501_files`,
the band would
have produced **gallery 6,226 / query 1,774 over 395 identities**, versus the recorded 15,913 / 3,368 over
750 — a 60 % / 47 % shortfall that no recorded artefact shows. **Proof by counts: the band did not produce
any recorded accuracy row.** Consequently REV02 §5 item 2 (following REV01 P5) calling it *"a live defect in
the default path"* **overstates its current reach**: it produced none of the recorded numbers *and* it no
longer captures unwary runs.

Where the false claim still stands on disk, and what this pass did about it:

| Location | Text | Status in this report |
|---|---|---|
| `REP/ES-EXP07-TASK7-PROPOSAL-01-REV01.md:35` (P3 row) and `:125` | the 512‑D SVD arm is "a **different population**" | **FALSE** — corrected in `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md` §2/§5/§9/Appendix A, which is the version this report cites |
| `SNAP/…/scripts/analyze_task7_statistics.py:1212-1214` | "the recorded run used the invalid id_band population … not the same population as this one" | **FALSE reason, correct conclusion.** Documented, **not patched**: patching a string would require a §1‑style backup plus a re‑run of its 68 tests, and the output label (`NOT APPLICABLE`) does not change either way. Recorded in `REP/ES-EXP07-TASK7-REPORT-01-REV02.md` §6 threat 10 |
| `REP/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` §5 row 26 | this finding, as measured | `MEASURED`, superseding both rows above |

**6.9.8 CA‑13 — the recorded production recommendation contradicts the recorded device status, in one
file.** `MEASURED` (both strings read from the same JSON in this pass).
`gate3_real_results_reconciled.json` contains, simultaneously:

- `"jetson_orin_nano_status": "NOT MEASURED — Laptop timing rejected for edge deployment decisions"`, and
- `"final_architectural_determination"` with winner `"OSNet x1.0"` and `"recommendation": "Deploy OSNet
  x1.0 under TensorRT FP16 on Jetson Orin Nano. Reject TransReID due to 40x memory overhead and inferior
  edge efficiency."`

The charter's hypothesis for Task 7 was whether TransReID *"might break the critical TTO handoff budget"* —
a **timing** hypothesis. This file states on its own that the timing quantity was never measured on the
target device, and then decides the hypothesis on that quantity. Separately, its `"40x memory overhead"`
rests on the mislabelled `parameter_ratio` (CA‑06, §6.8.1: the true ratios are **40.5725** with heads,
**46.4481** parameters, **39.8789** file bytes, none of which is a *memory* measurement — the only recorded
peak, `peak_runtime_gpu_memory_mb`, ranks the **smaller** model higher, CA‑12). The same file also carries
`"root_cause": "Genuine gold-standard baseline verified against official Torchreid model zoo"` for the OSNet
row, a claim about an environment that is not this host (§6.10: on this host that import fails).
`UNVERIFIED` here.

**Verdict for the register:** the *refusals* in proposal REV02 §8 and report REV02 §7 are consistent with
the evidence; the *record* is not. **This report does not adopt the "Reject TransReID / Deploy OSNet"
recommendation**, and equally does not adopt its opposite: §13 records the deployment decision as
`BLOCKED` on device evidence that does not exist in this package.

**6.9.9 What a downstream reader may quote from §6.9, and what they may not.** Derived from CA‑01…CA‑14
above; the same list appears in `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md` §10.

**May quote**, with the named artefact beside it:
- Rank‑1 93.37885985748218 % / Rank‑5 97.95130641330167 % / Rank‑10 98.87173396674585 % / mAP
  85.12520406508766 % for `transreid_512_sie` — **and** the CA‑01 fact that the same arm reproduces
  `_v2_corrected` / `_v3_reproduce` and the packaged team reproduction to the last printed digit;
- Rank‑1 88.74703087885986 % / mAP 77.32400426226894 % for `transreid_512_agnostic`, same support;
- every §6.1–§6.5 number, since each has a declared population (directory, 3,368 × 15,913, pid rules
  quoted in §5.3), a per‑query unit, and a hypothesis test where a claim of difference is made;
- the element‑count → size reconstructions (CA‑05, §6.8.1) — they are exact;
- 71.16983372921615 / 51.46723674145579 **only** as "`gate3_real_results.json` records …", i.e. as a
  quotation of a superseded run, with CA‑03 + CA‑14 attached (it is no‑camera, no‑SIE, and used the
  `dc2b9072…` projection).

**Must not quote**: the `_reconciled.json` 2‑dp ablation row (88.64 / 89.91 / 86.42 …) as a measurement
(CA‑11: three mutually inconsistent values for one named arm); any `delta_rank1_vs_baseline` string without
correcting `%` → pp; any latency ratio without naming its artefact row (CA‑07 — the sign itself depends on
the choice); the `phase8`/`phase9` medians (CA‑08 — absent from disk); `verification_threshold_audit.json`
FNMR/FMR/EER without stating that its pool is undeclared (CA‑10); OSNet 94.32897862232778 % as "our
baseline run" (it is `INHERITED` — §6.10); `40.5×` as a parameter ratio (CA‑06); and the
"Deploy OSNet / Reject TransReID" recommendation (CA‑13).

### 6.10 The OSNet comparator — what the package supports, and the four things it does not

This section had to be rewritten during this pass. §6.8.4 and the `not_measured` block of
`RES/statistics/task7_statistics.json` label OSNet retrieval **`BLOCKED` on this host**, and that label is
correct for the *production extractor path*. It is **not** correct for the whole question, because the
package also contains a **completed OSNet reproduction on this population**, with its own log and exit
status. Both facts are reported; neither is allowed to cancel the other.

**6.10.1 The baseline number was reproduced, exit 0, on this host.** `MEASURED` (re‑read the JSON *and* the
log in this pass): `REP/evidence/repro_osnet_full_rgb.json`, corroborated line‑for‑line by
`REP/evidence/full_rgb.log` which ends `DONE RGB 0` / `REAL_EXIT=0`:

| Field (as written) | Value |
|---|---|
| `route` | *"torchreid osnet.py via sys.prefix path (run_real_pipeline.py:718-731)"* |
| `preprocess` | *"BGR->RGB, resize (128,256), ImageNet mean/std (run_real_pipeline.py:274-285)"* |
| population (from the log) | `variant RGB n_gallery 15913 n_query 3368` |
| `osnet_reproduced_rgb_route` | Rank‑1 **94.32897862232778**, Rank‑5 97.80285035629454, Rank‑10 98.60451306413302, mAP **83.5692348350484** |
| `n` / `dim` / `zero_rows` / `seconds` | 19,281 / 512 / 0 / 537.5 |
| `recorded_reference` | the **same four values** |

`CALCULATED`: Δ Rank‑1 = 0.0 and Δ mAP = 0.0 between the reproduction and `recorded_reference`, so
94.32897862232778 / 83.5692348350484 is **the one baseline figure in Task 7 with a completed, exit‑0,
same‑population reproduction on disk**. `MEASURED` limits on it, and they are not cosmetic: the file records
**no timestamp, no host, no environment block, no script argv, and no per‑query outcomes**; the log records
progress lines only; the JSON names its flawed predecessor's artefact `bgr-order-run.json`, a file that is
**not in this package** (what is shipped is `repro_osnet_BUGGY_BGR.json`, §6.10.2). So this row is
`MEASURED` *as a re‑read of a packaged artefact* (§6.0 sense), not as something re‑executed today, and it
cannot be turned into a paired test because no per‑query array accompanies it.

**6.10.2 The audit's own first attempt was wrong by 84 Rank‑1 points, and that is disclosed on disk.**
`MEASURED`: `REP/evidence/repro_osnet_BUGGY_BGR.json` (+ `.log`, `REAL_EXIT=0`) records Rank‑1
**10.243467933491686**, Rank‑5 25.029691211401424, Rank‑10 36.22327790973871, mAP 8.607295284945122 on the
same n = 19,281 — `CALCULATED` **−84.0855106888361 pp** Rank‑1 and **−74.96193955010328 pp** mAP versus the
RGB route (a 9.21× ratio). The cause was the audit script feeding **BGR** pixels into **RGB**‑ordered
ImageNet statistics, while the repo harness converts at `run_real_pipeline.py:280`.
`REP/ES-EXP07-AUDIT-FORENSIC-01.md:42` states this in the first person and adds a sentence about the
*product*, not the audit: "the production node performs **no `sensor_msgs/Image` encoding check**".
**That sentence is `INHERITED` (audit date 2026‑10‑19) and, re‑inspected against the delivered source in
this pass, it is superseded — so it is recorded here as a conflict, not silently dropped.**
`IMPLEMENTED` + `TESTED`: the delivered `SNAP/edgesync_identity/edgesync_identity/core/reid_input_adapter.py`
resolves the channel order from `msg.encoding` (`:42`), accepts `rgb8` (`:69`), converts `bgr8`/`8UC1` by
reversing the last axis (`:72‑73`), and raises `ValueError` for anything else (`:75‑78`, supported sets at
`:20‑21`); the node calls it on **both** crop paths — `identity_node.py:556` inside the service handler with
its rejection branch at `:557‑559`, and `identity_node.py:354` in `_on_target_crop` with its rejection branch
at `:355‑358` — and the offline loader converts explicitly at `evaluate_offline.py:241, :246`
(`cv2.imread` then `COLOR_BGR2RGB`). `TST/test_transreid_input_adapter.py` asserts exactly this: `:57`
accepts upper‑case and `8UC1`, and `:77‑81` refuses `mono8`, `bgra8`, `16UC1` and `''` with the
`unsupported image encoding` diagnostic; its 31 cases are inside the 360 passed (§7.3). The change is traced
to **TR‑04** in `REP/ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md:32` ("the ROS image boundary ignored
`msg.encoding`, so BGR arrived at a model trained on RGB").

What the diagnostic still establishes is the **cost of the mistake**, not its presence in the shipped code:
`REP/evidence/channel_order_diagnostic.json`, gallery 600 / query 281 (n = 881): RGB Rank‑1
84.82758620689656 / mAP 88.6288731383559 versus BGR Rank‑1 49.6551724137931 / mAP 59.1790431683047, i.e.
`CALCULATED` **35.17241379310345 pp** Rank‑1 lost by a one‑line ordering mistake, with
`cosine_rgb_vs_bgr_same_image` mean **0.6062065269565474**, min 0.32869184017181396, max 0.9902685880661011,
p10 0.4679720401763916 — for the *same image and the same weights*. At τ = 0.70 (§6.3) a genuine pair whose
two channel orders score 0.606 apart is not a subtle case, which is why the gate was worth adding.
**T7‑D20 (§8) is therefore registered as what the current evidence supports, not as the audit's sentence:
the encoding gate exists and is unit‑proven, but nothing in the package exercises it on a message received
from a live graph, and no artefact records which `encoding` a real crop producer will publish.**

`osnet_backbone_equivalence.json` closes one adjacent question and opens another: for 8 images, the repo's
OSNet definition and the installed `torchreid` build share **0 missing / 0 unexpected** keys and give
`cosine_repo_vs_installed` mean 1.0000000298023224, min 1.0, `identical: true`, output dim [512] — so the
reproduction's backbone is the checkpoint's backbone. Its `train_mode_probe` records
`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])`,
confirming that the equivalence holds **in eval mode only**, which is the mode the reproduction used; nothing
in the package measures the backbone under train mode, and nothing needs to.

**6.10.3 The production OSNet path still cannot execute here — three distinct layers.** `MEASURED` /
`IMPLEMENTED` by inspection only; charter §3B forbids repair, and this pass performed no repair.
`RES/embeddings/task7_embeddings_manifest.json → osnet_status` = *"ARM NOT REQUESTED"* and
`osnet_factory_attempt` = **FAILED** (`MEASURED`, re‑read; the same two fields appear in §5.1's plan‑versus‑executed row).

| Layer | What fails | Evidence |
|---|---|---|
| `osnet_trt` | `SNAP/…/models/osnet_trt_extractor.py` has two halves, and re‑reading the whole method in this pass sharpens the finding. `_execute_trt()` (`:72-75`) is a stub: its body is a comment plus `return np.zeros(512, dtype=np.float32)`, so even a **loaded** engine would produce zeros; but it is called only when `self._trt_context is not None` (`:83`), and on every host in this evidence set it is `None`, so the `else` branch at `:86-89` substitutes a **tiled channel‑mean pseudo‑embedding** (`np.tile(mean_vals, 512 // 3 + 1)[:512]`) that is then L2‑normalised, so nothing downstream distinguishes it from a real feature by shape. **No `.engine`/`.plan`/`.trt` file exists anywhere** (§6.8.4), so only the pseudo‑embedding branch is reachable here | `MEASURED` (source re‑read in this pass) + `NOT MEASURED` (behaviour, no engine) — **defect T7‑D08** |
| `osnet_torch` | `osnet_torch_extractor.py:64` `from torchreid.models import build_model` is recorded as raising `RuntimeError: operator torchvision::nms does not exist`; the fallback at `:74` catches only `ImportError`, so the `RuntimeError` escapes. **Re‑inspected in this pass:** that `try` block lives in `_load_model()` (`:56`), called from `__init__` at `:54`, so the escape is at **construction** — out of `extractor_factory._build_osnet_torch()` (`:109`) — **not** out of `extract_embedding`, which has its own catch‑all at `:187-189` that does return the `zeros(512)` sentinel. `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md` CA‑16 states the stronger version ("propagates out of `extract_embedding`"); this report corrects it, and the correction is registered as part of T7‑D06 | `MEASURED` (recorded failure string + source re‑read) + `IMPLEMENTED` (the `except` clause as written) — **defect T7‑D06** |
| placeholder backbone | `_build_osnet_standalone()` (`:93-124`) is a stand‑in: `MEASURED` **8 entries / 42,945 elements** (42,816 trainable + 129 buffers) versus the checkpoint's **2,554,771** trainable elements, **0 of 567** key names shared, `strict=False` → `missing_keys = 7 / unexpected_keys = 567`; it consumes none of `osnet_x1_0_market1501.pth`, and its `preprocess()` (`:140-143`) uses ImageNet stats | `MEASURED` from `RES/provenance/task7_provenance_measurements.json → checkpoints.osnet_placeholder_backbone` — **defect T7‑D09** |

**6.10.4 What may therefore be said about TransReID versus OSNet.** Two statements are supported; three are
not.

*Supported*: (a) the recorded baseline row is reproducible on this host through the historical harness route
(§6.10.1); (b) for **the arm the shipped ROS 2 runtime can actually reach** — no camera label, because
`identity_node.py:266-272` calls `extract_embedding(crop)` with none, `IdentityVerify.srv` carries no camera
field, and `base_identity_extractor.py`'s signature has no `cam_label` parameter — reproduced TransReID is
**below** reproduced OSNet on all four metrics. `MEASURED` from
`REP/evidence/repro_transreid_both_arms.json → deltas` (`NO_CAM_deployed_path` minus `recorded_osnet`):
Rank‑1 **−5.58194774346792** pp, mAP **−6.24523057263802** pp. `CALCULATED` against §6.1's own mAP cell
(77.32400426226894 − 83.5692348350484 = **−6.245230572779462** pp) — the 1.4e‑10 difference is because that
file differences against its *own* mAP float, as CA‑01/CA‑04 predict for any mAP comparison.

The file states the hypothesis consequence in its own words, quoted verbatim: *"H7a states accuracy
increases. For the only arm the shipped ROS 2 runtime can reach (no SIE), reproduced TransReID is BELOW the
reproduced OSNet baseline on every metric. H7a is contradicted for the deployed configuration and holds on
mAP only (+1.56 pp) with SIE, while Rank‑1 is still ‑0.95 pp."* The `WITH_CAM_SIE` deltas in the same block
are Rank‑1 **−0.9501187648456** pp and mAP **+1.55596922980124** pp — `CALCULATED` check: 93.37885985748218 −
94.32897862232778 = **−0.9501187648455982** and 85.12520406484964 − 83.5692348350484 = **+1.555969229801235**
✓ (the mAP operand again being `gate3_real_results_v2_corrected.json`'s float, not §6.1's).

*Not supported*: (1) **a paired statistical test** against OSNet — it needs per‑query predictions from both
arms, and no OSNet per‑query array exists anywhere in the package (§6.4). Charter §8 forbids inferring a
handoff conclusion from an unpaired comparison; the correct label is `NOT APPLICABLE`, for the *baseline's*
evidence, not for the population (CA‑02, §6.9.7). (2) **any statement about OSNet versus TransReID at the
verification layer** — `verification_threshold_audit.json`'s OSNet row (FNMR 18.703290940081292 %) declares
no pool (CA‑10), and §6.3's TransReID rates do declare theirs, so the two are not differencable. (3) **any
deployment ranking** — the size facts are §6.8.1 and the device facts are §6.8.4 / CA‑13; `NOT MEASURED` for
both arms on Jetson, and OSNet has never been run through TensorRT here either.

§13's verdict inherits (b) as the single most consequential negative result in Task 7 and (1)–(3) as the
reason no accuracy claim in this report is a deployment recommendation.

### 6.11 Negative and unfavourable results — complete register

The brief requires that unfavourable results not be omitted. Twelve items, ordered by how much they change
what may be concluded. None is reconciled by the favourable numbers in §6.1–§6.5; all twelve also appear in
§13.

| # | Negative finding | The number | Class / evidence |
|---|---|---|---|
| N‑1 | **For the configuration the shipped runtime can actually reach, TransReID is worse than the frozen baseline on every metric.** | Rank‑1 **−5.58194774346792** pp, mAP **−6.24523057263802** pp (`MEASURED` deltas in `REP/evidence/repro_transreid_both_arms.json`) | H7a **contradicted** for the deployed arm; §6.10.4 |
| N‑2 | Even with camera conditioning, Rank‑1 is **still below** the baseline; only mAP exceeds it. | Rank‑1 **−0.9501187648456** pp vs mAP **+1.55596922980124** pp | §6.10.4; unpaired, so not a test |
| N‑3 | **The shipped threshold is far from the operating point the data supports.** | τ = 0.70 sits at `CALCULATED` **2.137×** the SIE arm's EER threshold (0.7 ÷ 0.32755059003829956) and **2.289×** the agnostic arm's; TAR at τ is only **0.5184595862957965** | `MEASURED` §6.3; no calibration is `IMPLEMENTED` (§6.8.4) |
| N‑4 | **At τ = 0.70 roughly one genuine pair in two is rejected** — and one query in five falsely accepts an impostor. | FNMR 48.154041370420353 % (SIE) / 60.73838345183187 % (agnostic) pair‑level; query‑level miss **11.312351543942989 %** / **18.913301662707838 %**; query false accept **20.160332541567696 %** / **19.922802850356294 %** | `MEASURED` §6.3, two denominators quoted separately |
| N‑5 | **No latency, throughput, memory, power or accuracy figure in this task is a device measurement.** Both arms, both the PyTorch and the ONNX route. | Jetson/TensorRT: `BLOCKED`, harness **executed zero calls**; `probe_build_report.json` environment block all‑negative | `NOT MEASURED` §6.8.4 — and N‑5 is why no timing claim is made anywhere in §6 |
| N‑6 | **The delivered output — a published `ReidSignature` — has never been produced by a running node.** | 0 executions at Tier R; no `rclpy`/DDS on this host; and **no crop publisher exists**, so the input is starved too | `BLOCKED` T7‑D02, §10 levels 5–8 |
| N‑7 | **The parity gate exercised no genuine identity.** | All 8 cross‑checked images are `pid 0` (the Market‑1501 distractor class), all `camid 1`, `cam_label: 0` | `MEASURED` §6.7; 8/8 PASS is still a real gate, but not on identities |
| N‑8 | **H3 is a power‑limited non‑finding, not a demonstrated absence of cost.** | 154 discordant queries; point estimate SIE **worse** by 0.23752969121140222 pp; p = 0.572845968999707 | `MEASURED` §6.4 |
| N‑9 | **The recorded history contradicts itself in ways this pass cannot settle.** | Rank‑1: 71.17 / 88.75 / 88.75 / 88.85 for one named arm; latency sign flip (0.711550× vs 2.084496×); `_reconciled.json` recommends a device deployment while declaring the device `NOT MEASURED` | `UNRESOLVED` CA‑07/CA‑11/CA‑13, §6.9.4–§6.9.8 |
| N‑10 | **My own first baseline reproduction was wrong by 84 Rank‑1 points** (BGR into RGB statistics). The audit's inference that the *product* therefore has no encoding check is **superseded** — the delivered adapter gates on `msg.encoding` (§6.10.2); what remains unfixed is that the gate is proven only on duck‑typed stand‑ins, never on a message from a live graph. | Rank‑1 10.243467933491686 vs 94.32897862232778; on a stratified mini‑population 84.82758620689656 → 49.6551724137931 = **35.17241379310345** pp lost, same‑image cosine mean **0.6062065269565474** | `MEASURED` §6.10.2 — disclosed, not suppressed |
| N‑11 | **A defect in this pass's own consumer tooling shipped with a false explanation** — `analyze_task7_statistics.py:1212-1214` justifies its (correct) `NOT APPLICABLE` with a population claim that is false. | id_band would give 6,226 / 1,774 vs the recorded 15,913 / 3,368 | `MEASURED` CA‑02 §6.9.7; documented, **not patched** (§2 forbids code edits) → **T7‑D27** |
| N‑12 | **Nothing in §6.1–§6.5 can be recomputed from this package alone.** | The three embedding `.npz` (55,193,927 + 37,094,828 + 37,095,723 B) live outside it, under `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory\` | `MEASURED` from `RES/embeddings/task7_embeddings_manifest.json → files`; §12.2 |

Two consequences of reading them together. First, **N‑1 + N‑3 + N‑6 bound the claimable result**: the arms
in §6.1 are correct and reproducible for what they are — an offline, single‑frame, appearance‑only gate on a
fixed dataset — and they do not license a handoff claim, a threshold choice, or a deployment ranking.
Second, **N‑1 is not the same statement as "TransReID is worse than OSNet"**: it is the statement "the arm
reachable through `identity_node.py:266-272` (no `cam_label`, no camera field in `IdentityVerify.srv`) is
worse than the reproduced baseline", which is a *wiring* finding as much as a *model* finding, and the
remediation for it (owner gate G.1/G.3, §11 action A‑2) is a decision rather than an experiment.

---

## 7. Tests and validation

### 7.1 Three evidence families, kept separate

No test in this section is described as "passing" without its run, its date and the tree it ran against.
Three families exist, and conflating them is the main way this evidence can be misread:

| Family | Date | Tree used | How it is labelled here | Where the record is |
|---|---|---|---|---|
| **A — project runs during execution** | 2026‑10‑15 → 2026‑10‑19 | the working project tree (`C:\Semester 7\EdgeSync-main (3)Last\…`) | `INHERITED` | `REP/test-execution-log.md`, `REP/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md`, `RES/execution_evidence/pytest_baseline_prefix.txt`, `REP/evidence/pytest_full.log`, `REP/evidence/pytest_ignored.log` |
| **B — freshly executed for the handoff, against the delivered copies** | captured **2026‑10‑21 07:54:08** local | `TASK07/source_snapshot/…` **inside this package** | `TESTED` (this package's own files) | `VALIDATION/TEST_EXECUTION_EVIDENCE.md` + `VALIDATION/fresh_evidence/*.log`, `*.txt` |
| **C — performed by this report pass** | 2026‑10‑22 | the delivered copies | `MEASURED` / collection‑only | recorded inline in §7.3, with the exact commands |

Family B is the important one: it means a reader who unpacks this package can re‑run the same commands on
the same files. Family C is deliberately minimal — this pass ran `pytest --collect-only` (read‑only, with
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`) to obtain per‑file counts that family B's logs report
only at suite level. **This pass did not re‑execute any Task 7 test, experiment, export or benchmark**; §2 of
the update brief forbids modifying any other file, and re‑execution of the analyser or the generator is not
possible from the package anyway (§6.11 N‑12, §12.2). After the collection runs, the only file under
`TASK07/` with a modified timestamp was this report itself (`MEASURED`: `find TASK07 -newermt "-20 minutes"`
→ 1 hit, `EXECUTIVE_SUMMARY_TASK07.md`).

Environment for family B, verbatim from `VALIDATION/TEST_EXECUTION_EVIDENCE.md:10-17`: Windows 10
(10.0.26300) AMD64, Git Bash, Python 3.10.11, pytest 9.0.3, torch 2.5.1+cu121, numpy 1.26.4; **"Not
present:** `rclpy`, generated `edgesync_msgs`, `launch` / `launch_ros`, `ros2` CLI, a DDS implementation,
TensorRT, any Jetson target"; guards `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, per‑package `cwd`,
`--continue-on-collection-errors`, nothing installed system‑wide.

### 7.2 Suite‑level results, with the reconciliation arithmetic shown

`TESTED` (family B). Command, per package:
`python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors`, run with `cwd` inside the
copied package so `PYTHONPATH` pointed at the copied `src/`.

| Package | Passed | Skipped | Failed | Collection errors | Exit | Log |
|---|---:|---:|---:|---:|---:|---|
| `edgesync_identity` | **360** | **15** | 0 | 1 (`test_identity_service.py`) | **1** | `VALIDATION/fresh_evidence/identity_copy_run.txt` (+ `_skipreasons.txt`) |
| `edgesync_bringup` | **55** | 0 | **2** | 1 (`test_robot_bringup_launch.py`) | **1** | `VALIDATION/fresh_evidence/bringup_copy_run.txt` |
| `edgesync_coordination` (Task 13, for context only) | 360 | 0 | 0 | 3 (all `rclpy`) | 1 | `coordination_copy_run.txt` |

**No suite in this table is "fully passing."** Both Task 7 suites exit non‑zero. `TESTED` counts must be
read with `CALCULATED` reconciliation, which closes exactly — family B's own §3 asserts this and the
assertion holds:

- `edgesync_identity`: **375** collected = 360 passed + 15 skipped (`CALCULATED` from `--collect-only`, §7.3;
  the 16th file cannot be imported at all). Against the inherited record "372 passed, 3 skipped"
  (`VALIDATION/inherited_evidence/exp13_20261021/identity_suite_excluded.txt`): `CALCULATED` 360 + 12 = 372
  and 3 + 12 = 15 — the 12‑test delta is tests whose **asset** is absent from the copy, not tests that
  changed behaviour. No new failure.
- `edgesync_bringup`: **57** collected = 55 passed + 2 failed (`CALCULATED`, §7.3). Against the inherited
  `RES/…/FINAL_bringup.log` (custom runner, 57 passed on 2026‑10‑19): `CALCULATED` 55 + 2 = 57, and both
  failures are the omitted checkpoint (§7.6), not a regression.
- `REP/evidence/pytest_full.log` / `pytest_ignored.log` (family A, 2026‑10‑19 project tree) show the same two
  faces of the `rclpy` problem: with collection continuing, the run is interrupted by
  `ModuleNotFoundError: No module named 'rclpy'` at `tests/test_identity_service.py:11` → `REAL_EXIT=2`;
  deselecting that file gives **56 passed, 3 warnings, `REAL_EXIT=0`**. `MEASURED`: 56 is the *subset* that
  excludes the uncollectable file and the heavy‑fixture skips of that day; it is not comparable with 360.

### 7.3 Per‑file counts for every Task 7 test file, and what each proves

`MEASURED` (family B logs where a per‑file run exists; this pass's `--collect-only` for the collected count
where it does not). Identity package, `cwd` = the copied `edgesync_identity`:

| Test file | Collected | Passed | Skipped | Per‑file log (family B) | What it proves | What it does **not** prove |
|---|---:|---:|---:|---|---|---|
| `test_analyze_task7_statistics.py` | 68 | 68 | 0 | yes, 5.58 s | The §6.1–§6.5 statistics engine: Wilson, percentile bootstrap, exact McNemar + Holm step‑down, pair pooling, τ gate that **refuses** a non‑0.7 τ, canonical‑vs‑per‑query mAP equality | Nothing about the embeddings it would consume — it is fed synthetic pools |
| `test_evaluate_offline_population.py` | 28 | 28 | 0 | yes, 0.40 s | B‑12/TR‑07 fix: `DEFAULT_POPULATION = directory`, opt‑in named `id_band_752_1501`, name echoed to the manifest (§6.9.7) | That the recorded Gate‑3 rows were produced by this code (they were not; §6.9.1) |
| `test_exp7_offline_exp_id.py` | 24 | 24 | 9 warn | yes, 12.47 s | `exp_id` is written to, and gates, the offline outputs; degenerate populations render as `null`, not as 0 or NaN (§6.0) | Any accuracy claim; the 9 warnings are the empty‑slice path being exercised on purpose |
| `test_experiment7_fixes.py` | 6 | 6 | 0 | no (suite only) | The named Experiment‑7 fixes at unit level | Runs with `UserWarning: Initializing zero-element tensors is a no-op` (§7.7) — it touches the OSNet placeholder, not the checkpoint |
| `test_experiment7_forensic_fixes.py` | 13 | **9** | **4** | no | The forensic‑audit fixes (metric code, population naming, sentinel handling) | 4 of 13 skip: 3 for the absent assets, 1 for the absent Market‑1501 directory (§7.5) |
| `test_export_transreid_onnx.py` | 33 | 33 | 0 | no | The ONNX exporter's contract: symbolic axes, projection kept outside the graph, checker invocation | That a graph was produced on this host by these tests — §6.7's artefacts are family A output |
| `test_extractor_factory_registry.py` | 23 | 22 | 1 | yes, 0.50 s | `BACKBONE_REGISTRY` ids, `DEFAULT_BACKBONE_ID = osnet_mock`, `EDGESYNC_EXPERIMENT_CONFIG` selection (§4) | 1 skip = `ES07_HEAVY` gate; registry ≠ real weights |
| `test_extractor_factory_transreid_trt.py` | 16 | 16 | 0 | no | TRT extractor is *selectable* with stubbed `tensorrt`/`pycuda` | **No inference**: stubs only; no engine exists (§6.8.4) |
| `test_gate1_scaffolding.py` | 32 | 24 | **8** | yes, 0.36 s | Gate‑1 wiring of the identity stack | 8 skip as *"TransReID assets not available"* — the synthetic fixtures were excluded by the size policy (§7.5) |
| `test_identity_fusion.py` | 5 | 5 | 0 | no | Fusion of signature fields | Not ROS 2 behaviour (Tier N stubs) |
| `test_identity_node_model_resolution.py` | 23 | 23 | 0 | no | `reid_model_type` vs `reid_model` alias resolution at node level — the T7‑D03/T7‑D11 evidence | Not that the resolved extractor then ran on real pixels |
| `test_identity_node_transreid_integration.py` | 20 | 20 | 18 warn | yes, 0.31 s | Node ↔ TransReID extractor wiring, camera‑id parameter path | `MEASURED`: node object under stub `rclpy`, **Tier N** — no DDS, no graph, no crop publisher (§6.11 N‑6) |
| `test_transreid_extractor_guardrails.py` | 18 | 17 | 1 | yes, 2.27 s | Input guardrails: shape refusal, sentinel return, dimension contract | The 1 skip is asset‑dependent; guardrails prove refusal, not accuracy |
| `test_transreid_input_adapter.py` | 31 | 31 | 0 | no | `reid_input_adapter` / crop → tensor contract: `encoding`-driven channel order, padded‑row stride, refusal of `mono8`/`bgra8`/`16UC1`/`''` (`:57`, `:77‑81`) | Only the pure‑function boundary. Messages are duck‑typed stand‑ins, so the gate is **not** shown to fire on a `sensor_msgs/Image` received from a graph — the residual half of T7‑D20, with the §6.10.2 measurement behind it |
| `test_transreid_trt_extractor.py` | 35 | 34 | 1 | no | The self‑contained TRT extractor's logic under stubs | 1 skip = `ES07_HEAVY`; engine inference never executed on any host |
| `test_identity_service.py` | **uncollectable** | — | — | no | — | `ModuleNotFoundError: No module named 'rclpy'` at `:11` — **0 % of this file executed**, on any host in this evidence set |
| **Total (identity)** | **375** | **360** | **15** | 7 of 16 files | | |

Bringup package, `cwd` = the copied `edgesync_bringup`:

| Test file | Collected | Passed | Failed | Notes |
|---|---:|---:|---:|---|
| `test_exp7_config_overlay.py` | 41 | 40 | **1** | Failure = `TestShippedFiles::test_reused_assets_exist_in_the_package_tree` |
| `test_reid_launch_defaults.py` | 16 | 15 | **1** | Failure = `TestAgainstTheShippedConfig::test_shipped_assets_exist_in_the_package_tree` |
| `test_robot_bringup_launch.py` | **uncollectable** | — | — | `ImportError: cannot import name 'LaunchDescription' from 'launch' (unknown location)` at `:21` |
| **Total** | **57** | **55** | **2** | exit 1 |

### 7.4 Tier accounting — what 360 + 55 green tests do and do not establish

`MEASURED` from the assertion sources and imports visible in the files above:

- **Tier U** (pure Python / numpy / torch on synthetic inputs): the large majority of the 360. Establishes
  that the *arithmetic and selection logic* in `analyze_task7_statistics.py`, `evaluate_offline.py`,
  `export_transreid_onnx.py` and the extractor guardrails behaves as documented on crafted inputs.
- **Tier N** (node and adapter classes under a **stubbed `rclpy`**, message classes generated from the
  on‑disk `.msg`/`.srv`): `test_identity_node_*`, `test_transreid_input_adapter.py`, `test_identity_fusion.py`.
  Establishes import‑level and call‑level wiring. It does **not** establish that a node ever joined a graph.
- **Tier R** (real ROS 2 / DDS): **0 tests executed, 0 assertions**. `rclpy` is absent, `colcon build` was
  never run, and `test_identity_service.py` cannot even be collected. §10 level 5 is therefore `BLOCKED`,
  not "untested by choice".
- **Tier H/J** (Jetson / TensorRT): **0 tests, 0 measurements** (§6.8.4). `test_*_trt_*` run against stubs.

The brief's own caution applies literally: *a passing unit test does not establish full system correctness,
ROS 2 integration, hardware performance, or deployment readiness.* Concretely for Task 7, the 415 green
assertions collectively prove (i) the statistics engine computes what §6.1–§6.5 reports, (ii) the extractor
factory selects and refuses correctly, (iii) the ONNX exporter emits the documented graph contract, (iv) the
population defect B‑12/TR‑07 is closed with a self‑labelling opt‑in. They collectively prove **nothing** about
a running system, because no running system exists in this evidence set: the delivered `ReidSignature` has
never been published (§6.11 N‑6), and every hardware number is `NOT MEASURED`.

### 7.5 All 15 skips, verbatim, with the site that skipped them

`TESTED` / `MEASURED` from `VALIDATION/fresh_evidence/identity_copy_run_skipreasons.txt:54-68`. Every skip is
an **asset** or **dataset** absence, not a behavioural failure — but the consequence is that 15 of 375
collected assertions never touched the real checkpoint:

| # | Site | Reason, verbatim |
|---:|---|---|
| 1 | `test_experiment7_forensic_fixes.py:84` | *"Market-1501 dataset directory not present on this path"* |
| 2 | `test_experiment7_forensic_fixes.py:125` | *"transformer_120.pth not present"* |
| 3 | `test_experiment7_forensic_fixes.py:195` | *"TransReID checkpoint or projection matrix not present"* |
| 4 | `test_experiment7_forensic_fixes.py:216` | *"TransReID checkpoint or projection matrix not present"* |
| 5 | `test_extractor_factory_registry.py:255` | *"loads the 414 MB checkpoint; set ES07_HEAVY=1"* |
| 6–13 | `test_gate1_scaffolding.py:368, 375, 382, 388, 393, 398, 404, 417` | *"TransReID assets not available. Run: python scripts/generate_test_fixtures.py OR set TRANSREID_CHECKPOINT_PATH / TRANSREID_PROJECTION_PATH"* (×8) |
| 14 | `test_transreid_extractor_guardrails.py:103` | *"asset absent: …\\TASK07\\source_snapshot\\…\\edgesync_bringup\\models\\transreid\\transformer_120.pth"* |
| 15 | `test_transreid_trt_extractor.py:149` | *"set ES07_HEAVY=1 to load transformer_120.pth for the parity check"* |

`CALCULATED`, by cause: **12** skips are a missing TransReID asset (sites 2, 3, 4, 6–13, 14); **2** are the
opt‑in `ES07_HEAVY=1` gate (sites 5 and 15), which also skips on the project tree; **1** is the missing
Market‑1501 directory (site 1) → `12 + 2 + 1 = 15` ✓. The packaging document groups the same 15 as
`8 (gate1) + 4 (forensic 3 + guardrails 1) + 2 (registry + trt) + 1 (dataset)` = 15 ✓ — two different
partitions of one set, both of which close.

**The load‑bearing consequence:** **no test in this package loads `transformer_120.pth` and asserts a numeric
output against it** — the two tests that would (sites 5 and 15) are opt‑in heavy gates, and the file itself is
absent from the package (§12.2). The only place the real checkpoint is exercised end‑to‑end is family A's
embedding generation (§6.0), whose output artefacts are also outside the package. **A reader who runs the
delivered suite verifies the plumbing, not the weights.**

### 7.6 Failures and collection errors, with cause

`TESTED` / `MEASURED`.

| Item | File / test | Cause | A code defect? |
|---|---|---|---|
| Failure 1 | `test_exp7_config_overlay.py::TestShippedFiles::test_reused_assets_exist_in_the_package_tree` | Asserts `os.path.isfile(<copied pkg>/models/transreid/transformer_120.pth)`; the 414,724,314 B checkpoint is referenced, not copied, by the size policy (`bringup_copy_run.txt:14-27`) | **No** — packaging policy. It *is* evidence that the shipped YAML's asset‑existence contract cannot be satisfied from this package |
| Failure 2 | `test_reid_launch_defaults.py::TestAgainstTheShippedConfig::test_shipped_assets_exist_in_the_package_tree` | The identical assertion against `REAL_YAML` defaults, at `:47` | **No** — same cause, same consequence |
| Error 1 | `edgesync_identity/tests/test_identity_service.py` | `ModuleNotFoundError: No module named 'rclpy'` at `:11` | **No** — absent ROS 2 stack; **but 0 % of this file's coverage has ever been realised on any host in this evidence set** |
| Error 2 | `edgesync_bringup/tests/test_robot_bringup_launch.py` | `ImportError: cannot import name 'LaunchDescription' from 'launch' (unknown location)` at `:21` | **No** — ROS 2 `launch` absent; the launch file it tests has never been imported, let alone executed (§10 level 6) |

One **documentation** defect was found inside the evidence while re‑reading it:
`VALIDATION/TEST_EXECUTION_EVIDENCE.md:87-92` says *"five test files … cannot be collected"*, then *"None is
a code defect; all four causes are the absent ROS 2 stack."* `MEASURED`: five files and **two** distinct
causes (`rclpy` ×4, `launch` ×1) — "four causes" is a miscount for "four files". The substance is unaffected;
recorded here rather than patched in the shared file, which §2 of this brief forbids modifying
(**T7‑D22**, §8).

### 7.7 Warnings that the green counts hide

`MEASURED` from the same logs: 41 warnings in the identity whole‑package run, 9 in the
`exp7_offline_exp_id` per‑file run, 3 in family A's `pytest_ignored.log`.

| Warning | Site | Reading |
|---|---|---|
| `DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead` | `edgesync_identity/nodes/identity_node.py:239` and `:261` (18 hits under `test_identity_node_transreid_integration`, 9 under `test_identity_node_model_resolution`) | Cosmetic against current `rclpy`; becomes a silent logging failure if the alias is removed. Not a Task 7 blocker; **not fixed** (§2 forbids code edits) |
| `RuntimeWarning: Mean of empty slice` | `analyze_task7_statistics.py:871` and `:874` | The degenerate‑population path exercised *deliberately* by `TestDegenerateSmokePopulation`; the assertion is that the value becomes `null`, not `NaN`, so §6.0's rule absorbs it and the warning is expected evidence |
| `RuntimeWarning: invalid value encountered in scalar divide` | `numpy/core/_methods.py:129` | Same path, one layer down |
| `UserWarning: Initializing zero-element tensors is a no-op` | `torch/nn/init.py:511`, from `test_experiment7_fixes.py` ×2 and `test_experiment7_forensic_fixes.py` ×1 | **Substantive**: those tests instantiate the OSNet placeholder backbone (§6.10.3 layer 3, **T7‑D09**). The warning is the observable trace of a stand‑in model holding empty tensors |

### 7.8 Not tested, and not testable from this package

| Item | Class | Why |
|---|---|---|
| Mutation / adversarial checks on Task 7's guards | **`NOT TESTED`** | `VALIDATION/inherited_evidence/exp13_20261021/mutation_check_20261021_011202.txt` (8/8 killed) is a **Task 13** artefact. No equivalent exists here, so nothing demonstrates that the 360 assertions would fail if a guard were deleted |
| Property / metamorphic testing of the metric code (permute gallery, scale invariance) | `NOT TESTED` | The canonical‑vs‑per‑query mAP equality check (§6.9.3, CA‑04, measured difference 0.0) is the only cross‑implementation check that exists |
| Coverage measurement (`--cov`) | `NOT MEASURED` | No coverage number exists for Task 7; "360 passed" is not a coverage statement and is not used as one anywhere in this report |
| Concurrency / executor timing of `identity_node` | `NOT TESTED` | Tier N stubs are single‑threaded; no DDS, no executor (§10 level 7) |
| `ReidSignature` publication; `IdentityVerify.srv` call | `BLOCKED` | Tier R absent — and the service carries no camera field at all (T7‑D01, §6.10.4) |
| TensorRT engine build/inference; Jetson latency, memory, power | `BLOCKED` / `NOT MEASURED` | `RES/trt_blocked/probe_build_report.json` environment block is all‑negative; the harness executed **zero** calls (§6.8.4) |
| OSNet arm through the production factory | `BLOCKED` | Charter §3B freeze + the three failure layers of §6.10.3 |

### 7.9 Re‑run consistency — including the one command that ran twice

`MEASURED`: the identity whole‑package command was executed **twice** for family B, into
`identity_copy_run.txt` and into `identity_copy_run_skipreasons.txt`. Both report
`360 passed, 15 skipped, 41 warnings, 1 error`, with wall times **20.84 s** and **18.75 s**
(`CALCULATED` Δ = 2.09 s, 11.2 % — scheduling noise on the same host, and no performance claim is made from
it, per charter §7). Identical outcome counts across two independent invocations is the determinism evidence
available here; the seeds, hash pins and integrity gates of §5.7 and §6.0 are the other half.

Against family A: `RES/execution_evidence/pytest_baseline_prefix.txt` (project tree, 2026‑10‑19) shows both
faces of the `rclpy` problem exactly as `REP/evidence/pytest_full.log` (`1 error during collection`,
`REAL_EXIT=2`) and `REP/evidence/pytest_ignored.log` (`56 passed, 3 warnings`, `REAL_EXIT=0`, 9.89 s vs
12.23 s for the same deselected configuration in the baseline prefix). `CALCULATED`: family A's 56 and
family B's 360 are not comparable quantities — both merely exclude `test_identity_service.py`, while family
B additionally includes the 14 test files added afterwards (TransReID, ONNX, TRT, input‑adapter and
statistics groups, `MEASURED` from the 29 `NEW` entries in
`RES/integrity/es07_final_integrity_report.txt`). **No fresh run contradicted an inherited record**, and each
numeric difference between the families is attributable to an omitted asset or to the absent ROS 2 stack —
a claim of `TEST_EXECUTION_EVIDENCE.md:117-118` that this pass re‑derived rather than repeated.

### 7.10 Test‑side items entering the defect register

Six §7 findings are carried into §8 rather than left in prose: **T7‑D08 / T7‑D09** (the two stub/placeholder
layers made visible by §7.7's `zero-element tensors` warning), **T7‑D20** (the encoding gate is
unit‑proven only, and the audit's "no encoding check" sentence is superseded — both recorded, per §6.10.2), **T7‑D22** (the "four causes" miscount inside the delivered validation evidence),
**T7‑D23** (no mutation or coverage evidence for Task 7's guards), and **T7‑D24 / T7‑D25** (the omitted
414 MB checkpoint and fixtures: one root cause producing 2 failures, 12 of 15 skips and the §7.5 blind spot;
seen from the packaging side and from the test side).

## 8. Defect register and forensic findings

### 8.1 Register conventions, severity basis, id policy and the complete index

**Every entry carries the same seven fields.** ID · finding · evidence (package‑relative path, with the
line or field inside it) · classification from §Evidence vocabulary · impact · severity · corrective action
and acceptance criterion. Nothing in this register is asserted without a path that a reviewer can open in
this package, or an explicit label saying the path is missing.

**Severity is an assessment made by this report, not a project‑specified scale.** `MEASURED`: no document
in `TASK07/reports/`, `TASK07/manifests/` or the shared configuration defines a severity vocabulary — the
requirement matrix uses *verdicts* (`FAIL — VERIFIED`, `UNVERIFIED CLAIM`, …) and the charter uses gates.
The four levels below are therefore this report's own definitions, applied consistently:

| Level | Definition used here |
|---|---|
| **Critical** | The task's central question cannot be answered, or a reader would be misled about what was measured. |
| **High** | A delivered code or configuration path produces a wrong, meaningless or unverifiable result if used. |
| **Moderate** | A real gap that bounds a claim (coverage, provenance, packaging) but does not corrupt a delivered number. |
| **Low** | Documentation, naming or citation defects with no numeric consequence. |

**Id policy.** Ids are cited from the section that first discusses the finding, so they are **not** in
numeric order by cluster. **T7‑D05, T7‑D07, T7‑D10, T7‑D12, T7‑D13 and T7‑D17 are deliberately unassigned**
(`MEASURED`: `grep` over this file returns zero occurrences of those six). They were consumed by an earlier
numbering pass and left as gaps rather than re‑used, so that no id in this report carries two meanings.
T7‑D26 and T7‑D27 are new in this pass and had no earlier numbering claim on them. **T7‑D28 is not a
new finding** — it is the inherited forensic item **B‑08** (`ES-EXP07-AUDIT-FORENSIC-01.md:493`, which
already names `_zero_streak` / `_zero_streak_count` and the constant −1) re‑verified against the current
tree at a different line (`identity_node.py:296` then, `:618` now); it carries a T7 id so that this
register is self‑contained, and §8.6 keeps the cross‑walk. **T7‑D29 is new in this pass** (found by source
re‑read; its dependency half is RTM R‑38). Two
corrections made in this pass are recorded rather than silently applied: the code pointer first written
for T7‑D27 was `analyze_task7_statistics.py:1140-1142`, which is actually the arm‑absent
`logger.warning` (`:1140`), and it has been re‑verified to `:1212-1214` in all four places that cited it;
and the finding first registered as T7‑D20 ("no encoding check") was too strong and is restated below as
a Tier‑R coverage gap.

**Complete index.** "Cluster" is the subsection holding the full entry. Statuses use the §Evidence
vocabulary; a status of `FIXED` always names the verification that supports it.

| ID | Finding in one line | Cluster | Status | Severity (assessment) |
|---|---|---|---|---|
| T7‑D01 | The charter's own trade‑off question (H7a/H7b/H7c) is unanswered on both of its axes | 8.5 | `NOT MEASURED` / `BLOCKED` | Critical |
| T7‑D02 | No `sensor_msgs/Image` producer exists, so the delivered output has never been produced | 8.5 | `BLOCKED` | High |
| T7‑D03 | The integration guide's switch value does not select the file the guide names | 8.5 | `UNRESOLVED` | Moderate |
| T7‑D04 | Projection assets are selected by **filename** across a namespace holding four contents | 8.3 | `IMPLEMENTED` (runtime), defect live | High |
| T7‑D06 | `osnet_torch` raises `RuntimeError` past an `except ImportError`, breaking the extractor contract | 8.2 | `MEASURED`, frozen by charter §3B | High |
| T7‑D08 | `_execute_trt()` returns `np.zeros(512)`, and without an engine `osnet_trt` ≡ `osnet_mock` | 8.2 | `IMPLEMENTED` + `NOT MEASURED` | High |
| T7‑D09 | The OSNet placeholder backbone shares 0 of 567 checkpoint keys (42,816 parameters) | 8.2 | `MEASURED` | High |
| T7‑D11 | The guide's `reid_model` is only an alias; the shipped canonical key is `reid_model_type` | 8.5 | `IMPLEMENTED` + `TESTED` | Low |
| T7‑D14 | Three unit‑norm deviation figures and two mean‑norm figures for one quantity | 8.3 | `RESOLVED` as a field‑name collision | Low |
| T7‑D15 | A status document cites `_real_v3_reproduce.npz` where the overlay pins `_real_v2.npz` | 8.3 | `MEASURED` | Low |
| T7‑D16 | The ONNX parity gate runs on 8 images, all of them distractor class `pid 0`, all `camid 1` | 8.4 | `MEASURED` | Moderate |
| T7‑D18 | `profile_edge_latency.py` synthesises Jetson numbers (+18.0 ms fudge) and writes a live config | 8.3 | `UNRESOLVED` (file absent) | High |
| T7‑D19 | The parity artefact never states which camera label was fed to the SIE graph | 8.3 | `MEASURED` (re‑read) | Low |
| T7‑D20 | The image‑encoding gate is proven only on duck‑typed stand‑ins, never on a live message | 8.4 | `IMPLEMENTED` + `TESTED` (Tier U) | Moderate |
| T7‑D21 | `analyze_task7_statistics.markdown()` crashed on a zero‑valid‑query arm while printing `COMPUTED 0.00 %` | 8.4 | `FIXED` — unit‑verified | Low (residual) |
| T7‑D22 | The delivered validation document miscounts "four causes" where it lists five files and two causes | 8.3 | `MEASURED` (re‑read) | Low |
| T7‑D23 | No mutation or coverage evidence exists for Task 7's guards | 8.5 | `NOT TESTED` / `NOT MEASURED` | Moderate |
| T7‑D24 | The 414,724,314 B checkpoint is not in the package: 2 failures, 12 of 15 skips | 8.3 | `MEASURED` | Moderate |
| T7‑D25 | `tests/fixtures/` is not in the package, so 8 scaffolding tests can never run here | 8.3 | `MEASURED` | Low |
| T7‑D26 | τ never reaches the runtime node from `algorithm_thresholds.yaml`; the code default is what fires | 8.4 | `IMPLEMENTED` + `TESTED` (non‑propagation pinned) | High |
| T7‑D27 | A shipped string in `analyze_task7_statistics.py:1212-1214` gives a false reason for a correct verdict | 8.4 | `MEASURED` | Moderate |
| T7‑D28 | `identity_node.py:618` reads `_zero_streak`, which the smoother never defines, so the log always says `smoother_streak=-1` | 8.4 | `MEASURED` (source re‑read) = inherited **B‑08** | Moderate |
| T7‑D29 | `preprocess()` silently switches resize algorithm when `cv2` is absent, and OpenCV is an undeclared dependency | 8.4 | `MEASURED` (source re‑read) | High |
| T7‑D30 | The requirement matrix's R‑10 row forwards its camera decision to the wrong row (R‑32 instead of R‑11) | 9.1 | `MEASURED` (matrix re‑read) | Low |
| T7‑D31 | `osnet_backbone.py` fetches un‑pinned weights from Google Drive under its own defaults, and R‑16/R‑17's recommended fix arms that path | 9.3 | `MEASURED` (source); path `NOT REACHED` in the shipped graph | Moderate |
| T7‑D32 | The device benchmark harness defaults to 20 warm‑up while the latency protocol is pre‑registered at 50, and its usage example passes 20 | 9.5 | `MEASURED` (source); harness `NOT TESTED`, `BLOCKED` at G.7 | Moderate |

Counts: **26 entries**. By severity (this report's assessment): **1 Critical** (T7‑D01),
**8 High** (T7‑D02, D04, D06, D08, D09, D18, D26, D29), **9 Moderate** (T7‑D03, D16, D20, D23, D24, D27,
D28, D31, D32), **8 Low** (T7‑D11, D14, D15, D19, D21, D22, D25, D30). Of these,
**3 are `FIXED` or `RESOLVED` with unit‑level verification** (T7‑D14, T7‑D21 and, in part, T7‑D20),
**5 are `BLOCKED`/`NOT MEASURED` on this host** (T7‑D01, T7‑D02, T7‑D18, T7‑D23, and the Tier‑R half of
T7‑D20), and **the remaining 18 are live findings about records, assets, delivered code or its
configuration channels**.
All twelve §6.11 negative results map onto this register in §8.6.

**Three entries carry their full seven‑field text in §9, not in §8**, because they were found while
re‑verifying requirement rows and the evidence that supports them belongs to those rows: **T7‑D30** (§9.1),
**T7‑D31** (§9.3), **T7‑D32** (§9.5). All three are new in this pass; none had an earlier numbering claim on
it, and none duplicates an inherited id — T7‑D31 and T7‑D32 both touch RTM **R‑38**'s undeclared‑dependency
theme but are distinct findings (a fetch path and a protocol mismatch, respectively).

### 8.2 Cluster A — the frozen OSNet comparator (T7‑D06, T7‑D08, T7‑D09)

These three are one story with three layers, and the story is bounded by charter §3B: OSNet is the frozen
baseline, so **this pass inspected it and repaired nothing**. `MEASURED` from the delivered source in this
pass; the recorded failure string comes from an artefact, not from a run performed here.

**T7‑D08 — the `osnet_trt` arm has no engine to run and its TRT call is a stub.**
*Evidence*: `SNAP/edgesync_identity/edgesync_identity/models/osnet_trt_extractor.py` — `_execute_trt()` at
`:72‑75` (body: comment, then `return np.zeros(512, dtype=np.float32)`); the dispatch `if self._trt_context
is not None:` at `:83` and the `else` branch at `:86‑89` that returns `np.tile(mean_vals, 512 // 3 + 1)[:512]`
from `np.mean(image_crop, axis=(0, 1))`; registry `extractor_factory.py:141‑157` binds **both**
`osnet_mock` and `osnet_trt` to `osnet_trt_extractor.OsnetTrtExtractor`. Engine absence: `MEASURED` by
recursive search of the whole package for `*.engine`, `*.plan`, `*.trt` → 0 hits (§6.8.4, §3.7);
`RES/onnx/transreid_engine_build_report.json` environment block is all‑negative.
*Classification*: `IMPLEMENTED` (the code as written) · `NOT MEASURED` (any real inference) ·
`BLOCKED` (no engine, no device).
*Impact*: on every host in this evidence set the two registry ids are **behaviourally identical** — both
produce a normalised tiled channel mean, a 512‑D unit vector that carries no identity information. Because
it is unit‑norm and non‑zero, it passes the §6.6 shape and norm checks and would pass a "did the extractor
return a feature?" smoke test. Any cosine computed against it is a function of mean pixel colour only.
`REP/ES-EXP07-AUDIT-FORENSIC-01.md` finding 6 records the same mechanism as the reason the shipped default
"silently runs a mock".
*Severity (assessment)*: **High** — a meaningless vector is indistinguishable from a real one at the
interface boundary.
*Corrective action*: make the arm fail closed (raise at construction when `_trt_context is None` and an
engine path is required, exactly as `osnet_torch_extractor.py:45‑49` already does for a missing checkpoint),
or return the `zeros(512)` sentinel with an attributable `last_failure`, and separate `osnet_mock` from
`osnet_trt` into distinct classes.
*Acceptance criterion*: a unit test that selects `osnet_trt` with no engine present and asserts either a
typed exception naming the missing artefact, or a sentinel return whose `last_failure['stage']` is
recorded; plus a registry test asserting `osnet_mock` and `osnet_trt` resolve to different classes.

**T7‑D06 — `osnet_torch` cannot be constructed here, and the intended fallback is unreachable.**
*Evidence*: `osnet_torch_extractor.py:62‑74` — `from torchreid.models import build_model` inside a `try`
whose only handler is `except ImportError` at `:74`; `_load_model()` declared at `:56`, invoked from
`__init__` at `:54`; recorded outcome `RES/embeddings/task7_embeddings_manifest.json → osnet_factory_attempt
.result` = *"FAILED — recorded, not repaired (charter §3B): RuntimeError: operator torchvision::nms does not
exist"*, and its `effect_on_comparison` field spells out the consequence for the paired test. The factory
frame is `extractor_factory.py:103‑109`.
*Classification*: `MEASURED` (the recorded string) + `IMPLEMENTED` (the `except` clause as written) +
frozen by charter §3B → **not repaired in this pass**.
*Impact*: selecting `osnet_torch` raises at construction rather than yielding the sentinel; on this host the
baseline arm cannot be produced at all, which is the direct cause of H7a being `NOT MEASURED` (T7‑D01).
*Correction to the inherited record*: CA‑16 in
`REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md` says the `RuntimeError` "propagates out of
`extract_embedding`". Re‑reading the file in this pass shows `extract_embedding` has its own catch‑all at
`:187‑189` that returns `zeros(512)`, so the escape happens one level earlier, in `_load_model()`. The
finding survives — the arm is unconstructable — but the mechanism is construction, not extraction, and the
§6.10.3 row now says so.
*Severity (assessment)*: **High**.
*Corrective action*: owner decision, because §3B forbids unilateral repair — either widen the handler to
`(ImportError, RuntimeError)` with an explicit "baseline unavailable" log, or remove `osnet_torch` from the
registry on hosts where `torchreid` cannot import, so selection fails at the factory with a named reason.
*Acceptance criterion*: on a host with a working `torchreid`, `build_extractor('osnet_torch', …)` returns an
`OsnetTorchExtractor` whose `load_state_dict` reports 0 missing / 0 unexpected keys for
`osnet_x1_0_market1501.pth`; on a host without it, the factory raises a typed error naming the import that
failed, and no zeros‑sentinel embedding is silently produced.

**T7‑D09 — the fallback backbone is a stand‑in that consumes none of the checkpoint.**
*Evidence*: `osnet_torch_extractor.py:93‑124` (`_build_osnet_standalone`, with its own inline `forward` at
`:118`) and `:140‑143` (`preprocess` using ImageNet mean/std);
`RES/provenance/task7_provenance_measurements.json → checkpoints.osnet_placeholder_backbone` =
`entries 8`, `total_elements 42,945`, `buffer_elements 129`, `trainable_elements 42,816`,
`key_overlap_with_checkpoint 0`, `strict_false_missing_keys 7`, `strict_false_unexpected_keys 567`,
`source "edgesync_identity/models/osnet_torch_extractor.py:93"`; contrast the real checkpoint at the same
path: 567 entries, 2,577,891 total, 2,554,771 trainable (`CA‑05`, §6.8.1). Observable trace in the delivered
test run: `RuntimeWarning: Initializing zero-element tensors is a no-op` at 3 sites (§7.7).
*Classification*: `MEASURED` from an in‑package provenance artefact, produced by the wrapped model built in
`RES/provenance/task7_provenance_measurements.json`; `IMPLEMENTED` for the code shape.
*Impact*: even if T7‑D06 were fixed by routing into the fallback, the fallback would load **none** of the
baseline weights (0 of 567 key names shared) and would emit embeddings from a randomly‑initialised 42,816‑
parameter CNN. `CALCULATED`: 42,816 ÷ 2,554,771 = **1.676 %** of the baseline's trainable mass. Any
"OSNet" number produced through that path is not the baseline.
*Severity (assessment)*: **High**.
*Corrective action*: §3B freeze applies; the record‑side action is that any future run must assert
`missing_keys == [] and unexpected_keys == []` **before** reporting an OSNet metric, and refuse otherwise.
*Acceptance criterion*: a regression test that loads the checkpoint into the built model and asserts the
element count equals 2,554,771 plus buffers, failing loudly if the placeholder is selected.

### 8.3 Cluster B — records, assets and provenance conflicts (T7‑D04, D14, D15, D18, D19, D22, D24, D25)

**T7‑D04 — projection assets are selected by filename across a namespace that holds four contents.**
*Evidence*: `CFG/exp7_transreid.yaml:57` and `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:47`
both give a **path**, not a digest. `MEASURED` in this pass by re‑hashing every `.npz` under
`TASK07/assets/edgesync_bringup/models/` and its `SNAP` twin: **16 files, 4 distinct SHA‑256**
(`9626bea5526df5b0` ×4 @1,466,587 B; `dc2b9072850c299d` ×6 @1,466,710 B; `c9bca7595cb67d63` ×4 @1,464,564 B;
`628b1dc6bcec1c22` ×2 @1,461,626 B — the ×2 is this package shipping every asset twice, so the project tree
count is 8 files / 4 contents). The variance fingerprints in
`RES/provenance/task7_provenance_measurements.json → projections` show what a filename swap does: R₅₁₂ is
**0.9864118099212646** for the pinned family and **0.9857138991355896** for the `dc2b9072` family, and
`gate3_real_results.json` carries the second while `_v2_corrected`/`_v3_reproduce` carry the first (§6.9.6,
CA‑14) — so a recorded "correction" silently changed matrix as well as preprocessing.
`mean_l2` separates them too (10.372914292029659 vs 19.389999510821742), but the orthonormality probe does
not: `columns_orthonormal_max_abs_deviation` is 8.75e‑09 … 1.25e‑08 for **all four**, so an operator
inspecting "is this a valid projection?" cannot tell them apart.
*Classification*: `IMPLEMENTED` (name‑based selection as written) · `MEASURED` (the census) · defect
**live** in the runtime path; `RES/provenance` and the embedding manifest pin the digest, and the *offline
consumer* re‑verifies it, but no runtime extractor does.
*Impact*: RTM **R‑13 `FAIL — VERIFIED`**. A rename or replacement changes the embedding space without
changing any check that the runtime performs, and every §6.1 arm figure is a function of the pinned matrix.
*Severity (assessment)*: **High**.
*Corrective action*: carry a SHA‑256 alongside each path (the overlay already names the file the manifest
pins), load it through the same integrity gate the offline consumer uses, and refuse to start on mismatch.
*Acceptance criterion*: a test that points `reid_projection_path` at a same‑shaped, different‑content `.npz`
and asserts a typed refusal naming both digests.

**T7‑D14 — three unit‑norm deviation figures and two mean‑norm figures for one quantity.** `RESOLVED`.
*Evidence and resolution*: §6.6 — the per‑arm block of `RES/embeddings/task7_embeddings_manifest.json` and
the `arms.<arm>.population.<side>` block of `RES/statistics/task7_statistics.json` use the **same field name
for two different statistics**, which is why 1.3248880670335694e‑07, 1.3504487839099966e‑07 and
1.2477772171060053e‑07 all appear "max deviation" while `mean_norm` appears twice per arm. Every value is
consistent with the others once the producing block is named; none is a numeric conflict, and all sit inside
the `1e‑4` contract tolerance in `interfaces/base_identity_extractor.py:5`.
*Impact*: the contract evidence was internally inconsistent until the field collision was identified — which
is why §6.6 lists every variant instead of choosing one.
*Severity (assessment)*: **Low** (naming), **High had it been reported as a numeric disagreement**.
*Corrective action*: distinct names per producer (e.g. `unit_norm_max_deviation_vs_contract` vs
`unit_norm_max_deviation_float32`). *Acceptance*: one name per quantity in both artefacts, asserted by a
schema test.

**T7‑D15 — a status document cites a different filename than the overlay pins.**
*Evidence*: `TASK07/TASK07_STATUS_AND_BLOCKERS.md:13` names
`svd_projection_768_to_512_real_v3_reproduce.npz`; `CFG/exp7_transreid.yaml:57` pins
`svd_projection_768_to_512_real_v2.npz`. `MEASURED`: the two files are **byte‑identical**
(`9626bea5526df5b0…`, 1,466,587 B). The same JSON records the pair under
`RES/provenance/task7_provenance_measurements.json → projections._relationships`, key
`"svd_projection_768_to_512_real_v3_reproduce.npz vs svd_projection_768_to_512_real_v2.npz"`, with
`byte_identical: true` and `max_abs_element_difference: 0.0`).
*Impact*: the citation is inconsistent while the content is not — the defect is in the record, not in the
asset. It matters because T7‑D04 makes filename‑level citations unsafe in general.
*Severity (assessment)*: **Low**. *Corrective action*: cite by digest with the filename as a label.
*Acceptance*: every projection citation in the delivered documents carries the `9626bea5…` prefix.

**T7‑D18 — the device profiling harness synthesised its own Jetson numbers and wrote a live config.**
*Evidence*: `MAN/requirement-traceability-matrix.csv` row R‑39 `FAIL — VERIFIED`:
`profile_edge_latency.py` "synthesises Jetson numbers, applies an +18.0 ms fudge, and writes into
`timing_budgets.yaml`"; the same statement appears in `REP/ES-EXP07-AUDIT-FORENSIC-01.md`. `MEASURED` in
this pass: a recursive search of the whole handoff package finds **no file named
`profile_edge_latency.py`** — the only artefact with "latency" in its name is
`RES/results/experiment7/benchmark_latency_memory.json`, whose generator is likewise absent (§6.9.1). The
target it wrote is live: `robot_bringup.launch.py:19` builds `timing_params` and passes it into the
`parameters=[…]` list of at least five nodes (`:116, :149, :162, :175, :189`), and
`SNAP/edgesync_bringup/config/timing_budgets.yaml:1‑2` states "All nodes consume timing parameters from this
file". `T_txn = 886 ms` is quoted from that file in §6.8.
*Classification*: `UNRESOLVED` in the strong sense — the evidence is a matrix row and a dated audit, not a
file that can be re‑inspected here.
*Impact*: a budget that any Task 13 timing claim inherits may have been produced by a fudge factor rather
than a measurement, and no artefact on disk distinguishes the two cases.
*Severity (assessment)*: **High**.
*Corrective action*: locate or discard the script, then re‑derive `timing_budgets.yaml` from a measured run
or mark each entry with its provenance; do **not** delete the file (it is a shared baseline asset).
*Acceptance*: every timing key in that YAML names either a measurement artefact or an explicit
`DESIGNED‑IN BUDGET, NOT MEASURED` marker.

**T7‑D19 — the ONNX parity artefact does not record the camera label it fed the SIE graph.**
*Evidence*: `RES/onnx/transreid_onnx_report.json` (and `…_rerun.json`) — eight per‑image records, each with
`pid 0`, `camid 1`, and a `cam_label` field reading **0** (§6.7). The SIE graph takes the label as a graph
input; the report never states which value was supplied at each invocation.
*Classification*: `MEASURED` (re‑read) + `UNRESOLVED` (intent of the recorded `cam_label`).
*Impact*: 8/8 PASS remains a real numerical gate, but the SIE leg cannot be attributed to a specific camera
conditioning — which is why §6.7 calls it "verified to exist, never verified to be the intended effect".
*Severity (assessment)*: **Low**.
*Corrective action*: have the export script echo `cam_label` per graph call into the report. Rerunning is out
of scope here: charter §5 and the standing instruction forbid regenerating artefacts to improve a report.
*Acceptance*: the report contains one `cam_label` per parity invocation and one per arm.

**T7‑D22 — the delivered validation document miscounts its own collection errors.**
*Evidence*: `VALIDATION/TEST_EXECUTION_EVIDENCE.md:87‑92` — "*five test files … cannot be collected*" then
"*None is a code defect; all four causes are the absent ROS 2 stack.*" `MEASURED` from the same file's table:
five files, **two** distinct causes (`rclpy` ×4, `launch` ×1). §7.6 of this report reproduces both strings.
*Impact*: none on any number; it is an internal inconsistency in the evidence document a reviewer is told
to trust. Recorded here rather than edited there — §2 of this brief forbids modifying that file.
*Severity (assessment)*: **Low**. *Corrective action*: erratum in the validation document ("four files, two
causes"). *Acceptance*: the sentence matches its own table.

**T7‑D24 / T7‑D25 — one packaging decision, two test‑facing consequences.**
*Evidence*: `models/transreid/transformer_120.pth` (414,724,314 B, `843055293e160da0…`) and
`tests/fixtures/` (`synthetic_transreid.pth` 414,718,880 B `2e518f28bb5ee501…`,
`synthetic_svd_768_to_512.npz`) are **referenced by digest, not copied**, per the package size policy
(`RES/integrity/es07_final_integrity_report.txt` records both sizes and digests; `find` over `TASK07/`
confirms absence). Consequences, `MEASURED` in §7.5–§7.6: 2 bringup failures
(`test_reused_assets_exist_in_the_package_tree`, `test_shipped_assets_exist_in_the_package_tree` — both
assert `os.path.isfile(…/transformer_120.pth)` inside the copied tree, `bringup_copy_run.txt:14‑27, :47`),
12 of the 15 identity skips (`identity_copy_run_skipreasons.txt:54‑68`), of which 8 are the `gate1`
fixtures‑absent skips (T7‑D25) and 4 the checkpoint/projection‑absent skips (T7‑D24).
*Classification*: `MEASURED` — and the arithmetic closes twice: `8 + 4 = 12` asset skips, `+2 ES07_HEAVY`
opt‑in gates, `+1` missing Market‑1501 directory = 15; and `360 + 12 = 372` matches the inherited
`identity_suite_excluded.txt` count exactly (§7.2).
*Impact*: the delivered suite verifies the **plumbing, not the weights**. No test in this package loads the
real checkpoint or the real projection, so no green count here says anything about the deployed model files.
*Severity (assessment)*: **Moderate** as packaging, **High** if read as a weight‑level verification.
*Corrective action*: either ship the checkpoint under a documented exception, or state in the package README
that the delivered suite is asset‑starved. *Acceptance criterion*: a run in which the 2 failures and 12
skips resolve, or a one‑line README statement that they cannot.

### 8.4 Cluster C — delivered code, its configuration channels and its verification state
(T7‑D16, T7‑D20, T7‑D21, T7‑D26, T7‑D27, T7‑D28, T7‑D29)

**T7‑D16 — the ONNX parity gate is decided by eight images that all come from one class and one
camera.**
*Evidence*: `SNAP/edgesync_identity/scripts/export_transreid_onnx.py:248` (`--parity-n`, `type=int`,
`default=8`) and `:222-230` — `pick_images()` returns `gallery[:count]` taken from
`collect_gallery_and_query(market_root, population='directory')`, so the sample is *the first eight
entries of the gallery in the loader's own order*, not a stratified draw. `MEASURED` (re‑read in this
pass) in `RES/onnx/transreid_onnx_report.json → parity` and identically in
`RES/onnx/transreid_onnx_report_rerun.json → parity`: `n_images_selected 8`, `n_images_scored 8`, and
every one of the eight rows carries `pid 0`, `camid 1`, `cam_label 0`, filename
`bounding_box_test/0000_c1s1_*.jpg`. The measured margins are large — agnostic arm worst
`max_abs_768` **1.239776611328125e‑05**, worst `max_abs_512_after_contract` **6.09084963798523e‑07**,
worst `cosine_768` **0.9999999999960024**; SIE arm **1.2874603271484375e‑05**,
**6.854534149169922e‑07**, **0.9999999999954788** — against `tolerance 0.002` and `cosine_floor 0.9999`,
so both arms read `verdict PASS`; `production_api_vs_torch_worst_max_abs_512` is **0.0** for both arms and
`sie_effect_worst_l2_on_512` is **0.15214425314786348**.
*Classification*: the numbers are `MEASURED`; the defect is in the sampling rule, which is
`IMPLEMENTED` exactly as described.
*Impact*: three limits follow, and none of them is a claim that the parity is wrong. (1) A gate decided
on 8 images carries no distributional statement — the report's own `caveat` field says the check
"validates the exported graph and the preprocessing/projection contract", and that is what 8 same‑class
images can do. (2) Because all eight `cam_label` values are **0**, the SIE graph is exercised on **1 of
the 6 supported label rows** (`MAX_CAMERA_LABELS = 6`, `reid_input_adapter.py:24`; the embedding table is
`[6, 1, 768]`), so a mis‑wired label index for rows 1‑5 could not be caught here. (3) All eight come from
`bounding_box_test`, the evaluation split; nothing is fitted from them, so this is not test‑set tuning,
but it does mean the gate and the projection fit share no image.
*Severity (assessment)*: **Moderate** — it bounds what the PASS verdict proves; it corrupts no delivered
number.
*Corrective action*: draw the parity sample across identities and cameras (e.g. ≥ 2 images × 6 cameras ×
≥ 3 identities including one `pid 0`) and add a label sweep that runs the SIE graph once per label in
`0…5`.
*Acceptance*: the parity artefact records ≥ 18 distinct `(pid, camid)` pairs **and** a six‑row
`cam_label` sweep in which the 768‑D output changes when the label changes; the gate fails if any label
row is constant.

**T7‑D20 — the image‑encoding gate exists and is tested, but only below the live‑message level.**
*Evidence*: the gate is real — `SNAP/edgesync_identity/edgesync_identity/core/reid_input_adapter.py:20‑21`
(`SUPPORTED_ENCODINGS_BGR = ('bgr8', '8uc3')`, `SUPPORTED_ENCODINGS_RGB = ('rgb8',)`), `:42` (encoding
normalised: `str(...).lower().split(';')[0].strip()`), `:69` (RGB passes through), `:72‑73`
(`arr[:, :, ::-1]` for BGR), `:75‑78` (`ValueError("unsupported image encoding …")`). Both production
entry points call it: `identity_node.py:556` (service path, `image_to_rgb(request.candidate_crop)`) and
`:354` (crop‑topic path). `TESTED`: `TST/edgesync_identity/tests/test_transreid_input_adapter.py` — 31
collected, 31 passed, 0 failed in this pass — covers case‑insensitivity/`8uc3` (`:57`) and the refusal of
`mono8`, `bgra8`, `16UC1` and `''` (`:77‑81`).
*What it does not prove*: every message in that suite is a duck‑typed stand‑in (`:25`
`"""Duck-typed stand-in for sensor_msgs/msg/Image."""`, `_Image` at `:27`), and so is the node‑level one
(`test_identity_node_transreid_integration.py:88`). No test has ever handed the adapter a
`sensor_msgs/Image` produced by a DDS subscription, because `rclpy` and `sensor_msgs` are absent here
(`NOT TESTED` at Tier R).
*Relationship to the inherited record*: `REP/ES-EXP07-AUDIT-FORENSIC-01.md:42` states that "the
production node performs **no `sensor_msgs/Image` encoding check**". That sentence is **superseded** —
TR‑04 (`ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md:32`) implemented the check and this pass
re‑read both the code and its test. The correction and the reason it mattered are in §6.10.2; the channel
cost measured there (35.17241379310345 pp of cosine mean between BGR‑ and RGB‑ordered input) stands.
*Severity (assessment)*: **Moderate** (`IMPLEMENTED` + `TESTED` at Tier U/N; the Tier‑R half is
`NOT TESTED`).
*Corrective action*: add one Tier‑R assertion — subscribe to a real `sensor_msgs/Image`, publish a
`bgra8` frame and require the service to refuse it — on a built workspace.
*Acceptance*: a test that **fails if the `encoding` field is ignored**, executed against a workspace where
`ros2 topic pub` produced the message.

**T7‑D21 — the statistics renderer crashed on a degenerate arm while labelling the row `COMPUTED`
(fixed, unit‑verified).**
*Evidence*: `REP/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md:472` (row 37) records the failure verbatim —
`analyze_task7_statistics.markdown()` raised `TypeError: 'NoneType' object is not subscriptable` on a
zero‑valid‑query arm, and *before* the exception the row was already labelled `COMPUTED` at `0.00 %` with
`NaN` coverage means; after the fix the same row renders `0.00 (—) / 0.00 / 0.00 / 0.00 (—)` with the
label `NOT MEASURED — none of the 2 queries has a gallery identity match, so Rank-k and mAP have no
denominator`, and `identity_coverage` means are `null`. Pinned by 3 `TestDegenerateSmokePopulation` cases
inside row 34's 92 (`TESTED`, Tier U). Same story restated at
`REP/ES-EXP07-TASK7-INTEGRATION-VERIFICATION-REPORT.md:100‑106`.
*Status*: **`FIXED`** — supported by the delivered code path plus the three regression cases; not
supported by any system‑level run.
*Residual*: the fix is renderer behaviour only. Because no recorded run has a zero‑valid population, **no
previously reported number changes** (row 37's own closing clause).
*Severity (assessment)*: **Low (residual)**. *Corrective action*: none in Task 7 scope; if `INTEGRATION/`
reuses the renderer, carry the three degenerate cases with it. *Acceptance*: already met at Tier U — a
zero‑valid arm yields `NOT MEASURED` and `null`, never `COMPUTED`/`NaN`.

**T7‑D26 — τ is mandated to come from YAML, and the channel that is supposed to deliver it cannot.**
This is the highest‑severity configuration finding in the register, and it changes no number today.
*Evidence chain*, each link re‑read in this package:
1. `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:2` — "ZERO-HARDCODING MANDATE: All algorithms, thresholds, and
   ablation flags are injected via parameters."; `:4` root key `algorithm_thresholds:`; `:5`
   `ros__parameters:`; `:7` `reid_similarity_threshold: 0.70`.
2. `identity_node.py:7‑8` — ISSUE‑10 "Parameter keys harmonized with algorithm_thresholds.yaml
   (reid_similarity_threshold → similarity_threshold, etc.)". The node therefore declares the
   **renamed** key: `:79` `declare_parameter('similarity_threshold', 0.70)`, read at `:110`, passed to the
   verifier at `:172`, logged at `:212` as `threshold={…:.2f}`.
3. `identity_verifier.py:22` constructor default `similarity_threshold: float = 0.70`; `:101` rejects when
   `sim < self.similarity_threshold`.
4. `robot_bringup.launch.py:122‑140` — `node_identity` is named `identity_node` (`:125`) and receives
   `parameters=[algo_params, {robot_id, reid_model_type, reid_checkpoint_path, reid_projection_path,
   reid_engine_path, exp_id, reid_camera_id}]` (`:127‑137`). **No `similarity_threshold` key appears in
   that dict**, and the file it also passes is keyed to a node named `algorithm_thresholds`.
5. `MEASURED` (grep over the whole snapshot): `reid_similarity_threshold` occurs in exactly one YAML
   (`algorithm_thresholds.yaml:7`) and in Python only as (a) the offline reader's lookup and provenance
   string (`analyze_task7_statistics.py:703`, `:705`) and (b) two test fixtures. `allow_undeclared_parameters`
   appears **nowhere**.
6. `TESTED`: the non‑propagation is pinned *as intended behaviour* by
   `SNAP/edgesync_bringup/tests/test_reid_launch_defaults.py:111‑115`, which writes
   `reid_similarity_threshold: 0.70` into a fixture and asserts `'reid_similarity_threshold' not in found`.
*Why it is a defect even though nothing is numerically wrong*: the YAML value and the code default are
both 0.70, so the delivered FNMR@τ figures and the node's accept rule agree today. The mandate at `:2` is
nevertheless unmet, and the failure mode is silent: a reviewer who edits
`reid_similarity_threshold` — which §6.7 shows is the single most consequential number in this task,
sitting far above every arm's EER (τ_EER ≈ 0.31‑0.37, FNMR 48‑61 % at 0.70) — changes a file the runtime
never reads, and the startup log at `:212` prints `threshold=0.70` identically whether the value arrived
by parameter or fell back to the default. The log cannot distinguish "injected" from "defaulted".
*Contradiction with the inherited record*: `MAN/requirement-traceability-matrix.csv` **R‑23** is
`PASS — VERIFIED` for "Threshold tau = 0.70 applied by the actual code path (not only written in YAML)",
with evidence "Parameter declared, **injected**, used". Against the current tree only *declared* and
*used* hold. §9 therefore records R‑23 as **PARTIAL** and names this defect as the reason. The RTM's own
line pointers (`identity_node.py:52`) belong to the older tree; the same statements resolve at `:79/:110/:172`
today.
*Severity (assessment)*: **High**. *Corrective action*: either add `'similarity_threshold': <value from
the resolved overlay>` to the launch dict at `:129‑137`, or rename the YAML key to the node's parameter
name under a root key that matches the node (`identity_node:` or a `/**` wildcard) — and, either way,
report the value's origin.
*Acceptance*: a test that sets τ ≠ 0.70 in the YAML and asserts the running node's
`similarity_threshold` equals it, plus a provenance field that distinguishes `yaml` from `default` (the
same mechanism `describe_reid_sources()` already uses for the model key, `reid_launch_defaults.py:317‑330`).

**T7‑D27 — a shipped string explains a correct verdict with a false reason.**
*Evidence*: `SNAP/edgesync_identity/scripts/analyze_task7_statistics.py:1212‑1214` —
`statistical_comparison_against_recorded_gate3_numbers` is printed as *"NOT APPLICABLE — the recorded run
used the invalid id_band population (TR‑07), so its per-query outcomes are not the same population as
this one"*. `MEASURED` refutation in §6.9.7: all four Gate‑3 artefacts record `n_query 3,368` /
`n_gallery 15,913`, i.e. the **directory** population, the same one this pass uses; the `id_band_752_1501`
rule is genuine (`evaluate_offline.py:155`, band registered at `:84`) but would give **gallery 6,226 /
query 1,774 over 395 identities** (`RES/provenance/task7_provenance_measurements.json →
dataset.identity_set_algebra.id_band_752_1501_files`), which is not what those files report.
*The verdict survives*: a paired comparison really is impossible, because no per‑query OSNet/Gate‑3
outcomes exist on disk anywhere in the package — the blocker is missing data, not a population mismatch.
*Status*: `MEASURED`, documented here and in `REP/ES-EXP07-TASK7-REPORT-01-REV02.md` §6 threat 10, **not
patched** — §2 of the brief governing this pass forbids editing code, and patching the string would also
obligate a re‑run of the script's 68 tests for a label that does not change.
*Severity (assessment)*: **Moderate** — a reader who accepts the printed reason believes the populations
were reconciled and found different; the actual state is that no comparison population can be built.
*Corrective action*: replace the clause with the measured reason and keep `NOT APPLICABLE`.
*Acceptance*: every `not_measured.*` string in the artefact asserts only what a file on disk supports.

**T7‑D28 — the node's smoother telemetry reads an attribute that does not exist, so it always reports
−1.** *This is the inherited forensic item **B‑08*** (`REP/ES-EXP07-AUDIT-FORENSIC-01.md:493`: "`identity_node.py:296`
reads `_zero_streak` (real attribute `_zero_streak_count`) → observability constant −1"), **re‑verified in
this pass** against the packaged tree at its current location and given a T7 id so this register is
self‑contained. Nothing here was produced by re‑running anything.
*Evidence*: `identity_node.py:618` — `smoother_streak = getattr(self.smoother, '_zero_streak', -1)`. The
attribute is not called that: `signature_smoother.py:45` defines `self._zero_streak_count`, incremented at
`:80`, compared at `:83`, reset at `:95` and `:127`, and published as the property `zero_streak_count` at
`:133‑136` (with `is_flushed` derived from the same counter, `:138‑142`). `MEASURED` (grep over the
snapshot): `_zero_streak` **without** the `_count` suffix appears in exactly one place in delivered code —
`identity_node.py:618`. The value flows only into the two log lines `:634` (MATCH) and `:641` (MISMATCH).
*Impact*: every identity log line this node can produce reports `smoother_streak=-1`, including after a
3‑frame flush. The smoothing decision logic itself is unaffected and is `TESTED` — eight assertions in
`TST/edgesync_identity/tests/test_gate1_scaffolding.py:38‑142` check `smoother.zero_streak_count`
directly. So this is an **observability** defect, not a decision defect: on the Jetson session it would
tell the operator that the stream had never produced an invalid frame, at the exact moment that
information matters. It is also the reason §10's level‑9 checklist cannot use existing logs as evidence.
*Why the suite missed it*: the tests assert the smoother's own counter; no test asserts the node's log
payload. That is a concrete instance of T7‑D23.
*Second finding in the same block*: `:605‑611` already measures `dt_id_ms` around
`verifier.verify_candidate(...)` and prints it at `:632`/`:639`. That is the cheapest H7b instrument in
the tree and it has never executed (`NOT MEASURED` — Tier R only); it must be read as **verifier‑only**
wall time, i.e. it excludes crop decode and extraction, so it is not the identity‑stage share of
`T_txn = 886 ms` that gate G.1 asks for.
*Severity (assessment)*: **Moderate**. *Corrective action*: read the documented property
(`getattr(self.smoother, 'zero_streak_count', -1)`).
*Acceptance*: a Tier‑N test that drives N invalid frames through the node's smoother and asserts the
logged `smoother_streak == N`, and a Tier‑R smoke log line whose streak value is non‑negative.

### 8.5 Cluster D — scope, integration and coverage findings (T7‑D01, T7‑D02, T7‑D03, T7‑D11, T7‑D23)

**T7‑D01 — the charter's own question is unanswered on two of its three axes.**
*Evidence*: the hypothesis as recorded in `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:41` —
"Hypothesis (H7): accuracy increases, but edge inference latency and edge memory footprint exceed the
handoff budget / 8 GB envelope, proving OSNet is the optimal trade‑off", with the per‑axis decision rules
at `:185‑187` (H7a "candidate mAP > baseline mAP"; H7b "candidate `end_to_end` p95 exceeds the identity
stage's share of `T_txn = 886 ms`"; H7c "candidate resident Δ plus the concurrent perception stack
exceeds the envelope the owner fixes before collection") and the delivered verdicts at `:195‑197`.
*Status by axis*: **H7a** `NOT VERIFIED BY MEASUREMENT HERE` — the baseline arm cannot be produced on this
host (T7‑D06/D08/D09) and the only OSNet figure available is the recorded aggregate
94.32897862232778 % Rank‑1 / 83.5692 mAP, which the same artefact's `n_query 3,368` / `n_gallery 15,913`
confirm is *the same population* as this pass's measurement, so the arithmetic comparison (mAP
**+1.5560 pp**, Rank‑1 **−0.9501 pp**) is computable but not a verification. **H7b** `NOT MEASURED` —
no engine, no device, and G.1 (the identity‑stage share) was never fixed. **H7c** `NOT MEASURED` — G.3
(8 GB envelope) open, G.7 (resident Δ) DENIED.
*Impact*: this is the reason §13's verdict cannot be "Task 7 answered its hypothesis". The accuracy axis
has evidence about **TransReID's own arms** (§6.1‑6.5); the trade‑off clause — which is the second half of
the sentence and the part that would recommend a model — has no device data at all, and the RTM records
that the report's own conclusion slot was left unfilled for exactly this reason (R‑40 `FAIL — VERIFIED`).
*Severity (assessment)*: **Critical** (the task's central question).
*Corrective action*: the two owner gates first (G.1, G.3 — A‑2/A‑5 in §11), then one Jetson session using
the locked harness (§12.2 lists what is missing to make that possible), then restate H7a with a runnable
baseline arm on a host where `torchreid` imports.
*Acceptance*: all three axes carry a measured value, each with a declared population/harness, and the
decision rule quoted above is applied literally rather than paraphrased.

**T7‑D02 — the delivered output has never been produced, because nothing publishes its input.**
*Evidence*: `MEASURED` (grep over the whole package): **zero** `create_publisher(...)` calls carrying a
`sensor_msgs/Image` exist in any delivered Python file — `TASK07/`, `TASK13/`, `SHARED_DEPENDENCIES/`,
`INTEGRATION/`, `VALIDATION/`. The only publisher in `identity_node.py` is
`create_publisher(ReidSignature, topic, 5)` (`:312`), and its crop **subscription** at `:332‑333` is only
reached when `reid_target_crop_topic` is non‑empty (`:314‑321`), which is exactly the case the node warns
about: `:317‑320` "…reid_target_crop_topic is unset: no observation feeds the extractor, so this node can
only ever publish an invalidation. Point reid_target_crop_topic at a sensor_msgs/Image target crop to
produce embeddings." The upstream cause is documented and independently confirmed:
`SNAP/edgesync_bringup/launch/exp13_bringup.launch.py:204‑208` leaves the parameter empty "on purpose",
because "perception_node computes a bounding box, deprojects it and publishes only the 3D centroid" — and
`SHARED_DEPENDENCIES/…/edgesync_perception/edgesync_perception/nodes/perception_node.py` indeed has a
single publisher, `PointStamped` on `output_centroid_topic` (`:138‑143`, published at `:283`), even though
it subscribes to `sensor_msgs.msg.Image` (`:18`).
*Status*: `BLOCKED` — and the block is upstream of Task 7's code, in the shared perception package.
*Impact*: the end‑to‑end artefact this task exists to produce — a live `ReidSignature` per target — has
never been emitted by a running node anywhere. The path is `IMPLEMENTED` (§3.9) and `TESTED` at Tier N
with stub `rclpy`, so this bounds the claim, not the code.
*Severity (assessment)*: **High**. *Corrective action*: publish the tracked crop — either a crop region out
of the frame `perception_node` already holds, or a small adapter node subscribing to the raw image plus the
`PointStamped`/bbox and republishing a `sensor_msgs/Image`; then set `reid_target_crop_topic`.
*Acceptance*: on a ROS 2 host, `ros2 topic info` shows a publisher on the crop topic and at least one
`ReidSignature` with a non‑zero embedding is captured from `ros2 topic echo`.

**T7‑D03 — the integration guide's switch value does not select the file the guide names, and the guide
is not in this package.**
*Evidence*: `SNAP/edgesync_identity/edgesync_identity/models/extractor_factory.py:137‑172` binds
`transreid` → `TransReIdExtractor` (PyTorch, "…centered truncated‑SVD 768->512 projection.") and
`transreid_trt` → `TransReIdTrtExtractor`; §6.8.4 establishes that no `.engine`/`.plan`/`.trt` file exists
anywhere in the package, so the `transreid_trt` branch has nothing to load. The guide's own text is the
half that cannot be checked here: `MEASURED` (searched) — **no file whose name contains "guide" exists
anywhere in `EdgeSync_My_Tasks_Handoff/`**, and the RTM says the same thing about its own audit at
`MAN/requirement-traceability-matrix.csv` R‑33: "guide not in tree so its exact mandate is unauditable".
*Status*: `UNRESOLVED` — the code half is `MEASURED`, the document half is **referenced but missing**
(§12.2). This report does not treat the guide's wording as verified.
*Impact*: a reviewer following a guide that says "select TransReID" gets the **PyTorch** arm, which is the
correct thing for accuracy numbers and the wrong thing for a deployment claim; the arm the guide names by
filename (`transreid_trt_extractor.py`) requires an asset that has never been built. Combined with
T7‑D11 (the guide's `reid_model` is an alias) and T7‑D26 (the same file's τ is not delivered), the pattern
is that the guide describes an injection channel the shipped code does not use.
*Severity (assessment)*: **Moderate**. *Corrective action*: replace the external guide with an
in‑package switch table (or restore the guide into `SHARED_DEPENDENCIES/`), stating per id the class, the
assets required, and which of them exist.
*Acceptance*: every documented switch → class → asset triple is checkable against this package, and any
id whose asset is absent is labelled `BLOCKED` in the table itself.

**T7‑D11 — the guide's parameter name is an alias, not the shipped key.**
*Evidence*: `SNAP/edgesync_bringup/edgesync_bringup/reid_launch_defaults.py:59‑67`
(`REID_LAUNCH_KEYS`, canonical `reid_model_type`), `:69‑70`
"Guide‑name alias accepted in config files, resolved onto the canonical key" / `MODEL_ALIAS_KEY = 'reid_model'`,
`:74` `RESOLVED_KEYS = REID_LAUNCH_KEYS + (MODEL_ALIAS_KEY,)`, `:84‑92` `HARDCODED_FALLBACKS`
(`reid_model_type: 'osnet_mock'`, alias `''`), `:96` `STATIC_BACKBONE_IDS` (the same five ids as the
registry), and the resolution rule at `:305‑330`: canonical wins (`canonical:<layer>`), else alias
(`alias:<layer>`), else fallback (`fallback`), with a same‑layer disagreement recorded as a conflict and
never dropped (`:311‑315`). Node side restates the precedence at `identity_node.py:20‑21`
("explicit canonical > explicit alias > registry default"). `TESTED`: 23 cases in
`TST/edgesync_identity/tests/test_identity_node_model_resolution.py` (§7.4) and the bring‑up overlay tests
that import `MODEL_ALIAS_KEY` (`test_exp7_config_overlay.py:15, :70, :228`).
*Status*: `IMPLEMENTED` + `TESTED` (Tier U/N). This is a documentation/naming defect, not a code defect.
*Impact*: a config that writes only `reid_model:` works, but its provenance is reported as `alias:…` and
the node's own declared parameter is `reid_model_type`; a config that writes **both** with different values
silently (well — with a WARN at `identity_node.py:240`) takes the canonical one. `MEASURED` in this pass:
no `.md` inside the package contains `reid_model: transreid` except this report, so the alias exists only
for a document that is not shipped here.
*Severity (assessment)*: **Low**. *Corrective action*: standardise documentation on `reid_model_type` and
keep the alias only while an external guide remains authoritative.
*Acceptance*: the delivered configuration files and any restored guide use the canonical key, and
`describe_reid_sources()` output asserting `canonical:`/`alias:`/`fallback` is part of the launch smoke test.

**T7‑D23 — there is no coverage or mutation evidence for Task 7's guards.**
*Evidence*: `MEASURED` (searched): no `.coveragerc`, no `coverage.xml`, no `.coverage`, no `mutmut*`
artefact anywhere in the package; the three `setup.cfg` files present are ament stubs containing only
`[develop] script_dir=…` and `[install] install_scripts=…`. No Task 7 document reports a line/branch
coverage figure or a mutation score — the only "coverage" statements in the tree are **checkpoint
parameter coverage** (100.0 %, `REP/ES-EXP07-TRANSREID-PREFLIGHT-REPORT.md:66`) and the
`identity_coverage` metric column, which is a retrieval quantity, not a test quantity.
*What the suite does prove*: §7.1‑7.6 enumerate 360 passed + 15 skipped (+2 failed in the bring‑up
package) with the guard‑bearing assertions named per file. *What nothing proves*: that those guards would
fail if removed. Two concrete blind spots were found by inspection in this pass and neither is covered by
any test — the node's telemetry attribute typo (T7‑D28: the smoother's counter is asserted 8 times, the
node's read of it never is) and the τ delivery channel (T7‑D26: the non‑propagation is asserted by a test
as intended behaviour, so a future fix would break that test rather than the runtime).
*Status*: `NOT TESTED` (mutation), `NOT MEASURED` (coverage).
*Severity (assessment)*: **Moderate** — it bounds the strength of every `TESTED` claim in §7 without
invalidating any of them.
*Corrective action*: run `pytest --cov` for both packages with a recorded floor and a stored `coverage.xml`,
then mutation‑test the guard‑bearing files (`identity_verifier.py`, `reid_input_adapter.py`,
`signature_smoother.py`, `extractor_factory.py`, `reid_launch_defaults.py`) with a named tool and version.
*Acceptance*: coverage and mutation artefacts land under `RES/` and every defect class in §8 that "no test
would notice" is either killed by a mutant or given an explicit test.

**T7‑D29 — `preprocess()` silently switches resize algorithm when OpenCV is missing, and OpenCV is not a
declared dependency.**
*Evidence*: the same construct is in all four delivered extractors — `transreid_extractor.py:222‑231`,
`transreid_trt_extractor.py:252‑260`, `osnet_torch_extractor.py:130‑138`, `osnet_trt_extractor.py:52‑60`
(`MEASURED`, re‑read): a `try: import cv2` / `cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)`
with an `except ImportError:` branch that substitutes index sampling
(`np.linspace(0, h-1, H).astype(int)` twice, then `crop[y_idx[:, None], x_idx]`) and **emits no log, no
`last_failure` record and no marker**. The dependency is undeclared twice over:
`SNAP/edgesync_identity/package.xml:10‑14` lists only `rclpy`, `sensor_msgs`, `edgesync_msgs` and
`<test_depend>pytest`, and `SNAP/edgesync_identity/setup.py:15` is `install_requires=['setuptools',
'numpy']`. RTM **R‑38** already records this half as `NOT TESTED` ("No torch/opencv/tensorrt declared"),
but does not connect it to the silent branch.
*Why every number in this report is still the `cv2` one*: `MEASURED` on this host — OpenCV **4.11.0**
imports, and the embedding generator itself calls `cv2.imread`/`cv2.cvtColor`
(`evaluate_offline.py:241, :246`), so all 19,281 crops and both ONNX parity arms went through
`cv2.resize`. The defect is therefore a **future/other‑host hazard**, not a flaw in a delivered result.
*Why the suite cannot see it*: `TST/edgesync_identity/tests/test_transreid_trt_extractor.py:147‑162`
asserts `trt_arm.preprocess(crop) == torch_arm.preprocess(crop).cpu().numpy()` — a **cross‑class** equality
that passes identically on both branches, because both classes take the same branch on the same host. No
test forces the `ImportError` path, and none compares the two resizers' outputs. This is the second
concrete instance of T7‑D23.
*Impact*: nearest‑neighbour index sampling and bilinear `INTER_LINEAR` are different functions; at the
contract geometry (a crop resized to 256×128) they differ in most pixels. A Jetson image without a Python
OpenCV binding would therefore produce a full set of embeddings that are internally consistent,
unit‑normalised, contract‑shaped, and **silently incomparable** to every number in §6 — while the node
would go on accepting or rejecting at τ. Because the fallback is not recorded anywhere, the resulting
mismatch would be attributed to the device, the engine, or fp16 rather than to preprocessing.
*Severity (assessment)*: **High** — a delivered code path changes the measurement with no diagnostic, and
the task's remaining work (T7‑D01) depends on those measurements being comparable.
*Corrective action*: declare the dependency (`<depend>python3-opencv</depend>` /
`install_requires=['opencv-python-headless']`, or an explicit `apt` line in the Jetson run sheet) **and**
make the fallback loud: log a WARN naming the branch, set `last_failure`/a `preprocess_backend` field on
the extractor, and have `EXP-METADATA` (`identity_node.py:217`) report which resizer was used.
*Acceptance*: (1) a test that removes `cv2` from `sys.modules` and asserts the node logs and reports the
fallback rather than proceeding silently; (2) a measured numeric gap between the two branches on at least
one pinned crop, so the cost is on record; (3) `rosdep`/`colcon build` succeeds from the declared
dependencies alone on the target.

### 8.6 Cross‑walk — inherited ids, negative results and requirement rows mapped onto this register

Task 7's findings have been numbered by **five independent systems** before this report existed, and all
five are still quoted inside the delivered documents. `MEASURED`: no file in this package except
`EXECUTIVE_SUMMARY_TASK07.md` contains the string `T7‑D`, so every mapping below is one‑directional and
none of the inherited ids has been rewritten in place.

| System | Lives in | What it numbers | Used here |
|---|---|---|---|
| `TR‑01 … TR‑11` | `REP/ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md:29‑39` | integration work items | 8.6.2 |
| `B‑01 … B‑19` | `REP/ES-EXP07-AUDIT-FORENSIC-01.md:486‑504` (severity class 1/2 columns) | forensic defects | 8.6.3 |
| `CA‑01 … CA‑17` | `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md:49‑209` | record conflicts | 8.6.3 |
| `D‑02, D‑04, D‑11, D‑12, D‑21, D‑22` | `REP/ES-EXP07-AUDIT-FORENSIC-01.md` (§absent‑artefact list) | missing deliverables | 8.6.3, §12.2 |
| `R‑01 … R‑42` | `MAN/requirement-traceability-matrix.csv` | requirement checks | §9 |
| `G.1, G.3, G.5, G.6, G.7` | `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:15‑17` + owner record | owner gates | 8.5, §11 |
| `P1` | `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:68` | frozen‑baseline status row | 8.2 |
| `N‑1 … N‑12` | §6.11 of this report | negative results | 8.6.1 |
| `T7‑D01 … T7‑D29` | §8 of this report | defect register | this section |

**8.6.1 §6.11's negative results, and where each one is now carried.**

| Negative | What it is | Carried as |
|---|---|---|
| N‑1 | Deployed arm is worse than the recorded baseline on every metric | evidence for **T7‑D01** (H7a contradicted for the reachable arm); §6.10.4 |
| N‑2 | With SIE, Rank‑1 still below baseline, mAP above | evidence for **T7‑D01**; §6.10.4 |
| N‑3 | τ = 0.70 sits at 2.137×/2.289× the arms' EER thresholds | **not a code defect** — a measured calibration finding; §6.3/§6.7 and action A‑4 in §11. Its *channel* half is **T7‑D26** |
| N‑4 | FNMR ≈ 48 %/61 % at τ, query‑level miss 11.3 %/18.9 % | measurement; §6.3, quoted in §13 |
| N‑5 | No device measurement of latency/throughput/memory/power/accuracy | **T7‑D01** (H7b/H7c) + §10 levels 8‑10; harness `BLOCKED` |
| N‑6 | The delivered output has never been produced | **T7‑D02** (+ Tier‑R absence, §10 levels 5‑7) |
| N‑7 | Parity used 8 images, all `pid 0`, all `camid 1` | **T7‑D16**, with the `cam_label` half as **T7‑D19** |
| N‑8 | H3 is power‑limited (154 discordant queries, p = 0.572845968999707) | **not a defect** — a limitation of the pre‑declared test; action A‑8 in §11 |
| N‑9 | The inherited record contradicts itself (three values for one arm, latency sign flip, a recommendation against its own `NOT MEASURED`) | carried under the inherited **CA‑07 / CA‑11 / CA‑13** ids in 8.6.3; deliberately *not* renumbered, because the artefacts concerned are frozen historical records this pass must not rewrite |
| N‑10 | First baseline reproduction was wrong by 84 Rank‑1 points; the audit's inference from it is superseded | **T7‑D20** (residual = Tier‑R proof only); correction recorded in §6.10.2 |
| N‑11 | Consumer tooling shipped a false explanation | **T7‑D27** |
| N‑12 | Nothing in §6.1‑§6.5 can be recomputed from this package alone (the three embedding `.npz` are outside it) | **packaging decision, not a defect** — 8.7 item 6, and listed with sizes in §12.2 |

**8.6.2 What the TR items changed, and what each one still leaves open.**

| TR | Delivered change (source of record) | Status in this package | Residual carried by |
|---|---|---|---|
| TR‑01 | `reid_model_type` derived from the resolved config instead of a literal launch dict (`reid_launch_defaults.py`, `robot_bringup.launch.py:28‑31, :38‑42`) | `IMPLEMENTED`, `TESTED` (Tier N/U, incl. `test_reid_launch_defaults.py`) | supersedes the override half of **B‑04**/R‑35; the *mock‑as‑fallback* remains by design (`reid_launch_defaults.py:85`, `extractor_factory.py:40`) and is now logged (`identity_node.py:23‑24`) → T7‑D11 context |
| TR‑02 | failed inference refuses at the node, attributes at the extractor, excludes at the evaluator (`identity_node.py` TR‑02, `transreid_extractor.py:196‑209`, `evaluate_offline.py:216`) | `IMPLEMENTED`, `TESTED` | the OSNet side is frozen and still mocks → **T7‑D08** |
| TR‑03 | `reid_camera_id` → `resolve_camera_label()` → `cam_label`, srv untouched (`identity_node.py:147`, `:126`) | `IMPLEMENTED`, `TESTED` (Tier N) | the *service* still carries no camera field (`IdentityVerify.srv` frozen) → **T7‑D01**'s reachable‑arm half (inherited **B‑06**) |
| TR‑04 | encoding‑aware adapter at the image boundary (`reid_input_adapter.py:20‑21, :42, :72‑78`, 31 tests) | `IMPLEMENTED`, `TESTED` (Tier U/N) | **T7‑D20** (no live‑message proof) |
| TR‑05 | CWD‑independent asset resolution (`identity_node.py:128‑144` calling `resolve_asset_path`, share lookup in `reid_input_adapter.py:129‑130`) | `IMPLEMENTED`, `TESTED` with a stubbed share dir (`test_transreid_input_adapter.py:164`, `test_identity_node_transreid_integration.py:379`) | supersedes **B‑05**/R‑37 in code; installed‑share resolution is `INTEGRATION PENDING` → §10 level 5 |
| TR‑06 | `load_state_dict()` result validated (missing ⊆ ∅, unexpected == tolerated set, parameter count) | `IMPLEMENTED`, `TESTED`; the heavy real‑checkpoint cases are 2 of the 15 skips | **T7‑D24** explains why they skip here |
| TR‑07 | named, CLI‑selected evaluation populations with exclusion metadata (`evaluate_offline.py`, 28 cases) | `IMPLEMENTED`, `TESTED`, `EXECUTED` (19,281 embeddings) | R‑22 still `FAIL — VERIFIED`: the `id_min`/`id_max` band remains a function default with no CLI override → 8.6.3 |
| TR‑08 | integrity‑gated statistics with Wilson/bootstrap/McNemar/Holm and declared pools | `IMPLEMENTED`, `TESTED` (68 cases), `EXECUTED` | its own `not_measured` string is **T7‑D27**; the τ *lookup* half is **T7‑D26** |
| TR‑09 | ONNX export, graph validation, Torch↔ONNX parity, byte‑identical re‑export | `IMPLEMENTED`, `TESTED` (33 cases), `EXECUTED` (2 reports) | **T7‑D16**, **T7‑D19**; the graphs themselves are outside the package (§12.2) |
| TR‑10/10b | `transreid_trt_extractor.py` (19,753 B) + `transreid_trt` registry entry, fail‑closed, no silent mock | `IMPLEMENTED`, `TESTED` (stubbed TRT) | supersedes **B‑02**; the engine does not exist → §10 level 8, `BLOCKED` |
| TR‑11 | build/benchmark harnesses that **refuse** on a non‑target host and label `--allow-non-target` output `NOT A DEPLOYMENT ARTEFACT` | `IMPLEMENTED`, refusal `MEASURED` (exit 1 + BLOCKED reports) | the older simulated tool (`profile_edge_latency.py`) is a **different** artefact and is absent here → **T7‑D18** |

**8.6.3 Inherited findings carried under their own ids (no T7 number assigned).** Renumbering these would
cut them off from the documents that quote them, so they stay where they are and their current state is
recorded here.

| Inherited id | Statement in the delivered record | State in this package now |
|---|---|---|
| **B‑01** / **CA‑16** layer 1 | `osnet_trt` inference is a hard‑coded zero vector (`osnet_trt_extractor.py:72‑75`) | **live, frozen** → `IMPLEMENTED` + `NOT MEASURED`; = **T7‑D08** |
| **B‑03** | wrong exception guard + placeholder fallback with random weights | **live, frozen** → **T7‑D06** (guard) and **T7‑D09** (placeholder), with CA‑16's "escapes `extract_embedding`" half corrected in §6.10.3 |
| **B‑04** | launch dict silently overrides YAML and ships a mock as the default | **override removed** by TR‑01 (`TESTED` Tier N); the mock remains the documented *fallback*, now WARN/NOTE‑logged |
| **B‑05** | asset paths relative, never resolved through the share directory | **fixed in code** by TR‑05; installed‑share path `INTEGRATION PENDING` (§10 level 5) |
| **B‑06** | camera/SIE unreachable through the frozen interface; headline accuracy belongs to an unreachable arm | **live by charter** — `IdentityVerify.srv` frozen, G.5 answered only for the per‑robot parameter; measured cost 35.17241379310345 pp (§6.10.2) and −4.632 pp / −7.801 pp (§6.4) → evidence for **T7‑D01** |
| **B‑07** | no `sensor_msgs/Image` encoding check at the node boundary | **superseded** by TR‑04; residual Tier‑R proof = **T7‑D20** (§6.10.2 records the supersession) |
| **B‑08** | `_zero_streak` read against `_zero_streak_count` → observability constant −1 | **still live** at `identity_node.py:618` → **T7‑D28** |
| **B‑09** | required acceptance statistics not implemented (McNemar, Holm, Wilson) | **closed** — `analyze_task7_statistics.py`, 68 cases, executed (§6.4); R‑26/R‑27's `FAIL — VERIFIED` rows are against the older tree |
| **B‑10** | false‑handoff / matched‑TAR FMR metrics absent | **half closed**: `FMR_at_TAR_0.90` and EER exist offline (`analyze_task7_statistics.py:612, :616`); cross‑robot handoff success is `NOT MEASURED` and not inferable (no trajectory data) → **T7‑D01** family |
| **B‑11** | recorded FNMR/FMR/EER unattributable — no pair counts, no population rule, no generating script (`verification_threshold_audit.json`, 1,542 B) | **unresolved** (the artefact is present, its provenance is not) → CA‑10; §6.7 restricts quoting it |
| **B‑12** | two incompatible populations in one package | **named populations closed** by TR‑07; the *band defaults* remain (R‑22 `FAIL — VERIFIED`) |
| **B‑13** | simulated Jetson numbers reported as device success; +18.0 ms fudge; writes into `timing_budgets.yaml` | → **T7‑D18**; the script itself is **absent** from this package, so the finding is `UNRESOLVED` here |
| **B‑18** | no version control anywhere; authorship cannot be attributed | **unchanged** — R‑41 `BLOCKED`, owner gate G.6 unobtainable; not a code defect |
| **B‑19** | projection loader asserts shape only — no dtype/finite/device assertion (`transreid_extractor.py`, R‑12 "NEXT") | **live, hardening only** — §11 action A‑9; today the loaded projection is `MEASURED` float32 `(768, 512)` |
| **CA‑01** | the SIE arm reproduces the record digit‑for‑digit (mAP Δ 2.3802e‑10) | `MEASURED` twice; **positive** finding, §6.9.1 |
| **CA‑02** | "different population" is false in three places, including this pass's own tooling | two places corrected in the documents; the third is **T7‑D27** |
| **CA‑03 / CA‑11** | one named arm, three values; the v1→v2 "correction" moved projection, camera and normalisation at once | **unresolved historical contradiction** → N‑9; §6.9.1‑§6.9.3 forbids difference‑quoting `delta_rank1_vs_baseline` |
| **CA‑04** | two mAP implementations differ by 6.0588e‑05 while Rank‑1 agrees exactly | documented float‑order/implementation spread; §6.9.3's equality check re‑measured 0.0 |
| **CA‑05 / CA‑06** | size figures reconstruct exactly; the "40.5×" label names the wrong definition (true: 40.5725 with heads, 46.448 parameters, 39.879 file) | `CALCULATED` and closed in §6.8.1; the run sheet's headline still needs the correction before any device run → §11 A‑10 |
| **CA‑07** | recorded latencies disagree **in sign** (0.711550× vs 2.084496×) | **unresolved offline** → N‑9; §10 level 8 requires the device harness |
| **CA‑08** | two cited latency medians (`phase8` 13.59, `phase9` 13.84) are not on disk | **referenced but missing** → §12.2 |
| **CA‑09 / CA‑10** | τ's provenance is machine‑checked; the recorded verification numbers are not | CA‑09 `MEASURED`; CA‑10 → **B‑11** |
| **CA‑12** | the recorded "memory footprint" ranks the **smaller** model higher | **unresolved** → N‑9, and a reason §13 refuses any deployment ranking |
| **CA‑13** | one file contains `jetson_orin_nano_status: NOT MEASURED` **and** "Deploy OSNet … Reject TransReID" | **unresolved**, never adopted; §6.9.8 and the may/may‑not‑quote list bar quoting the recommendation |
| **CA‑14** | `svd_variance_retained` fingerprints which projection file each recorded run used (0.9857138991355896 = `dc2b9072…` family, 0.9864118099212646 = deployed `9626bea5…`) | `MEASURED` → evidence for **T7‑D04** and CA‑03 |
| **CA‑15 / CA‑17** | training‑exposure evidence; `run_real_pipeline.py:33` points at an absent OneDrive tree | both `MEASURED`; CA‑17 is why R‑19/R‑20 rest on the artefacts, not the producer → **T7‑D04**/§12.2 |
| **D‑02, D‑04, D‑11, D‑12, D‑21, D‑22** | six artefacts the audit required and the tree did not contain | current state in §12.2 (some now exist: TR‑08's statistics report, TR‑09's ONNX; some still do not: `tests/fixtures/`, `profile_edge_latency.py`) |
| **R‑23** | "τ = 0.70 applied by the actual code path … declared, **injected**, used" | **downgraded in §9 to PARTIAL** — "injected" is unsupported; → **T7‑D26** |
| **P1** | OSNet frozen by §3B, and "worse than REV01 recorded" | `MEASURED` restatement in §6.10.3 → **T7‑D06/D08/D09** |

**8.6.4 Renumbering and collision ledger (this pass).**
* **T7‑D19** carried two different meanings mid‑pass (the ONNX parity camera‑label gap and the YAML
  root‑key finding). The parity gap kept the id because it had already declared itself a new item; the
  YAML finding was renumbered **T7‑D26**. `MEASURED`: the six §8.1 gaps (D05, D07, D10, D12, D13, D17) are
  unreused, and no external document uses any `T7‑D` id, so the renumbering cannot break a citation.
* **T7‑D27**'s code pointer was re‑verified from `:1140-1142` to `:1212-1214` in all four places.
* **T7‑D20**'s statement was weakened from "no encoding check" to "the check is not proven on a live
  message" after the adapter, node, offline loader and its 31 tests were re‑read (§6.10.2).
* **T7‑D28** is the same finding as inherited **B‑08**; both ids are kept, with the cross‑walk above as
  the join, and the entry says so rather than claiming novelty.

### 8.7 Items deliberately **not** registered as defects

Each was examined and found to be correct, intentional, or a pending decision rather than a defect. They
are listed so that a reviewer can see they were checked, not missed.

1. **The four unexpected `classifier_1..4.weight` keys** — JPM auxiliary heads the feature model does not
   own; `missing_keys == []`, 100.0 % coverage of 100,769,512 parameters (`RES/onnx/*→
   reference_extractor.checkpoint_coverage`, `REP/ES-EXP07-TRANSREID-PREFLIGHT-REPORT.md:66`). The real
   hazard (discarding `load_state_dict`'s return) was closed by TR‑06.
2. **Refusing an off‑contract resolution** — `runtime_checks.off_contract_128x64 = "refused by the graph:
   InvalidArgument"` in both arms is the intended behaviour of a 210‑token position embedding
   (`shape_assumptions.pos_embed`), not a bug.
3. **`providers: ['AzureExecutionProvider', 'CPUExecutionProvider']`** — ORT enumerating what is installed;
   `ORT_is_CPU_only: true` and `test_device: "onnxruntime CPUExecutionProvider (fp32)"` state what actually
   ran, so the list is not a claim.
4. **The parity thresholds themselves** (`--parity-tolerance 2e-3`, `--cosine-floor 0.9999`) — a documented
   judgement with its reason inline (`export_transreid_onnx.py:249‑253`), and the measured worst values
   (1.287e‑05 / 0.9999999999954788) sit two to five orders of magnitude inside them.
5. **`dt_id_ms: 120` in `timing_budgets.yaml:11`** — an on‑disk *candidate* for the identity‑stage share,
   explicitly not a decision (G.1 is OPEN), so it is tracked as an owner action (A‑2), not as a defect. The
   defect is only that the file's own "all nodes consume this" mandate (`:1‑2`) is unmet in the same way as
   τ — which is **T7‑D26**'s generalisation, recorded there rather than twice.
6. **N‑12 — the three embedding `.npz` outside the package** (55,193,927 + 37,094,828 + 37,095,723 B under
   `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory\`, per
   `RES/embeddings/task7_embeddings_manifest.json → files`) — a packaging decision taken when the handoff
   tree was built, restated with its consequence in §12.2 and §13. It is not a defect in Task 7's code, and
   this pass must not re‑generate embeddings to fix a packaging gap (charter §3A).
7. **`identity_coverage` reporting `null`** for a degenerate arm — correct post‑T7‑D21 behaviour (`NaN`
   would have been the defect).
8. **The 15 skips and 2 failures as counts** — they are the *consequences* of **T7‑D24** and **T7‑D25**,
   which are registered once, rather than a third entry per affected file.
9. **τ's YAML value equalling the code default** — a coincidence that keeps every current number correct;
   the registered defect is the missing channel (**T7‑D26**), not the agreement.
10. **The `reid_model` alias** — accepted deliberately, with documented precedence and provenance labels
    (`reid_launch_defaults.py:69‑74, :305‑330`); its registration (**T7‑D11**) is about the *external guide*
    that uses it, not about the code.

## 9. Requirements traceability

### 9.1 Source of record, verdict vocabulary and how this section was built

The requirement set is **not** authored by this report. Its source of record is the packaged matrix
`MAN/requirement-traceability-matrix.csv` — `MEASURED` in this pass: 15,600 B, **42 data rows**, header
columns `ID, Source document and section, Requirement, Relevant file/function, Evidence collected,
Verification method, Actual status, Gap / blocker, Next action`. Every row is reproduced below under its own
`R‑nn` id, with its requirement text taken from the matrix (and, where the matrix names a section of a
proposal, run sheet or charter, that name is kept). **No requirement has been added, merged or invented in
this pass**, and none of the 42 was dropped.

Two vocabularies are in play and are kept apart:

* **The matrix's `Actual status` column** uses its own verdict set — `PASS — VERIFIED`,
  `FAIL — VERIFIED`, `UNVERIFIED CLAIM`, `PARTIAL`, `BLOCKED`, `NOT TESTED`, `NOT APPLICABLE`. Those rows
  are `INHERITED` statements: each records what the authoring audit saw **in the tree as it stood when that
  audit ran**, and several are now stale because the TR‑01…TR‑11 work landed afterwards.
* **This report's verdict** uses the §5.9 vocabulary — `PASS` / `PARTIAL` / `FAIL` / `NOT TESTED` /
  `BLOCKED` — and is stated **only** where this pass could support it from a path inside this package or
  from an executed check recorded here. Where this pass could not re-verify a row, the verdict column says
  `INHERITED` and repeats the matrix's verdict rather than asserting a new one.

**The matrix's own verdict distribution, recomputed in this pass** from the CSV (`MEASURED` — a
`collections.Counter` over the `Actual status` column, not copied from any report):

| Matrix verdict | Rows | Ids |
|---|---|---|
| `FAIL — VERIFIED` | **16** | R‑01, R‑03, R‑05, R‑11, R‑16, R‑22, R‑26, R‑27, R‑28, R‑30, R‑31, R‑35, R‑37, R‑39, R‑40, R‑13 |
| `PASS — VERIFIED` | **15** | R‑02, R‑04, R‑06, R‑07, R‑08, R‑09, R‑10, R‑12, R‑15, R‑17, R‑18, R‑19, R‑20, R‑21, R‑23 |
| `UNVERIFIED CLAIM` | 3 | R‑14, R‑24, R‑25 |
| `PARTIAL` | 3 | R‑29, R‑33, R‑34 |
| `BLOCKED` | 2 | R‑32, R‑41 |
| `NOT TESTED` | 2 | R‑36, R‑38 |
| `NOT APPLICABLE` | 1 | R‑42 |

16 + 15 + 3 + 3 + 2 + 2 + 1 = **42** ✓ (`CALCULATED`, and it checks against the row count).

**A defect inside the matrix itself.** `MEASURED` by reading row R‑10 against row R‑32:

> **T7‑D30 — the requirement matrix's R‑10 row forwards its own camera‑interface decision to the wrong row.**
> *Finding*: R‑10 (`SIE / camera conditioning available to the deployed runtime`) ends with
> `Next action: Interface decision recorded separately at R-32`, but R‑32 is
> *"Engine size + device measurement protocol executed"* — a Jetson row with no camera content. The camera
> interface requirement is **R‑11** ("Camera conditioning through an approved frozen interface OR documented
> unavailable", `BLOCKED`/`FAIL — VERIFIED`, gap "G.5 unanswered"). *Evidence*: `MAN/requirement-traceability-matrix.csv`,
> rows R‑10, R‑11, R‑32, read in full in this pass. *Classification*: `MEASURED`. *Impact*: a reader who follows
> the pointer lands on a device row and may conclude the camera interface was escalated for hardware rather
> than left open as an owner decision — it misroutes the single decision (G.5) that bounds Task 7's headline
> accuracy claim. Nothing numeric changes. *Severity*: **Low** (this report's scale: a citation defect with no
> numeric consequence, the same class as T7‑D15 and T7‑D22). *Corrective action*: correct R‑10's `Next action`
> to name R‑11. *Acceptance criterion*: `grep` over the matrix resolves "R‑32" only in device context, and
> R‑10's row names R‑11. **This pass does not edit the matrix** (§2 forbids modifying any file but the two
> reports), so the correction is recorded here and queued as action **A‑11** in §11.

### 9.2 Group A — extractor output contract, assets and the projection (R‑02, R‑03, R‑04, R‑06, R‑07, R‑09, R‑10, R‑12, R‑13, R‑14, R‑15)

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑02 | Output embedding exactly 512‑D L2‑normalised float32 (proposal §3) | `SNAP/…/models/transreid_extractor.py` `extract_embedding`; contract pinned in `SNAP/…/interfaces/base_identity_extractor.py:19‑21` ("Output MUST be exactly (512,) float32 … ‖output‖₂ = 1.0") | `INHERITED`: 19,281 embeddings, dim 512, zero‑norm rows 0 (§6.1, §6.2). This pass re‑read the contract and the arrays' recorded metadata; it did **not** re‑generate embeddings (charter §3A) | `PASS — VERIFIED` | **PASS** (offline arms) | Contract holds for the offline arms; the *runtime* has never produced one → **T7‑D02**, §10 levels 5‑7 |
| R‑03 | Required candidate extractor = `transreid_trt_extractor.py` (proposal §3) | File **present** at `ASSET/transreid_trt_extractor.py` and `SNAP/…/models/transreid_trt_extractor.py`, 19,753 B; registered as `transreid_trt` (`extractor_factory.py:137‑172`) | `MEASURED` this pass: `find` returns both copies; docstring `:24‑27` states the engine was never built here. `find` for `*.plan`/`*.onnx` package‑wide → **0 hits** | `FAIL — VERIFIED` ("0 hits; engines/ empty") | **PARTIAL**, device half **BLOCKED** | Matrix row is stale — superseded by TR‑10 (8.6.2). Code `IMPLEMENTED` and fail‑closed; **no engine exists**, so the arm cannot run → §10 level 8, A‑1 |
| R‑04 | Real TransReID model loading and inference (proposal §3, run sheet §2 A1‑A2) | `transreid_extractor.py:75‑128`‑region loader (line numbers have drifted; the loader and its `load_state_dict` validation are re‑read in §4) | `INHERITED`: `transformer_120.pth` sha256 `84305529…`, 100,769,512 params logged at load. This pass **cannot recompute it** — the 414,724,314 B checkpoint is not in the package (`MEASURED`: `find` for `*.pth` returns only `osnet_x1_0_market1501.pth`) → **T7‑D24** | `PASS — VERIFIED` | **PARTIAL** | Matrix's gap ("`strict=False` return discarded `:128`") is closed by TR‑06 (`IMPLEMENTED`, `TESTED`); the real‑checkpoint assertions are 2 of the 15 skips → **NOT TESTED** here |
| R‑06 | Checkpoint provenance pinned by hash (run sheet §1.1 #12) | `RES/embeddings/task7_embeddings_manifest.json → assets`; `MAN` inventories | `INHERITED` for the TransReID digest (file absent → not recomputable in this package). `MEASURED` by contrast: the baseline weights **are** here — `TASK07/assets/edgesync_bringup/models/osnet/osnet_x1_0_market1501.pth` re‑hashed in this pass = 10,399,605 B, sha256 `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`, matching the §3.11 row | `PASS — VERIFIED` | **PASS as recorded / NOT RE‑VERIFIABLE here** | Restore the checkpoint or accept the manifest digests as the pin → §12.2, A‑1 |
| R‑07 | TransReID normalisation mean = std = [0.5, 0.5, 0.5] | `transreid_extractor.py:235‑236` (`mean = np.array([0.5,0.5,0.5], dtype=np.float32)`, same for `std`), docstring `:212` and `:250` | `MEASURED` this pass by re‑read. The matrix cites `:159‑163`, which is **stale** — the file grew and the constants moved; the values did not | `PASS — VERIFIED` | **PASS** | None for the value. The **resize** half is now a live finding → **T7‑D29** |
| R‑09 | Input geometry 256×128 NCHW batch 1 enforced | `transreid_extractor.py:211‑236 preprocess()`; `osnet_torch_extractor.py` `preprocess`; harness `preprocess_image:274‑285` | `MEASURED` this pass: both `cv2.INTER_LINEAR` and the `np.linspace` index‑sampling fallback emit 1×3×256×128 float32 | `PASS — VERIFIED` | **PASS with caveat** | Geometry holds on both branches; **the branch taken is host‑dependent** → **T7‑D29**, A‑7 |
| R‑10 | SIE / camera conditioning available to the deployed runtime | `transreid_extractor.py:170‑171`‑region SIE path; `identity_node.py:126 declare_parameter('reid_camera_id', …)`, `:147 resolve_camera_label()` | `INHERITED` measurement: absence costs −4.632 pp Rank‑1 / −7.801 pp mAP; `MEASURED` this pass: the **base API has no camera parameter** (`base_identity_extractor.py:27 extract_embedding(self, image_crop)`), and `IdentityVerify.srv` carries **no camera field** (full text quoted in §4.5) | `PASS — VERIFIED` | **PARTIAL** | True at node‑parameter level, false at service level; headline accuracy belongs to the arm the service cannot reach → **B‑06**, **T7‑D01**, and the mis‑pointer **T7‑D30** |
| R‑12 | Centred truncated SVD 768→512 fitted on the training partition only | `SNAP/…/scripts/fit_svd_projection.py`; loader asserts at `transreid_extractor.py:102, :108‑109` (`mean` cast to float32, shape asserted `(768,)`) | `INHERITED`: fit ids 0001‑0650 vs eval 0752‑1501. `MEASURED` this pass: the loader asserts **shape only** — no dtype/finite/device guard, matching the row's own gap | `PASS — VERIFIED` | **PASS** for the fit scope | Hardening gap still open → **B‑19**, action A‑9 |
| R‑13 | Projection identified by hash, not filename (run sheet §1.1 #11, structural note C.1) | `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:47` and `CFG/exp7_transreid.yaml:57` — both select `…_real_v2.npz` **by name** | `MEASURED`: **6 projection names → 4 distinct sha256** (`RES/provenance/task7_provenance_measurements.json → projections`: `dc2b9072…` held by both `…_768_to_512.npz` and `…_real.npz`; `9626bea5…` held by both `…_real_v2.npz` and `…_real_v3_reproduce.npz`; `628b1dc6…` and `c9bca759…` unique), and `find` in this package counts **16 copies** of those names under `models/transreid/` **and** `models/projections/`, each mirrored once by `TASK07/assets/` and once by `SNAP/`. `_relationships` records `_v2` ≡ `_v3_reproduce` byte‑identical (Δ 0.0) and `_norm05` ≠ `_v2` (Δ 0.42987602949142456) | `FAIL — VERIFIED` ("8 files → 4 sha256") | **FAIL** (live) | Name‑based selection over a duplicated namespace → **T7‑D04**. The two counts differ (this package: 6 names / 16 copies / 4 hashes; matrix: 8 files / 4 hashes) but agree on the defect → §9.8 |
| R‑14 | Projection fit reproducible and metadata recorded | `fit_svd_projection.py` present; the six `.npz` carry `svd_variance_retained` metadata (`MEASURED`: 0.9864118099212646 = deployed `9626bea5…` family, 0.9857138991355896 = `dc2b9072…` family) | The row itself records that the fit was **not re‑executed** ("NOT re‑run because it writes into the tree"). This pass also does not re‑run it (§2 forbids modifying any existing file; charter §3A) | `UNVERIFIED CLAIM` | **NOT TESTED** (reproducibility of the fit) | Reproducibility remains an inherited claim → §10 level 3, A‑9/A‑6 |
| R‑15 | Production path cannot accidentally regenerate a projection (charter §296) | `identity_node.py:79‑144` (parameters → `resolve_asset_path` → `create_extractor`); `extractor_factory.py:137‑172` | `MEASURED` this pass by trace: the factory only **loads** a file; `fit_svd_projection.py` is under `scripts/` and is not imported by `edgesync_identity` (grep of imports found no producer/consumer link) | `PASS — VERIFIED` | **PASS** | None. This is one of the strongest rows in the matrix and it survives re‑inspection |

### 9.3 Group B — the frozen OSNet comparator (R‑05, R‑08, R‑16, R‑17)

These four rows are the reason Task 7 has no baseline arm. Charter §3B freezes OSNet, so every finding here
is recorded and **not repaired** — see §6.10.3 and T7‑D06/D08/D09.

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑05 | Implementation must not falsely claim TensorRT while using PyTorch (proposal §3, P‑claim) | Two TRT classes exist: `ASSET/osnet_trt_extractor.py` and `ASSET/transreid_trt_extractor.py` (19,753 B) | `MEASURED` this pass: `osnet_trt_extractor.py:71-74 _execute_trt()` is `# Simulated or actual CUDA buffer exchange` / `return np.zeros(512, dtype=np.float32)` — the claim‑without‑inference **is** the shipped OSNet TRT path (T7‑D08). By contrast `transreid_trt_extractor.py:10‑15` states "NO silent mock … raises at construction" and `:24‑27` that no engine was built here | `FAIL — VERIFIED` | **FAIL** (frozen) | The requirement is violated **only** by the frozen comparator. The TransReID side now complies by construction → TR‑10 (8.6.2). Nothing on this host can fix it without breaching §3B → A‑2, A‑1 |
| R‑08 | OSNet normalisation = ImageNet mean/std (run sheet §1.1 #7) | `osnet_torch_extractor.py:141-142` and `osnet_trt_extractor.py:64-65` (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`, `/255.0` at `:63`/`:140`) | `INHERITED` diagnostic: RGB 84.83 vs BGR 49.66 Rank‑1 on a stratified mini‑population (§6.10.2, N‑10). `MEASURED` this pass: the matrix's own pointer `osnet_torch_extractor.py:141-143` is still accurate; `run_real_pipeline.py:44-45` holds the same constants | `PASS — VERIFIED` | **PASS** | The row's gap ("No encoding check at the ROS boundary", B‑07) is closed **in code** by TR‑04 (`reid_input_adapter.py`, 31 tests, Tier U) → residual is Tier‑R only = **T7‑D20** |
| R‑16 | OSNet baseline correctness in the shipped extractor (proposal §3, charter §259‑264) | `osnet_torch_extractor.py:62-72` (torchreid route) → `:74 except ImportError:` → `:77-83` placeholder route with `_build_osnet_standalone()` at `:93` | `INHERITED`: `import torchreid` raises `RuntimeError: operator torchvision::nms does not exist`, and `:74` catches only `ImportError`. `MEASURED` this pass by re‑read: the guard really is `except ImportError` (`:74`) and both routes end in `load_state_dict(..., strict=False)` (`:72`, `:83`). **Additionally**: this violates the frozen wire contract in the project's own interface — `base_identity_extractor.py:42-43` states implementations "MUST NOT raise exceptions to callers; errors are logged and result in the sentinel `zeros(512)`" | `FAIL — VERIFIED` | **FAIL** (live, frozen) | B‑03 confirmed twice over. The un‑caught `RuntimeError` breaks the base API's no‑raise rule, so the baseline cannot even reach its own placeholder fallback → T7‑D06, A‑2 |
| R‑17 | OSNet baseline architecture available and weight‑compatible | `ASSET/osnet_backbone.py` — vendored `OSNet` (`:287`), `OSBlock` (`:228`), factory `osnet_x1_0(num_classes=1000, pretrained=True, …)` (`:527`), weight merge `init_pretrained_weights` (`:446`) → `model.load_state_dict(model_dict)` (`:503`) | `INHERITED`: 0 missing / 0 unexpected keys, cosine 1.00000003 vs installed torchreid (`identical=true`). `MEASURED` this pass: the module **is** imported, but only by tests (`TST/edgesync_identity/tests/test_experiment7_fixes.py:20`, `test_experiment7_forensic_fixes.py:23`) and two audit scripts (`REP/evidence/diag_backbone_equiv.py:29`, `REP/preflight_probe.py:74`) — **no production file**; and all three test call sites pass `pretrained=False` (`test_experiment7_fixes.py:32, :39`, `test_experiment7_forensic_fixes.py:150`) | `PASS — VERIFIED`, gap "Module is not imported by any production file" | **PASS** for the module, **FAIL** for integration | The matrix's next action — "Repoint to `osnet_backbone.py` (fixes R‑16)" — is correct **and unsafe as written**; see **T7‑D31** below |

**T7‑D31 — the vendored baseline module performs an un‑pinned third‑party network fetch of model weights
under its own default arguments, and the requirement matrix's recommended fix for R‑16 arms that path.**
*Finding*: `SNAP/…/models/osnet_backbone.py:527` defines `osnet_x1_0(num_classes=1000, pretrained=True, …)`;
when `pretrained` is true it calls `init_pretrained_weights(model, key='osnet_x1_0')` (`:537-538`), which
`import gdown` (`:453`) and runs `gdown.download(pretrained_urls[key], cached_file, quiet=False)` (`:485`)
against one of five hard‑coded Google Drive links (`:16-26`, e.g.
`https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY`). The download is un‑pinned: nothing in
the module checks a digest of the fetched object, and `:505-508` only warns ("**ignored and continue**")
when `len(matched_layers) == 0`.
*Evidence*: the line numbers above, all re‑read in this pass; `grep` for `osnet_backbone` across the package
(importers = 2 test modules + 2 audit scripts, **no production module**); `grep` for `pretrained=False` at
the three test call sites.
*Classification*: `MEASURED` (source inspection). **This pass did not execute the download path and does not
claim to** — the branch is unreachable from the shipped runtime because the module is not wired in, so the
fetch is `NOT TESTED` and, in the delivered graph, `NOT REACHED`.
*Impact*: two‑fold. (1) Dormant today; (2) the matrix's own corrective action for R‑16/R‑17 ("Repoint to
`osnet_backbone.py`", "Wire it in") is the step that would put a default‑`pretrained=True` constructor on a
robot's cold start — a non‑hermetic build, an unaudited weight source, and a hard failure (or a silent
"ignored and continue" partial load) exactly where §3B says the baseline's identity must not change. A
reviewer who follows the CSV without reading the module inherits that hazard.
*Severity*: **Moderate** (this report's scale — a real gap that bounds a claim; it corrupts no delivered
number, because no number was produced through this path).
*Corrective action*: when R‑17's wiring is authorised, construct with `osnet_x1_0(num_classes=0,
pretrained=False)` — the same form the tests already use — and load the **packaged** weights
`TASK07/assets/edgesync_bringup/models/osnet/osnet_x1_0_market1501.pth` (10,399,605 B, sha256
`2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`, re‑hashed in this pass) explicitly;
never rely on `init_pretrained_weights`. Do not make this change in this pass: §2 forbids modifying any file
but the two reports, and §3B freezes the comparator.
*Acceptance criterion*: the module is imported by the runtime **only** with `pretrained=False`, a test
asserts the loaded digest equals `2809d322…`, and a run with networking disabled completes — i.e. no host
`gdown` call appears in any log. → §11 action **A‑12**.

### 9.4 Group C — metric validity, populations and the threshold (R‑18 … R‑25)

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑18 | Rank‑1/5/10/mAP computed on the canonical protocol (proposal §2, run sheet §4.3) | `SNAP/…/scripts/run_real_pipeline.py:425 compute_cmc_map(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50)` — pointer re‑verified, still `:425` | `INHERITED`: same‑pid **and** same‑cam removal, junk `−1` dropped, distractors kept, `num_valid` denominator; exercised in three reproductions. `MEASURED` this pass: `RES/…/gate3_real_results*.json` and `REP` records all report `n_query 3,368` / `n_gallery 15,913` (§6.2) | `PASS — VERIFIED` | **PASS** | Row's gap ("skipped‑query count not recorded") is closed by TR‑07's manifest (`evaluate_offline.py:179` writes the population record) → §9.8 |
| R‑19 | Reproduced accuracy matches recorded values | `RES/…/gate3_real_results_v3_reproduce.json`, `repro_transreid_both_arms.json`, `repro_osnet_full_rgb.json` | `MEASURED` twice, including by this pass's re‑read: SIE arm 93.37885985748218 / 85.12520406484964 (and the reconciled file's 85.12520406508766 — CA‑04's float‑order spread) and OSNet 94.32897862232778 / 83.5692348350484; CA‑01's Δ for the SIE arm is 2.3802e‑10 | `PASS — VERIFIED` | **PASS** | The row's gap ("laptop GPU only — not an edge result") is exactly right and is why §10 stops at level 4 for accuracy |
| R‑20 | Dataset identity and counts recorded from executed code (proposal §2, charter §393‑394) | Counts recorded in `REP/TASK7_DATA_PROVENANCE.md`; consumer `evaluate_offline.py` | `INHERITED`. **`MEASURED` negative this pass**: `find` over the package for `*market*` returns **nothing** — Market‑1501 (68,036 files, 275 MB per the row) is **not** in this package, so none of these counts can be recomputed here → §12.2 | `PASS — VERIFIED` | **INHERITED / NOT RE‑VERIFIABLE here** | Restore the dataset or treat every §6.1‑§6.5 figure as inherited → N‑12, A‑6 |
| R‑21 | Train/eval identity overlap and projection leakage (charter §378‑380) | `evaluate_offline.py` population rules (`:74-99`), `fit_svd_projection.py` fit range | `INHERITED`: train 0001‑0751 ∩ test 0752‑1501 = ∅; projection fit restricted to 0001‑0650. Not recomputable here (same absent dataset). Independently supported this pass by `RES/provenance/...` recording the fit's id range metadata | `PASS — VERIFIED` | **INHERITED** | No leakage found in any record this pass could open; the check cannot be re‑run without the dataset → A‑6 |
| R‑22 | Canonical vs bundled split selection is **configurable** (proposal §5 P5, charter §380) | `evaluate_offline.py:74-99 POPULATION_RULES` (`'directory'`, `'id_band_752_1501'`), `:99 DEFAULT_POPULATION = 'directory'`, `:473 --population` with `choices=sorted(POPULATION_RULES)`, `:480 --id-min` (default 752), `:482 --id-max` (default 1501), threaded at `:498` | `MEASURED` this pass by re‑read — and the row's stated reason is **no longer true**: the CLI overrides exist. What remains is the *warning* path (`:171-173`: changing the band away from 752‑1501 makes the manifest say the recorded numbers "do not describe this population") and the `--help` text's own verdict that the band is "measured invalid as a split guard, kept only to reproduce older numbers" | `FAIL — VERIFIED` ("no CLI override") | **PARTIAL** | Configurability: closed by TR‑07. Population *canonicality*: settled the other way — `directory` is the default and the band is quarantined. The matrix's second half ("`run_real_pipeline.py:489-498` uses an unfiltered population") is still true of that **older** script, which remains on disk → §9.8, A‑6 |
| R‑23 | τ = 0.70 applied by the actual code path, not only written in YAML (proposal §2) | Three τ sites, `MEASURED` this pass: `SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:7 reid_similarity_threshold: 0.70`; `identity_node.py:79 declare_parameter('similarity_threshold', 0.70)` read at `:110` and handed to the verifier at `:172`; `evaluate_offline.py:471 --threshold` default 0.70. Boundary semantics confirmed in `identity_verifier.py` (accept iff `sim ≥ τ`) | `TESTED` Tier U (the 0.70 boundary cases in `TST/test_gate1_scaffolding.py:160-227`) + `MEASURED` six‑link propagation chain (§8.4) | `PASS — VERIFIED`, evidence says "declared, **injected**, used" | **PARTIAL** | **Downgraded.** "Injected" is unsupported: no launch file or parameter channel delivers `similarity_threshold` to the node (`robot_bringup.launch.py:127-137` omits it), so today's 0.70 is the **code default**, not the YAML value — the two coincide numerically, which is why no number changes. → **T7‑D26**, §11 A‑4/A‑13 |
| R‑24 | FNMR/FMR/EER with a declared genuine‑pair population (charter §403‑408) | `RES/results/experiment7/verification_threshold_audit.json` — **1,542 B, present**, 4 rows | `MEASURED` this pass by re‑read: rows are "OSNet x1.0 Native 512‑D" (genuine_mean 0.7809563875198364, FNMR **18.703290940081292 %**, FMR 0.114 %, EER 2.051920807657008 %, **eer_tau 0.555**), "TransReID Corrected Native 768‑D (With Cam)" (FNMR 42.84122197456405 %, FMR 0.014 %, EER 1.7306935885669332 %, eer_tau 0.387), "…Centered SVD 512‑D (With Cam)" (genuine_mean 0.682079017162323, impostor_mean 0.007322206627577543, FNMR **48.6626458633801 %**, FMR 0.014 %, EER 1.678248328307329 %, **eer_tau 0.33**), "…Uncentered SVD 512‑D (With Cam)". Every row carries `tau: 0.7`. Still **no pair counts, no population rule, no generating script** → B‑11 holds
exactly, and this pass found it more strongly: `grep` over every packaged `.py` for the string
`verification_threshold_audit` returns **0 files**, so no script in this package even names the artefact. | `UNVERIFIED CLAIM` | **NOT TESTED** (unattributable) | Quotable only with its `INHERITED` label (§6.7). This pass's own pair‑level numbers in §6.3 (genuine 49,649 / impostor 53,532,220) are the attributable replacement → A‑4 |
| R‑25 | Two model variants evaluated under comparable pair definitions (charter §405) | Same file | `MEASURED` this pass: the file has **four** rows, not two, spanning native‑768, centred‑512 and uncentred‑512 variants, all "(With Cam)" — i.e. **the recorded verification audit contains no camera‑agnostic row at all**, while the arm the runtime can reach is the agnostic one. Comparability rule remains unstated | `UNVERIFIED CLAIM` | **NOT TESTED** | Same blocker as R‑24, plus a new consequence: the recorded verification evidence describes an arm the deployed service cannot select → B‑06, **T7‑D01** |

### 9.5 Group D — acceptance statistics, latency, memory and the device (R‑26 … R‑32)

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑26 | Paired exact McNemar on per‑query correctness with Holm correction, α = 0.05 (proposal measurement plan) | `SNAP/…/scripts/analyze_task7_statistics.py:124 exact_mcnemar(b, c)` and `:139 holm_corrections(pairs)` — **both now exist**; the matrix searched the whole repository and found none | `IMPLEMENTED` + `TESTED` (68 cases in `TST/test_analyze_task7_statistics.py`) + `EXECUTED`: the McNemar family and Holm‑adjusted p‑values are the numbers in §6.4 (H3 p = 0.572845968999707 on 154 discordant queries). This pass did **not** re‑run the analyzer | `FAIL — VERIFIED` (B‑09) | **PASS** as an offline tool | The row is **stale** — closed by TR‑08. What remains is not the code but the *arm*: the family cannot be run against a baseline that does not execute (R‑16) → §9.8, A‑2 |
| R‑27 | Wilson 95 % intervals on false‑handoff rate and FNMR (proposal measurement plan) | `analyze_task7_statistics.py:109 wilson_interval(successes, trials, z = 1.959963984540054)` (`z` is the 97.5 % quantile ⇒ two‑sided 95 %) | `IMPLEMENTED` + `TESTED` + `EXECUTED` for the **verification** quantities (FNMR intervals in §6.3). **False‑handoff rate**: `NOT MEASURED` — it is a cross‑robot quantity and there is no trajectory or handoff data (§6.8, B‑10) | `FAIL — VERIFIED` (B‑09) | **PARTIAL** | Half closed (FNMR Wilson, offline); the other half is unreachable offline → §10 levels 9‑10, A‑8 |
| R‑28 | Latency harness: 50 warm‑up / 200 timed / 3 trials / batch 1 / cuda events / idle ≤ 5 % (charter §418‑442, run sheet §1.1) | `run_real_pipeline.py:595 N_WARMUP, N_TIMED, N_TRIALS = 50, 200, 3`, echoed into its own output at `:658-660` and `:758-760`. A **second, device‑grade** harness exists: `SNAP/…/scripts/benchmark_transreid_jetson.py` (384 lines, 19,598 B) | `MEASURED` this pass: `run_real_pipeline.py:566` times `np.random.randint(0, 255, (256, 128, 3))` **noise** and `:570` pins `dummy_cam = torch.tensor([0])` — the row's B‑14 criticism is exactly right for *this* script, and `:563` self‑labels it "STEP 4: Latency Measurement (Laptop -- NOT Jetson Orin Nano)". By contrast `benchmark_transreid_jetson.py:198-199` **requires** `--market1501` ("Real crops are required; without them this is not a benchmark"), refuses at `:265` if no crop decodes, loads the **production** `TransReIdTrtExtractor` (`:11-12`), and samples `tegrastats` (`:112-126`) | `FAIL — VERIFIED` (B‑14) | **PARTIAL** | The named pre‑registered harness is still a noise‑input laptop tool; the real‑crop harness exists but **has never been run** (G.7 `DENIED`) and mismatches the pre‑registered warm‑up → **T7‑D32**, A‑1, A‑10 |
| R‑29 | Trial‑count consistency across the three governing documents (run sheet §1.1 #1‑#3) | Proposal `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:145` ("50 warm‑up calls over 200 timed calls, 3 independent trials, batch 1, `torch.cuda.Event`… v01 said 1,000 calls here while every artifact on disk used 200"), run sheet §2, report REV02 §0 `:20` | `MEASURED` this pass: the three **documents** agree at 50/200/3 and the **code** agrees at `run_real_pipeline.py:595`. New finding: the device harness defaults to `--warmup 20` (`benchmark_transreid_jetson.py:202`) and its own usage example passes `--warmup 20` (`:32`) | `PARTIAL` | **PARTIAL** | Document‑to‑document consistency is now `MEASURED`; document‑to‑*device‑harness* consistency is not → **T7‑D32** |
| R‑30 | Latency results are internally consistent (`results/experiment7/*.json`) | `RES/results/experiment7/gate3_real_results.json` and `benchmark_latency_memory.json` (1,125 B, present) | `MEASURED` this pass by re‑read of both files: gate3 13.431399998808047 vs 18.876250000175787; benchmark `full_pipeline_latency_median_ms` **58.718050000607036** (TransReID) vs **28.16894999705255** (OSNet), `backbone_latency_median_ms` 56.770099996356294 vs 24.78759999939939. `CALCULATED`: 58.718050000607036 ÷ 28.16894999705255 = **2.084495517466962** → the recorded "2.084496×" figure **is** this file's full‑pipeline ratio, and 13.431399998808047 ÷ 18.876250000175787 = **0.711549…** → the opposite‑signed "0.711550×" is gate3's. Both numbers in the CA‑07 contradiction are now attributed to a file in this package (§6.9.4) | `FAIL — VERIFIED` (B‑14) | **FAIL** | No protocol fields in `benchmark_latency_memory.json`; its generator is not on disk. Neither value may be quoted as an edge number → §6.9.4, A‑10 |
| R‑31 | Memory footprint measured as a model‑resident Δ on unified memory (charter §438‑442, run sheet §4.2) | `benchmark_latency_memory.json` `peak_runtime_gpu_memory_mb` 410.7265625 vs 413.26025390625. Method on record: proposal `:149` mandates `tegrastats` MEM + `VmRSS` + `MemAvailable` + `torch.cuda.max_memory_allocated()`, median of 3 trials | `MEASURED` this pass: `CALCULATED` 410.7265625 ÷ 413.26025390625 = **0.99386901744772** — the *smaller* model reports the *larger* footprint, so the recorded pair is context‑dominated (CA‑12). The row's claim "no VmRSS/tegrastats code in repo" is **now false**: `benchmark_transreid_jetson.py:103` parses `VmRSS:`, `:95` parses `MemAvailable`, `:112-126` runs `tegrastats` and keeps raw lines, `:308` reads `nvpmodel -q` after the run | `FAIL — VERIFIED` | **FAIL** for the recorded numbers; method **IMPLEMENTED**, `NOT MEASURED` on device | The method exists and has never produced a number → §10 level 8, A‑1/A‑3 |
| R‑32 | Engine size + device measurement protocol executed (run sheet §4.2, charter §451‑453) | Requires a Jetson. Harness `benchmark_transreid_jetson.py:233-239` refuses off‑target | `BLOCKED`, and `MEASURED` as blocked: `RES/trt_blocked/probe_build_report.json` environment block is all‑negative; `RES/trt_blocked/{bench_run2.log,build_run2.log}` are refusal artefacts, not results. Gate G.7 `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:20` = **DENIED FOR THIS SESSION** ("Not reachable now — I'll connect it myself later") | `BLOCKED` | **BLOCKED** | Nothing on this host can change it. Any deployment claim is out of scope until G.7 is granted → §10 levels 8‑10, A‑1 |

**T7‑D32 — the device benchmark harness's default warm‑up contradicts the pre‑registered latency protocol,
and its own usage example encodes the contradiction.**
*Finding*: the latency protocol is pre‑registered as **50 warm‑up / 200 timed / 3 trials** in
`REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:145`, in `REP/JETSON-RUN-SHEET-PREP-01-REV01.md` §2 and in
`REP/ES-EXP07-TASK7-REPORT-01-REV02.md:20` ("pre‑registered, no longer `<fill>`"), and the laptop harness
honours it (`run_real_pipeline.py:595`). The **device** harness does not:
`SNAP/…/scripts/benchmark_transreid_jetson.py:202` sets `--warmup` **default 20**, and the copy‑paste command
in its own docstring (`:32`) is `--iterations 200 --repeats 3 --warmup 20`.
*Evidence*: the line numbers above, each re‑read in this pass; `--iterations` (`:200`) and `--repeats` (`:201`)
do match the pre‑registration, so the deviation is confined to warm‑up.
*Classification*: `MEASURED` (source inspection). The harness has **never been executed on any target** —
`NOT TESTED`, `BLOCKED` at G.7 — so this is a pre‑flight finding about a future measurement, not a
re‑measurement of anything.
*Impact*: an operator who follows the docstring produces a device run with 20 warm‑up inferences while the
governing documents promise 50. H7b — the latency axis that owner gate **G.1** exists to decide — would then
be answered under an undeclared protocol deviation, and the pre‑registration's stated purpose ("fixed **before**
any data is collected") is defeated by the default rather than by the operator. One mitigation, and it is the
reason this is not rated High: the harness writes the actual value into its own report at `:220`
(`'protocol': {'warmup_inferences': args.warmup, …}`), so the deviation is *recorded* and detectable after the
run — unlike **T7‑D26**, whose failure mode is indistinguishable in the log.
*Severity*: **Moderate** (this report's scale — it bounds a future claim; it corrupts no delivered number,
because no device number exists).
*Corrective action*: make the harness default match the pre‑registration (`--warmup 50`) **or** amend the
pre‑registration through the owner, and change the docstring example to the value the documents promise;
have the harness refuse or warn when `args.warmup` differs from the pre‑registered constant. Do it in the
same change that runs A‑1, so the first device artefact is conforming.
*Acceptance criterion*: the Jetson report JSON contains `warmup_inferences = 50` (or the documented new
value) **and** the pre‑registration table in the results report names the same three numbers as the harness
defaults. → §11 action **A‑1**, sub‑item (a).

### 9.6 Group E — scope, naming, model selection and the camera interface (R‑01, R‑11, R‑33, R‑34, R‑35)

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑01 | Experiment 7 scope = isolated OSNet vs TransReID ablation **with device latency/memory measurement** (proposal §2‑§3) | Scope text `REP/…PROPOSAL-01-REV02.md`; arms defined in run sheet §2 | `MEASURED`: 0 of 3 H7 axes decided on device; accuracy axis has a candidate arm but **no runnable baseline** (R‑16/R‑05); the report's own deviation notice `REP/ES-EXP07-TASK7-REPORT-01-REV02.md:7` states the pre‑registered paired family was **replaced** by a within‑TransReID family | `FAIL — VERIFIED` | **FAIL** (scope not met) | Two independent blockers: frozen broken comparator (§3B) and `DENIED` device (G.7) → T7‑D01, A‑1 + A‑2 |
| R‑11 | Camera conditioning through an approved frozen interface **or** documented unavailable (proposal §3, charter §298‑309) | `SNAP/…/edgesync_msgs/srv/IdentityVerify.srv` — full field list re‑read this pass: `transaction_id`, `candidate_crop`, `reference_feature_vector`, `candidate_feature_vector` → `match`, `similarity`, `message`, `match_confirmed`, `similarity_score`. **No camera field**, and none was added (`edgesync_msgs` untouched) | `MEASURED` + `INHERITED`: G.5 is now **answered** — `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:18` records "DECIDED by the owner (logged as PF‑10c): *add a per‑robot `reid_camera_id` ROS parameter*; implemented, `edgesync_msgs` untouched". Implemented at `identity_node.py:126`, consumed at `:147 resolve_camera_label()`, `TESTED` at Tier N. The service still cannot carry a per‑query camera, so the SIE arm is reachable only through a **static per‑robot parameter** | `FAIL — VERIFIED`, gap "G.5 unanswered" | **PARTIAL** | Row's blocker is obsolete (G.5 answered) but its consequence is not: the headline accuracy belongs to an arm the *service* cannot reach (B‑06). Cost `MEASURED`: −4.632 / −7.801 pp (§6.4) and 35.17241379310345 pp in the BGR diagnostic (§6.10.2) → T7‑D01, A‑2 |
| R‑33 | Config filename `exp7_transreid.yaml` vs `ablation_task7_{osnet,transreid}.yaml` (team guide vs proposal §3) | `CFG/exp7_transreid.yaml` (4,183 B) **exists** and is the shipped overlay; `SNAP/…/launch/exp13_bringup.launch.py:104` joins `config/exp7_transreid.yaml` and `:194` documents it as "the Exp‑7 overlay"; `reid_launch_defaults.py:239, :252-253` implement the two accepted path forms; `TST/edgesync_bringup/tests/test_exp7_config_overlay.py:22` loads it | `MEASURED` this pass: the row's claim "Neither filename exists" is **half false** — `exp7_transreid.yaml` exists in two copies. `find` across the package for `ablation_task7*` and `task7_identity_ablation.launch.py` → **0 hits**; `tools/ablation/task7/run_task7_benchmark.sh` → **0 hits**; the team guide is still **not in this package**, so the naming mandate cannot be read at source | `PARTIAL` | **PARTIAL**, naming conflict **UNRESOLVED** | One of two filenames delivered; the *other* convention remains untestable because its document is absent → §12.2, T7‑D03, A‑6 |
| R‑34 | Are cross‑robot handoff metrics required by Experiment 7 or by Task 13? (report §4.4 vs proposal §2) | Report §4.4 tables vs proposal §3 scope; Task 13 side lives in `TASK13/` and `SHARED_DEPENDENCIES/` | `INHERITED` document conflict, `UNRESOLVED` by this pass. `MEASURED` support: `benchmark_transreid_jetson.py:368` explicitly records `cross_robot_handoff = 'NOT MEASURED and NOT inferable from latency or Market‑1501 retrieval'` — i.e. the shipped device harness itself refuses to imply a handoff result | `PARTIAL` | **UNRESOLVED** (needs an owner reply) | Scope text, not code. Until answered, §13 must not claim any handoff property → A‑2 |
| R‑35 | Model selectable at runtime via `reid_model_type` through the **real launch path** (proposal §3, charter §320‑325) | `SNAP/…/launch/robot_bringup.launch.py:28-43` — TR‑01 comment records the old literals (`'osnet_mock'`, `''`) that "silently overwrote the model selection… so the stack could never launch the TransReID arm", now replaced by `select_experiment_config` (`:38`) → `resolve_reid_config` (`:39`) → injected dictionary `:127-137`; `reid_launch_defaults.py:61-66 REID_LAUNCH_KEYS`, `:74 RESOLVED_KEYS`, `:85 HARDCODED_FALLBACKS['reid_model_type'] = 'osnet_mock'`; `extractor_factory.py:40 DEFAULT_BACKBONE_ID = BACKBONE_ID_OSNET_MOCK` | `TESTED` Tier N/U (`TST/edgesync_bringup/tests/test_reid_launch_defaults.py`, `test_exp7_config_overlay.py`; 57/57 bringup and 25/25 launch‑description checks recorded in `MAN/ES-EXP07-TASK7-INTEGRATION-HANDOFF-CHECKLIST.md` row 2). `NOT TESTED` at Tier R — `robot_bringup.launch.py` has never been executed by a real launch system here (§10 level 5) | `FAIL — VERIFIED` (B‑04: "launch dict silently overrides YAML") | **PARTIAL** | Override half **closed** by TR‑01 (`MEASURED` in code + `TESTED` Tier N). Mock‑as‑default half **remains by design** (`:85`, `:40`) and is logged (`identity_node.py:23-24`) → T7‑D11 context, A‑6/A‑13 |

### 9.7 Group F — dependencies, build, version control, misleading indicators and the report itself (R‑36 … R‑42)

| ID | Requirement (success criterion as stated in the matrix) | Implementation evidence in this package | Validation evidence, classified | Matrix verdict | This pass | Gap → next action |
|---|---|---|---|---|---|---|
| R‑36 | Invalid parameter values fail clearly (charter §326‑327) | `extractor_factory.py:137-172 BACKBONE_REGISTRY` + unknown‑key `ValueError`; `:280` "It will NOT fall back to '{DEFAULT_BACKBONE_ID}'"; `identity_node.py:128-144` `TR-05` asset resolution logging (`exists=`, `bytes=`, `attempted=`) | `TESTED` Tier U/N (factory and node suites re‑executed 2026‑10‑22, §7). **`NOT TESTED` at the level the row asks for**: no executed ROS 2 launch‑level negative test exists (§10 level 6) | `NOT TESTED` | **NOT TESTED** (launch level) | Add a launch‑level negative test on a ROS 2 host → A‑6 |
| R‑37 | Asset paths resolve from the **installed** ROS 2 package (charter §328, run sheet G‑b) | `identity_node.py:128-144` calls `resolve_asset_path`; `reid_input_adapter.py:129-130` uses `get_package_share_directory`; `edgesync_bringup/setup.py:8-13` records "FIX B16 (2026‑10‑16): models/ was never installed, so every `reid_*_path` pointed at a file absent from the install space", with `model_data_files()` (`:14-22`) appended at `:36`; `data/` and `results/` deliberately excluded (`:13`) | `IMPLEMENTED` + `TESTED` Tier U/N with a **stubbed** share directory (`test_transreid_input_adapter.py:164`, `test_identity_node_transreid_integration.py:379`). `NOT TESTED` against a real install tree — colcon is present, ROS 2 is not (§10 level 5) | `FAIL — VERIFIED` (B‑05) | **PARTIAL** | Code half closed (TR‑05); installed‑share resolution `INTEGRATION PENDING`. Note the row's cited evidence "no `get_package_share_directory`" is now stale — it exists in `reid_input_adapter.py:129` |
| R‑38 | Runtime dependencies declared for aarch64 / torchreid resolvable (run sheet §6 G‑c/G‑d) | `SNAP/…/edgesync_identity/setup.py:15 install_requires=['setuptools', 'numpy']`; `package.xml:10-14` declares only `rclpy`, `sensor_msgs`, `edgesync_msgs`, `test_depend pytest`; `edgesync_bringup/setup.py:37 install_requires=['setuptools']` | `MEASURED` this pass by reading both files: **no torch, no torchvision, no OpenCV, no tensorrt, no gdown declared anywhere**, although the code imports all of them (`osnet_torch_extractor.py:58 import torch`, `:133 cv2`, `osnet_backbone.py:453 gdown`, `transreid_trt_extractor.py:42-46` tensorrt bindings). The row's second claim (a harness that builds a Windows `sys.prefix` path) is consistent with `RES/trt_blocked/probe_build_report.json` | `NOT TESTED` | **FAIL** (undeclared deps) | Blocks any aarch64 install and makes **T7‑D29**'s silent cv2 fallback reachable on a fresh Jetson image → A‑7, A‑6 |
| R‑39 | No tool may report simulated results as device success (charter §14) | The row names `profile_edge_latency.py:53, 83-87, 274` (+18.0 ms fudge, `is_jetson = os.path.exists('/usr/bin/tegrastats')`, write‑back into `timing_budgets.yaml`). `find` in this package → **the file is absent**. Two **counter‑examples on disk** now: `run_real_pipeline.py:563` labels itself "Laptop -- NOT Jetson Orin Nano", and `benchmark_transreid_jetson.py:21-22, :233-239` refuses off‑target and stamps `NOT A JETSON BENCHMARK` under `--allow-non-target` | `UNRESOLVED` for the named script (`D`‑class absent artefact); `MEASURED` for the two compliant harnesses. The older finding (B‑13) therefore cannot be re‑verified here and is carried as `UNRESOLVED` → T7‑D18 | `FAIL — VERIFIED` (B‑13) | **UNRESOLVED** (named tool missing); the *rule* is honoured by the tools that are present | Do not re‑introduce the old tool; run the refusing harness instead → T7‑D18, A‑10 |
| R‑40 | `ES-EXP07-TASK7-REPORT-01-REV01.md` complete (charter §541) | Two revisions are packaged: REV01 (131 lines, 10,123 B) and **REV02** (348 lines, 52,908 B), which supersedes it (`REP/ES-EXP07-TASK7-REPORT-01-REV02.md:5`) | `MEASURED` this pass by count: `<fill>` occurrences — **REV01 = 110** (the row's number reproduces exactly), **REV02 = 109**. REV02's own defence is at `:5`: it "fills exactly the cells that this host can measure and leaves the device cells unfilled with their blocker named", and most surviving tokens are *quoted* REV01 cells (e.g. `:17-19`, `:52-53`) or deliberately unfilled device rows (`:42-43`, `:229-230`, `:262`). Reproduction paths still unresolvable: `:289` names `edgesync_bringup/results/experiment7/task7/{osnet,transreid}.json`, `comparison.json`, `comparison.md`, `tegrastats.log`, `memory_trials.json`, and `find` shows `experiment_results/results/experiment7/` holds **7 JSON files and no `task7/` subdirectory**; `tools/ablation/task7/run_task7_benchmark.sh` is absent. The row's "setup.py installs only config/launch" is now stale (`bringup/setup.py:36` installs models too) | `FAIL — VERIFIED` (B‑17) | **PARTIAL** | Filled‑where‑measurable holds; the §8 reproduction recipe still points at an absent results subtree → §12.2, A‑6. The **stale sub‑claim** is recorded in §9.8, not silently dropped |
| R‑41 | Changes committed vs working tree; authorship attributable (charter §35) | No `.git` directory anywhere in the package (`MEASURED`: `find` returns none) | `BLOCKED` by the owner, not by the tools: `REP/…REV02.md:19` gate **G.6** = "OPEN / NOT OBTAINABLE — not a git repository; `git init`/commit are prohibited by charter §2 and the owner handles Git. Results carry `NO-COMMIT`". This pass ran **no** Git write command | `BLOCKED` (B‑18) | **BLOCKED** | Accepted substitute: SHA‑256 provenance (§3.11, §12.1). Any hash‑pinned claim in this report is reproducible without VCS |
| R‑42 | No parameter tuning on the test set; τ not re‑derived (proposal §8, standing rules) | τ sites listed at R‑23; no calibration code path exists in `SNAP/…/scripts/` | `MEASURED` this pass: `grep` found no τ re‑derivation, and the analyzer **refuses** if the loaded τ ≠ 0.70 (`analyze_task7_statistics.py:28-32, :703-705`; the 0.62 refusal is execution log §5 row 21, quoted in §6.7). This pass changed **no** threshold, no split and no weighting | `NOT APPLICABLE` | **PASS** (rule honoured) | None. But note the distinction the brief asks for: the **configured** τ is 0.70 and the **applied** τ in the runtime today is the node's **code default** (T7‑D26); and the data supports an operating point near each arm's EER (0.3276 / 0.3058) — calibrating it is an owner decision, **not** something this pass may do → A‑4 |

### 9.8 Verdict‑change ledger — where this pass moved away from the matrix, and on what evidence

**23 of the 42 rows changed verdict class; 19 did not** (`CALCULATED` from the two lists below; 23 + 19 = 42).
A change is recorded **only** with the packaged evidence that forced it. Rows whose *stated reason* is simply
no longer true of this tree are marked **STALE** — that is the category the brief asks not to be hidden,
because a reader who trusts the CSV would keep looking for a defect that has been fixed, or would keep
asserting a gap that has been closed.

| ID | Matrix | This pass | Why it moved | Evidence |
|---|---|---|---|---|
| R‑03 | `FAIL — VERIFIED` ("0 hits") | **PARTIAL** / device **BLOCKED** | **STALE.** The file the row says does not exist is in the package, twice | `ASSET/transreid_trt_extractor.py` = `SNAP/…/transreid_trt_extractor.py`, 19,753 B; `find` for `*.plan`/`*.onnx` = 0 → engine still absent |
| R‑04 | `PASS — VERIFIED` | **PARTIAL** | `strict=False` half closed (**STALE**); the checkpoint is not in this package, so the load assertion cannot be re‑run | TR‑06 in `transreid_extractor.py`; `find` for `*.pth` returns only the OSNet weights → **T7‑D24** |
| R‑06 | `PASS — VERIFIED` | **INHERITED** | Digest cannot be recomputed here; the recorded 414,724,314 B / `84305529…` survives as metadata only | `RES/provenance/task7_provenance_measurements.json → checkpoints.transreid_checkpoint`; §12.2 |
| R‑10 | `PASS — VERIFIED` | **PARTIAL** | True for the class and the node parameter, false for the service and the base API | `base_identity_extractor.py:27`; `IdentityVerify.srv` field list |
| R‑11 | `FAIL — VERIFIED`, gap "G.5 unanswered" | **PARTIAL** | **STALE.** G.5 was answered and implemented | `REP/…REV02.md:18` (PF‑10c); `identity_node.py:126, :147` |
| R‑14 | `UNVERIFIED CLAIM` | **NOT TESTED** | Same fact, precise label: the fit has not been re‑executed by any pass that must not write into the tree | `fit_svd_projection.py` present; charter §3A + §2 of this brief |
| R‑17 | `PASS — VERIFIED` | **PARTIAL** (module PASS, integration FAIL) | The row's own gap is confirmed, **and** wiring it in as written arms a network fetch | `grep osnet_backbone` importers = tests + audit scripts only; `osnet_backbone.py:527, :537-538, :453, :485, :16-26` → **T7‑D31** |
| R‑20 | `PASS — VERIFIED` | **INHERITED** | Market‑1501 is not in this package | `find` for `*market*` = 0 hits |
| R‑21 | `PASS — VERIFIED` | **INHERITED** | Same root cause as R‑20 | as above |
| R‑22 | `FAIL — VERIFIED` ("no CLI override") | **PARTIAL** | **STALE — the row's central claim is false now.** The CLI overrides exist | `evaluate_offline.py:473 --population` (with `choices`), `:480 --id-min`, `:482 --id-max`, threaded at `:498` |
| R‑23 | `PASS — VERIFIED` ("declared, **injected**, used") | **PARTIAL** | **Injected is not supported.** No channel carries τ to the node; the code default fires | `reid_launch_defaults.py:61-66` (six keys, no threshold), `robot_bringup.launch.py:127-137` (dict has no `similarity_threshold`), and `test_reid_launch_defaults.py:115` which **asserts** `'reid_similarity_threshold' not in found` → **T7‑D26** |
| R‑24 | `UNVERIFIED CLAIM` | **NOT TESTED** | Label sharpened; the artefact is present but wholly unattributable | file re‑read; `grep` of all packaged `.py` for its name = 0 files |
| R‑25 | `UNVERIFIED CLAIM` | **NOT TESTED** | New content: all four recorded rows are "(With Cam)", so the audit contains **no camera‑agnostic row** — the deployed arm is absent from the verification evidence | `verification_threshold_audit.json` rows re‑read in full |
| R‑26 | `FAIL — VERIFIED` (B‑09) | **PASS** (tool) | **STALE.** McNemar and Holm now exist and were executed | `analyze_task7_statistics.py:124, :139`; §6.4 |
| R‑27 | `FAIL — VERIFIED` (B‑09) | **PARTIAL** | **STALE for Wilson**; the false‑handoff half is unmeasurable offline | `analyze_task7_statistics.py:109`; `benchmark_transreid_jetson.py:368` |
| R‑28 | `FAIL — VERIFIED` (B‑14) | **PARTIAL** | B‑14 is confirmed for the *named* harness, but a real‑crop device harness exists and was not credited | `run_real_pipeline.py:566, :570, :595`; `benchmark_transreid_jetson.py:198-199, :265` |
| R‑31 | `FAIL — VERIFIED`, gap "no VmRSS/tegrastats code in repo" | **FAIL** (numbers) / method **IMPLEMENTED** | **STALE gap.** The method is implemented, just never run on a target | `benchmark_transreid_jetson.py:95, :103, :112-126, :308` |
| R‑34 | `PARTIAL` | **UNRESOLVED** | A scope question, not a coverage percentage; only an owner reply settles it | `REP/…REV02.md:13-21` has no gate covering it |
| R‑35 | `FAIL — VERIFIED` (B‑04) | **PARTIAL** | **STALE half.** The launch‑dict override is gone | `robot_bringup.launch.py:28-43` (TR‑01 comment + `resolve_reid_config`) |
| R‑37 | `FAIL — VERIFIED` (B‑05), evidence "no `get_package_share_directory`" | **PARTIAL** | **STALE.** It exists, and the install‑space bug was fixed earlier than the row assumes | `reid_input_adapter.py:129-130`; `edgesync_bringup/setup.py:8-13` "FIX B16 (2026‑10‑16)", `:36` |
| R‑38 | `NOT TESTED` | **FAIL** | Moving it *down*, not up: undeclared dependencies are directly visible in the two packaged `setup.py` files | `edgesync_identity/setup.py:15`; `package.xml:10-14`; `edgesync_bringup/setup.py:37` → **T7‑D29**, **T7‑D31** |
| R‑39 | `FAIL — VERIFIED` (B‑13) | **UNRESOLVED** | The named tool is not in this package, so this pass cannot confirm or clear it; meanwhile the two tools that **are** here honour the rule | `find profile_edge_latency.py` = 0; `run_real_pipeline.py:563`; `benchmark_transreid_jetson.py:21-22, :233-239` |
| R‑40 | `FAIL — VERIFIED` (B‑17) | **PARTIAL** | REV02 exists and fills what is fillable; its §8 reproduction recipe still points at an absent subtree; its "setup.py installs only config/launch" note is stale | `:5`, `:7`, `:289` of REV02; `find results/experiment7` = 7 JSON, no `task7/`; `bringup/setup.py:36` |
| R‑42 | `NOT APPLICABLE` | **PASS** | The rule was honoured and this pass can say so from evidence, not from a disclaimer | `analyze_task7_statistics.py:28-32, :703-705` (refusal path); no τ/weight/split edit in §12.4's write scope |

**Two cross‑references in the matrix that this pass found wrong, recorded rather than corrected** (the file
must not be edited — §2): R‑10 → "R‑32" (**T7‑D30**, §9.1), and R‑13 vs gate **G.2**, which the proposal
declares "SATISFIED by measurement, not by owner reply" (`REP/…REV02.md:16`) on the strength of a *recorded*
digest while the shipped config still selects that file **by name** (`algorithm_thresholds.yaml:47`,
`exp7_transreid.yaml:57`). G.2 is therefore satisfied as a **record** and not as a **mechanism**, which is
exactly the distinction **T7‑D04** makes; a reader who takes G.2 as closed would believe the name‑collision
namespace is pinned. It is not.

### 9.9 Requirement status roll‑up

Counting this pass's verdicts over the 42 rows (the classes are this report's §5.9 vocabulary):

| Verdict | Rows | Share | Reading |
|---|---|---|---|
| `PASS` | **10** | `CALCULATED` 10 ÷ 42 = **23.81 %** | The requirement is met and this pass could point at the evidence |
| `PARTIAL` | **14** | **33.33 %** | Met in code or for one population of the requirement, unproven or unmet for the other |
| `FAIL` | **7** | **16.67 %** | Live defect, or a delivered number that does not support the requirement |
| `NOT TESTED` | **4** | **9.52 %** | No executing evidence exists at the level the requirement demands |
| `BLOCKED` | **2** | **4.76 %** | R‑32 (device, G.7 `DENIED`), R‑41 (VCS, G.6 `NOT OBTAINABLE`) |
| `UNRESOLVED` | **2** | **4.76 %** | R‑34 (scope, needs an owner), R‑39 (named tool absent) |
| `INHERITED`, not re‑verifiable in this package | **3** | **7.14 %** | R‑06, R‑20, R‑21 — the checkpoint and the dataset are outside the handoff tree |

10 + 14 + 7 + 4 + 2 + 2 + 3 = **42** ✓.

**What this roll‑up does not mean.** A `PASS` here never means "deployed" or "verified on a robot": 8 of the
10 `PASS` rows (R‑02, R‑07, R‑09, R‑12, R‑15, R‑18, R‑19, R‑26) are **offline or unit‑tier** results, one is a
policy observation (R‑42), and one (R‑08) is a constant. The highest requirement class Task 7 reaches on this
host is Tier U/N — see §10. And per the brief's own caution, a passing unit test establishes neither ROS 2
integration, nor hardware performance, nor deployment readiness, while a reported metric is not reproducible
merely because a document prints its value.

### 9.10 What no action on this host can close

Three requirements are not gaps in the code and cannot be closed by any further inspection, test run or
recomputation on this machine; each needs something that does not exist here:

1. **R‑01, R‑32, and the device halves of R‑28/R‑31** — need Jetson Orin Nano access. Gate **G.7** is
   `DENIED FOR THIS SESSION` (`REP/…REV02.md:20`), and the packaged refusal artefacts
   (`RES/trt_blocked/probe_build_report.json`, `build_run2.log`, `bench_run2.log`) are the proof that the
   harness itself says so. Only the owner can change this → A‑1.
2. **R‑11's service half, R‑34, and the budget halves of R‑24/R‑25** — need owner decisions, not
   measurements: whether `IdentityVerify.srv` may ever carry a camera field (G.5 answered only for the
   per‑robot parameter), which experiment owns cross‑robot handoff metrics, and what `dt_id_ms` /
   `unified_memory_limit_mb` are (G.1, G.3 both `OPEN`) → A‑2.
3. **R‑06, R‑20, R‑21 and every number in §6.1‑§6.5** — need the assets that were packaged *around* this
   handoff: the 414,724,314 B checkpoint, Market‑1501 (68,036 files) and the three embedding `.npz`
   (55,193,927 + 37,094,828 + 37,095,723 B). Without them these figures are `INHERITED`, and no QC step in
   this pass can promote them to `MEASURED` → §12.2, A‑6.

## 10. Integration and deployment readiness

This section answers one question in ten separate ways: **how far has Task 7 actually been validated, as
opposed to how far has it been written?** Each level below is a gate with its own evidence class. The rule
applied throughout is the one this report's §1.6 states: a level counts as **reached** only if some
verification has *executed* at that level and left an artefact in this package or a log named here. Reading
code that would work at that level is `IMPLEMENTED`, not reached. Writing a launch file is not running one;
exporting an ONNX graph is not building an engine; a laptop number is not a Jetson number.

Two cross-cutting constraints set the ceiling before any level is examined:

* **Middleware.** `MEASURED` in this pass (transcribed here; §12.4 records that this pass filed no new
  logs): `colcon` is on `PATH`, while `ros2`, `rclpy`, `launch` and `sensor_msgs` are **absent** — the same
  split the inherited record shows (`REP/test-execution-log.md` E‑4/E‑7, which additionally shows
  `python -m pytest` working and `ros2`/`colcon` returning 127 on 2026‑10‑19). `MEASURED`: no `build/`,
  `install/` or `log/` directory exists anywhere under `TASK07/`. Therefore **no ROS 2 build of any package
  in this task has ever run on this host**, which caps levels 5‑10 at the outset.
* **Target hardware.** Gate **G.7** is recorded `DENIED FOR THIS SESSION`
  (`REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:20`), and the harness agrees: `RES/trt_blocked/bench_run2.log`
  reads "BLOCKED — Pending Jetson Orin Nano access: none of `/etc/nv_tegra_release`,
  `/proc/device-tree/model`, `/etc/nvidia-l4t-version` exist on this host. Charter §7: a laptop benchmark is
  not a Jetson benchmark." Levels 8‑10 are `BLOCKED` by that, not by unfinished code.

### 10.1 The ladder at a glance

| # | Level | Reached? | Highest evidence class produced at this level | What stops it going further | Next action |
|---|---|---|---|---|---|
| 1 | Source inspection | **YES** | `IMPLEMENTED`, `MEASURED` (inspection) | nothing — this is the level this pass operates at | — |
| 2 | Unit / package tests | **YES, at Tier U/N** | `TESTED` (§7) | 1 uncollectable identity file, 1 uncollectable + 2 failing bringup tests, absent assets | A‑6 |
| 3 | Offline experiments | **YES, 2026‑10‑19, off‑package** | `INHERITED` here, `MEASURED` there | the 414 MB checkpoint, Market‑1501 and the three embedding `.npz` are not in the package | A‑6 |
| 4 | Simulation / benchmark | **PARTIAL** | `INHERITED` (ONNX parity, laptop latency) | no engine, no device, and the latency harness feeds random noise | A‑1, A‑10 |
| 5 | ROS 2 build & interface compilation | **NO** | `IMPLEMENTED` (correct CMake recipe, never executed) | `ros2`/`rclpy`/`launch`/`sensor_msgs` absent; no build tree exists | A‑6 |
| 6 | Node‑to‑node | **NO** | nothing executed | level 5 + no crop publisher + 3 of 7 launched packages absent from the package | A‑6 |
| 7 | DDS / discovery / timing | **NO** | nothing executed | level 5 + no code consumes the budgets, τ or the DDS profile | A‑13 |
| 8 | Jetson + TensorRT | **NO — `BLOCKED`** | refusal artefacts only | G.7 denied; `tensorrt`/`pycuda`/`trtexec` unimportable | A‑1 |
| 9 | Physical robot | **NO** | nothing, anywhere | no robot session exists in any evidence in this package | A‑2 |
| 10 | End‑to‑end safety | **NO** | nothing | levels 6‑9 all empty; the Task 7 ↔ Task 13 handoff is `NOT MEASURED` by its own harness | A‑1, A‑2 |

**One-sentence answer.** Task 7 is ready as an **offline analytical component with a machine-checked
export path**; it is **not ready as an integrated system**, and nothing in this package can move it past
level 4 on this host.

### 10.2 Level 1 — source inspection: reached

The whole delivered surface was read in this pass and is inventoried with sizes and digests in §3: 3 ROS
packages in the Task 7 snapshot (`edgesync_bringup`, `edgesync_identity`, `edgesync_msgs`), the production
extractor/verifier/adapter stack, the two launch files, 8 bringup configs, the 16 identity and 2 bringup test
files under `TST/` (§7.3 enumerates each),
and every result, provenance and refusal artefact under `RES/`. §9 then re-derived all 42 requirement rows
against that source rather than against the inherited matrix, which is what surfaced **T7‑D30**, **T7‑D31**
and **T7‑D32** and the 11 stale rows listed in §9.8. Level 1 is the only level this pass can *add* evidence
at, and it is where the report's confidence genuinely comes from.

*What it does not establish.* Inspection cannot produce a single runtime number. Every statement below it in
the ladder that this report marks `NOT MEASURED` stays `NOT MEASURED` no matter how thoroughly the code was
read — a correctly written subscription that has never been created is still an unexecuted subscription.

### 10.3 Level 2 — unit and package tests: reached at Tier U/N, never "fully passing"

Facts are in §7 and are not restated here except to fix their ladder meaning:

* `edgesync_identity` copy: **360 passed, 15 skipped, 1 collection error, exit 1**
  (`VALIDATION/fresh_evidence/identity_copy_run.txt`).
* `edgesync_bringup` copy: **55 passed, 2 failed, 1 uncollectable file, exit 1**
  (`VALIDATION/fresh_evidence/bringup_copy_run.txt`).
* Both exit codes are non‑zero. The inherited pair is no greener: `REP/test-execution-log.md` T‑1 records
  `pytest tests/ -q` at **exit 2 with 0 tests executed** (`ModuleNotFoundError: No module named 'rclpy'` at
  `tests/test_identity_service.py:11`), and T‑2 records the `--ignore` workaround at **56 passed, exit 0** —
  a subset, obtained by deleting the file that cannot run, on a day when fixtures were satisfied with
  "mocked / synthetic tensors — no checkpoint loaded".

The tier boundary is stated by the suite itself. `TST/edgesync_identity/tests/test_identity_node_transreid_integration.py:3-8`:
"Test class: UNIT. ROS 2 middleware (rclpy, edgesync_msgs, sensor_msgs) is not installed on this host, so
the stubs installed below replace ONLY the middleware scaffolding … They are NOT ROS 2 integration tests: no
DDS, no live service client, no real executor spin. Integration status is recorded separately as **BLOCKED**
— Pending ROS 2 runtime." Its stubs are fabricated in the test file (`:127-151`): `rclpy.init` is a function
that raises (`:133-134`), and `edgesync_msgs.msg` is a `types.ModuleType` holding two stand-in classes.

*Consequence found while fixing this report's own §4.4 table.* Because that stub provides only
`HandoffConfirm`/`HandoffCancel`, the `from edgesync_msgs.msg import ReidSignature` at
`identity_node.py:302` cannot succeed under it, so the node takes the error branch at `:304-307` and returns.
`MEASURED`: `ReidSignature` appears in Task 7's snapshot **only** in `identity_node.py` and in
`exp13_bringup.launch.py` text — **no Task 7 test executes the signature producer**. §4.4 now records that
as `NOT TESTED` on the producer side instead of the inherited "generated message classes from the `.msg`
text" claim, which this pass could not support.

### 10.4 Level 3 — offline experiments: reached on 2026‑10‑19, on the owner's laptop, off this package

`RES/embeddings/task7_embeddings_manifest.json` (7,352 B, `generated_utc` 2026‑10‑19T14:47:29Z) is the
strongest instrument in the package, because it records its own inputs and asserts its own population rather
than assuming a published one:

| Field | Value | Class |
|---|---|---|
| `extractor_embedding_count` | **19,281** | `INHERITED` (manifest present, arrays absent) |
| gallery | 19,732 scanned → **15,913 kept**, 3,819 `junk_identity_-1` excluded, 751 identities, 2,798 distractors at pid 0 | `INHERITED` |
| query | 3,368 scanned → **3,368 kept**, 750 identities, id range 1‑1501 | `INHERITED` |
| `CALCULATED` check | 15,913 + 3,368 = **19,281** = the recorded embedding count exactly — the manifest is internally consistent | `CALCULATED` |
| `parity_check` | `parity_tolerance: 2e-05`, `every_nth: 1`, i.e. **all 19,281 images** were cross-checked production‑vs‑re‑derived contract; `result: PASS`, "a violation aborts the run" | `INHERITED` |
| `extractor_failure_counts` | `{}` and `extractor_last_failure: null` | `INHERITED` |
| checkpoint coverage | 211 keys → 210 after filter (`classifier.weight` dropped), 206 model entries, `missing_keys: []`, `coverage_pct: 100.0`, `torch_parameters: 100,769,512` | `INHERITED` |
| OSNet arm | `osnet_factory_attempt.result`: **"FAILED — recorded, not repaired (charter §3B): RuntimeError: operator torchvision::nms does not exist"** | `INHERITED` |
| host | Windows‑10‑10.0.26300, Python 3.10.11, torch 2.5.1+cu121, CUDA available, `NVIDIA GeForce RTX 3050 Laptop GPU`, `elapsed_s: 956.4` | `INHERITED` |

Two ladder-relevant readings of that table:

1. **The OSNet row is why H7a cannot be answered, not merely unmeasured.** No OSNet embeddings ⇒ no paired
   per-query outcomes ⇒ no McNemar pair ⇒ the baseline arm of the charter's own hypothesis does not exist.
   §6.9 and T7‑D01 carry this; it is level‑3's ceiling, not a gap that more offline work fixes.
2. **The population was chosen from directory semantics, and the manifest says so** (`rules.gallery` keeps
   the pid‑0 distractors as negatives; `counts_asserted_from: "this run only — no standard dataset count is
   assumed"`). That is what B‑12/TR‑07 fixed, and §6.9.7 records the `DEFAULT_POPULATION = 'directory'`
   opt‑in in `evaluate_offline.py:99`.

*Why the label is `INHERITED` here.* The three embedding arrays the numbers were computed from are outside
the package (N‑12: 55,193,927 + 37,094,828 + 37,095,723 B under `C:\Users\PC\edgesync_es07_runs\embeddings_full_directory\`,
per `RES/embeddings/task7_embeddings_manifest.json → files`), as are Market‑1501 and `transformer_120.pth`
(414,724,314 B, sha256 `843055293e160da0…`). This pass re‑executed **nothing** at level 3 and would not be
authorised to: charter §3A forbids regenerating embeddings to repair a packaging gap. What *was* executed in
this pass at level 3's boundary is the statistics engine over synthetic pools (Tier U, §7.3), which proves
the arithmetic, not the accuracy.

### 10.5 Level 4 — simulation and benchmark: partially reached, with two quality problems

**4a. ONNX export, graph validation and Torch↔ONNXRuntime parity — `INHERITED`, and internally reproducible.**
`RES/onnx/transreid_onnx_report.json` (2026‑10‑19T14:54:12Z) and `…_rerun.json` (15:11:35Z) both record:

* two graphs exported with `opset 17`, `ir_version 8`, `dynamo: false`; agnostic input `image`, SIE input
  `image, cam_label`; `n_nodes` 779 / 783; `checker: PASS (onnx.checker.check_model)`;
* bytes and digests **identical across the two runs** — `343,380,183 B / 318b6857585d5d7c…` (agnostic) and
  `343,399,114 B / 2cd8ccc774c59337…` (SIE);
* parity over `n_images_selected = 8`, `n_images_scored = 8`, population `directory`, `test_device:
  onnxruntime CPUExecutionProvider (fp32)`, `ORT_is_CPU_only: true`, `reference_device: cpu`;
* worst-case numbers, identical in both files to the last digit — agnostic
  `worst_max_abs_768 = 1.239776611328125e-05`, `worst_max_abs_512_after_contract = 6.09084963798523e-07`,
  `worst_cosine_768 = 0.9999999999960024`; SIE `1.2874603271484375e-05`, `6.854534149169922e-07`,
  `0.9999999999954788`; against `tolerance 0.002` and `cosine_floor 0.9999`, `verdict: PASS` in both arms;
* `production_api_vs_torch_worst_max_abs_512 = {agnostic: 0.0, sie: 0.0}` — the *deployed*
  `extract_embedding()` path reproduces bit-for-bit, which is a stronger statement than the graph check;
* `sie_effect_worst_l2_on_512 = 0.15214425314786348`, reported separately on purpose (the arms legitimately
  differ); `runtime_checks`: batch 1 → `(1, 768)`, batch 2 → `(2, 768)`, off-contract 128×64 → "refused by
  the graph: InvalidArgument";
* and its own limits, verbatim: `not_claimed.tensorrt_engine = "BLOCKED — Pending Jetson Orin Nano access
  (no tensorrt, trtexec or JetPack on this host)"`, `not_claimed.target_latency = "NOT MEASURED — an
  ONNX/CPU parity check is not a speedup claim"`, `not_claimed.accuracy_parity_on_device = "NOT MEASURED —
  requires the TensorRT arm on the target"`.

`CALCULATED`: the two runs differ in `exports.*.seconds` (5.3→9.9 agnostic, 2.7→5.2 SIE) while every digest
and every parity figure matches. So reproducibility at this level is **established for the artefacts** and
**explicitly absent for wall-clock fields** — which is exactly the pattern §10 level 8 must not confuse.

*What 4a does not establish.* Eight images, one resolution, one dtype, CPU providers. It validates the
exported graph and the preprocessing/projection contract (`parity.caveat`, quoted in §6.7); it does not
validate an engine, an fp16 numerics path, a token count at any other resolution, or the 210‑token
`pos_embed` assumption (`shape_assumptions.pos_embed`: any other input size "change[s] the token count and
[is] recorded as off-contract"). `deployment_notes.engine_build_requirements` lists three conditions for a
build that has not happened, including "the TensorRT parser has to resolve the same dim during the engine
build" — the output dims are symbolic (`["batch", "BatchNormalizationfeature_768_dim_1"]`).

**4b. Latency and memory benchmarks — reached as *laptop* numbers only, on synthetic input.**
`RES/results/experiment7/benchmark_latency_memory.json` (1,125 B) records TransReID 100,769,512 parameters /
384.4051818847656 MB / median backbone 56.770099996356294 ms / median full pipeline 58.718050000607036 ms,
and OSNet 2,169,508 parameters / median backbone 24.78759999939939 ms / median full pipeline
28.16894999705255 ms. `CALCULATED`: 58.718050000607036 / 28.16894999705255 = **2.084495517466962** —
TransReID ≈ 2.08× slower than OSNet end-to-end on this host, and that is the provenance of the ratio CA‑07
traced in §6.8. The same file pair also carries the contradictory 13.431399998808047 / 18.876250000175787 =
**0.711549** from `gate3_real_results.json`, i.e. the opposite sign; §8 keeps that conflict `UNRESOLVED`.

The method is the problem, and the harness admits it in its own text: `run_real_pipeline.py:563` prints
"STEP 4: Latency Measurement (Laptop -- **NOT** Jetson Orin Nano)", `:566` builds the timed crop from
`np.random.randint(0, 255, (256, 128, 3))` — **uniform random noise, not a person** — and `:570` passes a
`dummy_cam` tensor. T7‑D23 registers this. So level 4's latency column is:
`MEASURED` **as a host property** (the export ran and reported real medians) and `NOT MEASURED` **as any
deployment-relevant quantity** — no locked clocks, no target, no real pixels, no p95, no concurrent
perception stack. A noise image also does not exercise the JPEG decode path the Jetson session would pay
for, so the laptop figure is not even a conservative bound.

### 10.6 Level 5 — ROS 2 build and interface compilation: not reached, on any host in this evidence set

*What exists.* The build recipe is correct on inspection, including the Task 7 additions:
`SNAP/edgesync_msgs/CMakeLists.txt` declares `find_package` for `ament_cmake`, `builtin_interfaces`,
`rosidl_default_generators`, `sensor_msgs`, `geometry_msgs` (`:8-12`), registers `msg/ReidSignature.msg`
**additively** with the comment "no existing type above was edited or reordered" (`:27-29`), keeps the
service list `set(srv_files "srv/IdentityVerify.srv")` (`:32-34`), and calls
`rosidl_generate_interfaces(${PROJECT_NAME} ${msg_files} ${srv_files} DEPENDENCIES … ADD_LINTER_TESTS)`
(`:36-44`); `package.xml:22` puts the package in `member_of_group rosidl_interface_packages`. That is the
*right way* to add a type to a frozen IDL surface, and it is `IMPLEMENTED`.

*What has never happened.* `MEASURED` this pass and `INHERITED` on 2026‑10‑19, in both directions: **no
colcon build of `edgesync_msgs`, `edgesync_identity` or `edgesync_bringup` appears anywhere in this
package** — no `build/`, `install/` or `log/` tree exists (§10 preamble), `REP/test-execution-log.md` E‑7
records `ros2`/`colcon` returning 127, and `rclpy` cannot be imported (E‑4, and again at
`tests/test_identity_service.py:11`, which is why that file is uncollected in §7). `test_robot_bringup_launch.py`
is uncollectable for the same family of reason — `ImportError: cannot import name 'LaunchDescription' from
'launch' (unknown location)` at `:21` — so even the launch file's own syntax test has never parsed.
Therefore `ReidSignature.msg` and `IdentityVerify.srv` have **never been compiled into generated code**, and
`ADD_LINTER_TESTS` has never run.

*Two packaging defects sit underneath this level and would survive a ROS 2 install.*
`edgesync_identity/setup.py:11-14` installs only the `rosidl_adapter` resource index marker and `package.xml`
— no `models/`, no `config/` — and its `install_requires` is `['setuptools', 'numpy']` (`:15`), so `torch`,
`opencv-python`, `tensorrt` and `gdown` are declared nowhere in either package (`edgesync_bringup/setup.py:37`
likewise lists only `setuptools`; T7‑D29). By contrast `edgesync_bringup/setup.py:8-13` carries "# FIX B16
(2026-10-16): models/ was never installed…" and its `model_data_files()` (`:14-22`) does install assets into
that share. Consequence: the `reid_checkpoint_path` / `reid_projection_path` values in
`SNAP/edgesync_bringup/config/algorithm_thresholds.yaml:46-47` resolve only against the **bringup** share,
which is why `reid_input_adapter.py:129-130` needs `get_package_share_directory` at all — the resolution path
is real code with no installed-share test behind it (§9 R‑37 correction; `INTEGRATION PENDING` at this level).

### 10.7 Level 6 — node-to-node integration: not reached, and two gaps are deliberate

Nothing in this package shows two Task 7 nodes exchanging one message. Three independent facts fix that:

1. **The composed graph is incomplete in the package.** `SNAP/edgesync_bringup/launch/robot_bringup.launch.py`
   builds **seven** nodes: `edgesync_perception:perception_node` (`:101-103`),
   `edgesync_estimation:estimation_node` (`:110-112`), `edgesync_identity:identity_node` (`:122-124`),
   `edgesync_coordination:coordination_node` (`:142-144`), `edgesync_acquisition:target_acquisition_node`
   (`:155-157`), `edgesync_navigation:safety_authority_node` (`:168-170`) and
   `edgesync_observability:monitoring_node` (`:182-184`). `MEASURED` across the whole handoff tree:
   `edgesync_perception`, `edgesync_coordination` and `edgesync_acquisition` exist only under
   `SHARED_DEPENDENCIES/` or `TASK13/`, and **`edgesync_estimation`, `edgesync_navigation` and
   `edgesync_observability` appear nowhere in this package**. Whether they exist in the project tree is a
   question this pass did not and must not ask of the old ZIP (§2.9); *as packaged*, the graph cannot be
   assembled.
2. **The identity node's observation leg is switched off on purpose, and a test pins it off.**
   `identity_node.py:103` declares `reid_target_crop_topic` with default `''`; `:314-321` warns and returns
   without subscribing when it is empty — "no observation feeds the extractor, so this node can only ever
   publish an invalidation". `exp13_bringup.launch.py:204-208` sets it to `''` with the comment "Empty on
   purpose. No node in this repository publishes a target crop: `perception_node` computes a bounding box,
   deprojects it and publishes only the 3D centroid"; `:36-41` states the same in the file's own header, and
   adds that with `with_identity:=true` "the signature leg is live, the observation leg still needs an image
   source". `MEASURED`: `SHARED_DEPENDENCIES/…/edgesync_perception/edgesync_perception/nodes/perception_node.py`
   (12,376 B) creates exactly one subscription (`:117`) and one publisher (`:139`, the centroid) — no crop
   publisher. And the Task 13 suite **asserts the gap stays**: `test_exp7_exp13_signature_integration.py:502`
   requires `"reid_target_crop_topic': ''"` to remain in the launch text, with the message "the launch must
   not pretend a crop source exists: nothing in this repository publishes one". That is **T7‑D02**, and this
   is the second time this report has found a Task 7 integration gap locked in by a test rather than left by
   oversight (the first is T7‑D26's threshold assertion).
3. **The service surface has never been crossed.** `IdentityVerify.srv` is served at `identity_node.py:187`
   and its only test file is the uncollected `tests/test_identity_service.py` (§7.3): `NOT TESTED` at
   Tier R, `TESTED` nowhere. The client side does not exist in the package.

### 10.8 Level 7 — DDS, discovery and timing: not reached; three declared contracts have no consumer

*DDS.* `SNAP/edgesync_bringup/config/fastdds_profiles.xml` is 4,753 B with digest `7f636a3b31886500…`
(`REP/artifact-inventory.md:40`, which describes it as the "DDS discovery profile for multi-robot runs").
`MEASURED`: a `grep` for `fastdds` / `FASTRTPS` over the whole of `TASK07/` returns **no executable
reference at all**. The five hits are all prose or metadata — `SNAP/edgesync_bringup/package.xml:6` (the
package *description* string), `REP/artifact-inventory.md:40`, `REP/evidence/artifact-inventory.json:15`,
and two report lines. No `.py`, `.yaml`, `.sh` or launch file sets `ROS_FASTRTPS_PROFILES` or
`FASTRTPS_DEFAULT_PROFILES_FILE` or names the profile path. `REP/lab-readiness-checklist.md:45` goes one
step further and records the mismatch as an open item: "B‑6 | FastDDS multi-host profile parses | `ros2
launch` with `FASTDDS_PROFILE` set **as the bringup expects** | discovery works across two hosts (needed
before any handoff test)" — a checklist step that presumes a bringup mechanism the bringup does not contain,
and which has never been executed. Same class as T7‑D26: the mechanism exists on disk as a file, the runtime
path does not exist.

*Timing.* `SNAP/edgesync_bringup/config/timing_budgets.yaml:1-2` states "STRICT ZERO-HARDCODING MANDATE: All
nodes consume timing parameters from this file", and the Task 7‑relevant budget `dt_id_ms: 120` sits at `:11`
inside `t_txn_ms: 886` (`:15`). `MEASURED`: the only `dt_id_ms` in delivered code is `identity_node.py:611`,
which **computes** a local elapsed value and prints it at `:632`/`:639` — it never **reads** the parameter,
and no node declares `dt_id_ms` or `t_txn_ms`. The inherited reports say the same thing from the other side:
`REP/ES-EXP07-TASK7-REPORT-01-REV02.md:237` — "`dt_id_ms = 120` remains an unconsumed config value", with
G.1 unassigned so there is "no number to compare a p95 against" (§8.5 T7‑D01; §12 item 5).

*Threshold timing.* `identity_node.py:79` declares `similarity_threshold` = `0.70`, `:110` reads it and `:172`
passes it to `IdentityVerifier`, which compares at `core/identity_verifier.py:101-102`. So the accept rule
*is* consumed by the node — but the launch layer never delivers the configured value
(`reid_launch_defaults.py:61-66` excludes any threshold key; `robot_bringup.launch.py:127-137` injects
robot_id + six reid/exp keys and no `similarity_threshold`; `test_reid_launch_defaults.py:115` asserts its
absence). Today the two agree by coincidence (both `0.70`), so **no measured number in this report changes**;
that coincidence is why T7‑D26 is a High‑severity *latent* defect rather than a live wrong-answer defect
(§12 item 9).

*Level verdict.* `NOT TESTED`. There is no executed DDS discovery, QoS, domain-ID, clock-offset or
latency-budget measurement anywhere in Task 7's evidence; every figure in §6.8 that looks like a timing
result is a Tier U/CPU number under a T7‑D23 caveat.

### 10.9 Level 8 — Jetson Orin Nano + TensorRT: `BLOCKED`, with executed refusals

The most important property of this level is that the harness **fails closed and says why**, three times in
the package:

* `RES/trt_blocked/probe_build_report.json` (1,812 B, 2026‑10‑19T15:09:22Z) attempted the fp16 engine build
  from the digest-matched ONNX (`onnx_sha256: 318b6857585d5d7c…`, batch profile `[1, 1, 1]`, image contract
  `[1, 3, 256, 128]`, workspace 4.0 GiB) and enumerated the environment: `tensorrt_version: "NOT IMPORTABLE:
  No module named 'tensorrt'"`, `pycuda: "NOT IMPORTABLE"`, `nvpmodel`/`jetson_clocks`/`free` "NOT AVAILABLE",
  `trtexec_on_path: null`, `gpu_busy_percent: cat: /sys/class/devfreq/17000000.gpu/gpu_busy_percent: No such
  file or directory`, `is_jetson_marker_present: false` on `LAPTOP-2B5ROUGO`. Verdict, verbatim: "**BLOCKED**
  — this host presents none of `/etc/nv_tegra_release`, `/proc/device-tree/model`, `/etc/nvidia-l4t-version`.
  Charter §6 step 6 requires the engine to be built in the actual target environment; re-run on the Orin
  Nano, or pass `--allow-non-target` for a bring-up build that is **not a deployment artefact**."
* `build_run2.log` (385 B) and `bench_run2.log` (202 B) are the two console refusals for the same session.
* `SNAP/…/scripts/benchmark_transreid_jetson.py` is a 384-line, 19,598 B harness that has **never produced a
  device number**: it refuses off-target (`:21-22`, `:233-239`), stamps `NOT A JETSON BENCHMARK` under
  `--allow-non-target` (`:21-22`), refuses to benchmark noise ("FAILED — no real crops could be decoded",
  `:261-265`; `:198-199` "Real crops are required; without them this is not a benchmark"), reads
  `tegrastats` (`:112-126`, error at `:126`), `VmRSS` (`:103`), `MemAvailable` (`:95`) and `nvpmodel -q`
  (`:308`), defaults to `--iterations 200 --repeats 3 --warmup 20` (`:200-202`, protocol echoed at `:220`),
  and closes with `'cross_robot_handoff': 'NOT MEASURED and NOT inferable from latency or Market-1501
  retrieval'` (`:368`).

`MEASURED`: **zero** `.engine`, `.plan` or `.trt` artefacts exist anywhere in the package, and
`algorithm_thresholds.yaml:48` ships `reid_engine_path: ""` with the note "no engine exists on disk
(device-side, UNVERIFIED)". `osnet_trt_extractor.py:71-74` `_execute_trt` is `# Simulated or actual CUDA
buffer exchange` followed by `return np.zeros(512, dtype=np.float32)` — a placeholder that returns the
sentinel. Everything at this level is `BLOCKED` or `NOT MEASURED`: H7b (p95 vs a budget that was never
assigned), H7c (resident Δ against the 8,192 MB envelope, G.3 `OPEN`), H3 power (T7‑D18/N‑8), and the fp16
accuracy-parity claim that `not_claimed.accuracy_parity_on_device` pre-emptively disavows. Nothing on this
host can change any of it → A‑1.

### 10.10 Level 9 — physical robot: nothing, anywhere

`NOT TESTED` in the strongest sense: no robot session, no rosbag, no TF tree, no DDS discovery log, no
`cmd_vel` interaction and no operator record appears in `TASK07/` at all. The nearest things to robot
evidence are both negative. First, `identity_node.py:618` reads `getattr(self.smoother, '_zero_streak', -1)`
while the smoother's documented attribute is `zero_streak_count` (`T7‑D24`), so **every** identity log line
the node can emit reports `smoother_streak=-1` — §8.4 records that this is precisely why a future robot
session could not reuse existing log lines as evidence. Second, the two real-robot-adjacent hypotheses are untestable in the
package: cross-robot handoff is `NOT MEASURED` by the harness's own field (`:368`), and G.5's per-robot
`reid_camera_id` (`algorithm_thresholds.yaml:49-54`) says "Set per robot on the Jetson" — a decision whose
runtime effect has never been observed, only resolved in a unit test (`test_identity_node_transreid_integration.py`,
§7.3).

### 10.11 Level 10 — end-to-end safety: not reached, and out of Task 7's authority

Task 7 can contribute three things to a safety argument, and all three are currently `NOT MEASURED`:

1. **A committed-ownership interaction.** Ownership transitions, dual ownership and the watchdog are Task 13
   objects; the Task 7 half is only the signature that the arm installs. With the observation leg starved
   (§10.7 point 2) the producer path can emit only invalidations, so Task 7 cannot supply evidence that an
   identity decision influenced a handoff. `exp13_bringup.launch.py:32-35` says this itself: with
   `with_identity:=false` "nothing produces a `ReidSignature` … That is the defect the ES-EXP07 ↔ ES-EXP13
   contract exists to close, and it is not closed by a run of this file that leaves the switch off."
2. **A timing guarantee.** `watchdog_max_missed: 3` / `watchdog_period_ms: 100` (`:16-17`), `t_ack_ms: 100`
   (`:18`) and `t_claim_ms: 200` (`:19`) are consumed by Task 13's node, not by identity; the identity stage
   has no assigned share of `t_txn_ms: 886` (G.1 `OPEN`), so there is no budget that an identity latency
   measurement could violate or satisfy.
3. **A calibration guarantee.** τ = 0.70 is applied (`identity_verifier.py:101`), and the only packaged
   verification audit — `RES/results/experiment7/verification_threshold_audit.json`, four rows, all at
   τ = 0.7, all "(With Cam)" — gives FNMR **18.703290940081292 % / 42.84122197456405 % / 48.6626458633801 % /
   38.51448800314671 %** against EER thresholds (`eer_tau`) of **0.555 / 0.387 / 0.33 / 0.399** for OSNet
   native 512, TransReID native 768, centred SVD 512 and uncentred SVD 512 respectively; `MEASURED`: the file
   carries **no pair counts and no population rule**, and no `.py` in the package references its filename. A
   deployed arm whose FNMR at the shipped τ is 48.66 % sits at an operating point roughly 0.37 cosine above
   its own measured EER (`CALCULATED`: 0.70 − 0.33), while FMR is near zero (0.014 %) — the classic signature
   of a threshold that was never recalibrated for the 512‑D projected space. That is the calibration need
   recorded as A‑4 and, at the requirement level, R‑24/R‑42 `NOT TESTED`.

**Safety statement this report can defend:** no configuration in this package has been shown to be safe on a
robot, and none has been shown to be unsafe. The correct label is `NOT TESTED`, not `PASS`.

### 10.12 The ceiling, and what each rung actually needs

Read as a single sentence: **Task 7's evidence ceiling on this host is Tier N**, levels 1‑2 are genuinely
met, level 3‑4 evidence is real but inherited and narrow, and levels 5‑10 are empty for one shared reason
(no ROS 2 runtime) plus three task-specific reasons (no crop publisher, no engine, no device access).

| Rung | Minimum new evidence that would move it | Owner |
|---|---|---|
| 5 → 6 | one `colcon build --packages-select edgesync_msgs` succeeding, with generated `ReidSignature`/`IdentityVerify` types importable, and `test_robot_bringup_launch.py` collecting | A‑6 (ROS 2 runtime) |
| 6 → 7 | an executed launch showing `identity_node` in a real graph, plus a crop source: either a publisher for `reid_target_crop_topic` or a documented, test-pinned alternative input | A‑6 + T7‑D02 |
| 7 → 8 | the budgets actually consumed (`dt_id_ms`, τ via a real parameter channel) and the DDS profile loaded by something | A‑13 |
| 8 → 9 | an Orin Nano session run by the packaged refusing harness, with `tegrastats` and real crops | A‑1 (G.7) |
| 9 → 10 | a two-robot episode where a `ReidSignature` reaches an ownership commit, plus the A‑4 τ calibration | A‑1 + A‑2 + A‑4 |

Two guardrails for whoever reads this table, both required by the brief:

* **Offline performance is not real-world performance.** §6.8's medians, §6.1‑§6.5's statistics and §10.5a's
  parity verdicts are all computed at Tier U/CPU on a laptop GPU with random-noise or 8-image inputs. None
  of them bounds the device case, in either direction: the noise crops under-measure decode cost, the fp32
  parity says nothing about fp16, and a 2.08× laptop ratio coexists in this very package with a 0.71×
  recorded ratio (§8, CA‑07 `UNRESOLVED`).
* **Reaching a level is not deploying.** Even if all five rungs above land, this section's evidence classes
  would become `TESTED`/`MEASURED` at Tier R/H; they would still not establish H7b/H7c verdicts, because
  G.1 and G.3 (`OPEN`) must be answered by the owner before a measured latency or memory figure has a
  threshold to be judged against → A‑2.

## 11. Remaining work and prioritized actions

Every action below is a **forward reference already made in this report**, collected here with its completion
criterion — §11.9 lists where each id was first cited. The action ids are this report's; they are not
inherited from any project tracker, and no action has been executed by this pass.

Two global rules govern the whole list, both inherited from the charter and this brief:

* **This pass executes none of them.** Writing them down is the deliverable; §2 forbids modifying code,
  configs, tests, logs or artefacts, and charter §3A forbids regenerating embeddings or refitting the
  projection to improve a number.
* **Priority is by what unblocks what.** Several High defects cannot be closed by inspection at all (§10).
  Ordering therefore follows the dependency chain *owner decision → environment → device session →
  measurement*, not severity alone. Where an action is blocked behind another, the blocker is named.

### 11.0 Master priority order

| Rank | Action | Category | Blocks / unblocks | Executable by |
|---|---|---|---|---|
| P0 | **A‑4** calibrate τ against the deployed 512‑D space | 1 | R‑24, R‑42, §10 level 10 | owner‑authorised analysis run |
| P0 | **A‑2** G.1 and G.3 decisions | 7 | H7b, H7c, R‑24/R‑25, A‑1's acceptance test | supervisor |
| P0 | **A‑13** give τ a real parameter channel | 1/2 | T7‑D26, R‑23, §10 level 7 | implementer |
| P1 | **A‑6** restore assets, dataset and the missing tests | 3 | R‑06/R‑20/R‑21, §6.1‑§6.5 promotion, levels 2‑3 | team (copy) |
| P1 | **A‑1** one Jetson Orin Nano session with the packaged refusing harness | 3 | G.7, H7b/H7c, T7‑D01, T7‑D18 | supervisor + device |
| P1 | **D02‑fix** supply a crop source or declare the producer out of scope | 1 | T7‑D02, level 6, R‑11 | implementer + owner |
| P2 | **A‑12** wire `osnet_backbone` safely, or not at all | 2 | T7‑D31, R‑16/R‑17 | implementer |
| P2 | **A‑7** declare the runtime dependencies | 2 | T7‑D29, R‑38, level 5 | implementer |
| P2 | **A‑9** loader hardening + fit reproducibility | 2/5 | B‑19, R‑12/R‑14 | implementer |
| P3 | **A‑3, A‑8, A‑10, A‑11** and the §11.5/§11.6 items | 3/5/6 | see rows | mixed |

### 11.1 Category 1 — critical correctness and safety issues

**A‑4 — Calibrate τ for the centred 512‑D projected space, or declare the verification stage not validated.**
*Evidence of need*: `RES/results/experiment7/verification_threshold_audit.json` puts the deployed arm at
FNMR **48.6626458633801 %** at τ = 0.70 with FMR 0.014 % and `eer_tau` **0.33** — an operating point
`CALCULATED` 0.37 cosine above its own measured EER, i.e. the cost falls almost entirely on rejects. The file
declares no pair counts and no population (`CA‑10`), and τ itself reaches the verifier by a coincidence, not
a channel (T7‑D26).
*Completion criterion*: a re-generated audit that (i) states the pool and the genuine/impostor pair counts in
the artefact, (ii) reports FNMR/FMR at the EER and at ≥ 3 candidate thresholds with Wilson intervals from
`analyze_task7_statistics.py:109`, and (iii) is produced by a `.py` file that references the artefact name —
today that grep returns **0 files**. Accept only if the owner also states which side (reject versus false
accept) the handoff budget penalises.
*Blocked behind*: nothing analytically; the assets in A‑6 make it cheap.

**A‑13 — Close T7‑D26: make the configured threshold actually reach the node.**
*Evidence*: `reid_launch_defaults.py:61-66 REID_LAUNCH_KEYS` has no threshold key;
`robot_bringup.launch.py:127-137` injects seven keys and no `similarity_threshold`;
`test_reid_launch_defaults.py:111, :115` asserts `'reid_similarity_threshold' not in found`, so the current
behaviour is **deliberate and pinned**. Today YAML 0.70 (`algorithm_thresholds.yaml:7`) equals the code
default (`identity_node.py:79`), so no published number changes — that is why this is High‑latent, not
High‑live.
*Completion criterion*: one added key in `REID_LAUNCH_KEYS`, one entry in the launch parameter dict, and the
inverted assertion — a Tier‑N test that sets `reid_similarity_threshold: 0.62` in a temp overlay, runs the
resolver, and asserts the node parameter equals 0.62 (the value `test_analyze_task7_statistics.py:728`
already uses for the analyzer side). Until then R‑23 stays **PARTIAL**, not PASS.

**A‑2 — Land the two open owner gates.** G.1 (identity stage's share of `t_txn_ms: 886`) and G.3 (the memory
envelope and which quantity fills it) are both `OPEN` (`REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md:13-21`).
*Why it is first*: with G.1 unset, `benchmark_transreid_jetson.py` and `identity_node.py:611` can produce a
p95 but **nothing can pass or fail**, so H7b is undecidable even after a successful device session
(`…REV02.md:237`). With G.3 unset, H7c has no threshold, and CA‑12/N‑9 show the current recorded "memory
footprint" ranks the *smaller* model higher (`CALCULATED` 410.7265625 ÷ 413.26025390625 = 0.99386901744772).
*Completion criterion*: a written decision naming (a) an integer ms budget, (b) the statistic compared to it
(p95 or median, `end_to_end` or backbone-only, warm or cold), (c) the memory quantity (VmRSS Δ vs
`max_memory_allocated` vs `tegrastats` MEM), and (d) the concurrent-load condition. Then A‑1 can be scored.

**D02‑fix — Resolve the crop feed.** `identity_node.py:103, :314-321` will not subscribe while
`reid_target_crop_topic` is empty; `exp13_bringup.launch.py:204-208` sets it empty with the reason "no node
in this repository publishes a target crop", and `test_exp7_exp13_signature_integration.py:502` asserts the
empty value stays. So Task 7's signature producer **cannot** run in any packaged configuration.
*Completion criterion*: one of two, chosen explicitly — (i) a crop publisher added to the perception stack
plus a Tier‑R test that a `sensor_msgs/Image` reaches `_on_target_crop` and a `ReidSignature` is published,
or (ii) an owner statement that the observation leg is out of Task 7's scope, in which case §9 R‑11 is
recorded **BLOCKED (scope)** rather than PARTIAL, and this report's §10 level 6 note is replaced by that
decision. Silence is not a completion.

**T7‑D24 (fix, one line, safety-relevant observability).** `identity_node.py:618` reads
`getattr(self.smoother, '_zero_streak', -1)` while the documented attribute is `zero_streak_count`, so every
identity log line prints `smoother_streak=-1`. A robot session's post‑hoc audit would read "never produced an
invalid frame" at exactly the moment that matters.
*Completion criterion*: `getattr(self.smoother, 'zero_streak_count', -1)` plus a Tier‑N test driving N invalid
frames through the node and asserting the logged streak equals N (§8.4).

**A‑12 guardrail — do not apply the matrix's own R‑16/R‑17 fix as written.** The RTM remediation text says
"repoint to `osnet_backbone.py`", but that module's `osnet_x1_0(... pretrained=True)` default (`:527,
:537-538`) calls `init_pretrained_weights` → `gdown.download` against five hard-coded Google Drive URLs
(`:16-26, :453, :485`). Wiring it in as recommended arms an unreviewed network fetch in a supposed‑frozen
baseline path. Detailed as **T7‑D31**; the safe variant is in §11.2.

### 11.2 Category 2 — required implementation fixes

**A‑12 — Wire the OSNet baseline without a fetch, or leave the module unimported.**
*Current state `MEASURED`*: `osnet_backbone.py` has **no production importer** — the only references are
`test_experiment7_fixes.py:20`, `test_experiment7_forensic_fixes.py:23`, `reports/evidence/diag_backbone_equiv.py:29`
and `reports/preflight_probe.py:74`, and both test call sites pass `pretrained=False` (`:32, :39, :150`).
*Completion criterion*: either (i) the registry gains an `osnet_x1_0` builder constructed with
`pretrained=False` followed by `load_state_dict` from the packaged `osnet_x1_0_market1501.pth`
(10,399,605 B, sha256 `2809d3227f7d078f…`), with the `strict=` result **checked**, and a test asserting no
network call is attempted (e.g. by stubbing `gdown` to raise); or (ii) a recorded decision that the Torch
baseline stays unavailable offline, which keeps §6.10's OSNet figures `INHERITED` forever and closes R‑16/R‑17
as **BLOCKED** rather than PASS.

**A‑9 — Projection loader hardening (inherited B‑19, still live).** `transreid_extractor.py:102` casts
`mean` and `:108-109` asserts its shape is `(768,)`; nothing asserts dtype, finiteness, or device for the
projection matrix itself.
*Completion criterion*: three asserts (or equivalent refusals returning the `zeros(512)` sentinel per the
`interfaces/base_identity_extractor.py:19-21, :42-43` contract) plus one negative test per assert — feed a
float64 `V`, a `V` containing `NaN`, and a CPU/GPU‑mismatched pair, and assert the loader refuses. R‑12 then
moves from "PASS for the fit scope, hardening open" to PASS unqualified.

**A‑7 — Declare what the code imports (T7‑D29).** `MEASURED`: `edgesync_identity/setup.py:15` is
`install_requires=['setuptools', 'numpy']`, `edgesync_bringup/setup.py:37` is `['setuptools']`, and
`edgesync_identity/package.xml:10-14` lists only `rclpy`, `sensor_msgs`, `edgesync_msgs`, `pytest`. Yet
`osnet_torch_extractor.py:58` imports `torch`, `:133` imports `cv2`, `osnet_backbone.py:453` imports `gdown`,
and `transreid_trt_extractor.py:42-46` imports `tensorrt`/bindings.
*Completion criterion*: a pinned dependency set (torch/torchvision versions matched to the CUDA the target
carries, `opencv-python`, `gdown`, and `tensorrt` as an *optional* extra so a missing engine fails with a
message instead of an `ImportError` storm), plus the aarch64 note run sheet §6 G‑c/G‑d asked for. Side
effect: it removes the host-dependent silent path in `transreid_extractor.py:225` `cv2.INTER_LINEAR` versus
`:229-230` `np.linspace` fallback (§9 R‑09 caveat).

**A‑11 — Fix the traceability matrix's own pointer (T7‑D30).** `MAN/requirement-traceability-matrix.csv`
row R‑10 forwards to "R‑32"; the camera decision is R‑11. Low severity, zero-risk fix.
*Completion criterion*: one corrected cell, and §9.1's recomputed Counter still summing 16+15+3+3+2+2+1 = 42.
This pass could not make the edit (§2.5).

**A‑6b — Restore what the guide documents.** T7‑D11's external-guide references and the three shared configs
that exist only under `SNAP/edgesync_bringup/config/` (§3 alias note). *Criterion*: every path printed in the
guide resolves against an installed tree, verified by a script rather than by reading.

### 11.3 Category 3 — required experimental validation

**A‑1 — Execute one Orin Nano session, with the packaged harness, under a granted G.7.**
*Instrument already on disk*: `SNAP/…/scripts/benchmark_transreid_jetson.py` (384 lines, 19,598 B) loads the
production `TransReIdTrtExtractor` (`:11-12`), requires real decoded crops (`:14-15, :198-199`), defaults to
`--iterations 200 --repeats 3 --warmup 20` (`:200-202`) and echoes the protocol (`:220`), reads
`tegrastats` (`:112-126`), `VmRSS` (`:103`), `MemAvailable` (`:95`) and `nvpmodel -q` (`:308`), and refuses
off-target (`:21-22, :233-239`).
*Completion criterion*: a run whose report (i) exists under the device path with `is_jetson_marker_present:
true`, (ii) contains a p95 for both `end_to_end` and backbone, (iii) contains the G.3 memory quantity under
the G.3 concurrent condition, and (iv) is scored against the A‑2 numbers. Then, and only then, may §6.8's
`NOT MEASURED` cells be replaced — and **`--allow-non-target` must not be used**, because the script itself
stamps such a run `NOT A JETSON BENCHMARK`.
*Sub-item (a) — D32*: the proposal's §6 measurement plan and the harness disagree on warm‑up (`REP/…REV02.md:145`
records "50 warm-up calls over 200 timed calls, 3 independent trials", while the harness defaults to 20 at
`:202`). Pick one, record it, and re-run; the recorded v01 "1,000 calls" claim is already known‑false.

**A‑3 — Produce the first real memory number (pairs with A‑1).** Today `benchmark_latency_memory.json`
reports 410.7265625 MB and 413.26025390625 MB peak *CUDA context* figures on a laptop, which is CA‑12's
misranking. *Criterion*: a resident Δ measured as the A‑2 definition specifies, on device, with the raw
`tegrastats` lines retained (the harness already keeps them).

**A‑6 — Restore the missing inputs so recorded numbers become verifiable.** Specifically: `transformer_120.pth`
(414,724,314 B, `843055293e160da0…`), the three embedding arrays (55,193,927 + 37,094,828 + 37,095,723 B),
Market‑1501 (`bounding_box_test` 19,732 + `query` 3,368 files per the manifest), the two `.onnx` graphs
(343,380,183 B `318b6857…`, 343,399,114 B `2cd8ccc7…`), `tests/fixtures/synthetic_transreid.pth`
(414,718,880 B `2e518f28bb5ee501…`), and the `results/experiment7/task7/` artefacts that `…REV02.md:289`
cites and that do not exist (§12.2).
*Criterion*: re‑hash against the digests already in `RES/provenance/task7_provenance_measurements.json` and
`RES/embeddings/task7_embeddings_manifest.json`; every §6 number is promoted from `INHERITED` to `MEASURED‑VERIFIED`
only on a digest match, and mismatches are recorded, not smoothed.

**A‑8 — Measure the two quantities that are structurally offline-unreachable.** (i) **false‑handoff rate**
(R‑27's other half): it needs trajectories and handoff episodes, so §9 keeps it `NOT MEASURED`; criterion =
a Wilson interval computed by `wilson_interval()` over episode‑level outcomes from a real two‑robot run.
(ii) **H3 power** (N‑8): the recorded test is power‑limited at 154 discordant queries, p = 0.572845968999707;
criterion = either a larger paired pool or a pre-declared statement that H3 stays undecided. This report will
not present either as closable offline.

**A‑10 — Retire the two latency artefact families that cannot support their citations.** CA‑08 shows two of
six historically cited medians (`phase8`/`phase9`) are **not on disk**; CA‑06 shows `40.5×` was a misread of
`parameter_ratio` — the correct figures per §6.8.1 are **40.5725** with the classifier heads included,
**46.448** for parameters excluding them, and **39.879** by file size, so the label names the wrong definition
rather than the wrong arithmetic).
*Criterion*: every surviving number in §6.8 traces to a packaged file and a named field; the unsourced medians
are struck from downstream documents, and the corrected ratio is quoted with heads included or excluded
explicitly.

### 11.4 Category 4 — integration and deployment tasks

**Level 5 first.** One `colcon build --packages-select edgesync_msgs edgesync_identity edgesync_bringup` on a
host with ROS 2. *Criterion*: generated `edgesync_msgs.msg.ReidSignature` and
`edgesync_msgs.srv.IdentityVerify` importable at runtime (they have **never** been compiled — §10.6), the
`rosidl_generate_interfaces(... ADD_LINTER_TESTS)` linters
(`edgesync_msgs/CMakeLists.txt:36-44`) passing, and `test_robot_bringup_launch.py` collecting instead of
raising `ImportError: cannot import name 'LaunchDescription' from 'launch' (unknown location)` at `:21`.

**Two tests that must exist and do not.** (i) A launch-level negative test that the bringup actually delivers
`reid_model_type` to the running node — today the strongest evidence is Tier N, and §9 marks the corresponding
requirement `NOT TESTED` at launch level. (ii) An installed-share resolution test: after
`rosidl` + `ament` install, `reid_input_adapter.py:129-130` must resolve `models/transreid/transformer_120.pth`
from the **bringup** share (`edgesync_bringup/setup.py:8-22, :34-36` installs it;
`edgesync_identity/setup.py:11-14` installs no assets at all). Criterion: the node logs the resolved absolute
path and byte size, which `algorithm_thresholds.yaml:41-44` claims it does — an assertion never yet observed
firing.

**DDS/timing.** Make something consume `fastdds_profiles.xml` (today: zero executable references, §10.8) and
make the identity stage read its budget rather than merely printing `dt_id_ms` (`identity_node.py:611`).
*Criterion*: a two-host `ros2 launch` where discovery succeeds with the profile loaded — checklist item B‑6 in
`REP/lab-readiness-checklist.md:45`, still unexecuted.

**Then, and only then, levels 6→10** in the §10.12 order. No deployment claim in this report survives without
them, and none of them is achievable by editing documentation.

### 11.5 Category 5 — methodology and reproducibility improvements

1. **Re‑run the projection fit in a scratch directory** (R‑14). `fit_svd_projection.py` is present and the
   matrix itself explains why it was not re-run: "NOT re-run because it writes into the tree". *Criterion*: a
   fit reproduced on the same id range (0001‑0650) whose `V`/`mean` digest matches the deployed
   `9626bea5…` family, or a recorded statement that reproducibility is **UNVERIFIED**. Needs explicit owner
   permission — it writes files, and the 16 packaged copies (T7‑D04) already show how easily a re-fit creates
   a seventh name for a fourth matrix.
2. **Select projections by digest, not filename (T7‑D04, R‑13).** Eight names → four distinct SHA‑256;
   `algorithm_thresholds.yaml:47` selects by name. *Criterion*: the config or loader pins a digest and refuses
   a mismatch, with one negative test.
3. **Fix the analyzer's own false rationale (N‑11 / T7‑D27).** `analyze_task7_statistics.py:1212-1214`
   justifies a *correct* `NOT APPLICABLE` with a population claim that §6.9.7 disproves. *Criterion*: the
   comment states the true reason, and `test_analyze_task7_statistics.py` gains an assertion that the rationale
   text matches `DEFAULT_POPULATION` (`evaluate_offline.py:99`).
4. **Keep the artefact label vocabulary and this report's evidence classes from drifting together**
   (T7‑D28): the statistics file emits `COMPUTED / REPRODUCED / BLOCKED / NOT MEASURED / NOT APPLICABLE`
   (`analyze_task7_statistics.py:75, :1216`); this report emits nine classes (§1.6). *Criterion*: a mapping
   table in the artifact README, or a legend field, so a downstream reader cannot read `COMPUTED` as
   `MEASURED`.
5. **Declare the pair pools everywhere they are missing.** `CA‑10`: `verification_threshold_audit.json` has
   four rows and no counts. *Criterion*: every FNMR/FMR/EER figure in every artefact carries
   `n_genuine`/`n_impostor` and `population`, so §6.3 and the audit can be compared at all.
6. **Property/metamorphic tests for the metric code** (§7.7 `NOT TESTED`): permute gallery order, duplicate a
   query, scale‑invariance of cosine. Today the only cross-implementation check is the canonical‑vs‑per‑query
   mAP equality (difference exactly 0.0, CA‑04).

### 11.6 Category 6 — optional improvements

* Add a Tier‑N producer test for `ReidSignature` (`§10.3`'s correction): stub `edgesync_msgs.msg.ReidSignature`
  in `test_identity_node_transreid_integration.py:147-151` so `identity_node.py:302` stops taking the
  `unavailable` branch, and assert the published vector/unit‑norm/`quality`/`stamp_utc_ns` fields. Cheap, and
  it converts one `NOT TESTED` cell into real coverage without any ROS 2.
* Deduplicate the 16 projection `.npz` copies (§3, T7‑D04) down to the four digests actually referenced.
* Retire or document the `reid_model` legacy alias (`reid_launch_defaults.py:69-74, :305-330`; T7‑D11) once
  the external guide stops using it.
* Make `osnet_trt_extractor.py:71-74` `_execute_trt` raise instead of returning `zeros(512)` behind a
  "Simulated or actual CUDA buffer exchange" comment — currently a placeholder that is contract-legal and
  information-free (§10.9).
* Replace the `getattr(…, -1)` idiom at `identity_node.py:614-618` with an explicit typed property so the
  next renamed attribute fails loudly rather than printing `-1`.

### 11.7 Category 7 — decisions requiring supervisor or team approval

| Id | Question | Why it cannot be answered inside this report | Consequence if it stays open |
|---|---|---|---|
| **G.1** | Identity stage's share of `t_txn_ms: 886` | It is a budget assignment, not a measurement (`timing_budgets.yaml:15, :11`) | H7b undecidable even with device data (A‑2) |
| **G.3** | The 8,192 MB envelope: which memory quantity, under what concurrency | CA‑12 shows the recorded metric ranks the smaller model higher; only the owner defines the metric | H7c undecidable; §13 keeps refusing any deployment ranking |
| **R‑34** | Which experiment owns cross‑robot handoff metrics | Task 7's harness disclaims it: `benchmark_transreid_jetson.py:368` `'cross_robot_handoff': 'NOT MEASURED and NOT inferable…'` | The charter's motivating claim stays unowned |
| **G.5 scope** | May `IdentityVerify.srv` ever carry a camera field? | G.5 DECIDED only for the per-robot `reid_camera_id`; the frozen `.srv` has no such field (§4.4) | R‑11's service half stays `NOT TESTED`/`BLOCKED` |
| **G.7** | Grant Jetson Orin Nano access | Recorded `DENIED FOR THIS SESSION` (`…REV02.md:20`) | Levels 8‑10 stay `BLOCKED`; nothing in §6.1‑§6.8 becomes a device result |
| **G.6** | `NOT OBTAINABLE` per the proposal gate table | Unobtainability is itself the finding | Record as permanently open, or the report is silently promising a number no one can produce |
| **Fit re-run permission** | May `fit_svd_projection.py` execute into a scratch copy? | §2.5 and charter §3A forbid this pass writing into any tree | R‑14 reproducibility remains `INHERITED` (§11.5 item 1) |
| **Charter §3B repair** | May the OSNet baseline's `torchvision::nms` failure be *fixed* rather than recorded? | The manifest records `osnet_factory_attempt.result` = "FAILED — recorded, not repaired (charter §3B)" | The paired-comparison arm never exists; H7a stays unanswerable (§10.4) |
| **pytest authorisation on a ROS 2 host** | Approve re-running the two suites after A‑6/level 5 | §7's numbers came from **copies** of the packages run on a host without ROS 2; an installed-tree run is a different validation level and needs the owner's go‑ahead | §7 numbers stay Tier U/N copies, not installed-tree results |

### 11.8 What this report explicitly does **not** propose

No action in this section may be read as "run it again to get a better number". Concretely excluded, per the
standing constraints: refitting the projection to change mAP, regenerating embeddings to change Rank‑1,
retuning τ to lower FNMR (A‑4 measures a calibration need; it does not authorise changing the frozen τ without
a decision), altering benchmark settings to improve convergence or latency figures, editing a test to make a
failing bringup test pass, or upgrading this host's packages to make an uncollectable file collect. Each of
those would silently convert an `INHERITED` finding into a fabricated `MEASURED` one.

### 11.9 Where each action was first cited

A‑1 §9.10/§10.9 · A‑2 §8.5/§9.10/§10.11 · A‑3 §9 R‑31 · A‑4 §6.3/§9 R‑24/§10.11 · A‑5 (merged into A‑2's
G.3 row) · A‑6 §3/§9 R‑06/§10.4/§12.2 · A‑7 §6.6/§9 R‑09/R‑38 · A‑8 §6.4/§9 R‑27 · A‑9 §9 R‑12/R‑14 ·
A‑10 §6.8/§6.9 CA‑06/CA‑08 · A‑11 §9.1 · A‑12 §9.3/§11.1 · A‑13 §8.6/§9.5/§10.8 · D02‑fix §8.5/§10.7 ·
T7‑D24 §8.4/§10.10 · T7‑D30/D31/D32 §9.1/§9.3/§9.5.

## 12. Evidence index and reproducibility

### 12.0 How to read this index

Four rules, then the tables.

1. **Two states only.** Every entry is either **physically present in the package** (§12.1) or **referenced
   but not packaged** (§12.2). "Present" means a file at the exact path given, re-listed in this pass.
   "Referenced" means some packaged artefact names it; that does **not** make it evidence available to a
   reviewer who unpacks this folder.
2. **Prefixes are this report's**, defined in §3: `SNAP`, `ASSET`, `CFG`, `RES`, `REP`, `MAN`, `TST`, and the
   package-level `VALIDATION/`. Absolute paths are given where a reviewer would otherwise have to guess.
3. **Digests are quoted only as they appear in packaged artefacts.** Where this pass re-hashed a file it says
   so; where it did not, it says `declared` and names the declaring artefact. Nothing here is a hash this pass
   invented.
4. **Size/count figures in §12.1.8 are `MEASURED` from `VALIDATION/fresh_evidence/HASH_VERIFY_20261021.json`**
   (86,103 B, `started_local` 2026‑10‑21 09:12:59, `elapsed_s: 15.2`), which is the packaging run's own
   integrity record, not this report's.

### 12.1 Evidence physically present in the package

#### 12.1.1 Design, proposal and reporting documents — `TASK07/reports/` (21 entries)

| Path | Size (B) | What it is | Evidence class for this report |
|---|---:|---|---|
| `REP/ES-EXP07-TASK7-PROPOSAL-01-REV02.md` | 67,939 | Authoritative proposal: gates table `:13-21`, H7 `:41`, per-axis rules `:185-187`, verdicts `:195-197`, measurement plan `:145, :149`, G.1 consequence `:30` | `INHERITED` (re-read in full this pass) |
| `REP/ES-EXP07-TASK7-PROPOSAL-01-REV01.md` | 21,351 | Superseded proposal; kept because CA‑06's "1,000 calls" claim originates there | `INHERITED` |
| `REP/ES-EXP07-TASK7-REPORT-01-REV02.md` | 52,908 | Authoritative results report; 348 lines, `<fill>` count **109** (`MEASURED`), supersession `:5`, deviation notice `:7`, pre-registration `:15-21`, absent-artefact pointer `:289` | `INHERITED`, partly contradicted by disk (§6.9) |
| `REP/ES-EXP07-TASK7-REPORT-01-REV01.md` | 10,123 | Earlier results report; 131 lines, `<fill>` count **110** (`MEASURED`) — i.e. a template that was never filled | `INHERITED` |
| `REP/TASK7_PROPOSAL_RESULTS_CONSISTENCY_AUDIT.md` | 38,038 | The conflict register CA‑01…CA‑17 + §9 self-disclosed tooling defects + §10 may/may-not-quote lists | `MEASURED` where re-read this pass |
| `REP/ES-EXP07-AUDIT-FORENSIC-01.md` | 101,495 | Forensic audit: assignment restatement `:89`, embedding timing `:330` ("≈18/≈19/≈28 ms per image … **not** latency-harness numbers") | `INHERITED` |
| `REP/ES-EXP07-EXECUTION-AND-CHANGE-LOG.md` | 120,496 | Chronology of every TR/B/CA action, incl. TR‑01/TR‑03/TR‑05/TR‑06 and B‑16/B‑17/B‑19 | `INHERITED` |
| `REP/ES-EXP07-TRANSREID-PREFLIGHT-REPORT.md` | 34,373 | Pre-flight gate evidence | `INHERITED` |
| `REP/JETSON-RUN-SHEET-PREP-01-REV01.md` | 28,413 | The run sheet: `dt_id_ms` `:58`, H7b decision rule `:142`, §4.2 memory method, §6 G‑c/G‑d | `INHERITED`; carries the CA‑06 mislabel |
| `REP/ES-EXP07-TASK7-INTEGRATION-IMPLEMENTATION-REPORT.md` | 21,401 | Integration implementation record | `INHERITED` |
| `REP/ES-EXP07-TASK7-INTEGRATION-VERIFICATION-REPORT.md` | 23,995 | Integration verification record | `INHERITED` |
| `REP/TASK7_DATA_PROVENANCE.md`, `REP/TASK7_WEIGHTS_PROVENANCE.md` | 14,965 / 13,677 | Split and weights provenance (CA‑15's training-exposure evidence) | `INHERITED` |
| `REP/artifact-inventory.md`, `REP/evidence/artifact-inventory.json` | 15,245 / 28,032 | Sizes + digest prefixes, incl. `fastdds_profiles.xml` 4,753 B `7f636a3b31886500…` at `:40` and `timing_budgets.yaml` 2,031 B `65dc40f7214fab73…` at `:43` | `INHERITED` (only mention of the DDS profile in prose) |
| `REP/test-execution-log.md` | 13,069 | E‑1…E‑8 environment records, T‑1 exit 2 / 0 tests, T‑2 56 passed exit 0 | `INHERITED`, partially superseded by §7 |
| `REP/audit-summary.json` | 28,032 | Machine summary of the 2026‑10‑19 audit | `INHERITED` |
| `REP/lab-readiness-checklist.md` | 13,827 | Unexecuted checklist; B‑6 (`:45`) FastDDS two-host discovery, C‑8 (`:62`) budget fit | `INHERITED`; **every item still open** |
| `REP/audit_20261019_141628/` | dir | `exp07_forensic_audit_report.md` (`:354` CHECK‑EDGE‑03), `model_asset_inventory.md` (`:71` budget digests), `requirement_traceability_matrix.md` (`:41` REQ‑23 UNVERIFIED) | `INHERITED` |
| `REP/preflight_probe.py`, `REP/transreid_probe2.py`, `REP/evidence/` | 16,689 / 11,846 / dir | Audit-side probes; one of only four importers of `osnet_backbone.py` (`preflight_probe.py:74`) | `IMPLEMENTED`, outputs `INHERITED` |

Also present at the task root: `TASK07/README_TASK07.md` (13,676 B) and
`TASK07/TASK07_STATUS_AND_BLOCKERS.md` (8,136 B), both listed among the handoff-authored documents in
`HASH_VERIFY_20261021.json → uncovered_delivered_files` — they have no original in the source tree, so they
are packaging products, not project evidence.

#### 12.1.2 Source files, configurations and tests

Enumerated exhaustively in §3 (per-file size, digest where applicable, role, write status) and in §7.3
(per-test-file counts). Counts re-verified in this pass by `find`:

* `SNAP/` = 3 packages — `edgesync_bringup`, `edgesync_identity`, `edgesync_msgs`.
* Identity test files: **16** in `TST/edgesync_identity/tests/` and 16 in the snapshot (same set).
* Bringup test files: **3** in the snapshot (`__init__.py` plus `test_exp7_config_overlay.py`,
  `test_reid_launch_defaults.py`, `test_robot_bringup_launch.py`) but only **2** were copied to
  `TST/edgesync_bringup/tests/` — `MEASURED`. The uncollected launch test therefore cannot be run from the
  test copy at all; it exists only under `SNAP/`. That is a packaging inconsistency, recorded here rather
  than invented into §7's arithmetic.
* `SNAP/edgesync_bringup/config/` = **8** files; `TASK07/configs/` = **2** files (see §3 alias note).
* `ASSET/` = 7 model modules + `edgesync_identity/package.xml`; `assets/edgesync_bringup/` = 1 OSNet
  checkpoint + 8 projection `.npz`.
* `edgesync_msgs`: 13 `.msg` + 1 `.srv` + `CMakeLists.txt` + `package.xml` (`§10.6`), i.e. the full IDL
  surface present as **text**, never compiled.

#### 12.1.3 Model assets physically present, re-hashed in this pass

| Path | Size (B) | SHA‑256 | Status |
|---|---:|---|---|
| `assets/edgesync_bringup/models/osnet/osnet_x1_0_market1501.pth` | 10,399,605 | `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154` | `MEASURED` this pass; equals the value declared in `RES/embeddings/task7_embeddings_manifest.json → assets.osnet_checkpoint` |
| 8 projection `.npz` under `assets/edgesync_bringup/models/{transreid,projections}/` + 8 under `SNAP/…/models/` | — | **16 copies → 4 distinct digests** (`dc2b9072850c299d…` ×2 names, `9626bea5526df5b0…` ×2, `628b1dc6…`, `c9bca759…`) | `MEASURED` this pass (all 16 re-hashed); variance ratios 0.9864118099212646 vs 0.9857138991355896 → T7‑D04 |

The TransReID checkpoint itself is **not** here (§12.2), so nothing in this table can run the TransReID arm.

#### 12.1.4 Execution evidence produced during packaging — `VALIDATION/`

| Path | Size (B) | Content |
|---|---:|---|
| `VALIDATION/fresh_evidence/FRESH_RUN_RECORD.json` | 14,352 | 16 recorded runs + `blocked_validation`; `captured_local` 2026‑10‑21 07:54:08; policy line quoted verbatim in §12.4 |
| `VALIDATION/fresh_evidence/identity_copy_run.txt` | 5,022 | 360 passed / 15 skipped / 1 collection error / exit 1 |
| `VALIDATION/fresh_evidence/identity_copy_run_skipreasons.txt` | 7,438 | the reason strings behind all 15 skips (§7.5) |
| `VALIDATION/fresh_evidence/bringup_copy_run.txt` | 5,391 | 55 passed / 2 failed / 1 collection error / exit 1 |
| `VALIDATION/fresh_evidence/coordination_copy_run.txt` | 2,743 | Task 13 package, 360 / 0 / 0 / 3 errors — **context only**, not Task 7 evidence |
| 10 × `perfile_test_*.log` | 101‑2,242 | per-file executions; **7 belong to Task 7** (analyze_task7_statistics, evaluate_offline_population, exp7_offline_exp_id, extractor_factory_registry, gate1_scaffolding, identity_node_transreid_integration, transreid_extractor_guardrails), matching §7.3's "7 of 16" |
| `VALIDATION/fresh_evidence/HASH_VERIFY_20261021.json` | 86,103 | integrity record — see §12.1.5 |
| `VALIDATION/HASH_AND_INTEGRITY_REPORT.md`, `TEST_EXECUTION_EVIDENCE.md`, `VALIDATION_SUMMARY.md` | 5,687 / 14,481 / 8,287 | prose summaries; §8 already corrects `TEST_EXECUTION_EVIDENCE.md:87-92`'s "five test files … cannot be collected" against its own follow-on sentence |
| `VALIDATION/inherited_evidence/exp07_20261019/` | 5 files | `FINAL_identity.log` (55,670), `FINAL_bringup.log` (5,170, the 57-passed custom runner), `FINAL_change_inventory.json` (7,707), `FINAL_protected_assets.json` (3,251), `scope_rehash_after_task7close_20261019.json` (3,301) |
| `VALIDATION/inherited_evidence/exp13_20261021/identity_suite_excluded.txt` | 4,091 | the "372 passed, 3 skipped" record reconciled arithmetically in §7.2 |

#### 12.1.5 Package integrity as measured, including its one honest exception

`HASH_VERIFY_20261021.json`, `MEASURED` by reading the file:

* **originals** — 327 rows, 305 re-hashed, **305 digest matches**, `mismatch: []`, `unreadable: []`,
  34 `out_of_repo`, 22 `generated_rows`;
* **copies** — 453 entries, 453 re-hashed, 452 match their stored digest, 431 match the current original; the
  single non-match is declared `self_referential` (the file's own digest cannot be stored inside itself;
  stored `23ccd0d2473c77dd…`), with `mismatch_stored: []`, `mismatch_original: []`, `unreadable: []`;
* **protected** — 51 baseline rows + 3 other-team rows; **54 re-hash equal to the archive digest**, 0 differ,
  0 not in archive — the strongest available statement that the frozen charter surface (§3A/§3B) was not
  touched;
* **package tree** — 468 files / 61,669,303 B, of which `TASK07` = **206 files**; by extension `.json`
  5,234,382 B, `.py` 2,725,528 B, `.md` 1,402,980 B, `.yaml` 81,252 B, `.csv` 329,633 B.

`CALCULATED`: 453 copies − 431 matching their original = 22 = exactly the `generated_rows` count on the
originals side, so every copy that does not match a live original is a file the handoff itself created. The
accounting closes with no unexplained residue.

#### 12.1.6 Experiment outputs and provenance — `TASK07/experiment_results/`

| Path | Size (B) | What it evidences | Class |
|---|---:|---|---|
| `RES/embeddings/task7_embeddings_manifest.json` | 7,352 | 19,281 embeddings, the `directory` population, per-image parity at 2e‑05, host identity, the failed OSNet attempt (§10.4) | `INHERITED` (arrays absent) |
| `RES/embeddings/task7_transreid_{512_agnostic,512_sie,768_agnostic}_exclusions.json` | 37 each | `{"gallery": [], "query": []}` — **no image excluded from any arm** | `INHERITED`, verified by reading |
| `RES/results/experiment7/gate3_real_results.json` (+ `_v2_corrected`, `_v3_reproduce`, `_reconciled`) | 1,557 / 3,247 / 1,667 / 2,263 | the four recorded arm variants; source of the 0.711549 ratio and CA‑01/CA‑03 | `INHERITED`, cross-checked this pass |
| `RES/results/experiment7/benchmark_latency_memory.json` | 1,125 | the only latency/memory pair on disk (laptop; §10.5b) | `INHERITED` |
| `RES/results/experiment7/verification_threshold_audit.json` | 1,542 | 4 τ = 0.7 rows: FNMR 18.70/42.84/48.66/38.51 %, `eer_tau` 0.555/0.387/0.33/0.399 | `INHERITED`; **producer not on disk** (§12.2) |
| `RES/results/experiment7/forensic_ablation_results.json` | 2,180 | CA‑04's second mAP implementation (D1 = 51.46717615365994) | `INHERITED` |
| `RES/statistics/task7_statistics.json` / `.md` | 63,412 / 5,298 | the §6.1‑§6.5 engine output incl. `labels_legend` (`:75, :1216` vocabulary) | `INHERITED` |
| `RES/statistics/task7_population.json`, `analyze_all_arms.log`, `analyze_arm1_512agnostic.log` | 7,676 / 1,013 / 612 | population/pair bookkeeping and two analyzer logs | `INHERITED` |
| `RES/provenance/task7_provenance_measurements.json` | 17,275 | projection digests + `_relationships`, checkpoint digests | `MEASURED` (re-hashed against it this pass) |
| `RES/onnx/transreid_onnx_report{,_rerun}.json` | — | the two-file parity pair (§10.5a): identical digests, identical worst values, differing export seconds | `INHERITED` |
| `RES/trt_blocked/probe_build_report.json`, `build_run2.log`, `bench_run2.log` | 1,812 / 385 / 202 | the executed **refusals** — the only device-adjacent evidence in the package | `INHERITED` (level 8) |
| `RES/execution_evidence/preflight_probe.json`, `transreid_probe2.json`, `pytest_baseline_prefix.txt`, `dataset_manifest_query_gallery.npy` | 14,653 / 11,641 / 1,725 / 3,983,184 | probe outputs and the gallery/query manifest array (present as `.npy`, unlike the embeddings) | `INHERITED` |

#### 12.1.7 Manifests and traceability — `TASK07/manifests/`

`MAN/requirement-traceability-matrix.csv` (15,600 B, 42 rows × 9 columns — all re-read and re-derived in §9),
`ES-EXP07-FINAL-DELIVERABLE-MANIFEST.md` (39,534 B), `ES-EXP07-TASK7-INTEGRATION-CHANGE-MANIFEST.md`
(17,005 B), `ES-EXP07-TASK7-INTEGRATION-HANDOFF-CHECKLIST.md` (6,871 B), `FINAL-DELIVERABLE-MANIFEST.md`
(4,919 B). §9.8 is the ledger of where the CSV's verdicts no longer match the code.

### 12.2 Referenced but **not** packaged

**12.2.1 Externally referenced, verified by digest at packaging time, absent from this folder.**
All nine rows come from `HASH_VERIFY_20261021.json → reference_only`, all `exists: true`, all `match: true` at
2026‑10‑21 09:12:59 — so they are *recoverable and identified*, but a reviewer holding only this package
cannot open them.

| Reference | Bytes | SHA‑256 | Named by |
|---|---:|---|---|
| `…/edgesync_bringup/data/Market-1501-v15.09.15` (directory) | 145,885,626 across **68,036 files** | not hashed (directory) | every §6 arm; `run_real_pipeline.py:33` hard-codes an absent OneDrive path (CA‑17) |
| `…/models/transreid/transformer_120.pth` | 414,724,314 | `843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5` | `RES/onnx/*`, embeddings manifest, `algorithm_thresholds.yaml:46` |
| `…/edgesync_identity/tests/fixtures/synthetic_transreid.pth` | 414,718,880 | `2e518f28bb5ee5019329f0a96647300fd90a47bae218cb8313bfc0960180658c` | `test_gate1_scaffolding.py:326-347`; regenerable only by running `generate_test_fixtures.py`, whose default output dir is `tests/fixtures/` (`:237, :243`) — a **write into the tree**, so not performed here |
| `…/embeddings_full_directory/task7_transreid_512_agnostic.npz` | 37,094,828 | `cb4c8b8654ff410a12bf379ce14e01bd000efba20408c2ff63dccd6f14e671b0` | §6.1‑§6.5 arm 1 |
| `…/embeddings_full_directory/task7_transreid_512_sie.npz` | 37,095,723 | `d2d50a8ed50fbb54b46b0229241280e8bc9f38c40c53c2a87c1239503f9ef144` | §6.1‑§6.5 arm 2 |
| `…/embeddings_full_directory/task7_transreid_768_agnostic.npz` | 55,193,927 | `459cfbb31236d558636813d59484e626dfc7e631af752e2596599a439d226ac2` | §6.1‑§6.5 arm 3 |
| `…/onnx_trt09/transreid_feature_agnostic.onnx` | 343,380,183 | `318b6857585d5d7c4882a42156a7080c11be37f84a53957b05fb1b0a9aa82b22` | `RES/onnx/*`, `RES/trt_blocked/probe_build_report.json → inputs` |
| `…/onnx_trt09/transreid_feature_sie.onnx` | 343,399,114 | `2cd8ccc774c5933771e5b6b653d593de3862faddfd59b471b9cecccf4bda4362` | `RES/onnx/*` |
| `…/onnx_trt09_rerun/transreid_feature_agnostic.onnx` | 343,380,183 (same digest) | as above | `RES/onnx/transreid_onnx_report_rerun.json → graph_validation` |

**12.2.2 Referenced by a packaged document and **not found anywhere**, this pass.** Verified by `find` /
`grep` over the whole of `TASK07/`; each row states what was searched. These are the gaps this report refuses
to fill with inference.

| Missing item | Cited by | Search result | Consequence |
|---|---|---|---|
| `results/experiment7/task7/` artefacts | `REP/…REPORT-01-REV02.md:289` | `find experiment_results -type d` → no `task7/` subdirectory | the authoritative report points at a directory that does not exist; its tables stay `INHERITED` with an unresolved pointer |
| `phase8` / `phase9` latency medians | historically cited tables (CA‑08) | `find -iname "*phase8*"`, `*phase9*` → **0 hits** | two of six historically cited medians have no artefact → A‑10 |
| `profile_edge_latency.py` (the +18.0 ms fudge tool) | RTM R‑39, B‑13 | `find` → **absent** | the `REQ-23`-era finding B‑13 cannot be re-verified → T7‑D18 stays `UNRESOLVED` |
| producer of `verification_threshold_audit.json` | `grep -l verification_threshold_audit` over all packaged `.py` → **0 files** | the four FNMR rows have no attributable generator and no pair counts (CA‑10) → A‑4 |
| `ablation_task7_*.yaml`, `task7_identity_ablation.launch.py`, `tools/ablation/task7/run_task7_benchmark.sh` | proposal/run-sheet ablation plan | `find` → 0 hits each | the ablation arms cannot be reproduced |
| `bgr-order-run.json`, `es07_audit/` | inherited conflict discussion (CA/B‑0x threads) | `find` → 0 hits each | the *specifically named* run file and audit folder are absent — but the substantive channel-order evidence **is** packaged as `REP/evidence/channel_order_diagnostic.json` (see §12.1.5a and §6.10.2), so this gap is naming, not data |
| the external team guide that uses `reid_model` | T7‑D11 | not in the package | the alias cannot be retired by editing a file we do not have |
| `.engine` / `.plan` / `.trt` | `algorithm_thresholds.yaml:48` `reid_engine_path: ""` "no engine exists on disk (device-side, UNVERIFIED)" | 0 hits | level 8 starts from nothing, not from a stale engine |
| `tests/fixtures/` directory itself | `test_gate1_scaffolding.py:327` | absent (only the generator script is present) | the 8 gate1 skips persist on any clean unpack → §12.3 |
| ROS 2 runtime (`rclpy`, `launch`, `sensor_msgs`, `ros2` CLI) | §10 preamble | unimportable this pass; `rclpy` `ModuleNotFoundError` at `tests/test_identity_service.py:11` | levels 5‑7 unexecutable here |

### 12.3 Reproducibility, result by result

| Result / claim | Producing command, as recorded | Inputs present in package? | Reproducible in this package? |
|---|---|---|---|
| 360 passed / 15 skipped (identity) and 55 / 2 (bringup) | `python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors`, cwd inside the copied package | **yes** | **yes** — executed on 2026‑10‑21, logs in `VALIDATION/fresh_evidence/`. 15 skips are asset-dependent and 2 bringup failures are the checkpoint (§7.5/§7.6) |
| 68 / 28 / 24 / 22+1 / 24+8 / 20 / 17+1 per-file results | same, one file at a time | yes | **yes**, and logged per-file for 7 files |
| `test_identity_service.py` | — | code present | **NO** — uncollectable without `rclpy`; 0 % executed on every host in this evidence set |
| 19,281 embeddings; Rank‑1/mAP/FNMR arms | `run_real_pipeline.py` → `evaluate_offline.py` → `analyze_task7_statistics.py` | **no** (checkpoint, dataset, `.npz` all §12.2.1) | **NO** offline here; and charter §3A forbids regeneration. Promotable only by A‑6 digest match |
| ONNX export + parity verdicts | `scripts/export_transreid_onnx.py --parity-n 8 --reference-device cpu` (argv recorded in `…_rerun.json → invocation`) | **no** (checkpoint + dataset + graphs external) | **NO** — but the two existing reports agree to the last digit and share graph digests, which is the strongest reproducibility evidence in §6 |
| TRT engine build / device latency / memory Δ | `build_transreid_trt_engine.py`, `benchmark_transreid_jetson.py --iterations 200 --repeats 3 --warmup 20` | harness yes; device **no** | **BLOCKED** by design — the scripts refuse off-target (§10.9). Reproducing them here would reproduce a refusal |
| Projection digest families, variance ratios | `fit_svd_projection.py` | `.npz` yes; training partition no | **NO** — re-running writes into the tree (§2.5, charter §3A) → A‑9/§11.5 item 1 |
| §9's 42-row verdict table | read + `grep`/`find` + `--collect-only` against packaged source | yes | **yes, as an inspection**: it can be re-derived from these files, but it is not a measurement of runtime behaviour |
| §7.2's 375 / 57 collected counts | `pytest --collect-only -q` per package | yes | **yes** — executed in this pass; the output is transcribed in §7 rather than filed as a new log (§2.3) |
| This report's arithmetic (ratios, sums, differences) | shown inline with `CALCULATED` labels | — | **yes**, each derivation is printed next to its result |

Three honesty notes required by the brief: a reported value is **not** reproducible merely because an artefact
prints it (`transreid_onnx_report.json` prints `seconds: 5.3` and its rerun prints `9.9` for the same
deterministic graph); a passing suite is **not** a passing system (§10.3); and where two packaged artefacts
disagree — CA‑01's mAP at the 1e‑10 pp level, CA‑04's 6.0588e‑05, CA‑07's sign flip, CA‑11's three values for
one arm — this report preserves both figures and marks the conflict `UNRESOLVED` instead of choosing the
flattering one (§8.6, §13).

### 12.4 What this pass wrote, and what it ran

* **Written: exactly one file so far — this report**, in place, at
  `C:\Users\PC\edgesync_handoff_20261021\EdgeSync_My_Tasks_Handoff\TASK07\EXECUTIVE_SUMMARY_TASK07.md`, and
  subsequently `TASK13/EXECUTIVE_SUMMARY_TASK13.md`. No report was created, renamed, moved or deleted; no code,
  config, dataset, weight, test, log or result file was touched (§2.3/§2.4/§2.5).
* **Executed in this pass:** read-only inspection (`Read`/`grep`/`find`/`ls`/`awk`), digest recomputation on
  packaged `.npz` and the OSNet checkpoint, `pytest --collect-only` against the two copied packages, and
  environment probes for `colcon`/`ros2`/`rclpy`/`launch`/`sensor_msgs`. **No test suite, experiment,
  benchmark or build was re-run in this pass**, and this report does not claim one. Outputs of the collect-only
  and probe steps are transcribed inline because §2.3 forbids filing new evidence files.
* **Executed earlier, and cited as such:** the suite and per-file runs dated 2026‑10‑21, whose own policy line
  reads: "Every row below was executed during this packaging run against the COPIES inside this handoff
  package. Nothing was installed system-wide, nothing in the project tree was written, and no historical
  experiment artifact was overwritten." (`FRESH_RUN_RECORD.json → policy`).
* **Deliberately not touched:** `C:\Semester 7\EdgeSync-main (3)Last.zip` (older version, banned as a source)
  and the project tree, which is outside this pass's authority. Where a `reference_only` row in §12.2.1 quotes
  a path inside the project tree, that path was read **by the 2026‑10‑21 packaging run**, not by this pass.

### 12.5 Named things that can never be re-verified from this package

For completeness, so that no future reader mistakes silence for evidence: **B‑13** (the +18.0 ms fudge in the
absent `profile_edge_latency.py`), **the OSNet "94.32897862232778 %" baseline** (§6.10 — quoted from
literature/inherited records, never produced by packaged code on this host), **the `RES`-only claims about
`phase8`/`phase9`** (CA‑08), **any statement about the 2026‑10‑19 `ros2`/`colcon` exit‑127 result being still
true** (this pass measures colcon **present**), and **the training provenance of `transformer_120.pth`**
beyond what `TASK7_WEIGHTS_PROVENANCE.md` and CA‑15's exposure check can support. Each is recorded
`UNRESOLVED` or `INHERITED` at its first appearance, and §13 repeats the ones that change the verdict.

<!-- CONTINUE-TASK07-SECTION-13 -->



