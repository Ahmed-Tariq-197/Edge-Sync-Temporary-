# ES-EXP07 ↔ ES-EXP13 — Ownership-Scoped Integration Audit, Minimal Fixes and Final Validation

Pass id: `20261021_001417` (local, EDT). Generated 2026-10-21 00:14 local / 2026-10-20 21:01 UTC.
Auditor scope: Experiment 7 (TransReID Re-ID) and Experiment 13 (CBAA handoff), plus the files needed
to integrate them.
Host: Windows-10-10.0.26300-SP0, AMD64, Python 3.10.11, pytest 9.0.3, torch 2.5.1+cu121,. **The repository is not a Git working tree**, so every scope claim below rests on
SHA-256 comparison against a pre-edit snapshot, not on `git diff`.

Evidence produced by this audit lives outside the repository:

| file | what it is |
| --- | --- |
| `C:/Users/PC/edgesync_int_backup/pre_audit_20261020_233257/` | bytes of every file before this audit edited it, + `BACKUP_MANIFEST.txt` digests |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/change_scope_20261020_211055.txt` | protected-file and change-scope verification, executed → `RESULT: PASS` |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/mutation_check_20261020_211049.txt` | 9 mutations of the new guards, each killed by a test → `RESULT: 9/9 mutations killed` |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/collection_errors_20261020_210322.txt` | the real cause of all 11 collection errors |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/capability_evidence_20261020_205834.txt` | host capability probe, executed |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/probe_exp7_inference_console.txt` | real TransReID forward pass through the production factory |
| `C:/Users/PC/edgesync_int_runs/audit_20261020/workspace_tests_20261020_210136.txt` | per-package test sweep across all 8 packages |

Historical evidence from earlier passes was read, not overwritten:
`docs/experiments/exp13/` (6 reports, pass `20261020_210721`), `docs/experiments/exp07/{audit,execution}`,
`C:/Users/PC/edgesync_es13_runs/completion_20261020/`, `C:/Users/PC/edgesync_es07_runs/`.

---

## A. Executive verdict

| area | status | the one sentence that justifies it |
| --- | --- | --- |
| Experiment 7 standalone | **PARTIAL** | The inference chain is genuine and re-executed here (real 414 MB checkpoint → 512-D unit-norm embedding → Exp-7's own guard passes), and 372 tests pass; but its two headline evaluation claims are not supported by the evidence on disk (§B F-06, F-07, F-08) and no target-hardware number exists. |
| Experiment 13 standalone | **PARTIAL** | 336 tests pass at Tier-N, the CBAA arm is wired end-to-end and its guards are mutation-tested; but the benchmark does not compare the two node arms and its dual-ownership metric is a self-consistency check (§B F-01, F-02), so protocol correctness is established at unit level only. |
| ES-EXP07 ↔ ES-EXP13 integration | **PARTIAL — real, but only inside one process and only for the leg I own** | The missing carrier is now implemented and 23 dedicated integration tests pass, including an end-to-end crop → trigger leg adjudicated by Exp-7's own guard; three of the four links in the intended flow are still outside my authorization (§B E-01…E-03). |
| ROS 2 / DDS runtime readiness | **BLOCKED** | No `rclpy`, `sensor_msgs`, `launch`, `launch_ros`, `ament_index_python`, no `ros2`, no `ROS_DISTRO`; `edgesync_msgs` has never been generated on this host (§D). |
| Hardware / deployment readiness | **NOT VALIDATED** | No Jetson, no TensorRT, ONNX providers are `AzureExecutionProvider, CPUExecutionProvider` only; every latency figure is a laptop number. |
| Overall review readiness | **READY for supervisor review as an audit; NOT ready for ROS 2 integration testing; NOT ready for hardware testing** | Everything claimed here was executed and is reproducible from the evidence paths above; the blockers are environmental and named. |

---

## B. Findings and fixes

Categories are the brief's §7: A = Exp-13 defect, B = Exp-7 defect, C = integration defect,
D = validation blocker, E = external/baseline, F = evaluation/evidence defect.

### C-01 — the handoff trigger could never carry a real reference embedding · **HIGH · Category C** · **FIXED**

- **Where:** `edgesync_coordination/nodes/coordination_node.py:133-134` (`_latest_reid_signature`
  initialised to `np.zeros(512)`), `:423-438` (`set_reid_signature`), `:475-476` (the only writer of
  `req.reid_feature_vector`).
- **Evidence (executed):** `set_reid_signature` has **zero call sites** in the whole `src` tree; the only
  production call of `broadcast_handoff_request()` is `coordination_node.py:277` with no argument, so the
  self-built path always runs and `reid_feature_vector` is the zeros it was initialised with. Before the
  fix, the integration test for a fresh signature failed exactly here (mutation **M4** reproduces the
  original defect and is killed by 2 tests).
- **Root cause:** no message type existed to carry an embedding from Exp-7 to the coordination node, and
  neither side produced or consumed one. It was an absent contract, not a bug in either experiment.
- **Fix (smallest correct form):** a new message `edgesync_msgs/msg/ReidSignature.msg`; an **opt-in**
  producer in Exp-7's `identity_node`; a consumer in the Exp-13 arm that installs the vector through the
  inherited setter, so the baseline's own publish line carries it. Nothing in a protected file changed.
- **Regression:** `tests/test_exp7_exp13_signature_integration.py` I1–I17, 23 tests, all passing;
  mutations M1–M9 all killed.
- **Remaining risk:** the leg is proven at Tier-U/Tier-N only. It has never crossed a DDS network, and the
  crop source that would feed it does not exist yet (E-03).

### C-03 — a delayed older signature could overwrite a fresher one · **MEDIUM · Category C (in my own new code)** · **FOUND BY THIS AUDIT, FIXED**

- **Where:** `exp13_coordination_node.py:_on_reid_signature`, the block that stores `_exp13_sig_state`.
- **Justification:** the consumer adopted whichever signature arrived last and compared stamps only for
  age at broadcast, never for arrival order. Two publications of one topic have no DDS ordering
  guarantee, so an old observation arriving late replaced the current one, and the next trigger carried
  an identity the camera had already moved past — precisely the stale-association failure the contract
  exists to prevent. Found while working through the brief's §6 item 8.
- **Fix:** reject an incoming signature stamped older than the held one; counted as
  `refused_out_of_order`, logged at WARN, held state untouched.
- **Regression:** `test_an_older_signature_that_arrives_last_does_not_replace_a_newer_one` and
  `test_a_duplicate_signature_is_idempotent_and_not_counted_twice_as_state`; mutation **M9** removes the
  guard and the first test fails.

### B-01 — Exp-7 had no way to turn an observation into anything anyone could consume · **MEDIUM · Category B** · **FIXED (opt-in)**

- **Where:** `edgesync_identity/nodes/identity_node.py`. `store_candidate_vector` (`:262`) and
  `set_live_target_vector` (`:335`) — the two methods that hold the smoothed signature — had **test-only**
  callers; the node's only real interface was the `IdentityVerify` service.
- **Root cause:** Exp-7 was built as a verifier that is asked questions, not as a publisher of state. That
  is defensible standalone and fatal for integration: the coordination node had nothing to ask.
- **Fix:** four parameters, all defaulting to the previous behaviour —
  `publish_reid_signature` (False), `reid_signature_topic`, `reid_target_crop_topic` (''),
  `reid_signature_target_id` (''). When enabled, a `sensor_msgs/Image` crop goes through
  `image_to_rgb` → the injected extractor (with `cam_label` routing) → `SignatureSmoother.update` →
  `is_invalid_embedding`, and either publishes a valid signature or publishes `valid=false`.
  The frozen `IdentityVerify.srv` and every existing path are untouched; `372 passed, 3 skipped, 1 error`
  both before and after the edit (§E).
- **Design point worth keeping:** an invalid observation is announced, not silently omitted. A consumer
  that never hears about the failure keeps handing over a target nobody can any longer see.

### B-02 — the fitted projection carries no provenance at all · **MEDIUM · Category B/F** · **DOCUMENTED, NOT FIXABLE RETROSPECTIVELY**

- **Evidence (executed):** `np.load` on
  `edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz`
  (1,466,587 B, sha256 `9626bea5526df5b0…`) returns exactly four arrays:
  `mean (768,)`, `projection (768,512)`, `singular_values (512,)`, `total_variance_ratio = 0.98641`.
  No identity list, no id range, no split name, no seed, no code version.
  `svd_projection_768_to_512_real.npz` and `…_v3_reproduce.npz` carry the same four keys.
- **Consequence:** the claim "the projection was fitted on IDs 1–650 of `bounding_box_train`" cannot be
  verified from the artifact, only from `TASK7_DATA_PROVENANCE.md`. The leakage question the brief asks in
  §5.2 therefore cannot be closed by inspection of the weights — it depends entirely on a markdown file.
  Confirms the earlier register's D-07 by execution.
- **No fix attempted:** refitting is forbidden by the ES-EXP07 charter, and writing metadata into a
  shipped `.npz` would silently change an artifact that existing results were computed with.

### E-01 — the deployed baseline node still ignores the signature · **HIGH · Category E (protected)** · **NOT APPLIED — patch proposed**

- **Where:** `coordination_node.py:423-438` and `:441-481`.
- **Defect:** `set_reid_signature` accepts any `np.ndarray` of shape `(512,)` whose values are finite; it
  does not reject the all-zeros sentinel, and it logs a rejection at **debug**, which is invisible at the
  default INFO level. `broadcast_handoff_request` then publishes whatever is held. A caller that never
  sets a signature publishes zeros and the receiver answers `INVALID_EMBEDDING: reference` — after a
  round trip, in the window where ownership could be handed over.
- **Why an adapter cannot solve it:** for the *Exp-13 arm* an adapter does solve it, and that is what was
  shipped. For the *deployed* `coordination_node` there is no seam — the zeros guard belongs inside the
  setter.
- **Proposed minimal patch (backward compatible, ~6 lines, requires authorization):**
  in `set_reid_signature`, after the existing shape/finite checks, add
  `if not np.any(arr): log WARN and return`; change the rejection log from `debug` to `warn`.
- **Risks of the patch:** it makes a currently-silent path noisy; any node that intentionally installs
  zeros (none exists in this repo today) would start logging. It changes no behaviour that any existing
  test depends on.

### E-02 — the receiver still cannot verify a handed-off target · **HIGH · Category E (other team's package)** · **NOT APPLIED — patch proposed**

- **Where:** `edgesync_acquisition/nodes/target_acquisition_node.py:248-262`. `center_and_verify()` is
  called from the centroid path with neither a crop nor a vector, so reference resolution reaches
  `np.zeros(512)` (`identity_node.py:374-386`, `reference_source='none'`) and the service returns
  `match=False, similarity=0.0` (`:444-449`).
- **Impact:** even with E-01 and C-01 fixed, the *receiving* side has no crop to encode, so the identity
  leg cannot close end-to-end on a real robot. This is the second half of the same missing contract.
- **Proposed follow-up (their ownership):** pass the acquired crop into `center_and_verify()`, or call
  the extractor on the region behind the centroid before verifying.

### E-03 — no node in the repository publishes a target crop · **HIGH · Category E (other team's package)** · **NOT APPLIED — patch proposed**

- **Where:** `edgesync_perception/nodes/perception_node.py:232-284` computes `primary_bbox` (ByteTrack or
  max-confidence detection) and then publishes **only** a `PointStamped` centroid. The bounding box never
  reaches the wire.
- **Impact:** Exp-7's new producer has `reid_target_crop_topic` defaulting to `''`, and
  `exp13_bringup.launch.py` ships it as `''` deliberately. With `with_identity:=true` the node arms,
  warns that nothing feeds the extractor, and can only ever publish an invalidation. **This is the single
  remaining reason the integration is not real on a robot.**
- **Proposed follow-up:** a `ReidTargetCrop.msg` (or a `sensor_msgs/Image` + ROI pair) published beside
  the centroid, from the code that already knows the bbox. Deliberately not invented here: it belongs to
  the perception team's topic contract.

### F-01 — the Exp-13 benchmark does not compare the two node arms · **HIGH · Category F** · **CORRECTED IN REPORTING ONLY**

- **Evidence (read, then verified by me at the file):**
  `edgesync_coordination/edgesync_coordination/scripts/run_exp13_unit_benchmark.py:41-42` states
  verbatim: *"It is a MODEL, not the node: CoordinationNode cannot be imported on this host because there
  is no `rclpy`."* Its only imports are `UtilityEngine`, `CBBANegotiator`, `ConsensusStatus`, `TaskInfo`
  (`:88-89`). `negotiation_protocol` is echoed into the artifact (`:880`) while `:848` states it is
  *"written into this artefact only; no selector exists in coordination_node.py"*.
- **What the arms therefore are:** `baseline` = a stdlib re-implementation of the selection logic;
  `exp13` = the real `CBBANegotiator` under a lockstep broadcast model. The node's own
  `negotiation_protocol:=cnp|cbba` selector — the thing Exp-13 is studying — is never exercised.
- **Root cause of the wrong claim:** my own earlier reports labelled the columns "`cnp` vs `cbba`", which
  reads as a live arm comparison. Corrected here; the artifact and the script are unchanged (the brief
  forbids changing a benchmark protocol after the fact).

### F-02 — "zero dual ownership" is a per-agent self-consistency check · **HIGH · Category F** · **DOCUMENTED**

- **Evidence (read):** `run_exp13_unit_benchmark.py:430-434` —
  `held = set(neg.bundle)`, `owned = {t for t, w in neg.winning_agents.items() if w == neg.agent_id}`,
  `if not held <= owned: res.dual_owner_violations += 1`, evaluated **per agent**.
- **Why that is weaker than it sounds:** `owned` is derived from the same agent's own belief about who
  holds what. Two agents can each hold the task and each believe *itself* the winner and the counter
  stays at zero. The cross-agent quantity that does exist, `participants_naming_same_winner`
  (`:421-423`), is only computed for decided episodes and is reported as agreement, not as dual ownership.
- **Action:** none against the code — a real global check would be a new measurement, not a fix, and
  re-running under a changed metric would be exactly the prohibited protocol edit. The claim is narrowed
  here and the correct statement is: *no per-agent bundle/ownership contradiction was observed in 600
  simulated episodes; dual ownership across agents was not measured.*

### F-03 — the order-independence claim is demonstrated for N ≤ 4 only · **MEDIUM · Category F** · **DOCUMENTED**

`run_exp13_unit_benchmark.py:824-825` filters both probes to `[s for s in sizes if s <= 5]`, and the size
ladder is (2,3,4,6,8), so the probes cover N ∈ {2,3,4} while the convergence table reports through N = 8.
`EXP13_BASELINE_COMPARISON_20261020_210721.md:182-183` asserts winner independence "over 24 orders"
without that bound. Narrowed claim: order-independence is demonstrated for N ≤ 4, through `run_round` only.

### F-04 — the tie defect is unreachable from the benchmark · **MEDIUM · Category F** · **DOCUMENTED**

The known merge defect requires belief ingestion while `winning_agents` is empty. Both probes enter only
through `run_round` (`:583`, `:659`), which always prices first, and the lockstep barrier swaps beliefs
only after *all* agents have acted (`:383-394`). `sequential_activation_probe` also reads
`converged[0].winning_agents` without checking that peers agree (`:661-663`). A concrete exposing input:
two agents, identical bid (axis 0.0, battery 80.0), deliver ugv_2's belief to ugv_1 **before** ugv_1's
first `run_round`; ugv_1's equal bid is then suppressed by the positive-improvement gate
(`cbba_negotiator.py:545`, `bid > winning + 1e-6`) and the winner is ugv_2 by schedule, contradicting
`_prefer`'s lexicographic rule (`cbba_negotiator.py:702-715`). This confirms and sharpens the earlier note
that *lockstep probes hide the tie defect*.

### F-05 — "exp13 converges 60%" is 3 conditions out of 5, not a measured rate · **MEDIUM · Category F** · **DOCUMENTED**

Recomputed from the raw JSON: each of the five conditions contributes exactly 60 episodes; three
converge 60/60 (`clean`, `equal_utilities`, `replayed_traffic`), `dead_participant` stalls 60/60 and
`one_silent_candidate` is undecided 60/60. The 180/300 figure is a property of the condition set, so it
must not be read as "60 % of attempts under load".

### F-06 — no valid OSNet ↔ TransReID comparison exists · **HIGH · Category F (Exp-7)** · **DOCUMENTED, NOT FIXABLE HERE**

- OSNet is frozen by charter and its arm errors on this host: `torchvision` fails to import with
  `RuntimeError: operator torchvision::nms does not exist` (reproduced in §E probe), and
  `torch.ops.torchvision.nms` raises `AttributeError`. No paired measurement exists.
- The only head-to-head numbers come from an older laptop artifact in which **OSNet wins Rank-1**
  (94.32897 / 83.56923 vs TransReID 93.37885 / 85.12520). `task7_statistics.md` measures TransReID arms
  only (sie 93.38 / 85.13; agnostic 88.75 / 77.32; McNemar H1 p = 2.18039e-41, H2 p = 8.09013e-73,
  H3 p = 0.572846).
- **Verdict:** any "TransReID improves over the OSNet baseline" claim is **unsupported** by the evidence
  on disk. What is supported: TransReID camera-conditioning (sie) beats camera-agnostic on the same
  population, with paired McNemar tests.

### F-07 — the evaluation population is self-labelled as an invalid leakage guard · **HIGH · Category F (Exp-7)** · **DOCUMENTED**

The `id_band_752_1501` band carries the verbatim note *"MEASURED AS INVALID AS A LEAKAGE GUARD"*. The
default selector is `directory`. `TASK7_DATA_PROVENANCE.md` D-02 states the documented band "does not
describe this dataset", disjointness holds only by folder (D-07, and see B-02), and there is an internal
inconsistency: D-03/`POPULATION_RULES` claim `directory` reproduces the recorded arm while
`task7_statistics.md:48` says the recorded run used the invalid `id_band`. Until that is resolved, the
leakage status of the headline TransReID numbers is **open**.

### F-08 — a recorded reason in my own earlier report is factually wrong · **LOW · Category F (self-correction)** · **CORRECTED HERE**

`EXP13_FINAL_DELIVERY_MANIFEST_20261020_210721.md:88` states there is no `colcon` on this host. The probe
found `colcon`, `cmake`, `gcc` and `g++` all present. ROS 2 itself is still absent (`rclpy`,
`ament_index_python`, `ros2`, `ROS_DISTRO`), so the **conclusion is unchanged** — an `ament_cmake` package
cannot be configured without a ROS distribution — but the stated reason was wrong and is corrected.

### A-01 — process defect found in my own audit tooling · **LOW · self-inflicted** · **FOUND AND REVERTED**

The first mutation-check run restored the two implementation files in text mode, which on Windows rewrote
the repository's LF endings as CRLF: `exp13_coordination_node.py` 50,792 → 51,701 B and
`identity_node.py` 32,275 → 32,931 B, with **content unchanged**. A text-mode digest comparison reported
"restored byte-identical" and was therefore blind to it. The byte-level workspace sweep caught it. Both
files were restored to their exact pre-mutation digests (`0309d714a8b5`, `5b0ecb4c9260`) after confirming
LF→CRLF accounted for the whole delta, the script was switched to binary writes, and the mutation check
was re-run clean. **Lesson recorded: on this host, "identical when read as text" is not "identical".**

---

## C. Integration contract (as actually implemented)

**Intended flow.** camera observation → TransReID 768-D → fitted centered SVD 768→512 → EMA signature →
carried into the handoff trigger → receiver verifies the incoming reference against its own live target.

**Realised flow (this audit):**

```
perception_node ──(PointStamped centroid only)──✗ no crop exists                      [E-03]
      │
      ▼  sensor_msgs/Image  (reid_target_crop_topic, default '')
identity_node  ── image_to_rgb ──▶ extractor.extract_embedding(crop, cam_label)
      │                            ──▶ SignatureSmoother.update  ──▶ is_invalid_embedding
      │                            valid ──▶ set_live_target_vector + ReidSignature(valid=true)
      │                            invalid ─▶ ReidSignature(valid=false)   [never a silent gap]
      ▼
/edgesync/<robot_id>/edgesync/identity/reid_signature   (absolute, per-robot, NOT in SWARM_TOPICS)
      ▼
exp13_coordination_node._on_reid_signature
      │  refuse: foreign sender_id | dim != 512 | non-finite or ‖v‖ ≤ 1e-6 |
      │          target_id != fsm.active_target_id | stamp older than the held one
      │  accept: _exp13_sig_state = {stamp_ns, target_id, model_id, quality}; set_reid_signature(v)
      ▼
broadcast_handoff_request(request_msg=None)          [coordination_node.py:441-481, inherited]
      │  _exp13_signature_usable(): none | target_mismatch | stale(age>max_age) | fresh
      │      not fresh + require_valid_reid_signature → publish NOTHING, keep the target
      │      not fresh + reason != none               → drop to zeros, count, WARN
      ▼
HandoffRequest.reid_feature_vector  = the real 512 floats            [coordination_node.py:475-476]
      ▼
target_acquisition_node → IdentityVerify.srv ──✗ has no crop of its own                [E-02]
```

**Interfaces.** `ReidSignature.msg` — `sender_id`, `target_id`, `valid`, `vector float32[512]`,
`quality float32`, `stamp_utc_ns uint64`, `model_id`. Registered additively in `edgesync_msgs/CMakeLists.txt`
(12 pre-existing types untouched, now 13). `IdentityVerify.srv` untouched and asserted compatible.

**Identity semantics.** The signature is *the sender's own current smoothed belief about the target it
owns*. It is not a person identity, not a track id and not a lease. `valid=false` is a first-class message:
the absence of an identity is announced. `target_id` is checked against `fsm.active_target_id` on arrival.
Camera metadata (`cam_label`) is used during extraction but is **not** propagated into the message, so a
receiver cannot know which camera produced a reference — an accepted limitation, not a silent one.
`quality` is carried and defaulted to 0.0 with **no writer**; nothing in either experiment scores it, so
low-confidence must not be claimed as handled (it is handled as *invalid*, never as *weak*).

**Timestamp semantics.** `stamp_utc_ns` is the observation time from the crop header (falling back to the
node clock). The consumer uses it twice: against a per-message maximum age at the instant of sending the
trigger, and as an arrival-order monotonicity check. Default `reid_signature_max_age_ms = 2000` is set
against the smoother's own horizon: `SignatureSmoother` flushes after 3 invalid frames, which at 2 Hz is
1.5 s, so a 2 s bound admits at most one flush interval of lag before a signature stops authorising a
handoff. **Verified on this host only as a clock-arithmetic behaviour under a stub clock.**

**Transaction / epoch semantics.** Unchanged from Exp-13: `header.transaction_id` *is* the episode id
(`<target_id>@e<ownership_epoch>#<attempt>`), and `header.ownership_epoch` fences beliefs from before an
abort-and-reclaim. The signature deliberately carries **no** transaction id: it describes a target, not an
attempt, so it survives a rejected negotiation and is re-adjudicated at each broadcast. That is a design
choice with a stated consequence — a rejected transaction does not invalidate an identity belief, and the
age bound is what limits how long the belief can outlive the transaction that failed.

**Ownership-transfer conditions.** A transfer still requires the full inherited handoff protocol
(Request → Response → Confirm → ConfirmAck, plus the OwnershipEvent). The CBBA belief exchange only decides
*which candidate the owner's commit step names*; `agreement_established` is the sender's own verdict
reported for the peer to record — not a quorum, not a lease, not a commitment. **Consensus is never proof
that Exp-7 identified the right person**, and nothing in the code path lets it be mistaken for that: the
belief topic is separate from the handoff topic, and the arm that wins a belief exchange still has to
publish a `HandoffConfirm` to take ownership.

**Failure behaviour.** Producer-side: bad crop → WARN + refusal counter, nothing published; extractor
exception → ERROR + `valid=false` published; smoothed embedding invalid → `valid=false` published.
Consumer-side: seven named refusal counters (§B C-01) and a hard opt-in gate
(`require_valid_reid_signature`) that makes the node withhold the trigger and keep the target.

**Verified vs assumed.**

| statement | status |
| --- | --- |
| 512-float contract matches across `.msg` → setter → `HandoffRequest` → `IdentityVerify.srv` | **VERIFIED** (static field parse + executed end-to-end test) |
| a fresh signature reaches the wire and passes Exp-7's own `is_invalid_embedding` | **VERIFIED** Tier-N executed |
| refusals for foreign sender / dimension / non-finite / zeros / wrong target / stale / out-of-order | **VERIFIED** Tier-N executed, each mutation-killed |
| `cnp` arm subscribes to nothing and still publishes zeros | **VERIFIED** Tier-N executed |
| two robots on one host do not cross signature streams | **ASSUMED** from per-robot absolute topic naming; untested without a real graph |
| a real crop source exists to drive the producer on a robot | **FALSE TODAY** — E-03, documented in the launch file |
| delivery is asynchronous, lossy, QoS-negotiated | **ASSUMED** — the stub delivers synchronously on publish, which is *stronger* than the network |
| anything about Jetson, DDS, or a running graph | **NOT VALIDATED** |

---

## D. Benchmark evidence

### Experiment 13 — what exists, and what it is

- **Artifact:** `C:/Users/PC/edgesync_es13_runs/completion_20261020/benchmark_phaseF_20261020_223131/`
  (JSON + per-trial JSONL + console).
- **Producing script, identified three ways:** `run_exp13_unit_benchmark.py` is the only script that writes
  those filenames; the console header matches its `print` lines verbatim; and the script's SHA-256 matches
  the digest asserted in the Phase-A delivery manifest.
- **Protocol as implemented:** 5 conditions × 5 participant counts (2,3,4,6,8) × 12 trials = **300 rows per
  arm, 600 total**; seed `"20261020"`, per-cell `random.Random(f"{seed}|{condition}|{n}|{trial}")`; the two
  arms are stamped from the *same* utilities, batteries, axes and arrival order (paired), though the
  baseline receives the agent list while exp13 receives the permuted order.
- **Accounting, recomputed from the raw JSON in this audit:** baseline 276 CONVERGED / 24 UNDECIDED /
  0 stalled / mean 4.2 messages / 1.0 round; exp13 180 CONVERGED / 60 UNDECIDED / 60 STALLED / mean 14.08
  messages / 55.64 ingested / 3.32 rounds. `agreed_by_all` and winner-matched equal 180 for exp13 and 276
  for baseline. **Every aggregate matches the recorded table to the digit; no rows are missing.**
- **Tier: Tier-U**, self-declared and independently corroborated. **No inflation found** in the tier
  labelling itself — the inflation was in the *arm naming* (F-01).
- **Limitations, all of them material:** it is a model vs the real negotiator, not `cnp` vs `cbba`
  (F-01); dual ownership is a self-consistency check (F-02); N ≤ 4 for the order probes (F-03); the tie
  defect is unreachable (F-04); 60 % is a condition count (F-05); wall-clock is a 0.015 s simulated tick,
  so it says nothing about real time; and it involves **no identity signal at all** — zero hits for `reid`
  or `signature` — so it neither supports nor threatens the new integration leg.
- **Verdict on the headline claim:** the CBAA arm is **unfavourable and honest** — more messages, more
  ingested beliefs, more rounds, and a lower convergence count than the baseline model, in exchange for
  `agreement_by_all` that the baseline model never achieves (0). It is not evidence of improvement on a
  robot.

### Experiment 7 — what exists

- Recorded retrieval results in `edgesync_bringup/results/experiment7/` (7 JSON files, dated 2026-10-15),
  device strings `cuda` / `NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1)`,
  `n_query=3368, n_gallery=15913, n_training_identities=751`. **These are laptop numbers on a laptop GPU.**
- Two files record verbatim that `tau` is *not* applied in the evaluating script — the reported metrics are
  threshold-free retrieval metrics. τ = 0.70 is applied in `identity_verifier.py:101` and in distractor
  rejection offline, not in the Rank-1/mAP path. Any claim that the configured threshold gates the
  benchmark result is **false**.
- `benchmark_latency_memory.json` contains no device or precision string; `trt_blocked_probe/bench/`
  records the Jetson path as blocked with the line *"a laptop benchmark is not a Jetson benchmark"*.
  Historical results were **not** modified.

### New measurement taken by this audit (standalone Exp-7 inference, laptop)

Executed through the production factory and the real config values:
`TransReIdExtractor` built in **2.72 s** from the configured 414,724,314 B checkpoint
(sha256 `843055293e160da0…`); one forward pass on a 256×256×3 `uint8` crop in **595.5 ms**; output shape
`(512,)`, finite, L2-norm **1.000000**, `is_invalid_embedding → None`; the same crop twice gives
max-abs-difference **0.000e+00**; two independent crops give cosine **0.945780** (input-dependent, so the
extractor is not returning a constant); after 3 zero observations the smoother flushes and the guard
returns *"embedding is the invalid-crop sentinel (zero norm) - not scored"* — the exact invalidation the
arm is designed to receive.
**This is a correctness probe, not a performance result.** 595 ms on an RTX 3050 laptop GPU at an
unspecified batch of 1 is not a throughput claim, and it is certainly not a Jetson number.

### Cross-experiment evaluation

No combined Exp-7 ↔ Exp-13 benchmark was created, deliberately: with no crop source (E-03) there is no
scientifically valid joint scenario to measure, and inventing one by injecting synthetic embeddings would
be a mock presented as an integration. The signature leg is validated at Tier-N instead, and the
integration's runtime cost is unmeasured — it is a `create_subscription` plus an O(512) numpy check per
message, which is an argument, not a measurement.

---

## E. Validation matrix

| # | what was tested | exact command / procedure | actual outcome | kind | evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | Exp-13 integration tests (new) | `python -m pytest tests/test_exp7_exp13_signature_integration.py -q` (cwd `edgesync_coordination`) | **23 passed in 0.30 s** | unit/Tier-N, mocked middleware | coordination test dir |
| 2 | Exp-13 package regression | `python -m pytest tests -q --ignore=…test_auto_trigger.py --ignore=…test_decision_and_receiver_services.py --ignore=…test_response_collector.py` | **336 passed in 0.73 s** (334 before the last two tests) | unit/Tier-N | same |
| 3 | Exp-7 package regression | `python -m pytest tests -q --continue-on-collection-errors` (cwd `edgesync_identity`) | **372 passed, 3 skipped, 41 warnings, 1 error in 22.25 s** — identical to the pre-edit baseline | unit | same |
| 4 | Exp-7 skips, individually | `pytest -q -rs` | `test_extractor_factory_registry.py:255`, `test_transreid_extractor_guardrails.py:105`, `test_transreid_trt_extractor.py:149` — all gated on `ES07_HEAVY=1` because each loads the 414 MB checkpoint | unit | this session |
| 5 | Collection errors, individually | `report_collection_errors.py` (11 files, one pytest run each) | 8 × `No module named 'rclpy'`; 1 × `'edgesync_coordination'`; 1 × `'sensor_msgs'`; 1 × `ImportError: cannot import name 'LaunchDescription' from 'launch'`. **None is a code defect and none was introduced here** | static/import | `collection_errors_20261020_210322.txt` |
| 6 | Whole-workspace sweep | `run_workspace_tests.sh` — all 8 packages, per package | **793 passed, 3 skipped, 11 collection errors.** Reported per package; **this is not `colcon test` and not a ROS 2 test run** | unit | `workspace_tests_20261020_210136.txt` |
| 7 | Guard strength | `mutation_check.py` — 9 guards removed one at a time, suite re-run, files restored in binary | **9/9 KILLED**, `arm=True producer=True` byte-identical, final suite 23 passed | unit | `mutation_check_20261020_2110*.txt` |
| 8 | Protected files unchanged | `verify_change_scope.py` vs `preserve/` snapshot + `SHA256SUMS.before.txt` | **7/7 IDENTICAL**; digests e.g. `robot_bringup.launch.py 45550fc4…`, `coordination_node.py 60d72037…`, `HandoffHeader.msg 24097614…` | static/hash | `change_scope_*.txt` |
| 9 | Nothing undeclared changed | same script, section [4], sweep of 4 packages since the Phase-A baseline | 14 files, **all** matched to a delivery record; extended to the whole `src` tree: only those 14, no Exp-12/14 or other-team file touched | static/hash | same + this session |
| 10 | Exp-7 assets real | `sha256` of the configured checkpoint and projection | 414,724,314 B `843055293e160da0…`; 1,466,587 B `9626bea5526df5b0…` | executed hash | probe [3] |
| 11 | Projection provenance | `np.load(...).files` | only `mean, projection, singular_values, total_variance_ratio` — **no provenance keys** | executed | probe [4] |
| 12 | Genuine inference | `probe_exp7_inference.py` | 512-D, ‖v‖ = 1.000000, finite, non-constant, deterministic; guard passes; smoother flush reproduced | **real runtime test (torch, laptop GPU)** | `probe_exp7_inference_console.txt` |
| 13 | Host capability | `probe_capability_evidence.py` | ROS 2 absent across the board; `colcon/cmake/gcc/g++` **present** (corrects F-08); `ros2` absent; `ROS_DISTRO` unset; `tensorrt` absent; torchvision import fails with the `torchvision::nms` RuntimeError; ONNX providers `Azure, CPU` only; CUDA available, RTX 3050 Laptop | executed | `capability_evidence_*.txt` |
| 14 | Message contract on disk | I1 test parses `ReidSignature.msg` and compares field-by-field; asserts 13 registered types and untouched pre-existing names | passed | static | test file |
| 15 | Launch wiring | I17 asserts `with_identity`, `IfCondition`, `exp7_transreid.yaml`, matching topic, and the deliberately empty crop topic | passed | **static only — `launch` cannot be imported, the file has never been executed** | test file |

**Explicitly not claimed:** that any ROS 2 node started; that `edgesync_msgs` compiled; that a launch file
ran; that a message crossed a network; that any number here is a Jetson number.

---

## F. Scope and preservation statement

### 1. Experiment 7 changed

| file | change | why |
| --- | --- | --- |
| `edgesync_identity/edgesync_identity/nodes/identity_node.py` | 24,591 → 32,275 B; `4def479ca2f8…` → `5b0ecb4c9260…` | opt-in signature production: 4 parameters (all default to previous behaviour), `_wire_signature_production`, `_crop_stamp_ns`, `_on_target_crop`, `_publish_signature`, one call site, one docstring bullet. Fixes B-01. |

`exp7_transreid.yaml` was read and deliberately **not** modified — enabling the producer there would
change the frozen Exp-7 default run.

### 2. Experiment 13 changed

| file | change | why |
| --- | --- | --- |
| `edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py` | 42,146 → 51,565 B; `ff54987fed0c…` → `5e2001c35dca…` | signature intake (3 parameters, `_wire_signature_intake`, `_on_reid_signature`, `_exp13_signature_usable`, `_exp13_drop_signature`, the pre-`super()` check in `broadcast_handoff_request`, 8 counters). Fixes C-01/C-03 on the consumer side. |
| `edgesync_bringup/launch/exp13_bringup.launch.py` | 8,230 → 11,774 B | `with_identity` argument (default `false`), conditionally launched `identity_node`, per-robot absolute signature topic, Exp-7 overlay passed to it, docstring corrected. |

### 3. Integration-specific additions

| file | change |
| --- | --- |
| `edgesync_msgs/msg/ReidSignature.msg` | **new**, 1,292 B — the carrier that did not exist. |
| `edgesync_coordination/tests/test_exp7_exp13_signature_integration.py` | **new**, 22,722 B — 23 tests, I1–I17. |

### 4. Shared build / registration file

`edgesync_msgs/CMakeLists.txt`: 999 → 1,204 B; `+3` lines (one path + two comment lines), appended after
`ConsensusBelief.msg`. All 12 pre-existing type names verified present and unedited; the list is asserted
at 13 entries by test I1. A demonstrated dependency: a new message type is not compiled unless
registered.

### Protected and unrelated files verified **unchanged** by digest

`robot_bringup.launch.py`, `coordination_node.py`, `fsm/ownership_fsm.py`, `fsm/epoch_manager.py`,
`protocol/negotiation_manager.py`, `decision/utility_engine.py`, `edgesync_msgs/msg/HandoffHeader.msg`
— **7/7 byte-identical** to the pre-work snapshot. No file from Experiments 12 or 14, and no other
team's package, was touched: a sweep of the entire `src` tree since the Phase-A baseline returns exactly
the 14 files above plus the 6 Phase-A files already declared in that pass's manifest.

**Deleted: nothing.** No test was weakened, no assertion removed, no failing test deleted. All Exp-7
historical result JSON, both benchmark console files and the Phase-F benchmark artifact are untouched; the
corrections in §B are new text, not edits to old measurements.

**Backward compatibility.** `publish_reid_signature` defaults to `false`, `reid_target_crop_topic` to `''`,
`with_identity` to `'false'`, `require_valid_reid_signature` to `false`; the `cnp` arm creates no
subscription and still publishes zeros (executed test). A deployment that updates the code and changes no
parameter behaves as before. The only shared file changed is additive registration.

---

## G. Remaining blockers

**I can still fix within my assignments**
1. The `cnp`/`cbba` labels in `EXP13_BASELINE_COMPARISON_20261020_210721.md:19` and the unbounded
   order-independence sentence at `:182-183` should be restated per F-01/F-03. Not done here because that
   file is a delivered artifact of an earlier pass; say the word and I will issue a dated correction
   section rather than editing it in place.
2. `reid_quality` has no writer. Either populate it from the verifier's similarity score or delete the
   field, rather than carrying an unwired float.
3. A real dual-ownership check needs cross-agent state; I can add it as a **new, separately named**
   measurement without touching the existing benchmark, if that is wanted.
4. The Exp-7 population inconsistency (F-07) is mine to resolve, and doing it properly means re-running
   the evaluation under `directory` and reconciling `task7_statistics.md:48` — a measurement task, not an
   edit.

**Requires ROS 2 access (this host cannot)**
- `colcon build` the workspace so `ReidSignature.msg` and `ConsensusBelief.msg` are actually generated —
  they have never been compiled.
- `ros2 topic echo` a real signature; verify QoS and cross-process delivery; confirm the two-robot topic
  naming does not cross-talk.
- Execute `exp13_bringup.launch.py` with and without `with_identity:=true`; the file has never run and its
  parameter names have never been accepted by a real node.
- `ros2 test`/launch-logging integration; the 11 collection errors here are all missing-ROS-2 imports and
  are expected to disappear on a ROS host.

**Requires Jetson / target hardware**
- Any latency, memory, TensorRT or fp16 number. TensorRT is not installed; the ONNX runtime here has no
  CUDA provider. Until measured on the Orin Nano, all deployment claims stay **NOT VALIDATED**.

**Requires supervisor or team approval**
- E-01: the `coordination_node.set_reid_signature` zeros-guard patch (protected file).
- E-02: acquisition passing a crop into `center_and_verify()`.
- E-03: perception publishing a target bbox/crop — **the one change without which the identity leg cannot
  close on a robot at all.**
- Any refit of the SVD projection (forbidden by the ES-EXP07 charter as it stands).

**Scientific and evaluation limitations that remain open**
- Zero dual ownership is not a safety proof, and the current benchmark could not have detected a
  cross-agent double claim anyway (F-02).
- No valid OSNet ↔ TransReID comparison exists (F-06); the TransReID population's leakage status is
  unresolved (F-07); the fitted projection's training identities are unrecoverable from the artifact
  (B-02).
- The CBAA arm's benchmark result is unfavourable on messages, ingested beliefs, rounds and convergence
  count, and favourable only on cross-agent agreement. No robot evidence exists either way.
- Everything about the new signature leg is executed **inside one Python process with a synchronous fake
  bus**. That is a real test of the contract and the guards; it is not a test of the system.

---

## H. Final completion checklist

| deliverable | state |
| --- | --- |
| Protected-file inventory, re-verified by digest against a snapshot | **COMPLETE** |
| Experiment 13 standalone audit (§4.1–4.3 requirements, test-quality review) | **COMPLETE** — carried forward from prior passes, re-verified against the current tree, and extended by F-01…F-05, which overturn parts of the earlier arm comparison claim |
| Experiment 7 standalone audit (model, embeddings, evaluation validity, deployment claims) | **COMPLETE** — inference re-executed live here; B-02, F-06, F-07 recorded |
| Integration contract audit (all 10 of §6) | **COMPLETE** — §C, each row labelled verified / assumed / false-today |
| Findings classified A–F | **COMPLETE** — §B, 13 findings |
| Category A/B/C fixes applied within scope | **COMPLETE** — C-01, C-03, B-01 |
| Category E patches | **PROPOSED ONLY — not applied, as §2.2 requires authorization** |
| Category D validations that this environment supports | **COMPLETE** — §E rows 5, 6, 13; blocker statements are per-component |
| Category F corrections without destroying evidence | **COMPLETE** — reporting corrections in this file; originals byte-unchanged |
| Focus tests + integration tests + regressions | **COMPLETE** — 23 new, 336 coordination, 372 identity, 793 workspace, 9/9 mutations |
| Change-scope verification | **COMPLETE** — 7/7 protected identical, 0 undeclared changes, LF integrity restored after A-01 |
| Benchmark re-run under the new integration | **NOT DONE — deliberately.** The existing benchmark contains no identity signal, so a re-run would reproduce the same numbers and imply an integration measurement that does not exist. |
| ROS 2 / DDS runtime validation | **BLOCKED** — no ROS 2 on this host |
| Hardware / deployment validation | **NOT VALIDATED** — no Jetson |
| End-to-end identity→handoff on a real graph | **BLOCKED** — E-03: no crop publisher exists |
| Real-robot readiness | **NOT READY**, and not claimed |

**Overall:** the Exp-7 ↔ Exp-13 identity leg is implemented, tested and mutation-verified at unit level; the
two experiments are individually more complete than their earlier reports claimed and, in three specific
places (arm comparison, dual-ownership metric, OSNet-vs-TransReID) **less** well-evidenced than those
reports implied. Nothing here is ready for hardware testing, and integration is not real on a robot until
the perception team publishes a target crop.
