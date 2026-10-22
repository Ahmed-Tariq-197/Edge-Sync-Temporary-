# Task 7 ↔ Task 13 dependency map

Built by reading the delivered source, not by reading a design document. Every arrow states the
producer file and function, the consumer file and function, the contract, the test that touches it,
the **verification level actually reached**, what is missing, and who owns the missing piece.

Levels used (`SOURCE ONLY` means the endpoints were read but no test exercised the edge):

| Level | Meaning |
|---|---|
| Tier U | Exercised by a unit/model-simulation test in one process |
| Tier N | Exercised by a node-level test against a **stubbed** `rclpy`, with message classes generated from the on-disk `.msg` |
| Tier R | Would require a real ROS 2 graph — **not reached anywhere in this package** |
| NOT WIRED | One endpoint exists, the other does not |

The explicit warning from the brief is honoured: **both endpoints existing is not evidence that the
edge works.** Nothing below was patched during packaging.

## The chain

```
camera frames -> target crop -> identity extraction -> 512-D signature -> ROS transport
  -> Exp-13 intake -> target/epoch association -> CBAA negotiation
  -> ownership / handoff protocol -> final ownership event
```

| # | Edge | Producer (file:function:line) | Consumer | Contract | Test | Level | Missing | Owner |
|---|---|---|---|---|---|---|---|---|
| A1 | camera frames → target crop | `perception_node.py:_on_synced_frames` :232 computes `primary_bbox` :263-267 and deprojects it :270 | `identity_node.py:_on_target_crop` :350 (subscribes `sensor_msgs/Image` on `reid_target_crop_topic`) | `sensor_msgs/Image` of the tracked crop + a target id to match `reid_signature_target_id` | none possible | **NOT WIRED** | No `create_publisher(sensor_msgs/Image …)` exists anywhere in the workspace; `perception_node` publishes only `PointStamped` (`:139-141`, publish at `:283`) | Perception / acquisition team (other-team owned); leader must assign |
| A2 | crop → 512-D embedding | `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/nodes/identity_node.py:_on_target_crop` :350 → `extract_embedding` (`Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/models/transreid_extractor.py`) | same call stack, `core/reid_input_adapter.py` | `interfaces/base_identity_extractor.py:27` — valid crop → unit-norm float32[512]; invalid → `zeros(512)` | `test_transreid_extractor_guardrails.py` (17 passed, 1 skipped), `test_identity_node_transreid_integration.py` (20 passed) | Tier U + Tier N | Needs `transformer_120.pth` (referenced, not copied) and a live crop input | Task 7 |
| A3 | embedding → ReidSignature message | `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/nodes/identity_node.py:_publish_signature` :392 (called :377/:385 invalid, :390 valid), publish :409 | the wire | `Implementation Phase/edgesync_ws/src/edgesync_msgs/msg/ReidSignature.msg` — `sender_id, target_id, valid, float32[512] vector, quality, uint64 stamp_utc_ns, model_id` | `test_exp7_exp13_signature_integration.py` (30 passed) | Tier U + Tier N | Dormant by default: `publish_reid_signature` = False (`:101`); with the publisher on and no crop topic the node logs the A1 gap itself (`:317`) | Task 7 (producer side) / owner (activation) |
| A4 | ReidSignature transport | `Implementation Phase/edgesync_ws/src/edgesync_identity/edgesync_identity/nodes/identity_node.py` :312 `create_publisher(ReidSignature, '/edgesync/identity/reid_signature', 5)` | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py` :527 `create_subscription(ReidSignature, topic, self._on_reid_signature, 5)` | DDS topic name + QoS depth 5, both ends declared in source | only through a stub `rclpy` | **Tier N only — never Tier R** | `ReidSignature.msg` and `ConsensusBelief.msg` have never been compiled here; no ROS 2, no DDS (`Implementation Phase/edgesync_ws/src/edgesync_msgs/msg/ReidSignature.msg` digest `44fcb363e51881ed…`) | Environment / owner (needs `colcon build`) |
| A5 | intake → association in the node | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py:_on_reid_signature` :535, refusal branches :544, :561, :567, :584, :592, :603, install :619 | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py:set_reid_signature` :423 → `_latest_reid_signature` :434 | Foreign sender / dimension / invalid flag / target mismatch / out-of-order stamp / future stamp are each refused; 100 ms skew is a **refusal** threshold, not a freshness rule; ageing is conservative (`worst = max(stamp_age, arrival_age)`) | `test_exp7_exp13_signature_integration.py`, `test_exp13_node_integration.py` (101 passed) | Tier U + Tier N | The **baseline** node has the setter (`coordination_node.py:423`) and the comment "Updated externally via set_reid_signature() from the identity smoother" (`:133`) but **no caller**; only the Exp-13 node supplies one | Task 13 (arm) / owner (baseline) |
| A6 | association → handoff request payload | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py` :476 `req.reid_feature_vector = self._latest_reid_signature.tolist()`; publisher :186-187 | peer's `_on_handoff_request` (`:161`); cross-verification in `edgesync_identity/core/identity_verifier.py:75` | `Implementation Phase/edgesync_ws/src/edgesync_msgs/msg/HandoffRequest.msg:16` `float32[512] reid_feature_vector` | `test_exp13_node_integration.py` `_on_handoff_request` :1062, :1078 (101 passed, Tier N). The baseline-only tests that would cover the service leg (`test_decision_and_receiver_services.py`, `test_response_collector.py`) were **not collected** here — `ModuleNotFoundError: No module named 'rclpy'` | Tier N (arm subclass only) | Never observed on a graph; the verifier's scored path is exercised in unit tests, not in a running handoff | Baseline owner |
| A7 | CBAA negotiation leg | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py:_publish_belief` :247 (per-peer publishers :209-211) | `_on_belief_msg` :309 → `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/negotiation/exp13_cbba_adapter.py` → `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/negotiation/cbba_negotiator.py` `build_bundle` :525, `_admit` :625, `_merge`/`_prefer` | `Implementation Phase/edgesync_ws/src/edgesync_msgs/msg/ConsensusBelief.msg`; winner rule = higher rounded bid, tie → lexicographically smaller agent id; `bid_epsilon` 1e-6, `utility_precision` 4 | `test_cbba_negotiator.py` (106 passed), `test_exp13_adapter.py` (76 passed) | Tier U (600-episode benchmark) + Tier N | `ConsensusBelief.msg` never compiled → no Tier R; under `dead_participant` / `one_silent_candidate` the arm never terminates (120/300 episodes) | Task 13 (arm) / owner (liveness rule) |
| A8 | negotiation outcome → ownership protocol | consensus winner is **advisory local belief** | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py:_on_handoff_confirm` :563 → ack :569 → `_on_handoff_confirm_ack` :580 | Request → Response → Confirm → ConfirmAck, `HandoffHeader` transaction + `ownership_epoch` | `test_exp13_node_integration.py` `_on_handoff_confirm` :851, :877-906, `_on_handoff_confirm_ack` :916-957 (101 passed, Tier N against stub `rclpy`) | Tier N | The benchmark arm does **not** execute this path, which is why `committed_dual_ownership_total` is `null` rather than 0 (F-02) | Baseline owner (M1/M2 safety fixes) |
| A9 | final ownership event | `Implementation Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py` `pub_claim` :198-199 on `/{self.robot_id}/edgesync/ownership_event`; mirrored subscription :176; claim built :234-242 and :587-594 | `_on_ownership_event` :598 (`ROLE_OWNER` from another sender at :600); Exp-13 mirror at :936-943 | `OwnershipEvent.msg` role + epoch + header | `test_exp13_node_integration.py` :742 asserts `pub_claim.topic == /ugv_1/edgesync/ownership_event` (101 passed) | Tier N — never Tier R | No real event has ever been observed; the safety audit's two-owner witness lives in the **baseline** path | Baseline owner |

## Reading the table honestly

* **Every edge at or beyond A4 is Tier U / Tier N at best.** There is no real ROS 2 evidence for any
  arrow in this chain on this host.
* **A1 is the break.** A2/A3 exist, A4 exists in source, A5 exists and is guarded — but nothing
  publishes the crop that starts it. The project says so itself in `exp13_bringup.launch.py:37-40`
  and `:204-208`, and `identity_node.py:317` logs the same condition at runtime.
* **A7 → A8 is advisory, not committed.** A CBAA winner is a local belief; ownership changes only
  on the ACK path, which the model arm never runs.
* **A5/A6 cross into baseline files.** `coordination_node.py` and `ownership_fsm.py` are
  `BASELINE_PROTECTED`; the fix the safety audit asks for (M1/M2) is an owner action, not a
  Task 13 action, and was not performed here.

## Configuration and launch surface for the chain

| File | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `exp7_transreid.yaml` | 4,183 | `a9f2a61beb165fcf…` | Task 7 model/asset/protocol selection |
| `exp13_cbba.yaml` | 11,618 | `8cd9001f412e1108…` | Task 13 protocol, timing, metric units, thresholds |
| `exp13_bringup.launch.py` | 11,774 | `a14721e20ae66f69…` | Selects the CBAA node; leaves `reid_target_crop_topic` empty |
| `ReidSignature.msg` | 2,171 | `44fcb363e51881ed…` | The Exp-7 → Exp-13 interface |
| `ConsensusBelief.msg` | 4,092 | `2d34df85da0ffc9d…` | The CBAA belief carrier (never compiled here) |
| `HandoffRequest.msg` | 1,035 | `cbf21a97b2830c23…` | Carries `reid_feature_vector` into the handoff |