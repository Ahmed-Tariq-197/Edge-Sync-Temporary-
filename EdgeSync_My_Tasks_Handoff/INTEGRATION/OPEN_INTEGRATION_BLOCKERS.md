# Open integration blockers

Ordered by what stops the two tasks from being a working end-to-end system. Each one states the
measured evidence, the effect, the action needed and who owns it. **None was patched during
packaging** (brief §8: "Do not patch these issues during packaging").

### B-01 — No producer for the target crop (severity: blocking; cross-task)
* **Evidence:** `grep -rn create_publisher` over every delivered `.py` file finds **0** publishers
  carrying `sensor_msgs/Image`. `perception_node.py` holds `primary_bbox` in-process (`:263-270`) and
  publishes only a `PointStamped` centroid (`:139-141`, `:283`). `exp13_bringup.launch.py:208` sets
  `reid_target_crop_topic: ''` and `:37-40` explains why.
* **Effect:** the Exp-7 signature producer has no input, so no running node has ever emitted a
  `ReidSignature`; Exp-13's intake guards are reachable only from tests.
* **Needs:** the perception or acquisition owner publishes a per-target crop topic that carries the
  target identity; then set `reid_target_crop_topic` and `reid_signature_target_id` in the launch.
* **Owner:** perception / acquisition team, assigned by the leader.

### B-02 — Message definitions have never been compiled; Tier R is unreachable (blocking; environment)
* **Evidence:** no `rclpy`, no generated `edgesync_msgs`, no `launch`/`launch_ros`, no `ros2` CLI, no
  DDS implementation on this host. Five test files across the workspace are **not collected**
  (`test_auto_trigger.py`, `test_decision_and_receiver_services.py`, `test_response_collector.py`,
  `test_identity_service.py`, `test_robot_bringup_launch.py`), all with `ModuleNotFoundError` /
  `ImportError` on ROS packages. `colcon`, `cmake` and a compiler are present but nothing is built.
* **Effect:** every integration claim in this package is Tier U or Tier N. Nothing may be described
  as ROS 2 or DDS validated.
* **Needs:** a ROS 2 install matching the project's declared distro, `colcon build`, then
  `ros2 launch edgesync_bringup exp13_bringup.launch.py`.
* **Owner:** environment / leader.

### B-03 — Consensus is advisory; the commit path is never exercised (design)
* **Evidence:** the benchmark's own `ownership_commit_basis` string, present on all 300 Exp-13 rows:
  "consensus winner is an advisory local belief; a committed owner is only observable on the
  Request → Response → Confirm → ConfirmAck path, which this model arm does not execute".
* **Effect:** `committed_dual_ownership_total` is `null`, not 0. Dual-ownership safety at the
  **commit** level is unmeasured by this package.
* **Needs:** a Tier R (or at minimum Tier N-with-services) run of the ACK exchange.
* **Owner:** Task 13 + baseline owner jointly.

### B-04 — CBAA does not terminate under two fault conditions (design decision)
* **Evidence:** 60/60 `dead_participant` → `STALLED_NOT_TERMINAL`, 60/60 `one_silent_candidate` →
  `UNDECIDED`, at every N ∈ {2,3,4,6,8}; the baseline arm decides in all those cells except N = 2.
* **Effect:** the CBAA arm trades liveness for symmetric agreement. Presented in the headline of
  `../TASK13/README_TASK13.md` §5, not in an appendix.
* **Needs:** an owner ruling between the conservative no-unilateral-agreement rule and a
  timeout/quorum fallback. A failure detector, quorum, lease or transaction protocol must not be
  invented inside this task.
* **Owner:** leader / supervisor.

### B-05 — The safety audit's blocking findings live in files Task 13 may not edit
* **Evidence:** `../TASK13/reports/EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:400-429` concludes
  **BLOCKED PENDING DESIGN APPROVAL**; M1 + M2 + M3 must be "executed on the baseline", i.e.
  `coordination_node.py` and `ownership_fsm.py` (`BASELINE_PROTECTED`). `EXP13_SAFETY_FINDINGS_20261020_210721.md:177`
  records M3 as "baseline hazard, fixed in arm only".
* **Effect:** the Exp-13 arm is safe in isolation; the deployed baseline path it wraps is not.
* **Needs:** an authorisation decision on the baseline patch (a proposal exists in the audit's own
  deliverables).
* **Owner:** baseline owner / leader.

### B-06 — Deployment evidence is absent (blocking for the robot claim)
* No Jetson, no TensorRT: `benchmark_transreid_jetson.py` and `build_transreid_trt_engine.py` were
  read, never executed; `../TASK07/experiment_results/trt_blocked/probe_build_report.json` records
  the blocked attempt.
* **Effect:** no latency or memory figure in this package may be quoted as a Jetson number.
* **Owner:** leader (hardware access).

### B-07 — Unresolved naming / scope labels
* A1 (algorithm naming), A2 (N > 2 vs 2-party selection) and D1 (whether the deliverable may be
  called CBAA at all) are open in the audit's own words. This package uses the implementation's
  wording: `negotiation_protocol: cbba`, `algorithm_variant: CBAA (CBBA with bundle length L = 1)`.
* Historical documents disagree with the current tree in one place worth knowing: the safety audit
  states the belief-carrier `.msg` "still does not exist", while `ConsensusBelief.msg`
  (2d34df85da0ffc9d…) is present in the project and delivered here.
  Both readings are preserved; the audit is dated earlier.
* **Owner:** leader.

## What would clear the package for a real handoff

1. B-01 and B-02 are the only two that block the end-to-end demonstration; everything else is a
   decision or a measurement.
2. With ROS 2 and a crop publisher, the minimum new evidence is: one `ReidSignature` observed on a
   real topic, one `ConsensusBelief` round-trip between two built nodes, and one
   Request → Response → Confirm → ConfirmAck sequence producing a single `OwnershipEvent`.
3. Until then, Tier U / Tier N is the ceiling of what this package can support.