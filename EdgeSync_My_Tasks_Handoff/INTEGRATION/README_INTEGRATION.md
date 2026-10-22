# Integration

This folder answers one question: **can Task 7 and Task 13 run together as the system the
proposal describes?** It is derived from the delivered source, not from the design documents.

| File | What it is |
|---|---|
| `TASK07_TASK13_DEPENDENCY_MAP.md` | Nine arrows covering the whole chain, each with producer,
  consumer, contract, the test that touches it, the verification level actually reached, what is
  missing, and who owns the missing piece |
| `OPEN_INTEGRATION_BLOCKERS.md` | B-01 … B-07, each with measured evidence, effect, action needed
  and owner |
| Inventory rows | 2 (these two documents; the
  folder holds no code) |

## The finding, in one line

The identity→negotiation interface is **built and tested on both ends but has no producer**:
`perception_node.py` keeps the target crop in-process and publishes only a `PointStamped`
centroid, and no file in the delivered tree creates a `sensor_msgs/Image` publisher. So arrow A1
is NOT WIRED, and every downstream arrow is exercised only inside one process.

## The ceiling on every claim in this folder

* **No ROS 2, no DDS, no `rclpy` on this host.** Consequently nothing here is Tier R.
* Node-level evidence is Tier N: a stub `rclpy` with message classes generated from the on-disk
  `.msg` files, which were never compiled by `colcon`.
* Two arrows (A8, A9) are additionally limited by design: the benchmark arm never executes the
  Request → Response → Confirm → ConfirmAck commit path, so it reports
  `committed_dual_ownership_total = null` rather than 0.

## What was deliberately not done

Per the brief, no blocker in this folder was patched during packaging: no crop publisher was
added, no baseline file was edited, no failure detector or quorum rule was invented, and the
message set was not re-declared. The dependency map records the gaps as they are.

## Where to read the same evidence first-hand

| Claim | Look at |
|---|---|
| No Image publisher | `SHARED_DEPENDENCIES/source_snapshot/…/edgesync_perception/edgesync_perception/nodes/perception_node.py` :139-141, :263-283 |
| Launch leaves the crop topic empty | `TASK13/configs/edgesync_bringup/launch/exp13_bringup.launch.py` :37-40, :204-208 |
| Producer side of the signature | `TASK07/source_snapshot/…/edgesync_identity/edgesync_identity/nodes/identity_node.py` :101-104, :302, :312, :350, :392-410 |
| Consumer side | `TASK13/source_snapshot/…/edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py` :527-535, :544-619 |
| Intake refusals | same file, :544, :561, :567, :584, :592, :603 |
| Fresh test evidence | `VALIDATION/fresh_evidence/FRESH_RUN_RECORD.json` |