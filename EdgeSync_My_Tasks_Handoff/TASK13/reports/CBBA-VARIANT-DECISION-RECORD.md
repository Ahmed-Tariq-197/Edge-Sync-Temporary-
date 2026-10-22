# ES-EXP13 — CBBA Pre-Implementation Decision Record

Record ID: ES-EXP13-DECISION-01
Date: 2026-10-19 (prepared 21:55–22:10 local, Windows host)
Author: agent session, under ES-EXP07 MASTER EXECUTION PROMPT §1–§12 and §9 Additional Execution Controls
Status: **DECISION REQUIRED FROM OWNER — implementation intentionally NOT started**
Purpose: satisfy §9.2 ("produce a pre-implementation decision record before implementing
Experiment 13") without inventing a protocol, scoring function or message contract.

---

## 0. Evidence basis and what is verifiable on disk right now

| Claim | Evidence class | Verifiable now? |
|---|---|---|
| No CBBA/CBAA code exists anywhere in the workspace | STATIC — repo-wide scan of every `.py/.md/.yaml/.csv/.txt/.json/.tex` (excluding `__pycache__`, `.pytest_cache`, `build`, `install`, `log`): **0** matches for `cbba`, `cbua`, `PROTOCOL_ID`, `CBBANegotiator`, `protocol_factory`, `negotiation_protocol`, `exp_id` | **YES** |
| `edgesync_coordination/edgesync_coordination/negotiation/` contains no `.py` files (only `__pycache__`) | STATIC directory listing | **YES** |
| `edgesync_bringup/config/exp13_cbba.yaml` and `edgesync_coordination/config/exp13_cbba.yaml` | STATIC — **MISSING** (deleted by the 2026-10-19 21:42:25 rollback) | **YES (absent)** |
| Team Execution Guide (the normative source that defines Experiment 13) | **ABSENT from the repo** — recorded independently at `docs/experiments/exp07/audit/artifact-inventory.md:113` | **NO — cannot be read** |
| A prior implementation existed and was removed | BACKED-UP COPY: `C:\Users\PC\edgesync_es07_backup\pre_s9session_20261019\Implementation Phase\edgesync_ws\src\edgesync_coordination\edgesync_coordination\negotiation\cbba_negotiator.py` — 20,856 B, 543 lines, SHA-256[:16] `7b0637c9b9181839` | **YES (read-only copy)** |
| Parameter values that prior YAML used (`max_bundle_size: 1`, `max_rounds: 50`, `convergence_threshold: 3`, `timeout_s: 5.0`) | TRANSCRIPT-ONLY — read in-session before deletion; the file was never backed up, so it **cannot be re-verified on disk** | **NO** |

Charter §9 applies: nothing below is presented as a test result. No pytest, ROS 2, colcon,
training, export, engine build or hardware run was performed for this record. `rclpy`,
`launch`, `launch_ros`, `ament_index_python`, `colcon` and `tensorrt` are all absent on this host.

---

## 1. Variant — which algorithm the evidence actually supports

**Names are not interchangeable.** The three candidates:

- **Full CBBA** (Choi, Brunet & How, ICRA 2009): per-agent bundle of length ≤ *L*, a per-task
  **candidate list** `cand_i`, and a **path-bid consistency check** in the consensus phase that
  trims candidate lists when a peer's bid exceeds the local bid at the corresponding bundle index.
- **CBAA** (Brunet, Choi & How, AAMAS 2008): *L = 1*, one task per agent, **no candidate list**,
  consensus is a pure winning-bid / winning-agent vector exchange.
- **Bounded single-round auction (L = 1)**: one highest-score-wins selection, no iterative consensus.

What the removed code did (read-only copy, line numbers cited from that backup):

| Behaviour | Location | CBBA? | CBAA? |
|---|---|---|---|
| Greedy bundle fill, bids only where it can outbid the current winner (`bid > winning_bid + ε`) | `cbba_negotiator.py:271-297`, gate at `:283` | yes (n≤L) | yes |
| `max_bundle_size` default **1** | `cbba_negotiator.py:106` | degenerates | matches |
| Consensus payload = `winning_bids`, `winning_agents`, `bundle` | `cbba_negotiator.py:306-319` | **no `cand_list` field** | matches |
| Conflict resolution = 4 bid-vector rules | `cbba_negotiator.py:321-405` | **no path-bid consistency check** | matches |
| Cascade release of tasks added after a released task | `cbba_negotiator.py:407-434` | nominal only — vacuous when *L = 1* (slice `bundle[idx:]` is always length ≤ 1) | matches |

**Classification: the removed implementation was CBAA (equivalently CBBA with *L = 1*), not full
CBBA.** Calling it "CBBA" in a report, config key or log line would be a false label.

**Independent corroboration from the frozen baseline:** `_on_negotiation_timeout` already performs
an *L = 1*, single-round, N-scalable highest-score-wins auction —
[coordination_node.py:537-539](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py#L537)
sorts valid ACCEPT responses by `utility` descending, takes `[0]`, drops the rest
(`# Scalable selection for N robots` is the existing comment at that line). Under *L = 1*, that is
exactly the allocation CBAA converges to, reached in one round from self-reported scores instead of
iterated consensus.

**Recommendation (not applied):** implement **CBAA with *L* = 1 explicitly labelled as such**, and
only as the *increment* the baseline lacks — iterated bid-vector consensus across the already-decoded
telemetry, reusing `UtilityEngine.compute_receiver_utility`
([utility_engine.py:86](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/decision/utility_engine.py#L86),
score `w_fov*s_fov + w_dist*s_dist + w_battery*s_battery` at `:128-130`) as the bid function.
Full CBBA is **not** supported by any available evidence and must not be built speculatively
(§9.2: "Do not implement a full protocol or invent a new scoring/message contract when the available
evidence does not support that choice").

---

## 2. Termination / convergence condition

- Removed code: **local** predicate only — `stable_rounds >= convergence_threshold` with no inbound
  belief change (`cbba_negotiator.py:496-506`), plus `timeout_s` (`:462-468`) and `max_rounds`
  (`:471-476`) guards.
- CBAA/CBBA theory requires **global** agreement: every agent's winning-bid vector must match. A
  locally stable agent can still be unaware of a newer peer bid, and DDS best-effort delivery gives
  no guarantee.
- **Consequence if carried over unchanged:** two agents can each terminate believing it holds the
  task. In EdgeSync that maps onto the *zero dual-ownership* invariant
  ([ownership_fsm.py:1-7](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/fsm/ownership_fsm.py#L1)).
  **Blocking design constraint, not a detail.**
- Proposed fix to be confirmed by owner: keep the local predicate as a *send* trigger only, and make
  the allocation result **advisory** (it chooses whom to address a `HandoffRequest` to). Ownership
  transfer must stay on the existing monotonic 4-step path (`HANDOFF_REQUEST` → `RESPONSE` →
  `CONFIRM` at epoch *e* → `CONFIRM_ACK` at *e+1*, `ownership_fsm.py:106-135`, `:199-207`). With that
  constraint, a wrong consensus cannot create two owners, because the ACK exchange is the only
  ownership-mutating event.

## 3. Round / iteration bound

- Transcript-only prior values: `max_rounds: 50`, `timeout_s: 5.0`.
- Frozen budgets: `dt_neg_ms: 60`, `t_neg_timeout_ms: 80`, `t_txn_ms: 886`
  ([timing_budgets.yaml:7-8,15](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_bringup/config/timing_budgets.yaml#L7)).
- **Arithmetic conflict:** 50 × 60 ms = 3,000 ms and 5,000 ms timeout are both **far larger** than the
  886 ms total transaction window (≈ 3.4× and 5.6×). Either the negotiation window is extended —
  which breaks a certified timing budget and the TTO safety margin that triggers handoff in the first
  place — or rounds must fit inside 886 ms, i.e. at most ⌊886/60⌋ = **14 rounds**, and realistically
  **1–2 rounds** if each round is one DDS exchange inside the existing 80 ms cutoff.
- Required owner decision before coding: the round bound and whether timing budgets may be touched.
  Timing budgets are baseline-certified; I will not change `timing_budgets.yaml`.

## 4. Allocation, tie-break and conflict rules (proposed, minimal)

1. Eligibility first, unchanged: `evaluate_hard_gate` + `evaluate_feasibility`
   ([eligibility_gate.py:49](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/decision/eligibility_gate.py#L49),
   `:72`) are the admissibility filter; a filtered-out agent cannot hold a task.
2. Bid = existing receiver utility, no new scoring terms, no `priority` multiplier.
3. Positive-improvement rule: an agent bids only if it strictly outbids its recorded winning bid
   (`> winner + ε`), preserving `cbba_negotiator.py:283` semantics.
4. Conflict: higher bid wins; ties broken deterministically by **lowest agent ID among agents that
   actually hold the task** — the removed code compared ID strings (`:386-393`) but left a task tied
   at bid 0 with an empty winner permanently unassigned; that dead branch must be tested or dropped.
5. Release: with *L* = 1 a release is a single-task clear; **do not** port the cascade block at
   `:407-434`, which is dead code at *L* = 1 and would imply *L* > 1.

## 5. Relation to the existing ownership / handoff FSM

Everything Exp 13 would touch sits **upstream** of ownership transfer, in candidate selection only.

- Admission is 2-party and epoch-monotonic: protocol version must equal `0x0203`, `receiver_id` must
  be `self_id` or `"*"`, stale epochs are rejected
  ([negotiation_manager.py:51](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/protocol/negotiation_manager.py#L51),
  `:55`, `:63-67`).
- The sender addresses exactly one peer: `req.header.receiver_id = self.peer_id`
  ([coordination_node.py:457](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py#L457)),
  even though the response window collects several responses (`:486-495`, `:498-513`).
- Trigger and recovery are unchanged: TTO-based auto-trigger (`:251-277`), watchdog → `RECOVERY` →
  reclaim at epoch *e+2* (`:289-294`, `ownership_fsm.py:126-135`).
- **Hard scope conclusion:** multi-agent CBBA consensus needs an *N*-party broadcast. There is no
  such topic today; the only swarm topics are the seven handoff/ownership/liveness topics in
  [robot_bringup.launch.py](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_bringup/launch/robot_bringup.launch.py)
  (`handoff_request/response/confirm/confirm_ack/cancel`, `ownership_event`, `state_heartbeat`).
  Adding a consensus topic **and** a message type = inventing a message contract → **not authorized**
  without the guide or an explicit owner decision. `state_heartbeat` is the one existing multi-writer
  channel that could legitimately carry per-agent telemetry for consensus without a new contract;
  that reuse must be approved, because it changes a certified topic's semantics.

## 6. Defects in the removed code that must not be re-created

1. `compute_bid` fell back to `battery_pct = 85.0` hardcoded and, on calculator exception, returned
   `task.priority * 0.5` (`cbba_negotiator.py:223-257`). Both are invented scores; the live telemetry
   already exists (`_on_local_heartbeat`, `coordination_node.py:279-283`). Charter §5 forbids
   substituting fabricated values for a failed computation.
2. Its module docstring claimed OwnershipFSM / NegotiationManager / HandoffRequest integration that
   the code does not perform.
3. `protocol_factory.py` (4,117 B) raised a reproducible circular `ImportError`
   (`cannot import name 'PROTOCOL_ID_PROACTIVE' from partially initialized module … protocol_factory`)
   — so the rollback removed code that had never been demonstrated to import.
4. No CBBA-focused test file exists or is referenced; there is no evidence any of it was ever executed.

## 7. Unresolved ambiguities — owner input required (§9.2 closing item)

| # | Question | Why I cannot decide it |
|---|---|---|
| A1 | Supply the **Team Execution Guide** (absent from the repo, `artifact-inventory.md:113`). Does it name CBBA, CBAA or "*L* = 1"? | It is the only normative source; naming drives the deliverable's claims. |
| A2 | Is Exp 13 about **N > 2 robots** (needs a new broadcast contract), or about **choosing better between the existing 2 parties**? | Determines whether a topic/message contract may be added at all. |
| A3 | May consensus rounds consume the 886 ms transaction budget, and if not, what is the round bound? | Timing budgets are baseline-certified; changing them is an ownership decision. |
| A4 | May `state_heartbeat` semantics be extended to carry bid vectors? | Modifies a certified swarm topic used by the watchdog. |
| A5 | Is a *label-only* deliverable acceptable — i.e. document that the certified baseline **already implements the *L* = 1 allocation**, and measure that — instead of adding a protocol? | Cheapest honest path; only the guide can rule it out. |
| A6 | Restore-vs-rewrite for Exp 7 (`extractor_factory.py`, `reid_launch_defaults.py`, `robot_bringup.launch.py`, `exp7_transreid.yaml` were all rolled back at 21:42:25) — restore from `pre_s9session_20261019\` or rewrite cleanly? | A rollback is a deliberate owner action; reversing it without instruction would overwrite that intent. |

## 8. Configuration-contract defect found while preparing this record (affects any Exp 13 bid)

`coordination_params.yaml` declares utility/gate weights under names **no node declares**:
`candidate_gate_cpu_max_pct`, `candidate_gate_battery_min_pct`, `utility_weight_feasibility`,
`utility_weight_fov`, `utility_weight_network`
([coordination_params.yaml:31-36](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_bringup/config/coordination_params.yaml#L31)),
while `coordination_node.py` declares `max_cpu_pct`, `min_battery_pct`, `w_fov`, `w_dist`,
`w_battery` ([coordination_node.py:54-60](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py#L54)).
The YAML's three weights are also a different triple (feasibility/FOV/network) from the engine's
(FOV/dist/battery). ROS 2 does not error on undeclared `ros__parameters` keys, so the file is
silently ignored and node defaults win. STATIC evidence; no run needed to observe the name mismatch;
the runtime consequence is inferred from ROS 2 parameter semantics, not measured on this host
(`rclpy` absent). Reported, **not fixed** — out of scope for §9.2 and it would modify a certified
baseline config file.

---

## 9. Files touched while producing this record

Created (new file, additive only): `docs/experiments/exp13/CBBA-VARIANT-DECISION-RECORD.md`.
No existing file was modified, moved or deleted in this step; no protected asset was touched;
no Git command other than read-only forms was run (the workspace is not a Git repository).
This record was deliberately not appended to `ES-EXP07-EXECUTION-AND-CHANGE-LOG.md`, because that
would modify an existing file and §1 requires a verified backup before any modification.
