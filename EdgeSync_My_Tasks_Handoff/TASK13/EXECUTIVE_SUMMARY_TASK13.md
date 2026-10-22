# Executive Summary — Task 13: CBAA / CBBA Decentralised Target Handoff

Prepared: 2026-10-22, from the files packaged in this handoff folder plus the source project the
copies came from. This document changes nothing: it adds one file and cites existing evidence.

Package-root-relative path prefixes used below (all verified to exist in this package):

* `SNAP` = `TASK13/source_snapshot/Implementation Phase/edgesync_ws/src`
* `RES` = `TASK13/benchmark_results/edgesync_bringup/results/experiment13`
* `REP` = `TASK13/reports`
* `FRESH` = `VALIDATION/fresh_evidence`

Labels used throughout, per the brief's evidence-integrity rules:
**MEASURED** (re-derived by me from a packaged artefact at the line/field cited),
**INHERITED** (a figure that exists only inside a report or an older artefact),
**NOT MEASURED**, **BLOCKED**, **NOT VERIFIED**.

---

## A. Executive overview

**What the task is for.** EdgeSync's baseline handoff allocator is a one-round request/response
protocol: the current owner broadcasts a `HandoffRequest`, collects `HandoffResponse`s inside a fixed
window (`t_neg_timeout_ms: 80`), and commits on its own ranking. Only the owner holds a decision.
Task 13 replaces that single decision-maker with a decentralised consensus allocator in which every
declared participant prices the target, exchanges beliefs, and must reach the same winner before the
owner-side node is allowed to transfer. The intended contribution to EdgeSync is an allocation that
is order-independent, admission-controlled, and verifiable from every participant's own view rather
than from the owner's.

**Why this strategy was chosen.** The selected mechanism is a consensus-based allocator whose belief
exchange converges without a broker. The delivered implementation is **CBAA — CBBA restricted to
bundle length `L = 1`** — one task per agent, one target per episode. Full CBBA features (candidate
lists, path-bid consistency trim, cascade release) are deliberately absent, and the module refuses to
run with `max_bundle_size != 1` (`SNAP/edgesync_coordination/edgesync_coordination/negotiation/cbba_negotiator.py:283-285`),
so the code cannot silently present itself as generic CBBA. The selector token fixed by the team
execution guide is `negotiation_protocol: cbba`, which is the value the YAML uses; `algorithm_variant`
is reporting-only and switches no code.

**Status, in one paragraph.** The algorithm, adapter, node subclass, message definition,
configuration, launch file and benchmark harness are **implemented and executed at Tier U (unit /
model simulation) and Tier N (node tests on a stub `rclpy`)**: the coordination package ran **360
passed with 3 collection errors, exit 1**, and a 600-episode two-arm benchmark was executed and then
**re-executed from the copies in this folder** with 588/600 trial rows byte-identical. On the two
required metrics the measured result is **unfavourable to Exp-13** (14.08 vs 4.2 mean application
messages per episode; 3.32 vs a definitional 1 mean rounds; convergence *time* not measurable on this
host in either direction), and two of five fault conditions never reach a terminal decision
(60/60 `dead_participant` stall, 60/60 `one_silent_candidate` undecided). Nothing has ever run on a
real ROS 2 graph: no `rclpy`, no DDS, `ConsensusBelief.msg` never compiled, `exp13_bringup.launch.py`
never executed (**BLOCKED**). The post-implementation safety audit that gates integration reads
**"BLOCKED PENDING DESIGN APPROVAL"** (`REP/EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:400`) because the
measured dual-`OWNER` defect lives in two shared files Task 13 is not permitted to modify. Task 13 is
therefore **ready for review** at Tier U/N, **not ready for integration**, and **not deployment-validated**.

---

## B. Requirements and success criteria

### B.1 Original requirements as recorded in the package

The requirement set Task 13 itself wrote down is the 25-row acceptance checklist at
`REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:70-96` (R1–R25), plus the guide's five structural rules
(unique plugin identifier, experiment YAML, dedicated pytest file, parameter switch, `exp_id`
propagation). The table below maps each to the classification vocabulary this brief requires. No
criterion was added or reworded.

| # | Original requirement | Classification | Evidence |
|---|---|---|---|
| R1 | `cbba_negotiator.py` implements the approved allocation mechanism | **PARTIALLY VERIFIED** — as CBAA `L = 1` only | `cbba_negotiator.py:66` `PROTOCOL_ID = "cbba"`, `:69` `ALGORITHM_VARIANT = "CBAA (CBBA with bundle length L = 1)"`, `:283-285` raises above `L = 1`; 106 tests pass (F). Full-CBBA semantics are absent by decision, listed at `:25-26` |
| R2 | Algorithm identity officially approved | **BLOCKED** — owner decision D1 not granted | `REP/EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:329` (AUD-12): repository record says CBAA, external REV04 §3.1 says IBBM-1 and reserves the label to supervisor approval, "which was never granted"; `REP/CBBA-VARIANT-DECISION-RECORD.md` §1/§5 is marked "DECISION REQUIRED FROM OWNER" |
| R3 | Deterministic, order-independent winner | **VERIFIED** at Tier U | `determinism_probe` and `sequential_activation_probe` in both artefacts: 1 distinct winner over 2 / 6 / 24 orders (`RES/exp13_unit_benchmark.json`, MEASURED); `test_tie_winner_is_independent_of_activation_order` |
| R4 | Unequal utilities still select the correct winner | **VERIFIED** at Tier U | `winner_matched_expected = 180` of 180 converged episodes, all 300 rows (MEASURED) |
| R5 | Local stability ≠ local agreement ≠ global convergence | **VERIFIED** | `cbba_negotiator.py:118` `LOCAL_STABLE`, `:1013`, `is_converged()` true only for `AGREED` (`:1050-1056`); `exp13_cbba.yaml:70-72` records `convergence_threshold` as diagnostic |
| R6 | Global convergence verified across participants, not just locally | **PARTIALLY VERIFIED** | `agreed_by_all = 180 / 300` (MEASURED); the ack leg exists (`:183` `agreement_established`, `:800` `ingest_agreement_report`, `:869` `check_distributed_agreement`, gated at adapter `:738`) — but `:761-778` states in the code that the reports are **not** a quorum, lease or failure detector, so distributed commitment is still not established |
| R7 | Stale / mismatched messages refused | **VERIFIED** at Tier U | `_admit` `:625`, epoch mismatch → `STALE` at `:650-651`; classes `:128-136`; 2,952 refusals observed across 300 episodes (MEASURED by re-summing rows) |
| R8 | Ownership epoch bound into the belief carrier on the wire | **PARTIALLY VERIFIED / BLOCKED at ROS level** | `ConsensusMessage.ownership_epoch` exists in Python (`:181`) and is in `key()` (`:195`); `ConsensusBelief.msg` exists (4,092 B, `2d34df85da0ffc9d…`) but has **never been compiled** — BLOCKED, see B3 |
| R9 | Reset and episode isolation | **VERIFIED** at Tier U | `:51` `(episode_id, ownership_epoch)` fence; `test_reusing_a_published_episode_id_is_refused`, `test_replay_from_a_previous_ownership_epoch_is_refused` |
| R10 | Participant support | **VERIFIED** for N = 2…8 at Tier U | per-N aggregates 60 episodes each at N = 2, 3, 4, 6, 8 (MEASURED) |
| R11 | Malformed messages validated before touching state | **VERIFIED** at Tier U | `_admit` runs before `_merge` (`:684`); tests assert `beliefs_delivered == 0` |
| R12 | `exp13_cbba.yaml` correct and nested as the project loads it | **VERIFIED** | `TASK13/configs/edgesync_bringup/config/exp13_cbba.yaml:44-45` `/**: ros__parameters:`; keys at `:48, :50, :52, :60-62, :65-75, :81, :87-94`; echoed verbatim into `exp13_config` in the result JSON (MEASURED match) |
| R13 | A YAML key is functional, not just present | **FAILED** for the two selector keys | `config_consistency.keys_not_consumed` lists `negotiation_protocol`, `exp_id`, `swarm_participant_count`, `target_count`, `battery_pct`; `key_accounting`: "no ROS 2 consumer exists", "no selector exists in coordination_node.py" |
| R14 | `negotiation_protocol` selects the strategy in production | **BLOCKED** (needs shared-file approval) | Baseline `SNAP/edgesync_coordination/edgesync_coordination/nodes/coordination_node.py:39-66` declares **24** parameters and contains no `negotiation_protocol` and no `exp_id` (verified in the packaged copy). The selector exists only on the subclass (`exp13_coordination_node.py:115, :140-147`), and that node has never run on a real graph |
| R15 | `exp_id` drives automated result logging | **NOT VERIFIED** — and the premise is contested | Declared `:116`, read `:136`, logged `:229` on the Exp-13 node only; `grep exp_id` over `edgesync_observability` returns zero matches; `REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:86` records "the project's established convention is per-experiment JSON written by a `--out` flag; `exp_id` is a provenance/gate label, not a logging key" |
| R16 | Dedicated unit tests exist and pass | **VERIFIED** | 5 dedicated files, 329 of the 360 passing tests (see §F) |
| R17 | Tests genuinely varied, not seed-labelled | **VERIFIED** at Tier U | `trial_design`: 245 distinct utility maps, 148 distinct orders over 300 episodes (MEASURED in artefact) |
| R18 | Benchmark uses one shared message-accounting boundary | **VERIFIED** | `metric_definitions.app_messages_sent` / `.app_messages_ingested` state one boundary for both arms; `hardcoded_fields_removed` lists 3 former constants |
| R19 | Outcomes derived from observed state, not hardcoded | **VERIFIED** | same field; `outcome_class` rule shared by both arms |
| R20 | Non-converged / failed episodes reported, not excluded | **VERIFIED** | 600 rows retained; 24 / 120 non-converged surfaced (MEASURED) |
| R21 | Convergence rounds reported alongside elapsed time | **VERIFIED** | rounds primary; 235–238 of 240 intervals at or below the host tick (§E, §conflicts) |
| R22 | Benchmark conclusion stated without spin | **VERIFIED** | `REP/EXP13_BASELINE_COMPARISON_20261020_210721.md:180-181`: "runs 3.32 rounds to the baseline's definitional 1, converges on 60 % of episodes against 92 %" |
| R23 | ROS 2 integration verified | **FAILED / BLOCKED** | No `rclpy`, no DDS; `exp13_cbba.yaml` appears in no `parameters=[…]` of `SNAP/edgesync_bringup/launch/robot_bringup.launch.py:148-151`; `ConsensusBelief.msg` never compiled |
| R24 | Multi-robot / network reliability demonstrated | **NOT MEASURED** | `message_layer`: "application protocol message; DDS/network packets NOT MEASURED" |
| R25 | On-device (Jetson) performance | **NOT MEASURED** | `FRESH/FRESH_RUN_RECORD.json` → `blocked_validation`: Tier H NOT MEASURED |

### B.2 Experimental hypotheses / success criteria and their outcome

| Hypothesis as originally framed | Outcome | Evidence |
|---|---|---|
| H-A: a consensus allocator reduces coordination overhead vs the baseline | **FAILED (measured)** | mean application messages/episode 4.2 → **14.08** = 3.35×, mean ingested 3.92 → **55.64** = 14.2× (MEASURED) |
| H-B: a consensus allocator converges faster | **NOT MEASURED** | baseline `convergence_time_s` n = 0 of 300; Exp-13 240 of 300 intervals exist and 235–238 of them are ≤ one clock tick. No direction is supportable |
| H-C: fewer / equal non-converged episodes | **FAILED (measured)** | converged 276/300 (92 %) baseline vs 180/300 (60 %) Exp-13; 120 vs 24 non-converged |
| H-D: no dual ownership | **NOT MEASURED at the property that matters** | `committed_dual_ownership_total = null` in both artefacts; the reported `dual_owner_violations_total = 0` is a **local** check (`dual_owner_basis`: "each agent's own bundle against its own winning vector"). External REV04 `:421` INV-09 assigns this property to tiers I + H and states "U cannot establish it" |
| H-E (added by the work itself): order-independent winner, working admission control | **VERIFIED at Tier U** | 1 distinct winner over 2/6/24 orders (baseline 1/2/3); 2,952 classed refusals vs a baseline collector with no admission test |

### B.3 Guide structural requirements (independent check against the guide's own paths)

| Guide requirement | Classification | Evidence |
|---|---|---|
| Deliverable file named `cbba_negotiator.py` at the member's negotiation path | **VERIFIED** | `SNAP/edgesync_coordination/edgesync_coordination/negotiation/cbba_negotiator.py` — 58,026 B, 1,138 LF, a real implementation, not a placeholder |
| Experiment YAML `exp13_cbba.yaml` | **VERIFIED as a file; FAILED as a live switch** | `TASK13/configs/edgesync_bringup/config/exp13_cbba.yaml`; loaded only by `exp13_bringup.launch.py` (`:12-19` of that file, its own words) which has never run |
| Dedicated pytest file | **VERIFIED** | `TASK13/tests/edgesync_coordination/tests/test_cbba_negotiator.py` (106 passed) plus 4 further Exp-13 files |
| Unique plugin identifier registered through the selection mechanism | **PARTIALLY VERIFIED** | `PROTOCOL_ID = "cbba"` (`cbba_negotiator.py:66`, adapter `:63`), `BASELINE_PROTOCOL_ID = "cnp"` (`:66`), `SUPPORTED_PROTOCOL_IDS` (`:74`), `UNIMPLEMENTED_PROTOCOL_IDS = ("auction_cnp", "centralized_broker")` (`:69`), resolver `resolve_negotiation_protocol` (`:201`) raises on unknown tokens (`:220`) rather than falling back. This is a **hand-written resolver in the Exp-13 subclass**, not the project's registry/factory mechanism (`BACKBONE_REGISTRY` is Re-ID-side, Task 7's). No `negotiation_protocol` selector exists on the baseline node |
| `exp_id` propagated as required and triggers automated logging | **NOT VERIFIED** | see R15 |

---

## C. Implementation completed

### C.1 Files written by Task 13 (all new relative to the 2026-10-16 archive)

`TASK13_STATUS_AND_BLOCKERS.md:3-6` and `README_TASK13.md:176-184` classify the 97 inventory rows of
this slice as **16 created** after the archive, **1 modified** (`edgesync_coordination/setup.py`,
needed to register the new package entry point) and **10 byte-identical** to it. Sizes and line
counts below were measured on the packaged copies.

| File (prefix `SNAP/…` unless stated) | Size / lines | Responsibility |
|---|---:|---|
| `edgesync_coordination/negotiation/cbba_negotiator.py` | 58,026 B / 1,138 LF | The CBAA belief layer: bundle construction (`build_bundle` `:525`), bid/consensus merge (`_merge` `:684`, `_prefer` `:720`), admission (`_admit` `:625`), status enum (`:107-122`), rejection classes (`:128-136`), local agreement (`check_agreement` `:747`), distributed agreement (`:869`), self-contradiction / dual-claim invariants (`:910-927`), metrics (`:1068-1073`) |
| `edgesync_coordination/negotiation/exp13_cbba_adapter.py` | 45,609 B / 866 LF | Maps live baseline task/agent/telemetry state onto negotiator inputs; owns the commit decision. `CommitAction` (`:94`) with `COMMIT` `:103`, `ABORT` `:107`; `arm_commit()` `:686-777`; late-infeasibility downgrade of a cached `COMMIT` `:573-587`; `resolve_negotiation_protocol` `:201` |
| `edgesync_coordination/nodes/exp13_coordination_node.py` | 54,578 B / 962 LF | `class Exp13CoordinationNode(CoordinationNode)` (`:80`) — subclass adding exactly four things: the selector (`:115`, read `:140`, resolved `:142`, dispatched `:147`), `exp_id` (`:116/:136/:229`), adapter construction (`:193`), and the Exp-7 `ReidSignature` intake (`_on_reid_signature` `:535` → `set_reid_signature()` `:619`) |
| `edgesync_bringup/config/exp13_cbba.yaml` | 11,618 B / 149 LF | Protocol, timing, metric-unit and measurement-definition configuration, with its own load-path disclaimer at `:11-33` |
| `edgesync_bringup/launch/exp13_bringup.launch.py` | 11,774 B / 225 LF | Bring-up that starts the Exp-13 node with this YAML remapped onto it. **Never executed** — no ROS 2 on this host |
| `edgesync_msgs/msg/ConsensusBelief.msg` | 4,092 B / 58 LF | Belief carrier: bundle, winning bids/agents, round, `ownership_epoch`, episode id, `agreement_established`, `schema_version`. **Never compiled** |
| `edgesync_coordination/scripts/run_exp13_unit_benchmark.py` | 76,660 B / 1,366 LF | Tier-U harness: baseline arm (a stdlib model of `coordination_node.py:441-545`) vs Exp-13 arm, 5 conditions × 5 sizes × 12 trials, probes, aggregation, table rendering |
| `edgesync_coordination/scripts/run_fault_injection.py` | — | Fault-injection driver (present in the snapshot) |
| 5 dedicated test files | see §F | `TASK13/tests/edgesync_coordination/tests/test_cbba_negotiator.py`, `test_exp13_adapter.py`, `test_exp13_node_integration.py`, `test_exp13_benchmark_metrics.py`, `test_exp7_exp13_signature_integration.py` |
| 3 result artefacts + 13 reports + 2 manifests | see §I | `RES/`, `REP/`, `TASK13/manifests/` |

### C.2 Important algorithmic and architectural decisions

* **Identity.** CBAA at `L = 1`; four explicit equivalence statements at `cbba_negotiator.py:73-76`
  ("IS CBAA (Brunet, Choi & How, AAMAS 2008)", outcome-equivalent to CBBA *only* for the winning-bid
  component at `L = 1`, **not** equivalent for bundle ordering, candidate-list trimming or cascade).
  Full-CBBA machinery is absent by decision, and the list of what is *not* ported is at `:25-26`.
* **Winner rule (tie rule).** Higher rounded bid wins; exact tie → lexicographically smaller agent id.
  Applied in **both** decision sites, `_merge` and `build_bundle`. The second site is the F-04 fix:
  before it a tied *local* bid failed the strict-improvement gate, so the winner became whoever
  published first.
* **Protocol constants deliberately not parameters.** `MIN_AGREEMENT_ROUND = 2` (`:98`, clause (iv))
  and `BELIEF_SCHEMA_VERSION = 3` (`:79`) are excluded from the YAML on purpose; `exp13_cbba.yaml:126-134`
  explains that a tunable able to reach 1 or 0 would let a run weaken a correctness clause to chase a
  benchmark number. Schema history is recorded at `:84` (1 → 2 added `ownership_epoch`; 2 → 3 added
  `agreement_established`).
* **Safety-by-default commit.** `arm_commit()` returns `COMMIT` only for the owner when the episode is
  `AGREED`, no invariant is violated and — with `require_distributed_agreement: true`
  (adapter `:434`, gate `:738`) — every declared peer has *reported* agreement for this episode and
  epoch. Otherwise `RETAIN` / `STANDBY` / `WAIT` / `ABORT` / `UNSAFE`; a non-owner never receives
  `ABORT` (`:718-720`).
* **Advisory lane.** REQ-109 is honoured: the negotiator imports only stdlib and never calls an FSM,
  never writes an epoch, never publishes (audit AUD-16, `:333`). **But** the audit also records that
  the module's own docstring derived an unsound conclusion from that lane ("a wrong consensus cannot
  create two owners") — AUD-23, `:341`; the current copy has corrected it: `:779-781` now states the
  commit path "contains a measured dual-ownership defect … unrelated to this module".
* **Baseline left alone.** `coordination_node.py`, `fsm/ownership_fsm.py`, `fsm/epoch_manager.py`,
  `protocol/negotiation_manager.py`, `decision/*` and `edgesync_msgs/*` are shared, owned elsewhere, and
  are **unmodified**; the same copies also appear under `SHARED_DEPENDENCIES/`, which is the correct
  classification.

### C.3 Configuration and parameter-selection mechanism, stated exactly

* `exp13_cbba.yaml:50` sets `negotiation_protocol: "cbba"`; `:48` sets `exp_id: "EXP13"`; `:52` sets the
  reporting-only `algorithm_variant`.
* The **only** runtime consumer is the subclass node: `exp13_coordination_node.py:115` declares it
  with default `BASELINE_PROTOCOL_ID` (`"cnp"`), `:140` reads it, `:142` resolves it, `:147` dispatches,
  `:151-152` logs the baseline path when not selected, `:224` logs the armed CBAA path. An unknown
  token raises at startup (`adapter :220`) rather than falling back.
* The **baseline node does not have this switch** (24 declared parameters, `coordination_node.py:39-66`;
  no `negotiation_protocol`, no `exp_id`), and `robot_bringup.launch.py:148-151` passes only the
  original four YAML files, so baseline runs are unaffected by anything in `exp13_cbba.yaml`.
* The project's existing experiment-configuration convention (`EDGESYNC_EXPERIMENT_CONFIG` +
  `select_experiment_config`, `SNAP/edgesync_bringup/edgesync_bringup/reid_launch_defaults.py:80, :229`,
  wired at `robot_bringup.launch.py:38-48`) reaches **only the Re-ID path**; `node_coordination` never
  sees it. Recorded at `REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:179`.

---

## D. Experimental methodology

**Harness and arms.** `run_exp13_unit_benchmark.py`, Tier U: one process, no ROS graph, no network, no
Jetson (`measurement_scope`). The **baseline arm is a stdlib model of `coordination_node.py:441-545`**
(`baseline_config.selection_source_lines`, verified: `:441 broadcast_handoff_request`, `:484-495
start_negotiation_window`, `:498 _on_handoff_response`, `:534 abort_to_recovery`), **not** a live
execution of the node — `rclpy` is absent. So no conclusion transfers to the real node.

**Design.** 5 conditions (`clean`, `equal_utilities`, `replayed_traffic`, `dead_participant`,
`one_silent_candidate`) × 5 participant counts (N = 2, 3, 4, 6, 8) × 12 trials = **300 episodes per
arm, 600 rows total**, one seeded stream (`trial_design.seed = 20261020`), 1 target per episode.
Variation is proven, not asserted: **245 distinct utility maps and 148 distinct arrival orders**;
`trial_design.interpretation` states that a count of 1 would mean the sweep varied nothing.

**Metrics and measurement procedure.** One accounting boundary for both arms:
`message_count_unit: belief_builds_counted_at_publish`, `ingestion_count_unit: payloads_examined_at_receive`,
`convergence_time_unit: s,monotonic,single-process` (YAML `:109-124`). DDS packets and network bytes
are **NOT MEASURED**. Three formerly hardcoded fields were removed and are listed verbatim in
`hardcoded_fields_removed`: `baseline res.decided = True`, `baseline res.converged_agents = 1`,
`exp13 ingestion_attempts = current_round * (N - 1)`. Rounds are arm-specific and never converted
(`metric_definitions.rounds`). `expected_winner` is **not** a shared oracle: for the baseline it is the
model's own rule and therefore tautological; for Exp-13 it is an independent statement of the
documented tie rule (`metric_definitions.expected_winner`). Four asymmetries that cannot be made
equivalent are disclosed in `non_equivalences_disclosed` and reproduced verbatim in
`README_TASK13.md:122-125`.

**Parameters used.** From `exp13_config` in the artefact: `deadline_s 1.0`, `max_rounds 4`,
`max_bundle_size 1`, `convergence_threshold 2`, `bid_epsilon 1e-06`, `utility_precision 4`; utility
weights `w_receiver_fov 0.5 / w_dist 0.3 / w_battery 0.2` sourced from `coordination_node.py:58-60`
(verified); baseline `t_neg_timeout_ms 80` (`:47`). The YAML overrides no weights and no baseline
timing budgets (`exp13_cbba.yaml:145-149`), and `battery_pct` is **not** configured: the node supplies
live telemetry or the negotiator abstains (`:136-143`, DEF-01).

**Hardware / software environment.** `host`: Windows-10-10.0.26300-SP0, AMD64, Python 3.10.11,
monotonic clock `GetTickCount64()` nominal resolution 0.015625 s with a **measured** tick of
0.014999999999417923 s, simulator wall clock 0.235 s (inherited artefact) / 0.484 s (fresh). No GPU,
ROS, DDS or TensorRT involvement.

**Reproducibility limitations.** (1) The design gate fixed **2 declared participants** while the sweep
deliberately runs N > 2; `config_consistency.declared_vs_sweeped_participants` warns that those rows
"are NOT the Tier-U configuration the YAML
declares and must not be reported as such". (2) Any interval at or below one host tick is
duration-blind, and the boundary count moves between identical-seed reruns. (3) The fresh re-execution
of the whole 600-episode sweep from the packaged copies is itself the reproducibility test: see §E.6.

---

## E. Results and interpretation

All figures below were re-derived from the packaged artefacts unless marked INHERITED. Two artefacts
exist for the same experiment and are reported **separately**, because they are not the same
measurement:

* **INHERITED** `RES/exp13_unit_benchmark.json` — 1,407,601 B, sha256 `207f41c38b75374c…`, produced by
  the project pass dated 2026-10-20 19:16 (its `dual_owner_basis` is the pre-F-02 local check and it
  has no `committed_dual_ownership_total` / `cross_agent_*` fields).
* **FRESH** `FRESH/benchmark_copy_repro/exp13_unit_benchmark.json` — 1,644,861 B, executed
  2026-10-21 during packaging against the copies in this folder (`FRESH_RUN_RECORD.json`, exit 0).

### E.1 Headline, all 300 episodes per arm (`__all__`) — identical in both artefacts

| Metric | Baseline | Experiment 13 | Label |
|---|---:|---:|---|
| Episodes executed | 300 | 300 | MEASURED |
| Converged | **276 (92 %)** | **180 (60 %)** | MEASURED |
| Undecided | 24 | **60** | MEASURED |
| Stalled (non-terminal) | 0 | **60** | MEASURED |
| Failed (safety class violated) | 0 | 0 | MEASURED |
| All-participant-agreed episodes | **0** | **180** | MEASURED |
| Winner matched the arm's expected winner | 276 | 180 | MEASURED (not a shared oracle, §D) |
| Mean application messages sent | **4.2** (med 4.0, min 1, max 8) | **14.08** (med 12.0, min 3, max 30) | MEASURED |
| Mean payloads ingested | 3.92 (med 3.0, max 14) | **55.64** (med 24.0, max 224) | MEASURED |
| Mean / median rounds | 1 / 1.0 — *definitional* | **3.32 / 3.0** (min 3, max 4) — observed | MEASURED, **not convertible** |
| Mean `convergence_time_s` | `null` (n = 0 of 300) — nothing to average | 0.000383 s INHERITED / 0.000521 s FRESH (n = 240) | **NOT a speed claim**, §E.5 |
| Mean refused inbound payloads | 0 — *definitional*, its collector has no admission test | **9.84** | MEASURED |
| `dual_owner_violations_total` | 0 | 0 | MEASURED, **local proxy only** |
| `committed_dual_ownership_total` | not present (INHERITED) / `null` (FRESH) | same | **NOT MEASURED** |
| `cross_agent_disagreeing_pairs_total` | not present / `null` | not present / **0 over 300 measurable rows** | MEASURED (FRESH) |

**What this demonstrates.** Exp-13 pays real overhead (3.35× sent, 14.2× ingested) and does not
terminate more often than the baseline; it buys two properties the baseline cannot express — all
participants reaching the same named winner (180 episodes where the baseline structurally has 0) and
an admission gate that demonstrably operates (2,952 classed refusals).
**What it does not demonstrate.** No network or DDS efficiency; no speed; no deployment-level ownership
safety; nothing about the real node, because the baseline arm is a model of it.

### E.2 Per fault condition — where the result actually breaks (both arms, 60 episodes each)

| Condition | Baseline | Experiment 13 | Exp-13 rounds | Exp-13 sent | Exp-13 ingested | Exp-13 refused |
|---|---|---|---|---:|---:|---:|
| `clean` | 60/60 CONVERGED | 60/60 CONVERGED | 3–3 | 13.6 | 42.4 | 0.2 |
| `equal_utilities` | 60/60 CONVERGED | 60/60 CONVERGED | 3–3 | 13.6 | 42.4 | 0.2 |
| `replayed_traffic` | 60/60 CONVERGED | 60/60 CONVERGED | 3–3 | 13.6 | 84.8 | **42.4** |
| `dead_participant` | 48/60 CONVERGED, 12 UNDECIDED | **0/60 — 60 × `STALLED_NOT_TERMINAL`** | 3–4 | 13.4 | 56.0 | 4.6 |
| `one_silent_candidate` | 48/60 CONVERGED, 12 UNDECIDED | **0/60 — 60 × `UNDECIDED`** | 3–4 | 16.2 | 52.6 | 1.8 |

MEASURED in both artefacts and confirmed row-by-row: the outcome-class counts over the 300 Exp-13 rows
are exactly `CONVERGED` 180, `STALLED_NOT_TERMINAL` 60, `UNDECIDED` 60, `FAILED` 0.
The failure mode is the design, not an accident: `require_distributed_agreement: true` means a missing
peer never yields `AGREED`, and design decision D4 forbids silently dropping a declared participant
from the consensus set. The baseline completes because it never required the missing agent's answer.
**This is reported as a liveness regression, not as a safety win** (`REP/EXP13_BASELINE_COMPARISON_20261020_210721.md:103-105`).

### E.3 Per participant count (N = 2 is the configuration the YAML declares)

| N | Arm | Mean sent | Mean ingested | Mean rounds | Converged | All-agree | Refused |
|---:|---|---:|---:|---:|---:|---:|---:|
| **2** | baseline | 1.6 | 0.80 | 1 | **36 / 60** | 0 | 0 |
| **2** | exp13 | 4.6 | 3.60 | 3.0 | **36 / 60** | 36 | 1.8 |
| 3 | baseline | 2.6 | 2.00 | 1 | 60 / 60 | 0 | 0 |
| 3 | exp13 | 8.8 | 13.60 | 3.4 | 36 / 60 | 36 | 3.4 |
| 4 | baseline | 3.6 | 3.20 | 1 | 60 / 60 | 0 | 0 |
| 4 | exp13 | 12.2 | 29.40 | 3.4 | 36 / 60 | 36 | 6.0 |
| 6 | baseline | 5.6 | 5.60 | 1 | 60 / 60 | 0 | 0 |
| 6 | exp13 | 19.0 | 79.00 | 3.4 | 36 / 60 | 36 | 13.6 |
| 8 | baseline | 7.6 | 8.00 | 1 | 60 / 60 | 0 | 0 |
| 8 | exp13 | **25.8** | **152.60** | 3.4 | 36 / 60 | 36 | 24.4 |

MEASURED (identical in both artefacts). **Interpretation, as the report states it**
(`REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:145-149`): at the declared N = 2 configuration
Exp-13 commits **exactly as many** episodes as the baseline (36/60 both arms) — the "commits fewer
episodes" headline is a property of the N > 2 sweep rows, not of the declared configuration. The N > 2
divergence has an identified cause: Exp-13 withholds commitment when a declared participant is silent,
while the baseline model commits on one receipt — 276 of the 300 baseline rows carry the note "owner
COMMITTED from 1 receipt(s)" (MEASURED by re-reading `RES/exp13_unit_benchmark_trials.jsonl`).
Sent cost grows ~N·rounds against ~N; ingested grows ~N²·rounds — 3.39× and 19.1× at
N = 8 (`REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:224`, INHERITED-from-report but consistent with
the measured rows).

### E.4 Refusal accounting (re-derived by me from the 300 Exp-13 rows, both artefacts and both file formats)

`DUPLICATE` 2,532 + `POST_TERMINAL` 420 = **2,952 refused payloads**, `Σ refused_ingress = 2,952`, of
which `replayed_traffic` accounts for 2,544. MEASURED. The baseline reports 0 refusals in the same
condition "not because nothing is wrong, but because nothing is checked"
(`REP/EXP13_BASELINE_COMPARISON_20261020_210721.md:132`).

### E.5 Order-independence and the three probes

| Probe | Result | Label |
|---|---|---|
| `determinism_probe` (equal utilities, all orders) | Exp-13: **1 distinct winner** at N = 2 (2 permutations), N = 3 (6), N = 4 (24); baseline: **1 / 2 / 3** distinct winners (1 / 2 / 6 permutations) | MEASURED |
| `sequential_activation_probe` (one participant per turn) | Exp-13: 1 winner over 2 / 6 / 24 activation orders, 0 orders ending without a commitment | MEASURED |
| `order_independence_probe` (**FRESH artefact only**) | N = 2 / 3 / 4 / 6 / 8 → **28 / 35 / 42 / 98 / 112** schedules over 7 modes (`lockstep`, `sequential`, `late_inbox`, `late_delivered`, `duplicate`, `delayed`, `post_convergence`); distinct winners 1, tie rule honoured, 0 cross-agent disagreements, 0 dual claims, 0 no-commitment; **`exhaustive: false`** | MEASURED (FRESH), **bounded** — N ≥ 6 uses a seeded sample (6 permutations + 6 rotations + identity/reverse), not an exhaustive enumeration |
| `metric_selfcheck` (**FRESH artefact only**) | `all_passed: true`; 4 synthetic views incl. "one dissents of four" → actual 3 / expected 3 | MEASURED (FRESH). Its stated purpose is anti-vacuity for F-02: without it, a zero could not be told apart from a counter that never fires |
| Guard cost (`MIN_AGREEMENT_ROUND` 2 → 1) | Identical: mean rounds 3.32, converged 180, mean msgs 14.08 → clause (iv) **costs zero** under lockstep broadcast | INHERITED (`REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:240-247`) — the patched rerun is not in this package; the mechanism is stated in-code at `cbba_negotiator.py:98-99` |

**Important methodology note carried in the reports:** the **lockstep** probe cannot see the pre-F-04
defect at all (in round 1 no agent holds a peer belief), which is why `sequential_activation_probe`
exists; the earlier claim that the benchmark demonstrated the defect did not hold
(`REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:236`).

### E.6 Reproducibility, measured

* Suite: the **`edgesync_coordination` package** suite is **360 passed** in the fresh copy run, which
  equals the inherited 2026-10-21 record exactly. (Task 7's `edgesync_identity` suite separately
  reports 360 passed **with different companions** — 15 skipped + 1 error vs this suite's 3
  uncollected; the matching headline is a coincidence of two packages, not a shared run.)
* Benchmark: **588 of 600 trial rows byte-identical**; the 12 differing rows differ **only** in
  `convergence_time_s_max` (`TASK13_STATUS_AND_BLOCKERS.md:69-70`). MEASURED by re-execution.
* Every count-based aggregate above is identical between INHERITED and FRESH; only the three
  timing-derived fields move: mean interval 0.000383 → 0.000521 s, intervals ≤ one tick 238 → 235 of
  240, resolvable-above-tick n 2 → 5, simulator wall clock 0.235 → 0.484 s.
* Refusal classes and totals identical (re-derived independently, §E.4).

### E.7 Conflicting numbers across reports — preserved, not resolved

| Quantity | Report A | Report B | Packaged artefact | Note |
|---|---|---|---|---|
| Mean `convergence_time_s`, Exp-13 `__all__` | `EXP13_BASELINE_COMPARISON_…:62` → **0.00045 s**, ≤ tick **237** of 240, resolvable n = 2 | `EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:155` → **0.000383 s**, ≤ tick **238**, n = 2 | INHERITED **0.000383 / 238 / n=2**; FRESH **0.000521 / 235 / n=5** | Three different runs of the same seeded sweep. None is a speed measurement; the reports themselves say the tick boundary is clock alignment (§E.6) |
| Resolvable-above-tick n | 2 (both reports) | — | 2 (INHERITED), **5** (FRESH) | Sample of 2–5 out of 300; `EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:202` already says "supports no comparison" |
| `test_cbba_negotiator.py` count | **105 passed** (`EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:98, :87`) | 105 collected (same file) | **106 passed** (FRESH, 2026-10-21) | One test was added by the later F-02/F-03 pass; both numbers were true of their own pass |
| Full-suite exit code | **exit 2 — BLOCKED**, 3 collection errors (`…:100, :104`) | — | **exit 1**, 3 collection errors, with `--continue-on-collection-errors` (`FRESH_RUN_RECORD.json`) | The packaging run added that flag; the exit code differs because of the flag, not because the tests differ. Both record the same 3 uncollected modules |
| `cbba_negotiator.py` size | "970-line module" (`EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:72`) | — | **1,138 LF / 58,026 B** | Version drift: later passes added the epoch field, the ack leg and the probes |
| Was the benchmark rerun? | "**The benchmark was not rerun**" (`EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:112`) | — | Re-executed 2026-10-21 into `FRESH/benchmark_copy_repro/` | Both true in sequence: the 2026-10-21 delivery pass declined to rerun; the packaging pass re-executed to test reproducibility |
| `README_TASK13.md:172` | "The 600-episode benchmark was re-executed from the copied source, not re-read: **16 passed in 0.31s**" | — | the benchmark run is the artefact pair; "16 passed in 0.31 s" is the `test_exp13_benchmark_metrics.py` line | Documentation defect: a test count has been pasted onto the benchmark sentence. The underlying claim (re-executed) is correct and independently evidenced |
| Belief carrier `.msg` | `EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md` (AUD-11, §8 M3a era): belief has **no** epoch field | — | `ConsensusBelief.msg` exists (4,092 B, `2d34df85da0ffc9d…`) and `ConsensusMessage.ownership_epoch` is implemented at `:181/:650-651` | The audit predates the schema-2/schema-3 changes; see §G.1 for which AUD items are now closed |

No number was selected for favour; where an artefact and a report disagree, the artefact's stored row
values are given as MEASURED and the report's as INHERITED.

---

## F. Tests and validation status

### F.1 Executed fresh, in this package, on 2026-10-21 (`FRESH/FRESH_RUN_RECORD.json`)

```
command : python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors
cwd     : TASK13\source_snapshot\Implementation Phase\edgesync_ws\src\edgesync_coordination
result  : 360 passed, 3 errors during collection, exit code 1
tier    : Tier U + Tier N (stub rclpy) — NOT Tier R
evidence: FRESH/coordination_copy_run.txt
```

| Dedicated Exp-13 file | Fresh result | Evidence |
|---|---|---|
| `test_cbba_negotiator.py` | **106 passed** in 0.60 s | `FRESH/perfile_test_cbba_negotiator.py.log` |
| `test_exp13_adapter.py` | **76 passed** in 0.41 s | `FRESH/perfile_test_exp13_adapter.py.log` |
| `test_exp13_node_integration.py` | **101 passed** in 0.53 s | `FRESH/perfile_test_exp13_node_integration.py.log` |
| `test_exp13_benchmark_metrics.py` | **16 passed** in 0.31 s | `FRESH/perfile_test_exp13_benchmark_metrics.py.log` |
| `test_exp7_exp13_signature_integration.py` | **30 passed** in 0.53 s | `FRESH/perfile_test_exp7_exp13_signature_integration.py.log` |

Those five files contribute **329 of the 360**; the remaining 31 are pre-existing baseline-package
tests that the Exp-13 changes must not break (one of the backward-compatibility assertions,
`test_baseline_negotiation_manager_admission_is_unchanged`, actually lives inside
`test_cbba_negotiator.py` itself). **0 failed. 0 skipped.**

### F.2 Not collected, and never reported as passed

`tests/test_auto_trigger.py`, `tests/test_decision_and_receiver_services.py`,
`tests/test_response_collector.py` — all three `ModuleNotFoundError: No module named 'rclpy'`. These
are baseline-package tests importing `rclpy` at module scope. Nothing about their behaviour is known
from this package.

### F.3 Real dependencies vs stubs — read this before treating any pass as integration

| Layer | Reached? | What actually ran |
|---|---|---|
| **Tier U** unit / model simulation | **YES — executed** | Negotiator, adapter, metric modules on real Python objects; the 600-episode benchmark; 1,138-line negotiator under test, stdlib only |
| **Tier N** node tests on **stub** middleware | **YES — executed** | `test_exp13_node_integration.py` (101) under a stub `rclpy` with message classes **generated from the on-disk `.msg`/`.srv` text**. Exercises parameter declaration/dispatch and the ReidSignature intake (`exp13_coordination_node.py:535-619`). **This is not ROS 2 integration** |
| **Tier R** real ROS 2 / DDS | **NO — BLOCKED** | No `rclpy`, no generated `edgesync_msgs` Python module, no `launch`/`launch_ros`, no `ros2` CLI, no DDS implementation. `ConsensusBelief.msg` **has never been compiled**; `exp13_bringup.launch.py` **has never been executed**; no topic, service or QoS setting has ever been exercised |
| **Tier H / J** Jetson, TensorRT, physical robots | **NO — NOT MEASURED** | No such hardware on this host. Nothing in Task 13 was run on a robot |
| Fault injection | Partial | The 5-condition sweep in the harness is Tier U. `run_fault_injection.py` is present; no Tier-R fault run exists in this package |

Consequences that must not be glossed over:

* A passing Tier-N test proves the node's **code paths** under a fake parameter/driver API. It does not
  prove the parameter is accepted by a real daemon, that the QoS profile works, that two processes
  discover each other, or that an epoch survives a DDS boundary.
* The 0/60 `dead_participant` and 0/60 `one_silent_candidate` results are **model-level** outcomes of
  the agreement rule, not measured behaviour of a partitioned swarm.
* The `bringup` package fresh run (`FRESH/bringup_copy_run.txt`: 55 passed, 2 failed, 1 collection
  error) contains `test_robot_bringup_launch.py` failing to import `launch` — the environment gap, not
  a Task 13 code failure. Its 2 failures are Task 7 asset assertions and are out of scope here.

### F.4 Missing assets / dependencies that blocked execution

`rclpy`, ROS 2 launch, a DDS implementation, a colcon-built `edgesync_msgs`; the
`tests/fixtures/` synthetic TransReID assets (excluded by the packaging size policy, Task 7 side); and
any second machine or robot. The timing measurements were not blocked by a missing asset but are
invalid as latency claims because of the host clock's 0.015 s tick.

---

## G. Known limitations and blockers

### G.1 Implementation defects

| ID | State in the delivered code | Evidence |
|---|---|---|
| F-04 pre-round-1 tie-break (winner became first publisher) | **Fixed** in this arm; tie rule now applied at both decision sites; held by `test_a_belief_arriving_before_the_first_round_is_order_independent_at_the_episode_level` | `TASK13_STATUS_AND_BLOCKERS.md:31`; `cbba_negotiator.py:525`/`:684` |
| F-02 `committed_dual_ownership_total` computed from one agent's local view | **Fixed as reporting**: the committed-owner figure is now `null` where the commit path did not execute, and a cross-agent counter with an anti-vacuity self-test was added | `TASK13_STATUS_AND_BLOCKERS.md:32`; FRESH artefact fields `committed_dual_ownership_total`, `cross_agent_disagreeing_pairs_total`, `metric_selfcheck` |
| F-03 order-independence asserted without a denominator | **Fixed**: bounded deterministic probe with mode list and `exhaustive: false` recorded | `TASK13_STATUS_AND_BLOCKERS.md:33`; FRESH `order_independence_probe` |
| Equal bids resolved by arrival order | **Fixed**: tie → lexicographically smaller agent id, both sites | `TASK13_STATUS_AND_BLOCKERS.md:34` |
| **Open** — unilateral `AGREED` remains possible at the negotiator level | Not a defect claim against the lane but a live limitation: `check_agreement` is a local predicate; the ack leg narrows it, it does not close it — "a silence-shaped hole is closed only for peers that are talking" | `cbba_negotiator.py:761-778`; D3, `REP/EXP13_DESIGN_DECISIONS_20261020_210721.md:74` |
| **Open** — REQ-107 clause (iii) (recompute the merged **full bid set**) unimplemented | The carrier holds one winning bid per task (`winning_bids: Dict[str, float]`, `:176`), not the full bid set; `check_agreement` compares winner vectors. Deliberately not implemented because the requirement is unanswered (decision B-2/D1) | `REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:189` (status 3), AUD-17 `:334` |
| **Open** — `deadline_s` is sender-local | Belief timestamps are sender-local monotonic and never used for elapsed time; cross-host deadline has no clock basis | `exp13_cbba.yaml:78-81`; D7 |
| Superseded AUD items | AUD-05, AUD-11, AUD-18, AUD-19 (clause iv) describe the **pre-fix** module; they are closed in the delivered copy (epoch field `:181/:650-651`, `MIN_AGREEMENT_ROUND = 2` `:98`). AUD-06/07/08/10 concern **shared baseline files** and remain live | `REP/EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:322-336` vs the packaged `cbba_negotiator.py` |

### G.2 Integration gaps

* **B5 / the gating one.** The measured dual-`OWNER` witness is a **baseline** defect: one lost
  BEST_EFFORT `AcqStatus` → A's `on_confirm_received()` returns without transitioning, the node
  publishes `HandoffConfirmAck epoch=1` anyway, B becomes `OWNER` at epoch 1 while A stays `OWNER` →
  **two owners, same epoch**, and `e+2` recovery does not reconcile (AUD-08 `:325`, §3.3 `:162`;
  reproduced by an external saved probe, `EXP13_DECISION_AND_BLOCKER_REGISTER.md:74-110`).
  M1/M2/M3 would close it and "their correctness follows from the existing contract", but they edit
  `coordination_node.py` and `ownership_fsm.py`, files Task 13 is forbidden to modify. Hence
  `**BLOCKED PENDING DESIGN APPROVAL**` (`:400`). Attribution is explicit in the audit: "not caused by
  Exp-13" (`:325`).
* **No selector in production.** The baseline node has no `negotiation_protocol` (`:39-66`), so the
  guide's switch works only on the Exp-13 subclass node — which nothing outside the Exp-13 launch
  starts. P-1…P-4 patch proposals exist in `REP/EXP13_INTEGRATION_GAP_ANALYSIS.md:107-144` and are
  **written, not applied**.
* **`exp_id` has no consumer.** Declared/logged on the Exp-13 node; zero matches in the observability
  package. `exp13_cbba.yaml:27-28` overstates it as reaching "a real log line — on the Exp-13 node
  only", which is the accurate reading; the guide's "trigger automated logging into the results file"
  is **NOT VERIFIED**.
* **M3 measured prerequisite.** `open_transaction` has **zero** production call sites, so simply
  routing handlers through `NegotiationManager.admit_message` would reject every confirm/ACK/claim as
  an unknown transaction; a prerequisite M0 is therefore proposed, not applied
  (`REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:169`; the repo's own
  `test_phase5_chaos_races.py:58-62` already records the sibling dead-code fact).
* **B2 (cross-task, technically relevant here).** No `sensor_msgs/Image` crop producer exists anywhere
  in the delivered source — `perception_node` keeps its bounding box in-process and publishes only a
  3D centroid — so `exp13_bringup.launch.py:204-208` leaves `reid_target_crop_topic` empty on purpose.
  The signature leg is wired (`exp13_coordination_node.py:535-619`) but starved; the real
  identity-conditioned handoff path has never carried data.

### G.3 Experimental limitations

1. **Tier U only.** The baseline arm is a model of `coordination_node.py:441-545`, not the node; no
   conclusion transfers to the deployed node (rule §7.1 (10), `EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:153`).
2. **Application messages ≠ packets.** No serialisation, no DDS, no retransmit, no bytes.
3. **The clock is disqualified.** Tick measured 0.014999999999417923 s; 235–238 of 240 intervals are
   below it, and only 2–5 resolve above it — the boundary count moves between identical seeds.
4. **Declared vs swept participants.** The YAML declares `swarm_participant_count: 2`; the sweep runs
   N up to 8. The N > 2 rows "must not be reported as" the Tier-U configuration.
5. **Local-proxy ownership.** `dual_owner_violations_total = 0` is each agent's bundle against its own
   winning vector. Deployment dual ownership is **NOT MEASURED**; external REV04 `:421` (INV-09)
   assigns that property to tiers I + H and states Tier U cannot establish it.
6. **Arms are structurally asymmetric** — the four `non_equivalences_disclosed` items; the baseline's
   `agreed_by_all = 0` and `refused = 0` are definitional, and its `winner_matched_expected` is
   tautological.
7. **No tuning was applied** to improve any figure: thresholds, `max_rounds`, `deadline_s`,
   `bid_epsilon`, `commit_grace_ms` are the shipped YAML values, unchanged against these results
   (`EXP13_BASELINE_COMPARISON_…:171-173`, audit `:364-367`).
8. **Probes are bounded, not proofs.** The order-independence probe is explicitly `exhaustive: false`;
   the audit's state-space probe (247,510 states, N ∈ {2,3}, depth cap 8–10) is exhaustive only within
   those bounds and models no loss, partition or clock skew (`EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:389-391`).

### G.4 Missing infrastructure or assets

No `rclpy`/DDS/launch/`ros2` CLI (3 baseline test modules uncollected); `colcon` present but ROS
absent; `edgesync_msgs` never built, so no generated message module ships; no Jetson, no TensorRT, no
robots; large shared assets referenced by digest rather than copied (see `README.md` §5); the design
authorities (`ES-EXP13-PROPOSAL-01-REV01…REV04.md`) live **outside** the repository under
`C:\Users\PC\edgesync_es13_proposal\` and are not part of this package (AUD-22, `:339` — a teammate
working from the repo alone cannot recover the REQ/T/INV contract the code was written against).

### G.5 Decisions requiring the team leader's or supervisor's approval

The full register is `REP/EXP13_DECISION_AND_BLOCKER_REGISTER.md:51-74` (B-1…B-12) and
`REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:196` (status 10). Headlines, with the register's own
caution at `:24` — **"No approval below is granted, assumed, or implied."**

* **D1 / A1 — the algorithm label.** CBAA (L = 1) vs IBBM-1 vs "CBBA"; the in-repo record forbids the
  CBBA label in reports/config keys/log lines while the team execution guide assigns the file name
  `cbba_negotiator.py` and the token `cbba`; `CBBA-VARIANT-DECISION-RECORD.md:19` records that the
  guide itself was **absent from the repository** at the time and "cannot be read" from it.
* **A2 / B-11 — production participant count** (the declared 2-UGV pair vs the N > 2 sweep).
* **B-2 — is the full bid set mandatory** (REQ-107 clause iii), which decides whether the current
  winner-vector carrier is compliant.
* **B-3 — the meaning of `AGREED`**: local belief vs distributed commitment; whether an ack/quorum/lease
  is authorised at all (audit M5).
* **B-10 — M0–M4 ownership**: who edits `coordination_node.py` / `ownership_fsm.py` /
  `epoch_manager.py` / `negotiation_manager.py`. The remediation proposal is explicit that its items
  were **not applied** because every one edits a shared file and no approval of record exists
  (`REP/EXP13_SAFETY_REMEDIATION_PROPOSAL.md:5, :212-232`; 18 regression tests are specified to be
  written *with* an approved patch, not before).
* **B-7 / D10 — the belief `.msg`** and who rebuilds/deploys `edgesync_msgs`.
* **Liveness vs safety (B1)** — keep the conservative no-unilateral-agreement rule and accept
  non-termination under partition, or add a timeout/quorum fallback. Nothing was changed during
  packaging; inventing a failure detector, quorum, lease or transaction protocol is outside this
  task's authority.

**Do-not-claim list**, taken from the reports and binding on any downstream text
(`EXP13_SAFETY_FINDINGS_20261020_210721.md:239-253`, `EXP13_SAFETY_REMEDIATION_PROPOSAL.md:226-232`):
the overall verdict is **"SAFE TO RUN AS AN ISOLATED EXPERIMENT ARM, NOT CLEARED FOR DEPLOYMENT"**
(`:251`); `fsm_witness.py` must not be described as a ROS 2/DDS test; no message-count figure may be
quoted as network overhead; no sub-tick interval may be quoted as convergence speed; zero local
dual-owner counters may not be quoted as a safety proof.

---

## H. Decisions and next steps (prioritised)

| Pri | Concrete issue | Why it matters | Evidence | Verification needed to close | Type |
|---|---|---|---|---|---|
| **1** | Baseline dual-`OWNER` path (M0–M3) unfixed; Exp-13 would inherit it | It violates the system's own zero-dual-ownership invariant and is reachable by losing **one** best-effort sample; it is the reason the safety audit blocks integration | `EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:325, :346-362, :400`; `EXP13_DECISION_AND_BLOCKER_REGISTER.md:74-110` | Written approval to edit `coordination_node.py` + `ownership_fsm.py`, the approved patch, the 18 specified regression tests, and a Tier-R run showing one owner per target | **Owner decision → then technical change** |
| **2** | Liveness under partition: 0/60 in two fault conditions at every N | A handoff that never terminates is an operational failure regardless of safety | §E.2 rows; `config_consistency`/D4 in `EXP13_DESIGN_DECISIONS_20261020_210721.md:109-133` | A ruling on the agreement rule (keep conservative vs add a bounded fallback), then re-run the same 5-condition sweep and report the new counts | **Owner decision → then experiment** |
| **3** | `negotiation_protocol` does not exist on the production node, so the guide's switch cannot be operated as specified | Without it no ablation run can select CBAA on the real system | `coordination_node.py:39-66` (24 params, verified); `EXP13_INTEGRATION_GAP_ANALYSIS.md:107-144` (P-1…P-4) | Apply P-1/P-3 with approval, then `ros2 param get` + a two-node run showing the CBAA arm selected and a third value rejected at startup | **Owner decision → then technical change** |
| **4** | Tier R has never run: `.msg` uncompiled, launch unexecuted, 3 modules uncollected | Every claim in §E is Tier U; nothing is ROS 2 or DDS validated | `FRESH_RUN_RECORD.json → blocked_validation`; `TASK13_STATUS_AND_BLOCKERS.md:92-95` | Install ROS 2, `colcon build`, `pytest` per package, `ros2 launch edgesync_bringup exp13_bringup.launch.py`, capture topic/QoS/epoch behaviour | **External dependency (environment)** |
| **5** | `exp_id` is a label, not a logging key; no consumer | The guide's "automated logging into the results file" is not achieved for either experiment | `grep exp_id` in `edgesync_observability` = 0 matches; `EXP13_IMPLEMENTATION_AND_TEST_REPORT.md:86` (R15) | Either point `exp_id` at an existing per-experiment `--out` convention or formally retire the requirement | **Owner decision** |
| **6** | REQ-107 clause (iii) unimplemented; carrier holds one winning bid per task | Decides whether the delivered allocator is contract-compliant at all | `cbba_negotiator.py:176`; `EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:189`; AUD-17 `:334` | Ruling on B-2; if mandatory, schema bump (full bid set), re-implement recompute, re-run probe suite, re-derive the benchmark | **Owner decision → then technical change** |
| **7** | Label decision D1/A1 unresolved while deliverables carry both names | Reviewers will cite "CBBA" for an algorithm the code says is CBAA L = 1 | `EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:329`; `CBBA-VARIANT-DECISION-RECORD.md:19` | A supervisor designation; then update the reporting-only strings, not the token | **Owner decision** |
| **8** | No image-crop producer, so the identity-conditioned handoff never receives a signature in a real run | The Exp-7 ↔ Exp-13 value proposition is untested end to end | `exp13_bringup.launch.py:37-40, :204-208`; `INTEGRATION/OPEN_INTEGRATION_BLOCKERS.md` | Perception/acquisition to publish the tracked crop with target identity, then a Tier-R two-node run with real signatures | **Cross-task dependency** |
| **9** | Timing is unmeasurable on this host, so "convergence time" is an open metric in both arms | A required comparison metric currently has no evidence | §E.1, §E.7 | Measure on a Tier R clock with explicit start/end events, or replace the metric and say so | **Experiment** |
| **10** | Design authorities are outside the repository (unversioned external proposal REV01–REV04) | Reproducibility of the requirement set itself | AUD-22 `:339`; `C:\Users\PC\edgesync_es13_proposal\` (external, not in this package) | Land the approved proposal revision under `docs/experiments/exp13/` in the project | **Owner decision / documentation** |

No item above is assigned to a named individual; the report register uses **roles** (supervisor,
coordination owner, bringup owner, perception/acquisition) and that is preserved here
(`EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md:200-208`, W1–W5).

---

## I. File and evidence index

Paths are package-root-relative. `SNAP`, `RES`, `REP`, `FRESH` are defined at the top of this file.

| Purpose | Path | Notes |
|---|---|---|
| Entry point, benchmark design, honest limits | `TASK13/README_TASK13.md` | §4 design, §5 results, §6 proxy metrics, §10 tiers |
| Status, defects, blockers, conflicts | `TASK13/TASK13_STATUS_AND_BLOCKERS.md` | §2 F-04/F-02/F-03/tie, §3 limits, §4 B1–B6, §5 preserved conflicts |
| Implementation, primary | `SNAP/edgesync_coordination/edgesync_coordination/negotiation/cbba_negotiator.py` | 58,026 B / 1,138 LF, `002045be0d346e51…` |
| Adapter + commit decision | `SNAP/edgesync_coordination/edgesync_coordination/negotiation/exp13_cbba_adapter.py` | 45,609 B / 866 LF |
| Node subclass (selector + signature intake) | `SNAP/edgesync_coordination/edgesync_coordination/nodes/exp13_coordination_node.py` | 54,578 B / 962 LF |
| Baseline node (shared, unmodified, no selector) | `SNAP/…/nodes/coordination_node.py` and `SHARED_DEPENDENCIES/source_snapshot/…/nodes/coordination_node.py` | 618 LF; `:39-66`, `:423`, `:441-545` |
| Baseline FSM / epoch (shared, hazard live) | `SHARED_DEPENDENCIES/…/fsm/ownership_fsm.py`, `…/fsm/epoch_manager.py` | Off-limits to Task 13 |
| Experiment YAML (configs copy) | `TASK13/configs/edgesync_bringup/config/exp13_cbba.yaml` | 11,618 B / 149 LF, `8cd9001f412e1108…` |
| Same YAML inside the snapshot | `SNAP/edgesync_bringup/config/exp13_cbba.yaml` | Kept so the launch path is traceable |
| Launch | `SNAP/edgesync_bringup/launch/exp13_bringup.launch.py` | 11,774 B / 225 LF; never executed |
| Baseline launch (proves the YAML is inert for baseline runs) | `SNAP/edgesync_bringup/launch/robot_bringup.launch.py` | `:38-48`, `:148-151` |
| Belief message definition | `SNAP/edgesync_msgs/msg/ConsensusBelief.msg` | 4,092 B, `2d34df85da0ffc9d…`; never compiled |
| Signature message consumed by the arm | `SNAP/edgesync_msgs/msg/ReidSignature.msg` | 2,171 B, `44fcb363e51881ed…` (Task 7's type, consumed here) |
| Benchmark harness | `SNAP/edgesync_coordination/edgesync_coordination/scripts/run_exp13_unit_benchmark.py` | 76,660 B / 1,366 LF |
| Main results, INHERITED | `RES/exp13_unit_benchmark.json`, `…_trials.jsonl` (600 rows, 1,030,180 B), `…_table.md` (295 lines) | json sha `207f41c38b75374c…`; mtime 2026-10-20 19:16 |
| Main results, FRESH | `FRESH/benchmark_copy_repro/exp13_unit_benchmark.json`, `…_trials.jsonl` (1,227,552 B), `…_table.md` (354 lines), `console.txt` (90,864 B) | Re-executed 2026-10-21 from the copies |
| Test execution record (all 16 runs) | `FRESH/FRESH_RUN_RECORD.json` | `blocked_validation` lists Tier R / Tier H / Task 7 OSNet |
| Suite logs | `FRESH/coordination_copy_run.txt`, `FRESH/perfile_test_cbba_negotiator.py.log`, `…test_exp13_adapter.py.log`, `…test_exp13_node_integration.py.log`, `…test_exp13_benchmark_metrics.py.log`, `…test_exp7_exp13_signature_integration.py.log` | Per-file counts in §F.1 |
| Inherited test records | `VALIDATION/inherited_evidence/exp13_20261020/*`, `VALIDATION/inherited_evidence/exp13_20261021/*` | Includes F-04 pre/post repro, mutation check, per-file counts |
| Requirements acceptance (R1–R25) | `REP/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md` | `:66-96` checklist, `:100-136` defect→fix→test, `:168-267` benchmark, `:271-308` limitations |
| Design decisions D1–D8 | `REP/EXP13_DESIGN_DECISIONS_20261020_210721.md` | D3 ack leg `:74`, D4 participant rules `:109`, D5 additive msg `:134`, D6 selector `:175`, D7 topology `:202`, D8 safety seam `:221` |
| Baseline comparison | `REP/EXP13_BASELINE_COMPARISON_20261020_210721.md` | `:39-74` metrics, `:88-105` fault conditions, `:113-136` what Exp-13 buys, `:157-173` confounds |
| Safety audit (the gate) | `REP/EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md` | AUD-01…AUD-24 `:314-343`, M-series `:346-367`, verdict `:400` |
| Safety findings S-1…S-10 | `REP/EXP13_SAFETY_FINDINGS_20261020_210721.md` | `:239-253` limits + verdict |
| Remediation M1–M4 (proposed, unapplied) | `REP/EXP13_SAFETY_REMEDIATION_PROPOSAL.md` | `:5` why unapplied, `:184-211` 18 tests, `:226-232` do-not-claim |
| Decision / blocker register | `REP/EXP13_DECISION_AND_BLOCKER_REGISTER.md` | Gate 1a `:13`, authorities `:36`, B-1…B-12 `:51`, witness `:74-118` |
| Integration gaps + patch proposals | `REP/EXP13_INTEGRATION_GAP_ANALYSIS.md` | 9 inventory items `:10-106`, P-1…P-4 `:107-144`, runtime verification blocked `:144-156` |
| Variant decision record (CBAA L = 1) | `REP/CBBA-VARIANT-DECISION-RECORD.md` | `:19` guide absent, §1/§5 label rule, §7 unresolved owner input |
| Final statuses (10 required) | `REP/EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md` | `:183-196` |
| Manifests | `TASK13/manifests/ES-EXP13-FINAL-DELIVERABLE-MANIFEST.md`, `TASK13/manifests/EXP13_FINAL_DELIVERY_MANIFEST_20261020_210721.md` | Note: older manifests refer to a `test_benchmark.py` that does not exist (§5 of the status doc) |
| Cross-task dependency (only where technical) | `INTEGRATION/TASK07_TASK13_DEPENDENCY_MAP.md`, `INTEGRATION/OPEN_INTEGRATION_BLOCKERS.md`, `INTEGRATION/ES-EXP07-EXP13-INTEGRATION-AUDIT-20261021_011521.md` | ReidSignature + clock/quality contract |
| Package inventory | `MASTER_FILE_INDEX.csv` (466 rows; `task == "TASK13"` for 118), `FINAL_PACKAGE_MANIFEST.json` | Original absolute paths + both digests |

**Referenced but not included in this package** (so their contents could not be independently
verified from the package alone, and each is labelled external where cited above):
`C:\Users\PC\edgesync_es13_proposal\ES-EXP13-PROPOSAL-01-REV04.md` (84,810 B, 913 lines) and REV01–REV03;
`C:\Users\PC\edgesync_es13_probe\gate1b_20261020\fsm_witness.py` (15,476 B) + `witness_console.txt`
(10,246 B / 162 lines); `C:\Users\PC\edgesync_es13_runs\…` benchmark console transcripts and the rerun
directory; the pre-edit backups `C:\Users\PC\edgesync_es13_backup\…`; the Team Execution Guide itself at
`C:\Users\PC\OneDrive\Documents\EdgeSync_Team_Execution_Guide.pdf` (the in-repo decision record
explicitly notes the guide was not readable from the repository); `tests/fixtures/` synthetic assets and
the large model/dataset files. The REQ/T/INV identifiers cited in §B (REQ-107/108/109/110, AC-C3,
INV-01, INV-09) come from the external proposal, not from a package file.

---

## J. Final status statement

**Complete.** The Exp-13 arm exists as real, self-consistent code: a 1,138-line CBAA negotiator, an
866-line adapter that owns a safe-by-default commit decision, a node subclass carrying the
`negotiation_protocol` selector and the Exp-7 signature intake, one additive `.msg`, one experiment
YAML, one launch file, one 1,366-line benchmark harness, and five dedicated test files. The metric
definitions, the disclosure blocks (`non_equivalences_disclosed`, `keys_not_consumed`,
`hardcoded_fields_removed`, `dual_owner_basis`) and the fail-fast behaviours are in place. Four defects
(F-04, F-02, F-03, the arrival-order tie) were fixed and are held by named tests. Nothing in the
shared baseline was modified.

**Demonstrated by measured evidence (Tier U / Tier N only).** 360 coordination tests pass with 3
baseline modules uncollected; a 600-episode two-arm benchmark ran and was re-executed with 588/600
rows byte-identical. Within that scope the arm shows deterministic, order-independent winner
selection (1 winner across 2/6/24 orders, bounded probes at N ≤ 8), operating admission control
(2,952 classed refusals; 42.4/episode under replay while still converging 60/60), all-participant
agreement in 180 episodes where the baseline structurally has 0, and 0 false-convergence events. It
also shows, equally measured, that it costs **3.35× the application messages**, **14.2× the ingested
payloads**, **3.32 rounds against a definitional 1**, and **fails to terminate in 120 of 300 episodes**
across two fault conditions at every N tested.

**Incomplete or blocked.** No production selector on the baseline node; `exp_id` reaches no logging
consumer; REQ-107 clause (iii) and any distributed-commitment guarantee are unimplemented pending
decision; unilateral `AGREED` remains possible at the negotiator; the baseline dual-`OWNER` defect is
reproduced and unfixed because the two files that close it are off-limits, leaving the safety audit at
**BLOCKED PENDING DESIGN APPROVAL**; the algorithm label (D1/A1) is unratified; no image-crop producer
means the identity-conditioned handoff has never carried data.

**Readiness, separated as required.**
* *Ready for review* — yes, as a Tier-U/Tier-N artefact, on the evidence in this folder.
* *Ready for integration* — **no.** Integration is gated by owner approvals (M0–M3, P-1…P-4, D1) that
  do not exist, and by a Tier-R environment that this host cannot provide. `ConsensusBelief.msg` has
  never been compiled and `exp13_bringup.launch.py` has never been executed, so no statement of the
  form "ROS 2 validated", "DDS tested" or "multi-robot verified" is supportable for Task 13.
* *Ready for deployment validation* — **not claimable.** The strongest defensible statement is the
  audit's own: **"SAFE TO RUN AS AN ISOLATED EXPERIMENT ARM, NOT CLEARED FOR DEPLOYMENT."** A
  single-process simulation establishes neither ownership safety nor swarm behaviour under partition,
  and no deployment-level guarantee may be inferred from these numbers.

*This summary does not make Task 13 complete. It records what Task 13 has and has not evidenced.*
