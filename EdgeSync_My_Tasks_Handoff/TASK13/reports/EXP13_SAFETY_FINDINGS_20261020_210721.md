# EXP13 — Safety findings (autonomous pass `20261020_210721`)

**What is new here:** the nine findings below were surfaced by *executing* the arm against the real node
handlers, not by reading the baseline. S-1…S-6 are defects introduced or exposed by the experiment's own
integration and are **fixed in the arm with a named regression test each**. S-8/S-9 are defects that exist in
the **deployed baseline** and are *not* fixed there — brief §5 keeps shared files conservative, and the baseline
patch stays as the proposal in `EXP13_SAFETY_REMEDIATION_PROPOSAL.md`. S-10 is the prior dual-OWNER witness,
restated only as a pointer plus its current status.

Every claim in this file is either (a) a code path with a line reference, or (b) a test that was run. No finding
is marked "prevented" unless the listed test passes in the executed suite
(`C:\Users\PC\edgesync_es13_runs\completion_20261020\phase_g_final_test_execution_20261020.txt`).

Line references: `exp13_coordination_node.py` (arm node), `exp13_cbba_adapter.py` (adapter),
`coordination_node.py` (deployed baseline, unmodified), `ownership_fsm.py` (shared, unmodified).
Test names are in `tests/test_exp13_node_integration.py` unless prefixed `adapter:` → `tests/test_exp13_adapter.py`
or `neg:` → `tests/test_cbba_negotiator.py`.

---

## S-1 Arm-owned state was read before it existed, aborting node construction

**Mechanism.** The baseline `CoordinationNode.__init__` publishes its initial ownership event and heartbeat.
With synchronous callback delivery — intra-process comms, a concurrent executor, or the fake middleware used
here — a subscription callback can be delivered *while the parent constructor is still running*. The overrides
added for this arm (`_on_local_heartbeat`, `_on_handoff_response`, `_drive_round`) read `self.adapter`, which
was only assigned later in the subclass body.

**Consequence.** `AttributeError` inside a callback during construction; the node never finished starting.
Observed, not theorised: **73 of the 101 arm tests failed** with
`'Exp13CoordinationNode' object has no attribute 'adapter'` on the first execution of the suite in this pass.

**Fix.** Every arm-owned attribute is declared before `super().__init__()` (`exp13_coordination_node.py:67-90`):
`adapter`, `pub_belief`, `sub_belief`, `peers`, `timer_round`, `_round_period_s`, `_exp13_episode_target`,
`_exp13_heartbeat_seen`, `exp_id`, `negotiation_protocol`. The guards therefore always have something to read,
and `adapter is None` means "not armed" rather than "not yet assigned".

**Evidence.** The suite as a whole is the regression (101 passed, 0 skipped, 0 error). The specific re-entrant
paths are `test_the_cbba_selector_builds_the_adapter_and_the_belief_endpoints`,
`test_an_unconfigured_node_gets_the_declared_baseline_default`,
`test_a_peer_heartbeat_does_not_overwrite_local_telemetry`.

---

## S-2 M4 — a verdict for a superseded episode could hand the target over

**Mechanism.** `arm_commit()` caches its `CommitDecision` (`exp13_cbba_adapter.py:686`, `_set_decision` `:804`).
The node can be holding a COMMIT for episode *E* while the FSM has already moved to a different active
transaction (abort → reclaim at e+2 → new attempt). Applying the cached verdict names a receiver in the *current*
FSM, so ownership of the target transfers under a transaction id that no longer exists.

**Consequence.** A silent, unattributed transfer — the worst class in this experiment, because the log and the
ownership event disagree about which attempt produced it.

**Fix.** `_apply_commit` (`exp13_coordination_node.py:348`, guard at `:373`) compares
`decision.episode_id` with `fsm.active_transaction_id` for both COMMIT and ACQUIRE; on mismatch it logs an error
naming "not the FSM's active transaction" and closes the attempt *without* transfer. `ACQUIRE` is covered by the
same gate, which matters because ACQUIRE starts an acquisition task rather than announcing ownership.

**Evidence.** `test_a_commit_aimed_at_another_transaction_does_not_hand_the_target_over`,
`test_an_acquire_aimed_at_another_transaction_starts_no_acquisition`,
`test_an_acquire_for_the_episode_the_fsm_is_in_is_accepted_without_a_claim`.

---

## S-3 Applying a closing decision left the adapter RUNNING, disabling every later trigger

**Mechanism.** `CBBANegotiator.abort()` (`cbba_negotiator.py:913-920`) is a no-op unless status is RUNNING, and
`Exp13CbbaAdapter.start_episode` (`:448`) refuses to open an episode while `self._negotiator` is non-terminal.
A node that *applied* a COMMIT/ABORT/RETAIN outcome (i.e. acted on it) never transitioned the adapter out of
RUNNING — it only stopped the timer.

**Consequence.** The next handoff trigger in the same process reached `start_episode`, was refused, and the node
handed over nothing while appearing healthy. One closed episode silently disabled negotiation for the lifetime
of the robot.

**Fix.** The `closes_episode` branch of `_apply_commit` now stops the round driver **and** aborts the adapter
episode (`exp13_coordination_node.py:348` ff.), so "acted upon" and "terminal" cannot diverge.

**Evidence.** `test_a_second_trigger_never_silently_replaces_a_running_episode`,
`test_a_closing_decision_stops_the_round_driver`, `test_the_driver_is_rearmed_for_the_next_attempt`,
`test_a_watchdog_cutoff_ends_the_episode_it_is_abandoning`.

---

## S-4 An idle tick after closure re-applied the verdict and advanced the epoch twice

**Mechanism.** The round timer fires independently of the FSM. After a closing decision had already been applied
once, a subsequent tick could call `arm_commit()` again and re-run `_apply_commit`. Each
`_close_attempt_without_transfer` runs `abort_to_recovery()` + `recovery_to_solo_tracking()`, and that pair
advances the epoch **e+2 per call** — so a stale tick bumps the epoch for an attempt that had already been
closed, fencing beliefs that belong to the *new* attempt.

**Fix.** `_drive_round` (`exp13_coordination_node.py:291`) returns when the adapter has no negotiator **or**
the negotiator `is_terminal()` (`cbba_negotiator.py:1040-1041`). A closing decision is applied exactly once.
Note that `arm_commit()` returns the *cached* verdict, so an abort-after-apply cannot relabel an earned COMMIT —
the cache is the reason this bug was invisible at the adapter layer and only appears at the node layer.

**Evidence.** `test_a_tick_after_the_episode_closed_does_not_re_apply_the_verdict`,
`test_driving_with_no_episode_open_does_nothing_at_all`.

---

## S-5 The watchdog cutoff abandoned an episode but left it negotiating

**Mechanism.** `_on_watchdog_timeout` in the baseline (`coordination_node.py:289-294`) aborts to RECOVERY and
reclaims at e+2 when a candidate goes silent. Under `cbba` the round driver was still ticking at that moment, so
the node kept publishing beliefs for a transaction it had just given up on, and left a RUNNING episode behind
(S-3's mechanism).

**Fix.** Override at `exp13_coordination_node.py:424`: stop the driver, `adapter.abort("watchdog cutoff:
candidate silent", is_owner=...)`, close that transaction in the negotiation table, then delegate to the
inherited handler. The closing decision is deliberately **not** applied inside the override — the inherited
`abort_to_recovery()` owns the epoch, and doing both would double-advance it.

**Evidence.** `test_a_watchdog_cutoff_ends_the_episode_it_is_abandoning`, and the control-arm twin
`test_the_control_arm_keeps_the_watchdog_exactly_as_the_baseline_runs_it`, which proves the baseline watchdog is
untouched when the adapter is absent.

---

## S-6 A trigger the arm refused left its own transaction OPEN

**Mechanism.** M0 registers the trigger's transaction so that admission checks have a real record to validate
against (`exp13_coordination_node.py:505`). If `start_episode` then raised `ProtocolConfigError` — participant
set refusing to fit, `max_bundle_size > 1`, deadline/round configuration rejected — the handler logged the
refusal and returned, but the transaction it had just registered stayed in the table.

**Consequence.** A permanently OPEN transaction for a handoff that never happened: a leak that also makes
`active_transaction_id` point at something no agent will ever commit. Rejecting unknown transactions is only
safe if legitimate ones are properly closed first — exactly the check brief §6 warns about ("avoid introducing a
check that blocks every valid handoff").

**Fix.** The `except ProtocolConfigError` branch in `broadcast_handoff_request` now closes the transaction it
opened and records the no-transfer attempt.

**Evidence.** `test_a_trigger_that_reached_no_transaction_opens_no_episode`,
`test_a_participant_set_that_omits_this_node_is_repaired_and_reported`,
`test_a_multi_target_episode_is_refused_rather_than_allocating_one_of_n`,
adapter-side `adapter:` C3 participant-refusal cases.

---

## S-7 A late infeasibility report must downgrade a cached COMMIT (behaviour, not a bug)

**Mechanism.** Under `cbba`, a `HandoffResponse` REJECT is a *feasibility report*, not a vote
(`exp13_coordination_node.py:601`). Consensus can still award the vetoing agent. The veto is recorded via
`mark_infeasible` (`exp13_cbba_adapter.py:569`), which downgrades a cached COMMIT to ABORT.

**Verified property.** `test_a_veto_that_lands_after_the_computed_commit_still_stops_the_transfer` (a veto that
arrives *after* the commit was computed still prevents the transfer) and
`test_a_reject_vetoes_that_agent_for_the_commit`. The complementary honest statement is
`test_an_accept_is_reported_but_selects_no_receiver_in_this_arm` — an ACCEPT never chooses a receiver in this
arm, and `_on_negotiation_timeout` (`:620`) closes the window selecting nothing.

---

## S-8 Any node that closes a failed attempt declares itself OWNER — **baseline hazard, unfixed**

**Mechanism.** `abort_to_recovery()` sets `role = OWNER` (`ownership_fsm.py:126-135`) and
`recovery_to_solo_tracking()` does the same (`:137-143`). Both are the baseline's normal exit from a failed
attempt. A candidate — whose role is CANDIDATE after `accept_request` (`ownership_fsm.py:160`) — that runs this
path emerges claiming ownership of a target it never held.

**Status.** Not fixed in `ownership_fsm.py`: it is a shared file used by Experiments 12/14 and the deployed node,
and brief §5/§6 keep the fix separate from the consensus implementation. The arm instead refuses to run the
recovery path unless it already holds the target — `if self.fsm.role is not OwnershipRole.OWNER:` logs an error
and returns (`exp13_coordination_node.py:458-463`).

**Evidence.** `test_a_non_owner_never_runs_the_recovery_path_even_when_the_episode_closes`,
`test_an_owner_closes_the_attempt_at_e_plus_2_and_resumes_solo_tracking`,
`test_the_subclass_keeps_the_constructors_signature`.
**Owner action required:** a decision on whether the FSM's recovery path should set role at all.

---

## S-9 M3 — a peer heartbeat overwrites local telemetry that then drives bidding — **baseline hazard, fixed in arm only**

**Mechanism.** `state_heartbeat` is in the swarm remap list (`robot_bringup.launch.py:84-92`), so both robots'
heartbeats arrive on one shared topic and the baseline handler is last-writer-wins. `_my_telemetry` feeds the
hard gate G(j) and `UtilityEngine.compute_receiver_utility`, i.e. a peer's battery level can change *this*
agent's bid and feasibility verdict.

**Status.** Fixed in the arm by adopting only this robot's own `sender_id`
(`exp13_coordination_node.py:639`, filter `:642`, refusal log `:662`) before delegating to the inherited
telemetry update. The deployed baseline still has the hazard.

**Evidence.** `test_a_peer_heartbeat_does_not_overwrite_local_telemetry`,
`test_the_peers_heartbeat_still_reaches_the_shared_swarm_topic`,
`test_own_heartbeat_is_adopted_and_enables_bidding`,
`test_bidding_uses_the_live_battery_rather_than_the_parameter_default`,
`test_an_episode_opened_without_telemetry_abstains_instead_of_bidding_on_the_default`, and the disclosure test
`test_the_control_arm_keeps_the_baselines_last_writer_wins_telemetry`.

---

## S-10 Prior dual-OWNER witness (M0/M1/M2) — status carried forward

`EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md` recorded a reproducible **dual-OWNER** outcome in the deployed
baseline: an unconditional ACK plus epoch-agnostic ownership adoption. The arm closes that path with M1 (ACK
only an accepted CONFIRM, known transaction — `exp13_coordination_node.py:669`) and M2 (ownership only on
`ACK(e+1)` for the active transaction — `:710`).

**Evidence in the arm.** `test_an_accepted_confirm_is_acked_at_e_plus_one_and_recorded`,
`test_a_confirm_the_fsm_refuses_gets_no_ack`, `test_a_refused_confirm_does_not_disarm_the_watchdog`,
`test_an_ack_at_e_plus_one_for_the_active_transaction_transfers_ownership`,
`test_an_ack_that_is_not_e_plus_one_or_not_this_transaction_takes_nothing`,
`test_a_replayed_ack_cannot_announce_ownership_twice`,
`test_a_unsafe_dual_claim_transfers_nothing_and_is_logged_as_a_violation`,
plus the control-arm twins `test_the_control_arm_keeps_the_baselines_unconditional_ack` and
`test_the_control_arm_still_takes_whatever_epoch_the_peer_names`.

**What is *not* established:** the baseline node itself is unchanged (`coordination_node.py` digest verified
byte-identical), so **the deployed system retains the dual-OWNER defect**. `test_the_deployed_coordination_node_source_has_no_experiment_branch`
proves the fix was not smuggled into the shared node, which is the correct outcome under §5 but does not make
production safe.

---

## Limits of this evidence — read before quoting any number in this file

1. **`dual_owner = 0` in the Tier-U benchmark is not a safety proof.** That counter lives in the harness's
   single-process ownership model (`run_exp13_unit_benchmark.py`). It shows the model did not produce two
   owners over 300 exp13 + 300 baseline episodes. It says nothing about DDS delivery, QoS mismatch, or a real
   FSM interleaving. The guards above are demonstrated at node level, under stub middleware.
2. **No ROS 2 execution.** `rclpy`, `edgesync_msgs`, `launch`, `launch_ros` are all absent and `ros2` is not on
   PATH (probed; see the test report §4). `ConsensusBelief.msg` has never been compiled by `rosidl`;
   `test_a_published_belief_survives_the_message_definition_field_by_field` checks the codec against the
   definition text, not against a generated type.
3. **Stub delivery ordering is not DDS ordering.** The fake middleware delivers synchronously, which is *stricter*
   than the real graph for re-entrancy (S-1) and *weaker* for concurrency (no executor threads, no serialization
   boundary, no wall-clock timer jitter).
4. **The cbba arm is not the deployment.** Its delta is exactly M0–M4 + the round timer + the belief endpoints,
   and those guards fix hazards for the experiment only.
5. **No robot, no network, no Jetson.** No claim in this file depends on one.

---

## Verdict

* **Standalone implementation complete** — yes, for the selected contract (see the test report).
* **Integration implementation complete** — yes at the code level: the selector reaches the adapter through a
  documented launch path (§2 of the implementation report) and the adapter is invoked by a runtime timer.
* **Locally validated** — yes: 282 Exp-13 tests and 313 package tests executed, all passing, none skipped.
* **ROS 2 integration validated** — **BLOCKED — Pending ROS 2 runtime.** Nothing in this pass can be called a
  ROS 2 test.
* **Network/robot validated** — **NOT STARTED** (no hardware on this host; owner action).
* **Performance improvement demonstrated** — **NO.** The measured comparison is unfavourable on both official
  metrics; see `EXP13_BASELINE_COMPARISON_20261020_210721.md`.

**Overall: SAFE TO RUN AS AN ISOLATED EXPERIMENT ARM, NOT CLEARED FOR DEPLOYMENT.** Two baseline hazards
(S-8, S-9) and the S-10 dual-OWNER defect remain live in the shared code by design, and every guard is verified
only against stub middleware.
