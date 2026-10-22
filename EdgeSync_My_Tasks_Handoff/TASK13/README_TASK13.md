# Task 13 — CBAA/CBBA decentralised target handoff

Owner: Task 13 (me). Everything here is a **copy** from
`C:/Semester 7/EdgeSync-main (3)Last/EdgeSync-main (3)Last/EdgeSync-main`; the project tree was not
written to by this packaging run.

## 1. What the experiment implements

**CBAA is implemented as CBBA with bundle length L = 1.** In `exp13_cbba.yaml`:
`negotiation_protocol: cbba` is the selector token that chooses the implementation, and
`algorithm_variant: CBAA (CBBA with bundle length L = 1)` is **reporting-only** — it does not switch code.

| File (original relative path) | Bytes | Role |
|---|---:|---|
| `cbba_negotiator.py` | 58,026 | The CBAA belief layer: phase-1 pricing, `_merge`/`_prefer` winner rule, `_admit` refusal classes, status enum |
| `exp13_cbba_adapter.py` | 45,609 | Maps baseline task/agent state onto negotiator bundles and bids |
| `exp13_coordination_node.py` | 54,578 | Node: baseline coordination node + protocol dispatch + the Exp-7 ReidSignature intake |
| `run_exp13_unit_benchmark.py` | 76,660 | Tier-U harness: baseline arm vs Exp-13 arm, 5 fault conditions, 5 agent counts |
| `ConsensusBelief.msg` | 4,092 | Belief/bundle wire definition (bundled here; **never compiled** on this host) |
| `exp13_cbba.yaml` | 11,618 | Protocol, timing, metric-unit and threshold configuration |
| `exp13_bringup.launch.py` | 11,774 | Bring-up that selects the CBAA node (never executed here) |

Folder layout: `source_snapshot/` 126 files (original paths), `tests/` 14, `configs/` 2,
`benchmark_results/` 3, `reports/` 13, `manifests/` 2.

## 2. Protocol design, in the terms a reviewer needs

* **Bundle length.** `max_bundle_size: 1` — one task per agent, so the algorithm is task-selection
  CBAA rather than a multi-task CBBA chain. `test_variant_refuses_bundle_length_greater_than_one()`
  makes the deviation from generic CBBA explicit and enforced rather than incidental.
* **Bidding.** Utility is rounded to `utility_precision: 4` decimals before comparison;
  `bid_epsilon: 1e-06` is the positive-improvement tolerance for admitting a task into a bundle and
  for out-bidding a recorded bid.
* **Winner rule (the tie rule).** Higher rounded bid wins; on an exact tie the **lexicographically
  smaller agent id** wins. The rule is applied in **both** places a winner can be decided:
  `_merge` (on incoming beliefs) **and** `build_bundle` (on the agent's own local record). The second
  one is the F-04 fix: before it, a tied local bid failed the strict-improvement gate, so the winner
  silently became whoever published first. `_merge`/`_prefer` is order-independent by construction.
* **Dispatch.** The node subscribes to the belief topic, runs each payload through `_admit`, and
  refuses with a named class rather than by exception: malformed, foreign sender, wrong dimension,
  invalid flag, target mismatch, out-of-order stamp, future/abnormal stamp, replay from an old
  ownership epoch, re-used episode id, mid-flight registration, abstention.
* **Status.** `ConsensusStatus` distinguishes RUNNING / LOCAL_STABLE / AGREED / DUAL_CLAIM, and
  `agreed_by_all` requires every participant to name the same winner — a local stable bundle is
  diagnostic only and never implies agreement.
* **Timing.** `belief_round_period_ms: 250`, `commit_grace_ms: 500`, `deadline_s: 1.0`.
  `convergence_threshold: 2` is in **rounds**, not milliseconds — `convergence_time_unit: s,monotonic,single-process`
  and the harness records `convergence_time_s` as null wherever a single-process interval is shorter
  than the host clock tick (measured this run: 0.01500 s). Rounds are the primary convergence
  measure; **elapsed time is not a valid speed claim at this tier**.

## 3. Tie-breaking and order-independence: the regression tests

Executed fresh against the copy in this folder (see §7). Named tests that hold the rules above:

```
test_equal_bids_tie_break_to_smaller_agent_id_for_both_participants
test_tie_break_is_repeated_across_reversed_participant_order
test_tie_winner_is_independent_of_activation_order
test_highest_bidder_wins_under_every_activation_order
test_outcome_is_independent_of_arrival_order
test_a_belief_arriving_before_the_first_round_is_order_independent_at_the_episode_level   <- F-04
test_agreement_cannot_be_claimed_while_vectors_disagree
test_dual_claim_is_detected_when_injected_state_contradicts_the_bundle
test_all_agents_abstaining_yields_no_winner_not_convergence
test_bundle_can_never_hold_a_task_the_agent_does_not_win
test_replay_from_a_previous_ownership_epoch_is_refused
test_reusing_a_published_episode_id_is_refused
test_sender_timestamp_is_never_used_for_elapsed_time
test_metrics_report_what_the_agent_actually_observed                                      <- F-02
test_baseline_negotiation_manager_admission_is_unchanged
```

`order_independence_probe` in the result file enumerates arrival permutations per episode
(modes: lockstep, sequential, late_inbox, late_delivered, duplicate, delayed, post_convergence); it is **bounded**, not exhaustive, and says so.

## 4. Benchmark design (as executed)

| Parameter | Value |
|---|---|
| Seed | `20261020` |
| Episodes per arm | 300 (rows in file: 600 across 2 arms) |
| Trials per cell | 12 |
| Distinct utility maps / arrival orders actually varied | 245 / 148 |
| Agent counts | N = 2, 3, 4, 6, 8 |
| Conditions | clean, equal_utilities, replayed_traffic, dead_participant, one_silent_candidate |
| Targets per episode | 1 |
| Participants declared vs swept | 2 declared, ids ['ugv_1', 'ugv_2'] (see `config_consistency`) |
| Evidence tier | **Tier U (unit simulation)** |
| Measurement scope | unit simulation; one process; no ROS graph; no network; no Jetson |
| Message layer | application protocol message; DDS/network packets NOT MEASURED |

`trial_design.interpretation` is worth quoting: "a count of 1 for either field would mean the sweep
varied nothing and must not be described as randomised" — the harness reports its own variation.

## 5. Results — including where CBAA does not converge

**Headline, and it is not a universal success.** In the Exp-13 (CBAA) arm, two of the five fault
conditions never reach a terminal decision at **any** agent count: every `dead_participant` episode
(60 of 300) ends `STALLED_NOT_TERMINAL` and every
`one_silent_candidate` episode (60 of 300) ends
`UNDECIDED`. The baseline arm instead decides in all conditions except N = 2. That is the real
trade-off this experiment measured, and it is stated here rather than buried in an appendix.

Per condition, aggregated over N = 2..8 (12 episodes × 5 agent counts = 60 each):

| Arm | Condition | Decided | Outcome classes | Rounds (min–max) | App msgs sent (min–max) | Ingested (min–max) | All-agree | Winner == expected |
|---|---|---:|---|---|---|---|---:|---:|
| baseline | `clean` | 60/60 | CONVERGED × 60 | 1–1 | 2–8 | 1–7 | 0 | 60 |
| baseline | `equal_utilities` | 60/60 | CONVERGED × 60 | 1–1 | 2–8 | 1–7 | 0 | 60 |
| baseline | `replayed_traffic` | 60/60 | CONVERGED × 60 | 1–1 | 2–8 | 2–14 | 0 | 60 |
| baseline | `dead_participant` | 48/60 | CONVERGED × 48, UNDECIDED × 12 | 1–1 | 1–7 | 0–6 | 0 | 48 |
| baseline | `one_silent_candidate` | 48/60 | CONVERGED × 48, UNDECIDED × 12 | 1–1 | 1–7 | 0–6 | 0 | 48 |
| exp13 | `clean` | 60/60 | CONVERGED × 60 | 3–3 | 5–24 | 4–112 | 60 | 60 |
| exp13 | `equal_utilities` | 60/60 | CONVERGED × 60 | 3–3 | 5–24 | 4–112 | 60 | 60 |
| exp13 | `replayed_traffic` | 60/60 | CONVERGED × 60 | 3–3 | 5–24 | 8–224 | 60 | 60 |
| exp13 | `dead_participant` | 0/60 | STALLED_NOT_TERMINAL × 60 | 3–4 | 3–27 | 0–168 | 0 | 0 |
| exp13 | `one_silent_candidate` | 0/60 | UNDECIDED × 60 | 3–4 | 5–30 | 2–147 | 0 | 0 |

Read it with the asymmetry stated in the file itself (`non_equivalences_disclosed`):

* one-sided baseline vs symmetric Exp-13: only the baseline's owner holds a decision
* stopping rules differ by construction (fixed window vs all-agreement / round bound / deadline) and are not convertible
* the baseline excludes the owner from its candidate pool; Exp-13 lets it bid and retain
* candidate refusal counts are definitional for the baseline, not observed, because the modelled collector has no admission test

In particular the baseline arm is **one-sided**: only the owner holds a decision, so its
`agreed_by_all` is 0 in every episode and its apparent convergence is a weaker claim than the
CBAA arm's `agreed_by_all = 60/60` on the non-fault conditions.

## 6. Ownership metrics, message counts and what is only a local proxy

| Metric | Value in this run | Honest label |
|---|---|---|
| `dual_owner_violations` | 0 in all 600 rows | **Local proxy**: each agent's own bundle against its own winning vector — not a cross-agent measurement |
| `committed_dual_ownership_total` | `null` | **NOT MEASURED**: a committed owner is only observable on the Request → Response → Confirm → ConfirmAck path, which this model arm does not execute. Reporting `null` instead of 0 is deliberate (F-02). |
| `cross_agent_disagreeing_pairs_total` | 0 over 300 measurable rows | Measurable only in the symmetric Exp-13 arm; the baseline arm's rows are `null` because its candidates hold no state |
| `app_messages_sent` / `ingested` | 14.08 / 55.64 mean per episode (Exp-13 arm) | **Application protocol messages counted in-process.** `message_layer` states DDS/network packets are NOT MEASURED |
| `refused_ingress` | 9.84 mean, counted per refusal class | Observed by the negotiator; the baseline arm's collector has no admission test (`coordination_node.py:498-513`), so its value is **definitional**, not measured |
| `false_convergence` | 0 | No episode was counted as converged while its vectors disagreed |
| `convergence_time_s` | `null` where below the host tick (0.01500 s) | A single-process interval is **not** robot or network latency |

Units are declared, not implied: `message_count_unit: belief_builds_counted_at_publish`, `ingestion_count_unit: payloads_examined_at_receive`,
`convergence_time_unit: s,monotonic,single-process`. The naming change is recorded in the file (`naming_change`):
`frames_sent → app_messages_sent`, `ingestion_attempts → app_messages_ingested`, and the previous
formula `rounds × (N−1)` was replaced by counted inboxes. `hardcoded_fields_removed` lists the
three fields that used to be constants: baseline res.decided = True; baseline res.converged_agents = 1; exp13 ingestion_attempts = current_round * (N - 1).

## 7. Tests — fresh execution against the copies in this folder

```
command : python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors
cwd     : C:\Users\PC\edgesync_handoff_20261021\EdgeSync_My_Tasks_Handoff\TASK13\source_snapshot\Implementation Phase\edgesync_ws\src\edgesync_coordination
result  : 360 passed, 3 errors during collection, exit code 1 (pytest returns non-zero when collection fails)
tier    : Tier U + Tier N (stub rclpy) - NOT Tier R
```

Not collected (ModuleNotFoundError: No module named 'rclpy'): tests/test_auto_trigger.py, tests/test_decision_and_receiver_services.py, tests/test_response_collector.py.
Those three are baseline-package tests that import `rclpy` at module scope; they are reported as
**not collected**, never as passed. Exp-13's own 5 test files contribute 5 of the 360:

| File | Result | Evidence |
|---|---|---|
| `test_cbba_negotiator.py` | 106 passed in 0.60s | `perfile_test_cbba_negotiator.py.log` |
| `test_exp13_adapter.py` | 76 passed in 0.41s | `perfile_test_exp13_adapter.py.log` |
| `test_exp13_node_integration.py` | 101 passed in 0.53s | `perfile_test_exp13_node_integration.py.log` |
| `test_exp13_benchmark_metrics.py` | 16 passed in 0.31s | `perfile_test_exp13_benchmark_metrics.py.log` |
| `test_exp7_exp13_signature_integration.py` | 30 passed in 0.53s | `perfile_test_exp7_exp13_signature_integration.py.log` |

**360 passed matches the inherited 2026-10-21 record exactly**

The 600-episode benchmark was **re-executed from the copied source**, not re-read: 16 passed in 0.31s
Reproducibility measured: 588/600 trial rows are byte-identical to the inherited record; the 12
differing rows differ only in `convergence_time_s_max`, i.e. wall-clock jitter in one process.

## 8. Created vs modified vs unchanged (measured)

* **created after the 2026-10-16 archive**: 16 rows — the whole `negotiation/` implementation, the
  Exp-13 node, adapter, config, launch, benchmark script, the 12 documents in `reports/` and every
  Exp-13 test file. The archive contains **no** `cbba*`, `exp13*`, `ConsensusBelief` or `docs/` entry,
  which is why this folder cannot honestly be built from it.
* **modified after it**: 1 rows —
  `setup.py` (pre-change digest `6a09ab0ebf95241c…` in edgesync_es07_backup/pre_s9session_20261019)
* **byte-identical to it**: 10 rows.

## 9. Runtime requirements

* Python 3.10, `numpy`, `pytest` — enough for Tier U and Tier N (that is how §7 ran).
* A **stubbed** `rclpy` plus message classes generated from the on-disk `.msg` for the node tests.
  Real `rclpy`, a DDS implementation, `launch`/`launch_ros` and the ROS 2 CLI are **absent** here.
* To reproduce from the copy: put this folder's
  `source_snapshot/Implementation Phase/edgesync_ws/src/edgesync_coordination` (plus `edgesync_msgs`, `edgesync_identity`,
  `edgesync_bringup`) on `PYTHONPATH` and run pytest inside that package directory.

## 10. Verification level actually reached

| Level | Reached? | What exists |
|---|---|---|
| Tier U (unit / model simulation) | **YES — executed** | 360 passing tests, the 600-episode two-arm benchmark |
| Tier N (node tests on stub middleware) | **YES — executed** | `test_exp13_node_integration.py` under a stub `rclpy` with generated messages. **This is not ROS 2 integration.** |
| Tier R (real ROS 2 graph, DDS) | **NO — blocked** | No rclpy/DDS/launch; `ConsensusBelief.msg` never compiled; `exp13_bringup.launch.py` never executed |
| Tier H / J (Jetson, robot) | **NO — not measured** | No such hardware on this host |

See `TASK13_STATUS_AND_BLOCKERS.md` for what has to happen to clear each one.