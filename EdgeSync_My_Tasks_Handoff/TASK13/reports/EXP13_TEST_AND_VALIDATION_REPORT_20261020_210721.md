# EXP13 — Test and validation report (autonomous pass `20261020_210721`)

Every number in this file came from a command that was executed on this host during this pass. Nothing here is
projected from a previous run; where a prior run is quoted, it is quoted as *prior evidence* and labelled.

## 1. Evidence tiers used throughout Exp-13

| Tier | What it is | What it can prove | What it cannot prove |
|---|---|---|---|
| **Tier-U** | unit simulation, `run_exp13_unit_benchmark.py`, pure Python, no node objects | protocol-level message counts, convergence, refusals, determinism across enumerated orders | anything about DDS, QoS, timers, or real clocks |
| **Tier-N** | node tests under a generated stub `rclpy` + message types built from the on-disk `.msg` files | that the arm's code paths do what the source says, including guard ordering and lifecycle | that a real graph delivers those messages |
| **Tier-R** | a real ROS 2 build + graph (`colcon`, `ros2 launch`, `ros2 topic`) | integration | **not reached — see §5** |
| **Tier-J** | Jetson Orin Nano robots | network/robot validation | **not reached — no hardware on this host** |

Stub harness (Tier-N): message classes are generated from `edgesync_msgs/msg/*.msg` at import time and fake
`rclpy`, `rclpy.node`, `qos`, `time`, `duration`, `tf2_ros`, `edgesync_msgs` modules are installed and then
uninstalled inside the test session — `test_the_stubs_were_uninstalled_after_the_import`,
`test_the_stub_messages_came_from_the_definitions_on_disk` and
`test_a_stub_message_refuses_a_field_the_definition_does_not_declare` constrain the harness itself, so a stub
cannot silently diverge from the definition. Repo precedent: `test_identity_node_transreid_integration.py`.

---

## 2. Executed runs

All commands run from `Implementation Phase/edgesync_ws/src/edgesync_coordination`. Full log:
`C:\Users\PC\edgesync_es13_runs\completion_20261020\phase_g_final_test_execution_20261020.txt`.

| Command | Result |
|---|---|
| `pytest tests/test_exp13_node_integration.py -q` | **101 passed** in 0.40 s |
| `pytest tests/test_exp13_adapter.py -q` | **76 passed** in 0.36 s |
| `pytest tests/test_cbba_negotiator.py -q` | **105 passed** in 0.46 s |
| `pytest tests/test_exp13_node_integration.py tests/test_exp13_adapter.py tests/test_cbba_negotiator.py -q` | **282 passed** in 0.67 s |
| `pytest tests -q --continue-on-collection-errors` (whole coordination package) | **313 passed, 3 collection errors** in 0.89 s |
| per-package run of all 8 packages, same flags | **772 passed, 3 skipped, 11 collection errors** |

Workspace-wide breakdown (`...phase_g_workspace_pytest_20261020_final.txt`, executed 22:38 local):

| Package | Passed | Skipped | Collection errors |
|---|---:|---:|---:|
| edgesync_acquisition | 0 | 0 | 1 |
| edgesync_bringup | 57 | 0 | 1 |
| edgesync_coordination | **313** | 0 | 3 |
| edgesync_estimation | 18 | 0 | 1 |
| edgesync_identity | 372 | 3 | 1 |
| edgesync_navigation | 3 | 0 | 1 |
| edgesync_observability | 2 | 0 | 2 |
| edgesync_perception | 7 | 0 | 1 |
| **Total** | **772** | **3** | **11** |

Correction, recorded rather than hidden: an earlier working note in this pass cited *12* collection errors. The
executed run says **11**. The 3 skipped tests are all pre-existing in `edgesync_identity` and are not Exp-13 files.
**No Exp-13 test is skipped** in any of the runs above.

### The 11 collection errors, all of them environment, not code

| Package | File | Reason |
|---|---|---|
| acquisition | `test_target_acquisition.py` | `ModuleNotFoundError: No module named 'rclpy'` |
| bringup | `test_robot_bringup_launch.py` | `ImportError: cannot import name 'LaunchDescription' from 'launch' (unknown location)` |
| coordination | `test_auto_trigger.py`, `test_decision_and_receiver_services.py`, `test_response_collector.py` | `ModuleNotFoundError: No module named 'rclpy'` (×3) |
| estimation | `test_estimation_node.py` | `ModuleNotFoundError: No module named 'rclpy'` |
| identity | `test_identity_service.py` | `ModuleNotFoundError: No module named 'rclpy'` |
| navigation | `test_safety_authority_node.py` | `ModuleNotFoundError: No module named 'rclpy'` |
| observability | `test_profiler_and_calibration.py` | `ModuleNotFoundError: No module named 'edgesync_coordination'` |
| observability | `test_thesis_logging.py` | `ModuleNotFoundError: No module named 'rclpy'` |
| perception | `test_perception_hardware_integration.py` | `ModuleNotFoundError: No module named 'sensor_msgs'` |

The bringup row deserves its own sentence: `launch` is genuinely **not installed** (`import launch` from anywhere
else raises `ModuleNotFoundError`), but pytest inserts the package rootdir onto `sys.path`, and
`src/edgesync_bringup/launch/` is the ROS *launch-script directory*, so `import launch` resolves there as an empty
namespace package. Verified and logged in the same evidence file. It is a pre-existing condition of this tree on
a non-ROS host, not something this pass introduced.

---

## 3. Coverage against brief §7 Phase D (the 14 required categories)

Section labels are the comment headers in the delivered files.

| # | Required category | Where it is tested | Representative executed tests |
|--:|---|---|---|
| 1 | Deterministic winner under arrival order | neg §15; N12; benchmark determinism probe | `test_the_awarded_candidate_sees_the_same_winner_from_its_own_view`; 1 distinct winner over all 24 participant orders at N=4 (table §5) |
| 2 | Bid conflicts / full-bid-set reconciliation | neg §2 (winner-vector merge). **Full-bid-set carriage rejected in D2**, so no recompute path exists to test | `adapter:` C4/C5 round-trip cases |
| 3 | Ties and equal bids | neg §8; benchmark `equal_utilities` condition | equal bid at comparison precision → lexicographically smaller agent id; 60/60 converged with 0 refusals |
| 4 | Duplicate messages | neg §4; adapter report de-duplication by `belief_id` | `adapter:` C6 |
| 5 | Stale / future epochs | neg §5; `ownership_epoch` field on the belief | `test_a_replayed_ack_cannot_announce_ownership_twice` |
| 6 | Old-episode messages | neg §17 replay fence; adapter C2; N5 | `test_beliefs_for_a_different_episode_are_not_merged_into_the_running_one`, `test_a_canonical_id_naming_another_target_is_refused_instead_of_negotiating_along` |
| 7 | Missing / late participants | neg §7; N3; benchmark `dead_participant`, `one_silent_candidate` | `test_a_declared_participant_is_never_dropped_to_make_the_episode_fit`, `test_a_single_participant_set_is_refused_rather_than_running_a_vacuous_consensus`, `test_a_node_that_bounds_out_before_the_peer_has_emitted_anything_hands_over_nothing` |
| 8 | Timeout / non-convergence | neg §11 bounded termination; N10/N12 | `test_a_watchdog_cutoff_ends_the_episode_it_is_abandoning`, `test_the_response_window_closes_without_choosing` |
| 9 | Invalid configuration | adapter C1; N1; N2; neg §12 | `test_an_unknown_selector_refuses_to_start_and_never_reaches_the_baseline`, `test_every_rejected_selector_value_raises_instead_of_redirecting`, `test_a_missing_required_key_raises_rather_than_defaulting_mid_experiment` |
| 10 | Message / schema validation | adapter C4/C5; N4 | `test_a_published_belief_survives_the_message_definition_field_by_field`, `test_the_wire_sequence_is_the_round_number_with_no_second_counter`, `adapter:` C5 malformed-payload cases |
| 11 | Selector dispatch + baseline default preserved | N1; adapter C1; N13 | `test_the_cbba_selector_builds_the_adapter_and_the_belief_endpoints`, `test_the_cnp_selector_runs_the_inherited_baseline_with_no_belief_plumbing`, `test_an_unconfigured_node_gets_the_declared_baseline_default` |
| 12 | Adapter lifecycle / failure handling | adapter C6–C9; N10; N12 | `test_a_second_trigger_never_silently_replaces_a_running_episode`, `test_a_closing_decision_stops_the_round_driver`, `test_the_driver_is_rearmed_for_the_next_attempt`, `test_a_unsafe_dual_claim_transfers_nothing_and_is_logged_as_a_violation` |
| 13 | Ownership-safety regression | N6 (M0), N7 (M1), N8 (M2), N9 (M3), N10b (M4) | see `EXP13_SAFETY_FINDINGS_20261020_210721.md` S-1…S-10, each with its named test |
| 14 | Regression for other experiment selectors | N13 + the 7 "control arm" twins + the workspace run | `test_the_baseline_bringup_launch_is_untouched_by_this_experiment`, `test_the_deployed_coordination_node_source_has_no_experiment_branch`, `test_the_arm_does_not_redeclare_any_parameter_the_baseline_owns`, `test_the_control_arm_still_picks_the_highest_utility_answer` |

Counting note for transparency: the three Exp-13 files contain 90 + 63 + 59 = **212 test functions**, which
collect as **282 tests** because of parametrised cases. 282 is the executed count; 212 is the def count.

---

## 4. Environment capability probes (executed, exact output)

```
python -c "import rclpy"          -> ModuleNotFoundError: No module named 'rclpy'
python -c "import edgesync_msgs"  -> ModuleNotFoundError: No module named 'edgesync_msgs'
python -c "import launch"         -> ModuleNotFoundError: No module named 'launch'
python -c "import launch_ros"     -> ModuleNotFoundError: No module named 'launch_ros'
which ros2                        -> not found on PATH
python                            -> 3.10.11
pytest                            -> 9.0.3
host                              -> Windows 10 (10.0.26300) AMD64
```

No dependency was installed and no environment was altered (standing constraint).

---

## 5. Commands that could **not** be run, and why

| Command | Status | Reason |
|---|---|---|
| `colcon build --packages-select edgesync_msgs edgesync_coordination edgesync_bringup` | not executed | no ROS 2 workspace setup / `colcon` on this host |
| `rosidl` generation of `ConsensusBelief.msg` | not executed | same; the message has therefore **never been compiled** |
| `ros2 launch edgesync_bringup exp13_bringup.launch.py …` | not executed | `launch`/`launch_ros` absent — the file is read, never run |
| `ros2 topic echo /edgesync_swarm/consensus_belief` | not executed | no graph exists to echo |
| `pytest` on the 11 ROS-dependent files | not executed | they fail at collection for the reasons in §2 |
| Multi-node run over real DDS | not executed | Tier-R absent |
| Jetson / robot runs | not executed | no hardware reachable from this session |

**The launch file's own status line says the same thing** (`exp13_bringup.launch.py:55-59`), so a reader of the
code cannot mistake it for a validated integration.

What was done instead, and how it is labelled: the N-sections drive real `Exp13CoordinationNode` objects against
generated message types, including **two nodes exchanging beliefs over their real topic names**
(`test_two_nodes_exchange_beliefs_over_their_real_topic_names`) and the peer-with-the-live-higher-battery outcome
(`test_the_peer_with_the_live_higher_battery_wins_and_the_owner_hands_to_it`). That is Tier-N. It is *not* a ROS 2
integration test and is never called one in these reports.

---

## 6. Suite hygiene, including what this pass changed about its own tests

* **No existing test was deleted or weakened.** `test_cbba_negotiator.py` only gained cases; the pre-existing
  baseline-regression section (neg §14) still runs.
* **Two of this pass's own assertions were removed as vacuous**, which is a self-criticism, not a fix:
  `assert node.pub_confirm.published == [] or True` (always true) was deleted, and a test that used `pytest.skip`
  when the configuration had not armed a COMMIT was rewritten to drive the pair deterministically. Exp-13 now
  runs **0 skipped**.
* Two earlier N12 pair failures were **test-schedule artefacts, not code defects**: driving one node to its round
  bound before the peer had emitted anything left the pair in `LOCAL_STABLE`. The harness gained `drive_pair()`
  (interleaved rounds) and the partition behaviour was kept as its own explicit test
  (`test_a_node_that_bounds_out_before_the_peer_has_emitted_anything_hands_over_nothing`) rather than hidden.
* The launch-wiring assertion was rewritten against the file's actual positional style instead of the keyword
  style it had assumed — a test-authoring error, corrected, with the corrected test now passing.
* Six node-level defects found *because* the tests were strict are recorded in the safety report (S-1…S-6). None
  of them was fixed by relaxing a test.

---

## 7. Artifact and configuration validation (brief §7 Phase E items)

| Phase E requirement | Result |
|---|---|
| Dedicated Exp-13 unit tests | executed — 282 passed (§2) |
| ROS-free package suite | executed — 313 passed, 3 collection errors |
| Full test collection | executed per package — 772 passed, 3 skipped, 11 errors |
| ROS 2 interface/build tests | **could not run** (§5) |
| Integration with multiple simulated participants | executed at Tier-N (two real node objects, real topic names) and at Tier-U (N = 2, 3, 4, 6, 8) |
| Baseline regression tests | executed — all pre-existing coordination tests pass alongside the new files; `cnp` control-arm twins prove the inherited behaviour is intact |
| Artifact / configuration validation | `exp13_cbba.yaml` is loaded by the launcher and its keys are asserted to reach the negotiator at episode open (`test_the_parameters_take_effect_on_the_negotiator_the_moment_an_episode_opens`, `test_the_round_period_parameter_sets_the_driver_period`, `test_require_distributed_agreement_false_is_honoured_by_the_adapter`) |
| Hash comparison of preserved files | executed — `phase_e_digest_check_20261020_223018.txt`: preserve set **7/7 bit-identical**, to-modify set **5/5 intentionally changed** |

---

## 8. What these tests do not establish

1. **Nothing about a real graph.** No message has ever crossed a DDS boundary in this work. QoS compatibility of
   RELIABLE + depth 10 against a real middleware, topic remapping under a live launch, and executor threading are
   untested.
2. **Nothing about clocks.** The round timer's period is asserted to be *configured* correctly, not to fire at
   250 ms on a real clock. The benchmark's `conv_s` column is duration-blind (see the comparison report §4).
3. **Nothing about `ConsensusBelief` compilation.** Its invariants are enforced by `belief_from_mapping()` in
   Python; the `.msg` file has not been through `rosidl`.
4. **`dual_owner = 0` in the Tier-U run is a model observation**, not proof of guard correctness on robots. The
   guards themselves are proven at Tier-N.
5. **No statistical claim is made.** 12 trials per cell on 245 distinct utility maps is a reproducible sweep, not
   a hypothesis test; no confidence interval is quoted anywhere in these reports.
