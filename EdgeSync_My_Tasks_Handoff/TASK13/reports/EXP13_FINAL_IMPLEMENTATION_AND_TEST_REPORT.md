# EXP13 — Final implementation and test report (Gate 0–4 pass)

**Experiment:** ES-EXP13 — consensus-based task allocation
**Date:** 2026-10-20
**Brief worked:** "Controlled Completion, Design-Authority Resolution, Safety Remediation, and Verification"
**Final status:** `IMPLEMENTATION PARTIALLY COMPLETE — DESIGN DECISION REQUIRED`, with baseline ownership safety `REPRODUCED BUT UNFIXED` and shared-file work `BLOCKED — SHARED-FILE APPROVAL REQUIRED`.
**Evidence tier of every number here:** **Tier U** — one Python process, one Windows host. No ROS graph, no DDS transport, no network, no robot, no Jetson. Nothing in this file upgrades that.

This pass was **not** asked to write an algorithm. It was asked to establish what is true, preserve what exists, reproduce the safety claim, and name the decisions that are not ours to make. That is what it did.

Prior documents remain in force and were **not** overwritten:

| Document | Size | Role |
|---|---:|---|
| `EXP13_IMPLEMENTATION_AND_VERIFICATION_REPORT.md` (repo root) | 46,110 B · `bac4f6f9f96c…20ad89e7` | Gate A–E implementation report — untouched |
| `EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md` (repo root) | 49,228 B · `1e86bfd64612…324920e7` | read-only safety audit, source of M1–M4 — untouched |
| `docs/experiments/exp13/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md` | was 53,867 B · `b150885e3f84…d652c3e` → now 55,943 B · `44df91264c36…694c11b` | completion report — **corrected in place**, see §2 |
| `docs/experiments/exp13/CBBA-VARIANT-DECISION-RECORD.md` | 15,619 B · `1aaf5144295f…d9270788` | in-repo design authority — untouched, still `DECISION REQUIRED FROM OWNER` at `:6` |

New deliverables from this pass: `EXP13_DECISION_AND_BLOCKER_REGISTER.md`, `EXP13_SAFETY_REMEDIATION_PROPOSAL.md`, `EXP13_INTEGRATION_GAP_ANALYSIS.md`, and this file — all under [docs/experiments/exp13/](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/docs/experiments/exp13).

---

## 1. Environment and workspace (re-verified, not assumed)

| Item | Value | How known this pass |
|---|---|---|
| Repo root | `C:\Semester 7\EdgeSync-main (3)Last\EdgeSync-main (3)Last\EdgeSync-main` | path inspected |
| Colcon workspace | `Implementation Phase\edgesync_ws\src` (9 packages) | directory listing |
| Git repository | **No.** `git status` → `fatal: not a git repository`, exit **128**; no `.git`. Git was **not** initialised and no branch was created | executed in Gate 0 |
| Python / pytest | 3.10.11 / 9.0.3 | carried from Gate 0, unchanged host |
| `rclpy` | **Absent.** `python -c "import rclpy"` → `ModuleNotFoundError: No module named 'rclpy'`, exit **1** | executed this pass, transcript saved |
| `colcon` | not exercised — no ROS environment | not attempted, not claimed |
| Jetson / robots | absent, unused | environment |

The brief warned not to assume the previous working directory still exists. It does, and every path used here was resolved from the live tree; `run_exp13_unit_benchmark.py` in particular lives at `edgesync_coordination/edgesync_coordination/scripts/`, not `edgesync_coordination/scripts/`.

---

## 2. Files changed, and why

**Two files were modified. Both are Exp-13-owned. Both are documents. No code, no configuration, no shared file was changed.**

| File | Before | After | Why |
|---|---|---|---|
| [docs/experiments/exp13/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/docs/experiments/exp13/EXP13_IMPLEMENTATION_AND_TEST_REPORT.md) | 53,867 B `b150885e3f840d86a6a2f2f9a7728b7fb3ab1fef4df47bbf27253adb7d652c3e` | 55,943 B `44df91264c36af975a6eb9bd1b4f6614b78416f55616f499d8cc64d29694c11b` | Gate 0 found two factual errors in it. (a) §3 R8 and §8.3 A3 said the message set has **9** files; `ls -1 … \| wc -l` returns **11**. (b) both rows implied the message set carries no epoch field, but `HandoffHeader.msg:8` declares `uint64 ownership_epoch` and `HandoffConfirmAck.msg:4` declares `uint64 new_epoch`. §4's "all nine" wording corrected to match. §8.6 gained two rows recording the checks, the pre-edit digest and the backup path |
| four new deliverables (A, B, C, D) | did not exist | written this pass | brief §10 A–D |

**Both corrections change a *status* nowhere** — R8 and A3 remain `BLOCKED — SHARED FILE` — but they shrink the believed shared-file gap: the missing carrier is the belief payload (`winning_bids`/`winning_agents`/`episode_id`), not the epoch field, and a belief `.msg` can embed the existing `HandoffHeader`. That is recorded as a correction rather than quietly folded in, and it is carried into `EXP13_INTEGRATION_GAP_ANALYSIS.md` §1.5.

**Backup and manifest.** Pre-edit copy: `C:\Users\PC\edgesync_es13_backup\pre_pass13gate_20261020_201935\` — `EXP13_IMPLEMENTATION_AND_TEST_REPORT.md` (53,867 B) plus `SHA256SUMS.txt` recording `b150885e3f84…`. The backup digest equals the digest the completion report recorded for itself, so the file had not drifted since the prior pass. Nothing was deleted. No prior report, backup, transcript or artefact was removed or replaced.

### 2.1 Exp-13-owned code and configuration — verified unchanged

The brief forbids reworking the already-implemented and tested changes without first reproducing a specific defect. No such defect was found, so **all six were left byte-identical**, confirmed by digest against the completion report's own §4 record:

| File | Lines | SHA-256 (first 20) | Matches prior record |
|---|---:|---|---|
| `…/negotiation/cbba_negotiator.py` | 970 | `360da2269a8d6d18cb3b` | yes |
| `…/negotiation/__init__.py` | 30 | `12b3321940c33da9389a` | yes |
| `…/scripts/run_exp13_unit_benchmark.py` | 1,083 | `cd5b372a8658ca036412` | yes |
| `…/tests/test_cbba_negotiator.py` | 1,297 | `8d64e810f0a3f2c24fc9` | yes |
| `…/config/exp13_cbba.yaml` | 113 | `ce38b603149070765cd5` | yes |
| `…/results/experiment13/` ×3 artefacts | — | `207f41c38b75374c…`, `7fee23366d46b93f…`, `3d58c8c6cbec324e…` | yes (revalidated, §5) |

Specifically **kept as-is and re-confirmed present in the live tree**: sequential-activation tie fix (`build_bundle()` at `:822`, ingestion loop at `:824`, `_episode_open` set at `:818`); `ConsensusMessage.ownership_epoch` (`:163`) with epoch-aware `key()` (`:169`) and `STALE` classification (`:593`); `BELIEF_SCHEMA_VERSION = 2` (`:73`); `MIN_AGREEMENT_ROUND = 2` (`:80`) enforced at `:836`; episode-reuse fence (`:358`) with `_episode_open` (`:292`, `:353`, `:400`); the honesty corrections at `:32` and in the `convergence_time_s` docs; `max_bundle_size != 1` raising at `:251`.

### 2.2 Shared files — read-only, digests recorded

`coordination_node.py` `60d720373a422cfb477f` · `fsm/ownership_fsm.py` `99670d02479bc9418c3d` (219 lines) · `fsm/epoch_manager.py` `e4a9d9774bf4f61988bf` (49 lines) · `protocol/negotiation_manager.py` `5654621cf5ba334b97be` · `edgesync_coordination/setup.py` `6a09ab0ebf95241ca29e` · `launch/robot_bringup.launch.py` `45550fc4b3ed408428d9` · all 11 `edgesync_msgs/msg/*.msg` (mtimes 2026-10-14 19:30). **None was opened for writing.**

---

## 3. Exact commands executed, with exit codes

All run this pass; transcripts saved under `C:\Users\PC\edgesync_es13_runs\completion_20261020\gate_tests\` and `…\gate4_console.txt`.

| # | Command | Working directory | Exit | Output (verbatim, last line) |
|---|---|---|---:|---|
| C1 | `python -m pytest tests/test_cbba_negotiator.py -q` | `src\edgesync_coordination` | **0** | `105 passed in 1.46s` |
| C2 | `python -m pytest tests -q --ignore=tests/test_auto_trigger.py --ignore=tests/test_decision_and_receiver_services.py --ignore=tests/test_response_collector.py` | same | **0** | `136 passed in 1.34s` |
| C3 | `python -m pytest tests -q` (no ignores) | same | **2** | `Interrupted: 3 errors during collection` / `3 errors in 1.76s` |
| C4 | `python -c "import rclpy"` | same | **1** | `ModuleNotFoundError: No module named 'rclpy'` |
| C5 | `python fsm_witness.py > witness_console.txt 2> witness_err.txt` | `C:\Users\PC\edgesync_es13_probe\gate1b_20261020` | **0** | 162 lines, stderr 0 B, `EXIT 0 = witness reproduced in the current tree` |
| C6 | `python gate4_validate_artefacts.py` | `C:\Users\PC\edgesync_es13_runs\completion_20261020` | **0** | `checks: 26/26 match` |
| C7 | `git status` | repo root | **128** | `fatal: not a git repository` |
| C8 | `ls -1 edgesync_msgs/msg/ \| wc -l` | `src` | **0** | `11` |
| C9 | `grep -n epoch edgesync_msgs/msg/*.msg` | `src` | **0** | hits at `HandoffConfirmAck.msg:4`, `HandoffHeader.msg:8` only |

**Not executed, and not claimed as executed:** `colcon build`, `colcon test`, `colcon test-result --verbose`, `ros2 launch edgesync_bringup robot_bringup.launch.py`, any multi-agent integration test, any DDS or network measurement, any Jetson run. No dependency was installed and no environment was altered.

---

## 4. Tests — passed, failed, blocked

| Suite | Result | Tier | Notes |
|---|---|---|---|
| Dedicated Exp-13 (`test_cbba_negotiator.py`) | **105 passed, 0 failed**, exit 0 | unit-tested | the prior pass's expected count reproduced exactly |
| ROS-free package suite (3 ignores) | **136 passed, 0 failed**, exit 0 | unit-tested | |
| Full package suite | **exit 2 — BLOCKED BY MISSING ROS 2 DEPENDENCY** | blocked | 3 collection errors, each `ModuleNotFoundError: No module named 'rclpy'`: `tests/test_auto_trigger.py`, `tests/test_decision_and_receiver_services.py`, `tests/test_response_collector.py`. This is **not** a pass and **not** a test failure; the three modules were never collected, so nothing about their behaviour is known |
| Gate 1b FSM witness | exit 0 | unit-executed (FSM layer) / statically-inspected (node handlers) | §6 |
| M1–M4 regression suite | **does not exist** | not applicable | 18 tests specified in `EXP13_SAFETY_REMEDIATION_PROPOSAL.md` §"Regression test specification", to be written **with** an approved patch, not before |

The full-suite exit code 2 is expected and correct on this host; it must not be reported as 136/139 or as "mostly passing". 3 modules are uncollected.

No new tests were added this pass, per the brief: tests are maintained or extended "only when there is a confirmed defect or approved contract requirement", and the confirmed defects (M1–M4) live in shared files that may not be edited, while the REQ-107 requirement is unanswered. Adding an assertion that encodes one unapproved reading of REQ-107 would be a design decision taken silently.

---

## 5. Benchmark — validity, revalidation, and the new per-N finding

**The benchmark was not rerun.** Rule §7.2 requires a concrete correction or missing evidence to justify one; §5.1 of the revalidation found none: the three artefacts' digests are unchanged and **every headline figure re-derives exactly from the 600 stored rows** (26/26 checks, exit 0). Re-running would have overwritten evidence for no gain. All artefacts and prior console transcripts are preserved (deliverable F).

Re-derived vs stored (`__all__` over 300 episodes per arm):

| Metric | Baseline | Exp-13 | Re-derived from rows? |
|---|---:|---:|---|
| Mean application messages **sent** | 4.2 | **14.08** | yes, exactly |
| Mean rounds | 1 | **3.32** | yes, exactly |
| Converged / committed episodes | 276 / 300 | **180 / 300** | yes, exactly |
| Non-converged | 24 (undecided 24, stalled 0) | 120 (undecided 60, stalled 60) | yes |
| Overhead ratio | — | **3.35×** | recomputed: 14.08 / 4.2 = 3.35 |

**Conclusion unchanged and not softened: Exp-13 shows no improvement on either required metric.** It sends 3.35× the application messages and terminates after 3.32 rounds against a baseline whose "1" is **definitional** — `rounds_basis` in the rows says so verbatim: baseline *"definitional: one request plus one fixed collection window (coordination_node.py:484-495); not an observed iteration count"*, Exp-13 *"observed: highest per-participant round counter when the sweep stopped"*. The two are not comparable stopping rules, and rule §7.1 (9) forbids pretending otherwise.

### 5.1 New finding this pass — the declared N = 2 configuration is not the aggregate headline

Aggregating all five participant counts is what produced "180 vs 276 commits". Broken out per N from the trial rows (60 episodes per arm per N):

| N | Arm | Mean msgs sent | Mean rounds | Converged | All-participant agreement |
|---:|---|---:|---:|---:|---:|
| **2 (declared in YAML)** | baseline | 1.6 | 1.0 | **36 / 60** | 0 |
| **2 (declared in YAML)** | exp13 | **4.6** | 3.0 | **36 / 60** | **36** |
| 3 | baseline | 2.6 | 1.0 | 60 / 60 | 0 |
| 3 | exp13 | 8.8 | 3.4 | 36 / 60 | 36 |
| 4 | baseline | 3.6 | 1.0 | 60 / 60 | 0 |
| 4 | exp13 | 12.2 | 3.4 | 36 / 60 | 36 |
| 6 | baseline | 5.6 | 1.0 | 60 / 60 | 0 |
| 6 | exp13 | 19.0 | 3.4 | 36 / 60 | 36 |
| 8 | baseline | 7.6 | 1.0 | 60 / 60 | 0 |
| 8 | exp13 | 25.8 | 3.4 | 36 / 60 | 36 |

Three honest readings, none of them a spin:

1. **At the configuration the YAML actually declares** (`swarm_participant_count: 2`), Exp-13 commits **exactly as many** episodes as the baseline (36/60 both arms) — the "commits fewer episodes" headline is a property of the N > 2 sweep rows, not of the declared configuration. In the `clean` condition at N = 2 both arms commit 12/12.
2. **At N = 2 the overhead penalty is still real and smaller than the aggregate suggests:** 4.6 vs 1.6 application messages = **2.88×**, and 3 rounds vs 1. There is still **no** overhead or speed win to claim.
3. **The N > 2 divergence has a cause the rows identify, not a mystery:** Exp-13 withholds commitment whenever a declared participant is silent (`one_silent_candidate`, `dead_participant`), while the baseline model commits on one receipt — its own row note reads *"owner COMMITTED from 1 receipt(s)"*. So the baseline's 60/60 at N > 2 is **not** evidence of better reliability; it is the absence of an agreement condition. Symmetrically, Exp-13's flat 36/60 at every N is the cost of requiring all declared participants. Neither direction is presented as a win here.

`agreed_by_all` is 36/60 for Exp-13 and 0 for the baseline at every N **by construction** — the baseline protocol has no all-participant agreement state, so that column is a property difference, not a score.

### 5.2 Standing limitations of the evidence (unchanged)

- **Tier U only.** The baseline arm is a stdlib **model** of `coordination_node.py:441-545`, not a live execution of that node, because `rclpy` is absent. Rule §7.1 (10) applies: no conclusion transfers to the real node.
- **Application messages are not packets.** Every row carries `message_layer: "application protocol message; DDS/network packets NOT MEASURED"`. One accounting boundary is used for both arms.
- **Timing is disqualified.** Host monotonic tick measured at 0.014999999999417923 s; 238 of 240 non-null intervals are at or below one tick, and only **n = 2** resolve above it (both at 0.016 s). Mean 0.000383 s is clock quantisation, not speed. Rounds are the primary measure; no latency claim is made. (The prior pass's rerun showed this same boundary count move by 2 episodes between identical seeds — recorded in the completion report §7.7.)
- **The zero counters are not safety proofs.** `dual_owner_violations_total = 0` and `false_convergence_total = 0` are in-process allocation counters on a lane that never touches the ownership FSM; §6 shows the actual dual-owner witness needs no negotiator at all.
- Unilateral `AGREED` remains possible (no acknowledgement leg); REQ-107's clause (iii) is not evaluated; `deadline_s` is sender-local; 3 conditions deliberately withhold commitment and are reported rather than removed.

---

## 6. Baseline ownership safety — reproduced in the current tree, unfixed

Full detail in `EXP13_SAFETY_REMEDIATION_PROPOSAL.md`. Headline:

- The audit's probe was ephemeral and unsaved (`EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:374-383`), so a saved isolated probe was written **outside** the repository and executed against the **real** `OwnershipFSM` and `EpochManager`: `C:\Users\PC\edgesync_es13_probe\gate1b_20261020\fsm_witness.py` (15,476 B, `2ce4a6454fd74a9c…`), exit **0**.
- **PATH 2 reproduced:** one lost `AcqStatus` sample on the best-effort lane → A's `on_confirm_received()` returns **1** without transitioning, the node publishes `CONFIRM_ACK epoch=1` anyway, B becomes `OWNER` at epoch 1, A stays `OWNER` at epoch 1 → **two owners, same epoch**. `is_stale(1)` → `False` and `validate_incoming_epoch(1)` → `True`, so the existing validators would not have caught it.
- **PATH 1 control:** 1 owner (probe is non-vacuous). **PATH 3:** after `abort_to_recovery` → epoch 3, A `OWNER` epoch 3 and B `OWNER` epoch 1 — recovery does **not** reconcile them.
- Node-handler transcription was validated against the live source by 5 runtime fidelity assertions, including `validate_incoming_epoch`/`is_stale` appearing **0 times** in the node and `admit_message` reached at exactly **1** of 6 inbound handlers.
- **New this pass, measured:** M3 as written is not implementable by adding a call — `open_transaction` has **zero** production call sites, so `admit_message` would reject every confirm/ACK/claim as an unknown transaction. The repository's own test already records the sibling fact (`test_phase5_chaos_races.py:58-62`: *"BUG DETECTION: mark_confirmed() is never called from coordination_node.py… it's dead code"*). A prerequisite M0 is therefore proposed, not applied.
- **M1–M4 were not applied.** No approval of record exists; the patch text, invariants, failure modes and 18 regression tests are delivered as proposals.
- **Not Exp-13's defect.** `cbba_negotiator.py` is not on the path; the two `OWNER` states were produced by the ACK exchange with no consensus participant involved.

---

## 7. ROS 2 production integration — not implemented, blocked by approval and environment

Detail in `EXP13_INTEGRATION_GAP_ANALYSIS.md`. Verified this pass: `negotiation_protocol` has **0 production consumers** (all 11 workspace hits are inside Exp-13's own files); the node declares **24** parameters and none is the selector; `allow_undeclared_parameters` is absent (exit 1) and the runtime fate of an undeclared key is **explicitly recorded as unknown on this host**; no production code imports `CBBANegotiator`; `exp13_cbba.yaml` is referenced by **no** launch file; `exp_id` has **0** production hits, so the brief's warning applies — it is a provenance label, not a runtime selector, and it was not added anywhere.

The existing convention was located and reused rather than reinvented: `EDGESYNC_EXPERIMENT_CONFIG` + `select_experiment_config` (`reid_launch_defaults.py:229-254`, wired at `robot_bringup.launch.py:38-48`) — but it reaches **only** the Re-ID path, and `node_coordination`'s `parameters=[…]` (`:149`) never sees it. So the YAML is inert for **two independent reasons**, and patch P-3 is blocked because applying it alone also hijacks Re-ID resolution. New selector frameworks: **0 invented.**

---

## 8. The ten required statuses, reported separately (§11)

| # | Status area | Verdict |
|---|---|---|
| 1 | **Standalone negotiator** | **verified (Tier U only).** 105/105 dedicated tests pass, exit 0; deterministic order-independent winner measured over 24 activation orders; epoch staleness, episode-replay fencing, minimum agreement round and rejection classes are unit-tested with positive **and** negative controls. Not verified at Tier I/H |
| 2 | **Approved algorithm identity** | **unresolved.** D1 unanswered; in-repo record says CBAA (L = 1) and is itself marked `DECISION REQUIRED FROM OWNER`; REV04 `:211-219` designates IBBM-1 and reserves any official label to supervisor approval. The code's `PROTOCOL_ID = "cbba"` / `ALGORITHM_VARIANT` mixture is a silent pick between two written authorities and is **not** approved either way |
| 3 | **Contract compliance** | **partial / blocked on the union of both authorities.** REQ-108 abstain, REQ-109 advisory lane, REQ-110 in-process epoch binding, clause (iv) and clause (v): satisfied and tested. REQ-107 clause (iii) (recompute on the merged **full** bid set): **not implemented** — `check_agreement` (`:690-735`) compares a winner vector, and the full-bid-set requirement is unanswered, so it was not implemented "merely because it appears in REV04". Acknowledgement leg / distributed commitment: absent, undecided (B-3) |
| 4 | **Baseline ownership safety** | **reproduced but UNFIXED.** Executed witness in the current tree (§6). Repair requires shared-file approval that does not exist; M0–M4 delivered as unapplied proposals with 18 specified tests |
| 5 | **ROS 2 production integration** | **not implemented, and blocked by both approval and environment.** G5 needs D5/D7/D10/D11 (all unanswered, "no shared file may change"); `rclpy` exit 1; `colcon` never run |
| 6 | **Benchmark evidence** | **verified for its stated Tier-U scope.** 26/26 revalidation checks, digests unchanged, all headline figures re-derived from the stored rows; per-N breakdown now reported so the declared N = 2 configuration is separated from the exploratory sweep. Its limits are structural: modelled baseline, application-layer counting boundary, disqualified clock |
| 7 | **Network reliability** | **not measured.** No DDS, no packet, no byte, no loss-rate figure exists here. The one loss-related statement (best-effort `AcqStatus` lane) is a **configuration reading** at `:153-157`/`:167` and `target_acquisition_node.py:110,257`, not a measurement |
| 8 | **Jetson performance** | **not measured.** Hardware absent; a laptop number is not a Jetson number |
| 9 | **Shared-file approval** | **pending** for every one of `coordination_node.py`, `fsm/ownership_fsm.py`, `fsm/epoch_manager.py`, `protocol/negotiation_manager.py`, `launch/robot_bringup.launch.py`, `edgesync_msgs/msg/*`. No approval artefact exists; no per-file owner exists (`UNKNOWN (package-generic metadata only)`); nothing was edited |
| 10 | **Remaining owner decisions** | listed in full in `EXP13_DECISION_AND_BLOCKER_REGISTER.md` §3 — B-1/D1 designation · B-2/REQ-107 full-bid-set mandatory · B-3 meaning of `AGREED` (local belief vs distributed commitment) · B-4/REQ-110 wire contract · B-5/A1-D5 selector declaration + fallback · B-6/A2 `exp_id` role · B-7/D10 belief `.msg` + rebuild/deploy owner · B-8/D11 adapter & hosting · B-9/D7 branch vs wrapper · B-10 M1–M4 ownership · B-11 production participant count · B-12/D3–D4 timing budget placement and metrics ownership |

---

## 9. Remaining work and responsible owners

| Item | Work | Owner (role) | Prerequisite |
|---|---|---|---|
| W1 | Rule on D1 and REQ-107 | Supervisor | nothing — unblocks the most code |
| W2 | Rule on the meaning of `AGREED`; authorise or forbid an ack/quorum/lease | Supervisor + coordination owner | W1 |
| W3 | M0 + M1 + M2 + M3, with the 18 specified regression tests | Coordination owner | written approval to edit 4 shared files; W1 not required |
| W4 | Choose M4's candidate (a)–(d) | Supervisor + coordination owner | W3 (M0) |
| W5 | Declare the selector, wire the overlay, approve D11/D7 | Coordination owner + bringup owner | W1, W2 |
| W6 | Approve and build the belief `.msg`; rebuild 8 consumer packages | msgs owner | W1, W2 |
| W7 | Tier-I run: `colcon build/test/test-result`, launch + multi-agent fault injection | User (environment + authorisation) | W3–W6 + a ROS 2 Humble workstation |
| W8 | Tier-H run | User + Jetson Orin Nano | W7 |
| W9 | Name a per-file owner of record, or nominate the approver for each shared file | Supervisor / project owner | nothing — and it is what makes W3–W6 actionable at all |

---

## 10. What would make this report wrong

Stated so a reviewer can attack the right things: the PATH-2 witness depends on the node-handler transcription, which is fidelity-checked against the source but **not executed as a node**; the per-N breakdown assumes each arm's 60 episodes per N are matched inputs, which the harness reports but a reviewer should re-derive; the 3.35×/2.88× overhead figures inherit whatever the shared counting boundary includes; the baseline arm is a model; and every number is Tier U. If D1 lands as "IBBM-1 with mandatory REQ-107", then `ConsensusMessage` needs a bid-set field, `BELIEF_SCHEMA_VERSION` goes to 3, `check_agreement` must evaluate clause (iii), and the current agreement statistics — including the 180/300 and 36/60 commit counts — are re-measurable only after that change.
