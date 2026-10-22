# EXP13 — Design decisions (autonomous pass)

**Run id:** `20261020_210721`  **Date:** 2026-10-20
**Authority:** brief *"Autonomous Design Decision, Isolated Integration, Implementation, and Validation"* §1
("I am authorizing you to make the engineering decisions necessary for Experiment 13 within the boundaries below")
and §4 ("Resolve the Design Decisions Yourself").
**Supersedes for this run:** the `BLOCKED — DESIGN DECISION REQUIRED` posture recorded in
`EXP13_DECISION_AND_BLOCKER_REGISTER.md` §1, which was correct **under the previous brief** (that brief withheld
decision authority) and is now replaced by owner grant, not by new evidence.
**Backup:** `C:\Users\PC\edgesync_es13_backup\pass15_autonomous_20261020_210721\` (`SHA256SUMS.before.txt`).

Every decision below is stated as **selected option · reasoning · consequences · what would falsify it**, so a
reviewer can overturn one item without re-litigating the set.

---

## D1 — Algorithm identity: **CBAA (CBBA with bundle length L = 1)**, selector token `cbba`

**Selected.** The algorithm is CBAA (Berinet/Brunet–Choi–How, AAMAS 2008): single-slot bundle, positive-improvement
bidding, consensus by winning-bid / winning-agent vector exchange. The **selector value stays `cbba`** because brief
§D6 fixes the assignment's identifier list (`cnp`/`auction_cnp`/`cbba`/`centralized_broker`) — that token is an
interface, not a claim about the algorithm. Every human-readable string (config comment, log line, report,
`ALGORITHM_VARIANT`) says CBAA / L = 1.

Explicit specification, as §D1 requires (this is now the normative text; see `EXP13_INTEGRATION_GAP_ANALYSIS.md`
for the previous "undefined" state):

| Element | Rule | Source |
|---|---|---|
| Bundle construction | greedy single slot: take the highest task whose bid beats the recorded winning bid by more than `bid_epsilon`; `len(bundle) >= L` ⇒ no-op | `cbba_negotiator.py:489-522` |
| Bidding | `UtilityEngine.compute_receiver_utility(position, battery)`, rounded to `utility_precision`; **abstain** (`None`) on missing calculator, missing telemetry, non-finite or out-of-[0,1] result — never a substitute score | `:449-478` |
| Consensus | per task, merge local and peer claims under `_prefer`; peer vectors retained verbatim for the agreement test | `:627-661` |
| Winner selection | argmax bid, total order | `:663-676` |
| Tie-break | equal bid at comparison precision ⇒ **lexicographically smaller agent id** wins; rule is symmetric and does not depend on arrival order or on which claim is "mine" | `:663-676` |
| Convergence | `check_agreement()` true ∧ `round >= MIN_AGREEMENT_ROUND` ⇒ `AGREED`; all-tasks-unassigned ⇒ `NO_WINNER`; bounds `max_rounds`, `deadline_s`; quiet-detector ⇒ `LOCAL_STABLE` (**not** agreement) | `:832-849` |

**Why not full CBBA.** With `target_count: 1` and `max_bundle_size: 1`, the three features that distinguish CBBA —
candidate lists, the path-bid consistency trim, and cascade release of later bundle entries — are vacuous. Shipping
them anyway would add unreachable code and would license a false label, which §D1 forbids.

**Consequences.** `max_bundle_size != 1` keeps raising `ValueError` (`:251-255`) — an honest refusal, not a gap. A
reviewer who wants genuine L > 1 must also add a multi-target task set; that is a **change of experiment**, not of
naming, and is out of scope here.

**What would falsify it:** evidence that the Exp-13 task set contains >1 allocable slot per agent, in which case L = 1
is a restriction imposed by the code rather than by the scenario, and full CBBA becomes mandatory.

---

## D2 — Full bid-set consensus (REQ-107): **winner-vector is sufficient; full bid set rejected**

**Selected.** CBAA/CBBA consensus *is* consensus on the winning-bid vector (plus the winning-agent vector). Each
agent keeps, per task, the best bid and its owner; the merge is a symmetric join, so the lattice of views converges to
the same fixed point regardless of order — that property is measured, not asserted (1 distinct winner over 24
activation orders, `EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md` §5). Carrying every agent's every bid would add
payload and a recompute path while **changing no reachable outcome**, because any bid that is not the maximum for its
task is invisible to the allocation at L = 1.

**Therefore REQ-107's "full bid set carriage + recompute" is NOT implemented.** It is a stronger contract than the
selected algorithm needs, and §D2 delegates exactly this judgement: "Determine whether winner-vector comparison alone
is sufficient for the selected algorithm."

**What IS added, because §D2 also requires "sufficient, versioned information":** a per-belief
`agreement_established` flag and explicit schema versioning. `BELIEF_SCHEMA_VERSION` goes **2 → 3**; a schema-2 belief
is refused as `UNSUPPORTED_SCHEMA` by the existing guard at `:577-578` rather than silently read as "no flag". This is
an explicit version bump with a rejection path, which §D2 demands ("version it explicitly … do not silently
reinterpret an existing schema"); it is not a silent reinterpretation, and Exp-13 has no deployed v2 peers to break.

**Consequence:** `check_agreement()` (the content test) is unchanged in meaning; the new predicate
`check_distributed_agreement()` (§D3) is additive.

---

## D3 — Meaning of `AGREED`: **a four-level ladder, and Exp-13 may only climb the first two**

**Selected.** Four states are distinguished and given separate names, because brief §D3 requires the distinction and
the previous implementation collapsed levels 1 and 2 into one boolean:

| # | State | Who sets it | Meaning |
|---|---|---|---|
| 1 | `LOCAL_CONSENSUS` | negotiator, `check_agreement()` | my vectors equal every declared participant's **last reported** vectors |
| 2 | `AGREED` | negotiator status | 1, ∧ round ≥ `MIN_AGREEMENT_ROUND`, ∧ some task assigned — **still an observation about my view** |
| 3 | `COMMIT_ARMED` | **adapter**, `check_distributed_agreement()` | 2, ∧ every declared peer has *itself reported* `agreement_established = true` for this episode and epoch |
| 4 | `OWNERSHIP_COMMITTED` | **the confirm/ACK exchange only** | epoch advanced and owner claim published |

**The acknowledgement leg, decided.** The audit's sharpest finding was that `AGREED` is local by construction and that
"an agent which receives from everyone but is itself never heard" satisfies it while its peers stay non-converged
(`cbba_negotiator.py:704-718`). The previous pass refused to close it because a failure detector / quorum / lease was
not authorised. Under this brief I resolve it with the **cheapest mechanism that is not a new consensus protocol**:
peers piggyback their own agreement result on the beliefs already in flight, and the adapter requires all peer reports
before it arms a commit. Cost: one boolean on a message that is already sent, and a bounded post-agreement collection
window. It is not a quorum (no voting), not a lease (no TTL), not a failure detector (silence yields
`not COMMIT_ARMED`, which is the safe direction).

**Never equated.** `AGREED` ≠ ownership transfer, explicitly: the negotiator writes no epoch, publishes nothing, and
touches no FSM (`:25-37` remains true). Level 4 lives in `_on_handoff_confirm` / `_on_confirm_ack_received` — and
those are where §6's defects are, which is why **D8 exists**.

**Consequences.** Some episodes that previously stopped at `AGREED` will now stop at `AGREED` but never reach
`COMMIT_ARMED`, and will abort instead of committing. That **lowers** the commit count. I accept it: §D3 says use the
safest contract compatible with the project, and brief §10 forbids tuning for a better number.

**What would falsify it:** if the peer-reported flag can be set `true` by an agent whose own view is stale within the
same epoch, the gate is cosmetic. Bounded by `(episode_id, ownership_epoch)` binding and by `belief_id` dedup; tested
for the false-witness direction in `test_exp13_adapter.py`.

---

## D4 — Participant and convergence rules

**Selected.** Authoritative participant set = **declared, never discovered**: the constructor's `participant_ids`
(`:258-262`, ≥ 2 enforced, self must be a member) supplied by the adapter from `robot_id`/`peer_id` +
`swarm_participant_count`. The two must agree or construction fails — a count of 2 with 3 names is a configuration
error, not a negotiation.

| Situation | Classification | Basis |
|---|---|---|
| Participant silent for the whole episode | never counted as agreeing; outcome `MAX_ROUNDS` / `TIMEOUT` / `LOCAL_STABLE`, **never removed from the set** | `:722-728`, `:813-815`, `:847-849` |
| Late participant (belief after bound reached) | `POST_TERMINAL` (counted, not applied) | `:573-574` |
| Belief from an older epoch | `STALE` | `:593-594` |
| Belief from an older round of the same episode | `STALE` | `:601-602` |
| Belief re-using a used `episode_id` | **refused at registration** (`ValueError`) — the round watermark is cleared by `reset()`, so only identity fences a replay | `:358-363` |
| Duplicate round belief | `DUPLICATE` | `:599-600` |
| Unknown sender | `UNKNOWN_SENDER` | `:583-584` |
| Task/target set mismatch | `TARGET_MISMATCH` | `:604-609` |
| Equal bids | lexicographic agent id (§D1) | `:663-676` |
| Timeout / retry | round bound + optional monotonic `deadline_s`; retry requires a fresh `episode_id` | `:813-815` |

**Silence is never assent, and absence is never deletion** — §D4's two hard rules, both already enforced and now
stated as the contract rather than as code behaviour.

---

## D5 — Wire protocol: **one additive message, `ConsensusBelief.msg`; no existing message changed**

**Selected.** Inspection of all 11 existing `.msg` files shows `HandoffHeader` already carries
`sender_id / receiver_id / transaction_id / target_id / ownership_epoch / sequence / stamp_utc_ns / protocol_version /
message_type`, and no existing message carries a belief payload. So:

```
# edgesync_msgs/msg/ConsensusBelief.msg
edgesync_msgs/HandoffHeader header      # identity, epoch, txn, stamp — NOT duplicated below
uint8  schema_version
string belief_id
uint32 round_number
string[] task_ids
string[] target_ids           # parallel to task_ids (REQ-206: carried, never inferred)
float64[] winning_bids        # parallel
string[] winning_agents       # parallel
string[] bundle               # 0 or 1 entry at L = 1
bool   agreement_established  # schema 3 (§D2/§D3)
uint8 MSG_TYPE_ALLOCATION_BELIEF = 9
```

**Identity mapping, decided explicitly as §D5 requires:** `episode_id ≡ header.transaction_id`. One negotiation
attempt gets one identity; there is no second `episode_id` field to fall out of sync with it. The adapter mints
`transaction_id = f"{target_id}@e{epoch}#{attempt}"`, which satisfies the replay-fencing rule in §D4 by construction
(different epoch ⇒ different id; different attempt ⇒ different id).

**`HandoffHeader.msg` is NOT modified.** Value 9 is declared as a constant on the new message instead of added to the
shared header, so the existing 8 message-type constants keep their exact values and every consumer package's
behaviour is unchanged.

**Consequences (accepted, and they are real).** `edgesync_msgs/CMakeLists.txt` uses an **explicit** `set(msg_files …)`
list, not a glob, so one line must be added to a shared file — the smallest possible change. That forces a rebuild of
`edgesync_msgs` and, for consistency, of the 8 packages that depend on it
(acquisition, bringup, coordination, estimation, identity, navigation, observability, perception). **No compatibility
claim is made until `colcon build` has actually run**, and it cannot run here (`ros2` absent from PATH, no
`rclpy`, no ROS 2 environment). The wire contract is therefore *specified*, not *validated*; the ROS-free schema test
in `test_exp13_adapter.py` parses this file and asserts that every field the mapping touches exists with the expected
type, which is the strongest evidence obtainable on this host.

---

## D6 — Selector and configuration: **new node declares it; `cnp` stays the default; invalid ⇒ hard failure**

**Selected.** All §5-mandated preservation is achieved **without touching the baseline node or launch file**:

* new `nodes/exp13_coordination_node.py` declares `negotiation_protocol` (default **`'cnp'`**) and dispatches —
  `cnp` ⇒ inherited one-round path, `cbba` ⇒ CBAA adapter path;
* unknown or unimplemented values (`auction_cnp`, `centralized_broker`, typos) ⇒ `ValueError` at construction naming
  the supported set. An invalid `cbba` configuration can therefore never be silently redirected to the baseline, as
  §D6 forbids;
* `robot_bringup.launch.py` and `coordination_node.py` are **not edited**. The old default path is preserved by
  non-modification, and its evidence is a byte-identical digest (§ see implementation report);
* `exp13_cbba.yaml` reaches a running node through the **new** `edgesync_bringup/launch/exp13_bringup.launch.py`,
  which passes it in the `parameters=[…]` list. The Re-ID overlay machinery (`select_experiment_config`,
  `resolve_reid_config`) is **not called and not modified** by that file, so §D6's warning about Re-ID configuration
  resolution is respected; it also sidesteps the known `EDGESYNC_EXPERIMENT_CONFIG` single-channel coupling.

**Defect found and fixed in the owned config:** `battery_pct: null` at `exp13_cbba.yaml:107` would be layered onto a
node that declares `battery_pct` as a `float` (`coordination_node.py:62`), which ROS 2 rejects as a type violation at
parameter load. Since integration is now the point, that line is a live bug, so the key becomes an explicit
`battery_pct: 0.0` **plus** the adapter rule that no telemetry ⇒ abstain (never a stand-in bid). This is a specific
reproduced defect, which is the standard §1 sets for reworking existing work.

**Every key in `exp13_cbba.yaml` is declared on the new node.** That is deliberate: an undeclared key is silently
dropped by rclpy, and a silently dropped `max_rounds` would change the experiment's outcome without a sound.

---

## D7 — Integration topology: **subclass the node; put every decision in ROS-free code**

**Selected.** The seam is `broadcast_handoff_request()` (`coordination_node.py:441-482`): it is the only thing the
production trigger calls (`:277`, from `_on_tto_estimate`), and the only place `trigger_candidate`/`send_request`
plus the request publish occur. Overriding it substitutes the negotiation strategy while perception ingestion, QoS,
topic names, FSM construction, epoch manager, receiver-side decision, watchdog and the commit handoff
(`:542`/`:551`/`:554`) are inherited untouched.

Consequences: the Exp-13 node enters `NEGOTIATING` exactly as the baseline does, so nothing downstream is surprised;
the belief exchange runs on a new topic; the commit decision is made by the adapter and then reproduces the
baseline's own three calls. All protocol logic lives in `negotiation/exp13_cbba_adapter.py`, which imports **no**
ROS package — which is what makes it testable on this host at all.

**One shared file is touched for packaging:** `edgesync_coordination/setup.py` gains one `console_scripts` line.
There is no alternative — ament_python registers executables only through entry points (the top-level `scripts/`
directory is not a registration path, per §1 of the workspace audit).

---

## D8 — Ownership-safety seam: **guard it in Exp-13-owned code; do not fix the baseline here**

**Selected.** §6 asks whether Exp-13 can integrate without relying on an unsafe transition. Answer, with the line
evidence gathered this pass:

1. `on_confirm_received()` returns `self.epoch_manager.current_epoch` **unchanged** when the guard at
   `ownership_fsm.py:108` refuses (`sender_state` not in IDENTITY_VERIFICATION/ACQUISITION), and the node discards
   the distinction: `_on_handoff_confirm` (`:566-577`) publishes the ACK with whatever value came back,
   unconditionally. A refused transition therefore emits a *commitment* — that is M1.
2. Nothing validates that an inbound ACK's `new_epoch` equals *pre-transaction epoch + 1*. `EpochManager` offers only
   `>=` / `<` semantics (`epoch_manager.py:21-34`), and both validators have **zero production call sites**;
   `set_epoch` ignores non-increasing values (`:46-50`) — that is M2.
3. **New this pass, and it changes the severity story:** `send_confirm()` (`ownership_fsm.py:191-197`) — the only
   producer of `ReceiverState.AWAITING_ACK` — has **no production caller anywhere in the tree** (grep: tests only).
   `on_confirm_ack_received()` requires AWAITING_ACK (`:201`). Meanwhile the *receiver's* `HandoffConfirm` is
   published by a different package, `target_acquisition_node.py:306-320`, with
   `header.ownership_epoch = self.active_epoch` and no coordination-FSM involvement. So in today's assembled system a
   successful handoff ends with A `CONFIRMED` and still `role=OWNER` at epoch e+1, B stuck at `role=CANDIDATE`, the
   owner-claim burst never fired, and **ownership never transferring**. The dual-OWNER state is reachable at FSM level
   (reproduced: 2 owners at epoch 1) and becomes reachable in production for anyone who adds the missing
   `send_confirm()` producer.

**Decision.** Exp-13 therefore (a) does not depend on the defective exchange for anything — its consensus output
remains advisory until `COMMIT_ARMED`; (b) implements M1 and M2 as **overrides inside the Exp-13 node**, so the
Exp-13 path publishes an ACK only when its FSM actually reached `CONFIRMED`, and accepts an ACK only when
`new_epoch == epoch_at_confirm_send + 1`; and (c) implements M3 **only for the belief stream**, where the
transaction-lifecycle prerequisite is satisfiable in Exp-13-owned code (`open_transaction` at the start of each
episode, `:80-88`), and **not** for confirm/ACK — because gating those on an empty transaction table would reject
every legitimate handoff, which is precisely the trap §6 warns about ("Confirm that all legitimate inbound messages
are associated with valid transactions before rejecting unknown transactions").

**Not done, deliberately:** the baseline node's handlers are not edited, `AWAITING_ACK`'s missing producer is not
created inside another package, and no claim of "zero dual ownership" is made for `cnp`. The defect stays open,
separately tracked, and is reported in `EXP13_SAFETY_FINDINGS_*.md` as **unfixed in baseline / guarded in Exp-13**.

**What would falsify it:** a single executed path where the Exp-13 node ends with two `OWNER` roles at one epoch.
That is what the regression test is for, and until it runs and passes, §6's "not fixed" wording stays in the reports.

---

## Decisions deliberately NOT taken

* No change to `convergence_threshold`, `max_rounds`, `bid_epsilon`, `utility_precision`, `MIN_AGREEMENT_ROUND` or
  the tie rule — none of these is implicated by D1–D8, and the previous pass measured that the round floor costs zero
  under lockstep, so it is not a performance knob either.
* No quorum, lease, or failure detector (§D3 rationale).
* No full CBBA (candidate lists / path-bid trim / cascade release) — vacuous at L = 1 (§D1).
* No `HandoffHeader` edit, no renaming or deletion of existing functions, no edits to Exp-07/12/14 files, no new
  dependency, no Git initialisation.
* No edit to the 105 existing tests: they read `coordination_node.py` and `robot_bringup.launch.py` by name
  (`test_cbba_negotiator.py:65,853-857,811-814`), neither of which is modified, so the tripwires stay green **and stay
  meaningful** — they now document the *baseline* node's non-selection rather than a missing feature. Their
  post-integration meaning is recorded in the test report instead of being silently left ambiguous.
