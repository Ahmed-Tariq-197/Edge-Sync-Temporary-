# EXP13 — Baseline ownership-safety remediation proposal (M1–M4)

**Experiment:** ES-EXP13 · **Date:** 2026-10-20 (Gate 1c)
**Status of everything in this file: `PROPOSED — NOT APPLIED`.**
**Why not applied:** every item edits a **shared** file (`coordination_node.py`, `fsm/ownership_fsm.py`, `fsm/epoch_manager.py`, `protocol/negotiation_manager.py`). No approval of record exists for any of them — see `EXP13_DECISION_AND_BLOCKER_REGISTER.md` §1 (decision-sheet rows all `☐`, "No approval below is granted, assumed, or implied", owning party `UNKNOWN (package-generic metadata only)`). The brief's §4.2 instruction is explicit: absent authorization, deliver the proposal and the test specification instead of the patch.

**Not Exp-13's defect.** The witness lives entirely in the pre-existing baseline. `cbba_negotiator.py` is not on the path (REQ-109 lane isolation: it never writes an epoch, never calls an FSM, never publishes).

**Reproduction of record (executed this pass, Tier U):**
`C:\Users\PC\edgesync_es13_probe\gate1b_20261020\fsm_witness.py` → `witness_console.txt` (162 lines, 10,246 B, `10fc9cb24b9b…97e9a12`), exit **0**, stderr empty. PATH 1 control 1 owner; PATH 2 (one lost `AcqStatus`) **2 owners at epoch 1**; PATH 3 (`e+2` recovery) **2 owners at epochs 3 and 1**. Node-handler transcription verified faithful against the live source by 5 runtime assertions. The FSM/epoch layer was **executed**; the DDS/QoS/topic layer was **statically inspected only** — no Tier-I measurement exists on this host (`import rclpy` → exit 1).

---

## 0. The prerequisite that gates M3 — measured first

The audit proposes "apply admission validation to responses, confirmations, ACKs and owner claims". Executed and static inspection shows that **cannot be done by adding a call**:

| Fact | Evidence |
|---|---|
| `admit_message` returns `(False, "Unknown transaction ID {txn}")` for any non-request message whose txn is not in the table | `protocol/negotiation_manager.py:69-73` |
| The table is populated only by `open_transaction`, which has **zero production call sites** | workspace-wide grep: `open_transaction` appears at its `def` (`:80`) and in **tests only** (`test_cbba_negotiator.py:868`, `test_phase3_fsm_timer_mocking.py:174,183,194`, `test_phase5_chaos_races.py:44,65`) |
| `mark_confirmed` is likewise never called from the node — the repository's **own test says so** | `tests/test_phase5_chaos_races.py:58-62`: *"BUG DETECTION: mark_confirmed() is never called from coordination_node.py. The NegotiationManager has the idempotent replay mechanism but it's dead code. This test documents the bug."* |
| `admit_message` is called at exactly **one** site in the node | `nodes/coordination_node.py:330`, inside `_on_handoff_request` (`:327`) — the request leg only |

**Consequence:** wiring `admit_message` into the confirm/ACK/claim handlers as-is would drop **every** legitimate message and break the handoff outright. M3 therefore has a **prerequisite (M0)**: open and confirm the transaction record in the production flow. M0 is itself a shared-file change and is listed below.

**Second measured obstacle for M2:** the epoch validators that exist cannot express the required rule. `validate_incoming_epoch` is `msg_epoch >= current` (`epoch_manager.py:30`) and `is_stale` is `msg_epoch < current` (`:34`). In PATH 2 the bad ACK carries epoch 1 while current is 1, so `is_stale(1)` → **False** and `validate_incoming_epoch(1)` → **True**. Measured in the probe. The contract needs `ack_epoch == pre_transaction_epoch + 1`, a predicate with no home in the current API.

---

## M1 — never acknowledge a confirmation the FSM refused

**Exact current path.** `nodes/coordination_node.py:563-578`:
`:565` `self.watchdog.disarm()` → `:566` `new_epoch = self.fsm.on_confirm_received()` → `:569-576` build ACK → `:577` `self.pub_confirm_ack.publish(ack)` → `:578` log.
`fsm/ownership_fsm.py:106-114`: the call **only** transitions from `IDENTITY_VERIFICATION` or `ACQUISITION`; otherwise it falls through to `:114 return self.epoch_manager.current_epoch`. The success path also returns an epoch (`:113 new_epoch`), so **the return value cannot distinguish "advanced to e+1" from "refused, here is e"** — and on refusal they are the same number.

**Invariant intended.** System Invariant 1 as stated in the FSM header (`ownership_fsm.py:1-7`): ownership transfers strictly upon an atomic `HANDOFF_CONFIRM_ACK` exchange at epoch `e+1`. An ACK must exist only if a confirmation transition actually happened.

**Observed breach (PATH 2).** A in `COMMITTED` (not `CONFIRMED`), `on_confirm_received()` returned `1`, and the node published `CONFIRM_ACK epoch=1`. That ACK is what makes B an owner.

### Option A — node-only, smallest possible change (4 lines, one file)

`proposal, not applied`:

```python
     def _on_handoff_confirm(self, msg: HandoffConfirm):
         """Peer verified Re-ID and sent CONFIRM."""
         self.watchdog.disarm()
         new_epoch = self.fsm.on_confirm_received()
+        if self.fsm.sender_state != SenderState.CONFIRMED:
+            self.get_logger().error(
+                f"REFUSED HANDOFF_CONFIRM: sender_state={self.fsm.sender_state.name}, "
+                f"no epoch advance, no ACK published (txn={msg.header.transaction_id}).")
+            return
```

Requires no FSM edit and changes no signature, so no test's expectations move. `SenderState` is already imported in the node (used at `:447`).
**Adverse effect to rule on before applying:** the watchdog is disarmed at `:565` **before** the refusal, so the refused node is left unarmed with an open transaction. Whether a refused confirm should (i) keep the watchdog armed, (ii) arm a recovery timer, or (iii) publish `HandoffCancel` is **D6 territory** (`HandoffCancel` wiring, receiver-side abort) and is deliberately **not** decided here. Option A without that ruling converts a dual-owner fault into a stall — safer, but a stall.

### Option B — make the FSM report the outcome (FSM + node)

Add, without touching the existing method or its 9 test call sites:

```python
     def confirm_or_refuse(self) -> Tuple[int, bool]:
         """Return (epoch, transitioned). False means no CONFIRMED state was entered."""
         if self.sender_state not in (SenderState.IDENTITY_VERIFICATION,
                                      SenderState.ACQUISITION):
             return self.epoch_manager.current_epoch, False
         return self.on_confirm_received(), True
```

and have the node branch on the boolean. Cleaner contract, larger blast radius (a second public method that can drift out of sync with the first). **Recommendation: Option A** as the minimal step; Option B only if the owner wants the outcome on the FSM's API rather than inferred from state.

**Owner / approval:** coordination owner (D5/D7/D11 area) + supervisor for the refusal-policy question. `ownership_fsm.py` and `coordination_node.py` are both shared.

---

## M2 — the received ACK epoch must equal `pre_transaction_epoch + 1` exactly

**Exact current path.** `fsm/ownership_fsm.py:199-207` `on_confirm_ack_received(ack_epoch)`: on `receiver_state == AWAITING_ACK` it calls `self.epoch_manager.set_epoch(ack_epoch)` (`:202`), then unconditionally sets `TRACKING` and `role = OWNER` (`:203-204`) and returns `True`.
`fsm/epoch_manager.py:46-50` `set_epoch` ignores any value that is not strictly greater, so a **stale-equal** epoch silently leaves B's epoch at 1 while B's role becomes OWNER — the node then stamps its claim burst with `msg.new_epoch` = 1 (`coordination_node.py:593`), i.e. the published epoch contradicts the asserted role.
`nodes/coordination_node.py:582` discards the return value entirely and proceeds to publish the 3× `OWNER_CLAIM` burst (`:586-596`) whatever the FSM answered.

**Invariant intended.** `epoch_manager.py:1-6` guarantee #1 ("monotonically ordered `e → e+1`") and #2 ("stale epoch messages `e <= current` are dropped immediately") — a guarantee **no production code implements**.

**Observed breach (PATH 2).** ACK epoch 1 accepted at current 1 → B `OWNER` epoch 1 while A is `OWNER` epoch 1. `contract_satisfied: False` printed by the probe.

### Proposed change (FSM, plus one node guard that makes it effective)

Record the pre-transaction epoch when the ACK window is armed, and require exact `+1`:

```python
     def send_confirm(self) -> bool:                       # :191
         if self.receiver_state == ReceiverState.VERIFYING:
+            self._ack_expected_epoch = self.epoch_manager.current_epoch + 1
             self.receiver_state = ReceiverState.AWAITING_ACK
             ...

     def on_confirm_ack_received(self, ack_epoch: int) -> bool:
-        if self.receiver_state == ReceiverState.AWAITING_ACK:
-            self.epoch_manager.set_epoch(ack_epoch)
+        if self.receiver_state != ReceiverState.AWAITING_ACK:
+            return False
+        if ack_epoch != self._ack_expected_epoch:
+            self._notify_change(f"ACK REFUSED: epoch {ack_epoch} != expected "
+                                f"{self._ack_expected_epoch}")
+            return False
+        self.epoch_manager.set_epoch(ack_epoch)
+        self.receiver_state = ReceiverState.TRACKING
+        self.role = OwnershipRole.OWNER
+        return True
```

```python
     def _on_handoff_confirm_ack(self, msg: HandoffConfirmAck):
-        self.fsm.on_confirm_ack_received(msg.new_epoch)
+        if not self.fsm.on_confirm_ack_received(msg.new_epoch):
+            self.get_logger().error(
+                f"REFUSED HANDOFF_CONFIRM_ACK epoch={msg.new_epoch}; no OWNER_CLAIM published.")
+            return
```

**Both halves are needed.** The FSM guard alone does not stop the claim burst, because `:582` throws the result away; the node guard alone does not stop B becoming OWNER. This pairing is why M1 and M2 must ship together.

**Compatibility risk (measured):** `on_confirm_ack_received` has 5 test references and `abort_to_recovery` 13. The `e+2` recovery path (`:126-135`) legitimately advances by 2, so any *test* that drives an ACK at `e+2` will now be refused — correct behaviour, but a contract change for whoever writes the next recovery test. `_ack_expected_epoch` must also be initialised in `__init__` (`:46-61`) and cleared in `on_ack_timeout` (`:209-215`), or a stale expectation leaks into the next transaction. ROS-free affected modules: `test_fsm_transitions.py`, `test_fault_injection.py`, `test_phase3_fsm_timer_mocking.py`, `test_phase5_chaos_races.py` (all run on this host); `test_auto_trigger.py`, `test_response_collector.py` need `rclpy` and **cannot be checked here at all**.

**Owner / approval:** coordination owner; the epoch contract should be countersigned by the msgs owner because `HandoffConfirmAck.msg:4` (`uint64 new_epoch # Incremented epoch (e+1)`) is where the `+1` rule is documented.

---

## M3 — admission/transaction/epoch validation on every inbound leg

**Exact current state (measured).** `admit_message` is reached only for `HandoffRequest` (`:330`). Unvalidated inbound handlers:

| Handler | Line | What it does with no validation |
|---|---|---|
| `_on_handoff_response` | `:498-513` | collects or **auto-commits**; see below |
| `_on_acq_status` | `:557-561` | feeds the watchdog and advances A to `IDENTITY_VERIFICATION` |
| `_on_handoff_confirm` | `:563-578` | advances epoch, publishes ACK (M1) |
| `_on_handoff_confirm_ack` | `:580-596` | makes a peer OWNER locally and bursts claims (M2) |
| `_on_ownership_event` | `:598-602` | demotes this robot to STANDBY |

**The out-of-window auto-commit path.** `:505-513`: if `_negotiation_window_open` is False, an `ACCEPT` response still executes `self.selected_receiver_id = sender_id; self.fsm.on_peer_accepted(); self.watchdog.arm()` — bypassing `_on_negotiation_timeout`'s utility selection (`:515-555`) entirely, with no protocol-version, receiver-id, transaction or epoch check. A late or duplicated response from any peer can therefore commit a transaction that has already been decided or aborted.

**Invariant intended.** One transaction, one admission decision per message, no state change from a message that does not belong to an open transaction.

**Smallest compatible change — and why it is not small.** Sequence required:
1. **M0 (new prerequisite):** call `negotiation.open_transaction(...)` where the transaction is actually created (A-side `broadcast_handoff_request` `:441-482`, B-side after `:330` admits), and `mark_confirmed(...)` / `close_transaction(...)` at the existing commit/abort points.
2. Then call `admit_message(..., is_request=False)` at the head of the five handlers above and `return` on refusal.
3. Replace the `:507-513` fallback with a validated path: admit → then decide, or drop and log. There is no third option that preserves the invariant.

Step 1 is what makes M3 a **design** change rather than a bugfix: it puts a stateful transaction table on the critical path of every experiment that uses the node, and its tombstone pruning (`:59`) then becomes a live timer. **Do not apply step 2 while step 1 is unwired** — §0 shows the result is total message loss.

**Duplicate-ACK interaction.** `negotiation_manager.py:63-67` already carves out an idempotent replay case (a duplicate `CONFIRM` at epoch `e` while current is `e+1` is admitted). Once `mark_confirmed` is actually called, that carve-out becomes load-bearing for M2's exact-`+1` rule, and the two must be tested together.

**Owner / approval:** coordination owner (node + `negotiation_manager`) and supervisor, because it changes the transaction contract every other experiment relies on. REV04 registers this area as **D11** ("may the existing collector and `OwnershipFSM` host the protocol") and **D6**.

---

## M4 — a receiver-side policy for an `e+2` recovery claim

**Exact current path.** `fsm/ownership_fsm.py:116-124` `on_owner_claim_received()` demotes **only** from `sender_state == CONFIRMED`. A robot that became OWNER through the ACK leg sits in `receiver_state == TRACKING` with `sender_state == STANDBY`, so an inbound `e+2` reclaim claim returns `False` and nothing happens. There is no other owner-claim consumer: `_on_ownership_event` (`:598-602`) calls that one method and logs "Transitioned to STANDBY" **unconditionally**, regardless of the return value — so the node reports a demotion it did not perform.

**Observed (PATH 3, measured).** After `abort_to_recovery` returns epoch 3, A is `OWNER` epoch 3 / `TRACKING`, B is `OWNER` epoch 1 / `TRACKING`, and B's `on_owner_claim_received()` → `False`. **Divergence is not resolved by recovery**, and the two owners disagree about the epoch, so the "highest epoch wins" claim in `epoch_manager.py:1-6` (guarantee #3) does not hold in the executed path.

**Invariant intended.** At most one OWNER per `(target_id, epoch)`; after a reclaim at `e+2`, any peer still asserting OWNER at `e+1` or below must converge.

**This one cannot be written as a patch, and here is why.** Every candidate is a different protocol with different liveness:

| Candidate | Behaviour | Cost |
|---|---|---|
| (a) receiver-side demotion | a peer whose `header.ownership_epoch > mine` and `role == ROLE_OWNER` demotes me unconditionally | simplest; a forged or stale high-epoch claim steals ownership, and it silently reverses a *legitimate* ownership B won |
| (b) verify-then-demote | demote only if the claim matches an open transaction this robot is party to (needs M0) | correct, but requires the transaction table; ties M4 to M3 |
| (c) refuse-to-serve + raise `DUAL_CLAIM` | neither side demotes; both stop tracking and a reconciler decides | safety-preserving, availability-reducing; needs a reconciler that does not exist |
| (d) lease/timeout | ownership expires unless refreshed | a new subsystem, new QoS, new timers; far outside a bugfix |

**Recommendation for the ruling, not a patch:** (b) is the only candidate consistent with the existing invariants, and it is only expressible after M0. The owner must additionally decide what `OwnershipEvent.ROLE_OWNER` means for a robot that is not party to the transaction, and whether `monitoring_node.py` (`edgesync_observability`) surfaces `DUAL_CLAIM`.

**Owner / approval:** supervisor + coordination owner. Flagged in REV04 as **D7** (integration shape) and **D6** (abort/cancel wiring).

---

## Regression test specification (to be written with the approved patch, not before)

All of §8.2 of the brief applies. The set below is what each item needs; every test must be paired with a **positive control** so it cannot pass vacuously. Today **none of these exists** — `EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:303` records that the dual-ownership witness is unprotected by any assertion.

| # | Test | Item | Expected |
|---|---|---|---|
| T-M1a | confirm arrives while A is `COMMITTED` (AcqStatus dropped) | M1 | no `CONFIRM_ACK` on the wire; A stays `COMMITTED`; error logged |
| T-M1b | **control:** confirm arrives after `AcqStatus(STATUS_ACQUIRED)` | M1 | ACK at `e+1` published; A → `CONFIRMED` |
| T-M1c | duplicate confirm (same txn, same epoch, arriving twice) | M1/M3 | second refused or idempotently admitted per `:63-67`; exactly one ACK |
| T-M1d | reordered: ACK for txn B arrives while txn A is open | M3 | refused — unknown or mismatched transaction |
| T-M2a | ACK epoch `e` (stale-equal) | M2 | refused; role stays CANDIDATE; **no claim burst** |
| T-M2b | ACK epoch `e+2`, `e-1`, `0`, negative | M2 | refused for all four (parametrised) |
| T-M2c | **control:** ACK epoch exactly `e+1` | M2 | accepted; `epoch_manager.current_epoch == e+1`; claim burst stamped `e+1` |
| T-M2d | role/epoch coherence | M2 | after acceptance, published `OwnershipEvent.header.ownership_epoch == fsm.epoch_manager.current_epoch` (catches the `set_epoch` no-op mismatch) |
| T-M2e | second transaction after a timeout | M2 | `_ack_expected_epoch` does not leak across transactions |
| T-M3a | response accepted while window closed | M3 | no `on_peer_accepted()`, no watchdog arm |
| T-M3b | `AcqStatus(STATUS_ACQUIRED)` from a foreign `sender_id` / wrong txn | M3 | refused; A stays `COMMITTED` |
| T-M3c | owner claim with `protocol_version` mismatch | M3 | refused before any FSM call |
| T-M3d | transaction lifecycle | M0 | `open_transaction` visible after request broadcast; `mark_confirmed` after ACK; `close_transaction` after claim; tombstone pruning does not drop a live txn |
| T-M4a | PATH-3 replay: e+2 reclaim after the PATH-2 fault | M4 | exactly one OWNER at the end; which robot is decided by the approved candidate (a)–(d) |
| T-M4b | both robots receive the same e+2 claim simultaneously | M4 | converge to one OWNER, no oscillation over 50 rounds |
| T-M5 | **full PATH-2 replay as a permanent regression test** | M1+M2 | single OWNER — the test that would have caught the original defect |
| T-M6 | message-loss sweep: drop each of {AcqStatus, Confirm, Ack, Claim} in turn | all | never two OWNERs; each drop yields a defined, logged outcome |

Execution tier for these tests: **ROS-free** for the FSM/epoch/manager layer (proven reachable — `ownership_fsm.py` and `epoch_manager.py` import stdlib only and executed in the probe); **Tier I required** for anything asserting an ACK or claim actually appeared on a DDS topic, which **cannot be run on this host**.

---

## Change manifest if M1–M4 were approved (all shared, all currently unapplied)

| File | SHA-256 (unchanged, live tree) | Items touching it | Approval needed |
|---|---|---|---|
| `nodes/coordination_node.py` | `60d720373a42…` (mtime 2026-10-19 21:42) | M1 (`:563-578`), M2 node half (`:580-596`), M3 (`:498-513`, `:557`, `:598`), M4 log honesty (`:602`) | coordination owner |
| `fsm/ownership_fsm.py` | `99670d02479b…` (mtime 2026-10-14 19:30) | M1 Option B (`:106-114`), M2 (`:191-207`), M4 (`:116-124`) | coordination owner |
| `fsm/epoch_manager.py` | recorded in Gate 0 (mtime 2026-10-14 19:30) | M2 needs an `== e+1` predicate (`:21-34`) | coordination owner |
| `protocol/negotiation_manager.py` | recorded in Gate 0 | M0/M3 — wire `open_transaction` / `mark_confirmed` / `close_transaction` | coordination owner |
| `edgesync_msgs/msg/HandoffConfirmAck.msg` | `8b4a8b41…` | **no edit proposed** — the `+1` contract is already documented at `:4` | — |

No digest in this table moved this pass. Nothing was patched.

---

## Do-not-claim list (for whoever writes the next report)

1. Do **not** claim zero dual ownership is proven by `cbba_negotiator.py` or by the Exp-13 benchmark. Its 0-count observations are in-process allocation events on a lane that never touches the FSM. Ownership safety must be validated across the transaction/FSM boundary.
2. Do **not** describe `fsm_witness.py` as a ROS 2 or DDS test. It executes the real FSM and models the node's two handlers under a fidelity check.
3. Do **not** apply M3's `admit_message` calls before M0 exists — see §0.
4. Do **not** pick M4's candidate silently. All four change availability differently.
5. Do **not** call the PATH-1 control a "fix". It is the pre-existing happy path, kept to prove the probe is non-vacuous.
