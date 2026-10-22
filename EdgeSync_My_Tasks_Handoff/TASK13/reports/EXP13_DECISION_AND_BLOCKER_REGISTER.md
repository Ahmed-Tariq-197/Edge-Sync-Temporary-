# EXP13 — Decision and blocker register

**Experiment:** ES-EXP13 — consensus-based task allocation
**Date:** 2026-10-20 (Gate 1a/1d pass)
**Prepared by:** Qoder agent working the "Controlled Completion, Design-Authority Resolution, Safety Remediation, and Verification" brief
**Purpose:** decide, on evidence, whether an authoritative design approval exists for each contested point; where none exists, name the exact decision required and who must make it. This file makes **no** design choice on anyone's behalf.
**Scope of this pass:** read-only inspection plus one Exp-13-owned report correction. No shared file was modified. Nothing in §5 was implemented.

Companion documents: `EXP13_SAFETY_REMEDIATION_PROPOSAL.md` (M1–M4, proposals only), `EXP13_INTEGRATION_GAP_ANALYSIS.md` (shared-file gaps A1–A5), `EXP13_FINAL_IMPLEMENTATION_AND_TEST_REPORT.md` (statuses and evidence).

---

## 1. Gate 1a determination

**Question put by the brief (§3.1):** does authoritative approval exist for D1, A1, A2, REQ-107, REQ-110, the meaning of `AGREED`, and ownership of the baseline safety remediation?

**Answer: no.** For every one of those items the evidence is a *recommendation*, a *requirement written by the proposal's own author*, or an *unanswered decision row*. Under the brief's rule — "A proposal or an unresolved decision record is not approval. A newer file is not automatically authoritative" — none of them is approval.

Measured evidence, all captured this pass:

| # | Checked | Command / method | Result |
|---|---|---|---|
| E1 | Decision rows answered? | read `ES-EXP13-SUPERVISOR-DECISION-SHEET-REV04.md` §4 (lines 189–201) | 11 rows `D1…D11`; **every row is `☐` with a blank answer and blank date**; no row carries a name |
| E2 | Does the sheet claim authority for itself? | same file, §4 preamble | states verbatim: **"No approval below is granted, assumed, or implied."** and "Nothing in this table is pre-ticked" |
| E3 | Is a person named anywhere? | same file §2 (lines 148–160) | "the workspace records **no per-file owner** — there is no `CODEOWNERS`, no `MAINTAINERS`"; owning party recorded as `UNKNOWN (package-generic metadata only)`; authority is expressed as **roles** (supervisor / msgs owner / coordination owner / observability owner / bringup owner / user) |
| E4 | Repo-level owner metadata | `find` for `CODEOWNERS`/`MAINTAINERS`; `setup.py:18-19` | no such file exists; `maintainer='EdgeSync Team'`, `maintainer_email='edgesync-dev@robotics.org'` — generic, not per-file |
| E5 | In-repo design record settled? | `docs/experiments/exp13/CBBA-VARIANT-DECISION-RECORD.md:6` | "Status: **DECISION REQUIRED FROM OWNER — implementation intentionally NOT started**" |
| E6 | Is REV04 signed off? | `ES-EXP13-PROPOSAL-01-REV04.md:219` | "Any *official* label other than \"IBBM-1\" is a **supervisor approval (D1)**, not a design choice" — i.e. REV04 itself defers its own designation to an approval it does not carry |
| E7 | Gate table | decision sheet §3 (lines 176–179) | `G5` needs **D5, D7, D10, D11** + D3 + D4, and "Without it → **no shared file may change**"; `G2` needs D1+D2 |
| E8 | Anything newer in the repo that grants approval | directory listing of `docs/experiments/exp13/` and repo root | only the three Exp-13 reports and the decision record; no approval artefact |

**Consequence.** Gate 2 (Exp-13-owned corrections) can proceed only for changes that are safe under *both* authorities. Gates 3 and 5 (shared baseline + ROS 2 integration) are `BLOCKED — SHARED-FILE APPROVAL REQUIRED`, and REQ-107 plus the `AGREED` semantics are `BLOCKED — DESIGN DECISION REQUIRED`.

---

## 2. The two live authorities, stated without resolving them

| | In-repo record | External REV04 |
|---|---|---|
| Path | `docs/experiments/exp13/CBBA-VARIANT-DECISION-RECORD.md` (15,619 B, `1aaf5144295f…9270788`) | `C:\Users\PC\edgesync_es13_proposal\ES-EXP13-PROPOSAL-01-REV04.md` (84,810 B, `c712c08b9b0f…fe61e461`) |
| Algorithm name | CBAA (CBBA with bundle length `L = 1`); "calling it 'CBBA' in a report, config key or log line would be a false label" | **IBBM-1**; states at `:211-219` it *is not* CBBA and *is not* CBAA; official label reserved to D1 |
| Scope | `L = 1`, 2 participants | `M = 2, T = 1, L = 1` (`:36` D2, recommendation only) |
| Belief content | winner vector | **full bid set**: `:274` "Every belief message carries **all bids the sender has seen**" (REQ-107); `:330-341` specifies `bids float64[2]`, `bid_present uint8[2]`, `winner_id`, `tie_rule_version`, `header = HandoffHeader` with a **new** `MSG_TYPE_ALLOCATION_BELIEF = 9` |
| Status | unresolved (`:6`) | authored, not approved (E6) |
| Selector convention | `negotiation_protocol: cbba` in `edgesync_bringup/config/exp13_cbba.yaml:49` while `max_bundle_size: 1` (`:60`) | same YAML value would name a protocol REV04 says is the wrong name |

Both documents are preserved unchanged. The current code follows the in-repo record for the human-readable label and REV04 for epoch binding; that mixture is **not** a resolution and is listed as B-1 below.

---

## 3. Decision register — the exact question and the exact owner role

Each row is a decision that cannot be inferred from the code. "Answer needed" is phrased so a single sentence from the owner unblocks it.

| ID | Contested point | Answer needed | Authority (role) | Blocks | Can proceed without it? |
|---|---|---|---|---|---|
| **B-1 / D1** | Official algorithm designation and the claims it licenses: CBAA (L = 1), IBBM-1, or another name. Includes whether `PROTOCOL_ID = "cbba"` may remain the selector string and whether `exp13_cbba.yaml` keeps its filename | "D1: adopt ☐ IBBM-1 ☐ CBAA L=1 ☐ other ____", plus which claim words are permitted in reports | Supervisor | Gate 2 design-dependent edits; the label in every artefact; report title | Partially — the tie-break fix and epoch binding were deliberately chosen to be safe under both readings |
| **B-2 / REQ-107** | Is a full-bid-set carrier **mandatory**, or is a winner-vector belief sufficient? If mandatory: fixed-size `float64[M]` array (`bids`/`bid_present`) or a variable-length bid list? | "REQ-107: mandatory / not mandatory; if mandatory, adopt REV04 §4.3 field set as written" | Supervisor (semantics) + msgs owner (carrier) | `ConsensusMessage` fields, `check_agreement` clause (iii), schema version 3, the belief `.msg` | **No** — implementing it changes the wire contract; not implementing it leaves clause (iii) unevaluated. Both are documented, neither is chosen |
| **B-3 / AGREED semantics** | What does `AGREED` mean for the *commit* path: local belief agreement (what exists today), all-declared-participant belief agreement, or a distributed commitment requiring an ack leg / quorum / lease? | "AGREED is ☐ a local diagnostic ☐ a commit precondition ☐ requires a certificate/ack exchange" | Supervisor + coordination owner | Whether unilateral `AGREED` is a defect or a documented property; whether an ack leg is authorised | Yes for Tier U — the module documents the limitation at `cbba_negotiator.py:704-718`; **no** for integration |
| **B-4 / REQ-110** | Confirm ownership-epoch binding is mandatory and fix the wire contract: reuse `HandoffHeader.ownership_epoch` (`HandoffHeader.msg:8`, already declared) rather than a new field | "REQ-110: mandatory; epoch rides on the existing `HandoffHeader.ownership_epoch`" | Supervisor + msgs owner | Whether `Belief` must embed `HandoffHeader`; whether schema 2 → 3 is needed | Already implemented **in-process**; the wire half is blocked with B-2 |
| **B-5 / A1 (D5)** | Is the `negotiation_protocol` selector approved for `coordination_node.py`, with what default and what fallback for an unknown value? | "D5: selector contract adopted ☐; default = ____; unknown value → ☐ error ☐ baseline" | Coordination owner | Any edit to `coordination_node.py`; any claim that the YAML is functional | Yes — the gap is measured and reported; no half-integration was produced |
| **B-6 / A2 (D5 side)** | Is `exp_id` a runtime selector, a provenance label, or not required on that node at all? | "exp_id is ☐ a selector ☐ provenance only ☐ not to be added" | Coordination owner + observability owner | Whether `coordination_node.py` gains a parameter | Yes — evidence says the project's provenance convention is a per-experiment JSON written by a `--out` flag, and no logging path keys off `exp_id` today (see `EXP13_INTEGRATION_GAP_ANALYSIS.md` §4) |
| **B-7 / D10** | Does the belief `.msg`, `MSG_TYPE_ALLOCATION_BELIEF = 9` and the `edgesync_msgs` rebuild/deploy plan get approved? | "D10: approve ☐ / decline ☐; if approve, who rebuilds and deploys `edgesync_msgs`" | msgs owner | Every Tier-I test; any cross-process belief exchange | Yes for Tier U only |
| **B-8 / D11** | May the existing collector and `OwnershipFSM` host the protocol, or is an adapter required? | "D11: ☐ adapter ☐ hosting on existing classes ☐ neither yet" | Coordination owner | Integration shape; G5 | Yes — specified, not built |
| **B-9 / D7** | Branch beside the baseline, or wrap it? | "D7: ☐ branch ☐ wrapper (equivalence capture first)" | Supervisor (shape) + coordination owner | G5 "hard", per the decision sheet | Yes |
| **B-10 / M1–M4 ownership** | Who owns the baseline dual-`OWNER` remediation, and is it in this experiment's scope at all? The witness is now reproduced (see §4) and is entirely in shared files. | "M1–M4 are owned by ☐ coordination owner as baseline work ☐ a separate defect ticket ☐ Exp-13 under written approval" | Coordination owner + supervisor | Any repair to `coordination_node.py` / `ownership_fsm.py`; Tier-I safety claims | Yes — the patch proposals are written and unapplied |
| **B-11 / Participant count** | Is the production contract `swarm_participant_count: 2` (YAML) or arbitrary `N ≥ 2` (what the benchmark sweeps)? | "Production scope: ☐ N = 2 ☐ N ≥ 2 arbitrary, with a certified max" | Supervisor (D2) | Whether N = 3…8 rows are results or exploratory; whether the YAML changes | Yes — the two are reported separately and the YAML was not changed |
| **B-12 / D3, D4** | Timing-budget placement and definitions; metrics-storage ownership and counters in `coordination_node.py` | per decision sheet rows | Supervisor + observability owner + bringup owner | G5 inputs; any latency claim | Yes — no timing claim is made anywhere in this pass |

Rows D6 and D9 of the sheet are the only ones REV04 marks deferrable with no consequence, and only because the recommended scope excludes withdrawal and impaired-network claims; neither is treated as approved here.

---

## 4. Gate 1b — baseline ownership-safety witness: **reproduced in the current tree**

The brief requires reproduction before any remediation proposal, and forbids claiming the earlier witness reproduces unless it actually does. The audit's probe was ephemeral stdin and was **not** saved (`EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md:374-383`), so per §4.1 a minimal isolated probe was written outside the repository and executed against the live code.

```
cd "C:\Users\PC\edgesync_es13_probe\gate1b_20261020"
python fsm_witness.py > witness_console.txt 2> witness_err.txt ; exit 0   (stderr 0 B, 162 lines)
```

The probe imports the **real** `OwnershipFSM` and `EpochManager` (`fsm/ownership_fsm.py`, `fsm/epoch_manager.py`) and transcribes the node's two handlers (`coordination_node.py:563-578`, `:580-596`, `:598-602`) into a shim, because `rclpy` is absent (exit 1, `ModuleNotFoundError: No module named 'rclpy'`) and `CoordinationNode` cannot be instantiated. The transcription is **checked against the live source at runtime** — five fidelity assertions, all `SHIM FAITHFUL`/`CONFIRMED` — so the probe fails closed if the node changes under it.

Tier statement: **executed** for the FSM/epoch layer; the ROS 2 topic, QoS and DDS lanes are **statically inspected only**, never run. This is a Tier-U reproduction of a Tier-I defect.

### 4.1 PATH 2 — one lost `AcqStatus` sample produces two owners

| Step | Event | FSM return | State after |
|---|---|---|---|
| — | initial | — | A `ugv_1` role=OWNER sender=TRACKING epoch=1 · B `ugv_2` role=STANDBY sender=STANDBY receiver=IDLE epoch=1 |
| EV1–EV7 | `trigger_candidate` / `send_request` / `on_request_received` / `accept_request` / `on_peer_accepted` / `start_acquisition` / `target_locked` | all `True` | A COMMITTED (still OWNER) · B VERIFYING |
| **EV8** | **`AcqStatus(STATUS_ACQUIRED)` lost** on the BEST_EFFORT depth-1 subscription (`:153-157`, `:167`) → `on_peer_verifying()` never called | — | A **stays COMMITTED** |
| EV9 | B `send_confirm()` | `True` | B AWAITING_ACK |
| EV10 | A `_on_handoff_confirm()` → `fsm.on_confirm_received()` | **`1`** (no transition; `ownership_fsm.py:114` returns the *unchanged* epoch) | A COMMITTED, role=OWNER — **yet the node publishes `CONFIRM_ACK epoch=1`** (`:575-577`) |
| EV11 | B `_on_handoff_confirm_ack()` → `fsm.on_confirm_ack_received(1)` | `True`; `set_epoch(1)` is a no-op (`epoch_manager.py:48`) | **B role=OWNER, epoch=1** |
| EV12 | A `_on_ownership_event()` → `fsm.on_owner_claim_received()` | `False` (requires CONFIRMED, `:116-124`) | **A role=OWNER, epoch=1** |
| **Final** | | | **A OWNER epoch 1 · B OWNER epoch 1 → owner count 2, same epoch** |

Observed invariant breaches, printed by the probe: *"A published a CONFIRM_ACK while its FSM never entered CONFIRMED"* and *"ACK epoch 1 is not e+1=2 yet was accepted"*.

### 4.2 PATH 1 — control (the probe is not vacuous)

Identical script, `AcqStatus` delivered: `on_confirm_received()` advances to **2**, ACK epoch 2, `2 == pre_transaction_epoch(1) + 1` → `contract_satisfied: True`, A demotes to STANDBY, **owner count = 1**. Non-vacuity check `PASS`.

### 4.3 PATH 3 — `e+2` recovery does not reconcile

After the PATH-2 fault, A `abort_to_recovery()` returns **epoch 3**, `recovery_to_solo_tracking()` → OWNER/TRACKING; B receives the e+2 claim and `on_owner_claim_received()` returns **False** (no receiver-side demotion path exists from `sender_state=STANDBY`). Final: **A OWNER epoch 3 · B OWNER epoch 1 — two owners at different epochs, divergence unresolved.**

### 4.4 Validators exist but are never called

Measured in the live tree: `validate_incoming_epoch` and `is_stale` appear **0 times** in `coordination_node.py`. Worse, for PATH 2 `is_stale(1)` → **False** and `validate_incoming_epoch(1)` → **True** (`epoch_manager.py:21`, `:32` use `>= current`), so wiring the existing validators in would **not** catch this case; the contract needs the predicate "ACK epoch == `pre_transaction_epoch` + 1", which the current API cannot express.

`admit_message` is called at exactly **one** site in the node (`:330`, inside `_on_handoff_request` at `:327`). The five other inbound handlers — `_on_handoff_response` `:498`, `_on_acq_status` `:557`, `_on_handoff_confirm` `:563`, `_on_handoff_confirm_ack` `:580`, `_on_ownership_event` `:598` — perform no admission, transaction or epoch validation. `_on_handoff_response` also contains an out-of-window auto-commit path (`:507-513`) that calls `on_peer_accepted()` and arms the watchdog with no validation.

`on_peer_acquisition_started` has **zero** production call sites (only `ownership_fsm.py:90` and nine test files); `on_peer_verifying` has exactly **one** (`coordination_node.py:561`). So `SenderState.ACQUISITION` is unreachable in production, and the tests exercise a transition the deployed node can never take.

### 4.5 Attribution

The defect is entirely in pre-existing shared files (`coordination_node.py`, `fsm/ownership_fsm.py`, `fsm/epoch_manager.py`). `cbba_negotiator.py` is not on the path: it never writes an epoch, never calls an FSM, never publishes (REQ-109 lane isolation). Reproducing it here is **not** evidence about Exp-13's algorithm, and the zero-dual-owner counts in the Exp-13 benchmark are **not** evidence about this defect. The two `OWNER` states were produced by the ACK exchange with no consensus participant involved.

---

## 5. What proceeds without choosing a protocol

Per brief §3.2 item 4 — these need no design ruling and are all Exp-13-owned:

1. **Preservation.** The five Exp-13-owned files, three benchmark artefacts and all three prior reports are unchanged this pass. Verified by digest (§2 of the final report).
2. **Evidence-quality correction of the Exp-13-owned report.** Two factual errors found in Gate 0 were fixed in place, backed up first: the message-set size (9 → **11** files) and the claim that the message set lacks an epoch field (`HandoffHeader.msg:8` already declares `uint64 ownership_epoch`). Statuses did not change.
3. **The reproduction of record.** §4 is an isolated, saved, re-runnable probe. No shared file was touched to produce it.
4. **Proposals and gap analysis.** Deliverables B, C and E are documents; they specify exact patches without applying them.
5. **Test re-verification.** 105 / 136 / exit-2 baseline re-executed and captured (§1 of the final report).

Explicitly **not** proceeding: REQ-107's clause (iii), any ack/quorum/lease, the belief `.msg`, `negotiation_protocol` on the node, `exp_id` on the node, M1–M4, and any threshold change to improve a benchmark number.

---

## 6. Status

`BLOCKED — DESIGN DECISION REQUIRED` for B-1…B-12 as design-dependent work; `BLOCKED — SHARED-FILE APPROVAL REQUIRED` for every M1–M4 edit and for A1–A5. The Exp-13 Tier-U module itself is unchanged and its tests still pass at the recorded counts. The fastest unblocking action is a one-line answer to **D1** and **REQ-107** (§3 rows B-1, B-2), which together decide whether the current belief schema is final.
