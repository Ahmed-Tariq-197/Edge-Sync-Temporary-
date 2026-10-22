# EdgeSync Experiment 13 — Post-Implementation Safety and Contract Audit

Audit ID: ES-EXP13-AUDIT-01
Date: 2026-10-20 (Windows host, executed in-session)
Mode: **STRICT READ-ONLY.** No source, test, configuration, documentation or project file was created,
edited, overwritten, renamed, moved or deleted other than this report. No fix was applied. No Git
command of any kind was run (the workspace is not a Git repository). No dependency was installed.
Author: agent session, under the Exp-13 audit instruction set and ES-EXP07 MASTER EXECUTION PROMPT §1–§12.

---

## 0. Evidence basis, classification key and what is *not* in this audit

| Class | Meaning in this report |
|---|---|
| **EXECUTED** | Result produced by running real code in this session; exact command in §9. |
| **STATIC** | Conclusion read from source; the file cannot be imported/executed on this host (`rclpy` absent). |
| **DOC** | Comparison between two documents/artefacts on disk. |
| `PASS` | Property holds under the evidence class stated. |
| `FAIL` | Property does not hold; a witness or measurement is given. |
| `BLOCKED` | Cannot be evaluated on this host, or cannot be evaluated until an owner decision exists. |
| `NOT VERIFIED` | Not evaluated at all in this audit; stated as an open item, never as a pass. |

**Evidence tier of everything in §2–§5: Tier U** (one process, no ROS graph, no DDS, no network, no
Jetson, no robot). `FAIL` findings about the node are FSM-level or static; none is a distributed or
on-hardware measurement. A laptop/simulation number is not a Jetson number, and no node-level
integration test passed here because none can run here (§6).

**Contract sources available (verified by `ls`/`find`).** In the repository:
`docs/experiments/exp13/CBBA-VARIANT-DECISION-RECORD.md` (15,619 B), `src/edgesync_bringup/config/exp13_cbba.yaml`
(95 lines) and `EXP13_IMPLEMENTATION_AND_VERIFICATION_REPORT.md` (256 lines, the record being audited).
**The authoritative design set lives outside the repository**, in
`C:\Users\PC\edgesync_es13_proposal\`: `ES-EXP13-PROPOSAL-01-REV04.md` (84,810 B, status line 6 =
"**DESIGN-CLOSED FOR THE TIER-U ALGORITHM MODULE** · **NOT AUTHORISED FOR ROS 2 INTEGRATION** (§11,
Verdict A / Verdict B)"), `ES-EXP13-TECHNICAL-DEFECT-REGISTER-REV04.md` (35,486 B),
`ES-EXP13-SUPERVISOR-DECISION-SHEET-REV04.md` (21,247 B), `ES-EXP13-FILE-CHANGE-MANIFEST-REV04.md`
(19,826 B), plus REV01–REV03 and two change logs. The implementation was compared against REV04's
REQ/INV/T contracts and the result is §4, §5 and findings AUD-16…AUD-24. **Caveat carried forward:**
these are the only Exp-13 authorities and none of them is version-controlled or stored in the repo, so a
teammate checking out the repository cannot see the contract the code was written against.

---

## 1. Safety-path trace: belief → local agreement → episode decision → request/response → FSM → ownership

The instruction was to prove presence or absence of the episode-level gate **from the actual call path**,
not by assuming the benchmark's gate exists in production. It does not exist in production, and the
stronger fact is that production never consults the Exp-13 negotiator at all.

### 1.1 What consumes the Exp-13 negotiator

Repo-wide grep for `CBBANegotiator` / `cbba_negotiator` excluding `__pycache__`, `.pytest_cache`, `.git`:
**every** hit is one of — the module itself
([cbba_negotiator.py:174](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/edgesync_coordination/negotiation/cbba_negotiator.py#L174)),
its package re-export (`negotiation/__init__.py:6,10,22`), the benchmark harness (`scripts/run_exp13_unit_benchmark.py:63,249,251,451`),
the dedicated test file, or comments inside `exp13_cbba.yaml`. There is **no** occurrence in
`nodes/`, `fsm/`, `decision/`, `protocol/`, any launch file, or any other experiment.
`negotiation_protocol` appears only in the YAML (`exp13_cbba.yaml:49`, self-labelled "inert until
approval A1") and as the module's selector constant (`:43`); `coordination_node.py` declares 24
parameters (`:39-66`) and neither `negotiation_protocol` nor `exp_id` is among them.
Two existing tests enforce exactly this absence and pass (§6):
`test_selector_is_declared_in_configuration_not_in_the_running_node` (asserts `"cbba_negotiator" not in node_src`,
[tests/test_cbba_negotiator.py:816](file:///C:/Semester%207/EdgeSync-main%20(3)Last/EdgeSync-main%20(3)Last/EdgeSync-main/Implementation%20Phase/edgesync_ws/src/edgesync_coordination/tests/test_cbba_negotiator.py#L816))
and `test_exp13_config_is_not_yet_loaded_by_the_bringup_launch`.
**Classification: `PASS` — the unilateral-`AGREED` hazard is latent, not live.** AUD-01.

### 1.2 The chain, stage by stage, with the verified line numbers

| # | Stage | Where it happens | What actually authorises it |
|---|---|---|---|
| 1 | incoming consensus belief | `CBBANegotiator._accept` `:496-546` — nine refusal classes in this order: POST_TERMINAL `:506`, MALFORMED `:508`, UNSUPPORTED_SCHEMA `:510`, MALFORMED `:512`, SELF `:514`, UNKNOWN_SENDER `:516`, FOREIGN_EPISODE `:518`, MALFORMED `:520`, **DUPLICATE `:524`, STALE `:526`, TARGET_MISMATCH `:529`/`:533`**, then three more MALFORMED `:541-545` | admission is *content + watermark*; there is no signature, nonce, epoch or membership attestation |
| 2 | local agreement status | `check_agreement()` `:614-647`; loop `:634`; a participant that has never reported ⇒ `return False` `:639-640` | **a purely local predicate over what peers last *told us*** — the module says so in its own docstring `:628-630` |
| 3 | episode decision (module) | `run_round` ladder `:689-744`: POST_TERMINAL absorb `:704`, deadline ⇒ TIMEOUT, round bound ⇒ MAX_ROUNDS, then DUAL_CLAIM / AGREED / NO_WINNER / LOCAL_STABLE; `is_converged()` is `status is AGREED` `:786` | `AGREED` is set **the instant this agent's local test passes** — nothing waits for peers |
| 4 | episode decision (harness only) | `run_exp13_unit_benchmark.py:299` `res.decided = terminal and len(converged) == len(negotiators)` | **this gate exists in the harness and nowhere else.** It is not a production control |
| 5 | handoff request | node: TTO trigger `:251-277` → `broadcast_handoff_request()` `:441-482` → FSM `trigger_candidate`/`send_request` → publish `:477` → window `:482-496` | local TTO threshold; response window `t_neg_timeout_ms` = 80 ms |
| 6 | response collection / selection | `_on_handoff_response` `:498-513`; `_on_negotiation_timeout` `:515-555`, sort `:538`, winner `:539`, commit `:551`, watchdog arm `:554` | highest **self-reported** `float(msg.utility)` among ACCEPTs; no peer-side confirmation required |
| 7 | ownership FSM | `on_peer_accepted` fsm `:82-88`; `on_confirm_received` fsm `:106-114`; `on_confirm_ack_received` fsm `:199-207`; `abort_to_recovery` fsm `:126-135` | FSM state guards only — per-process, no cross-node check |
| 8 | final ownership / command authority | node `:563-578` (publish `HandoffConfirmAck`) and `:580-596` (**become OWNER**, publish 3-copy `OWNER_CLAIM`) | **`HandoffConfirmAck.new_epoch` taken verbatim from the wire** (`:582`) |

### 1.3 Answer to the required question

> "Can an agent that locally reports `AGREED` cause an ownership transfer or robot action without
> evidence that the complete declared participant set converged?"

**Not today, and not by construction — but for a different reason than the module's design.**
Nothing in the repository wires `check_agreement()`, `is_converged()`, `get_assignment()` or
`metrics()` to any publisher, timer, FSM transition or actuation call (§1.1). The commit decision is
made entirely by stages 6–8, which never consult a consensus object. So the *Exp-13 module* cannot
cause ownership transfer. **The existing baseline path can, and §3 shows it does.**

If Exp-13 is integrated as designed, `AGREED` would enter stage 5 as a *selector* (whom to address).
The decision record §2 already requires that the allocation stay **advisory**
(`CBBA-VARIANT-DECISION-RECORD.md:85-89`) and states that with that constraint "a wrong consensus
cannot create two owners, because the ACK exchange is the only ownership-mutating event".
**This audit measured that the ACK exchange is itself capable of creating two owners** (§3.3) — so the
load-bearing premise of that safety argument is not sound as the code stands. This is the single most
important result of the audit.

---

## 2. Adversarial partition probes (all EXECUTED against the real negotiator/FSM objects)

Model for the search probes: lossless broadcast with **arbitrary delay and arbitrary order** — each
agent keeps a mailbox and may ingest one pending belief, all pending beliefs, or none, then publishes.
No message is ever dropped and no peer is ever unreachable in this model; it is the *best-case* network.
Search is a memoised depth-first sweep over live `CBBANegotiator` objects (`deepcopy` per branch), so
every conclusion is the real class's behaviour, not a re-implementation.

| Probe | Question | Result | Class |
|---|---|---|---|
| P10 | Can two agents, or an agent vs a peer, ever hold **different** `winning_agents` maps under asynchrony? (predicates (P) and (Q)) | **No witness** across **247,510 reachable states** in 4 profiles: N=3 equal bids 94,650 states; N=3 distinct 71,172; N=3 two-high tie 81,554; N=2 134. Max simultaneously `AGREED` reached 3. | EXECUTED `PASS` |
| P10-U | Can one agent be `AGREED` while a declared participant is not? | **Yes, pervasively**: 68/134 (N=2), 18,530/71,172, 27,761/81,554 states. | EXECUTED `FAIL` |
| P11 | Can an `AGREED` agent coexist with a participant that terminated **non-converged**? | **Yes**: 40/134 states (N=2 equal bids), 35/138 (N=2 distinct). In **100 %** of those states the `AGREED` agent's `metrics()['convergence_time_s']` is a **number**, not `None`. Witness: `ugv_2` `AGREED`, assignment `{'task_0': 'ugv_1'}`, `convergence_time_s = 0.0`, while `ugv_1` is `LOCAL_STABLE` (terminal, never converged) with the same assignment. | EXECUTED `FAIL` |
| P12 | Is the *winner identity* order-independent? (equal utilities, acting-order permutation sweep, restricted to `AGREED` agents) | **No.** N=2: 2 permutations give `ugv_1` in one and `ugv_2` in the other. N=3: three distinct winners (each in 2 of 6 permutations). N=4: four distinct winners. With distinct utilities the highest bidder always wins (correct). | EXECUTED `FAIL` (qualification) |
| P12b | Same sweep, distinct utilities | Winner is the true maximiser in every interleaving; only incomplete episodes differ. | EXECUTED `PASS` |
| P3–P5 | Post-timeout / post-terminal delivery; all nine refusal classes; restart re-admission | Retained from the implementation session and re-confirmed this session by the guard-order reading in §1.2: a real `TIMEOUT`/terminal episode cannot be revived (POST_TERMINAL absorbs at `:506`/`:704`); every refusal class fires; refusals never mutate consensus content. **`reset()` clears the watermark (`:317`), and `ConsensusMessage` (`:113-132`) carries no `ownership_epoch`, no nonce and no participant set**, so a stale belief from the previous attempt is re-admitted under the same `episode_id`. | EXECUTED + STATIC |
| F1 | Can `_on_handoff_confirm` tell that the FSM **refused** the CONFIRM? | **No** — EXECUTED: FSM in `TRACKING`, `on_confirm_received()` returns `1` = current epoch, state unchanged. The node uses that return value as the ACK epoch with no refusal test (`coordination_node.py:566,575-576`). | EXECUTED `FAIL` |
| F2 | Is the ownership epoch validated? | **No** — EXECUTED: receiver in `AWAITING_ACK`, `on_confirm_ack_received(999)` ⇒ `accepted=True`, role `OWNER`, state `TRACKING`, epoch `999`. Any `new_epoch` on the wire becomes the node's ownership epoch (`:582`, fsm `:199-207`). | EXECUTED `FAIL` |
| F5 / PATH 2 | Does a lost `AcqStatus` break the zero-dual-ownership invariant? | **Yes — measured, §3.3.** | EXECUTED `FAIL` |
| F4 | Abort arithmetic | `abort_to_recovery` after `COMMITTED` ⇒ epoch `3` (e+2), role `OWNER` — as documented. | EXECUTED `PASS` |
| node-level integration | Do the node handlers behave as traced? | `BLOCKED` — `rclpy` absent; `coordination_node.py` cannot be imported. §1.2 and the F-series are the strongest available substitute: FSM transitions executed, node lines read. `NOT VERIFIED` at runtime. | BLOCKED |

**Explicit non-claim:** none of the above is evidence about DDS behaviour, real radio loss, broker
partitions, clock skew, restart timing or Jetson latency. P10's "no witness" is a statement about the
*merge rule under a best-case network*, which is precisely the case in which distributed safety was
never at risk; it does not show the partitioned case is safe, because the partitioned case is not
modelled by the module at all (no failure detector, no lease, no quorum).

---

## 3. Exact unilateral-agreement and ownership-commit behaviour

### 3.1 What `AGREED` proves and what it does not

`check_agreement()` compares, task by task, *the belief each peer last sent us* against our own vector
(`:634-647`). There is no acknowledgement leg: a peer's `AGREED` is never reported to anyone, and
`ConsensusMessage` (`:113-132`) has no field that could carry it. Consequently a local `AGREED` is
evidence of exactly one thing — *this agent's mailbox contents are self-consistent with what each peer
asserted at the time it asserted it*. It is not evidence that any peer also reached `AGREED`, and P11
shows a peer may be **permanently non-converged** (`LOCAL_STABLE`, terminal) at the moment of, and
after, this agent's `AGREED`.

**No local observable distinguishes the safe from the unsafe case.** Measured earlier in this session:
in a one-way partition the unheard-but-all-hearing agent's `metrics()` dict is field-for-field identical
(diff `{}`) to the same agent's metrics in a fully converged episode. There is no key for peers'
statuses, no ack count, no epoch, and `convergence_time_s` is numeric in 40/40 unilateral states.

### 3.2 What commits an episode

* In the **harness**, `run_exp13_unit_benchmark.py:299` requires *every* declared participant to be
  terminal **and** `AGREED`, else `decided=False`; the fresh audit run (§6) shows **10 of 25 exp13 rows
  `decided=False`, and 5 of those still carry `converged_agents ≥ 1`** — i.e. the harness gate is
  withholding a commit that a local view would have approved. This is a harness property, not a
  production one.
* In **production**, the commit is `_on_negotiation_timeout:538-554` (pick highest self-reported
  utility, `fsm.on_peer_accepted()`, `watchdog.arm()`), or — with no window open at all — the
  unconditional fallback at `:506-512`, which commits on the first `DECISION_ACCEPT` received, subject
  to **no** admission check. `NegotiationManager.admit_message` is called from exactly **one** site in
  the whole node (`:330`, the `HandoffRequest` path); `_on_handoff_response`, `_on_acq_status`,
  `_on_handoff_confirm`, `_on_handoff_confirm_ack` and `_on_ownership_event` call it **zero** times.

### 3.3 The dual-ownership witness (measured at FSM level)

`SenderState.ACQUISITION` has **no producer**: `on_peer_acquisition_started` (fsm `:90`) has no call
site anywhere in the package. The only transition into `IDENTITY_VERIFICATION` is
`_on_acq_status → on_peer_verifying` (`coordination_node.py:557-561`) driven by `AcqStatus` on a
**BEST_EFFORT, depth-1** subscription (`:153-157`, `:166`). So one lost or overwritten `AcqStatus`
leaves A in `COMMITTED` when `HANDOFF_CONFIRM` arrives. Executed trace:

```
PATH 1 (AcqStatus delivered)      A on_confirm_received() -> ACK epoch 2, A CONFIRMED -> STANDBY
                                  B TRACKING role=OWNER epoch=2.                 single owner.
PATH 2 (AcqStatus LOST — one best-effort sample)
  A: on_confirm_received() returned 1 ; state COMMITTED (unchanged) ; role OWNER
  node publishes HANDOFF_CONFIRM_ACK stamped epoch 1  (coordination_node.py:566,575-577)
  B: on_confirm_ack_received(1) accepted=True -> role=OWNER state=TRACKING epoch=1
  A: on_owner_claim_received() accepted=False (requires CONFIRMED, fsm:116-124) -> A stays OWNER
  ==> ZERO DUAL-OWNERSHIP VIOLATED: both nodes role=OWNER at epoch 1
PATH 3 (watchdog recovery)
  A: abort_to_recovery -> epoch 3, role OWNER, TRACKING
  B: still role=OWNER, TRACKING, epoch 1; no receiver-side demotion path exists for an e+2 claim
  ==> divergence is NOT resolved by recovery; two owners at epochs 3 and 1
```

Note that `is_stale(1)` returned **False** and `validate_incoming_epoch(1)` returned **True** for this
case: the existing `EpochManager` validators (`epoch_manager.py:21`, `:32`) use `>= current`, so even
wiring them in would **not** catch it. The contract requires the ACK epoch to be **exactly `e+1`**, a
predicate the current API cannot express. Also `set_epoch` ignores a non-increasing value, so B becomes
`OWNER` while its epoch stays at the pre-handoff value — the node's own state is then internally
inconsistent with the claim burst it publishes (`:586-596`, stamped `msg.new_epoch` = 1).

**Scope and ownership of this defect:** it is entirely in the **pre-existing baseline** —
`coordination_node.py` and `fsm/ownership_fsm.py`, files Exp-13 was explicitly told not to modify
("Do not modify shared ROS files, message definitions, launch files"). It is *not* caused by
`cbba_negotiator.py`. It violates System Invariant 1 as stated in the FSM's own header
(`ownership_fsm.py:1-7`) and contradicts `epoch_manager.py:1-6` guarantee #2 ("Stale epoch messages
(e <= current) are dropped immediately"), which **no production code implements** — the two validators
are referenced only by `fsm/epoch_manager.py` and by `tests/test_fault_injection.py:281-286` and
`tests/test_phase5_chaos_races.py:193-197`. A mechanism that is unit-tested but never called is not a
safety mechanism. AUD-07/AUD-08/AUD-09.

---

## 4. Algorithm-identity ruling

| Item | Evidence | Ruling |
|---|---|---|
| Approved variant | `CBBA-VARIANT-DECISION-RECORD.md:31-51` classifies the removed code as CBAA (CBBA with L = 1) and states "Calling it 'CBBA' in a report, config key or log line would be a false label"; `:61-68` recommends CBAA L = 1 and forbids speculatively building full CBBA | Authority located and read |
| Implemented variant | `ALGORITHM_VARIANT = "CBBA with bundle length L = 1"` wording at `:46`, `max_bundle_size != 1` raises at `:208-212`, no candidate list in `ConsensusMessage` (`:113-132`), no path-bid consistency trim anywhere, `L = 1` release is a single-task clear (`_release_task`, `:602`), cascade explicitly not ported | **matches CBAA, not full CBBA** |
| Selector string | `PROTOCOL_ID = "cbba"` at `:43`, documented at `:44` as "Selector value used by the team guide's `negotiation_protocol` parameter" and inert (`exp13_cbba.yaml:49`) | `PASS` — retained for filename/selector continuity, not presented as an algorithm claim |
| Scope vs design | `swarm_participant_count: 2`, `target_count: 1`, `max_bundle_size: 1` (`exp13_cbba.yaml:56-60`) | `PASS` for the Tier-U module |
| Missing full-CBBA semantics if full CBBA were required | per-task candidate list `cand_i`; path-bid consistency check trimming candidate lists when a peer's bid exceeds the local bid at the corresponding bundle index; cascade release of bundle suffixes; bundle growth beyond L=1 | enumerated; **not** implemented, correctly |
| Naming in the deliverable filename (`cbba_negotiator.py`) and module name | `exp13_cbba.yaml:8` records the retention rationale | accepted, matches the record |

**Ruling (in-repo authority): identity is correct and not inflated** — the module is CBAA (L = 1), and
every artefact that states the algorithm says so. No relabelling and no scope expansion is proposed.

**Ruling against the authoritative design (REV04): the designation is NOT settled.** REV04 §3.1 closes
D1 as a *recommendation* named **IBBM-1**, and states explicitly
(`ES-EXP13-PROPOSAL-01-REV04.md:211-219`): "(a) IBBM-1 **is not** CBBA… **(c)** IBBM-1 is **not** CBAA
either: CBAA's phase-1 bundle construction and its task-selection rule are absent… Any *official* label
other than 'IBBM-1' is a **supervisor approval (D1)**, not a design choice". The delivered module *does*
implement phase-1 bundle construction (`build_bundle` `:423-459`), so it sits closer to CBAA than to
IBBM-1-as-designed — but the code, the YAML (`algorithm_variant`, `exp13_cbba.yaml:50`) and the
implementation report all adopt "CBAA (L = 1)" on the strength of the *repository* decision record
(`:31-51`, `:61-68`), while **D1 was never granted** and `docs/experiments/exp13/` contains no revision
of the design after the record. So the two written authorities disagree, the implementation silently
picked one, and the label in the deliverable is still an open owner decision (AUD-12).

The only identity-adjacent technical correction this audit adds is to the *tie* claim (AUD-17, §5):
`_prefer`'s docstring (`:587-596`) asserts the rule is "order-independent"; that is true **of the
pairwise comparison** and is confirmed by 247,510 states with zero divergence, but it is **not** true of
the *winner of an episode*, which is first-mover determined (P12). REV04 `:242` and REQ-107 (`:274`)
promise more than the carrier can deliver: "**Every belief message carries all bids the sender has
seen**, so each side can recompute the same rule from the same inputs". `ConsensusMessage` (`:113-132`)
carries `winning_bids`/`winning_agents` **vectors** — one bid per task — not the set of all bids seen, so
the merged-input recomputation REQ-107 relies on is not possible, and adopting a peer's vector on first
sight (`:566-570`, "not mine_winner ⇒ take") is exactly what makes the first publisher's winner stick.

Three further REV04 clauses are unimplemented (AUD-18, AUD-19):

| REV04 requirement | Implementation | Evidence |
|---|---|---|
| **REQ-110** (`:277`, schema row `:340`): "A belief is valid only for the `(target_id, ownership_epoch)` pair it names… a belief naming a different epoch is `STALE`"; `ownership_epoch uint64 — must equal sender's current epoch" | **Absent.** `ConsensusMessage` has no `ownership_epoch` field; `STALE` (`:525-526`) tests `round_number < last_seen_round` only — an *ordering* rule with no epoch semantics at all | EXECUTED/STATIC — this is the mechanism behind the restart/re-episode hazard (AUD-11) |
| **AGREEMENT_REACHED** (`:358`): requires (iii) local recomputation of REQ-107 on the merged bid set yields that winner **and (iv) `round_number ≥ 2`**, with evidence recorded as `(own belief, peer belief, round)` | **Absent.** `run_round` sets `AGREED` on `check_agreement() and any assigned` (`:736-739`); no round floor, no recompute, no recorded evidence tuple | STATIC |
| **REQ-207 admission on the receive lane** (`:717` register item 1: the node's `:498-513` "never calls `admit_message`") | Correctly identified by REV04 and classified **class 2 for integration**; the module implements its own admission, but the *node* still has the gap (§1.2 stage 6, AUD-06) | DOC — agreement with REV04 |

**Traceability:** the delivered test file contains **no `REQ-nnn`, `T-nn` or `INV-nn` identifier**
(grep: zero matches), so the REV04 test index T-01…T-27 cannot be traced to code from the tests
themselves. The substance of T-04 *is* covered by three real tests —
`test_outcome_is_independent_of_arrival_order` (`:354`),
`test_equal_bids_tie_break_to_smaller_agent_id_for_both_participants` (`:534`) and
`test_tie_break_is_repeated_across_reversed_participant_order` (`:547`) — but all three vary
**message arrival / participant-list order inside a lockstep run**, which is precisely the dimension
P12 shows to be insufficient: varying *acting order* gives 2/3/4 distinct winners at N = 2/3/4. AUD-20.

**One documentation risk found while re-checking the suite:** `test_convergence_is_repeatable`
(`:559-567`) is `@pytest.mark.parametrize("seed", range(12))` but uses **no RNG** — `seed % 3` selects
among three fixed bid deltas (0.60/0.61/0.62 vs 0.59). Calling it "12-seed repeatability" overstates it;
the benchmark *harness* genuinely contains no randomness. AUD-21.
**Unresolved decisions remain D1 (label), A1 (Team Execution Guide naming) and A2 (N > 2 vs better choice
between 2 parties)** — the last two open in `CBBA-VARIANT-DECISION-RECORD.md:157-158`.

---

## 5. Metrics and benchmark validity

Re-run for this audit (exit 0, 50 rows, JSON 48,489 B, SHA-256[:8] `1fcd4f94`, written **outside** the
repository to `C:\Users\PC\edgesync_es13_runs\exp13_audit_readonly_20261020\`).

| Check | Finding | Class |
|---|---|---|
| Same accounting boundary in both arms? | **No.** `frames_sent` is `1 + len(received)` in the baseline arm (`:206`) versus `sum(beliefs_published)` in the exp13 arm (`:283`); `ingestion_attempts` is `len(received)*(1+duplicate_factor)` (`:210`) versus `sum(current_round*(N-1))` (`:287`). The JSON's own `metric_definitions` says `frames_sent` = "application frames each arm hands to its transport" and `ingestion_attempts` names a different formula per arm. **Neither is an on-wire DDS packet count**, and the two arms are not counting the same event. | DOC/STATIC `FAIL` (comparability) |
| Same *decision* rule? | **No.** The baseline arm hard-codes `res.decided = True` (`:226`) and `res.converged_agents = 1` (`:227`, comment "the owner alone decides in the baseline"); the exp13 arm computes both (`:298-299`). Baseline `decided` is therefore an assumption, not a measurement. Fresh run: baseline 23 True / 2 False, exp13 15 True / 10 False. | DOC `FAIL` |
| Comparable participants/eligibility, or controlled difference? | Candidate sets differ **by design and are disclosed**: the baseline pool excludes the requester (`:189`, `:201`), the exp13 arm lets the owner bid for itself ("self-win = retain", `:314`). This is the §7.6 caveat of the implementation report and it is correctly stated. | `PASS` (disclosed) |
| Can the episode-level gate hide a unilateral `AGREED`? | In the harness, **no** — it withholds the commit (`:299`, and 5 rows prove it fires). In production there is no such gate at all (§1.2 stage 4). The harness result must never be cited as a production safety property. | `PASS` for harness / `FAIL` for portability |
| Rounds separated from wall clock? | Yes. `rounds` (`:286`) is the primary measure; `convergence_time_s_max` (`:294-296`) is explicitly annotated "host clock resolution ~15.6 ms … round counts are the primary convergence measure", and `milliseconds` is declared `PREDICTED_NOT_MEASURED`. 15.625 ms resolution is respected. | `PASS` |
| Simulated frame counts described as on-wire packets? | No such claim in the JSON, `measurement_scope` ("unit simulation; one process; no ROS graph; no network; no Jetson") or the report. | `PASS` |
| Determinism probe scope | The shipped probe permutes **delivery** under lockstep and reports `exp13_distinct_winners = 1` for 2/6/24 permutations at equal utilities (verified in the fresh JSON), while `baseline_distinct_winners = 1/2/3`. It does **not** permute *acting order*. Under acting-order permutation the exp13 winner set becomes 2/3/4 distinct winners for N=2/3/4 (P12). **The exp13 determinism advantage is partly a lockstep artefact.** | `FAIL` (qualification) |
| Baseline fidelity at the precision boundary | Production ranks by raw `float(msg.utility)` (`:538`) where `HandoffResponse.utility` is `float32`; the harness ranks by `round(u, 4)` (`:193`, `:215`), the same quantum the CBAA arm uses. Measured: float32 ulp at 0.72 is **5.96e-8**, i.e. **1677× finer** than the 1e-4 tie quantum; pairs exist (e.g. 0.500001 vs 0.499999) that are equal at 4 dp but distinct as float32, so production breaks such ties by value while the harness breaks them by stable arrival order. The harness therefore models a **different** baseline tie behaviour from the one that runs. | `FAIL` |
| Does any performance-improvement claim follow? | **No.** On the retained and freshly re-run 50 rows, exp13 is unfavourable on both axes (more rounds, more belief frames, `decided` 15 vs 23). The implementation report states this honestly; no optimisation was made to improve the benchmark. | `PASS` (report honesty) / `FAIL` (any improvement claim) |
| Reproducibility | Retained artefacts re-verified by digest: JSON 48,473 B `f1434d67`, `console.txt` 9,210 B `8f1750ff`, `a_console.txt` 9,203 B `3cb8dcb1`, `b_console.txt` 9,203 B `e9e6f052`. All match the report. **One documentation contradiction found:** report `:252` still says "two-run `diff` = one line" while `:203` gives the measured "14 of 81 lines differ (13 `conv_s` + the output-path line)". The measured value is 14. Also `console.txt` is a shell redirect, not script output — the script writes only the JSON (`:503-505`). | DOC `FAIL` (AUD-14) |

---

## 6. Verification: what ran, what could not

All commands from `…\Implementation Phase\edgesync_ws\src\edgesync_coordination`. Exit codes captured
directly (`$?`), never through a pipe.

| # | Command | Exit | Result | Class |
|---|---|---|---|---|
| V1 | `python -m pytest tests/test_cbba_negotiator.py -q` | **0** | **70 passed** in 0.42 s | EXECUTED `PASS` |
| V2 | `python -m pytest tests/test_fault_injection.py -q` | **0** | **5 passed** in 0.25 s (includes the epoch-validator assertions at `:281-286`) | EXECUTED `PASS` |
| V3 | `python -m pytest tests/test_phase5_chaos_races.py -q` | **0** | **11 passed** in 0.21 s (includes `:193-197`) | EXECUTED `PASS` |
| V4 | `python -m pytest tests -q` | **2** | **0 tests ran.** `Interrupted: 3 errors during collection` — `tests/test_auto_trigger.py`, `tests/test_decision_and_receiver_services.py`, `tests/test_response_collector.py`, each `ModuleNotFoundError: No module named 'rclpy'` | **BLOCKED — reported as not run** |
| V5 | `python -m pytest tests -q --ignore=tests/test_auto_trigger.py --ignore=tests/test_decision_and_receiver_services.py --ignore=tests/test_response_collector.py` | **0** | **101 passed** in 0.41 s (= 70 CBAA + 31 other) | EXECUTED `PASS` |
| V6 | `PYTHONPATH=. python -m edgesync_coordination.scripts.run_exp13_unit_benchmark --out C:/Users/PC/edgesync_es13_runs/exp13_audit_readonly_20261020/exp13_audit` | **0** | 50 rows written; SHA-256[:8] `1fcd4f94`; `dual_owner_violations` total **0**; `false_convergence` total **0** | EXECUTED `PASS` |
| V7 | `python edgesync_coordination/scripts/run_exp13_unit_benchmark.py` (no PYTHONPATH) / same with `PYTHONPATH=.` but no `--out` | **1** / **2** | invocation errors of my own, recorded for transparency; not code defects | n/a |

**Tests added in this audit: none.** Read-only mode forbids creating or editing test files, so
investigation §2 was executed as ephemeral in-memory probes against the real classes (§9). This means
the dual-ownership witness of §3.3 and the P10–P12 results are **not** yet protected by any assertion
in the repository, and will not be detected by CI if the code changes.

**Explicitly not claimed:** no ROS 2 test, no `colcon` build, no launch, no integration test, no
node-level execution, no `pytest -m` marker run, no Jetson measurement. `rclpy`, `launch`,
`launch_ros`, `ament_index_python`, `colcon` are absent on this host. The 3 blocked files above are the
only tests covering the node's response/collector/auto-trigger behaviour, so **the node-side commit path
is unverified by execution on this host** — hence AUD-09 `BLOCKED` and the FSM-level substitute in §3.3.

---

## 7. Findings register

| ID | Finding | Class |
|---|---|---|
| AUD-01 | No production consumer of the Exp-13 negotiator; unilateral-`AGREED` hazard is latent, not live. Enforced by two passing tests. | PASS |
| AUD-02 | `check_agreement()` is a local predicate with no acknowledgement leg; a local `AGREED` carries **no** evidence that any peer agreed. `metrics()` exposes nothing that would distinguish safe from unsafe agreement. | FAIL |
| AUD-03 | Under lossless-but-asynchronous delivery, consensus **content** never diverges: 247,510 reachable states, 4 bid profiles, zero witnesses of two agents (or an agent and a peer) holding different `winning_agents` maps. | PASS |
| AUD-04 | Reachable interleavings exist where one `AGREED` agent coexists with a declared participant that terminated non-converged (40/134 and 35/138), and in all of them `convergence_time_s` is numeric. `AGREED` ⇒ "all declared participants converged" is **false**. | FAIL |
| AUD-05 | With equal utilities the episode winner is first-mover determined (2/3/4 distinct winners at N=2/3/4), qualifying both `_prefer`'s "order-independent" docstring and the harness determinism probe, which permutes delivery but not acting order. | FAIL (scope of claim) |
| AUD-06 | `NegotiationManager.admit_message` guards only `HandoffRequest` (`:330`). Response, AcqStatus, CONFIRM, CONFIRM_ACK and OWNER_CLAIM handlers perform **no** admission/epoch/transaction validation; `_on_handoff_response:506-512` commits the FSM and arms the watchdog on an unvalidated ACCEPT when no window is open. | FAIL |
| AUD-07 | `_on_handoff_confirm:563-578` cannot detect refusal (measured F1) and publishes `HandoffConfirmAck` unconditionally. | FAIL |
| AUD-08 | **Zero dual-ownership invariant violated (measured, §3.3):** one lost BEST_EFFORT `AcqStatus` (its only producer is `:557-561`; `SenderState.ACQUISITION` has no producer at all) leads to two nodes both `role=OWNER` at epoch 1, and `e+2` recovery does **not** reconcile them. Pre-existing baseline defect; not caused by Exp-13. | FAIL (critical) |
| AUD-09 | Node-level behaviour not executable here (`rclpy`); AUD-06/07/08 proven at FSM level + static node lines. `SenderState.ACQUISITION` unreachable, and no receiver-side demotion path exists for an `e+2` reclaim. | BLOCKED / NOT VERIFIED |
| AUD-10 | `EpochManager.validate_incoming_epoch`/`is_stale` are documented as guarantee #2, tested by two existing suites, and called from **zero** production sites. Measured additionally: `>= current` cannot express the required "ACK epoch == e+1", so wiring them in as-is would not fix AUD-08. | FAIL |
| AUD-11 | Belief carries no `ownership_epoch`, nonce or participant-set field (`:113-132`); `reset()` clears the watermark (`:317`) ⇒ restart/re-episode fencing by the existing epoch interlock is impossible without a schema change. | FAIL (design gap) |
| AUD-12 | Algorithm identity is internally consistent and not inflated (`ALGORITHM_VARIANT` `:46`, L = 1 enforced at `:208-212`, full-CBBA semantics absent by decision). The *official* designation is still open: the repository record says CBAA, REV04 §3.1 says IBBM-1 and forbids any official label without supervisor approval **D1**, which was never granted. | PASS (internal) / **FAIL-BLOCKED (designation, D1)** |
| AUD-13 | Benchmark arms do not share an accounting boundary (`frames_sent`, `ingestion_attempts` formulas differ; baseline `decided`/`converged_agents` hard-coded at `:226-227`); harness models the baseline at 4-dp rounding while production ranks raw float32 (measured 1677× quantum mismatch). Counts are "application frames", never DDS packets. | FAIL (comparability) |
| AUD-14 | Implementation-report internal contradiction at `:252` ("diff = one line") vs the measured `:203` (14 of 81); `console.txt` is a shell redirect, not script output. Retained digests verified. | FAIL (documentation) |
| AUD-15 | No performance-improvement claim is supportable; the recorded result is unfavourable to Exp-13 on rounds, frames and `decided`, and is reported that way. | PASS (honesty) / FAIL (any improvement claim) |
| AUD-16 | **REV04 REQ-109 is honoured**: the module imports only stdlib (`:33-41`) and its sole `OwnershipFSM` mention is the disclaimer at `:26-27`. It never writes an epoch, never calls an FSM, never publishes: the allocation is advisory, exactly as contracted (REV04 `:276`). **But the module's own docstring at `:15-16` states the unsound half of the argument** — "never mutates an ownership epoch, *so a wrong consensus cannot create two owners*" — which §3.3 measured to be false of the ACK exchange it defers to. | PASS (lane) / FAIL (docstring safety claim) |
| AUD-17 | **REQ-107 tie determinism is not achieved as written.** The contract requires every belief to carry "all bids the sender has seen" so each side recomputes the rule from the same inputs; the carrier holds one winning bid per task (`:113-132`), so a first-seen peer vector is adopted wholesale (`:566-570`) and the episode winner becomes first-mover determined (measured: 2/3/4 distinct winners at N = 2/3/4 under acting-order permutation). Content agreement still holds (AUD-03). | **FAIL** |
| AUD-18 | **REQ-110 epoch binding unimplemented.** A belief is contracted to be valid only for its `(target_id, ownership_epoch)` pair, with a mismatching epoch classified `STALE`; `ConsensusMessage` has no epoch field and `STALE` (`:525-526`) compares round numbers only. This is what makes stale-belief re-admission after `reset()` (`:317`) possible and un-fenceable. | **FAIL** |
| AUD-19 | **AGREEMENT_REACHED clauses (iii)+(iv) unimplemented** (`:358`): `AGREED` needs only `check_agreement()` plus a non-empty assignment (`:736-739`) — no local recompute of the tie rule on the merged bid set, and no `round_number ≥ 2` floor, so agreement can be declared in the round a belief is first adopted. | **FAIL** |
| AUD-20 | **No REQ/T/INV traceability in the delivered tests** (zero identifier matches); T-04's substance exists but only over arrival/participant order (`:354`, `:534`, `:547`), which does not cover the acting-order dimension AUD-17 fails on. | **FAIL** |
| AUD-21 | `test_convergence_is_repeatable` (`:559`) is seed-*parametrised* but not random — 12 seeds select 3 fixed bid deltas; no RNG is used anywhere in the suite or harness. Describing it as a 12-seed repeatability test would overstate the evidence. | DOC **FAIL** |
| AUD-22 | The Exp-13 design authorities live **outside the repository** (`C:\Users\PC\edgesync_es13_proposal\`, unversioned); a teammate working from the repo alone sees only `CBBA-VARIANT-DECISION-RECORD.md` (dated 2026-10-19, "DECISION REQUIRED FROM OWNER — implementation intentionally NOT started") and cannot recover the REQ/INV/T contract the code was written against. | **FAIL** (governance) |

| AUD-23 | **The deliverable's own safety argument is unsound.** `cbba_negotiator.py:15-16` asserts the allocation is advisory and "never mutates an ownership epoch, **so a wrong consensus cannot create two owners**" — the same premise as `CBBA-VARIANT-DECISION-RECORD.md:85-89`. §3.3 measured that the ACK exchange it defers to *does* create two owners without any consensus involvement. The lane isolation (AUD-16) is fine; the conclusion drawn from it is not, and it is the sentence a future integrator will rely on. | **FAIL** |
| AUD-24 | The REQ-110 epoch binding is **already mandated by the approved contract**, so implementing it is an authorized change rather than new design — it is the smallest change that would fence restarts and re-episodes (AUD-11/AUD-18) and it needs no protocol invention. | DOC (basis for M3a) |

---

## 8. Minimum safe changes before ROS integration (none applied — read-only)

Ordered smallest-first. M1–M3a are isolated changes whose correctness **follows from the existing
approved contract** (SAD System Invariant 1, `epoch_manager.py:1-6` guarantee #2, the ACK-at-`e+1`
transition already written in `ownership_fsm.py:106-114`/`:199-207`, and REV04 REQ-110/`:358` which the
implementation simply never delivered), so they need only execution authorisation, not new design.
M4–M6 are genuine design decisions and must **not** be resolved silently.

| # | Change | Why it follows from the existing contract | Risk if not made |
|---|---|---|---|
| **M1** | Make `_on_handoff_confirm` ACK **only** on a real transition: give `on_confirm_received()` a refusal signal (return `Optional[int]`, or `(ok, epoch)`) and suppress the ACK when the sender is not in `{IDENTITY_VERIFICATION, ACQUISITION}`. | `ownership_fsm.py:106-114` already refuses by returning the *unchanged* epoch; the node just cannot see it. Same file, same invariant. | AUD-07/08 — an unaccepted CONFIRM is ACKed, creating two owners. |
| **M2** | In `_on_handoff_confirm_ack`, require `msg.new_epoch == pre_transaction_epoch + 1` **exactly** before `on_confirm_ack_received`, and drop anything else. | The contract already defines the transition as "ACK with epoch e+1" (`ownership_fsm.py:1-7`, `epoch_manager.py:1-6`). | AUD-08 — any epoch on the wire becomes the ownership epoch (measured `999`, measured stale `1`). |
| **M3** | Route `_on_handoff_response` (and CONFIRM/ACK/OWNER_CLAIM) through the already-present `NegotiationManager.admit_message` used at `:330`; remove the out-of-window auto-commit at `:506-512`. | Uses an existing, already-declared-invariant mechanism on the paths that currently skip it. | AUD-06 — duplicate/stale/foreign ACCEPTs can commit and arm the watchdog. |
| **M3a** | In `cbba_negotiator.py` only: add `ownership_epoch` to `ConsensusMessage` (`:113-132`) and classify a belief whose epoch differs from the local current epoch as `STALE` in `_accept` (currently a pure round-number test at `:526`); add the two missing `AGREEMENT_REACHED` clauses — (iii) recompute the merged bid set before declaring agreement and (iv) `round_number >= 2` — per REV04 `:358`. | **Already in the approved design**: REV04 REQ-110 (`:277`, schema `:340`) and AGREEMENT_REACHED (iii)(iv) (`:358`) are contract text the implementation never delivered, and the epoch semantic is the one `epoch_manager.py:1-6` guarantee #2 already states. No new protocol, no new topic, no shared file touched. | AUD-11/AUD-18 — a belief re-admitted after `reset()` or in a new episode is indistinguishable from a live one; and `AGREED` remains reachable in round 1, which is the clause (iv) the design already forbids. |
| **M4** | Add a receiver-side demotion path for an `e+2` reclaim (or prove one exists). | **Open design question.** Options: (a) let `ReceiverState.TRACKING` accept `OWNER_CLAIM` with epoch > current and demote; (b) require the OWNER to re-verify Re-ID after any epoch jump. Consequence of (a): a spoofed/late high-epoch claim can steal ownership; of (b): tracking outage during recovery. Not chosen here. | AUD-08/09 — divergence survives watchdog recovery. |
| **M5** | Decide what a production "commit" means for Exp-13 before any wiring: (i) keep the allocation advisory only (decision record §2 as written, needs the §3.3 fix to actually hold); (ii) promote harness `:299` into the node as an all-declared-participants gate — which requires a belief carrier `.msg` **and** fails under partition by AUD-02; or (iii) accept that Tier U cannot evidence global agreement and gate commit on a lease/quorum — **not** authorised without approval. | The architecture has no N-party consensus topic today (`CBBA-VARIANT-DECISION-RECORD.md:131-138`); inventing one is outside the contract. | Exp-13 output silently treated as a global decision. |
| **M6** | Answer **A1** (Team Execution Guide: CBBA / CBAA / L=1), **A2** (N > 2 robots vs better choice between the existing 2) and **D1** (REV04 `:211-219` reserves any official label other than IBBM-1 to supervisor approval; §4) before the variant claim is finalised. Then decide the **remaining** `ConsensusMessage` fields only: the epoch field is **not** part of this decision — it is already contract-mandated and belongs to M3a — but a participant-set/nonce field is not named anywhere in REV04 or the decision record, so adding one is a schema bump and an owner call (AUD-11). Alternatively, formally exclude restart/epoch fencing beyond M3a from scope. | A1/A2 are listed as owner-only in `CBBA-VARIANT-DECISION-RECORD.md:157-158`, D1 in REV04 `:219`; the extra fields have no source that authorises them. | Identity/scope claims in deliverables remain unratified; and either beliefs stay re-admittable across `reset()` beyond what M3a fences, or an unapproved contract change is made to close it. |

**Do not** tune `convergence_threshold`, `max_rounds`, `bid_epsilon` or the tie rule to improve the
benchmark. P12 shows the tie's *outcome* depends on who publishes first, which no parameter change
fixes; and the CBAA-vs-baseline tie asymmetry at the precision boundary (§5, "Baseline fidelity at the
precision boundary") is a carrier-precision fact, not an algorithm fact.

---

## 9. Reproduction

Executed test commands: V1–V7 in §6 (single-line, exit codes captured directly).
FSM/negotiator probes were run as ephemeral stdin scripts, so no probe file was created inside the
repository:

```
cd "…\Implementation Phase\edgesync_ws\src\edgesync_coordination"
PYTHONPATH=.:/tmp python - <<'PY'          # F1/F2/F4/F5 PATH-1/2/3 witnesses  (§3.3)
PYTHONPATH=.:/tmp python - <<'PY'          # P10/P10-U/P11  memoised async DFS over live negotiators
PYTHONPATH=.:/tmp python - <<'PY'          # P12  acting-order permutation sweep, AGREED-only
python - <<'PY'                            # P11b float32 vs 4-dp quantum (§5)
```

Probe model invariants (stated so results can be challenged): objects are the real
`CBBANegotiator`/`OwnershipFSM`, copied with `deepcopy` per branch; a state key is
(status, round, `winning_bids`, `winning_agents`, `bundle`, per-sender watermark, `_stable_rounds`,
mailbox contents); delivery is lossless; agents may ingest one/all/none of their mailbox. Coverage
bounds: N ∈ {2,3}, 1 task, 1 target, `max_rounds` ∈ {4,6}, `convergence_threshold` ∈ {2}, depth cap
8–10 actions, 247,510 states total. The search is **exhaustive within those bounds only** — it is not
a proof, and it models no message loss, no partition and no clock skew.

Probe outputs needed for §2/§3 that are **not** retained as files (read-only mode): re-run the commands
in §9; each prints its counts and the witness tuples verbatim.

---

## 10. Recommendation

# **BLOCKED PENDING DESIGN APPROVAL**

Not "proceed after listed fixes", for three reasons that are separable and each disqualifying:

1. **The safety premise of Exp-13 integration is false as the code stands.** The design keeps the
   consensus result advisory and relies on "the ACK exchange is the only ownership-mutating event"
   (`CBBA-VARIANT-DECISION-RECORD.md:85-89`). §3.3 *executed* a two-owner state produced by that very
   ACK exchange, reachable by losing one BEST_EFFORT `AcqStatus`, and showed watchdog recovery does not
   reconcile it. M1+M2 close it and their correctness follows from the existing contract, so they
   should be authorised **immediately as a baseline fix** — but they touch `coordination_node.py` and
   `ownership_fsm.py`, files Exp-13 is forbidden to modify, so they are an owner action, not an Exp-13
   action. Exp-13 must not inherit an invariant-violating commit path.
2. **The architecture cannot establish global agreement under partition, and that is a stated
   limitation, not a solvable detail here.** `check_agreement()` is local by construction (AUD-02),
   unilateral `AGREED` is pervasive in reachable states (AUD-04), and the all-converged gate that saved
   the benchmark result lives in the harness only (`run_exp13_unit_benchmark.py:299`). Closing this
   requires a failure detector, quorum, lease or transaction protocol — **all explicitly outside my
   authority** ("Do not invent a failure detector, quorum rule, lease, or distributed transaction
   protocol without project approval"). M5 states the alternatives and their consequences and does not
   choose.
3. **Two decisions that gate any integration are still open** — A1 (the guide's algorithm naming) and
   A2 (N > 2 vs 2-party selection) — and D1 (whether the deliverable may be labelled CBAA at all) is
   unresolved against REV04's own IBBM-1 designation (§4). The belief-carrier `.msg` still does not
   exist. The REV04 artefacts themselves **are** on disk, but outside the repository and unversioned
   (AUD-22), so the approved contract this audit compared against is not part of the deliverable tree.

Minimum conditions to move to `PROCEED ONLY AFTER LISTED FIXES`: M1 + M2 + M3 executed on the baseline
and covered by tests that fail without them; M3a executed inside `cbba_negotiator.py` and likewise
test-covered; M4 decided; a written ruling on M5 (recommended: allocation stays advisory, commit stays
on the corrected ACK path, and Exp-13 is measured at Tier I as a selector change only); A1/A2/D1 answered.
The only change this audit proposes to `cbba_negotiator.py` is M3a, and it is contract text (REV04
REQ-110, AGREEMENT_REACHED iii/iv) that was approved but never delivered — not new design. Its consensus
**content** is otherwise the strongest part of the deliverable (AUD-03) and must not be rewritten; the
`:15-16` safety sentence (AUD-23) must be corrected in the same change, since M1/M2 are what actually
close the two-owner hazard.
