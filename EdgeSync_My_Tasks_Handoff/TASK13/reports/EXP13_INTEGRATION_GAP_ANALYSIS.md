# EXP13 — Integration gap analysis (read-only inventory + unapplied patch proposals)

**Experiment:** ES-EXP13 · **Date:** 2026-10-20 (Gate 1c / §6 of the brief)
**Method:** live-tree inspection only. Every claim below is either a `grep`/`ls` result with its exit code or a line citation. **No shared file was read for writing and none was modified.** `rclpy` is absent on this host (`python -c "import rclpy"` → exit **1**, `ModuleNotFoundError`), so **nothing in this document is a runtime observation** — the ROS 2 layer is statically inspected, and where runtime behaviour decides a design option that is stated as unknown rather than guessed.

**Overall status: `BLOCKED — SHARED-FILE APPROVAL REQUIRED`.** G5 of the readiness gate needs D5, D7, D10, D11 (+ D3, D4) and the decision sheet states that without it "**no shared file may change**" (`ES-EXP13-SUPERVISOR-DECISION-SHEET-REV04.md:176`). None of those rows is answered.

---

## 1. The nine inventory items the brief requires

### 1.1 Selector declaration and default

| Fact | Evidence |
|---|---|
| `coordination_node.py` declares **24** parameters | `grep -c declare_parameter` on the file → 24 (`:39-66`) |
| None of them is `negotiation_protocol` | workspace grep for `negotiation_protocol`: 11 hits, **all** inside Exp-13's own files (`exp13_cbba.yaml:23,49`; `cbba_negotiator.py:61`; `run_exp13_unit_benchmark.py:848,880,979,1028`; `test_cbba_negotiator.py:766,767,814,851,855,923`). Zero in any node |
| `allow_undeclared_parameters` is not set anywhere in the node | grep → exit 1, no matches |
| Runtime fate of an undeclared key supplied by a params file | **UNKNOWN ON THIS HOST.** Either outcome — silent drop or a configure-time exception — is a `rclpy` behaviour that cannot be executed here. This is not a detail: if it raises, wiring `exp13_cbba.yaml` into the launch file without first declaring the keys **stops the stack from launching**, so item 1.4 must land before item 1.1's config is referenced |
| A tripwire already asserts the absence | `test_cbba_negotiator.py:855` asserts `"declare_parameter('negotiation_protocol'" not in node_src`, and `:814` asserts it is absent from the launch source |

The selector contract (values, default, fallback) is **D5** and is unanswered. There is no existing selector framework to reuse: today the negotiation strategy is hard-wired into the node's request/response methods (`:441-482` broadcast, `:498-513` collect, `:515-555` select).

### 1.2 Supported selector values and fallback

No production dispatch exists, so there is nothing to add a value to. Proposed minimal contract **if D5 approves it** — and only if:

- `cnp` (default, current behaviour, zero behaviour change for every other experiment) →
- `cbba` (constructs the Exp-13 negotiator; value string already fixed by `exp13_cbba.yaml:49` and `PROTOCOL_ID` at `cbba_negotiator.py:61`) →
- unknown value → **refuse to start the negotiation lane and log an error**, never silently fall back. A silent fallback would mean an experiment believed it was running Exp-13 while the baseline ran; the alternative (fail closed) is what `select_experiment_config` already does for Re-ID (`reid_launch_defaults.py:242-243`: "An override that does not exist is not dropped quietly… raises").

Which of "error" or "baseline" is correct is exactly the unanswered half of D5, so it is not decided here. Note also that **B-1/D1 is upstream of this**: the value string `cbba` names, per REV04 `:211-219`, an algorithm the module is explicitly said *not* to be.

### 1.3 Negotiator construction and lifecycle

| Item | State |
|---|---|
| Production instantiation | **none.** `from edgesync_coordination.negotiation import CBBANegotiator` appears only at `scripts/run_exp13_unit_benchmark.py:89`. The package's `__init__.py` exports it; no node imports it |
| Driver shape | The module is deliberately driver-agnostic: `register_tasks` → `build_bundle`/`run_round(incoming)` → `check_agreement` → `reset`. It has **no** timer, publisher, subscriber or clock of its own (REQ-109: never publishes, never calls an FSM — verified at `:26,33` where the only FSM references are docstring disclaimers) |
| What an adapter must own | episode id minting (the module refuses a reused published id, `cbba_negotiator.py:358`), the round timer, the belief publisher/subscription, the `deadline_s` budget (`:76` of the YAML; **sender-local only**), `ownership_epoch` supply at construction (`:248`), and `reset()` between episodes |
| Lifecycle ordering trap (measured) | `_episode_open` (`:292,353,400,818`) distinguishes "in progress" from "reset and idle". An adapter that calls `register_tasks` after a `reset()` with the **same** episode id is refused by design; the adapter must mint a fresh id. This is the fix that replaced a dead-end, and an integrator who mints ids from a ROS clock must not mint a *reused* id |
| Interaction with the FSM | none permitted (REQ-109). Allocation is advisory: the outcome must be handed to the existing commit path, which is where M1–M4 live. Integrating Exp-13 does **not** fix, and does not inherit a fix for, the dual-owner defect |

### 1.4 Configuration loading — the existing convention, and why Exp-13's YAML is inert

`exp13_cbba.yaml` is shaped correctly for ROS 2 (`/**: ros__parameters:` at `:44-45`) and its keys were verified line by line (`:48` `exp_id`, `:49` `negotiation_protocol`, `:60` `max_bundle_size`, `:63` `max_rounds`, `:65` `convergence_threshold`, `:68` `bid_epsilon`, `:69` `utility_precision`, `:76` `deadline_s`). It is referenced by **no launch file**: grep for `exp13_cbba` in `launch/` → **0 matches**.

The project already has one experiment-overlay mechanism, and Exp-13 does **not** use it:

```
robot_bringup.launch.py:38   experiment_config, experiment_origin = select_experiment_config(algo_params)
                       :39   reid = resolve_reid_config(algo_params, experiment_config)
                       :44   if experiment_config and not reid['model_is_valid']: raise ValueError(...)
reid_launch_defaults.py:229-254   select_experiment_config(shared_path, override=None)
                            :246   EDGESYNC_EXPERIMENT_CONFIG env var is the only supported selector
```

Two facts about that convention decide the patch shape:

1. The overlay is selected by an **environment variable**, deliberately — the launch file resolves defaults before the CLI is parsed, so a launch argument would either be invisible or compete for precedence (`robot_bringup.launch.py:33-37`, `reid_launch_defaults.py:233-241`). Any Exp-13 selector must respect the same precedence rule instead of inventing a second one.
2. `experiment_config` reaches **only the Re-ID resolution path**. The coordination node's parameter list is fixed and never sees it:

```python
    node_coordination = Node(                                  # :142-153
        package='edgesync_coordination', executable='coordination_node',
        name='coordination_node', namespace=node_namespace, remappings=global_remappings,
        parameters=[
            coordination_params, timing_params, algo_params, hw_params,   # :149  no overlay
            {'robot_id': robot_id, 'peer_id': peer_id, 'is_initial_owner': is_owner}
        ], output='screen')
```

So `exp13_cbba.yaml` is inert for **two independent reasons** — the node declares no selector keys (§1.1) *and* the launch never hands the overlay to that node (§1.4). Fixing one without the other changes nothing.

### 1.5 Message schema, epoch and episode fields

- `edgesync_msgs/msg/` contains **11** files: `AcqStatus`, `HandoffCancel`, `HandoffConfirm`, `HandoffConfirmAck`, `HandoffHeader`, `HandoffRequest`, `HandoffResponse`, `OwnershipEvent`, `StateHeartbeat`, `TTOEstimate`, `Target3DState` (re-counted this pass; an earlier Exp-13 report said 9 and has been corrected). All carry mtime 2026-10-14 19:30; none was touched.
- **The epoch field already exists on the wire.** `HandoffHeader.msg:8 uint64 ownership_epoch # Monotonic ownership counter (Anti-divergence FR-54)`; `HandoffConfirmAck.msg:4 uint64 new_epoch # Incremented epoch (e+1)`; `OwnershipEvent` embeds `HandoffHeader`. A `grep -n epoch` over all 11 files returns exactly those two declarations. **This materially shrinks the believed gap:** REQ-110's epoch binding does not need a new shared field, and a belief carrier can *embed* `HandoffHeader` as every other handoff message does.
- What genuinely has no carrier is the **belief payload**: `winning_bids`, `winning_agents`, `bundle`, `episode_id`, `round_number`, `schema_version`. REV04 §4.3 (`:330-341`) already specifies the intended contract — `header` (reused `HandoffHeader`) + `episode_id` + `target_id` + `round_number` + `ownership_epoch` + `bids float64[2]` + `bid_present uint8[2]` + `winner_id` + `tie_rule_version`, with `message_type` = a new constant `MSG_TYPE_ALLOCATION_BELIEF = 9` (existing constants occupy 1…8, `HandoffHeader.msg:13-20`). That specification is **design only** (`:330`: "no `.msg` file was created or modified in this task") and its approval is **D10**.
- One internal inconsistency to put in front of the msgs owner rather than resolve silently: `HandoffHeader` already carries `transaction_id`, and REV04 separately requires `episode_id`. Either `episode_id` duplicates `transaction_id` (then the field is redundant) or it is a distinct identity (then the relationship must be defined, because the module's replay fence keys on episode identity at `cbba_negotiator.py:358` and its staleness fence keys on epoch at `:593`).
- Rebuild blast radius: **8 packages** declare `edgesync_msgs` in `package.xml` (acquisition, bringup, coordination, estimation, identity, navigation, observability, perception). A new `.msg` or a new header constant forces a coordinated rebuild and redeploy of all eight — which is why D10 also asks "who rebuilds and deploys".

### 1.6 Launch-file parameter wiring

No `exp13_cbba` reference exists in `launch/` (0 matches). `config/` holds `exp7_transreid.yaml` and `exp13_cbba.yaml` side by side; only the first is reachable, and only through `EDGESYNC_EXPERIMENT_CONFIG`. Proposed wiring is in §2 P-3 — bringup owner.

### 1.7 Experiment result provenance

- The project's established convention is a per-experiment JSON artefact written by an explicit output flag; that is what `run_exp13_unit_benchmark.py` does, writing to `edgesync_bringup/results/experiment13/`.
- `exp_id` appears **nowhere** in production code outside Exp-13's own script: workspace grep for `exp_id` in `edgesync_coordination/edgesync_coordination` and `edgesync_observability` → **0 hits**.
- Therefore the brief's warning applies directly: **`exp_id` is a provenance label, not a runtime selector, and adding it to the node is not justified by any existing logging contract.** The prior report's R15 finding ("the premise is wrong") stands, and nothing was added.

### 1.8 Compatibility with the baseline and other experiments

- The baseline lane is untouched as long as the default selector keeps today's behaviour; the only structural risk is M3/M0, which would put a stateful transaction table on the path every experiment uses (see `EXP13_SAFETY_REMEDIATION_PROPOSAL.md` §0).
- **Cross-package producers/consumers measured:** `HandoffConfirm` is published by `edgesync_identity/nodes/identity_node.py:183` on the **global** topic `/edgesync_swarm/handoff_confirm`; `AcqStatus` is published by `edgesync_acquisition/nodes/target_acquisition_node.py:257` (`STATUS_ACQUIRED`) on `/{peer_id}/edgesync/acq_status` with **`best_effort_qos`** (`:110`) — which is the exact lane whose single-sample loss reproduces the dual-owner witness. So M1/M2 cannot be reasoned about inside one package: the confirm leg arrives from a different node with different topic semantics.
- 6 test modules depend on the FSM methods M1–M4 would change (`on_confirm_received` 9 refs, `on_confirm_ack_received` 5, `on_owner_claim_received` 4, `abort_to_recovery` 13). 4 of them are ROS-free and would run here; 2 (`test_auto_trigger.py`, `test_response_collector.py`) import `rclpy` and **cannot be verified on this host at all**.

### 1.9 Error handling when configuration is missing or invalid

- Re-ID path: missing/nonexistent override is **not** dropped quietly — it raises (`reid_launch_defaults.py:242-243`, surfaced as `ValueError` at `robot_bringup.launch.py:44-48`). That is the precedent to follow.
- Coordination path: no handling exists, because nothing reads Exp-13's configuration. `exp13_cbba.yaml:23` records this openly ("`coordination_node.py:39-66` and `negotiation_protocol` is not [declared]"), and `run_exp13_unit_benchmark.py:848` writes the same disclaimer into the result artefact (`"no selector exists in "` production).
- The module's own validation is real and executed: constructor requires ≥ 2 participants including self, rejects `max_bundle_size != 1` (`:251`), rejects non-int / bool / negative epochs (`:248`), and classifies inbound payloads into `STALE` / `DUPLICATE` / `POST_TERMINAL` / `FOREIGN_EPISODE` / `MALFORMED` (`:111`, `:572`, `:593`).

---

## 2. Patch proposals for the blocked shared changes — **written, NOT applied**

Order matters: P-1 before P-2 before P-3. Applying P-3 alone gives a launch that still silently ignores the file; applying P-1/P-2 without D1 gives a selector whose value names an unapproved algorithm.

### P-1 — declare the selector on the node (`nodes/coordination_node.py`, shared) — needs **D5**

```python
         # :39-66 parameter block, appended
+        self.declare_parameter('negotiation_protocol', 'cnp')   # 'cnp' = today's behaviour, unchanged
```

and, at the single point where the baseline strategy is chosen, dispatch on the value. **Not written further**, because the body depends on two unanswered questions: the unknown-value fallback (§1.2) and the official protocol name (D1). Default `'cnp'` is chosen so that no existing experiment changes behaviour by default.

### P-2 — instantiate and drive the negotiator (`edgesync_coordination`, shared) — needs **D11 + D7**

An adapter owning: episode-id minting, round timer, belief pub/sub, `deadline_s`, epoch supply, and `reset()`. It must **not** connect the allocation result to the FSM or epoch manager — REQ-109 keeps the lane advisory, and M1–M4 (unapproved) is what stands between a claim and ownership. Specified as a boundary with typed actions in REV04 §7.6/D11; that specification is the deliverable, not code.

### P-3 — wire the overlay to the coordination node (`launch/robot_bringup.launch.py`, shared) — needs **bringup owner + D5**

Reuse the existing selector instead of inventing one:

```python
-            coordination_params, timing_params, algo_params, hw_params,
+            coordination_params, timing_params, algo_params, hw_params,
+            *([experiment_config] if experiment_config else []),
```

Preconditions, all of which must hold before this line is worth applying: P-1 declares the keys (else the runtime behaviour of §1.1's unknown applies); `select_experiment_config` currently resolves against the **Re-ID** config directory and its docstring speaks only of Re-ID, so pointing `EDGESYNC_EXPERIMENT_CONFIG` at `exp13_cbba.yaml` today also drives Re-ID resolution — an overlay that carries both namespaces would need the resolver generalised first. **This is a real coupling, and it is why P-3 is listed as blocked rather than "small".**

### P-4 — belief carrier `.msg` (`edgesync_msgs`, shared) — needs **D10** (+ D1 for what it is called)

Adopt REV04 §4.3 verbatim as the starting point, with three changes the owner must rule on rather than inherit silently: embed `HandoffHeader` (which already supplies `ownership_epoch`, `transaction_id`, `target_id`, `sequence`, `stamp_utc_ns`) instead of re-declaring an epoch field; decide `episode_id` vs `transaction_id` (§1.5); and add `MSG_TYPE_ALLOCATION_BELIEF = 9` to `HandoffHeader.msg` (`:13-20` occupies 1…8). Then rebuild **8** consumer packages.

**If REQ-107 (B-2) is approved as mandatory**, the carrier must also hold the full bid set, and the Python-side model changes in the same step: `ConsensusMessage` today carries `winning_bids: Dict[str, float]` / `winning_agents: Dict[str, str]` (`:158-159`), i.e. a **winner vector**, while REV04 `:274` requires "**all bids the sender has seen**" and `:358` clause (iii) requires agreement to be *recomputed on the merged bid set*. `check_agreement` (`:690-735`) compares the peer's **winner vector** and therefore evaluates clauses (i), (ii), (iv via `MIN_AGREEMENT_ROUND` at `:836`) and (v), **not clause (iii)**. That is the precise, measured delta — and it is the one change that would move `BELIEF_SCHEMA_VERSION` from 2 to 3. It is not applied because REQ-107 is unanswered; per brief §5.2, "if REQ-107 is not approved, do not add the field merely because it appears in REV04."

---

## 3. Runtime verification — attempted, blocked

The brief's §6.3 commands (`colcon build`, `colcon test`, `colcon test-result --verbose`, then the launch and multi-agent integration tests) were **not run and are not represented as run**: there is no ROS 2 environment here and `colcon` was not exercised. Installing one is out of scope ("Do not install dependencies, alter the ROS environment…"). No stub or mock import was used to fake `rclpy`.

`BLOCKED — NO ROS 2 ENVIRONMENT ON THIS HOST`, with the exact error captured:

```
python -c "import rclpy"   →  ModuleNotFoundError: No module named 'rclpy'   exit 1
```

Consequently every Tier-I and Tier-H line in the final report is **unmeasured**, and no number in any Exp-13 artefact may be described as a ROS 2, DDS, network, robot or Jetson measurement.

---

## 4. What this analysis does **not** claim

- It does not claim the YAML is functional. It is inert for two independently stated reasons.
- It does not claim a half-integrated production path exists, nor propose one.
- It does not claim any of P-1…P-4 is authorised.
- It does not upgrade the Tier-U reproduction into a Tier-I defect confirmation; the DDS loss behaviour that makes PATH 2 reachable in the field is **inspected, not measured**.
