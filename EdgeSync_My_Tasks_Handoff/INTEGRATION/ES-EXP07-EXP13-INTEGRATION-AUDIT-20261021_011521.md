# ES-EXP07 <-> ES-EXP13 integration audit — pass `20261021_011521`

Supersedes nothing. The previous pass (`ES-EXP07-EXP13-INTEGRATION-AUDIT-20261021_001417.md`) is
retained unchanged; this file records a follow-up time-boxed pass that implemented four local fixes
that pass left as documentation.

Scope: Experiment 7 (TransReID re-identification) + Experiment 13 (CBAA/CBBA-style decentralized
target handoff) + the interfaces between them. Evidence tier: **Tier-U** (model simulation) and
**Tier-N** (node tests under a generated `rclpy` stub). This host has no ROS 2, no DDS, no Jetson and
no TensorRT, so **no Tier-R or Tier-J claim is made anywhere in this file.**

## A. Gate: baseline capture before any edit

`C:/Users/PC/edgesync_int_runs/audit2_20261021/inventory.py` produced
`baseline_20261021_004249.json` (SHA-256 computed on bytes, plus size, mtime, line count and CRLF
count) and a pre-edit copy of every editable target. Three paths in the brief were wrong and were
located before editing: `epoch_manager.py` and `negotiation_manager.py` live under
`edgesync_coordination/fsm/` and `.../protocol/`, `utility_engine.py` under `.../decision/`.

Baseline state, all pure LF: 7 protected files, 2 other-owner files, 8 editable targets.

## B. Findings

| ID | Finding | Previous status | Status after this pass |
|----|---------|-----------------|------------------------|
| F-04 | Winner depends on message arrival order when a belief arrives before an agent's first round | DOCUMENTED | **FIXED + VERIFIED** (reproduced, fixed, regression-tested, mutation-killed) |
| F-02 | `dual_owner_violations` was a per-agent self-consistency check presented as dual-ownership proof | DOCUMENTED | **FIXED + VERIFIED for the metric**; committed dual ownership remains **NOT MEASURED** by design |
| F-03 | Order-independence demonstrated only for N ≤ 4 while claims reached 8 | DOCUMENTED | **FIXED + VERIFIED** for seeded coverage to N = 8 |
| 5.3/5.4 | `ReidSignature` `quality` / `stamp_utc_ns` had no defined producer/consumer semantics | partly open | **FIXED, PARTIALLY VERIFIED** — clock-domain and absent-quality contract now enforced at Tier-N; no cross-host time error model exists |
| F-01 | The benchmark does not compare the two deployed node arms | CORRECTED IN REPORTING | unchanged — **BLOCKED** (needs ROS 2 + the real selector) |
| F-05 | "exp13 converges 60%" is 180/300 observed outcomes | DOCUMENTED | unchanged; re-measured 180/300 in this pass |
| B-02 | `svd_projection_768_to_512_real_v2.npz` carries no provenance metadata | DOCUMENTED, NOT FIXABLE RETROSPECTIVELY | unchanged — keys re-read this pass: `mean, projection, singular_values, total_variance_ratio` |
| F-06 | No valid OSNet ↔ TransReID comparison exists on disk | DOCUMENTED, NOT FIXABLE HERE | unchanged |
| F-07 | Evaluation population self-labelled as an invalid leakage guard | DOCUMENTED | unchanged — no Exp-7 evaluation was re-run in this pass (time-box) |
| — | Identity verification threshold `0.70` (`identity_verifier.py:22`) | separate from retrieval metrics | unchanged; still a pairwise gate, never validated against Rank-1/mAP |

## C. F-04 — pre-round-1 tie-break (fixed)

Reproduced first, before touching code, with `repro_f04_tiebreak.py`:

* Order A (both agents price in round 1, then exchange beliefs) → winner `ugv_1`.
* Order B (`ugv_2`'s belief delivered before `ugv_1`'s first round) → winner `ugv_2`.
* `RESULT: order-independent = False; documented winner honoured = False` (`repro_f04_pre_fix.txt`).

Root cause is two rules that meet at an empty slot. `_merge` adopts a peer claim wholesale when the
local slot is empty, and `build_bundle` then only takes a task when the local bid beats the recorded
bid by more than `bid_epsilon`. A tied local bid can therefore never displace an adopted claim, so
the episode winner was decided by which agent published first. Pricing-before-ingestion in
`run_round` does not cover it: the belief arrives from the inbox, not from a round.

Fix, at the place where the record is set (`cbba_negotiator.py`, inside `build_bundle`): a tied bid
may now take the task when the recorded winner is a *peer* and the documented tie rule
(`_prefer`: equal bid → lexicographically smaller agent id) favours this agent. The rule keeps one
definition — `_prefer` — used by both the merge path and the pricing path, and a zero bid on an
unassigned task still cannot take anything because `recorded_winner` must be non-empty.

Verified: `repro_f04_post_fix.txt` → `order-independent = True; documented winner honoured = True`;
two regression tests in `test_cbba_negotiator.py` (arrival alone adopts `ugv_2`, then
`build_bundle()` flips to `ugv_1`; and the episode-level order test) — both kill the mutation (N1,
see §G). The pre-existing order-sensitivity test was **rewritten, not deleted**, because it asserted
the defect still reproduces.

## D. F-02 — ownership consistency is now three separate numbers

`run_exp13_unit_benchmark.py` keeps the old counter and its JSON key
(`dual_owner_violations_total`), relabelled in the table to **"Local bundle/winner self-consistency
violations"**, and adds a cross-agent measure computed over the same episode:

* `cross_agent_winner_views` — each agent's recorded winner for the target;
* `cross_agent_disagreement_pairs(views)` — unordered pairs of *decided* agents naming different
  winners (extracted to a function so it can be driven with an inconsistent input);
* `committed_dual_ownership_total` — **`None`, "NOT MEASURED"**, with the basis recorded in the
  artefact: a committed owner is only observable on the `Request → Response → Confirm → ConfirmAck`
  path, which this model arm does not execute. The baseline arm reports `None` for the cross-agent
  count too, because a zero there would be a property of the model, not of the protocol.

Measured in this run (300 exp13 episodes): local violations 0, cross-agent pairs 0, cross-agent
disagreement episodes 0, measurable rows 300, committed dual ownership `None`. A self-check block in
the same artefact proves the counter can be non-zero — `all agree → 0`, `one dissents of four → 3`,
`undecided agents are not counted → 1`, `nobody has a winner → 0`, `all_passed=True` — and `main()`
returns 3 if it ever fails, so a zero in the table means "not observed", not "impossible to observe".

## E. F-03 — bounded, seeded order-independence to N = 8

The old probes were capped by `s <= 5` against a default size list of `2,3,4,6,8`, which is why
coverage stopped at 4. `order_independence_probe()` now runs seven delivery schedules (lockstep,
sequential, late_inbox, **late_delivered** — the F-04 trigger — duplicate, delayed,
post-convergence) for every requested N, samples acting orders from the run seed above N = 5, and
each row states its own denominator:

| N | schedules executed | sampling | distinct winners | documented winner | honoured | no-commitment | cross-agent disagreement | DUAL_CLAIM |
|---|------------------|----------|------------------|-------------------|----------|---------------|--------------------------|------------|
| 2 | 28 | fixed seed enumeration | 1 (`ugv_1`) | `ugv_1` | True | 0 | 0 | 0 |
| 3 | 35 | fixed seed enumeration | 1 (`ugv_1`) | `ugv_1` | True | 0 | 0 | 0 |
| 4 | 42 | fixed seed enumeration | 1 (`ugv_1`) | `ugv_1` | True | 0 | 0 | 0 |
| 6 | 98 | seeded sample: 6 permutations + 6 rotations + identity/reverse | 1 | `ugv_1` | True | 0 | 0 | 0 |
| 8 | 112 | seeded sample: 6 permutations + 8 rotations + identity/reverse | 1 | `ugv_1` | True | 0 | 0 | 0 |

These are 315 executed schedules, not a proof of universal order independence: at N = 6 and N = 8 the
acting orders are a documented sample of 8! and 6! permutations.

## F. ReidSignature clock and quality contract (Phase 5.3/5.4)

Consumer (`exp13_coordination_node.py`):

* `REID_SIGNATURE_CLOCK_SKEW_TOLERANCE_NS = 100_000_000`. A stamp more than 100 ms ahead of the node's
  own clock is refused (`refused_future_stamp`), not trusted — two hosts with no shared time error
  model cannot decide freshness from a timestamp.
* Local arrival time is stored as `received_ns` and staleness is now the **worst of two ages**
  (producer stamp age, only when non-negative, and local arrival age). Previously a producer whose
  clock ran ahead made `stamp_age` negative and the signature never expired.
* `quality` is stored as `None` when the field is 0.0, because 0.0 is what the producer emits when it
  measured nothing. Nothing in the arm decides anything from `quality`, and that is asserted by test
  (`quality` never authorises or withholds a handoff, for both 0.0 and 0.95).
* Counters added: `refused_out_of_order`, `refused_future_stamp`.

Producer and contract: `identity_node.py` documents that every call site passes `quality = 0.0`
because nothing measures it, and that smoother frame count and crop area are not confidence measures;
`ReidSignature.msg` now states that `stamp_utc_ns` is the **producer's clock domain** (Unix UTC only
under the default system-time clock and `use_sim_time=false`) and that `quality` is neither an
identity-match confidence nor a detector score. Field names and types are unchanged, so no consumer
breaks; `edgesync_msgs/CMakeLists.txt` is untouched.

Six new Tier-N tests cover this (`I18`–`I23`). One older test was corrected rather than weakened: it
used to simulate "an older signature arriving last" by stamping the newer one a full second into the
consumer's future, which the new skew guard legitimately refuses; it now advances the harness clock.
The end-to-end fixture's default crop stamp moved from 1.5 s to the harness clock instant (1.0 s),
with its paired assertion.

## G. Tests executed

| Batch | Command scope | Result |
|-------|---------------|--------|
| Exp-13 focused | `test_cbba_negotiator.py`, `test_exp13_adapter.py`, `test_exp13_node_integration.py`, `test_exp7_exp13_signature_integration.py` | **313 passed**, 0 failed, 0 skipped |
| `edgesync_coordination` package | all except 3 files | **344 passed**; 3 files NOT COLLECTED: `test_auto_trigger.py`, `test_decision_and_receiver_services.py`, `test_response_collector.py` (`No module named 'rclpy'`) |
| `edgesync_identity` package | all except 1 file | **372 passed, 3 skipped**; 1 file NOT COLLECTED: `test_identity_service.py` (`No module named 'rclpy'`) |
| Workspace sweep | all of `src/` | **429 passed, 3 skipped, 24 collection errors** (`sweep_workspace_20261021_final.txt`); causes: `rclpy` ×4, `sensor_msgs` ×1, and 12 uninstalled-package `ImportError`s — these host-side limits pre-date this pass and none names a file changed here |

Mutation check (`mutation_check_v2.py`, all reads and writes binary): 8/8 mutations killed. The first
run (`mutation_check_20261020_221040.txt`) left **N3 SURVIVED** — the arrival-age test still passed
with the arrival term deleted, because at the margin it had chosen the producer's stamp age exceeded
the limit too. The test was tightened (90 ms skew, 550 ms advance, 500 ms limit: arrival 550 ms stale,
stamp age 460 ms fresh) and the re-run killed it. Under the F-04 mutation the benchmark's own probe
reports `honours={2: False, 3: False}` with 2 distinct winners, so the fix is verified against the
claim rather than only against a test. Implementations were restored byte-identical
(`cbba_negotiator.py=True, exp13_coordination_node.py=True, identity_node.py=True`).

## H. Benchmark re-run

`C:/Users/PC/edgesync_int_runs/audit2_20261021/benchmark_20261021/` (JSON + trials.jsonl + table.md +
console.txt). The repository's own `src/edgesync_bringup/results/experiment13/` was **not written to**
— its files still carry 2026-10-20 19:16 mtimes. 300 episodes per arm, 5 conditions × 5 sizes × 12
trials, seed base 20261020, wall clock 1.25 s.

| arm | converged | undecided | stalled | failed | msgs mean/median | ingested mean | rounds mean/median | winner |
|-----|-----------|-----------|---------|--------|------------------|---------------|--------------------|--------|
| baseline (standalone model) | 276 | 24 | 0 | 0 | 4.2 / 4.0 | 3.92 | 1 / 1.0 | varies by order (`ugv_7`/`ugv_8`/…) |
| exp13 (real `CBBANegotiator`) | 180 | 60 | 60 | 0 | 14.08 / 12.0 | 55.64 | 3.32 / 3.0 | `ugv_1` in every equal-utility cell |

Both arms are **algorithm models executed in one process**; neither imports or invokes
`coordination_node.py`, and the `negotiation_protocol` selector is not exercised by this harness. The
timing column is a single-process interval against a measured host tick, 236 of 240 intervals sit at
or below one tick, so no speed claim is made; rounds remain the convergence measure. 180/300 is the
observed outcome of this scenario set, not a 60% success rate.

## I. Change inventory and integrity

Changed (5 implementation/contract files, 2 test files), all LF-only, verified by `verify.py`:

| File | before → after (SHA-256, 12) | bytes |
|------|------------------------------|-------|
| `.../negotiation/cbba_negotiator.py` | `3117499b58a6` → `002045be0d34` | 56,698 → 58,026 (+1,328) |
| `.../nodes/exp13_coordination_node.py` | `5e2001c35dca` → `6f0183d3b583` | 51,565 → 54,578 (+3,013) |
| `.../scripts/run_exp13_unit_benchmark.py` | `cd5b372a8658` → `4bae60290085` | 59,478 → 76,660 (+17,182) |
| `edgesync_msgs/msg/ReidSignature.msg` | `9915091f316b` → `44fcb363e518` | 1,292 → 2,171 (+879, comments only) |
| `.../nodes/identity_node.py` | `5b0ecb4c9260` → `2ec3ca8db8a2` | 32,275 → 32,635 (+360, comments only) |
| `tests/test_cbba_negotiator.py` | — → `3439ed8db02a` | 59,891 B |
| `tests/test_exp7_exp13_signature_integration.py` | — → `a5d9e25c93e4` | 28,220 B |

Unchanged and verified identical: `coordination_node.py` `60d720373a42`, `ownership_fsm.py`
`99670d02479b`, `epoch_manager.py` `e4a9d9774bf4`, `negotiation_manager.py` `5654621cf5ba`,
`utility_engine.py` `60bc70f353ec`, `robot_bringup.launch.py` `45550fc4b3ed`, `HandoffHeader.msg`
`240976141aaa` — **7/7 protected**. Other-owner: `perception_node.py` `380803b8f77c`,
`target_acquisition_node.py` `80a13dcb8abf` — **2/2**. Also untouched: `exp13_cbba_adapter.py`,
`exp13_bringup.launch.py`, `edgesync_msgs/CMakeLists.txt`. CRLF count is 0 in every file in scope.

## J. Blockers and owner handoffs

1. **Tier-R / Tier-J blocked by the host.** `rclpy`, `sensor_msgs`, `ros2`, a launch system and
   TensorRT are absent; `colcon`, `cmake`, `gcc` are present but nothing has been built, so no package
   is installed and 12 test files import by package name and fail to collect. This is why F-01 stays
   BLOCKED: the node-level comparison and the `negotiation_protocol` selector cannot be executed here.
2. **Committed dual ownership is not observable in the model arm.** Measuring it requires the
   `Request → Response → Confirm → ConfirmAck` / `OwnershipEvent` path under real middleware. The
   field is `None` and labelled NOT MEASURED rather than 0.
3. **Exp-7 evaluation validity (F-06, F-07, B-02) is not a code fix.** It needs a supervisor decision
   on the evaluation population and a defensible split before any re-run; the time-box forbade heavy
   re-extraction, and no Exp-7 metric was re-measured in this pass. The historical `.npz` projection
   keeps its missing provenance; a corrected fit must be written under a new name.
4. **`quality` remains unpopulated on the wire.** A producer that measures it is Exp-7 owner work; the
   contract now says what its absence means, which is the only safe local change.

## K. Verdict

All four time-boxed local items are implemented and verified at the tier the host allows: F-04 is a
confirmed defect, fixed at the pricing rule with a single tie-break definition, reproduced before the
fix and order-independent after it; F-02 and F-03 now report what was actually measured, with a
self-check that fails the run if a counter cannot fire; the `ReidSignature` leg refuses implausible
stamps, ages on local arrival time as well as the producer's stamp, and stops treating an unmeasured
`quality` as a score. Exp-13's convergence cost is unchanged and honestly reported (180 of 300
episodes committed in this arm versus 276 of 300 in the standalone model), and Exp-7's headline
evaluation claims remain unsupported on the evidence on disk. Nothing here is ROS 2, DDS, Jetson or
physical-robot validation.

**Single recommended next step:** run the node-level Tier-R comparison on a machine with ROS 2
installed — real `coordination_node.py`, the `negotiation_protocol` selector actually switching arms,
and the `Request → Response → Confirm → ConfirmAck` path instrumented so committed dual ownership
becomes a measured number instead of `None`.
