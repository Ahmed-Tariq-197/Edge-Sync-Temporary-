# EXP13 — Baseline comparison (autonomous pass `20261020_210721`)

**Evidence:** `C:\Users\PC\edgesync_es13_runs\completion_20261020\benchmark_phaseF_20261020_223131\`
(`console.txt`, `exp13_unit_benchmark.json`, `exp13_unit_benchmark_trials.jsonl`,
`exp13_unit_benchmark_table.md`, `comparison_vs_prior_run.txt`). Prior artefacts at
`src/edgesync_bringup/results/experiment13/` were **not** overwritten; the new run is a *re-execution of the same
harness and the same seed*, not a re-tuned one.

**Tier:** U — unit simulation. Application-layer protocol messages only. No DDS packet, no network, no robot.
**Headline, stated before any table: the comparison is unfavourable to Experiment 13 on both official metrics.
No performance improvement is claimed or demonstrable from this evidence.**

---

## 1. Run as executed

| Field | Value (recorded by the harness itself) |
|---|---|
| Arms | `cnp` baseline model vs `cbba` Exp-13 (CBAA, L = 1) |
| Episodes | 300 per arm, 600 total — 5 conditions × 5 participant counts × 12 trials |
| Seed base | `20261020` |
| Design variety | 245 distinct utility maps, 148 distinct orders (a sweep with variety = 1 would not be randomisable, and the harness records that caveat itself) |
| Participant counts | N = 2, 3, 4, 6, 8 |
| Conditions | `clean`, `equal_utilities`, `replayed_traffic`, `dead_participant`, `one_silent_candidate` |
| Host tick measured this run | 0.014999999999417923 s |
| Simulator wall-clock | 0.250 s for the whole Phase-F run (simulator cost, **not** a protocol latency) |
| Episodes dropped | none |

**Reproducibility check (executed, not asserted):** Phase F vs the prior run agrees on *every* count in the
`__all__` cells for both arms — 300/276/24/24/0/0, 0/0/0/0 and 14.08/12.0/55.64/24.0/3.32 for exp13. Only two
cells differ, both clock-quantisation artefacts of the same underlying intervals:
`convergence_time_at_or_below_host_tick` 238 → 237 and `convergence_time_s mean` 0.000383 → 0.00045 (n = 240 in
both). **The result is reproducible; it is still bad.**

---

## 2. The two official metrics

### 2.1 Application-layer messages per episode — Exp-13 is worse

| | Baseline | Exp-13 | Ratio |
|---|---:|---:|---:|
| Mean sent / episode | **4.2** | **14.08** | 3.35× |
| Median sent / episode | 4.0 | 12.0 | 3.0× |
| Mean ingested / episode | 3.92 | 55.64 | 14.2× |
| Median ingested / episode | 3.0 | 24.0 | 8.0× |

Scaling with N (mean sent): baseline 1.6 / 2.6 / 3.6 / 5.6 / 7.6 at N = 2/3/4/6/8; Exp-13
4.6 / 8.8 / 12.2 / 19 / 25.8. Ingestion is the sharper cost — Exp-13 at N = 8 examines **152.6** payloads per
episode against the baseline's 8, because every participant broadcasts a belief to every peer each round while
the baseline's owner collects a single round of responses.

The 14.2× ingestion gap is not padding: it is the price of every agent holding and merging the full belief
vector instead of the owner reading one reply. It is counted at `payloads_examined_at_receive`, and refusals are
counted separately (§4) so a reader can see how much of that traffic the arm actually *rejected*.

### 2.2 Consensus convergence time — **not measurable on this host, in either direction**

| | Baseline | Exp-13 |
|---|---:|---:|
| Episodes with an interval recorded | **0 of 300** | 240 of 300 |
| Mean interval | NOT MEASURED | 0.00045 s |
| Median interval | NOT MEASURED | 0.0 s |
| Intervals ≤ one host tick | 0 (no intervals exist) | **237 of 240** |
| Mean over resolvable intervals only | NOT MEASURED | 0.016 s |

237 of 240 intervals are at or below the clock's own resolution. They are **duration-blind, not fast**. The
baseline model records no interval at all, so "NOT MEASURED" there means "nothing to average", not "zero". Any
statement of the form "Exp-13 converges faster/slower than the baseline" is unsupported by this run, and the
honest position is that convergence time requires a Tier-R measurement with a real clock and explicit start/end
events.

**Rounds** are the usable convergence measure here, and on rounds the comparison is also unfavourable:
mean 1 (baseline, *definitional* — its model has one round) vs **3.32** (Exp-13, observed, median 3.0).

---

## 3. Commitment success and non-convergence reasons

| | Baseline | Exp-13 |
|---|---:|---:|
| Converged | **276 / 300 (92 %)** | **180 / 300 (60 %)** |
| Undecided (terminated, no winner) | 24 | 60 |
| Stalled (sweep ended non-terminal) | 0 | 60 |
| Failed (safety class violated) | 0 | 0 |
| Winner matched the scenario's expected winner | 276 | 180 |

Non-convergence is entirely concentrated in the two degradation conditions — Exp-13 gets **0/60** in each:

| Condition | Baseline converged | Exp-13 converged | Exp-13 mode of failure |
|---|---:|---:|---|
| `clean` | 60 | **60** | — |
| `equal_utilities` | 60 | **60** | — |
| `replayed_traffic` | 60 | **60** | — |
| `dead_participant` | 48 | **0 / 60** | **stalled** — one declared participant never emits, so distributed agreement is unreachable and the round bound is hit |
| `one_silent_candidate` | 48 | **0 / 60** | **undecided** — terminates with no winner |

The 3 healthy conditions converge at 100 %; the 2 degraded ones converge at 0 %. That is the real shape, and the
mechanism is D4: an unavailable participant is **never silently removed from the consensus set**, so the episode
refuses rather than completing. The baseline completes because its collector never required the missing agent's
answer in the first place.

This is a design trade, not an accident — but the report does not get to call it a win. Under brief §2 the
measured commitment success rate is lower, and under §D4 the refusal is the intended behaviour. Both statements
are true and are recorded together.

---

## 4. What Exp-13 buys with the extra traffic — the three cells that move the other way

| Property | Baseline | Exp-13 |
|---|---:|---:|
| All-participant agreement episodes | **0 / 300** (structurally unreachable: candidates hold no decision state) | **180 / 300** |
| Refused inbound payloads / episode | **0.0** (its modelled collector has *no admission test* — nothing to refuse) | **9.84** mean |
| Distinct winners over enumerated orders at equal utility (N = 2 / 3 / 4) | 1 / **2** / **3** | **1 / 1 / 1** |

### 4.1 Arrival-order independence — the one measured safety-relevant difference

With every utility equal and every order enumerated, the baseline names **2 different winners at N = 3 and 3 at
N = 4**: its outcome depends on the order responses arrive. Exp-13 names exactly **1 winner across all 2, 6 and
24 participant orders**, and the tie rule is symmetric (equal bid at comparison precision → lexicographically
smaller agent id). A second probe — sequential activation, one participant per turn, which is what an
asynchronous adapter actually does — also gives 1 winner across 2 / 6 / 24 activation orders with 0 orders ending
without a commitment. Lockstep cannot expose this defect (in round 1 nobody has a peer belief to ingest), which
is why the probe exists; that is a measurement-methodology finding worth carrying into any re-run.

### 4.2 Admission control is doing work under replay

`replayed_traffic` costs **42.4 refused payloads per episode** while still converging **60/60** and recording
**0** dual-ownership or false-convergence events. The episode-id/epoch fence (`cbba_negotiator.py:1040`, adapter
`on_belief` `exp13_cbba_adapter.py:633`) rejects replayed beliefs without breaking agreement. The baseline
reports 0 refusals in the same condition — not because nothing is wrong, but because nothing is checked.

### 4.3 Per-condition refusal load (Exp-13)

`clean` 0.2 · `equal_utilities` 0.2 · `one_silent_candidate` 1.8 · `dead_participant` 4.6 · `replayed_traffic` 42.4.

---

## 5. Safety counters — reported, and bounded

| Counter | Baseline | Exp-13 |
|---|---:|---:|
| Dual-owner violations | 0 | 0 |
| False convergence (converged flag without AGREED) | 0 | 0 |

**These zeros do not mean "safe".** They come from the harness's own single-process ownership model; the
guards that matter (M0–M4) are tested at Tier-N against stub middleware, and the deployed baseline still
contains the dual-OWNER path recorded in `EXP13_SAFETY_FINDINGS_20261020_210721.md` S-10. Per brief §2:
**fewer messages or faster convergence would not automatically mean safer ownership transfer, and in this run
Exp-13 has neither — so no safety advantage is claimed from the performance columns either.** The defensible
claim is narrower and comes from §4.1/§4.2: arrival-order-independent winner selection and an admission gate
that demonstrably operates.

---

## 6. Confounds, stated so the comparison is not over-read

1. **The arms are not the same code path.** `cbba` differs from the deployed node by exactly M0–M4 + the round
   timer + the belief endpoints. The baseline's model in this harness also has **no admission test at all**, so
   the refusal counts are not symmetric: 0 for the baseline is definitional, not observed.
2. **The baseline's "1 round" is a property of the model**, not a measurement of a protocol that could have taken
   more.
3. **Tier-U traffic ≠ network traffic.** Beliefs are counted at publish time
   (`message_count_unit: belief_builds_counted_at_publish`). No serialization, no DDS packet, no retransmit.
4. **Clocks.** Every interval in §2.2 is single-process and sub-tick.
5. **The degraded conditions encode D4's refusal policy.** A reviewer who wants Exp-13 to "recover" from a dead
   participant would be asking it to drop a declared participant from the consensus set, which brief §D4 forbids.
   The fair alternative is a *different experiment* with a re-configured participant set and an explicit
   re-participation rule.
6. **No tuning was applied to help Exp-13.** Thresholds, `max_rounds`, `deadline_s`, `bid_epsilon`,
   `commit_grace_ms` and the conditions are those in `exp13_cbba.yaml` and were not adjusted against these
   results.

---

## 7. Conclusion

* **Performance improvement demonstrated: NO.** Exp-13 sends 3.35× the application messages, ingests 14.2×,
  runs 3.32 rounds to the baseline's definitional 1, converges on 60 % of episodes against 92 %, and its
  convergence time is unmeasurable on this host.
* **What the evidence supports:** winner selection that is independent of arrival and activation order under
  equal utility (1 distinct winner over 24 orders vs the baseline's 3), all-participant agreement that the
  baseline cannot express (180 vs structurally 0), and an admission gate that refused 42.4 replayed payloads per
  episode while still converging 60/60.
* **What would change the verdict:** a Tier-R run with a real clock and explicit start/end events (convergence
  time is currently unmeasurable in either direction), plus either a re-participation rule for a dead
  participant or a scenario in which the extra rounds cost less than the refusals they prevent.
