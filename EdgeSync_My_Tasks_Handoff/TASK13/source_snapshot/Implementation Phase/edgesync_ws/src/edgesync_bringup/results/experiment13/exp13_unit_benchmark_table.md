# Exp-13 unit-simulation comparison — Tier U (unit simulation)

exp_id=EXP13 protocol=cbba variant=CBAA (CBBA with bundle length L = 1)
Episodes per arm: 300 (5 conditions x 5 sizes x 12 trials), seed base 20261020.
Message layer: application protocol message; DDS/network packets NOT MEASURED.
Host monotonic tick measured this run: 0.014999999999417923s.
Any millisecond figure below is a single-process interval, not a robot or network latency; differences at or below one tick are clock quantisation.

| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 300 | 300 | actual trial x cell count, no row dropped |
| Converged episodes | 276 | 180 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 24 | 120 | reported, never excluded |
| Undecided episodes | 24 | 60 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 60 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 180 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 4.2 | 14.08 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 4.0 | 12.0 | same boundary |
| Mean ingested messages | 3.92 | 55.64 | inbound application messages actually examined at receive time |
| Median ingested messages | 3.0 | 24.0 | same boundary |
| Mean convergence rounds | 1 | 3.32 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000383 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 300 | 60 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 238 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | 0.016 | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 9.84 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

## Per condition
### clean
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 60 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 0 | reported, never excluded |
| Undecided episodes | 0 | 0 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 0 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 60 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 4.6 | 13.6 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 4.0 | 12.0 | same boundary |
| Mean ingested messages | 3.6 | 42.4 | inbound application messages actually examined at receive time |
| Median ingested messages | 3.0 | 24.0 | same boundary |
| Mean convergence rounds | 1 | 3 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000267 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 0 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 59 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | 0.016 | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 0.2 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### one_silent_candidate
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 48 | 0 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 12 | 60 | reported, never excluded |
| Undecided episodes | 12 | 60 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 0 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 0 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 3.6 | 16.2 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 3.0 | 14.0 | same boundary |
| Mean ingested messages | 2.6 | 52.6 | inbound application messages actually examined at receive time |
| Median ingested messages | 2.0 | 27.0 | same boundary |
| Mean convergence rounds | 1 | 3.8 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 4.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.0005 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 0 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 60 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 1.8 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### dead_participant
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 48 | 0 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 12 | 60 | reported, never excluded |
| Undecided episodes | 12 | 0 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 60 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 0 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 3.6 | 13.4 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 3.0 | 11.0 | same boundary |
| Mean ingested messages | 2.6 | 56 | inbound application messages actually examined at receive time |
| Median ingested messages | 2.0 | 24.0 | same boundary |
| Mean convergence rounds | 1 | 3.8 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 4.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | NOT MEASURED | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | NOT MEASURED | single-process monotonic interval |
| Episodes with no convergence time | 60 | 60 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 0 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 4.6 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### replayed_traffic
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 60 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 0 | reported, never excluded |
| Undecided episodes | 0 | 0 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 0 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 60 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 4.6 | 13.6 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 4.0 | 12.0 | same boundary |
| Mean ingested messages | 7.2 | 84.8 | inbound application messages actually examined at receive time |
| Median ingested messages | 6.0 | 48.0 | same boundary |
| Mean convergence rounds | 1 | 3 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000517 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 0 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 59 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | 0.016 | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 42.4 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### equal_utilities
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 60 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 0 | reported, never excluded |
| Undecided episodes | 0 | 0 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 0 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 60 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 4.6 | 13.6 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 4.0 | 12.0 | same boundary |
| Mean ingested messages | 3.6 | 42.4 | inbound application messages actually examined at receive time |
| Median ingested messages | 3.0 | 24.0 | same boundary |
| Mean convergence rounds | 1 | 3 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.00025 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 0 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 60 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 0.2 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

## Per participant count
### N = 2
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 36 | 36 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 24 | 24 | reported, never excluded |
| Undecided episodes | 24 | 12 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 12 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 36 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 1.6 | 4.6 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 2.0 | 5.0 | same boundary |
| Mean ingested messages | 0.8 | 3.6 | inbound application messages actually examined at receive time |
| Median ingested messages | 1.0 | 4.0 | same boundary |
| Mean convergence rounds | 1 | 3 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.0 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 12 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 48 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 1.8 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### N = 3
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 36 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 24 | reported, never excluded |
| Undecided episodes | 0 | 12 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 12 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 36 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 2.6 | 8.8 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 3.0 | 9.0 | same boundary |
| Mean ingested messages | 2 | 13.6 | inbound application messages actually examined at receive time |
| Median ingested messages | 2.0 | 12.0 | same boundary |
| Mean convergence rounds | 1 | 3.4 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000312 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 12 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 48 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 3.4 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### N = 4
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 36 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 24 | reported, never excluded |
| Undecided episodes | 0 | 12 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 12 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 36 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 3.6 | 12.2 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 4.0 | 12.0 | same boundary |
| Mean ingested messages | 3.2 | 29.4 | inbound application messages actually examined at receive time |
| Median ingested messages | 3.0 | 24.0 | same boundary |
| Mean convergence rounds | 1 | 3.4 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.0 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 12 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 48 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | NOT MEASURED | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 6 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### N = 6
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 36 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 24 | reported, never excluded |
| Undecided episodes | 0 | 12 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 12 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 36 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 5.6 | 19 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 6.0 | 18.0 | same boundary |
| Mean ingested messages | 5.6 | 79 | inbound application messages actually examined at receive time |
| Median ingested messages | 5.0 | 75.0 | same boundary |
| Mean convergence rounds | 1 | 3.4 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000646 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 12 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 47 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | 0.016 | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 13.6 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

### N = 8
| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |
|---|---:|---:|---|
| Episodes executed | 60 | 60 | actual trial x cell count, no row dropped |
| Converged episodes | 60 | 36 | arm committed an allocation; derived from its own end state |
| Non-converged episodes | 0 | 24 | reported, never excluded |
| Undecided episodes | 0 | 12 | terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / MAX_ROUNDS / TIMEOUT) |
| Stalled (non-terminal) | 0 | 12 | sweep stopped before the arm terminated |
| Failed episodes | 0 | 0 | safety class violated (DUAL_CLAIM) |
| All-participant-agreed episodes | 0 | 36 | every declared participant names the same winner; structurally unreachable for the baseline, whose candidates hold no decision state |
| Mean application messages | 7.6 | 25.8 | application protocol messages produced per episode; NOT DDS packets |
| Median application messages | 8.0 | 24.0 | same boundary |
| Mean ingested messages | 8 | 152.6 | inbound application messages actually examined at receive time |
| Median ingested messages | 7.0 | 147.0 | same boundary |
| Mean convergence rounds | 1 | 3.4 | rounds until the arm stopped; baseline's 1 is definitional, not observed |
| Median convergence rounds | 1.0 | 3.0 | same stopping rule per arm |
| Mean convergence time | NOT MEASURED | 0.000958 | single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero' |
| Median convergence time | NOT MEASURED | 0.0 | single-process monotonic interval |
| Episodes with no convergence time | 60 | 12 | agreement never established, so no interval exists; not counted as 0 |
| Intervals at or below one host tick | 0 | 47 | the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only because it has no intervals to classify |
| Mean time over resolvable intervals only | NOT MEASURED | 0.016 | subset whose interval exceeded one tick; n reported in the JSON |
| Mean refused ingress | 0 | 24.4 | inbound payloads classified and dropped (baseline: definitionally 0) |
| Dual-owner violations | 0 | 0 | safety counter, both arms |
| False convergence | 0 | 0 | converged flag without AGREED |

## Determinism probe (all utilities equal, every order enumerated)
* N=2 (utilities equal: True): baseline 1 distinct winner(s) over 1 response orders ['ugv_2'] (0 orderings produced none) | Exp-13 1 over 2 participant orders ['ugv_1'] (0 produced none)
* N=3 (utilities equal: True): baseline 2 distinct winner(s) over 2 response orders ['ugv_2', 'ugv_3'] (0 orderings produced none) | Exp-13 1 over 6 participant orders ['ugv_1'] (0 produced none)
* N=4 (utilities equal: True): baseline 3 distinct winner(s) over 6 response orders ['ugv_2', 'ugv_3', 'ugv_4'] (0 orderings produced none) | Exp-13 1 over 24 participant orders ['ugv_1'] (0 produced none)

## Sequential activation probe (equal utilities, one participant per turn)

Lockstep cannot expose an activation-order defect: in round 1 nobody has a peer belief to ingest, so every agent prices before it merges. This probe turns participants one at a time, which is what an asynchronous adapter really does.
* N=2 (utilities equal: True): Exp-13 1 distinct winner(s) over 2 activation orders ['ugv_1'] (0 orders ended without a commitment)
* N=3 (utilities equal: True): Exp-13 1 distinct winner(s) over 6 activation orders ['ugv_1'] (0 orders ended without a commitment)
* N=4 (utilities equal: True): Exp-13 1 distinct winner(s) over 24 activation orders ['ugv_1'] (0 orders ended without a commitment)
