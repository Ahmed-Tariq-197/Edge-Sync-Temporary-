#!/usr/bin/env python3
"""Unit-level comparison harness for Experiment 13 (CBAA, L = 1) against the baseline selector.

WHAT THIS MEASURES AND WHAT IT DOES NOT
    Evidence tier: **Tier U (unit simulation)**. Every figure is produced inside one Python process
    on this host. There is no ROS graph, no DDS transport, no network stack, no robot and no Jetson.
    Nothing here is a multi-robot or on-device measurement and no number may be quoted as one.

    MESSAGE LAYER, STATED ONCE AND ENFORCED EVERYWHERE
        ``app_messages_sent``     one **application-level protocol message** produced by a
                                 participant and handed to its transport during one episode.
        ``app_messages_ingested`` one inbound application-level message a participant actually
                                 examined at receive time.
        Both counters are incremented inside the loop that performs the send or the receive, from
        the object the arm really produced or really read. Neither is computed from a formula about
        what the arm "would" have exchanged.
        DDS packets, serialised bytes and network frames are **NOT MEASURED** — no transport exists
        in this process, so any such number would be an estimate presented as an observation.

    ONE ACCOUNTING BOUNDARY FOR BOTH ARMS
        An episode is bounded by the moment the arm can no longer change its allocation: the
        baseline's collection window closing, or Exp-13's participant set becoming terminal.
        Everything each arm produces or reads inside that boundary is counted, including traffic
        that was refused, and including episodes that end with no allocation. No row is dropped.

    Outcomes are derived, never assigned. ``decided``, ``agreed_by_all``,
    ``participants_at_terminal_state`` and ``outcome_class`` are read out of each arm's own end
    state; the collector loop that fills them is the same loop that produces the messages.

THE TWO ARMS
    ``baseline``  A stdlib re-implementation of the selection logic that is actually in the
                  repository: one published request, a fixed response window, append every response
                  that arrives, sort the ACCEPTs by utility descending, take index 0.
                  coordination_node.py:441-479 (publish), :484-495 (window), :498-513 (collect,
                  which appends **unconditionally** — there is no duplicate or staleness admission),
                  :538 (``list.sort`` is stable, so equal utilities resolve by arrival order),
                  :542 (winner), :534 (abort to recovery on zero ACCEPTs).
                  It is a MODEL, not the node: CoordinationNode cannot be imported on this host
                  because there is no ``rclpy``.

    ``exp13``     The real deliverable class ``CBBANegotiator``, driven in lockstep under a broadcast
                  delivery model, bidding with the real ``UtilityEngine``.

TRIALS
    Every (condition, participant-count) cell runs ``--trials`` independent episodes. A trial varies
    per-agent battery and viewpoint jitter and the arrival/delivery order, from one seeded stream;
    **both arms receive the identical jittered inputs and the identical order**, so a difference
    between arms is a property of the allocator, not of the draw. ``--seed 0`` makes the whole sweep
    reproducible. The ``randomisation_audit`` block proves the stream actually changed the inputs —
    a run that varied nothing would report ``distinct_utility_maps == 1`` and would then be
    described as "not randomised", because it would not be.

STRUCTURAL DIFFERENCES THAT CANNOT BE MADE EQUIVALENT (disclosed, not smoothed over)
    * The baseline is one-sided: the current owner asks, candidates answer, the owner alone decides.
      Exp-13 is symmetric: every declared participant bids and recomputes the same winner vector.
      "Messages sent" is therefore counted over the same participant set but not over identical
      roles, and the baseline can never report agreement from more than one participant.
    * Stopping conditions differ by construction: the baseline stops when its fixed response window
      closes (``t_neg_timeout_ms``); Exp-13 stops on all-participant belief agreement, its round
      bound, or its deadline. One round bound cannot be translated into the other without a measured
      per-round period, which does not exist, so ``rounds`` is reported per arm and never converted.
    * Candidate-selection rules differ: the baseline excludes the owner from its candidate pool
      (:441-479), Exp-13 lets the owner bid and therefore retain (self-win). The same utilities are
      offered to both; who is allowed to use them is protocol, not tuning.

Both arms share every input except the allocator: same participants, same target geometry, same
per-agent utilities from the same UtilityEngine, same injection conditions, same trial jitter and
order, and parameters read from config/exp13_cbba.yaml (Exp-13 arm) and
config/coordination_params.yaml (baseline).
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from edgesync_coordination.decision.utility_engine import UtilityEngine
from edgesync_coordination.negotiation import CBBANegotiator, ConsensusStatus, TaskInfo

WS_SRC = Path(__file__).resolve().parents[3]
EXP13_CONFIG = WS_SRC / "edgesync_bringup" / "config" / "exp13_cbba.yaml"
BASELINE_CONFIG = WS_SRC / "edgesync_bringup" / "config" / "coordination_params.yaml"
DEFAULT_OUT = WS_SRC / "edgesync_bringup" / "results" / "experiment13"

EVIDENCE_TIER = "Tier U (unit simulation)"
MESSAGE_LAYER = "application protocol message; DDS/network packets NOT MEASURED"
NOT_MEASURED = "NOT MEASURED"
TASK_ID = "task_0"
TARGET_ID = "target_0"
Z_M = 5.0                      # both arms see the same geometry; only the allocator differs
AXIS_STEP_M = 0.35            # per-participant off-axis offset, so utilities are distinct
BATTERY_JITTER_MAH = 6.0      # per-trial telemetry variation, identical for both arms
AXIS_JITTER_M = 0.06


# --------------------------------------------------------------------- configuration

def load_ros_params(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]


def load_configs() -> Tuple[dict, dict]:
    return load_ros_params(EXP13_CONFIG), load_ros_params(BASELINE_CONFIG)


def build_engine(baseline_cfg: dict) -> Tuple[UtilityEngine, dict]:
    """Receiver utility with the baseline's own weights.

    coordination_params.yaml carries no w_fov/w_dist/w_battery keys, so a launched node uses the
    declare_parameter defaults at coordination_node.py:58-60 (0.50 / 0.30 / 0.20), read at :80-82
    and injected at :109-111. Those defaults are used here and the provenance is recorded.
    """
    weights = {
        "w_receiver_fov": float(baseline_cfg.get("w_fov", 0.50)),
        "w_receiver_dist": float(baseline_cfg.get("w_dist", 0.30)),
        "w_receiver_battery": float(baseline_cfg.get("w_battery", 0.20)),
    }
    provenance = {
        "weights": weights,
        "source": ("coordination_params.yaml" if all(k in baseline_cfg for k in
                                                     ("w_fov", "w_dist", "w_battery"))
                   else "coordination_node.py:58-60 declare_parameter defaults"),
    }
    return UtilityEngine(**weights), provenance


def measure_host_tick_s(budget_s: float = 0.25) -> Optional[float]:
    """Smallest positive ``time.monotonic()`` delta observable on this host, measured now.

    Sampling is bounded by a wall budget, not an iteration count: a fixed 4000-sample loop finishes
    inside one tick on this host and would report "no delta observed".

    Reported alongside every elapsed-time figure: an interval below one tick cannot be resolved, so
    a mean convergence time of a few ticks — or of zeros — is not evidence about relative speed.
    """
    best: Optional[float] = None
    started = time.monotonic()
    previous = started
    while time.monotonic() - started < budget_s:
        now = time.monotonic()
        delta = now - previous
        if delta > 0 and (best is None or delta < best):
            best = delta
        previous = now
    return best


class AgentUtility:
    """Per-participant bid calculator: the shared UtilityEngine at that participant's own viewpoint.

    The geometry is keyed by agent identity, never by position in a list, so re-ordering the
    participant list cannot silently change who has the better view — otherwise a "winner changed"
    result would be a harness artefact rather than a protocol property.
    """

    def __init__(self, engine: UtilityEngine, axis_m: float, battery_pct: float):
        self.engine = engine
        self.axis = float(axis_m)
        self.battery_pct = battery_pct

    def compute_receiver_utility(self, point_camera=None, battery_pct=None, distance=None):
        return self.engine.compute_receiver_utility(
            point_camera=[self.axis, 0.0, Z_M], battery_pct=self.battery_pct)


def task_for() -> TaskInfo:
    return TaskInfo(task_id=TASK_ID, target_id=TARGET_ID, position=(0.0, 0.0, Z_M),
                    tto_safe_s=2.0, sender_id="ugv_1")


def utility_of(engine: UtilityEngine, axis_m: float, battery_pct: float) -> float:
    score, _, _, _ = engine.compute_receiver_utility(
        point_camera=[float(axis_m), 0.0, Z_M], battery_pct=battery_pct)
    return score


def index_of(agent: str) -> int:
    """Stable identity-derived index (``ugv_3`` -> 2)."""
    return int(str(agent).rsplit("_", 1)[-1]) - 1


def base_axes(agents: Sequence[str], geometry: str) -> Dict[str, float]:
    if geometry == "distinct":
        return {a: index_of(a) * AXIS_STEP_M for a in agents}
    if geometry == "shared":
        return {a: 0.0 for a in agents}
    raise ValueError(f"unknown geometry {geometry}")


def base_batteries(agents: Sequence[str], profile: str) -> Dict[str, float]:
    if profile == "distinct":
        return {a: 90.0 - 7.0 * index_of(a) for a in agents}
    if profile == "equal":
        return {a: 80.0 for a in agents}
    raise ValueError(f"unknown battery profile {profile}")


def jitter(agents: Sequence[str], rng: random.Random, axes: Dict[str, float],
           batteries: Dict[str, float], keep_equal: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-trial telemetry variation, applied identically to both arms.

    ``keep_equal`` is set for the tie cells: jittering their batteries or viewpoints would make the
    utilities unequal and the row would stop testing tie-breaking. Those trials vary order only.
    """
    if keep_equal:
        return dict(axes), dict(batteries)
    jittered_axes = {a: round(axes[a] + rng.uniform(-AXIS_JITTER_M, AXIS_JITTER_M), 3)
                     for a in agents}
    jittered_bat = {a: round(min(100.0, max(5.0, batteries[a]
                                            + rng.uniform(-BATTERY_JITTER_MAH, BATTERY_JITTER_MAH))), 1)
                    for a in agents}
    return jittered_axes, jittered_bat


# --------------------------------------------------------------------- result record

@dataclass
class ArmResult:
    arm: str
    condition: str
    participants: int
    trial: int = 0
    app_messages_sent: int = 0
    app_messages_sent_max_per_agent: int = 0
    per_agent_app_messages_sent: Dict[str, int] = field(default_factory=dict)
    app_messages_ingested: int = 0
    refused_ingress: int = 0
    refused_ingress_basis: str = ""
    rejections_by_class: Dict[str, int] = field(default_factory=dict)
    rounds: int = 0
    rounds_basis: str = ""
    decided: bool = False
    agreed_by_all: bool = False
    participants_at_terminal_state: int = 0
    participants_naming_same_winner: int = 0
    converged_agents: int = 0
    outcome_class: str = ""
    outcome_blob: str = ""
    participant_states: Dict[str, str] = field(default_factory=dict)
    winner: Optional[str] = None
    expected_winner: Optional[str] = None
    selection_list_size: int = 0
    convergence_time_s_max: Optional[float] = None
    dual_owner_violations: int = 0
    false_convergence: int = 0
    # F-02. `dual_owner_violations` above only compares an agent's own bundle with its own winning
    # vector, so it can never see two *different* agents each believing they own the target. These
    # fields measure the two things that were previously conflated with it, and the thing that this
    # arm cannot measure at all, as `None` rather than as a reassuring zero.
    cross_agent_winner_views: Dict[str, str] = field(default_factory=dict)
    cross_agent_disagreeing_pairs: Optional[int] = None
    ownership_commit_basis: str = ""
    message_layer: str = MESSAGE_LAYER
    note: str = ""
    # the exact inputs this cell's trial fed to BOTH arms, recorded so any number can be re-derived
    utilities: Dict[str, float] = field(default_factory=dict)
    batteries: Dict[str, float] = field(default_factory=dict)
    axes_m: Dict[str, float] = field(default_factory=dict)
    arrival_order: List[str] = field(default_factory=list)


def classify(res: ArmResult) -> str:
    """One classification rule, applied to both arms from each arm's own observed end state."""
    if res.decided:
        return "CONVERGED"
    if "DUAL_CLAIM" in res.outcome_blob:
        return "FAILED"
    if "RUNNING" in res.outcome_blob:
        return "STALLED_NOT_TERMINAL"
    return "UNDECIDED"


# --------------------------------------------------------------------- baseline arm

def baseline_expected(agents: Sequence[str], sender: str, heard: Optional[Sequence[str]],
                      utilities: Dict[str, float]) -> Optional[str]:
    """Highest utility among those heard; ties keep arrival order because list.sort is stable."""
    pool = [a for a in agents if a != sender] if heard is None \
        else [a for a in heard if a != sender]
    if not pool:
        return None
    ordered = sorted(pool, key=lambda a: round(utilities[a], 4), reverse=True)
    return ordered[0]


def run_baseline(agents: Sequence[str], utilities: Dict[str, float], sender: str,
                 condition: str, arrival_order: Optional[Sequence[str]] = None,
                 duplicate_factor: int = 0, trial: int = 0) -> ArmResult:
    """One request/window episode, simulated step by step so every counter is an observation.

    The receipt loop mirrors ``_on_handoff_response`` (coordination_node.py:498-513), which appends
    every arriving response with no duplicate or staleness test; the ranking mirrors :538, whose
    ``list.sort`` is stable, so equal utilities keep arrival order.
    """
    res = ArmResult(arm="baseline", condition=condition, participants=len(agents), trial=trial)
    candidates = [a for a in agents if a != sender]
    heard = list(candidates) if arrival_order is None else \
        [a for a in arrival_order if a in candidates]

    # --- send phase: the owner publishes one request; each candidate that answers publishes one.
    # A transport duplicate is re-offered at receive time, so receipts grow but sends do not.
    res.app_messages_sent = 1 + len(heard)
    res.per_agent_app_messages_sent = {a: (1 if a == sender or a in heard else 0) for a in agents}
    res.app_messages_sent_max_per_agent = max(res.per_agent_app_messages_sent.values())
    res.rounds = 1
    res.rounds_basis = ("definitional: one request plus one fixed collection window "
                        "(coordination_node.py:484-495); not an observed iteration count")

    # --- receive phase: count every payload the collector examined, and append it unconditionally
    collected: List[Tuple[str, float]] = []
    for candidate in heard:
        for _ in range(1 + duplicate_factor):
            res.app_messages_ingested += 1
            collected.append((candidate, round(utilities[candidate], 4)))
    res.selection_list_size = len(collected)
    res.refused_ingress = 0
    res.refused_ingress_basis = ("definitional: the modelled collector has no admission test at "
                                 "coordination_node.py:498-513, so nothing can be refused")
    res.note = ("no duplicate/staleness admission control in the collector (:498-513); "
                "owner is requester and never a candidate (:441-479)")
    # Left deliberately unmeasured. This model has exactly one deciding agent, so a "0" here would
    # be an artefact of the model rather than an observation, and reporting it next to Exp-13's real
    # count would imply the two arms were tested for the same property.
    res.cross_agent_disagreeing_pairs = None
    res.ownership_commit_basis = (
        "owner-side COMMITTED is a model reconstruction of the deployed decision, not an executed "
        "Confirm/ConfirmAck exchange; candidates hold no state, so cross-agent disagreement is "
        "NOT MEASURABLE in this arm")

    # --- decision phase: the owner's state is read out of the loop above, never assigned
    owner_state = "ABORTED" if not collected else "COMMITTED"
    # The window closing makes the owner terminal. A candidate that answered holds no decision
    # state at all in this protocol, so it is never counted as terminal.
    res.participant_states = {a: ("WINDOW_CLOSED_" + owner_state if a == sender else
                                  ("ANSWERED" if a in heard else "SILENT")) for a in agents}
    res.participants_at_terminal_state = sum(
        1 for state in res.participant_states.values() if state.startswith("WINDOW_CLOSED"))
    res.outcome_blob = owner_state

    if not collected:
        res.decided = False
        res.converged_agents = 0
        res.outcome_class = classify(res)
        res.note += "; zero ACCEPTs in window -> :534 abort, owner keeps tracking"
        return res

    ordered = sorted(collected, key=lambda item: item[1], reverse=True)   # stable, as at :538
    res.winner = ordered[0][0]
    res.expected_winner = baseline_expected(agents, sender, heard, utilities)
    res.decided = res.winner is not None
    res.converged_agents = res.participants_at_terminal_state
    # Only the owner holds a decision; a candidate that answered has no state to agree with.
    res.participants_naming_same_winner = 1
    res.agreed_by_all = res.decided and res.participants_at_terminal_state == res.participants
    res.outcome_class = classify(res)
    res.note += f"; owner COMMITTED from {len(collected)} receipt(s)"
    return res


# --------------------------------------------------------------------- Exp-13 arm

def cross_agent_disagreement_pairs(views: Dict[str, str]) -> int:
    """Count unordered agent pairs that name different winners, ignoring agents that name nobody.

    Extracted from ``run_exp13`` so the counter can be fed a deliberately inconsistent input. The
    metric exists because the pre-existing ``dual_owner_violations`` only compared each agent with
    itself, so on honest agents it could report zero while two agents disagreed about the owner.
    """
    named = sorted(a for a, w in views.items() if w)
    return sum(1 for i, a in enumerate(named)
               for b in named[i + 1:] if views[a] != views[b])


def run_exp13(agents: Sequence[str], engine: UtilityEngine, battery: Dict[str, float],
              axes: Dict[str, float], exp_cfg: dict, condition: str,
              deliver: Optional[Dict[str, Sequence[str]]] = None,
              duplicate_factor: int = 0, silent: Optional[str] = None,
              trial: int = 0, episode_id: str = "bench") -> ArmResult:
    """Lockstep CBAA rounds against the real negotiator class.

    ``deliver`` restricts whose beliefs reach each agent; ``silent`` names an agent that runs no
    round at all. A partitioned member is declared in ``participant_ids``, so the all-converged
    gate can withhold the episode — that is the property being measured, not an accident.

    Every message counter here is incremented where the message is actually produced or actually
    handed to ``run_round``: ``app_messages_sent`` comes from each negotiator's own
    ``stats.beliefs_published`` and ``app_messages_ingested`` from the real inbox lengths.
    """
    max_rounds = int(exp_cfg["max_rounds"])
    res = ArmResult(arm="exp13", condition=condition, participants=len(agents), trial=trial)

    negotiators = build_negotiators(agents, engine, battery, axes, exp_cfg,
                                    f"{episode_id}:n{len(agents)}:t{trial}")

    inflight: Dict[str, object] = {}
    for _ in range(max_rounds + 1):
        if all(n.is_terminal() for n in negotiators.values()):
            break
        out = {}
        for agent, neg in negotiators.items():
            if agent == silent:
                continue          # participant is down: it neither publishes nor listens
            allowed = None if deliver is None else list(deliver.get(agent, []))
            inbox = [m for sender, m in inflight.items()
                     if sender != agent and (allowed is None or sender in allowed)]
            inbox = inbox + list(inbox) * duplicate_factor
            res.app_messages_ingested += len(inbox)      # observed, not derived from rounds
            _, belief = neg.run_round(inbox)
            out[agent] = belief
        inflight = out

    terminal = [a for a, n in negotiators.items() if n.is_terminal()]
    converged = sorted(a for a, n in negotiators.items() if n.is_converged())
    res.app_messages_sent = sum(n.stats.beliefs_published for n in negotiators.values())
    res.per_agent_app_messages_sent = {a: n.stats.beliefs_published for a, n in negotiators.items()}
    res.app_messages_sent_max_per_agent = max(res.per_agent_app_messages_sent.values())
    res.rounds = max(n.current_round for n in negotiators.values())
    res.rounds_basis = "observed: highest per-participant round counter when the sweep stopped"
    res.refused_ingress = sum(n.stats.total_rejected for n in negotiators.values())
    res.refused_ingress_basis = ("observed: each negotiator classified the payload and named the "
                                 "reason; counted per class in rejections_by_class")
    by_class: Dict[str, int] = {}
    for neg in negotiators.values():
        for cls, count in neg.stats.rejected.items():
            by_class[cls.name] = by_class.get(cls.name, 0) + count
    res.rejections_by_class = dict(sorted(by_class.items()))
    times = [n.stats.convergence_time_s for n in negotiators.values()
             if n.stats.convergence_time_s is not None]
    res.convergence_time_s_max = max(times) if times else None
    res.selection_list_size = len(converged)
    res.participants_at_terminal_state = len(terminal)
    res.converged_agents = len(converged)
    winners = {n.winning_agents.get(TASK_ID, "") for n in negotiators.values()}
    winners.discard("")
    res.decided = (len(terminal) == len(negotiators) and len(converged) == len(negotiators))
    if res.decided:
        res.winner = negotiators[converged[0]].winning_agents.get(TASK_ID)
        res.participants_naming_same_winner = sum(
            1 for n in negotiators.values() if n.winning_agents.get(TASK_ID) == res.winner)
    res.agreed_by_all = res.decided and res.participants_naming_same_winner == res.participants
    res.outcome_blob = "|".join(sorted({n.convergence_status.name
                                        for n in negotiators.values()}))
    res.participant_states = {a: n.convergence_status.name for a, n in negotiators.items()}

    for neg in negotiators.values():
        held = set(neg.bundle)
        owned = {t for t, w in neg.winning_agents.items() if w == neg.agent_id}
        if not held <= owned:
            res.dual_owner_violations += 1
        if neg.is_converged() and neg.convergence_status is not ConsensusStatus.AGREED:
            res.false_convergence += 1

    # Local self-consistency only: the counter above asks each agent whether its own bundle is
    # consistent with its own belief. It is silent on the safety question it was being read as
    # answering, so the cross-agent view is counted separately here.
    views = {a: n.winning_agents.get(TASK_ID, "") for a, n in sorted(negotiators.items())}
    res.cross_agent_winner_views = views
    res.cross_agent_disagreeing_pairs = cross_agent_disagreement_pairs(views)
    res.ownership_commit_basis = (
        "consensus winner is an advisory local belief; a committed owner is only observable on the "
        "Request -> Response -> Confirm -> ConfirmAck path, which this model arm does not execute")

    res.outcome_class = classify(res)
    _append_note(res, "every declared participant bids, including the owner (self-win = retain)")
    if not res.decided:
        _append_note(res, "no episode commitment: not every declared participant converged. A "
                          "member that hears every peer can still reach AGREED from its own "
                          "viewpoint; the all-converged gate is what withholds the handoff")
    return res


# --------------------------------------------------------------------- scenarios

def _append_note(res: ArmResult, text: str) -> None:
    res.note = f"{res.note}; {text}" if res.note else text


def expected_exp13(agents: Sequence[str], utilities: Dict[str, float]) -> Optional[str]:
    """Highest bid, ties broken by the smallest agent id — the module's symmetric rule."""
    best = max(round(u, 4) for u in utilities.values())
    tied = sorted(a for a in agents if round(utilities[a], 4) == best)
    return tied[0] if tied else None


CONDITIONS = {
    # condition -> (geometry, battery profile). "shared" makes the utilities genuinely equal, so the
    # equal_utilities row compares the two arms' tie rules rather than their viewpoints.
    "clean": ("distinct", "distinct"),
    "one_silent_candidate": ("distinct", "distinct"),
    "dead_participant": ("distinct", "distinct"),
    "replayed_traffic": ("distinct", "distinct"),
    "equal_utilities": ("shared", "equal"),
}


def one_episode(agents: Sequence[str], condition: str, engine: UtilityEngine, exp_cfg: dict,
                rng: random.Random, trial: int) -> List[ArmResult]:
    """Run the same trial inputs through both arms and return their two result rows."""
    geometry, battery_profile = CONDITIONS[condition]
    axes = base_axes(agents, geometry)
    batteries = base_batteries(agents, battery_profile)
    axes, batteries = jitter(agents, rng, axes, batteries,
                             keep_equal=(geometry == "shared" or battery_profile == "equal"))
    utilities = {a: utility_of(engine, axes[a], batteries[a]) for a in agents}

    sender = agents[0]
    quiet = agents[-1]
    order = list(agents)
    rng.shuffle(order)             # arrival order (baseline) and activation order (Exp-13)
    heard_order = [a for a in order if a != sender]        # every candidate, in arrival order
    dup = 1 if condition == "replayed_traffic" else 0
    rows: List[ArmResult] = []

    if condition in ("one_silent_candidate", "dead_participant"):
        heard = [a for a in heard_order if a != quiet]
        base = run_baseline(agents, utilities, sender, condition,
                            arrival_order=heard, duplicate_factor=dup, trial=trial)
        deliver = {a: [o for o in agents if o not in (a, quiet)] for a in agents}
        if condition == "dead_participant":
            deliver[quiet] = []
        exp13 = run_exp13(order, engine, batteries, axes, exp_cfg, condition, deliver=deliver,
                          duplicate_factor=dup,
                          silent=quiet if condition == "dead_participant" else None,
                          trial=trial)
        base.expected_winner = baseline_expected(agents, sender, heard, utilities)
        exp13.expected_winner = expected_exp13(agents, utilities)
        _append_note(exp13, f"{quiet} declared but silent"
                            if condition == "one_silent_candidate"
                            else f"{quiet} is down: publishes nothing, listens to nothing")
        _append_note(base, f"{quiet} never answered")
    else:
        base = run_baseline(agents, utilities, sender, condition,
                            arrival_order=heard_order, duplicate_factor=dup, trial=trial)
        exp13 = run_exp13(order, engine, batteries, axes, exp_cfg, condition,
                          duplicate_factor=dup, trial=trial)
        base.expected_winner = baseline_expected(agents, sender, heard_order, utilities)
        exp13.expected_winner = expected_exp13(agents, utilities)

    _append_note(base, "response arrival order permuted for this trial")
    _append_note(exp13, "participant activation/delivery order permuted for this trial")

    if dup:
        _append_note(base, f"each response replayed x{dup + 1} at receive time")
        _append_note(exp13, f"each belief replayed x{dup + 1} at receive time")

    for row in (base, exp13):
        row.trial = trial
        row.utilities = {a: round(utilities[a], 6) for a in agents}
        row.batteries = batteries
        row.axes_m = axes
        row.arrival_order = order
    rows.extend([base, exp13])
    return rows


def scenario_rows(sizes: Sequence[int], conditions: Sequence[str], engine: UtilityEngine,
                  exp_cfg: dict, trials: int, seed: int) -> Tuple[List[ArmResult], dict]:
    """Every (condition, size, trial) cell, each arm once, from one shared seeded stream.

    Each arm sees the identical utilities, jitter and order for a given trial, so the two rows in a
    cell differ only by allocator.
    """
    rows: List[ArmResult] = []
    utility_maps: set = set()
    orders: set = set()
    for condition in conditions:
        for n in sizes:
            agents = [f"ugv_{i + 1}" for i in range(n)]
            for trial in range(trials):
                rng = random.Random(f"{seed}|{condition}|{n}|{trial}")
                rows.extend(one_episode(agents, condition, engine, exp_cfg, rng, trial))
                utility_maps.add(tuple(sorted(round(u, 6) for u in rows[-1].utilities.values())))
                orders.add(tuple(rows[-1].arrival_order))
    audit = {
        "episodes_per_arm": len(rows) // 2,
        "trials_per_cell": trials,
        "seed": seed,
        "distinct_utility_maps": len(utility_maps),
        "distinct_orders": len(orders),
        "interpretation": ("a count of 1 for either field would mean the sweep varied nothing and "
                           "must not be described as randomised"),
    }
    return rows, audit


def determinism_probe(sizes: Sequence[int], engine: UtilityEngine, exp_cfg: dict) -> List[dict]:
    """Distinct winners across every arrival order with identical utilities for all participants.

    Baseline permutes the response arrival order (the owner's collector list); Exp-13 permutes the
    participant construction/delivery order. Viewpoints and batteries are fixed by identity, so any
    winner change here is an ordering effect, not a scoring effect. Orderings that produce no winner
    are counted, not skipped.
    """
    out = []
    for n in sizes:
        agents = [f"ugv_{i + 1}" for i in range(n)]
        axes = base_axes(agents, "shared")
        bat = base_batteries(agents, "equal")
        utils = {a: utility_of(engine, axes[a], bat[a]) for a in agents}
        sender = agents[0]
        candidates = [a for a in agents if a != sender]

        base_winners: List[Optional[str]] = []
        for order in itertools.permutations(candidates):
            base_winners.append(run_baseline(agents, utils, sender, "equal_utilities",
                                             arrival_order=list(order)).winner)
        exp_winners: List[Optional[str]] = []
        for order in itertools.permutations(agents):
            exp_winners.append(run_exp13(list(order), engine, bat, axes, exp_cfg,
                                         "equal_utilities").winner)

        out.append({
            "participants": n,
            "utilities_equal": len({round(u, 4) for u in utils.values()}) == 1,
            "baseline_permutations": len(base_winners),
            "baseline_distinct_winners": len({w for w in base_winners if w}),
            "baseline_winner_set": sorted({w for w in base_winners if w}),
            "baseline_permutations_without_winner": sum(1 for w in base_winners if w is None),
            "exp13_permutations": len(exp_winners),
            "exp13_distinct_winners": len({w for w in exp_winners if w}),
            "exp13_winner_set": sorted({w for w in exp_winners if w}),
            "exp13_permutations_without_winner": sum(1 for w in exp_winners if w is None),
        })
    return out


# --------------------------------------------------------------------- aggregation

def build_negotiators(agents: Sequence[str], engine: UtilityEngine, battery: Dict[str, float],
                      axes: Dict[str, float], exp_cfg: dict, episode_id: str
                      ) -> Dict[str, CBBANegotiator]:
    """One negotiator per participant, configured from the Exp-13 YAML, identity-keyed geometry."""
    negotiators: Dict[str, CBBANegotiator] = {}
    for agent in agents:
        neg = CBBANegotiator(
            agent_id=agent,
            participant_ids=list(agents),
            max_bundle_size=int(exp_cfg["max_bundle_size"]),
            max_rounds=int(exp_cfg["max_rounds"]),
            convergence_threshold=int(exp_cfg["convergence_threshold"]),
            deadline_s=(None if exp_cfg.get("deadline_s") in (None, "null")
                        else float(exp_cfg["deadline_s"])),
            bid_epsilon=float(exp_cfg["bid_epsilon"]),
            utility_precision=int(exp_cfg["utility_precision"]),
        )
        neg.register_tasks([task_for()], episode_id)
        neg.set_bid_calculator(AgentUtility(engine, axes[agent], battery[agent]))
        neg.set_telemetry(battery[agent])
        negotiators[agent] = neg
    return negotiators


def sequential_activation_probe(sizes: Sequence[int], engine: UtilityEngine, exp_cfg: dict) -> List[dict]:
    """Every participant takes a turn one at a time — the schedule that exposes activation order.

    Why this exists alongside ``determinism_probe``: in lockstep, nobody has a peer belief to ingest
    during round 1, so every agent prices before it merges and an activation-order defect cannot show
    up no matter which order the participants are constructed in. A real adapter publishes and
    receives asynchronously, and that is where a first mover can be adopted wholesale before the
    second mover has priced anything. This probe makes that schedule explicit, so the property is
    measured here and not only asserted by a unit test.
    """
    out = []
    for n in sizes:
        agents = [f"ugv_{i + 1}" for i in range(n)]
        axes = base_axes(agents, "shared")
        bat = base_batteries(agents, "equal")
        utils = {a: utility_of(engine, axes[a], bat[a]) for a in agents}
        max_rounds = int(exp_cfg["max_rounds"])

        winners: List[Optional[str]] = []
        refusals = 0
        for order in itertools.permutations(agents):
            negs = build_negotiators(agents, engine, bat, axes, exp_cfg,
                                     f"seq:n{n}:{''.join(order)}")
            published: Dict[str, object] = {}
            for _ in range(max_rounds):
                if all(neg.is_terminal() for neg in negs.values()):
                    break
                for agent in order:
                    neg = negs[agent]
                    if neg.is_terminal():
                        continue
                    inbox = [m for sender, m in published.items() if sender != agent]
                    _, belief = neg.run_round(inbox)
                    published[agent] = belief
            converged = [neg for neg in negs.values() if neg.is_converged()]
            if len(converged) == len(negs):
                winner = converged[0].winning_agents.get(TASK_ID)
            else:
                winner = None
                refusals += 1
            winners.append(winner)

        out.append({
            "participants": n,
            "utilities_equal": len({round(u, 4) for u in utils.values()}) == 1,
            "activation_orders": len(winners),
            "distinct_winners": len({w for w in winners if w}),
            "winner_set": sorted({w for w in winners if w}),
            "orders_with_no_commitment": refusals,
        })
    return out


def _drive_schedule(negs: Dict[str, CBBANegotiator], order: Sequence[str], mode: str,
                    max_rounds: int) -> None:
    """Run one delivery schedule over already-built negotiators.

    Modes are the shapes a real asynchronous graph produces, and each differs from the previous
    probes by exactly one thing:

    ``lockstep``          every agent's inbox is frozen at the start of the sweep, so no agent sees a
                          belief published in the same sweep. The shape the earlier probes used.
    ``sequential``        each agent sees everything published before it in this sweep.
    ``late_inbox``        the last agent in ``order`` starts after the others have published, but its
                          beliefs are passed through ``run_round``'s inbox, which prices first.
    ``late_delivered``    the same schedule, but the beliefs are handed to the receive path *before*
                          that agent has priced anything. This is F-04's trigger: an adapter's
                          subscriber can fire before the node's first timer tick.
    ``duplicate``         every published belief is delivered twice.
    ``delayed``           a belief becomes visible only one sweep after it was published.
    ``post_convergence``  two extra sweeps are driven after every agent is terminal.
    """
    terminal_extra = 2 if mode == "post_convergence" else 0
    published: Dict[str, object] = {}
    lagged: Dict[str, object] = {}
    snapshot: Dict[str, object] = {}
    late = order[-1]
    late_started = False

    def inbox_for(agent: str) -> List[object]:
        if mode == "lockstep":
            source = snapshot
        elif mode == "delayed":
            source = lagged
        else:
            source = published
        msgs = [m for sender, m in source.items() if sender != agent]
        if mode == "duplicate":
            msgs = msgs + list(msgs)
        return msgs

    for _ in range(max_rounds + terminal_extra):
        if all(neg.is_terminal() for neg in negs.values()) and mode != "post_convergence":
            break
        snapshot = dict(published)
        if mode == "delayed":
            # A belief becomes visible one full sweep after it was published.
            lagged = snapshot
        for agent in order:
            neg = negs[agent]
            if neg.is_terminal() and mode != "post_convergence":
                continue
            msgs = inbox_for(agent)
            if mode == "late_delivered" and agent == late and not late_started:
                for m in msgs:
                    neg.process_consensus_message(m)
                msgs = []
                late_started = True
            _, belief = neg.run_round(msgs)
            published[agent] = belief
    return None


def order_independence_probe(sizes: Sequence[int], engine: UtilityEngine, exp_cfg: dict,
                             seed: int, samples: int = 6) -> List[dict]:
    """Bounded deterministic order coverage through N = 8, with the tested set reported exactly.

    Why this exists: ``determinism_probe`` and ``sequential_activation_probe`` both enumerate
    ``itertools.permutations``, which is 40,320 episodes at N = 8, so ``main`` capped them at N <= 5
    and, with the shipped size list ``2,3,4,6,8``, they only ever ran N = 2, 3 and 4. A scaling claim
    to eight agents therefore had no eight-agent measurement behind it, and neither probe delivered a
    belief before an agent's first round.

    Above N = 5 the orders are sampled from one fixed seed rather than enumerated, and every row
    reports how many schedules were run. No conclusion here is universal: it is a statement about the
    schedules listed, which is what a bounded probe can honestly support.
    """
    out = []
    max_rounds = int(exp_cfg["max_rounds"])
    modes = ("lockstep", "sequential", "late_inbox", "late_delivered", "duplicate", "delayed",
             "post_convergence")
    for n in sizes:
        agents = [f"ugv_{i + 1}" for i in range(n)]
        axes = base_axes(agents, "shared")
        bat = base_batteries(agents, "equal")
        utils = {a: utility_of(engine, axes[a], bat[a]) for a in agents}
        expected = expected_exp13(agents, utils)
        identity = list(agents)

        orders: List[Tuple[str, List[str]]] = [("identity", identity),
                                               ("reverse", list(reversed(identity)))]
        for i in range(n):
            orders.append((f"rotation{i}", identity[i:] + identity[:i]))
        if n > 5:
            rng = random.Random(f"order_independence|{n}|{seed}")
            seen = {tuple(o) for _, o in orders}
            taken = 0
            while taken < samples:
                candidate = rng.sample(identity, n)
                if tuple(candidate) not in seen:
                    seen.add(tuple(candidate))
                    orders.append((f"sampled{taken}", candidate))
                    taken += 1

        rows: List[dict] = []
        for mode in modes:
            for label, order in orders:
                negs = build_negotiators(agents, engine, bat, axes, exp_cfg,
                                         f"oi:{mode}:{label}:n{n}")
                _drive_schedule(negs, order, mode, max_rounds)
                views = {a: neg.winning_agents.get(TASK_ID, "") for a, neg in negs.items()}
                named = sorted(a for a, w in views.items() if w)
                pairs = sum(1 for i, a in enumerate(named)
                            for b in named[i + 1:] if views[a] != views[b])
                converged = [neg for neg in negs.values() if neg.is_converged()]
                winner = (converged[0].winning_agents.get(TASK_ID)
                          if len(converged) == len(negs) else None)
                rows.append({"mode": mode, "order": label, "winner": winner,
                             "disagreeing_pairs": pairs,
                             "dual_claim": any("DUAL_CLAIM" in neg.convergence_status.name
                                               for neg in negs.values())})

        winners = sorted({r["winner"] for r in rows if r["winner"]})
        out.append({
            "participants": n,
            "utilities_equal": len({round(u, 4) for u in utils.values()}) == 1,
            "expected_winner": expected,
            "exhaustive": False,
            "modes": list(modes),
            "orders_per_mode": len(orders),
            "schedules_executed": len(rows),
            "sampling": ("fixed seed enumeration" if n <= 5 else
                         f"seeded sample: {samples} permutations + {n} rotations + identity/reverse"),
            "distinct_winners": len(winners),
            "winner_set": winners,
            "honours_documented_tie_rule": (winners == [expected] if winners else None),
            "schedules_without_commitment": sum(1 for r in rows if r["winner"] is None),
            "schedules_with_cross_agent_disagreement": sum(
                1 for r in rows if r["disagreeing_pairs"] > 0),
            "schedules_with_dual_claim": sum(1 for r in rows if r["dual_claim"]),
            "cross_agent_disagreement_total": sum(r["disagreeing_pairs"] for r in rows),
        })
    return out


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def aggregate(rows: Sequence[ArmResult], host_tick_s: Optional[float]) -> Dict[str, dict]:
    """One aggregate per arm, at three scopes: overall, per condition, per participant count."""
    scopes: Dict[Tuple[str, str], List[ArmResult]] = {}
    for r in rows:
        for key in ((r.arm, "__all__"), (r.arm, f"condition:{r.condition}"),
                    (r.arm, f"N={r.participants}")):
            scopes.setdefault(key, []).append(r)

    out: Dict[str, dict] = {}
    for (arm, scope), group in scopes.items():
        decided = [r for r in group if r.outcome_class == "CONVERGED"]
        times = [r.convergence_time_s_max for r in group if r.convergence_time_s_max is not None]
        # An interval at or below one clock tick is not a measurement of duration: it says only
        # that the episode finished without the clock advancing. Keep those apart from the rest.
        sub_tick = [t for t in times if host_tick_s is not None and t <= host_tick_s]
        resolved = [t for t in times if t > (host_tick_s or 0.0)]
        entry = {
            "episodes_executed": len(group),
            "converged_episodes": len(decided),
            "non_converged_episodes": len(group) - len(decided),
            "undecided_episodes": sum(1 for r in group if r.outcome_class == "UNDECIDED"),
            "stalled_episodes": sum(1 for r in group if r.outcome_class == "STALLED_NOT_TERMINAL"),
            "failed_episodes": sum(1 for r in group if r.outcome_class == "FAILED"),
            "app_messages_sent": _quantiles([r.app_messages_sent for r in group]),
            "app_messages_ingested": _quantiles([r.app_messages_ingested for r in group]),
            "convergence_rounds": _quantiles([r.rounds for r in group]),
            "convergence_time_s": _quantiles(times),
            "convergence_time_none_episodes": len(group) - len(times),
            "convergence_time_at_or_below_host_tick": len(sub_tick),
            "convergence_time_resolved_above_tick": _quantiles(resolved),
            "mean_refused_ingress": round(statistics.mean([r.refused_ingress for r in group]), 3),
            "dual_owner_violations_total": sum(r.dual_owner_violations for r in group),
            "dual_owner_basis": "local: each agent's own bundle against its own winning vector",
            "cross_agent_pairs_measurable_rows": sum(
                1 for r in group if r.cross_agent_disagreeing_pairs is not None),
            "cross_agent_disagreeing_pairs_total": (
                sum(r.cross_agent_disagreeing_pairs for r in group
                    if r.cross_agent_disagreeing_pairs is not None)
                if any(r.cross_agent_disagreeing_pairs is not None for r in group) else None),
            "cross_agent_disagreement_episodes": (
                sum(1 for r in group if (r.cross_agent_disagreeing_pairs or 0) > 0)
                if any(r.cross_agent_disagreeing_pairs is not None for r in group) else None),
            "committed_dual_ownership_total": None,
            "false_convergence_total": sum(r.false_convergence for r in group),
            "agreed_by_all_episodes": sum(1 for r in group if r.agreed_by_all),
            "winner_matched_expected": sum(1 for r in group
                                           if r.winner and r.winner == r.expected_winner),
        }
        out.setdefault(arm, {})[scope] = entry
    return out


REQUIRED_ROWS = [
    ("Episodes executed", "episodes_executed", "actual trial x cell count, no row dropped"),
    ("Converged episodes", "converged_episodes",
     "arm committed an allocation; derived from its own end state"),
    ("Non-converged episodes", "non_converged_episodes", "reported, never excluded"),
    ("Undecided episodes", "undecided_episodes",
     "terminated or stalled with no winner (baseline zero-ACCEPT abort; exp13 NO_WINNER / "
     "MAX_ROUNDS / TIMEOUT)"),
    ("Stalled (non-terminal)", "stalled_episodes", "sweep stopped before the arm terminated"),
    ("Failed episodes", "failed_episodes", "safety class violated (DUAL_CLAIM)"),
    ("All-participant-agreed episodes", "agreed_by_all_episodes",
     "every declared participant names the same winner; structurally unreachable for the baseline, "
     "whose candidates hold no decision state"),
    ("Mean application messages", ("app_messages_sent", "mean"),
     "application protocol messages produced per episode; NOT DDS packets"),
    ("Median application messages", ("app_messages_sent", "median"), "same boundary"),
    ("Mean ingested messages", ("app_messages_ingested", "mean"),
     "inbound application messages actually examined at receive time"),
    ("Median ingested messages", ("app_messages_ingested", "median"), "same boundary"),
    ("Mean convergence rounds", ("convergence_rounds", "mean"),
     "rounds until the arm stopped; baseline's 1 is definitional, not observed"),
    ("Median convergence rounds", ("convergence_rounds", "median"), "same stopping rule per arm"),
    ("Mean convergence time", ("convergence_time_s", "mean"),
     "single-process monotonic interval; NOT a speed claim, see the two rows below. The baseline "
     "model records no interval at all, so NOT MEASURED there means 'nothing to average', not 'zero'"),
    ("Median convergence time", ("convergence_time_s", "median"), "single-process monotonic interval"),
    ("Episodes with no convergence time", "convergence_time_none_episodes",
     "agreement never established, so no interval exists; not counted as 0"),
    ("Intervals at or below one host tick", "convergence_time_at_or_below_host_tick",
     "the clock could not resolve these; they are duration-blind, not fast. Baseline shows 0 only "
     "because it has no intervals to classify"),
    ("Mean time over resolvable intervals only",
     ("convergence_time_resolved_above_tick", "mean"),
     "subset whose interval exceeded one tick; n reported in the JSON"),
    ("Mean refused ingress", "mean_refused_ingress",
     "inbound payloads classified and dropped (baseline: definitionally 0)"),
    ("Local bundle/winner self-consistency violations", "dual_owner_violations_total",
     "F-02: per-agent check that an agent's own bundle is consistent with its OWN winning vector. "
     "Renamed from 'Dual-owner violations', which over-claimed: it cannot observe two different "
     "agents each believing they own the same target. Zero is not a dual-ownership proof"),
    ("Cross-agent winner disagreement (episodes)", "cross_agent_disagreement_episodes",
     "episodes in which two agents name different winners for the same target; NOT MEASURED for "
     "the baseline model, whose candidates hold no decision state, so no cross-agent view exists"),
    ("Cross-agent disagreement (agent pairs)", "cross_agent_disagreeing_pairs_total",
     "disagreeing unordered agent pairs summed over episodes; measured only where >=2 agents hold "
     "a winner view"),
    ("Rows where cross-agent disagreement was measurable", "cross_agent_pairs_measurable_rows",
     "denominator for the two rows above; a zero disagreement over zero measurable rows is not a "
     "pass"),
    ("Committed dual-ownership violations", "committed_dual_ownership_total",
     "NOT MEASURED in either arm by construction: a committed owner exists only on the Request -> "
     "Response -> Confirm -> ConfirmAck path, which neither model arm executes. It is observable in "
     "the Tier-N node tests and on a real graph, not here"),
    ("False convergence", "false_convergence_total", "converged flag without AGREED"),
]


def render_table(agg: Dict[str, dict]) -> List[str]:
    lines = ["| Metric | Baseline CNP / Request-Response | Experiment 13 | Definition / Notes |",
             "|---|---:|---:|---|"]
    for label, key, note in REQUIRED_ROWS:
        cells = []
        for arm in ("baseline", "exp13"):
            entry = agg.get(arm, {}).get("__all__", {})
            if isinstance(key, tuple):
                value = (entry.get(key[0]) or {}).get(key[1])
            else:
                value = entry.get(key)
            cells.append(NOT_MEASURED if value is None else f"{value}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {note} |")
    return lines


# --------------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default="2,3,4,6,8",
                        help="comma-separated participant counts (default 2,3,4,6,8)")
    parser.add_argument("--conditions",
                        default="clean,one_silent_candidate,dead_participant,replayed_traffic,"
                                "equal_utilities")
    parser.add_argument("--trials", type=int, default=12,
                        help="independent episodes per (condition, size) cell (default 12)")
    parser.add_argument("--seed", default="20261020",
                        help="base seed for the trial stream; per-cell streams are derived from it")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"directory for JSON + trials + table output (default {DEFAULT_OUT})")
    parser.add_argument("--order-sizes", default="",
                        help="participant counts for the bounded order-independence probe; empty "
                             "(default) uses every --sizes value, including N > 5 where orders are "
                             "sampled from the seed instead of enumerated")
    parser.add_argument("--order-samples", type=int, default=6,
                        help="seeded permutations per delivery mode at N > 5 (default 6)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    conditions = [c for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        print(f"unknown condition(s): {unknown}", file=sys.stderr)
        return 2

    exp_cfg, baseline_cfg = load_configs()
    engine, weights_prov = build_engine(baseline_cfg)
    host_tick_s = measure_host_tick_s()

    nego_logger = logging.getLogger("edgesync_coordination.negotiation.cbba_negotiator")
    previous_level = nego_logger.level
    nego_logger.setLevel(logging.ERROR)
    started = time.monotonic()
    rows, randomisation = scenario_rows(sizes, conditions, engine, exp_cfg,
                                        args.trials, args.seed)
    # The two exhaustive probes stay capped: 8! permutations is 40,320 episodes each. F-03 coverage
    # to N = 8 comes from order_independence_probe, which samples and reports its own denominator.
    determinism = determinism_probe([s for s in sizes if s <= 5], engine, exp_cfg)
    sequential = sequential_activation_probe([s for s in sizes if s <= 5], engine, exp_cfg)
    order_sizes = ([int(s) for s in args.order_sizes.split(",") if s.strip()] or sizes)
    order_independence = order_independence_probe(order_sizes, engine, exp_cfg, args.seed,
                                                 samples=args.order_samples)
    sim_wallclock_s = time.monotonic() - started
    nego_logger.setLevel(previous_level)

    agg = aggregate(rows, host_tick_s)
    timing_note = (
        "single-process interval, not a robot or network latency; the host monotonic tick measured "
        f"in this run is {host_tick_s}s, so any mean/median at or below one tick is clock "
        "quantisation, not a speed difference. Rounds are the primary convergence measure."
        if host_tick_s else "host tick unmeasurable in this run; rounds are the only usable "
                            "convergence measure")
    for scopes in agg.values():
        for entry in scopes.values():
            entry["convergence_time_note"] = timing_note

    consumed = {
        "max_bundle_size": "passed to every negotiator",
        "max_rounds": "passed to every negotiator and used as the sweep bound",
        "convergence_threshold": "passed to every negotiator",
        "bid_epsilon": "passed to every negotiator",
        "utility_precision": "passed to every negotiator",
        "deadline_s": "passed to every negotiator (None or 'null' means round-bound only)",
        "exp_id": "written into this artefact only; no ROS 2 consumer exists (approval A2)",
        "negotiation_protocol": "written into this artefact only; no selector exists in "
                                "coordination_node.py (approval A1)",
        "algorithm_variant": "written into this artefact only; never a selector",
        "swarm_participant_count": "NOT used as an input — the sweep deliberately exceeds it, see "
                                   "config_consistency",
        "target_count": "NOT used as an input — every cell uses 1 target, which matches it",
        "battery_pct": "NOT used — telemetry is supplied per trial, since a stand-in battery must "
                       "not decide an allocation",
        "message_count_unit": "definition followed by this harness: beliefs counted at publish",
        "convergence_time_unit": "definition followed: s, monotonic, single process, null when no "
                                 "agreement",
    }
    config_consistency = {
        "keys_consumed_by_harness": sorted(k for k in exp_cfg if k in
                                           ("max_bundle_size", "max_rounds", "convergence_threshold",
                                            "bid_epsilon", "utility_precision", "deadline_s")),
        "keys_not_consumed": [k for k in sorted(exp_cfg) if k not in
                              ("max_bundle_size", "max_rounds", "convergence_threshold",
                               "bid_epsilon", "utility_precision", "deadline_s")],
        "key_accounting": consumed,
        "declared_vs_sweeped_participants": {
            "swarm_participant_count": exp_cfg.get("swarm_participant_count"),
            "sweep_sizes": sizes,
            "note": ("the design gate fixed 2 declared participants; the sweep deliberately runs "
                     "N > 2 to probe scaling, so these rows are NOT the Tier-U configuration the "
                     "YAML declares and must not be reported as such"),
        },
        "target_count": {"declared": exp_cfg.get("target_count"), "used": 1},
    }

    disagree_cases = [
        ("all agree", {"ugv_1": "ugv_1", "ugv_2": "ugv_1"}, 0),
        ("one dissents of four", {f"ugv_{i}": ("ugv_9" if i == 4 else "ugv_1")
                                  for i in range(1, 5)}, 3),
        ("undecided agents are not counted", {"ugv_1": "ugv_1", "ugv_2": "", "ugv_3": "ugv_3"}, 1),
        ("nobody has a winner", {"ugv_1": "", "ugv_2": ""}, 0),
    ]
    metric_selfcheck = {
        "purpose": ("shows the cross-agent counter can be non-zero. Without it a zero in the table "
                    "cannot be told apart from a counter that never fires, which is the defect F-02 "
                    "recorded for the pre-existing local self-consistency metric"),
        "cross_agent_disagreement_pairs": [
            {"case": case, "views": views, "expected": expected,
             "actual": cross_agent_disagreement_pairs(views)}
            for case, views, expected in disagree_cases],
    }
    metric_selfcheck["all_passed"] = all(
        c["expected"] == c["actual"] for c in metric_selfcheck["cross_agent_disagreement_pairs"])

    payload = {
        "exp_id": exp_cfg.get("exp_id"),
        "negotiation_protocol": exp_cfg.get("negotiation_protocol"),
        "algorithm_variant": exp_cfg.get("algorithm_variant"),
        "evidence_tier": EVIDENCE_TIER,
        "measurement_scope": "unit simulation; one process; no ROS graph; no network; no Jetson",
        "message_layer": MESSAGE_LAYER,
        "metric_definitions": {
            "app_messages_sent": ("ONE application-level protocol message produced by a participant "
                                 "and handed to its transport inside the episode. baseline: 1 request "
                                 "+ 1 ACCEPT per answering candidate, counted at the send. exp13: 1 "
                                 "belief per non-censored round per active agent, read from each "
                                 "negotiator's own beliefs_published counter. Transport duplicates "
                                 "re-offered at receive time add receipts, never sends."),
            "app_messages_ingested": ("ONE inbound application-level message a participant examined "
                                      "at receive time. baseline: every payload the collector loop "
                                      "read (appended unconditionally, as at :506). exp13: the sum of "
                                      "the real inbox lengths passed to run_round(). Neither value is "
                                      "derived from an arithmetic formula about what the arm would "
                                      "have exchanged."),
            "dds_or_network_packets": f"{NOT_MEASURED} — no transport exists in this process",
            "rounds": "arm-specific: baseline 1 is definitional (single window); exp13 is the highest "
                      "observed per-participant round counter. Never converted between arms.",
            "decided": "derived from the arm's end state: baseline owner COMMITTED with a winner; "
                       "exp13 every declared participant terminal AND converged",
            "agreed_by_all": "every declared participant names the same winner, counted from each "
                             "participant's own recorded vector",
            "outcome_class": "one rule for both arms: CONVERGED / UNDECIDED / STALLED_NOT_TERMINAL "
                             "/ FAILED, from observed end state",
            "selection_list_size": ("per-arm and NOT comparable: baseline = receipts the sort keyed "
                                    "on; exp13 = participants whose own state reported converged. "
                                    "It is never used as a message count"),
            "expected_winner": ("per-arm rule, NOT a shared oracle. baseline: 'highest utility among "
                                "those heard, ties keep arrival order' — which is the model's own "
                                "rule, so winner_matched_expected is tautological for the baseline "
                                "and is not evidence about it. exp13: 'highest utility, ties by "
                                "lowest agent id', an independent statement of the module's "
                                "documented rule, so its match rate is evidence"),
            "sequential_activation_probe": ("distinct Exp-13 winners over every one-at-a-time "
                                            "activation order with identical utilities. Added because "
                                            "the lockstep determinism probe cannot see an "
                                            "activation-order defect at all"),
            "order_independence_probe": ("bounded N=2..8 coverage over seven delivery schedules "
                                         "(lockstep, sequential, late-inbox, late-delivered-before-"
                                         "first-round, duplicate, delayed, post-convergence). At "
                                         "N > 5 orders are seeded-sampled, so the claim is bounded by "
                                         "the reported schedules_executed and is NOT universal; the "
                                         "two exhaustive probes above stop at N <= 4 (8! permutations "
                                         "are not enumerable here)"),
            "convergence_time_s": ("slowest converged participant's monotonic interval from first "
                                   "belief to confirmed agreement, inside this one process; None when "
                                   "agreement never happened. NOT a robot or network latency."),
            "host_tick_s": host_tick_s,
            "milliseconds": "reported only as measured single-process intervals with the tick "
                            "disclosed; no millisecond figure here is a protocol latency",
        },
        "naming_change": ("this revision renamed frames_sent -> app_messages_sent and "
                          "ingestion_attempts -> app_messages_ingested, and replaced the previous "
                          "exp13 ingestion formula rounds x (N-1) with counted inboxes"),
        "hardcoded_fields_removed": [
            "baseline res.decided = True",
            "baseline res.converged_agents = 1",
            "exp13 ingestion_attempts = current_round * (N - 1)",
        ],
        "non_equivalences_disclosed": [
            "one-sided baseline vs symmetric Exp-13: only the baseline's owner holds a decision",
            "stopping rules differ by construction (fixed window vs all-agreement / round bound / "
            "deadline) and are not convertible",
            "the baseline excludes the owner from its candidate pool; Exp-13 lets it bid and retain",
            "candidate refusal counts are definitional for the baseline, not observed, because the "
            "modelled collector has no admission test",
        ],
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "monotonic_clock_info": str(time.get_clock_info("monotonic")),
            "monotonic_tick_s_measured_this_run": host_tick_s,
            "simulator_wallclock_s": round(sim_wallclock_s, 6),
            "wallclock_note": ("simulator cost only, not a protocol latency; the tick above bounds "
                               "every interval reported"),
        },
        "trial_design": randomisation,
        "utility_weights": weights_prov,
        "exp13_config": {k: exp_cfg[k] for k in sorted(exp_cfg)},
        "baseline_config": {
            "t_neg_timeout_ms": baseline_cfg.get("t_neg_timeout_ms", 80),
            "selection_source_lines": "coordination_node.py:441-545",
        },
        "config_consistency": config_consistency,
        "aggregate": agg,
        "results": [asdict(r) for r in rows],
        "determinism_probe": determinism,
        "sequential_activation_probe": sequential,
        "order_independence_probe": order_independence,
        "metric_selfcheck": metric_selfcheck,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp13_unit_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    trials_path = out_dir / "exp13_unit_benchmark_trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True, default=str) + "\n")

    table_path = out_dir / "exp13_unit_benchmark_table.md"
    lines: List[str] = [f"# Exp-13 unit-simulation comparison — {EVIDENCE_TIER}", "",
                        f"exp_id={exp_cfg.get('exp_id')} protocol={exp_cfg.get('negotiation_protocol')} "
                        f"variant={exp_cfg.get('algorithm_variant')}",
                        f"Episodes per arm: {randomisation['episodes_per_arm']} "
                        f"({len(conditions)} conditions x {len(sizes)} sizes x {args.trials} trials), "
                        f"seed base {args.seed}.",
                        f"Message layer: {MESSAGE_LAYER}.",
                        f"Host monotonic tick measured this run: {host_tick_s}s.",
                        "Any millisecond figure below is a single-process interval, not a robot or "
                        "network latency; differences at or below one tick are clock quantisation.",
                        ""]
    lines.extend(render_table(agg))
    lines.append("")
    lines.append("## Per condition")
    for condition in conditions:
        lines.append(f"### {condition}")
        lines.extend(render_table({arm: {c: agg[arm][f"condition:{condition}"] for c in ["__all__"]}
                                   for arm in ("baseline", "exp13")}))
        lines.append("")
    lines.append("## Per participant count")
    for n in sizes:
        lines.append(f"### N = {n}")
        lines.extend(render_table({arm: {c: agg[arm][f"N={n}"] for c in ["__all__"]}
                                   for arm in ("baseline", "exp13")}))
        lines.append("")
    lines.append("## Determinism probe (all utilities equal, every order enumerated)")
    for p in determinism:
        lines.append(f"* N={p['participants']} (utilities equal: {p['utilities_equal']}): baseline "
                     f"{p['baseline_distinct_winners']} distinct winner(s) over "
                     f"{p['baseline_permutations']} response orders {p['baseline_winner_set']} "
                     f"({p['baseline_permutations_without_winner']} orderings produced none) | "
                     f"Exp-13 {p['exp13_distinct_winners']} over {p['exp13_permutations']} "
                     f"participant orders {p['exp13_winner_set']} "
                     f"({p['exp13_permutations_without_winner']} produced none)")
    lines.append("")
    lines.append("## Sequential activation probe (equal utilities, one participant per turn)")
    lines.append("")
    lines.append("Lockstep cannot expose an activation-order defect: in round 1 nobody has a peer "
                 "belief to ingest, so every agent prices before it merges. This probe turns "
                 "participants one at a time, which is what an asynchronous adapter really does.")
    for p in sequential:
        lines.append(f"* N={p['participants']} (utilities equal: {p['utilities_equal']}): Exp-13 "
                     f"{p['distinct_winners']} distinct winner(s) over "
                     f"{p['activation_orders']} activation orders {p['winner_set']} "
                     f"({p['orders_with_no_commitment']} orders ended without a commitment)")
    lines.append("")
    lines.append("Order-independence, bounded and seeded (F-03). Seven delivery schedules per agent "
                 "count; at N > 5 the acting orders are sampled from the run seed, so each row "
                 "states how many schedules were actually executed.")
    for p in order_independence:
        lines.append(f"* N={p['participants']} ({p['schedules_executed']} schedules, "
                     f"{p['modes'][0]}..{p['modes'][-1]}, {p['orders_per_mode']} orders each; "
                     f"{p['sampling']}): {p['distinct_winners']} distinct winner(s) "
                     f"{p['winner_set']} against the documented rule "
                     f"{p['expected_winner']} (honoured: {p['honours_documented_tie_rule']}); "
                     f"{p['schedules_without_commitment']} without a commitment, "
                     f"{p['schedules_with_cross_agent_disagreement']} with cross-agent disagreement, "
                     f"{p['schedules_with_dual_claim']} DUAL_CLAIM")
    lines.append("")
    lines.append("## Metric self-check (F-02)")
    lines.append(metric_selfcheck["purpose"] + ".")
    for c in metric_selfcheck["cross_agent_disagreement_pairs"]:
        lines.append(f"* {c['case']}: views {c['views']} -> expected {c['expected']}, "
                     f"measured {c['actual']}")
    lines.append(f"* all cases passed: {metric_selfcheck['all_passed']}")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    width = 138
    print("=" * width)
    print(f"EdgeSync Exp-13 unit-simulation comparison — {EVIDENCE_TIER}")
    print(f"exp_id={payload['exp_id']} protocol={payload['negotiation_protocol']} "
          f"variant={payload['algorithm_variant']}")
    print(f"Message layer: {MESSAGE_LAYER}")
    print(f"Episodes/arm={randomisation['episodes_per_arm']}  trials/cell={args.trials}  "
          f"distinct utility maps={randomisation['distinct_utility_maps']}  "
          f"distinct orders={randomisation['distinct_orders']}")
    print(f"Host monotonic tick measured now = {host_tick_s}s; intervals at or below that are "
          f"quantisation, not speed")
    print("=" * width)
    header = (f"{'condition':22} {'N':>2} {'arm':7} {'msgs':>6} {'max/ag':>6} {'ingest':>7} "
              f"{'refused':>7} {'rounds':>6} {'term':>5} {'conv':>5} {'decided':>7} "
              f"{'allAgree':>8} {'local':>5} {'xPair':>5} {'fConv':>5} {'conv_s':>8} "
              f"{'winner':>7} {'expect':>7}")
    print(header)
    print("-" * width)
    for r in rows:
        conv = "None" if r.convergence_time_s_max is None else f"{r.convergence_time_s_max:.6f}"
        xpair = "-" if r.cross_agent_disagreeing_pairs is None \
            else str(r.cross_agent_disagreeing_pairs)
        print(f"{r.condition:22} {r.participants:>2} {r.arm:7} {r.app_messages_sent:>6} "
              f"{r.app_messages_sent_max_per_agent:>6} {r.app_messages_ingested:>7} "
              f"{r.refused_ingress:>7} {r.rounds:>6} {r.participants_at_terminal_state:>5} "
              f"{r.converged_agents:>5} {str(r.decided):>7} {str(r.agreed_by_all):>8} "
              f"{r.dual_owner_violations:>5} {xpair:>5} {r.false_convergence:>5} {conv:>8} "
              f"{str(r.winner):>7} {str(r.expected_winner):>7}")
    print("-" * width)
    print("\nAGGREGATE (all cells, all trials)")
    for arm in ("baseline", "exp13"):
        a = agg[arm]["__all__"]
        print(f"  {arm:9} episodes={a['episodes_executed']} converged={a['converged_episodes']} "
              f"non-converged={a['non_converged_episodes']} (undecided={a['undecided_episodes']}, "
              f"stalled={a['stalled_episodes']}, failed={a['failed_episodes']})")
        print(f"  {arm:9} msgs mean={a['app_messages_sent']['mean']} "
              f"median={a['app_messages_sent']['median']} | ingested "
              f"mean={a['app_messages_ingested']['mean']} | rounds "
              f"mean={a['convergence_rounds']['mean']} median={a['convergence_rounds']['median']} | "
              f"time mean={a['convergence_time_s']['mean']} "
              f"median={a['convergence_time_s']['median']} (n={a['convergence_time_s']['n']}, "
              f"no-interval episodes={a['convergence_time_none_episodes']}, at/below tick="
              f"{a['convergence_time_at_or_below_host_tick']}) -> NOT A SPEED CLAIM")
    print("\nDETERMINISM")
    for p in determinism:
        print(f"  lockstep    N={p['participants']} (equal utilities: {p['utilities_equal']}): "
              f"baseline {p['baseline_distinct_winners']} distinct over "
              f"{p['baseline_permutations']} orders {p['baseline_winner_set']} | exp13 "
              f"{p['exp13_distinct_winners']} over {p['exp13_permutations']} orders "
              f"{p['exp13_winner_set']}")
    for p in sequential:
        print(f"  sequential  N={p['participants']} (equal utilities: {p['utilities_equal']}): exp13 "
              f"{p['distinct_winners']} distinct winner(s) over {p['activation_orders']} activation "
              f"orders {p['winner_set']} ({p['orders_with_no_commitment']} ordered no commitment)")
    for p in order_independence:
        print(f"  order-indep N={p['participants']:>2} {p['schedules_executed']:>4} schedules "
              f"({p['sampling']}): winners {p['winner_set']} vs documented {p['expected_winner']} "
              f"-> honoured={p['honours_documented_tie_rule']} "
              f"no-commitment={p['schedules_without_commitment']} "
              f"x-disagreement={p['schedules_with_cross_agent_disagreement']} "
              f"dual_claim={p['schedules_with_dual_claim']}")
    print("\nMETRIC SELF-CHECK (F-02)")
    for c in metric_selfcheck["cross_agent_disagreement_pairs"]:
        print(f"  {c['case']:35} expected={c['expected']} measured={c['actual']}")
    print(f"  all_passed={metric_selfcheck['all_passed']}")
    if not metric_selfcheck["all_passed"]:
        print("METRIC SELF-CHECK FAILED: a cross-agent counter that cannot be driven non-zero "
              "reports a false zero, which is the defect F-02 records", file=sys.stderr)
        return 3
    print(f"\nwrote {json_path}")
    print(f"wrote {trials_path}")
    print(f"wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
