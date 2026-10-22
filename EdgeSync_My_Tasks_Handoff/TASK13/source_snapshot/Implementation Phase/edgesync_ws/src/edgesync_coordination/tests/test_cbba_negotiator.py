"""Unit tests for the Experiment 13 CBAA (L = 1) negotiator.

Deliberately **not** dependent on ``rclpy`` or ``edgesync_msgs``: the negotiation module imports no
ROS package by design, so these tests run off a ROS graph on a host with no ROS 2 installation.
That is a unit-level result and is reported as such — it is not a multi-robot or Jetson measurement.

Verifies the 14 cases required by the Experiment 13 execution brief, plus the defects found in the
recovered implementation that the repaired semantics must not re-create:

Consensus behaviour
1. Normal consensus/convergence for multiple participants (2 and 3 agents).
2. Conflicting bids/beliefs resolve to exactly one holder.
3. Two participants referring to different ``target_id`` values cannot agree.
4. Duplicate messages are rejected and counted, and do not inflate the message tally.
5. Stale/out-of-order messages are rejected; the outcome is arrival-order independent.
6. Missing or invalid fields are classified, counted, and never raise into the receive path.
7. Participant silence and a passed deadline censor the episode; neither yields agreement.
8. Deterministic conflict resolution (equal bids tie-break by agent id) and repeatability.
9. Reset between independent handoff attempts leaves no cross-episode state.
10. No false agreement and no dual ownership (bundle/winner consistency invariant).
11. Termination is bounded by ``max_rounds`` under the documented assumptions.
12. ``exp13_cbba.yaml`` selects ``cbba`` and ``EXP13`` and feeds the constructor.
13. Strategy interface/selector compatibility surface.
14. Baseline protocol behaviour is unchanged (``NegotiationManager`` admission + launch config set).

Recovered-implementation defects (DEF-nn refers to ES-EXP13-TECHNICAL-DEFECT-REGISTER-REV04)
D1. No hardcoded battery stand-in decides an allocation (DEF-01).
D2. No fabricated fallback bid: abstention is the absence of a bid (DEF-02).
D3. No exception after a terminal state (DEF-03).
D4. Tie-break is reachable, so two agents cannot both keep the task (DEF-04).
D5. Same-round replay cannot be re-applied (DEF-05).
D6. The variant is labelled CBAA, and ``L > 1`` is refused rather than faked (DEF-06, DEF-07).
D7. Agreement is content-based, not a round-number comparison (DEF-08).
D8. ``is_converged()`` is false for MAX_ROUNDS / TIMEOUT / LOCAL_STABLE / ABORTED (DEF-09).
D9. Malformed inputs are counted, not silently swallowed (DEF-10).
"""
import itertools
import json
import random
import time
from pathlib import Path

import pytest

from edgesync_coordination.negotiation import (
    ALGORITHM_VARIANT,
    BELIEF_SCHEMA_VERSION,
    MIN_AGREEMENT_ROUND,
    PROTOCOL_ID,
    CBBANegotiator,
    ConsensusMessage,
    ConsensusStatus,
    RejectionClass,
    TaskInfo,
)
from edgesync_coordination.protocol.negotiation_manager import NegotiationManager


# ------------------------------------------------------------------ helpers

PKG_ROOT = Path(__file__).resolve().parents[1]
WS_SRC = PKG_ROOT.parent
EXP13_CONFIG = WS_SRC / "edgesync_bringup" / "config" / "exp13_cbba.yaml"
COORDINATION_NODE_PY = PKG_ROOT / "edgesync_coordination" / "nodes" / "coordination_node.py"
BRINGUP_LAUNCH_PY = WS_SRC / "edgesync_bringup" / "launch" / "robot_bringup.launch.py"

EPISODE = "ep_exp13"


def make_tasks(n=1):
    """One task slot per target; positions are distinct so a per-task bid is addressable."""
    return [
        TaskInfo(task_id=f"task_{i}", target_id=f"target_{i}",
                 position=(float(i) + 1.0, 0.0, 5.0), tto_safe_s=2.0, sender_id="ugv_1")
        for i in range(n)
    ]


class FixedUtility:
    """Deterministic stand-in for UtilityEngine with the same call signature.

    Keyed by ``point_camera[0]`` so a test can say "this agent rates task at axis x with utility u".
    Records the battery value it was handed, which is what D1 asserts against.
    """

    def __init__(self, by_axis, default=0.0):
        self.by_axis = dict(by_axis)
        self.default = default
        self.calls = 0
        self.seen_battery = []

    def compute_receiver_utility(self, point_camera=None, battery_pct=None, distance=None):
        self.calls += 1
        self.seen_battery.append(battery_pct)
        u = self.by_axis.get(float(point_camera[0]), self.default)
        return u, u, u, u


class BrokenUtility:
    """Calculator that raises, e.g. because a telemetry topic never arrived."""

    def compute_receiver_utility(self, point_camera=None, battery_pct=None, distance=None):
        raise RuntimeError("no telemetry")


class ManualClock:
    """Injectable clock: reading it never advances time, so intervals are exact."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


def make_agent(agent_id, peers, utility=None, battery=90.0, n_tasks=1, **kwargs):
    kwargs.setdefault("max_rounds", 4)
    neg = CBBANegotiator(agent_id=agent_id, participant_ids=[agent_id] + list(peers), **kwargs)
    neg.register_tasks(make_tasks(n_tasks), EPISODE)
    if utility is not None:
        neg.set_bid_calculator(utility)
    if battery is not None:
        neg.set_telemetry(battery)
    return neg


def belief_of(neg, round_number=None, **overrides):
    """Snapshot a negotiator's current belief via public accessors, with optional corruption.

    Uses the non-counting path so a test fixture never inflates ``beliefs_published``.
    """
    values = {
        "sender_agent_id": neg.agent_id,
        "episode_id": neg.episode_id,
        "round_number": neg.current_round if round_number is None else round_number,
        "winning_bids": neg.winning_bids,
        "winning_agents": neg.winning_agents,
        "bundle": neg.bundle,
        "task_ids": neg.task_ids,
        "target_ids": neg.target_ids,
        "ownership_epoch": neg.ownership_epoch,
        "schema_version": BELIEF_SCHEMA_VERSION,
        "belief_id": f"{neg.episode_id}:{neg.agent_id}:{neg.current_round}",
    }
    values.update(overrides)
    return ConsensusMessage(**values)


def lockstep_round(agents, inflight):
    """One synchronous broadcast round: each agent ingests every *other* agent's last belief.

    Own belief is filtered out because a broadcast loop returns it to the sender; the SELF rejection
    path is exercised separately rather than being left to this driver.
    """
    out = {}
    results = {}
    for neg in agents:
        inbox = [m for sender, m in inflight.items() if sender != neg.agent_id]
        changed, msg = neg.run_round(inbox)
        out[neg.agent_id] = msg
        results[neg.agent_id] = changed
    return out, results


def run_to_completion(agents, rounds=8):
    """Drive agents in lockstep until all are terminal or the round budget is spent."""
    inflight = {}
    executed = 0
    for _ in range(rounds):
        if all(n.is_terminal() for n in agents):
            break
        inflight, _ = lockstep_round(agents, inflight)
        executed += 1
    return executed


def outcome(agents):
    return {n.agent_id: n.convergence_status.name for n in agents}


def assert_single_owner(agents, task_id):
    """Every agent's belief must name the same single holder, and holders must match bundles."""
    winners = {n.winning_agents.get(task_id, "") for n in agents}
    winners.discard("")
    assert len(winners) <= 1, f"dual ownership: {winners}"
    for neg in agents:
        for held in neg.bundle:
            assert neg.winning_agents.get(held, "") in ("", neg.agent_id), (
                f"{neg.agent_id} bundles {held} but believes {neg.winning_agents.get(held)} owns it"
            )


# ------------------------------------------------------------------ 1. normal convergence

def test_two_participant_consensus_converges_to_highest_bid():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.65}))

    run_to_completion([a, b])

    assert a.is_converged() is True
    assert b.is_converged() is True
    assert a.convergence_status is ConsensusStatus.AGREED
    # ugv_1 bids 0.80 > 0.65, so the higher utility agent must keep the target
    assert a.get_assignment() == {"task_0": "ugv_1"}
    assert b.get_assignment() == {"task_0": "ugv_1"}
    assert b.bundle == []          # outbid agent released its claim
    assert a.bundle == ["task_0"]
    assert a.get_target_assignment() == {"target_0": "ugv_1"}
    assert_single_owner([a, b], "task_0")


def test_three_participant_consensus_converges_for_all_declared_members():
    bids = {"ugv_1": 0.55, "ugv_2": 0.72, "ugv_3": 0.61}
    agents = [make_agent(me, [o for o in bids if o != me], FixedUtility({1.0: bids[me]}))
              for me in sorted(bids)]

    run_to_completion(agents)

    assert all(neg.is_converged() for neg in agents), outcome(agents)
    for neg in agents:
        assert neg.get_assignment() == {"task_0": "ugv_2"}, neg.metrics()
    # every participant's belief reports the same holder
    assert {n.winning_agents["task_0"] for n in agents} == {"ugv_2"}


# ------------------------------------------------------------------ 2. conflicting bids

def test_conflicting_claims_settle_on_one_holder_and_release_the_loser():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.40}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.90}))

    # round 1: both claim, producing a genuine conflict
    _, _ = lockstep_round([a, b], {})
    assert a.winning_agents["task_0"] == "ugv_1"
    assert b.winning_agents["task_0"] == "ugv_2"

    # round 2: a sees b's higher bid and must give up
    lockstep_round([a, b], {"ugv_2": belief_of(b)})
    assert a.winning_agents["task_0"] == "ugv_2"
    assert a.bundle == []
    assert "task_0" not in a.bundle


# ------------------------------------------------------------------ 3. disagreeing targets

def test_participants_with_different_target_declaration_cannot_agree():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))

    a.run_round([])
    peer = belief_of(b)
    # b believes task_0 refers to a different target
    peer.target_ids = {"task_0": "target_other"}

    changed, _ = a.run_round([peer])

    assert changed is False
    assert a.stats.count(RejectionClass.TARGET_MISMATCH) == 1
    assert a.is_converged() is False
    assert a.convergence_status is ConsensusStatus.RUNNING


def test_participants_with_different_task_sets_cannot_agree():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))

    a.run_round([])
    peer = belief_of(b)
    peer.task_ids = ["task_0", "task_1"]

    a.run_round([peer])

    assert a.stats.count(RejectionClass.TARGET_MISMATCH) == 1
    assert a.check_agreement() is False


# ------------------------------------------------------------------ 4. duplicates

def _open_episode_with_one_reporting_peer():
    """(self, one reporting peer, one silent participant) and the peer's round-1 belief.

    The silent third participant is what keeps this episode RUNNING: agreement needs every declared
    member, so the duplicate-rejection path can be observed over several rounds instead of the
    episode terminating with AGREED after the first delivery.
    """
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2", "ugv_3"],
                        max_rounds=8, convergence_threshold=99)
    a.register_tasks(make_tasks(1), EPISODE)
    a.set_bid_calculator(FixedUtility({1.0: 0.40}))
    a.set_telemetry(90.0)

    b = CBBANegotiator(agent_id="ugv_2", participant_ids=["ugv_1", "ugv_2"], max_rounds=8)
    b.register_tasks(make_tasks(1), EPISODE)
    b.set_bid_calculator(FixedUtility({1.0: 0.70}))
    b.set_telemetry(70.0)
    b.run_round([])

    a.run_round([])                                   # a claims task_0 at 0.40
    assert a.winning_agents["task_0"] == "ugv_1"
    return a, belief_of(b, round_number=1)


def test_same_round_replay_is_rejected_and_does_not_double_count_delivery():
    a, peer = _open_episode_with_one_reporting_peer()
    assert peer.winning_agents == {"task_0": "ugv_2"}

    first_changed, _ = a.run_round([peer])
    delivered_after_first = a.stats.beliefs_delivered
    assert first_changed is True          # 0.70 > 0.40, so a hands the task over
    assert a.winning_agents["task_0"] == "ugv_2"
    assert a.bundle == []
    assert delivered_after_first == 1

    # identical payload replayed inside the same round must be refused
    second_changed, _ = a.run_round([peer])
    assert second_changed is False
    assert a.stats.count(RejectionClass.DUPLICATE) == 1
    assert a.stats.beliefs_delivered == delivered_after_first
    assert a.winning_agents["task_0"] == "ugv_2"       # replay cannot flip ownership back


def test_duplicate_belief_does_not_inflate_published_message_count():
    a, peer = _open_episode_with_one_reporting_peer()
    a.run_round([peer])                                # the one real delivery
    published_before = a.stats.beliefs_published

    for _ in range(4):
        a.run_round([peer])

    # every completed round publishes exactly once; a refused duplicate adds nothing to the tally
    assert a.stats.beliefs_published == published_before + 4
    assert a.stats.count(RejectionClass.DUPLICATE) == 4
    assert a.stats.beliefs_delivered == 1
    assert a.is_terminal() is False, "fixture broke: the episode must still be open"


# ------------------------------------------------------------------ 5. stale / out of order

def test_stale_lower_round_belief_is_rejected():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))

    a.run_round([])
    a.run_round([belief_of(b, round_number=2)])
    assert a.stats.beliefs_delivered == 1

    # a round-1 belief arriving after round 2 is stale
    a.run_round([belief_of(b, round_number=1)])
    assert a.stats.count(RejectionClass.STALE) == 1
    assert a.stats.beliefs_delivered == 1


def test_outcome_is_independent_of_arrival_order():
    """Merge order must not change the settled assignment — the property the recovered tie branch
    broke, because it only ever compared "is the peer strictly better than me"."""
    results = {}
    for perm in itertools.permutations(["ugv_2", "ugv_3"]):
        a = make_agent("ugv_1", ["ugv_2", "ugv_3"], FixedUtility({1.0: 0.50}))
        a.run_round([])
        peers = {
            "ugv_2": ConsensusMessage(
                sender_agent_id="ugv_2", episode_id=EPISODE, round_number=1,
                winning_bids={"task_0": 0.75}, winning_agents={"task_0": "ugv_2"},
                task_ids=["task_0"], target_ids={"task_0": "target_0"}),
            "ugv_3": ConsensusMessage(
                sender_agent_id="ugv_3", episode_id=EPISODE, round_number=1,
                winning_bids={"task_0": 0.60}, winning_agents={"task_0": "ugv_3"},
                task_ids=["task_0"], target_ids={"task_0": "target_0"}),
        }
        for sender in perm:
            a.run_round([peers[sender]])
        results[perm] = (a.winning_agents["task_0"], a.winning_bids["task_0"])

    assert len(set(results.values())) == 1
    assert next(iter(results.values())) == ("ugv_2", 0.75)


# ------------------------------------------------------------------ 6. missing / invalid fields

def _peer_belief(**overrides):
    base = dict(
        sender_agent_id="ugv_2", episode_id=EPISODE, round_number=1,
        winning_bids={"task_0": 0.70}, winning_agents={"task_0": "ugv_2"},
        task_ids=["task_0"], target_ids={"task_0": "target_0"},
    )
    base.update(overrides)
    return ConsensusMessage(**base)


@pytest.mark.parametrize("overrides,expected_class", [
    ({"sender_agent_id": ""}, RejectionClass.MALFORMED),
    ({"episode_id": ""}, RejectionClass.MALFORMED),
    ({"round_number": -1}, RejectionClass.MALFORMED),
    ({"schema_version": BELIEF_SCHEMA_VERSION + 1}, RejectionClass.UNSUPPORTED_SCHEMA),
    ({"winning_agents": {"task_0": ""}}, RejectionClass.MALFORMED),          # bid with no holder
    ({"winning_bids": {"task_0": 0.0}, "winning_agents": {"task_0": "ugv_2"}},
     RejectionClass.MALFORMED),                                              # holder with no bid
    ({"winning_bids": {"task_0": float("nan")}}, RejectionClass.MALFORMED),
    ({"winning_bids": {"task_0": 1.7}}, RejectionClass.MALFORMED),           # utility out of [0, 1]
    ({"episode_id": "somewhere_else"}, RejectionClass.FOREIGN_EPISODE),
    ({"sender_agent_id": "ugv_9"}, RejectionClass.UNKNOWN_SENDER),
    ({"ownership_epoch": 7}, RejectionClass.STALE),                    # REQ-110: epoch mismatch
    ({"ownership_epoch": "1"}, RejectionClass.MALFORMED),              # not an int
    ({"ownership_epoch": True}, RejectionClass.MALFORMED),             # bool is not an epoch
])
def test_invalid_beliefs_are_classified_counted_and_never_raise(overrides, expected_class):
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])

    changed, _ = a.run_round([_peer_belief(**overrides)])   # must not propagate

    assert changed is False
    assert a.stats.count(expected_class) == 1
    assert a.stats.total_rejected == 1


def test_non_message_object_is_rejected_as_malformed():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])

    changed, _ = a.run_round([{"sender_agent_id": "ugv_2", "round_number": 1}])

    assert changed is False
    assert a.stats.count(RejectionClass.MALFORMED) == 1


# ------------------------------------------------------------------ 7. silence / timeout

def test_silence_from_a_participant_never_reads_as_agreement():
    """3 declared participants, only 1 replies. Silence is not assent."""
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2", "ugv_3"],
                       max_rounds=6, convergence_threshold=2)
    a.register_tasks(make_tasks(1), EPISODE)
    a.set_bid_calculator(FixedUtility({1.0: 0.80}))
    a.set_telemetry(90.0)

    for _ in range(6):
        a.run_round([])

    assert a.check_agreement() is False
    assert a.is_converged() is False
    assert a.convergence_status is ConsensusStatus.LOCAL_STABLE
    assert "unchanged" in a.stats.outcome_reason or "rounds" in a.stats.outcome_reason


def _one_way_partition():
    """Three declared participants where ``ugv_3`` hears everyone and no one hears ``ugv_3``.

    The point of this fixture is the asymmetry a content-based predicate cannot remove: the
    partitioned member is the only agent that can observe full agreement, so it reaches AGREED while
    the two reachable agents stall. Nothing here commits the episode, because the commitment gate is
    "every declared participant converged", not "some participant is satisfied".
    """
    clock = ManualClock()
    common = dict(max_rounds=6, convergence_threshold=2, time_fn=clock)
    agents = {
        "ugv_1": make_agent("ugv_1", ["ugv_2", "ugv_3"], FixedUtility({1.0: 0.80}), **common),
        "ugv_2": make_agent("ugv_2", ["ugv_1", "ugv_3"], FixedUtility({1.0: 0.90}), **common),
        "ugv_3": make_agent("ugv_3", ["ugv_1", "ugv_2"], FixedUtility({1.0: 0.70}), **common),
    }
    hears = {"ugv_1": {"ugv_2"}, "ugv_2": {"ugv_1"}, "ugv_3": {"ugv_1", "ugv_2"}}
    inflight = {}
    for _ in range(6):
        if all(n.is_terminal() for n in agents.values()):
            break
        out = {}
        for aid, neg in agents.items():
            inbox = [m for s, m in inflight.items() if s != aid and s in hears[aid]]
            _, belief = neg.run_round(inbox)
            out[aid] = belief
        inflight = out
    return agents


def test_one_way_partition_lets_the_unheard_member_agree_alone_but_commits_nobody():
    """Documented limit, pinned as a regression guard (see report §7.7 / §11.5)."""
    agents = _one_way_partition()
    quiet, reachable_a, reachable_b = agents["ugv_3"], agents["ugv_1"], agents["ugv_2"]

    # all three hold the *same* content: the highest bid wins, whoever can prove it
    for neg in agents.values():
        assert neg.winning_agents == {"task_0": "ugv_2"}

    assert quiet.convergence_status is ConsensusStatus.AGREED
    assert quiet.is_converged() is True
    assert quiet.stats.convergence_time_s is not None

    for neg in (reachable_a, reachable_b):
        assert neg.is_converged() is False, "a peer it never heard must not read as assent"
        assert neg.stats.convergence_time_s is None

    # no episode-wide commitment, and no agent violates bundle subset-of-won
    assert sum(1 for n in agents.values() if n.is_converged()) < len(agents)
    for neg in agents.values():
        won = {t for t, w in neg.winning_agents.items() if w == neg.agent_id}
        assert set(neg.bundle) <= won
    assert quiet.bundle == [] and reachable_a.bundle == []


def test_deadline_censors_the_episode_without_agreement():
    clock = ManualClock()
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=8, deadline_s=0.5, time_fn=clock)
    a.register_tasks(make_tasks(1), EPISODE)
    a.set_bid_calculator(FixedUtility({1.0: 0.80}))
    a.set_telemetry(90.0)

    a.run_round([])
    published = a.stats.beliefs_published
    clock.advance(0.6)
    changed, msg = a.run_round([])

    assert changed is False
    assert a.convergence_status is ConsensusStatus.TIMEOUT
    assert a.is_converged() is False
    assert a.is_terminal() is True
    assert a.stats.convergence_time_s is None
    assert published == a.stats.beliefs_published   # a censored round emits nothing


def test_abort_is_terminal_and_never_reports_convergence():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])

    a.abort("handoff deadline passed mid-negotiation")

    assert a.convergence_status is ConsensusStatus.ABORTED
    assert a.is_terminal() is True
    assert a.is_converged() is False
    assert a.convergence_time_s() is None
    assert a.get_assignment() == {"task_0": "ugv_1"}   # local claim, never presented as agreed


# ------------------------------------------------------------------ 8. deterministic conflict

def test_equal_bids_tie_break_to_smaller_agent_id_for_both_participants():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.5000}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.5000}))

    run_to_completion([a, b])

    assert a.winning_agents["task_0"] == "ugv_1"
    assert b.winning_agents["task_0"] == "ugv_1"
    assert a.is_converged() and b.is_converged()
    assert a.bundle == ["task_0"] and b.bundle == []
    assert_single_owner([a, b], "task_0")


def test_tie_break_is_repeated_across_reversed_participant_order():
    """The same physical scenario, with the participant list reversed, must give the same holder:
    the rule is a property of the two claims, not of local ordering."""
    holders = set()
    for peers in (["ugv_2", "ugv_3"], ["ugv_3", "ugv_2"]):
        agents = [make_agent(me, [o for o in ["ugv_1", "ugv_2", "ugv_3"] if o != me],
                             FixedUtility({1.0: 0.5})) for me in ["ugv_1"] + peers]
        run_to_completion(agents)
        holders.add({n.winning_agents["task_0"] for n in agents}.pop())
    assert holders == {"ugv_1"}


@pytest.mark.parametrize("seed", range(12))
def test_convergence_is_repeatable(seed):
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.60 + 0.01 * (seed % 3)}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.59}))

    run_to_completion([a, b])

    expected = "ugv_1" if (0.60 + 0.01 * (seed % 3)) >= 0.59 else "ugv_2"
    assert a.winning_agents["task_0"] == expected
    assert b.winning_agents["task_0"] == expected


# ------------------------------------------------------------------ 9. reset

def test_reset_clears_consensus_state_between_independent_attempts():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])

    assert a.is_converged() is True
    snapshot = (a.current_round, a.stats.beliefs_published, a.winning_agents["task_0"])
    assert snapshot[0] > 0 and snapshot[1] > 0

    a.reset()

    assert a.current_round == 0
    assert a.stats.beliefs_published == 0
    assert a.stats.beliefs_delivered == 0
    assert a.stats.total_rejected == 0
    assert a.stats.abstained_tasks == []
    assert a.stats.outcome is ConsensusStatus.RUNNING
    assert a.stats.convergence_time_s is None
    assert a.winning_agents == {"task_0": ""}
    assert a.winning_bids == {"task_0": 0.0}
    assert a.bundle == []
    assert a.is_terminal() is False


def test_registering_a_second_episode_mid_flight_is_refused():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])

    with pytest.raises(ValueError, match="reset"):
        a.register_tasks(make_tasks(1), "a_different_episode")


def test_same_episode_same_epoch_replay_is_admissible_by_design():
    """Boundary of the replay fence, stated rather than hoped for.

    After ``reset()`` the round watermark is deliberately cleared — a new attempt could not accept
    its peer's *first* belief otherwise — so within one ``(episode_id, ownership_epoch)`` identity a
    replayed belief is admissible. Its content is a peer's last published vector, which the merge
    rule re-derives from, so this is not a divergence path. What *is* fenced is the identity: see
    :func:`test_replay_from_a_previous_ownership_epoch_is_refused` and
    :func:`test_reusing_a_published_episode_id_is_refused`. The previous revision of this file named
    a test like this one "...are not applied" while asserting the opposite.
    """
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])
    stale = belief_of(b)
    a.reset()

    a.run_round([stale])

    assert a.stats.count(RejectionClass.FOREIGN_EPISODE) == 0
    assert a.stats.count(RejectionClass.STALE) == 0
    assert a.stats.beliefs_delivered == 1
    assert a.current_round == 1


def test_replay_from_a_previous_ownership_epoch_is_refused():
    """REQ-110: an epoch change (abort-and-reclaim is e -> e+2) invalidates the old beliefs."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}), ownership_epoch=1)
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}), ownership_epoch=1)
    run_to_completion([a, b])
    old = belief_of(b)

    a.reset()
    a.ownership_epoch = 3                      # the adapter re-read EpochManager after a reclaim
    a.run_round([old])

    assert a.stats.count(RejectionClass.STALE) == 1
    assert a.stats.beliefs_delivered == 0
    # The round still prices this agent's own bid (phase 1 precedes ingestion), but the stale claim
    # never entered the vectors: ugv_2 must not appear as a holder anywhere.
    assert a.winning_agents["task_0"] != "ugv_2"
    assert "ugv_2" not in set(a.winning_agents.values())
    assert a._peer_winners.get("ugv_2") is None


# ------------------------------------------------------------------ 10. no false agreement / no dual ownership

def test_all_agents_abstaining_yields_no_winner_not_convergence():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}), battery=None)
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}), battery=None)

    run_to_completion([a, b])

    assert a.convergence_status is ConsensusStatus.NO_WINNER
    assert b.convergence_status is ConsensusStatus.NO_WINNER
    assert a.is_converged() is False
    assert a.get_assignment() == {}
    assert a.stats.abstained_tasks == ["task_0"]


def test_agreement_cannot_be_claimed_while_vectors_disagree():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))

    a.run_round([])
    assert a.check_agreement() is False, "peer has not reported yet"

    a.run_round([belief_of(b, round_number=1)])
    assert a.check_agreement() is False, "peer still believes it holds the task"
    assert a.is_converged() is False


def test_bundle_can_never_hold_a_task_the_agent_does_not_win():
    """Safety invariant across a range of bid profiles: a bundled task must be one this agent wins,
    otherwise two robots would both act as owner and a handoff could be reported falsely."""
    profiles = [(0.80, 0.20), (0.20, 0.80), (0.50, 0.50), (0.50001, 0.50002), (0.0, 0.0),
                (1.0, 0.9999), (0.31, 0.32)]
    for u_a, u_b in profiles:
        a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: u_a}))
        b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: u_b}))
        run_to_completion([a, b])

        for neg in (a, b):
            for held in neg.bundle:
                assert neg.winning_agents.get(held, "") in ("", neg.agent_id), (
                    f"profile {(u_a, u_b)}: {neg.agent_id} bundles {held} while believing "
                    f"{neg.winning_agents.get(held)!r} owns it")
            if neg.convergence_status is ConsensusStatus.AGREED:
                assert neg.get_assignment(), "AGREED with no assignment must be NO_WINNER instead"
        assert_single_owner([a, b], "task_0")


def test_dual_claim_is_detected_when_injected_state_contradicts_the_bundle():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])
    assert a.bundle == ["task_0"]

    # simulate a corrupted merge that left the claim in the bundle but handed the win to a peer
    a._winning_agents["task_0"] = "ugv_2"
    a._winning_bids["task_0"] = 0.90

    changed, _ = a.run_round([])

    assert a.convergence_status is ConsensusStatus.DUAL_CLAIM
    assert a.is_converged() is False
    assert "safety invariant" in a.stats.outcome_reason


# ------------------------------------------------------------------ 11. bounded termination

def test_termination_is_bounded_by_max_rounds_without_agreement():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}), max_rounds=2)
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}), max_rounds=2)

    executed = run_to_completion([a, b], rounds=10)

    assert executed <= 3
    assert a.current_round <= 2 and b.current_round <= 2
    assert a.is_terminal() and b.is_terminal()
    for neg in (a, b):
        assert neg.convergence_status in (ConsensusStatus.MAX_ROUNDS, ConsensusStatus.AGREED,
                                          ConsensusStatus.LOCAL_STABLE)
        if neg.convergence_status is ConsensusStatus.MAX_ROUNDS:
            assert neg.is_converged() is False
            assert neg.convergence_time_s() is None


def test_round_bound_is_honoured_even_with_a_generous_deadline():
    clock = ManualClock()
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=3, convergence_threshold=99, deadline_s=None, time_fn=clock)
    a.register_tasks(make_tasks(1), EPISODE)
    a.set_bid_calculator(FixedUtility({1.0: 0.80}))
    a.set_telemetry(90.0)

    for _ in range(10):
        a.run_round([])

    assert a.current_round == 3
    assert a.convergence_status is ConsensusStatus.MAX_ROUNDS
    assert a.stats.rounds_executed == 3


# ------------------------------------------------------------------ 12. configuration

def _load_exp13_config():
    yaml = pytest.importorskip("yaml", reason="exp13_cbba.yaml is a ROS 2 parameter file")
    assert EXP13_CONFIG.is_file(), f"missing deliverable: {EXP13_CONFIG}"
    return yaml.safe_load(EXP13_CONFIG.read_text(encoding="utf-8"))["/**"]["ros__parameters"]


def test_exp13_config_selects_cbba_and_exp13():
    cfg = _load_exp13_config()

    assert cfg["exp_id"] == "EXP13"
    assert cfg["negotiation_protocol"] == "cbba"
    assert cfg["negotiation_protocol"] == PROTOCOL_ID
    assert cfg["algorithm_variant"] == ALGORITHM_VARIANT
    # the human-readable variant string must not be mislabelled as full CBBA
    assert cfg["algorithm_variant"] != "CBBA"
    assert "L = 1" in cfg["algorithm_variant"]


def test_exp13_config_feeds_the_negotiator_constructor():
    cfg = _load_exp13_config()

    assert cfg["max_bundle_size"] == 1
    assert cfg["target_count"] == 1
    assert cfg["swarm_participant_count"] == 2

    neg = CBBANegotiator(
        agent_id="ugv_1",
        participant_ids=["ugv_1", "ugv_2"],
        max_bundle_size=cfg["max_bundle_size"],
        max_rounds=cfg["max_rounds"],
        convergence_threshold=cfg["convergence_threshold"],
        deadline_s=cfg["deadline_s"],
        bid_epsilon=cfg["bid_epsilon"],
        utility_precision=cfg["utility_precision"],
    )
    assert neg.max_rounds == cfg["max_rounds"]
    assert neg.utility_precision == cfg["utility_precision"]
    assert neg.bid_epsilon == pytest.approx(cfg["bid_epsilon"])
    assert neg.deadline_s == cfg["deadline_s"]


def test_exp13_config_supplies_no_hardcoded_battery_or_baseline_overrides():
    cfg = _load_exp13_config()

    # DEF-01: a stand-in battery must never decide an allocation. The key is absent rather than
    # `null` because a float-declared ROS parameter cannot be given a null value without a type
    # violation; absence means "read telemetry or abstain", which is what the node does.
    assert "battery_pct" not in cfg, "a stand-in battery must not be configured"
    # baseline scoring weights and timing budgets belong to coordination_params.yaml; the Exp-13
    # overlay must not smuggle in a second copy that would silently change the ablation
    for key in ("w_fov", "w_dist", "w_battery", "t_neg_timeout_ms", "t_txn_ms",
                "max_cpu_pct", "min_battery_pct"):
        assert key not in cfg, f"unrelated baseline key {key} must not be overridden"


def test_exp13_config_is_not_yet_loaded_by_the_bringup_launch():
    """Tripwire on the **baseline** launch: `robot_bringup.launch.py` must stay free of Exp-13
    wiring, which is how the one-round default path is preserved by non-modification (design
    decision D6/D7). Exp-13 is loaded by its own launch file, `exp13_bringup.launch.py`, and the
    positive assertion — that the overlay reaches a running node there — lives in
    tests/test_exp13_node_integration.py. If this test starts failing, the shared launch was edited
    and the report needs updating."""
    launch_src = BRINGUP_LAUNCH_PY.read_text(encoding="utf-8")

    assert "exp13_cbba" not in launch_src
    assert "negotiation_protocol" not in launch_src


# ------------------------------------------------------------------ 13. interface compatibility

def test_strategy_interface_surface_is_present():
    """The names the repository's recovered negotiator exposed must still resolve, so a future
    adapter wires against the same surface."""
    required = ["register_tasks", "compute_bid", "build_bundle", "create_consensus_message",
                "process_consensus_message", "run_round", "get_assignment", "reset",
                "is_converged", "set_bid_calculator"]
    for name in required:
        assert callable(getattr(CBBANegotiator, name, None)), f"missing strategy method {name}"

    for prop in ["bundle", "path", "winning_bids", "winning_agents", "convergence_status",
                 "current_round"]:
        assert isinstance(getattr(CBBANegotiator, prop, None), property), f"missing {prop}"


def test_message_type_carries_participant_target_and_round_identity():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}), ownership_epoch=6)
    msg = a.create_consensus_message()

    assert msg.sender_agent_id == "ugv_1"
    assert msg.episode_id == EPISODE
    assert msg.round_number == a.current_round
    assert msg.task_ids == ["task_0"]
    assert msg.target_ids == {"task_0": "target_0"}     # REQ-206: target id explicit, not inferred
    assert msg.ownership_epoch == 6                      # REQ-110: epoch is a belief field
    assert msg.schema_version == BELIEF_SCHEMA_VERSION
    assert msg.agreement_established is False, "an in-flight belief must not claim agreement"
    assert BELIEF_SCHEMA_VERSION == 3, (
        "schema 3 added agreement_established (design decision D2/D3, EXP13_DESIGN_DECISIONS_"
        "20261020_210721.md). The pin is not decoration: a schema-2 belief carries no acknowledgement "
        "field and must be refused as UNSUPPORTED_SCHEMA rather than defaulted — see "
        "test_schema_2_belief_is_refused_by_both_intake_paths in tests/test_exp13_adapter.py"
    )
    assert "e6" in msg.key()
    assert msg.belief_id == f"{EPISODE}:ugv_1:{a.current_round}"
    assert json.dumps(msg.key())


def test_selector_is_declared_in_configuration_not_in_the_running_node():
    """Scope: the **baseline** node. ``coordination_node.py`` declares no ``negotiation_protocol`` and
    never imports the negotiator, and it must stay that way — Exp-13 integrates by subclassing it in
    ``nodes/exp13_coordination_node.py`` so the one-round default path is preserved by
    non-modification (design decision D6/D7). The positive counterpart — that the Exp-13 node does
    declare and dispatch the selector — is asserted in tests/test_exp13_node_integration.py, not
    weakened here."""
    node_src = COORDINATION_NODE_PY.read_text(encoding="utf-8")

    assert "declare_parameter('negotiation_protocol'" not in node_src
    assert "cbba_negotiator" not in node_src
    assert node_src.count("declare_parameter") > 0, "sanity: the node does declare parameters"


# ------------------------------------------------------------------ 14. baseline regression

def test_baseline_negotiation_manager_admission_is_unchanged():
    """The Exp-13 module must not alter the baseline transaction admission policy."""
    mgr = NegotiationManager(self_id="ugv_1", protocol_version=0x0203)

    ok, _ = mgr.admit_message(0x0203, "ugv_1", "txn_1", 1, 1, is_request=True)
    assert ok is True
    mgr.open_transaction("txn_1", "target_0", "ugv_2")

    ok, reason = mgr.admit_message(0x0204, "ugv_1", "txn_1", 1, 1)
    assert ok is False and "version" in reason.lower()

    ok, reason = mgr.admit_message(0x0203, "ugv_9", "txn_1", 1, 1)
    assert ok is False and "ugv_9" in reason

    ok, _ = mgr.admit_message(0x0203, "*", "txn_1", 1, 1)
    assert ok is True

    ok, reason = mgr.admit_message(0x0203, "ugv_1", "txn_1", 0, 1)
    assert ok is False and "epoch" in reason.lower()

    mgr.mark_confirmed("txn_1", ack_epoch=2)
    ok, _ = mgr.admit_message(0x0203, "ugv_1", "txn_1", 1, 2)   # idempotent CONFIRM replay
    assert ok is True

    mgr.close_transaction("txn_1", "handoff completed")
    ok, reason = mgr.admit_message(0x0203, "ugv_1", "txn_1", 2, 2)
    assert ok is False and "CLOSED" in reason


def test_baseline_utility_scoring_is_shared_with_the_exp13_arm():
    """The ablation changes the allocator only: the CBAA bid must come from the same UtilityEngine
    the baseline selector uses, so message-count differences are attributable to the protocol."""
    from edgesync_coordination.decision.utility_engine import UtilityEngine

    engine = UtilityEngine(w_receiver_fov=0.5, w_receiver_dist=0.3, w_receiver_battery=0.2)
    on_axis = [TaskInfo(task_id="task_0", target_id="target_0", position=(0.0, 0.0, 5.0))]
    baseline_utility, s_fov, s_dist, s_battery = engine.compute_receiver_utility(
        point_camera=list(on_axis[0].position), battery_pct=90.0)

    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"])
    a.register_tasks(on_axis, EPISODE)
    a.set_bid_calculator(engine)
    a.set_telemetry(90.0)
    bid = a.compute_bid("task_0")

    assert bid == pytest.approx(baseline_utility)
    assert (s_fov, s_dist, s_battery) == (1.0, 0.2, 0.9)
    assert bid == pytest.approx(0.5 * s_fov + 0.3 * s_dist + 0.2 * s_battery)


def test_baseline_coordination_parameters_are_untouched():
    """The new YAML must not have edited the file the baseline actually loads."""
    yaml = pytest.importorskip("yaml")
    baseline = yaml.safe_load(
        (WS_SRC / "edgesync_bringup" / "config" / "coordination_params.yaml")
        .read_text(encoding="utf-8"))["/**"]["ros__parameters"]

    assert baseline["robot_id"] == "ugv_1"
    assert baseline["t_txn_ms"] == 886
    assert baseline["candidate_gate_cpu_max_pct"] == 90.0
    assert baseline["utility_weight_feasibility"] == 0.50
    assert "negotiation_protocol" not in baseline
    assert "exp_id" not in baseline


def test_launch_still_loads_the_same_four_coordination_configs():
    launch_src = BRINGUP_LAUNCH_PY.read_text(encoding="utf-8")
    node_block = launch_src.split("node_coordination = Node(")[1].split("node_target_acquisition")[0]

    for expected in ("coordination_params", "timing_params", "algo_params", "hw_params"):
        assert expected in node_block, f"baseline config channel {expected} disappeared"


# ------------------------------------------------------------------ DEF-targeted checks

def test_no_hardcoded_battery_stand_in_reaches_the_calculator():
    """DEF-01: the recovered file passed battery_pct=85.0 regardless of the robot's state."""
    util = FixedUtility({1.0: 0.8})
    a = make_agent("ugv_1", ["ugv_2"], util, battery=12.5)
    a.compute_bid("task_0")

    assert util.seen_battery == [12.5]
    assert 85.0 not in util.seen_battery


def test_abstention_produces_no_bid_instead_of_a_substitute_score():
    """DEF-02: a fabricated ``priority * 0.5`` fallback used to decide allocations."""
    util = FixedUtility({1.0: 0.0}, default=0.0)
    a = make_agent("ugv_1", ["ugv_2"], util)
    b = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"])
    b.register_tasks(make_tasks(1), EPISODE)
    b.set_telemetry(90.0)          # telemetry but no calculator

    assert a.compute_bid("task_0") == 0.0
    assert b.compute_bid("task_0") is None
    assert b.bundle == []
    assert b.stats.abstained_tasks == ["task_0"]


def test_broken_calculator_abstains_without_inventing_a_value():
    a = make_agent("ugv_1", ["ugv_2"], BrokenUtility())

    assert a.compute_bid("task_0") is None
    assert a.build_bundle() is False
    assert a.stats.abstained_tasks == ["task_0"]
    assert a.winning_bids == {"task_0": 0.0}


def test_run_round_after_a_terminal_state_does_not_raise():
    """DEF-03: the recovered file raised RuntimeError once the episode ended, which would have
    propagated out of the caller's receive path on any late message."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}), max_rounds=1)

    a.run_round([])
    # The round bound is tested on entry, so round 1 still executes; the next call censors.
    assert a.is_terminal() is False
    a.run_round([])
    assert a.is_terminal() is True
    assert a.convergence_status is ConsensusStatus.MAX_ROUNDS

    for _ in range(3):
        changed, msg = a.run_round([_peer_belief(round_number=7)])
        assert changed is False
        assert isinstance(msg, ConsensusMessage)

    assert a.stats.count(RejectionClass.POST_TERMINAL) == 3
    assert a.stats.beliefs_published == 1, "a terminal episode must stop emitting beliefs"


def test_late_message_after_agreement_cannot_change_the_outcome():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])
    assert a.is_converged() is True

    before = (dict(a.winning_agents), dict(a.winning_bids), list(a.bundle))
    late = _peer_belief(round_number=99, winning_agents={"task_0": "ugv_2"},
                        winning_bids={"task_0": 0.99})
    a.run_round([late])

    assert (dict(a.winning_agents), dict(a.winning_bids), list(a.bundle)) == before
    assert a.convergence_status is ConsensusStatus.AGREED


def test_variant_refuses_bundle_length_greater_than_one():
    """DEF-06/DEF-07: full CBBA needs candidate lists and the path-bid consistency trim. Accepting
    L > 1 would silently ship a single-item auction under a CBBA label."""
    with pytest.raises(ValueError, match="max_bundle_size must be 1"):
        CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"], max_bundle_size=2)

    assert PROTOCOL_ID == "cbba"
    assert ALGORITHM_VARIANT.startswith("CBAA")


def test_convergence_and_termination_predicates_are_distinct():
    """DEF-09: is_converged() used to be true for every non-running state."""
    for status in (ConsensusStatus.MAX_ROUNDS, ConsensusStatus.TIMEOUT,
                   ConsensusStatus.LOCAL_STABLE, ConsensusStatus.NO_WINNER,
                   ConsensusStatus.ABORTED, ConsensusStatus.DUAL_CLAIM):
        neg = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
        neg._status = status
        neg.stats.outcome = status
        assert neg.is_terminal() is True
        assert neg.is_converged() is False, f"{status.name} must not count as convergence"
        assert neg.convergence_time_s() is None

    agreed = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    agreed._status = ConsensusStatus.AGREED
    assert agreed.is_converged() is True


def test_convergence_time_is_measured_on_one_monotonic_clock_in_one_process():
    """No cross-host subtraction: elapsed time is derived from the injected monotonic clock only."""
    clock = ManualClock(t=500.0)
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=4, time_fn=clock)
    b = CBBANegotiator(agent_id="ugv_2", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=4, time_fn=clock)
    for neg in (a, b):
        neg.register_tasks(make_tasks(1), EPISODE)
    a.set_bid_calculator(FixedUtility({1.0: 0.80}))
    a.set_telemetry(90.0)
    b.set_bid_calculator(FixedUtility({1.0: 0.70}))
    b.set_telemetry(70.0)

    lockstep_round([a, b], {})
    clock.advance(0.25)
    inflight, _ = lockstep_round([a, b], {"ugv_1": belief_of(a), "ugv_2": belief_of(b)})

    assert a.stats.started_mono == 500.0
    assert b.stats.started_mono == 500.0
    assert b.convergence_time_s() == pytest.approx(0.25)
    assert b.metrics()["clock_basis"] == "monotonic-single-process"
    # a has only heard b's pre-release belief, so it needs one more round; the interval stays inside
    # the single monotonic clock and never becomes a cross-host subtraction
    assert a.convergence_time_s() is None
    lockstep_round([a, b], inflight)
    assert a.is_converged() is True
    assert a.convergence_time_s() == pytest.approx(0.25)
    assert a.metrics()["clock_basis"] == "monotonic-single-process"


def test_sender_timestamp_is_never_used_for_elapsed_time():
    """Belief timestamps are sender-local, so subtracting two of them across hosts is meaningless.
    Advancing b's clock must not move a's measured convergence time."""
    clock_a = ManualClock(t=0.0)
    clock_b = ManualClock(t=9000.0)
    a = CBBANegotiator(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=4, time_fn=clock_a)
    b = CBBANegotiator(agent_id="ugv_2", participant_ids=["ugv_1", "ugv_2"],
                       max_rounds=4, time_fn=clock_b)
    for neg, axis in ((a, 0.80), (b, 0.70)):
        neg.register_tasks(make_tasks(1), EPISODE)
        neg.set_bid_calculator(FixedUtility({1.0: axis}))
        neg.set_telemetry(90.0)

    a.run_round([])
    clock_b.advance(50.0)
    b_msg = b.create_consensus_message()
    clock_a.advance(0.1)
    a.run_round([b_msg])

    assert a.stats.convergence_time_s is None or a.stats.convergence_time_s <= 0.11, \
        "a's elapsed time was contaminated by b's clock"


def test_malformed_input_is_counted_rather_than_swallowed():
    """DEF-10: ``except ValueError: pass`` hid both bad messages and the agent's own traffic."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    a.run_round([])

    a.run_round([_peer_belief(winning_bids={"task_0": float("inf")})])
    a.run_round([_peer_belief(sender_agent_id="ugv_1")])

    assert a.stats.count(RejectionClass.MALFORMED) == 1
    assert a.stats.count(RejectionClass.SELF) == 1
    assert a.stats.total_rejected == 2
    # every rejection is individually auditable, and none of it changed the belief
    assert a.winning_agents == {"task_0": "ugv_1"}


def test_metrics_snapshot_is_serialisable_and_labels_uncensored_outcomes():
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])

    m = a.metrics()
    serialized = json.dumps(m, sort_keys=True)      # must be a flat, loggable record

    assert m["status"] == "AGREED"
    assert m["algorithm_variant"] == ALGORITHM_VARIANT
    assert m["convergence_time_s"] is None or m["convergence_time_s"] >= 0.0
    assert m["rejected_by_class"] == {} or set(m["rejected_by_class"]) <= {
        r.name for r in RejectionClass}
    assert "task_0" in m["assignment"]
    assert serialized


def test_host_monotonic_resolution_is_recorded_for_the_report():
    """Not a protocol test: this pins the measurement caveat. On this host the OS tick is coarse,
    so a sub-tick convergence interval legitimately measures 0.0 and must never be reported as an
    averaged convergence time. Exact timings therefore come from the injected clock."""
    t0 = time.monotonic()
    while time.monotonic() == t0:
        pass
    granularity = time.monotonic() - t0

    assert granularity > 0.0
    assert repr(granularity) != "0.0"


# ------------------------------------------------------------------ 15. activation-order determinism
# AUD-17 / AUD-20 in EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md. The arrival-order tests in section 5
# permute one agent's *inbox*; they cannot see this defect, because it needs one agent to finish a
# round before another starts one. These tests therefore permute the order agents **act** in.

def drive_sequential(ids, utilities, order, rounds=6, **kwargs):
    """Run negotiators one at a time in ``order``, each seeing what has already been published.

    This is the schedule shape a real DDS network produces when agents do not share a tick.
    """
    agents = {i: make_agent(i, [o for o in ids if o != i],
                            FixedUtility({1.0: utilities[i]}), **kwargs) for i in ids}
    seq = [agents[i] for i in order]
    published = {}
    for _ in range(rounds):
        if all(n.is_terminal() for n in seq):
            break
        for neg in seq:
            if neg.is_terminal():
                continue
            inbox = [m for sender, m in published.items() if sender != neg.agent_id]
            _, msg = neg.run_round(inbox)
            published[neg.agent_id] = msg
    return {n.agent_id: n.winning_agents.get("task_0", "") for n in seq}, seq


def test_tie_winner_is_independent_of_activation_order():
    """Equal bids must resolve to the lowest agent id under *every* acting order, at 2, 3 and 4."""
    for ids in (("ugv_1", "ugv_2"),
                ("ugv_1", "ugv_2", "ugv_3"),
                ("ugv_1", "ugv_2", "ugv_3", "ugv_4")):
        utilities = {i: 0.5 for i in ids}
        seen = set()
        for order in itertools.permutations(ids):
            holders, agents = drive_sequential(ids, utilities, order)
            assert len(set(holders.values())) == 1, (order, holders)
            seen |= set(holders.values())
            for neg in agents:
                for held in neg.bundle:
                    assert neg.winning_agents.get(held) in ("", neg.agent_id)
        assert seen == {"ugv_1"}, f"winner varied with activation order: {sorted(seen)}"


def test_highest_bidder_wins_under_every_activation_order():
    """The fix must not have replaced order-dependence with a fixed agent: unequal bids still rank."""
    ids = ("ugv_1", "ugv_2", "ugv_3")
    utilities = {"ugv_1": 0.40, "ugv_2": 0.75, "ugv_3": 0.55}
    seen = set()
    for order in itertools.permutations(ids):
        holders, _ = drive_sequential(ids, utilities, order)
        assert len(set(holders.values())) == 1, (order, holders)
        seen |= set(holders.values())
    assert seen == {"ugv_2"}


def test_a_belief_arriving_before_the_first_round_is_corrected_at_pricing():
    """F-04, in two assertions that isolate *where* the winner is decided.

    Rewritten from ``test_activation_order_sensitivity_returns_if_pricing_follows_ingestion``, which
    asserted that ingesting before pricing still produced the first mover. That non-vacuity witness
    is obsolete: ``build_bundle`` now consults the tie rule against a peer-held record, so the old
    phase order no longer goes wrong. Asserting the old failure would have meant asserting a bug that
    has been fixed, so this keeps the same role — proving the outcome is not a coincidence — by
    exposing the intermediate state instead of the end state.

    (a) Arrival alone cannot resolve a tie: an agent that has priced nothing has no claim to compare,
        so ``_merge`` must take the peer record wholesale. If it did not, the belief would be dropped
        and the task left unassigned.
    (b) Pricing is what displaces it. ``build_bundle`` returning True and flipping the record to the
        lower agent id is exactly the corrected branch; delete that branch and this assertion reverts
        to ``ugv_2``, which is the mutation check's witness.
    """
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.5}))
    b.run_round([])                             # b prices, then publishes
    first_mover = belief_of(b, round_number=1)

    old = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.5}))
    changed, rej = old._accept(first_mover)     # delivered before old's very first round
    assert rej is None and changed
    assert old.winning_agents["task_0"] == "ugv_2", (
        "arrival of a peer claim for an empty local slot is adoption, not a tie-break; there is "
        "nothing local to compare it against yet")

    assert old.build_bundle() is True, (
        "a tied local bid against a peer-held record must still be able to take the task, or the "
        "episode winner is decided by which agent published first")
    assert old.winning_agents["task_0"] == "ugv_1"   # lowest id wins, as specified
    assert old.bundle == ["task_0"] and old.winning_bids["task_0"] == pytest.approx(0.5)

    new = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.5}))
    new.run_round([first_mover])                # production order: price, then ingest
    assert new.winning_agents["task_0"] == "ugv_1"


def test_a_belief_arriving_before_the_first_round_is_order_independent_at_the_episode_level():
    """The reported F-04 scenario, end to end: same logical state, both permissible orders, one winner."""
    for first_mover_id, expected in (("ugv_2", "ugv_1"), ("ugv_1", "ugv_1")):
        u1 = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.9}))
        u2 = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.9}))
        agents = {"ugv_1": u1, "ugv_2": u2}
        mover, other = agents[first_mover_id], agents["ugv_1" if first_mover_id == "ugv_2" else "ugv_2"]

        _, belief = mover.run_round([])
        other.process_consensus_message(belief)      # arrives before `other` has run any round
        _, reply = other.run_round([])
        mover.process_consensus_message(reply)

        holders = {n.agent_id: n.winning_agents.get("task_0") for n in (u1, u2)}
        assert len(set(holders.values())) == 1, (first_mover_id, holders)
        assert next(iter(holders.values())) == expected, (
            f"winner depended on which agent published first: {holders}")
        for neg in (u1, u2):
            for held in neg.bundle:
                assert neg.winning_agents.get(held) in ("", neg.agent_id)


# ------------------------------------------------------------------ 16. AGREEMENT_REACHED clause iv

def test_agreement_is_withheld_in_round_one_even_when_contents_match():
    """Peer vectors can be handed to a negotiator that has published nothing yet; clause (iv) in
    REV04's AGREEMENT_REACHED is what stops that from being reported as agreement."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.70}))
    peer = ConsensusMessage(
        sender_agent_id="ugv_2", episode_id=EPISODE, round_number=1,
        winning_bids={"task_0": 0.70}, winning_agents={"task_0": "ugv_1"},
        task_ids=["task_0"], target_ids={"task_0": "target_0"},
        bundle=[], ownership_epoch=0)

    a.run_round([peer])

    assert a.current_round == 1
    assert a.check_agreement() is True          # content matches ...
    assert a.is_terminal() is False             # ... but round 1 is not an exchange yet

    a.run_round([])

    assert a.current_round == MIN_AGREEMENT_ROUND
    assert a.convergence_status is ConsensusStatus.AGREED


# ------------------------------------------------------------------ 17. replay fence: episode identity

def test_reusing_a_published_episode_id_is_refused():
    """An adapter that retries under the same id would keep old beliefs admissible; refuse it."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])
    assert a.stats.beliefs_published > 0

    a.reset()
    with pytest.raises(ValueError, match="fresh id"):
        a.register_tasks(make_tasks(1), EPISODE)


def test_a_fresh_episode_id_after_reset_is_accepted():
    """The fence must not block the legitimate path: reset, then register a new attempt's id."""
    a = make_agent("ugv_1", ["ugv_2"], FixedUtility({1.0: 0.80}))
    b = make_agent("ugv_2", ["ugv_1"], FixedUtility({1.0: 0.70}))
    run_to_completion([a, b])
    a.reset()

    a.register_tasks(make_tasks(1), "ep_attempt_2")

    assert a.episode_id == "ep_attempt_2"
    assert a.current_round == 0 and a.is_terminal() is False
    a.run_round([belief_of(b)])                   # old-episode belief must now be refused

    assert a.stats.count(RejectionClass.FOREIGN_EPISODE) == 1
    assert a.stats.beliefs_delivered == 0
    assert "ugv_2" not in set(a.winning_agents.values())


# ------------------------------------------------------------------ 18. agreement evidence honesty

def test_metrics_report_what_the_agent_actually_observed():
    """AGREED must be distinguishable from "I am the only one who could see anything"."""
    a = make_agent("ugv_1", ["ugv_2", "ugv_3"], FixedUtility({1.0: 0.9}))
    b = make_agent("ugv_2", ["ugv_1", "ugv_3"], FixedUtility({1.0: 0.4}))
    run_to_completion([a, b])                   # ugv_3 declared but never runs

    m = a.metrics()
    assert m["peers_declared"] == 2
    assert m["peers_observed"] == 1
    assert a.is_converged() is False
    assert a.convergence_status is not ConsensusStatus.AGREED


@pytest.mark.parametrize("seed", range(24))
def test_randomised_repeatability_actually_randomises(seed):
    """Genuine randomisation: RNG draws the utilities *and* the acting order, and the settled holder
    must equal the rule's answer every time. Distinct from ``test_convergence_is_repeatable``, which
    is parametrised by seed but uses no randomness at all."""
    rng = random.Random(seed)
    ids = ["ugv_1", "ugv_2", "ugv_3"]
    utilities = {i: round(rng.uniform(0.05, 0.95), 4) for i in ids}
    order = list(ids)
    rng.shuffle(order)

    holders, agents = drive_sequential(tuple(ids), utilities, tuple(order))

    ranked = sorted(ids, key=lambda i: (-utilities[i], i))
    assert len(set(holders.values())) == 1, (utilities, order, holders)
    assert next(iter(holders.values())) == ranked[0], (utilities, order, holders)
    for neg in agents:
        if neg.is_converged():
            assert neg.winning_agents["task_0"] == ranked[0]
