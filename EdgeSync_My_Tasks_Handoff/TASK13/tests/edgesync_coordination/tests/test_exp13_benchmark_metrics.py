"""ES-EXP13 benchmark-metric tests: the counters that replaced the over-claiming ones.

TEST CLASS: UNIT. Runs the benchmark *script's* code under pytest. NOT a ROS 2 test and NOT the
benchmark run itself: what this file covers is the metric layer of
``edgesync_coordination/scripts/run_exp13_unit_benchmark.py``, which produces no pytest evidence of
its own and is therefore the part of the Tier-U deliverable that could silently rot.

WHY THIS FILE EXISTS
====================
Finding F-02 recorded that the benchmark's ``dual_owner_violations`` compared each agent only with
itself, yet had been read as a dual-ownership proof. The fix added a cross-agent counter and made the
unmeasurable committed state report ``None`` instead of ``0``. Before this file the only guard on that
fix lived inside the script's own ``metric_selfcheck`` block, so deleting the counter would not have
failed anything in ``pytest tests/``. F-04's schedule-level guard belongs here for the same reason:
``late_delivered`` is the delivery shape that reproduces the defect, and it is driven by this script.

Cases: X1-X4 the pair counter's own arithmetic; X5-X6 each arm's per-episode record; X7 what the
aggregate refuses to claim; X8 the table labels; X9-X10 delivery schedules against the documented tie
rule.
"""
import random
from itertools import permutations

import pytest

from edgesync_coordination.scripts import run_exp13_unit_benchmark as B

TASK = B.TASK_ID


@pytest.fixture(scope="module")
def harness():
    exp_cfg, baseline_cfg = B.load_configs()
    engine, _weights = B.build_engine(baseline_cfg)
    return exp_cfg, engine


def _tie_episode(harness, agents=("ugv_1", "ugv_2")):
    """One equal-utility episode: the only cell where the tie rule decides the winner."""
    exp_cfg, engine = harness
    rows = B.one_episode(list(agents), "equal_utilities", engine, exp_cfg,
                         random.Random("20261020|X|0"), 0)
    return {r.arm: r for r in rows}


def _drive(harness, agents, order, mode):
    exp_cfg, engine = harness
    axes = B.base_axes(agents, "shared")
    batteries = B.base_batteries(agents, "equal")
    negs = B.build_negotiators(agents, engine, batteries, axes, exp_cfg, "ep_test")
    B._drive_schedule(negs, order, mode, int(exp_cfg["max_rounds"]))
    return negs


# ------------------------------------------------------------------ X1-X4 the pair counter

def test_the_cross_agent_counter_is_zero_only_when_every_decided_agent_names_the_same_winner():
    assert B.cross_agent_disagreement_pairs({"ugv_1": "ugv_1", "ugv_2": "ugv_1"}) == 0


def test_one_dissenting_agent_of_four_is_three_disagreeing_pairs_not_one():
    views = {f"ugv_{i}": ("ugv_9" if i == 4 else "ugv_1") for i in range(1, 5)}
    assert B.cross_agent_disagreement_pairs(views) == 3, (
        "the metric counts pairs, so a single defector must scale with the swarm; counting agents "
        "would understate a partition as the participant count grows")


def test_agents_that_have_not_decided_are_excluded_from_the_denominator():
    assert B.cross_agent_disagreement_pairs({"ugv_1": "ugv_1", "ugv_2": "", "ugv_3": "ugv_3"}) == 1, (
        "an agent with no winner view holds no ownership claim, so it cannot disagree with anyone; "
        "scoring it as a dissent would report every undecided trial as a conflict")


def test_the_counter_returns_zero_for_an_empty_view_table_rather_than_raising():
    assert B.cross_agent_disagreement_pairs({}) == 0


# ------------------------------------------------------------------ X5-X7 what each arm records

def test_the_exp13_arm_records_a_winner_view_for_every_participant(harness):
    rows = _tie_episode(harness)
    exp = rows["exp13"]
    assert sorted(exp.cross_agent_winner_views) == ["ugv_1", "ugv_2"], (
        "the cross-agent count is only meaningful if every participant's view was actually read; a "
        "silently missing agent would make a disagreement uncountable")
    assert exp.cross_agent_disagreeing_pairs == 0
    assert "Request -> Response -> Confirm" in exp.ownership_commit_basis, (
        "the row must state where a committed owner would be observable, so a zero here cannot be "
        "quoted as proof about ownership")


def test_the_baseline_arm_reports_the_cross_agent_count_as_unmeasured_not_zero(harness):
    rows = _tie_episode(harness)
    base = rows["baseline"]
    assert base.cross_agent_disagreeing_pairs is None, (
        "the baseline model's candidates hold no decision state, so a 0 would be a property of the "
        "model rather than of the protocol — the exact over-claim F-02 recorded")


def test_the_aggregate_leaves_committed_dual_ownership_unmeasured(harness):
    rows = _tie_episode(harness)
    agg = B.aggregate([rows["baseline"], rows["exp13"]], None)
    exp_all = agg["exp13"]["__all__"]
    assert exp_all["committed_dual_ownership_total"] is None
    assert exp_all["cross_agent_pairs_measurable_rows"] == 1, (
        "the denominator must travel with the zero; without it the two rows above are indistinguishable"
    )
    assert agg["baseline"]["__all__"]["cross_agent_disagreement_episodes"] is None


# ------------------------------------------------------------------ X8 what the table says

def test_the_row_that_over_claimed_is_relabelled_and_the_note_says_so():
    labels = {label: (key, note) for label, key, note in B.REQUIRED_ROWS}
    assert "Local bundle/winner self-consistency violations" in labels
    assert "Dual-owner violations" not in labels, (
        "the old label described a different measurement than the one readers took from it")
    key, note = labels["Local bundle/winner self-consistency violations"]
    assert key == "dual_owner_violations_total", (
        "the JSON key is deliberately kept so historical artefacts stay readable")
    assert "Zero is not a dual-ownership proof" in note


def test_committed_ownership_is_a_reported_row_and_is_never_a_measured_number():
    labels = {label: key for label, key, _ in B.REQUIRED_ROWS}
    assert labels["Committed dual-ownership violations"] == "committed_dual_ownership_total"


# ------------------------------------------------------------------ X9-X10 delivery schedules

@pytest.mark.parametrize("mode", ["lockstep", "late_inbox", "late_delivered", "duplicate",
                                  "delayed", "sequential"])
def test_every_delivery_shape_reaches_the_documented_tie_winner_at_two_participants(harness, mode):
    """F-04's trigger is `late_delivered`; the other shapes are here so the fix is not fitted to it."""
    agents = ["ugv_1", "ugv_2"]
    for order in permutations(agents):
        negs = _drive(harness, agents, list(order), mode)
        views = {a: n.winning_agents.get(TASK, "") for a, n in negs.items()}
        decided = {w for w in views.values() if w}
        assert decided == {"ugv_1"}, (
            f"equal bids must resolve to the smaller agent id regardless of which agent published "
            f"first (mode={mode}, order={order}, views={views})")


def test_a_late_delivered_belief_does_not_leave_two_agents_owning_the_same_task(harness):
    agents = ["ugv_1", "ugv_2", "ugv_3"]
    holders = []
    for order in permutations(agents):
        negs = _drive(harness, agents, list(order), "late_delivered")
        views = {a: n.winning_agents.get(TASK, "") for a, n in negs.items()}
        assert B.cross_agent_disagreement_pairs(views) == 0, f"order={order} views={views}"
        holders.append([a for a, n in negs.items() if TASK in n.bundle])
    assert max(len(h) for h in holders) == 1, (
        "exactly one agent may hold the task in its bundle; two holders is the state the dual-owner "
        "metric was being mistaken for")
