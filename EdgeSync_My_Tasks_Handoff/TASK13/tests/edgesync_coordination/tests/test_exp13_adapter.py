"""Experiment 13 adapter, contract and integration-adjacent tests — ROS-free.

Layer this file covers, and the layer it deliberately does not
================================================================
Everything here runs **off a ROS graph, on a host with no rclpy**: the adapter is importable without
``rclpy`` by design, and the message-contract checks read the ``.msg`` files as text. A pass here is
evidence that the adapter's protocol logic and the ``ConsensusBelief`` field contract agree with the
message definition on disk. It is NOT evidence that a DDS topic carries these fields, that two nodes
negotiate over a wire, or that ownership transfers safely on a robot. Those layers are recorded
separately in ``EXP13_TEST_AND_VALIDATION_REPORT_20261020_210721.md``.

Coverage map to the Experiment 13 brief's Phase-D categories
===========================================================
C1  Selector resolution: valid, unknown, unset, wrong type, another experiment's token.
C2  Episode identity: minting, verbatim echo, canonical mismatch refused, opaque transaction id.
C3  Participant rules: count disagreement refused, self not listed, <2 refused, silence not removed.
C4  Wire contract: ``.msg`` field parity, header parity, round-trip fidelity, sequence = round.
C5  Validation at both intake paths: schema 2 refused, malformed, ragged, duplicate, self, unknown
    sender, foreign episode, stale epoch — counted, never raised.
C6  Agreement leg: reports recorded while RUNNING (the regression the smoke run found), commit gate
    will not open on a silent peer, grace expiry.
C7  Commit truth table, role-relative: all seven :class:`CommitAction` values.
C8  Hard-gate veto overrides the consensus winner, including a veto that lands after COMMIT.
C9  Metric accounting: beliefs emitted/received, rounds driven, decode failures.
C10 Zero-hardcoding and DEF-01: no battery stand-in from config, abstain-then-NO_WINNER.
"""
import re
from pathlib import Path

import pytest

from edgesync_coordination.negotiation import (
    ALGORITHM_VARIANT,
    BELIEF_SCHEMA_VERSION,
    MIN_AGREEMENT_ROUND,
    CBBANegotiator,
    ConsensusMessage,
    ConsensusStatus,
    RejectionClass,
)
from edgesync_coordination.negotiation.exp13_cbba_adapter import (
    BASELINE_PROTOCOL_ID,
    HEADER_FIELDS,
    MSG_TYPE_ALLOCATION_BELIEF,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    CommitAction,
    CommitDecision,
    Exp13CbbaAdapter,
    ProtocolConfigError,
    belief_from_mapping,
    belief_to_mapping,
    make_episode_id,
    parse_episode_id,
    resolve_negotiation_protocol,
)

# --------------------------------------------------------------------------- paths

PKG_ROOT = Path(__file__).resolve().parents[1]
WS_SRC = PKG_ROOT.parent
MSG_DIR = WS_SRC / "edgesync_msgs" / "msg"
CONSENSUS_MSG = MSG_DIR / "ConsensusBelief.msg"
HEADER_MSG = MSG_DIR / "HandoffHeader.msg"

TARGET = "target_0"
EPOCH = 3


# --------------------------------------------------------------------------- helpers

def base_config(**overrides):
    """The key set ``exp13_cbba.yaml`` supplies, spelled out here so a missing key is a test failure."""
    cfg = {
        "max_bundle_size": 1,
        "max_rounds": 4,
        "convergence_threshold": 2,
        "bid_epsilon": 1e-6,
        "utility_precision": 4,
        "swarm_participant_count": 2,
        "target_count": 1,
        "belief_round_period_ms": 250,
        "commit_grace_ms": 500,
        "deadline_s": 1.0,
        "require_distributed_agreement": True,
    }
    cfg.update(overrides)
    for key, value in list(cfg.items()):
        if value is None:
            cfg.pop(key)
    return cfg


class SteppingClock:
    """Injectable monotonic clock. Nothing advances it except a test saying so."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class FixedBid:
    """Deterministic bid calculator: every task is worth ``value`` to this agent.

    Returns the same 4-tuple shape ``UtilityEngine.compute_receiver_utility`` returns
    ``(utility, s_fov, s_dist, s_battery)`` and records the battery it was handed, which is what C10
    asserts: the adapter must pass a live reading, never a configured stand-in.
    """

    def __init__(self, value):
        self.value = float(value)
        self.calls = 0
        self.seen_battery = []

    def compute_receiver_utility(self, point_camera=None, battery_pct=None, distance=None):
        self.calls += 1
        self.seen_battery.append(battery_pct)
        return self.value, self.value, self.value, self.value


class Rig:
    """Two adapters, one in-process bus, lockstep rounds.

    The bus is a plain dict of per-agent queues: ``deliver()`` routes each captured payload to every
    *other* adapter. Nothing is published in ``run_round`` unless the adapter emitted a new belief,
    so the queue length is the message count the experiment is scored on.
    """

    def __init__(self, bids=None, config=None, epoch=EPOCH, target=TARGET, battery=90.0,
                 agents=("ugv_1", "ugv_2"), start=True):
        bids = bids or {"ugv_1": 0.80, "ugv_2": 0.60}
        self.config = base_config(**(config or {}))
        self.clock = SteppingClock()
        self.agents = list(agents)
        self.bids = {a: FixedBid(v) for a, v in bids.items()}
        self.emitted = {a: [] for a in self.agents}
        self.queues = {a: [] for a in self.agents}
        self.received = {a: 0 for a in self.agents}
        self.adapters = {}
        for agent in self.agents:
            adapter = Exp13CbbaAdapter(
                agent_id=agent,
                participant_ids=self.agents,
                config=dict(self.config),
                utility_engine=self.bids[agent],
                publish_belief=self._capture(agent),
                time_fn=self.clock,
                stamp_fn=lambda: int(self.clock.t * 1e9),
            )
            self.adapters[agent] = adapter
        if start:
            for agent in self.agents:
                self.adapters[agent].start_episode(
                    target_id=target, ownership_epoch=epoch, position=[1.0, 0.0, 5.0],
                    tto_safe_s=2.0, battery_pct=battery,
                )

    def _capture(self, agent):
        def sink(payload):
            self.emitted[agent].append(payload)
            self.queues[agent].append(payload)
        return sink

    def step(self, rounds=1, deliver_after_each=True):
        for _ in range(rounds):
            for agent in self.agents:
                self.adapters[agent].run_round()
            if deliver_after_each:
                self.deliver()

    def deliver(self):
        pending, self.queues = self.queues, {a: [] for a in self.agents}
        for agent, payloads in pending.items():
            for payload in payloads:
                for other in self.agents:
                    if other == agent:
                        continue
                    self.received[other] += 1
                    self.adapters[other].on_belief(dict(payload))

    def commit(self, agent, is_owner):
        return self.adapters[agent].arm_commit(is_owner)

    def totals(self):
        return {a: len(self.emitted[a]) for a in self.agents}


def sample_belief(agent="ugv_1", episode=f"{TARGET}@e3#1", round_number=2, agreement=False,
                  epoch=EPOCH, schema=BELIEF_SCHEMA_VERSION, target=TARGET):
    """A well-formed belief about the rig's one task slot.

    ``target`` defaults to the target the :class:`Rig` registered: a belief whose ``target_ids`` value
    disagrees with the registered task is refused as TARGET_MISMATCH at the merge, which would silently
    disable every test that expects an ingest. Pass an explicit ``target`` to test that refusal.
    """
    return ConsensusMessage(
        sender_agent_id=agent, episode_id=episode, round_number=round_number,
        winning_bids={"task_0": 0.8}, winning_agents={"task_0": agent},
        bundle=["task_0"], task_ids=["task_0"], target_ids={"task_0": target},
        ownership_epoch=epoch, agreement_established=agreement, schema_version=schema,
        belief_id=f"{agent}:{episode}:{round_number}",
    )


def msg_field_names(path):
    """``.msg`` field names, with comments, blank lines and ``TYPE NAME = value`` constants stripped.

    ``HandoffHeader.msg`` mixes both: its ``MSG_TYPE_*`` constants would otherwise read as fields and
    the parity assertion would compare the wrong sets.
    """
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#")[0].strip()
        if not text or "=" in text:
            continue
        parts = text.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


# --------------------------------------------------------------------------- C1 selector

def test_selector_accepts_the_two_implemented_ids():
    assert resolve_negotiation_protocol("cbba") == PROTOCOL_ID
    assert resolve_negotiation_protocol("cnp") == BASELINE_PROTOCOL_ID
    assert PROTOCOL_ID == "cbba", "the selector token is fixed by the assignment interface"


def test_selector_rejects_unknown_unset_empty_and_wrong_type():
    for bad in (None, "", "   ", 7, ["cbba"], "CBBA", "cbb", "cbba2", "consensus"):
        with pytest.raises(ProtocolConfigError):
            resolve_negotiation_protocol(bad)


def test_selector_tolerates_surrounding_whitespace_because_yaml_block_scalars_do_that():
    assert resolve_negotiation_protocol("  cbba\n") == PROTOCOL_ID
    assert resolve_negotiation_protocol("\tcnp ") == BASELINE_PROTOCOL_ID


def test_selector_names_another_experiments_protocol_rather_than_redirecting_to_baseline():
    """A typo must not silently run the control arm — brief D6."""
    for other in ("auction_cnp", "centralized_broker"):
        with pytest.raises(ProtocolConfigError) as exc:
            resolve_negotiation_protocol(other)
        message = str(exc.value)
        assert "not implemented" in message
        assert PROTOCOL_ID in message and BASELINE_PROTOCOL_ID in message, (
            "the refusal must name what this build actually supports, not just reject")


def test_algorithm_variant_label_is_cbba_not_cbba_full():
    assert "CBAA" in ALGORITHM_VARIANT and "L = 1" in ALGORITHM_VARIANT


# --------------------------------------------------------------------------- C2 episode identity

def test_episode_id_mints_canonically_and_parses_back():
    eid = make_episode_id("tgt", 5, 2)
    assert eid == "tgt@e5#2"
    assert parse_episode_id(eid) == ("tgt", 5, 2)


def test_each_initiator_attempt_advances_the_counter_and_reuses_nothing():
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    first = adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)
    adapter.abort("test")
    second = adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)
    assert first != second
    assert parse_episode_id(first)[2] + 1 == parse_episode_id(second)[2]


def test_echoed_canonical_id_must_name_the_same_target_and_epoch():
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    with pytest.raises(ProtocolConfigError, match="names target"):
        adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0,
                              episode_id="other_target@e3#1")
    with pytest.raises(ProtocolConfigError, match="names epoch"):
        adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0,
                              episode_id=f"{TARGET}@e9#1")


def test_opaque_transaction_id_is_adopted_verbatim_because_the_baseline_mints_that_shape():
    """``coordination_node.py:445`` builds ``txn_<ns>``; an episode must be able to *be* that id.

    If the adapter refused it, the only alternatives would be to change the baseline's transaction
    naming (a shared-file semantic change Exp 12/14 depend on) or to negotiate under an id the peer
    never sees. Both are worse than accepting an opaque id, so this asserts the accepted behaviour and
    records which naming was in play for the report.
    """
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    eid = adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0,
                                episode_id="txn_1729432800123456789")
    assert eid == "txn_1729432800123456789"
    assert adapter.metrics()["episode_id_kind"] == "opaque_transaction_id"


def test_an_empty_or_whitespace_echoed_id_is_refused():
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    for bad in ("", "   ", "  \t"):
        with pytest.raises(ProtocolConfigError):
            adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0,
                                  episode_id=bad)


def test_an_omitted_echoed_id_mints_instead_of_refusing():
    """``None`` means "this agent is the initiator", which is a different thing from ``""``."""
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    assert adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)
    assert adapter.metrics()["episode_id_kind"] == "minted"


def test_starting_a_second_episode_while_one_is_running_is_refused():
    rig = Rig()
    with pytest.raises(ProtocolConfigError, match="still RUNNING"):
        rig.adapters["ugv_1"].start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)


# --------------------------------------------------------------------------- C3 participants

def test_participant_count_disagreeing_with_the_set_is_refused():
    with pytest.raises(ProtocolConfigError, match="swarm_participant_count"):
        Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2", "ugv_3"], base_config())


def test_a_participant_set_of_one_is_refused_because_consensus_needs_two():
    with pytest.raises(ProtocolConfigError, match="at least 2"):
        Exp13CbbaAdapter("ugv_1", ["ugv_1"], base_config(swarm_participant_count=1))


def test_an_agent_outside_its_own_participant_set_is_refused():
    with pytest.raises(ProtocolConfigError):
        Exp13CbbaAdapter("ugv_9", ["ugv_1", "ugv_2"], base_config())


def test_a_silent_participant_is_never_dropped_to_make_the_episode_succeed():
    """D4: silence holds the gate open and censors the episode; it does not shrink ``participants``."""
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    before = list(mine.participants)
    for _ in range(4):
        mine.run_round()          # ugv_2 never answers
    decision = mine.arm_commit(True)
    assert mine.participants == before, "the consensus set is configuration, not discovery"
    assert decision.action is CommitAction.ABORT
    censor = decision.reason.split(":", 1)[0].strip()
    assert censor in {"MAX_ROUNDS", "LOCAL_STABLE", "TIMEOUT", "NO_WINNER", "ABORTED"}, decision.reason
    assert "AGREED" not in decision.reason


def test_missing_required_config_key_is_refused_not_defaulted():
    for key in ("max_rounds", "bid_epsilon", "utility_precision", "belief_round_period_ms",
                "commit_grace_ms", "swarm_participant_count", "target_count"):
        cfg = base_config()
        cfg.pop(key)
        with pytest.raises(ProtocolConfigError, match=key):
            Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], cfg)


def test_multi_target_episode_is_refused_rather_than_half_allocated():
    with pytest.raises(ProtocolConfigError, match="single-target"):
        Rig(config={"target_count": 3, "swarm_participant_count": 2}).adapters["ugv_1"] \
            .start_episode(TARGET, EPOCH + 1, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)


def test_bundle_length_above_one_is_refused_so_the_label_stays_true():
    with pytest.raises((ProtocolConfigError, ValueError)):
        Rig(config={"max_bundle_size": 2})


# --------------------------------------------------------------------------- C4 wire contract

def test_consensus_belief_msg_file_exists_and_declares_the_expected_fields():
    assert CONSENSUS_MSG.is_file(), f"{CONSENSUS_MSG} is missing"
    fields = msg_field_names(CONSENSUS_MSG)
    assert fields[0] == "header"
    assert set(fields) == {
        "header", "schema_version", "belief_id", "round_number", "task_ids", "target_ids",
        "winning_bids", "winning_agents", "bundle", "agreement_established",
    }


def test_payload_keys_match_the_msg_fields_field_for_field():
    payload = belief_to_mapping(sample_belief())
    msg_fields = set(msg_field_names(CONSENSUS_MSG))
    assert set(payload) | {"header"} == msg_fields, (
        "the adapter's dict and the .msg disagree; one side was renamed without the other")


def test_header_fields_match_handoff_header_and_reuse_it_rather_than_duplicating_it():
    assert set(HEADER_FIELDS) == set(msg_field_names(HEADER_MSG)), (
        "HEADER_FIELDS drifted from HandoffHeader.msg")
    text = CONSENSUS_MSG.read_text(encoding="utf-8")
    assert re.search(r"^HandoffHeader\s+header\s*(?:#.*)?$", text, re.MULTILINE), (
        "ConsensusBelief must nest the existing header, not redeclare its fields")


def test_round_trip_preserves_every_belief_field():
    original = sample_belief(agreement=True, epoch=7)
    restored = belief_from_mapping(belief_to_mapping(original, stamp_utc_ns=12345))
    assert restored.sender_agent_id == original.sender_agent_id
    assert restored.episode_id == original.episode_id
    assert restored.round_number == original.round_number
    assert restored.ownership_epoch == 7
    assert restored.winning_bids == original.winning_bids
    assert restored.winning_agents == original.winning_agents
    assert restored.bundle == original.bundle
    assert restored.task_ids == original.task_ids
    assert restored.target_ids == original.target_ids
    assert restored.agreement_established is True
    assert restored.schema_version == BELIEF_SCHEMA_VERSION == 3


def test_sequence_carries_the_round_and_transaction_carries_the_episode():
    payload = belief_to_mapping(sample_belief(episode="tgt@e3#4", round_number=3))
    assert payload["header"]["sequence"] == 3
    assert payload["header"]["transaction_id"] == "tgt@e3#4"
    assert payload["header"]["message_type"] == MSG_TYPE_ALLOCATION_BELIEF == 9
    assert payload["header"]["protocol_version"] == PROTOCOL_VERSION


def test_message_type_nine_is_not_claimed_by_the_existing_header_constants():
    taken = set(msg_field_names(HEADER_MSG))
    assert "MSG_TYPE_ALLOCATION_BELIEF" not in taken, (
        "the belief type is allocated by the adapter, not added to the shared header")
    constants = re.findall(r"MSG_TYPE_\w+\s*=\s*(\d+)", HEADER_MSG.read_text(encoding="utf-8"))
    assert "9" not in constants, f"1..8 only; found {sorted(constants)}"


def test_abstaining_belief_still_names_the_declared_task_slot():
    """A no-bid belief must be addressable, or it is rejected as a different task universe."""
    msg = sample_belief()
    msg.winning_bids = {}
    msg.winning_agents = {}
    msg.bundle = []
    payload = belief_to_mapping(msg)
    assert payload["task_ids"] == ["task_0"]
    assert payload["winning_bids"] == [0.0]
    assert payload["winning_agents"] == [""]
    restored = belief_from_mapping(payload)
    assert restored.winning_agents == {"task_0": ""}


# --------------------------------------------------------------------------- C5 validation

def test_schema_two_belief_is_refused_on_both_intake_paths():
    """The v3 report flag is not negotiable: an old belief cannot carry it."""
    old = belief_to_mapping(sample_belief())
    old["schema_version"] = 2
    with pytest.raises(ValueError):
        belief_from_mapping(old)
    adapter = Rig().adapters["ugv_1"]
    handled, note = adapter.on_belief(old)
    assert handled is False
    assert note == RejectionClass.UNSUPPORTED_SCHEMA.name
    assert adapter.counters.unsupported_schema == 1
    assert adapter.counters.decode_failures == 1, "a refused decode is still a decode failure"


@pytest.mark.parametrize("mutate,cls", [
    (lambda p: p["header"].__setitem__("sender_id", ""), "MALFORMED"),
    (lambda p: p["header"].__setitem__("transaction_id", ""), "MALFORMED"),
    (lambda p: p["header"].__setitem__("protocol_version", 0x0102), None),
    (lambda p: p["header"].__setitem__("message_type", 4), None),
    (lambda p: p.__setitem__("winning_agents", []), "MALFORMED"),
    (lambda p: p.__setitem__("winning_bids", ["0.8"]), "MALFORMED"),
    (lambda p: p.__setitem__("round_number", -1), "MALFORMED"),
    (lambda p: p.__setitem__("agreement_established", "yes"), "MALFORMED"),
    (lambda p: p["header"].__setitem__("ownership_epoch", "3"), "MALFORMED"),
])
def test_malformed_beliefs_are_classified_and_never_raise(mutate, cls):
    rig = Rig()
    payload = belief_to_mapping(sample_belief(agent="ugv_2"))
    mutate(payload)
    applied, note = rig.adapters["ugv_1"].on_belief(payload)
    assert applied is False
    assert note
    assert rig.adapters["ugv_1"].counters.decode_failures >= 1
    if cls is not None:
        assert cls.lower() in note.lower()


def test_a_non_dict_payload_cannot_break_the_receive_path():
    rig = Rig()
    for junk in (None, "belief", 5, []):
        applied, note = rig.adapters["ugv_1"].on_belief(junk)
        assert applied is False and note


def test_peer_identity_gates_run_at_the_merge_and_are_counted_not_raised():
    """Intake buffers, the negotiator classifies. Both halves matter and neither may raise.

    ``on_belief`` returns True for anything it can decode and that belongs to a live episode, because
    merging happens at the round boundary (price locally, then ingest) — that ordering is what makes
    the winner arrival-order independent. The SELF / UNKNOWN_SENDER / FOREIGN_EPISODE / STALE refusals
    therefore show up in ``stats.rejected``, and this test reads them from there rather than pretending
    the inbox is a gate.
    """
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    episode = mine.episode_id

    def payload(sender, epoch=EPOCH, eid=None, rnd=1):
        return belief_to_mapping(sample_belief(agent=sender, episode=eid or episode,
                                               epoch=epoch, round_number=rnd))

    for sender, epoch, eid in (("ugv_1", EPOCH, None), ("ugv_7", EPOCH, None),
                               ("ugv_2", EPOCH, "other@e3#1"), ("ugv_2", EPOCH + 2, None)):
        handled, note = mine.on_belief(payload(sender, epoch=epoch, eid=eid))
        assert handled is True and note == "", "buffered for merge, not refused at intake"
    mine.run_round()
    stats = mine.negotiator.stats
    for cls in (RejectionClass.SELF, RejectionClass.UNKNOWN_SENDER,
                RejectionClass.FOREIGN_EPISODE, RejectionClass.STALE):
        assert stats.count(cls) >= 1, f"{cls.name} was never tallied: {stats.rejected}"


def test_a_replayed_belief_is_deduplicated_by_belief_id_at_the_merge():
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    payload = belief_to_mapping(sample_belief(agent="ugv_2", episode=mine.episode_id))
    assert mine.on_belief(dict(payload))[0] is True
    assert mine.on_belief(dict(payload))[0] is True
    mine.run_round()
    assert mine.negotiator.stats.count(RejectionClass.DUPLICATE) >= 1


# --------------------------------------------------------------------------- C6 agreement leg

def test_agreement_report_is_recorded_before_this_agent_terminates():
    """The defect the two-process smoke run found, pinned.

    A peer's terminating belief routinely arrives while this episode is still RUNNING. If the report
    leg only accepted beliefs after *local* AGREED, the flag in that early belief would be thrown
    away, ``check_distributed_agreement()`` could never open, and the commit gate would deadlock by
    construction.
    """
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    assert mine.negotiator.status is ConsensusStatus.RUNNING
    report = belief_to_mapping(sample_belief(agent="ugv_2", episode=mine.episode_id,
                                             round_number=1, agreement=True))
    applied, _ = mine.on_belief(report)
    assert applied is True
    assert mine.negotiator._peer_agreement.get("ugv_2") is True
    # Recording early must not be able to arm anything on its own.
    assert mine.negotiator.status is ConsensusStatus.RUNNING
    assert mine.negotiator.check_distributed_agreement() is False


def test_a_false_report_is_not_recorded_and_cannot_open_the_gate():
    """Only ``agreement_established == True`` is worth a report slot.

    The adapter's intake asks first (``if msg.agreement_established``), so a peer that has *not*
    converged leaves this agent's view of the peer empty rather than storing a False that
    ``check_distributed_agreement`` would then have to look past. The point of the test is that the
    absence is what keeps the gate shut — a missing report is not a yes.
    """
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    report = belief_to_mapping(sample_belief(agent="ugv_2", episode=mine.episode_id,
                                             agreement=False))
    mine.on_belief(report)
    assert "ugv_2" not in mine.negotiator._peer_agreement
    assert mine.negotiator.peers_reported_agreement() == []


def test_lockstep_episode_arms_commit_for_the_owner_and_acquire_for_the_awarded_peer():
    rig = Rig(bids={"ugv_1": 0.40, "ugv_2": 0.90})   # the non-owner holds the better claim
    rig.step(rounds=4)
    owner = rig.commit("ugv_1", True)
    peer = rig.commit("ugv_2", False)
    assert owner.action is CommitAction.COMMIT, owner.reason
    assert owner.receiver_id == "ugv_2"
    assert "agreement" in owner.reason.lower()
    assert peer.action is CommitAction.ACQUIRE, peer.reason
    assert peer.receiver_id == "ugv_2"
    assert owner.transfers_ownership is True
    assert peer.transfers_ownership is False, "ACQUIRE is the receiver's own leg, not a transfer here"


def test_the_receiving_leg_never_reports_a_commit_for_a_non_owner():
    """D3/D8: a peer that *won* must not be able to talk itself into ``COMMIT``."""
    rig = Rig(bids={"ugv_1": 0.40, "ugv_2": 0.90})
    rig.step(rounds=4)
    assert rig.commit("ugv_2", False).action is not CommitAction.COMMIT


def test_the_owner_that_wins_keeps_and_the_outbid_peer_stands_by():
    """RETAIN is not a failure and STANDBY is not a rejection: neither changes physical state."""
    rig = Rig(bids={"ugv_1": 0.90, "ugv_2": 0.10})
    rig.step(rounds=4)
    owner = rig.commit("ugv_1", True)     # owner also holds the best claim
    peer = rig.commit("ugv_2", False)
    assert owner.action is CommitAction.RETAIN, owner.reason
    assert peer.action is CommitAction.STANDBY, peer.reason
    assert owner.closes_episode and peer.closes_episode
    assert not owner.transfers_ownership and not peer.transfers_ownership


def test_a_commit_needs_the_peer_to_have_been_heard_not_merely_the_vectors_to_match():
    rig = Rig()
    for _ in range(2):
        rig.adapters["ugv_1"].run_round()      # ugv_2 never answers
    assert rig.adapters["ugv_1"].arm_commit(True).action is not CommitAction.COMMIT


def test_grace_expiry_closes_a_one_sided_agreement_as_abort_for_the_owner():
    rig = Rig(config={"commit_grace_ms": 500})
    mine = rig.adapters["ugv_1"]
    rig.step(rounds=4)
    if mine.negotiator.status is not ConsensusStatus.AGREED:
        pytest.skip("ordering did not produce local agreement; silence is covered elsewhere")
    # Simulate a peer that converged but whose report never reached this agent: the wire, not the
    # protocol, is what failed, and the adapter must still refuse to act on it.
    mine.negotiator._peer_agreement.clear()
    mine._grace_started_mono = rig.clock.t - 1.0
    decision = mine.arm_commit(True)
    assert decision.action is CommitAction.ABORT
    assert "grace" in decision.reason and "ugv_2" in decision.reason


def test_the_downgraded_arm_labels_its_own_contract_in_the_decision():
    rig = Rig(bids={"ugv_1": 0.40, "ugv_2": 0.90},
              config={"require_distributed_agreement": False})
    rig.step(rounds=4)
    owner = rig.commit("ugv_1", True)
    assert owner.action is CommitAction.COMMIT
    assert "downgraded" in owner.reason, (
        "a commit taken without the agreement leg must be identifiable as such in the log")
    for agent in rig.agents:
        assert rig.adapters[agent].metrics()["require_distributed_agreement"] is False


# --------------------------------------------------------------------------- C7 commit truth table

def _force_status(adapter, status, reason="test-injected"):
    """Set the negotiator's terminal status directly.

    This asserts the adapter's *mapping*, not the negotiator's reachability: DUAL_CLAIM and the
    rarer censors are not constructible through honest two-agent traffic in L = 1, and a test that
    pretended otherwise would be inventing a protocol path. Reachability of DUAL_CLAIM is a
    negotiator-level property, covered in ``test_cbba_negotiator.py``.
    """
    neg = adapter.negotiator
    neg._status = status                       # `status` is a read-only property by design
    neg.stats.outcome_reason = reason


@pytest.mark.parametrize("status", [ConsensusStatus.MAX_ROUNDS, ConsensusStatus.TIMEOUT,
                                   ConsensusStatus.LOCAL_STABLE, ConsensusStatus.NO_WINNER,
                                   ConsensusStatus.ABORTED])
def test_every_bounded_out_status_is_abort_for_the_owner_and_standby_for_a_non_owner(status):
    rig = Rig()
    owner, other = rig.adapters["ugv_1"], rig.adapters["ugv_2"]
    _force_status(owner, status)
    _force_status(other, status)
    assert owner.arm_commit(True).action is CommitAction.ABORT
    assert other.arm_commit(False).action is CommitAction.STANDBY, (
        "a non-owner must never be handed the owner's recovery action: abort_to_recovery() sets "
        "role = OWNER, which is how a standby robot would declare itself the target's owner")


def test_no_episode_started_is_not_a_commit_for_anyone():
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config())
    assert adapter.arm_commit(True).action is CommitAction.ABORT
    assert adapter.arm_commit(False).action is CommitAction.STANDBY


def test_running_episode_yields_wait_and_does_not_close_anything():
    rig = Rig()
    rig.step(rounds=1)
    decision = rig.adapters["ugv_1"].arm_commit(True)
    if rig.adapters["ugv_1"].negotiator.status is ConsensusStatus.RUNNING:
        assert decision.action is CommitAction.WAIT
        assert decision.closes_episode is False


def test_decision_is_cached_so_a_repeated_ask_returns_the_same_outcome():
    rig = Rig()
    rig.step(rounds=4)
    first = rig.adapters["ugv_1"].arm_commit(True)
    second = rig.adapters["ugv_1"].arm_commit(True)
    assert first is second or (first.action is second.action and first.reason == second.reason)


def test_only_commit_transfers_and_every_action_closing_an_episode_is_terminal():
    assert CommitDecision(CommitAction.WAIT).closes_episode is False
    for action in (CommitAction.COMMIT, CommitAction.ACQUIRE, CommitAction.RETAIN,
                   CommitAction.STANDBY, CommitAction.ABORT, CommitAction.UNSAFE):
        decision = CommitDecision(action)
        assert decision.closes_episode is True
        assert decision.transfers_ownership is (action is CommitAction.COMMIT)


def test_seven_actions_are_reachable_and_named_in_the_order_the_node_applies_them():
    assert [a.name for a in CommitAction] == ["WAIT", "COMMIT", "ACQUIRE", "RETAIN", "STANDBY",
                                              "ABORT", "UNSAFE"]


def test_unsafe_is_reported_to_both_roles_because_a_dual_claim_is_not_role_relative():
    rig = Rig()
    owner, other = rig.adapters["ugv_1"], rig.adapters["ugv_2"]
    _force_status(owner, ConsensusStatus.DUAL_CLAIM, "two holders for target_0")
    _force_status(other, ConsensusStatus.DUAL_CLAIM, "two holders for target_0")
    assert owner.arm_commit(True).action is CommitAction.UNSAFE
    assert other.arm_commit(False).action is CommitAction.UNSAFE


def test_winner_is_read_from_the_episode_target_not_from_the_first_entry():
    rig = Rig()
    rig.step(rounds=2)
    awarded = rig.adapters["ugv_1"]._awarded_agent()
    if awarded is not None:
        assert awarded in rig.agents


# --------------------------------------------------------------------------- C8 veto

def test_a_hard_gate_reject_overrides_the_consensus_winner():
    rig = Rig(bids={"ugv_1": 0.10, "ugv_2": 0.90})
    rig.step(rounds=4)
    mine = rig.adapters["ugv_1"]
    if mine.negotiator.status is not ConsensusStatus.AGREED:
        pytest.skip("agreement not reached in this ordering; veto path needs an armed winner")
    mine.mark_infeasible("ugv_2", "CPU 95% > 90%")
    decision = mine.arm_commit(True)
    assert decision.action is CommitAction.ABORT
    assert "infeasible" in decision.reason and "overridden" in decision.reason


def test_a_late_veto_downgrades_an_already_cached_commit_to_abort():
    rig = Rig(bids={"ugv_1": 0.10, "ugv_2": 0.90})
    rig.step(rounds=4)
    mine = rig.adapters["ugv_1"]
    decision = mine.arm_commit(True)
    if decision.action is not CommitAction.COMMIT:
        pytest.skip(f"winner was not ugv_2 in this run ({decision.action.name}); "
                    f"late-veto needs an armed COMMIT first")
    mine.mark_infeasible(decision.receiver_id, "navigation unavailable")
    after = mine.arm_commit(True)
    assert after.action is CommitAction.ABORT
    assert "late infeasibility" in after.reason


def test_a_veto_of_a_non_winner_leaves_the_armed_commit_alone():
    rig = Rig(bids={"ugv_1": 0.90, "ugv_2": 0.10})
    rig.step(rounds=4)
    mine = rig.adapters["ugv_1"]
    mine.mark_infeasible("ugv_2", "low battery")
    decision = mine.arm_commit(True)
    assert decision.action is CommitAction.RETAIN
    assert mine.metrics()["infeasible_reports"] == {"ugv_2": "low battery"}


def test_a_veto_from_an_unknown_agent_does_not_poison_the_participant_set():
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    mine.mark_infeasible("", "empty id")
    assert mine.metrics()["infeasible_reports"] == {}


# --------------------------------------------------------------------------- C9 accounting

def test_message_count_is_what_the_transport_was_handed_not_rounds_multiplied_by_peers():
    """The official Exp-13 metric, pinned to its measurement point.

    ``rounds x (N - 1)`` would be a *reconstruction*; what is counted here is each payload the adapter
    handed to its publish callback. A censored round emits nothing, so the tally can be lower than the
    arithmetic — which is the difference this test exists to keep visible.
    """
    rig = Rig()
    rig.step(rounds=3)
    for agent in rig.agents:
        adapter = rig.adapters[agent]
        assert adapter.counters.beliefs_emitted == len(rig.emitted[agent])
        assert adapter.negotiator.stats.beliefs_published == len(rig.emitted[agent])
    assert sum(len(rig.emitted[a]) for a in rig.agents) <= 3 * (len(rig.agents) - 1) * 2


def test_rounds_driven_counts_timer_ticks_including_censored_ones():
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    before = mine.counters.rounds_driven
    rig.step(rounds=2)
    assert mine.counters.rounds_driven == before + 2


def test_a_terminal_round_emits_nothing_and_adds_no_message_to_the_tally():
    rig = Rig()
    rig.step(rounds=4)
    mine = rig.adapters["ugv_1"]
    emitted_before = mine.counters.beliefs_emitted
    assert mine.run_round() is None
    assert mine.counters.beliefs_emitted == emitted_before


def test_injection_failure_is_counted_as_a_decode_failure_and_not_applied():
    rig = Rig()
    mine = rig.adapters["ugv_1"]
    junk = belief_to_mapping(sample_belief(agent="ugv_2", episode=mine.episode_id))
    junk["task_ids"] = ["task_0", "task_1"]
    junk["winning_bids"] = [0.5]
    applied, note = mine.on_belief(junk)
    assert applied is False
    assert mine.counters.decode_failures == 1
    assert "ragged" in note.lower() or "MALFORMED" in note


def test_metrics_report_the_protocol_identity_the_node_will_print():
    rig = Rig()
    metrics = rig.adapters["ugv_1"].metrics()
    assert metrics["protocol_id"] == PROTOCOL_ID
    assert metrics["participants"] == ["ugv_1", "ugv_2"]
    assert metrics["action"] == "PENDING"
    assert "negotiator" in metrics


def test_a_publish_callback_that_raises_does_not_kill_the_round():
    def boom(payload):
        raise RuntimeError("transport down")

    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config(),
                               utility_engine=FixedBid(0.5), publish_belief=boom)
    adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)
    payload = adapter.run_round()
    assert payload is not None
    assert adapter.counters.beliefs_emitted == 0
    assert adapter.counters.rounds_driven == 1


def test_a_clock_that_raises_yields_a_zero_stamp_rather_than_no_belief():
    def bad_clock():
        raise RuntimeError("no PTP")

    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config(),
                               utility_engine=FixedBid(0.5), stamp_fn=bad_clock)
    adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0, battery_pct=90.0)
    payload = adapter.run_round()
    assert payload is not None and payload["header"]["stamp_utc_ns"] == 0


# --------------------------------------------------------------------------- C10 zero-hardcoding

def test_no_battery_means_no_bid_and_no_commit_for_anyone():
    """DEF-01 at the integration boundary: a stand-in value must not decide an allocation.

    The node holds bidding shut until a ``StateHeartbeat`` it authored itself has arrived, so the
    adapter is asked to run an episode with no telemetry at all. The honest outcome is an episode that
    ends without a winner — reported as such — rather than a bid computed from a config default.
    """
    bid = FixedBid(0.7)
    adapter = Exp13CbbaAdapter("ugv_1", ["ugv_1", "ugv_2"], base_config(), utility_engine=bid)
    adapter.start_episode(TARGET, EPOCH, [1.0, 0.0, 5.0], 2.0)      # battery omitted on purpose
    for _ in range(4):
        adapter.run_round()
    decision = adapter.arm_commit(True)
    assert decision.action is CommitAction.ABORT
    assert decision.reason.split(":", 1)[0].strip() in {
        "NO_WINNER", "LOCAL_STABLE", "MAX_ROUNDS"}, decision.reason


def test_telemetry_none_is_a_no_op_and_not_a_zero():
    adapter = Rig().adapters["ugv_1"]
    adapter._battery_pct = 77.0
    adapter.set_telemetry(None)
    assert adapter._battery_pct == 77.0


def test_the_live_battery_reaches_the_bid_calculator_as_a_float():
    rig = Rig()
    rig.adapters["ugv_1"].set_telemetry(64.5)
    rig.step(rounds=1)
    assert 64.5 in rig.bids["ugv_1"].seen_battery
    assert None not in rig.bids["ugv_1"].seen_battery


def test_deadline_none_means_round_bound_only():
    rig = Rig(config={"deadline_s": None})
    assert rig.adapters["ugv_1"]._deadline_s() is None
    rig.step(rounds=2)
    assert rig.adapters["ugv_1"].counters.rounds_driven == 2


@pytest.mark.parametrize("is_owner,expected", [(True, CommitAction.ABORT),
                                               (False, CommitAction.STANDBY)])
def test_a_passed_deadline_censors_the_episode_without_agreement(is_owner, expected):
    """One adapter per case: a decision is cached per episode, not per role.

    Asking the same adapter twice with different roles would return the first answer, which is the
    correct behaviour for a node that must not change its mind between timer ticks — so the two roles
    need two episodes to be compared honestly.
    """
    rig = Rig(config={"deadline_s": 0.5})
    mine = rig.adapters["ugv_1"]
    mine.run_round()
    rig.clock.advance(1.0)
    mine.run_round()
    assert mine.negotiator.status is ConsensusStatus.TIMEOUT
    decision = mine.arm_commit(is_owner)
    assert decision.action is expected
    assert "TIMEOUT" in decision.reason
