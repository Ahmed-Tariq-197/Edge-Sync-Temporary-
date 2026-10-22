"""Experiment 13 integration adapter — belief-exchange negotiation for target handoff.

Why this file exists and what it owns
=====================================
``cbba_negotiator.py`` is transport-free by design and stays that way. This module is the layer that
turns it into something a ROS 2 node can actually run: it mints the episode identity, registers the
task set, prices bids with the project's own :class:`UtilityEngine`, decides when a round is due,
encodes and decodes the belief payload, and — most importantly — converts a consensus result into an
explicit **commit decision** instead of an implicit ownership claim.

It imports **no** ROS package on purpose. ``nodes/exp13_coordination_node.py`` is the only file in
this experiment that touches ``rclpy``, and it holds no decision logic, so every rule below is
executable and testable on a host with no ROS 2 installation. That is a *testability* choice, not a
claim that the ROS graph behaves the same: see ``EXP13_TEST_AND_VALIDATION_REPORT_*.md`` for which
tier each assertion was earned at.

Episode identity (design decision D5)
====================================
``episode_id`` **is** ``HandoffHeader.transaction_id`` — one identity per attempt, and no second
episode field to drift out of sync with it. The initiator adopts the id it already put on the wire and
every other participant echoes that string **verbatim**; ``make_episode_id`` spells it
``<target_id>@e<epoch>#<attempt>`` for a caller that owns its own attempt counter (the harness, and any
node that has not yet run ``trigger_candidate``), while a node driven by the baseline's own trigger
passes the ``txn_<ns>`` string the FSM holds. Both shapes are accepted, and :meth:`metrics` records
which one was in play, because refusing the baseline's naming would have meant either renaming
transactions in a shared file or negotiating under an identity no peer ever sees. Deriving the id from
the attempt rather than re-deriving it locally at each node is what fences a replay: an
abort-and-reclaim moves the epoch, so beliefs from the failed attempt can never be admissible in the
next one (``cbba_negotiator.py`` refuses a mismatching ``episode_id`` outright).

The commit contract (design decision D3/D8)
===========================================
A consensus result is **advisory until it is armed**. ``arm_commit()`` returns a decision only when
:``CBBANegotiator.check_distributed_agreement()`` holds — this agent agreed *and* every declared peer
reported agreement for this episode and epoch. Anything else yields ``RETAIN`` or ``ABORT``, never a
silent fallthrough to the baseline commit. The adapter itself never advances an epoch, never publishes
an ownership claim and never touches the FSM: the caller applies the returned decision, so this
boundary is auditable from the return value alone.

Safety boundary: this layer cannot make ownership transfer safe. ``EXP13_SAFETY_FINDINGS_*`` records
that the confirm/ACK exchange below the commit seam has a measured dual-OWNER path and that
``ReceiverState.AWAITING_ACK`` has no production producer at all.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .cbba_negotiator import (
    BELIEF_SCHEMA_VERSION,
    CBBANegotiator,
    ConsensusMessage,
    ConsensusStatus,
    RejectionClass,
    TaskInfo,
)

_logger = logging.getLogger(__name__)

PROTOCOL_ID = "cbba"
"""Selector token fixed by the assignment (brief D6). The algorithm is CBAA, L = 1."""

BASELINE_PROTOCOL_ID = "cnp"
"""The one-round Request/Response baseline. Must remain the default everywhere."""

UNIMPLEMENTED_PROTOCOL_IDS = ("auction_cnp", "centralized_broker")
"""Experiment 12 and 14 identifiers. Their meaning is unchanged: nothing in this build interprets
them, and selecting one raises rather than silently falling back to ``cnp`` (brief D6: "Do not
silently redirect an invalid cbba configuration to the baseline")."""

SUPPORTED_PROTOCOL_IDS = (BASELINE_PROTOCOL_ID, PROTOCOL_ID)

TASK_ID_PREFIX = "task_"

PROTOCOL_VERSION = 0x0203
"""``HandoffHeader.protocol_version`` for Exp-13 beliefs — the same v2.3 the baseline uses. The belief
travels on its own topic, but it carries the project header so a single identity rule (``transaction_id``)
applies to it; see design decision D5."""

MSG_TYPE_ALLOCATION_BELIEF = 9
"""``HandoffHeader.message_type`` for a consensus belief. 1–8 are taken by the handoff protocol
(``edgesync_msgs/msg/HandoffHeader.msg:13-20``); 9 is the next free discriminator and is not claimed by
any other experiment in this tree."""

HEADER_FIELDS = ("protocol_version", "message_type", "sender_id", "receiver_id", "transaction_id",
                 "target_id", "ownership_epoch", "sequence", "stamp_utc_ns")
"""The nine ``HandoffHeader`` fields, in declaration order. Kept as data so the schema test can compare
this tuple against the ``.msg`` file instead of trusting a comment."""


class CommitAction(Enum):
    """What the caller may do with this episode's result. Every value is reachable and distinct.

    Two of them are role-relative, which is why :meth:`Exp13CbbaAdapter.arm_commit` takes the
    caller's ownership role instead of guessing: being the awarded participant means *take the
    target* for a standby robot and *keep it* for the one already tracking it.
    """

    WAIT = auto()        # still negotiating; publish/ingest more beliefs
    COMMIT = auto()      # owner-side: hand off to ``receiver_id``
    ACQUIRE = auto()     # awarded participant that is not the owner: begin acquisition
    RETAIN = auto()      # owner won its own target: keep tracking, no transfer, no failure
    STANDBY = auto()     # neither owner nor winner: this episode asks nothing of this agent
    ABORT = auto()       # bounded out, vetoed, or disagreement: do not transfer
    UNSAFE = auto()      # invariant violated (dual claim); caller must not transfer


class ProtocolConfigError(ValueError):
    """Raised for a selector or parameter that cannot be honoured. Never raised for protocol traffic."""


class UnsupportedSchemaError(ValueError):
    """A belief payload written under a schema this build does not interpret.

    Its own type rather than a plain ``ValueError`` because the two refusals mean different things to
    a report: "malformed" says a field was wrong, "unsupported schema" says *a different build is
    talking on this topic*, which is the finding an experiment has to be able to count.
    """


@dataclass
class CommitDecision:
    """Result of :meth:`Exp13CbbaAdapter.arm_commit`. The only object this adapter hands to a node."""

    action: CommitAction
    receiver_id: str = ""
    reason: str = ""
    episode_id: str = ""
    ownership_epoch: int = 0
    rounds: int = 0
    beliefs_published: int = 0
    status: str = ConsensusStatus.RUNNING.name
    is_owner: bool = False

    @property
    def transfers_ownership(self) -> bool:
        return self.action is CommitAction.COMMIT

    @property
    def closes_episode(self) -> bool:
        """True when the node must stop driving rounds for this episode.

        ``WAIT`` is the only action that keeps an episode open, so a node that loops on
        ``not closes_episode`` cannot spin forever after a bound is hit.
        """
        return self.action is not CommitAction.WAIT


@dataclass
class AdapterCounters:
    """Transport-side tallies, kept apart from the negotiator's own stats.

    ``decode_failures`` counts payloads rejected before a :class:`ConsensusMessage` existed, so a
    malformed wire frame is never mistaken for a protocol-level rejection inside the negotiator.
    ``unsupported_schema`` is the subset of those that named a schema this build does not interpret:
    a version finding belongs in a report separately from a field error, because the two say very
    different things about what is running on the other end of the topic.
    """

    beliefs_received: int = 0
    beliefs_emitted: int = 0
    decode_failures: int = 0
    unsupported_schema: int = 0
    rounds_driven: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "beliefs_received": self.beliefs_received,
            "beliefs_emitted": self.beliefs_emitted,
            "decode_failures": self.decode_failures,
            "unsupported_schema": self.unsupported_schema,
            "rounds_driven": self.rounds_driven,
        }


def make_episode_id(target_id: str, ownership_epoch: int, attempt: int = 0) -> str:
    """Mint the one identity an episode has (D5): ``<target>@e<epoch>#<attempt>``."""
    if not target_id:
        raise ProtocolConfigError("target_id must be non-empty to mint an episode id")
    if isinstance(ownership_epoch, bool) or not isinstance(ownership_epoch, int) or ownership_epoch < 0:
        raise ProtocolConfigError(f"ownership_epoch must be an int >= 0, got {ownership_epoch!r}")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ProtocolConfigError(f"attempt must be an int >= 0, got {attempt!r}")
    return f"{target_id}@e{ownership_epoch}#{attempt}"


def parse_episode_id(episode_id: str) -> Tuple[str, int, int]:
    """Inverse of :func:`make_episode_id`. Raises ``ValueError`` on anything it did not mint."""
    if not episode_id or "@e" not in episode_id:
        raise ValueError(f"not an Exp-13 episode id: {episode_id!r}")
    target, rest = episode_id.split("@e", 1)
    epoch_part, _, attempt_part = rest.partition("#")
    if not target or not epoch_part.isdigit():
        raise ValueError(f"not an Exp-13 episode id: {episode_id!r}")
    return target, int(epoch_part), int(attempt_part or 0)


def resolve_negotiation_protocol(value: Any, supported: Sequence[str] = SUPPORTED_PROTOCOL_IDS) -> str:
    """Validate a ``negotiation_protocol`` selector value.

    Nothing here falls back. Unset, empty, whitespace-only, a non-string, an unknown token or a token
    belonging to another experiment all raise, because a typo in an experiment configuration must never
    quietly run a different experiment (brief D6). The baseline is reached by *asking for it*:
    ``declare_parameter('negotiation_protocol', BASELINE_PROTOCOL_ID)`` is the default at the node, so
    the fallback is a declared value, not a silent one. Surrounding whitespace is tolerated because a
    YAML block scalar can carry it.
    """
    if value is None:
        raise ProtocolConfigError("negotiation_protocol is unset; supported: " + ", ".join(supported))
    if not isinstance(value, str):
        raise ProtocolConfigError(f"negotiation_protocol must be a string, got {type(value).__name__}")
    token = value.strip()
    if not token:
        raise ProtocolConfigError("negotiation_protocol is empty; supported: " + ", ".join(supported))
    if token in supported:
        return token
    if token in UNIMPLEMENTED_PROTOCOL_IDS:
        raise ProtocolConfigError(
            f"negotiation_protocol={token!r} names another experiment and is not implemented in this "
            f"build; supported here: {', '.join(supported)}"
        )
    raise ProtocolConfigError(
        f"unknown negotiation_protocol {value!r}; supported: {', '.join(supported)}"
    )


def belief_to_mapping(
    msg: ConsensusMessage,
    receiver_id: str = "*",
    protocol_version: int = PROTOCOL_VERSION,
    stamp_utc_ns: int = 0,
) -> Dict[str, Any]:
    """Belief → wire-shaped dict, matching ``edgesync_msgs/msg/ConsensusBelief.msg`` field for field.

    Two shapes meet here. The ROS message nests a ``HandoffHeader`` and carries parallel repeated
    fields (``map<string,double>`` is not available in the interface), so the flattening/nesting lives in
    these two functions and nowhere else — the ROS node and the ROS-free tests exchange the same dict.
    Field names and the header's field list are asserted against the ``.msg`` file on disk by
    ``tests/test_exp13_adapter.py``, so a rename on one side fails a test rather than corrupting a run.

    ``header.sequence`` carries the round number: within one transaction (episode) beliefs are strictly
    numbered, which is what makes an out-of-order or replayed round detectable by the transaction
    table. ``header.transaction_id`` is the episode id — one identity, not two (D5).

    ``task_ids`` is the *declared* task set, not the tasks this agent happens to have a bid for. An
    abstaining belief must still name ``task_0``, or the receiver's admission check
    (``cbba_negotiator.py:643``) reads it as a different task universe and rejects it as
    TARGET_MISMATCH — which would make "I cannot take this target" indistinguishable from a protocol
    error. Unassigned entries travel as an explicit ``0.0`` bid with an empty winner.
    """
    task_ids = list(msg.task_ids) if msg.task_ids else sorted(msg.winning_bids)
    if not task_ids:
        task_ids = sorted(set(msg.winning_agents) | set(msg.target_ids))
    target_id = next((msg.target_ids[t] for t in task_ids if msg.target_ids.get(t)), "")
    return {
        "header": {
            "protocol_version": int(protocol_version),
            "message_type": MSG_TYPE_ALLOCATION_BELIEF,
            "sender_id": msg.sender_agent_id,
            "receiver_id": receiver_id,
            "transaction_id": msg.episode_id,
            "target_id": target_id,
            "ownership_epoch": int(msg.ownership_epoch),
            "sequence": int(msg.round_number),
            "stamp_utc_ns": int(stamp_utc_ns),
        },
        "schema_version": int(msg.schema_version),
        "belief_id": msg.belief_id,
        "round_number": int(msg.round_number),
        "task_ids": list(task_ids),
        "target_ids": [str(msg.target_ids.get(t, "")) for t in task_ids],
        "winning_bids": [float(msg.winning_bids.get(t, 0.0)) for t in task_ids],
        "winning_agents": [str(msg.winning_agents.get(t, "")) for t in task_ids],
        "bundle": list(msg.bundle),
        "agreement_established": bool(msg.agreement_established),
    }


def belief_from_mapping(data: Dict[str, Any]) -> ConsensusMessage:
    """Wire-shaped dict → :class:`ConsensusMessage`, validating as it goes.

    Raises ``ValueError`` for a payload that cannot be a belief at all (wrong types, ragged parallel
    arrays, unknown schema). The *caller* classifies that as a decode failure and counts it; nothing
    here is a protocol decision.

    The schema check is strict-equality, not a range: a v1 or v2 belief means a different build is
    talking on this topic, and merging its vectors under v3 semantics would silently change what
    ``agreement_established`` means. Refusing it loudly is the safe answer (D2).
    """
    if not isinstance(data, dict):
        raise ValueError(f"belief payload must be a mapping, got {type(data).__name__}")
    header = data.get("header")
    if not isinstance(header, dict):
        raise ValueError("belief payload must carry a nested header mapping")
    missing = [f for f in ("sender_id", "receiver_id", "transaction_id", "ownership_epoch")
               if f not in header]
    if missing:
        raise ValueError(f"belief header missing required field(s): {', '.join(missing)}")

    def req(name: str) -> Any:
        if name not in data:
            raise ValueError(f"belief payload missing required field {name!r}")
        return data[name]

    sender = str(header["sender_id"])
    episode = str(header["transaction_id"])
    if not sender or not episode:
        raise ValueError("belief payload must carry sender and transaction identity")
    epoch = header["ownership_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"header.ownership_epoch must be an int, got {epoch!r}")
    version = int(header.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported belief protocol_version 0x{version:04x}, expected "
                         f"0x{PROTOCOL_VERSION:04x}")
    msg_type = int(header.get("message_type", MSG_TYPE_ALLOCATION_BELIEF))
    if msg_type != MSG_TYPE_ALLOCATION_BELIEF:
        raise ValueError(f"header.message_type {msg_type} is not an allocation belief")
    schema = req("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError(f"schema_version must be an int, got {schema!r}")
    if schema != BELIEF_SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"unsupported belief schema_version {schema}, expected {BELIEF_SCHEMA_VERSION}")
    agreement = req("agreement_established")
    if not isinstance(agreement, bool):
        raise ValueError(f"agreement_established must be a bool, got {agreement!r}")

    task_ids = [str(t) for t in req("task_ids")]
    target_ids = [str(t) for t in req("target_ids")]
    bids = list(req("winning_bids"))
    winners = [str(w) for w in req("winning_agents")]
    if not (len(task_ids) == len(target_ids) == len(bids) == len(winners)):
        raise ValueError(
            f"parallel arrays ragged: {len(task_ids)} task_ids, {len(target_ids)} target_ids, "
            f"{len(bids)} winning_bids, {len(winners)} winning_agents"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate task_ids in one belief: {task_ids}")
    for value in bids:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"winning_bids entries must be numeric, got {value!r}")

    round_number = req("round_number")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ValueError(f"round_number must be an int >= 0, got {round_number!r}")

    return ConsensusMessage(
        sender_agent_id=sender,
        episode_id=episode,
        round_number=int(round_number),
        winning_bids={t: float(v) for t, v in zip(task_ids, bids)},
        winning_agents={t: w for t, w in zip(task_ids, winners)},
        bundle=[str(b) for b in data.get("bundle", [])],
        task_ids=list(task_ids),
        target_ids={t: tgt for t, tgt in zip(task_ids, target_ids)},
        ownership_epoch=int(epoch),
        agreement_established=bool(agreement),
        schema_version=int(schema),
        belief_id=str(data.get("belief_id", "")),
    )


class Exp13CbbaAdapter:
    """One adapter instance per negotiating node. Drives one episode at a time.

    Args:
        agent_id: this node's robot id.
        participant_ids: the **declared** participant set. Never discovered, never grown at runtime:
            a participant that is not in this list cannot block or be blocked by consensus, and a
            declared participant that stays silent is what makes the episode fail bounded (§D4).
        config: the Exp-13 parameter block (``exp13_cbba.yaml``'s ``ros__parameters``). Every value
            is read from here; nothing meaningful is hardcoded, and a missing required key raises at
            construction rather than defaulting mid-experiment.
        utility_engine: the project's ``UtilityEngine``, injected so the CBAA arm bids with the exact
            same scorer the baseline uses. Without it every agent abstains and nothing is allocated.
        publish_belief: called with a wire-shaped dict for each outbound belief. Exceptions from the
            transport are logged and swallowed — a dead publisher must not kill the negotiating node.
        time_fn: monotonic clock, injectable so a test can step time without sleeping.
        stamp_fn: wall-clock nanoseconds for ``header.stamp_utc_ns``. The node passes its ROS clock;
            with none injected the field stays 0, which is honest — a laptop run has no PTP stamp.
    """

    def __init__(
        self,
        agent_id: str,
        participant_ids: Sequence[str],
        config: Dict[str, Any],
        utility_engine: Any = None,
        publish_belief: Optional[Callable[[Dict[str, Any]], None]] = None,
        time_fn: Callable[[], float] = time.monotonic,
        stamp_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        if not agent_id:
            raise ProtocolConfigError("agent_id must be non-empty")
        if not isinstance(config, dict):
            raise ProtocolConfigError("config must be the Exp-13 ros__parameters mapping")

        self.agent_id = agent_id
        self.participants = list(dict.fromkeys(str(p) for p in participant_ids))
        self._time = time_fn
        self._stamp_fn = stamp_fn
        self._publish_belief = publish_belief
        self._utility_engine = utility_engine
        self.counters = AdapterCounters()
        self._inbox: List[ConsensusMessage] = []

        # Required configuration. A default here would make a missing experiment key silently
        # invisible, which is the failure mode the whole file exists to avoid.
        for key in ("max_bundle_size", "max_rounds", "convergence_threshold", "bid_epsilon",
                    "utility_precision", "swarm_participant_count", "target_count",
                    "belief_round_period_ms", "commit_grace_ms"):
            if key not in config:
                raise ProtocolConfigError(f"exp13 config is missing required key {key!r}")

        declared_count = int(config["swarm_participant_count"])
        if declared_count != len(self.participants):
            raise ProtocolConfigError(
                f"swarm_participant_count={declared_count} disagrees with the declared participant "
                f"set {self.participants}; the participant set is configuration, not discovery"
            )
        if len(self.participants) < 2:
            raise ProtocolConfigError("CBAA needs at least 2 declared participants")
        if agent_id not in self.participants:
            raise ProtocolConfigError(f"{agent_id!r} is not in its own participant set")

        self.max_rounds = int(config["max_rounds"])
        self.target_count = int(config["target_count"])
        self.round_period_s = float(config["belief_round_period_ms"]) / 1000.0
        self.commit_grace_s = float(config["commit_grace_ms"]) / 1000.0
        self.require_distributed_agreement = bool(config.get("require_distributed_agreement", True))
        self._negotiator_config = config
        self._episode_attempt = 0
        self._negotiator: Optional[CBBANegotiator] = None
        self._episode_epoch = 0
        self._episode_target = ""
        self._grace_started_mono: Optional[float] = None
        self._decision: Optional[CommitDecision] = None
        self._battery_pct: Optional[float] = None
        self._infeasible: Dict[str, str] = {}
        self._episode_id_kind: str = ""

    # ------------------------------------------------------------------ episode lifecycle

    def start_episode(
        self,
        target_id: str,
        ownership_epoch: int,
        position: Sequence[float],
        tto_safe_s: float,
        battery_pct: Optional[float] = None,
        attempt: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """Begin (or restart) consensus for one handoff opportunity. Returns the episode id.

        ``battery_pct`` must come from a live reading. Passing ``None`` is allowed and means
        "abstain until telemetry arrives", which is the DEF-01 requirement: a stand-in battery value
        must never decide an allocation, so the adapter never substitutes one either.

        ``episode_id``: the initiator mints it (leave it ``None`` and the attempt counter advances);
        every other participant passes the id it received on the wire **verbatim**. The episode id is
        the transaction id, so there is exactly one of them per attempt, and a node that re-derived it
        locally would compute a different ``#attempt`` suffix from its own counter and negotiate a
        different transaction — silently, with both sides' vectors looking valid.
        """
        if self._negotiator is not None and not self._negotiator.is_terminal():
            raise ProtocolConfigError(
                f"episode {self._negotiator.episode_id!r} is still RUNNING; abort() it before "
                f"starting another"
            )
        if self.target_count != 1:
            # Deliberate refusal rather than a guess: the task set below names exactly one slot, and
            # a target_count of N would silently allocate 1 of N.
            raise ProtocolConfigError(
                f"this integration implements the single-target handoff episode "
                f"(target_count={self.target_count} requested): the CBAA bundle is L = 1 and one "
                f"target is in scope per episode"
            )
        if not target_id:
            raise ProtocolConfigError("target_id must be non-empty")
        if attempt is None and episode_id is None:
            self._episode_attempt += 1
            attempt = self._episode_attempt
        if episode_id is None:
            episode_id = make_episode_id(target_id, ownership_epoch, attempt)
            self._episode_id_kind = "minted"
        else:
            # The echoed id is validated, never trusted blind and never re-derived. Two shapes are
            # legitimate: the canonical form this module mints (used by the harness and by any caller
            # that owns its own attempt counter), and the *opaque transaction id* a running node
            # already put on the wire — ``coordination_node.py:445`` builds it as ``txn_<ns>``, and an
            # integration that renamed it would negotiate an episode the peer never joined.
            # For the canonical shape the target and epoch fields must agree with the id; for the
            # opaque shape there is nothing in the string to check, and the binding is carried by the
            # negotiator's own FOREIGN_EPISODE / STALE rejections on (episode_id, ownership_epoch).
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ProtocolConfigError("an echoed episode id must be a non-empty string")
            episode_id = episode_id.strip()
            try:
                parsed_target, parsed_epoch, _ = parse_episode_id(episode_id)
            except ValueError:
                self._episode_id_kind = "opaque_transaction_id"
            else:
                self._episode_id_kind = "canonical"
                if parsed_target != target_id:
                    raise ProtocolConfigError(
                        f"episode id {episode_id!r} names target {parsed_target!r}, not {target_id!r}"
                    )
                if parsed_epoch != int(ownership_epoch):
                    raise ProtocolConfigError(
                        f"episode id {episode_id!r} names epoch {parsed_epoch}, not {ownership_epoch}"
                    )

        neg = CBBANegotiator(
            agent_id=self.agent_id,
            participant_ids=self.participants,
            max_bundle_size=int(self._negotiator_config["max_bundle_size"]),
            max_rounds=self.max_rounds,
            convergence_threshold=int(self._negotiator_config["convergence_threshold"]),
            deadline_s=self._deadline_s(),
            bid_epsilon=float(self._negotiator_config["bid_epsilon"]),
            utility_precision=int(self._negotiator_config["utility_precision"]),
            ownership_epoch=int(ownership_epoch),
            time_fn=self._time,
        )
        if self._utility_engine is not None:
            neg.set_bid_calculator(self._utility_engine)

        task = TaskInfo(
            task_id=f"{TASK_ID_PREFIX}0",
            target_id=target_id,
            position=tuple(float(c) for c in position[:3]),
            tto_safe_s=float(tto_safe_s),
            sender_id=self.agent_id,
        )
        neg.register_tasks([task], episode_id)

        self._negotiator = neg
        self._episode_epoch = int(ownership_epoch)
        self._episode_target = target_id
        self._grace_started_mono = None
        self._decision = None
        # A belief buffered for the previous attempt belongs to the previous episode id; dropping it
        # here keeps the negotiator's rejection tallies about the wire, not about this adapter's queue.
        self._inbox.clear()
        self._infeasible.clear()
        if battery_pct is not None:
            self.set_telemetry(battery_pct)
        _logger.info("Agent %s started Exp-13 episode %s (epoch %d, participants %s)",
                     self.agent_id, episode_id, ownership_epoch, self.participants)
        return episode_id

    def _deadline_s(self) -> Optional[float]:
        value = self._negotiator_config.get("deadline_s")
        return None if value is None else float(value)

    def set_telemetry(self, battery_pct: Optional[float]) -> None:
        """Feed the node's live battery reading into bidding. ``None`` is a no-op, not a zero."""
        if battery_pct is None:
            return
        self._battery_pct = float(battery_pct)
        if self._negotiator is not None:
            self._negotiator.set_telemetry(self._battery_pct)

    def mark_infeasible(self, agent_id: str, reason: str = "hard gate G(j) failed") -> None:
        """Record that a participant reported itself unable to take the target.

        Called from the node's response path, where the baseline's hard gate has already decided. A
        veto that lands *after* ``arm_commit()`` had already returned COMMIT downgrades the cached
        decision to ABORT, because the safe direction for a late safety fact is to not transfer —
        the alternative is to honour a race in the wire's arrival order.
        """
        if not agent_id:
            return
        self._infeasible[agent_id] = reason
        d = self._decision
        if d is not None and d.action is CommitAction.COMMIT and d.receiver_id == agent_id:
            _logger.warning("Agent %s: late infeasibility report from %s overrides an armed COMMIT; "
                            "episode %s closed without transfer", self.agent_id, agent_id,
                            d.episode_id)
            self._decision = CommitDecision(
                action=CommitAction.ABORT, reason=f"late infeasibility report from {agent_id} "
                                                 f"({reason}) overrode a cached COMMIT",
                episode_id=d.episode_id, ownership_epoch=d.ownership_epoch, rounds=d.rounds,
                beliefs_published=d.beliefs_published, status=d.status,
            )

    @property
    def episode_id(self) -> str:
        return self._negotiator.episode_id if self._negotiator is not None else ""

    @property
    def negotiator(self) -> Optional[CBBANegotiator]:
        return self._negotiator

    # ------------------------------------------------------------------ round driving

    def run_round(self) -> Optional[Dict[str, Any]]:
        """Advance one consensus round and emit this agent's belief.

        Inbound beliefs are expected to have been delivered through :meth:`on_belief` beforehand;
        the negotiator's own inbox model is "the adapter hands me what arrived since last round",
        which keeps ordering effects visible to tests instead of hidden in a queue.

        Returns the wire-shaped belief emitted, or ``None`` when nothing new was published. The
        discriminator is the negotiator's own publish counter, not a flag: a censored round (round
        bound, deadline, already terminal) hands back the *previous* belief without counting a
        publish, and re-transmitting it would inflate the message tally that Experiment 13 is scored
        on.
        """
        neg = self._negotiator
        if neg is None:
            return None
        if neg.is_terminal():
            return None
        published_before = neg.stats.beliefs_published
        _, outgoing = neg.run_round(self._drain_inbox())
        self.counters.rounds_driven += 1
        if neg.stats.beliefs_published == published_before:
            return None
        # A round that ends the episode still emits its belief, and that belief is the one carrying
        # agreement_established=True — which is exactly what the peers need to arm their own commit.
        payload = belief_to_mapping(outgoing, stamp_utc_ns=self._stamp())
        self._emit(payload)
        if neg.status is ConsensusStatus.AGREED and self._grace_started_mono is None:
            self._grace_started_mono = self._time()
        return payload

    def on_belief(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Accept one inbound belief. Never raises for a protocol-level problem.

        While the episode runs this is a normal merge. Once the episode has reached ``AGREED`` the
        same stream carries peers' *reports*, which are recorded rather than merged — that is the
        acknowledgement leg of design decision D3, and it costs no extra messages because the belief
        was already being sent.

        Returns ``(handled, note)``. ``note`` is ``""`` when the belief entered this episode's inbox,
        and a class name otherwise: ``NO_EPISODE``, ``MALFORMED_WIRE`` (undecodable), or
        ``POST_TERMINAL`` (the report leg refused it). A belief that is *buffered* here can still be
        refused later by the negotiator's own admission gates — SELF, UNKNOWN_SENDER,
        FOREIGN_EPISODE, STALE, DUPLICATE, TARGET_MISMATCH — which are counted in
        ``negotiator.stats.rejected`` and read back from :meth:`metrics`. Buffering is not a bypass of
        those gates: it is what keeps the merge order fixed (price locally, then ingest), which is the
        property that makes the winner arrival-order independent.
        """
        neg = self._negotiator
        if neg is None:
            return False, "NO_EPISODE"
        self.counters.beliefs_received += 1
        try:
            msg = belief_from_mapping(payload)
        except UnsupportedSchemaError as exc:
            self.counters.decode_failures += 1
            self.counters.unsupported_schema += 1
            _logger.warning("Agent %s refused a belief on schema grounds: %s", self.agent_id, exc)
            return False, RejectionClass.UNSUPPORTED_SCHEMA.name
        except ValueError as exc:
            self.counters.decode_failures += 1
            _logger.warning("Agent %s could not decode a belief payload: %s", self.agent_id, exc)
            return False, "MALFORMED_WIRE"

        if neg.is_terminal():
            recorded = neg.ingest_agreement_report(msg)
            return recorded, ("" if recorded else "POST_TERMINAL")

        # Record first, then buffer. A peer's terminating belief is usually in flight while this
        # agent still has a round to run, so the merge path alone would swallow the declaration and
        # the commit gate could never open. Recording is deduplicated by belief_id and cannot arm
        # anything early: check_distributed_agreement() also requires this agent's own AGREED status.
        if msg.agreement_established:
            neg.ingest_agreement_report(msg)

        # Buffer for the next round rather than merging mid-round: run_round() prices the local bid
        # before it ingests, and that ordering is what makes the winner independent of arrival order.
        self._inbox.append(msg)
        return True, ""

    def _drain_inbox(self) -> List[ConsensusMessage]:
        inbox, self._inbox = self._inbox, []
        return inbox

    def arm_commit(self, is_owner: bool) -> CommitDecision:
        """Decide what the node may do. Safe by default: only ``COMMIT`` transfers ownership.

        ``is_owner`` is the caller's FSM role, and it is required rather than guessed: the same
        consensus result means *keep tracking* for the robot that already owns the target and *begin
        acquisition* for the one that does not. ``STANDBY`` is the third case — a participant that is
        neither owner nor winner is asked to do nothing, which is not a failure and must not be
        tallied as one.

        ``UNSAFE`` is a violated invariant. ``ABORT`` covers every bounded-out or disqualified case:
        rounds exhausted, deadline passed, agreement never established, a peer that never reported,
        or the awarded participant having reported itself infeasible.
        """
        neg = self._negotiator
        if neg is None:
            return CommitDecision(CommitAction.ABORT if is_owner else CommitAction.STANDBY,
                                  reason="no episode started", is_owner=is_owner)
        if self._decision is not None:
            return self._decision

        status = neg.status
        if status is ConsensusStatus.DUAL_CLAIM:
            # UNSAFE for *both* roles: a dual claim is a safety violation whether or not this robot
            # is the one that would have released the target, and it must be reported by every
            # participant that sees it.
            return self._set_decision(CommitAction.UNSAFE, is_owner=is_owner,
                                      reason=neg.stats.outcome_reason)
        if status in (ConsensusStatus.MAX_ROUNDS, ConsensusStatus.TIMEOUT,
                      ConsensusStatus.LOCAL_STABLE, ConsensusStatus.NO_WINNER,
                      ConsensusStatus.ABORTED):
            # A bounded-out episode means *release nothing and resume tracking* for the owner, which
            # is an epoch-bumping recovery action; for a participant that never held the target it
            # means *do nothing*. Returning ABORT to a non-owner would hand the node's recovery path
            # to a robot that has nothing to recover, so the role decides here too.
            return self._set_decision(CommitAction.ABORT if is_owner else CommitAction.STANDBY,
                                      is_owner=is_owner,
                                      reason=f"{status.name}: {neg.stats.outcome_reason}")
        if status is not ConsensusStatus.AGREED:
            return CommitDecision(CommitAction.WAIT, episode_id=neg.episode_id,
                                  ownership_epoch=self._episode_epoch,
                                  rounds=neg.current_round, status=status.name)

        winner = self._awarded_agent()
        if winner is None:
            return self._set_decision(CommitAction.ABORT, is_owner=is_owner,
                                      reason="agreement with no awarded agent")

        # The agreement leg is required before anything is *acted upon*. Both halves of a transfer
        # — the owner releasing and the receiver acquiring — wait for it, so a one-sided agreement
        # cannot set a robot slewing after a target nobody handed it over. RETAIN and STANDBY need no
        # gate because neither changes the physical state: keeping what this robot already tracks is
        # safe whether or not the peer was heard from.
        if self.require_distributed_agreement and not neg.check_distributed_agreement():
            if self._grace_expired():
                missing = [p for p in neg.participants
                           if p != self.agent_id and p not in neg.peers_reported_agreement()]
                return self._set_decision(
                    CommitAction.ABORT if is_owner else CommitAction.STANDBY, is_owner=is_owner,
                    reason=f"commit grace window expired; no agreement report from "
                           f"{', '.join(sorted(missing)) or 'any peer'}")
            return CommitDecision(CommitAction.WAIT, episode_id=neg.episode_id,
                                  ownership_epoch=self._episode_epoch,
                                  rounds=neg.current_round, status=status.name,
                                  reason="awaiting peer agreement reports")

        if winner in self._infeasible:
            # The awarded participant reported it cannot take the target (hard gate G(j): CPU,
            # battery, navigation, link age, or the TTO feasibility clause). Consensus chose it on
            # utility alone, and a veto overrides the winner. This is the concrete reason a locally
            # computed winner is not a commitment.
            return self._set_decision(
                CommitAction.ABORT if is_owner else CommitAction.STANDBY, is_owner=is_owner,
                reason=f"awarded participant {winner} reported itself infeasible "
                       f"({self._infeasible[winner]}); consensus winner overridden",
            )

        if winner == self.agent_id:
            if is_owner:
                return self._set_decision(CommitAction.RETAIN, is_owner=is_owner,
                                          receiver_id=self.agent_id,
                                          reason="this agent holds the best claim to the target it "
                                                 "already owns")
            return self._set_decision(CommitAction.ACQUIRE, is_owner=is_owner,
                                      receiver_id=self.agent_id,
                                      reason="consensus awarded this agent the target, and it is not "
                                             "the current owner")

        if is_owner:
            return self._set_decision(CommitAction.COMMIT, is_owner=is_owner, receiver_id=winner,
                                      reason=self._agreement_basis())
        # Neither owner nor winner: the allocation names somebody else, and this robot does nothing.
        return self._set_decision(CommitAction.STANDBY, is_owner=is_owner, receiver_id=winner,
                                  reason=f"consensus awarded {winner}; this agent is a standby "
                                         f"participant")

    def _agreement_basis(self) -> str:
        """Say, in the decision record, which contract actually armed this commit.

        The downgraded arm must be self-labelling: a log line reading "all declared peers reported
        agreement" on a run that skipped the check would misstate the safety property that was
        exercised.
        """
        if self.require_distributed_agreement:
            return "all declared peers reported agreement"
        return "local agreement only; require_distributed_agreement=false (downgraded contract)"

    def _awarded_agent(self) -> Optional[str]:
        neg = self._negotiator
        if neg is None:
            return None
        awarded = neg.get_target_assignment().get(self._episode_target, "")
        return awarded or None

    def _grace_expired(self) -> bool:
        if self._grace_started_mono is None:
            return False
        return (self._time() - self._grace_started_mono) >= self.commit_grace_s

    def _set_decision(self, action: CommitAction, receiver_id: str = "", reason: str = "",
                      is_owner: bool = False) -> CommitDecision:
        neg = self._negotiator
        self._decision = CommitDecision(
            action=action,
            receiver_id=receiver_id,
            reason=reason,
            episode_id=neg.episode_id if neg else "",
            ownership_epoch=self._episode_epoch,
            rounds=neg.current_round if neg else 0,
            beliefs_published=neg.stats.beliefs_published if neg else 0,
            status=neg.status.name if neg else ConsensusStatus.RUNNING.name,
            is_owner=bool(is_owner),
        )
        _logger.info("Agent %s episode %s -> %s: %s", self.agent_id, self._decision.episode_id,
                     action.name, reason)
        return self._decision

    def abort(self, reason: str = "handoff deadline", is_owner: bool = True) -> CommitDecision:
        """Cancel the episode from outside. Terminal, and terminal means *not* committed."""
        neg = self._negotiator
        if neg is not None:
            neg.abort(reason)
        return self.arm_commit(is_owner)

    def _stamp(self) -> int:
        if self._stamp_fn is None:
            return 0
        try:
            return int(self._stamp_fn())
        except Exception as exc:  # a clock fault must not stop negotiation
            _logger.warning("Agent %s could not read a wall-clock stamp: %s", self.agent_id, exc)
            return 0

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self._publish_belief is None:
            return
        try:
            self._publish_belief(payload)
            self.counters.beliefs_emitted += 1
        except Exception as exc:  # a transport fault must not kill the negotiating node
            _logger.error("Agent %s could not publish a belief: %s", self.agent_id, exc)

    # ------------------------------------------------------------------ reporting

    def metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.counters.as_dict())
        out.update({
            "protocol_id": PROTOCOL_ID,
            "require_distributed_agreement": self.require_distributed_agreement,
            "participants": list(self.participants),
            "infeasible_reports": dict(self._infeasible),
            "episode_id_kind": self._episode_id_kind,
            "action": self._decision.action.name if self._decision else "PENDING",
        })
        if self._negotiator is not None:
            out["negotiator"] = self._negotiator.metrics()
        return out

    def __repr__(self) -> str:
        return (f"Exp13CbbaAdapter(agent={self.agent_id!r}, episode={self.episode_id!r}, "
                f"rounds={self.counters.rounds_driven}, "
                f"action={self._decision.action.name if self._decision else 'PENDING'})")
