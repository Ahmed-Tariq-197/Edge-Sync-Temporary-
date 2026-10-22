"""Consensus-Based Assignment (bundle length 1) negotiator — Experiment 13.

Official deliverable filename retained from the team execution guide
(``negotiation/cbba_negotiator.py``). **The algorithm implemented here is CBAA, i.e. CBBA
restricted to bundle length ``L = 1``, not full CBBA.** That wording is deliberate: the in-repo
specification ``CBBA-VARIANT-DECISION-RECORD.md`` (``docs/experiments/exp13/``) §1 classifies a
bundle-length-1 consensus allocator as CBAA, and its §1/§5 forbid labelling it "CBBA" in a report,
config key or log line. See :data:`ALGORITHM_VARIANT` for the four equivalence statements.

What is genuinely implemented
    * bundle construction: a task is taken when this agent outbids the record by more than
      ``bid_epsilon``, or when it ties the record and the documented tie rule favours its agent id,
    * task/target assignment over an explicitly declared task set,
    * iterative consensus by winning-bid / winning-agent vector exchange,
    * conflict resolution that is order-independent and idempotent,
    * an **explicit mutual-agreement predicate** (not a local "nothing changed" counter),
    * a **peer-reported agreement acknowledgement** (:meth:`CBBANegotiator.ingest_agreement_report`
      plus :meth:`CBBANegotiator.check_distributed_agreement`) — the predicate an adapter must clear
      before acting on an allocation, which closes the "heard from everyone, heard by no one" blind
      spot at the cost of one boolean on a belief that was already being sent,
    * bounded termination by ``max_rounds`` and, optionally, by a declared monotonic deadline.

What is deliberately NOT implemented (and why)
    * per-agent candidate lists and the CBBA path-bid consistency trim — they do not exist in CBAA
      and are vacuous at ``L = 1``;
    * cascade release of later bundle entries — dead code at ``L = 1`` and would imply ``L > 1``;
    * full CBBA — not supported by any available project evidence, so it is not built speculatively.

Integration honesty (this replaces a false claim in an earlier revision of this file)
    * This module imports **no** ROS package. It does **not** talk to ``OwnershipFSM``,
      ``NegotiationManager`` or ``HandoffRequest``/``HandoffResponse``, and consensus results are
      **not** carried inside existing messages via "structured metadata" — no such carrier exists.
    * It returns belief objects and counters; an adapter owns transport. Allocation output is
      **advisory**: it names whom a handoff should be requested from and this module never mutates
      an ownership epoch.
    * **That is not a safety proof.** Zero dual ownership is enforced by the ACK exchange
      (``ownership_fsm.py`` ``on_confirm_received`` / ``on_confirm_ack_received``), which this module
      neither calls nor checks. ``EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md`` §3.3 measured a
      reachable two-``OWNER`` state in that exchange with no consensus input at all, so "the
      allocation is advisory" bounds this module's blast radius and buys no guarantee about the
      commit path. An integrator must not read the advisory design as ownership safety.

Two properties the belief layer is responsible for, and how they are obtained
    * **Order-independent winner selection** — the tie rule is applied in *both* places a winner can
      be set: on arrival (:meth:`_merge` via :meth:`_prefer`) and at pricing
      (:meth:`build_bundle`). Running phase 1 before ingestion in each round is not sufficient on its
      own, because a belief can arrive before an agent's *first* round, and a tied local bid that
      only tested strict improvement could then never displace the adopted peer claim — the episode
      winner would be decided by which agent published first.
    * **Fencing of replayed beliefs** — a belief is valid only for its ``(episode_id,
      ownership_epoch)`` pair. An epoch mismatch is classified ``STALE``; a re-used ``episode_id``
      is refused at registration, because ``reset()`` must clear the round watermark (otherwise a
      fresh attempt could never accept its peer's first belief) and the watermark alone therefore
      cannot fence a replay.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_logger = logging.getLogger(__name__)

PROTOCOL_ID = "cbba"
"""Selector value used by the team guide's ``negotiation_protocol`` parameter."""

ALGORITHM_VARIANT = "CBAA (CBBA with bundle length L = 1)"
"""Precise designation. Four labelled statements:

1. **Not full CBBA.** No candidate lists, no path-bid consistency check, no cascade release.
2. **Is** CBAA (Brunet, Choi & How, AAMAS 2008): ``L = 1``, one task per agent, consensus is a
   winning-bid / winning-agent vector exchange.
3. Outcome-equivalent to full CBBA **only** for the winning-bid component when ``L = 1``.
4. NOT equivalent for bundle ordering, candidate-list trimming or path-bid cascade behaviour.
"""

BELIEF_SCHEMA_VERSION = 3
"""Belief payload schema version carried in every message; not a ROS ``protocol_version``.

History, because a receiver decides what to refuse from it:

* **1 -> 2** when ``ownership_epoch`` became a belief field, so a schema-1 belief is refused as
  :class:`RejectionClass.UNSUPPORTED_SCHEMA` instead of silently reading as "epoch 0".
* **2 -> 3** (EXP13 design decision D2/D3, ``EXP13_DESIGN_DECISIONS_20261020_210721.md``) when
  ``agreement_established`` was added: the field is the peer-reported agreement acknowledgement that
  lets an adapter distinguish "my view matches what peers told me" from "every peer also reports it
  has matched". A schema-2 belief cannot express that, so it is refused as
  ``UNSUPPORTED_SCHEMA`` rather than defaulted to ``False`` — defaulting would silently downgrade
  every old peer to "never agrees", which is the safe direction but is not a detectable fact.

There is no deployed schema-2 population to migrate: this module has no production consumer, so the
bump breaks no running system. It does mean a mixed-version cluster cannot negotiate at all, which
is the intended failure mode — see :meth:`CBBANegotiator.check_distributed_agreement`.
"""

MIN_AGREEMENT_ROUND = 2
"""Round floor for declaring :attr:`ConsensusStatus.AGREED` (AGREEMENT_REACHED clause iv).

Agreement in round 1 would mean accepting a peer's claim about an episode in which this agent has
published nothing yet, so the floor is conservative by construction: it can delay termination, never
accelerate it. It is not sufficient on its own — the content predicate is what rejects disagreement.
"""


class ConsensusStatus(Enum):
    """Terminal/non-terminal consensus status. Distinct values on purpose.

    ``is_converged()`` is true **only** for :attr:`AGREED`. ``MAX_ROUNDS`` and ``TIMEOUT`` are
    censored outcomes: they must never be summed into an agreement statistic and a convergence
    duration measured for them is undefined (``None``, not ``0``).
    """

    RUNNING = auto()
    AGREED = auto()          # agreement predicate satisfied, as observed by THIS agent
    NO_WINNER = auto()       # agreement satisfied, but every task ended unassigned
    LOCAL_STABLE = auto()    # quiet detector fired; agreement NOT established
    MAX_ROUNDS = auto()      # round bound exhausted first
    TIMEOUT = auto()         # declared monotonic deadline exceeded first
    DUAL_CLAIM = auto()      # this agent holds a claim it does not win, or two holders for one target
    ABORTED = auto()         # externally cancelled, e.g. the handoff deadline passed mid-episode


class RejectionClass(Enum):
    """Classification of inbound beliefs that were not applied."""

    DUPLICATE = auto()
    STALE = auto()
    SELF = auto()
    UNKNOWN_SENDER = auto()
    FOREIGN_EPISODE = auto()
    MALFORMED = auto()
    TARGET_MISMATCH = auto()
    UNSUPPORTED_SCHEMA = auto()
    POST_TERMINAL = auto()


@dataclass
class TaskInfo:
    """One allocable unit. In EdgeSync this is one target-handoff opportunity.

    Attributes:
        task_id: Identifier of the allocable slot within the episode.
        target_id: Identifier of the *target* the slot refers to. Carried explicitly, never
            inferred from ordering or from a lane (REQ-206).
        position: Target position in this agent's camera frame, ``[X_c, Y_c, Z_c]``.
        tto_safe_s: Safe time-to-observe remaining, seconds.
        sender_id: Agent currently holding tracking responsibility, if any.
    """

    task_id: str
    target_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tto_safe_s: float = 2.0
    sender_id: str = ""


@dataclass
class ConsensusMessage:
    """A belief: this agent's current winning-bid / winning-agent vectors for the episode.

    ``schema_version`` and ``belief_id`` make an incompatible or replayed payload detectable
    instead of silently applied. ``ownership_epoch`` is the epoch this agent observed when the
    episode started; a belief carrying a different epoch is not the same negotiation and is dropped
    as :class:`RejectionClass.STALE`, which is what fences beliefs across an abort-and-reclaim
    (epoch ``e + 2``) or an adapter restart. An adapter that does not track epochs passes the same
    value to every participant, which reproduces the pre-epoch behaviour exactly. ``timestamp`` is
    **sender-local monotonic** and is therefore meaningless across hosts; it is retained for
    debugging only and never used in any elapsed-time calculation.
    """

    sender_agent_id: str
    episode_id: str
    round_number: int
    winning_bids: Dict[str, float] = field(default_factory=dict)
    winning_agents: Dict[str, str] = field(default_factory=dict)
    bundle: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    target_ids: Dict[str, str] = field(default_factory=dict)
    ownership_epoch: int = 0
    agreement_established: bool = False
    """Sender's own :class:`ConsensusStatus.AGREED` at emission time (schema 3, design decision D3).

    This is a *report*, not a commitment: it says "the sender's agreement predicate held for the
    vectors in this payload". It is what lets a receiver distinguish an agent that has genuinely
    converged from one that is merely still broadcasting while its own view disagrees — the case
    :meth:`check_agreement` cannot see, because that predicate is local by construction.
    """
    timestamp: float = field(default_factory=time.monotonic)
    schema_version: int = BELIEF_SCHEMA_VERSION
    belief_id: str = ""

    def key(self) -> str:
        return (f"{self.episode_id}@e{self.ownership_epoch}:"
                f"{self.sender_agent_id}:{self.round_number}")


@dataclass
class ConsensusStats:
    """Counters and timings for one episode. Reset by :meth:`CBBANegotiator.reset`."""

    beliefs_published: int = 0
    beliefs_delivered: int = 0
    agreement_reports_accepted: int = 0
    """Beliefs read by :meth:`CBBANegotiator.ingest_agreement_report` after this episode terminated.

    Counted apart from ``beliefs_delivered`` on purpose: those payloads were not merged, so folding
    them into the delivery tally would change the meaning of the metric the benchmark reports.
    """
    rounds_executed: int = 0
    rejected: Dict[RejectionClass, int] = field(default_factory=dict)
    abstained_tasks: List[str] = field(default_factory=list)
    started_mono: Optional[float] = None
    agreed_mono: Optional[float] = None
    outcome: ConsensusStatus = ConsensusStatus.RUNNING
    outcome_reason: str = ""

    def reject(self, cls: RejectionClass) -> None:
        self.rejected[cls] = self.rejected.get(cls, 0) + 1

    def count(self, cls: RejectionClass) -> int:
        return self.rejected.get(cls, 0)

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())

    @property
    def convergence_time_s(self) -> Optional[float]:
        """Monotonic, single-process only. ``None`` when agreement was never established.

        A returned ``0.0`` means the episode started and agreed inside one tick of the host clock;
        it does **not** mean convergence was instantaneous. On Windows the monotonic clock is
        ``GetTickCount64()`` with a 15.625 ms tick, so a sub-tick episode is duration-blind. An
        interval is therefore not comparable across processes, and cross-host convergence needs a
        synchronised clock basis, which is out of scope for this module.
        """
        if self.started_mono is None or self.agreed_mono is None:
            return None
        return self.agreed_mono - self.started_mono


class CBBANegotiator:
    """CBAA (``L = 1``) negotiator: bundle construction plus iterative bid consensus.

    Args:
        agent_id: This agent's identity; compared bytewise, must be unique in the participant set.
        participant_ids: The **declared** participant set for the episode. Agreement requires a
            belief from every member other than self, so silence is never read as assent.
        max_bundle_size: Must remain ``1``. Values above 1 require candidate lists and the path-bid
            consistency check, which this module does not implement, so they are rejected.
        max_rounds: Round bound. Declared input, never derived from a per-round duration.
        convergence_threshold: Consecutive locally unchanged rounds before
            :attr:`ConsensusStatus.LOCAL_STABLE`. Diagnostic only; it never implies agreement.
        deadline_s: Optional monotonic wall budget for the episode. ``None`` means the episode is
            bounded only by ``max_rounds``.
        bid_epsilon: Positive-improvement tolerance for bundle construction and outbidding.
        utility_precision: Decimal places used when comparing bids, so the tie rule is evaluated on
            the same quantisation the receiver-side utility engine already applies.
        ownership_epoch: The epoch the adapter observed when this episode started. Every belief
            carries it and a mismatching epoch is refused as ``STALE``; ``0`` is the default so an
            adapter without epoch tracking behaves as before. It is an **input**, read from
            ``EpochManager`` by the adapter — this module never mutates it.
    """

    PROTOCOL_ID = PROTOCOL_ID
    ALGORITHM_VARIANT = ALGORITHM_VARIANT

    def __init__(
        self,
        agent_id: str,
        participant_ids: Sequence[str],
        max_bundle_size: int = 1,
        max_rounds: int = 4,
        convergence_threshold: int = 2,
        deadline_s: Optional[float] = None,
        bid_epsilon: float = 1e-6,
        utility_precision: int = 4,
        ownership_epoch: int = 0,
        time_fn=time.monotonic,
    ) -> None:
        if max_bundle_size != 1:
            raise ValueError(
                "max_bundle_size must be 1: this module implements CBAA (L = 1). Full CBBA needs "
                "candidate lists and the path-bid consistency check (see CBBA-VARIANT-DECISION-RECORD.md §1)."
            )
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        members = list(dict.fromkeys(str(p) for p in participant_ids))
        if agent_id not in members:
            raise ValueError(f"agent_id {agent_id!r} is not in the declared participant set {members}")
        if len(members) < 2:
            raise ValueError("a consensus episode needs at least 2 declared participants")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if bid_epsilon < 0:
            raise ValueError("bid_epsilon must be non-negative")
        if isinstance(ownership_epoch, bool) or not isinstance(ownership_epoch, int):
            raise ValueError(f"ownership_epoch must be an int, got {ownership_epoch!r}")
        if ownership_epoch < 0:
            raise ValueError(f"ownership_epoch must be >= 0, got {ownership_epoch}")

        self.agent_id = agent_id
        self.participants: List[str] = members
        self.max_bundle_size = max_bundle_size
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.deadline_s = deadline_s
        self.bid_epsilon = float(bid_epsilon)
        self.utility_precision = int(utility_precision)
        self.ownership_epoch = int(ownership_epoch)
        self._time = time_fn

        self._tasks: Dict[str, TaskInfo] = {}
        self._bundle: List[str] = []
        self._own_bids: Dict[str, float] = {}
        self._winning_bids: Dict[str, float] = {}
        self._winning_agents: Dict[str, str] = {}
        self._peer_bids: Dict[str, Dict[str, float]] = {}
        self._peer_winners: Dict[str, Dict[str, str]] = {}
        self._peer_agreement: Dict[str, bool] = {}
        self._seen_report_ids: Set[str] = set()
        self._last_seen_round: Dict[str, int] = {}
        self._published_episode_ids: Set[str] = set()
        self._episode_open = False
        self._battery_pct: Optional[float] = None
        self._last_belief: Optional[ConsensusMessage] = None
        self._episode_id: str = ""
        self._current_round: int = 0
        self._stable_rounds: int = 0
        self._status = ConsensusStatus.RUNNING
        self._bid_calculator = None
        self.stats = ConsensusStats()

        _logger.info(
            "CBBANegotiator (CBAA L=1) init: agent=%s participants=%s max_rounds=%d threshold=%d "
            "deadline_s=%s eps=%g epoch=%d",
            self.agent_id, self.participants, self.max_rounds, self.convergence_threshold,
            self.deadline_s, self.bid_epsilon, self.ownership_epoch,
        )

    # ------------------------------------------------------------------ configuration

    def set_bid_calculator(self, calculator) -> None:
        """Inject the bid function.

        Must expose ``compute_receiver_utility(point_camera, battery_pct)`` returning a
        ``(utility, s_fov, s_dist, s_battery)`` tuple — i.e. ``UtilityEngine`` as used by
        ``coordination_node``. With no calculator injected, :meth:`compute_bid` abstains; it never
        substitutes a fabricated score.
        """
        self._bid_calculator = calculator

    def set_telemetry(self, battery_pct: float) -> None:
        """Supply this agent's own battery for bidding.

        Kept explicit rather than defaulted, so a hardcoded stand-in value cannot silently decide
        an allocation. Until it is called, :meth:`compute_bid` abstains.
        """
        if not isinstance(battery_pct, (int, float)) or not (0.0 <= float(battery_pct) <= 100.0):
            raise ValueError(f"battery_pct out of [0, 100]: {battery_pct!r}")
        self._battery_pct = float(battery_pct)

    # ------------------------------------------------------------------ episode lifecycle

    def register_tasks(self, tasks: Iterable[TaskInfo], episode_id: str) -> None:
        """Declare this episode's task/target set and reset all consensus state.

        Raises:
            ValueError: empty task list, duplicate ``task_id``, a task mapping to more than one
                ``target_id``, a non-empty ``episode_id`` that differs from the one already in use
                once a round of that episode has run, or an ``episode_id`` this negotiator has
                already published beliefs under. Reusing the last of those would let a replayed
                belief from
                the earlier attempt be re-admitted, because :meth:`reset` must clear the round
                watermark for the new attempt to work at all — so episode identity, not the
                watermark, is what fences a replay. Re-using an attempt requires an id that is
                unique per attempt (an adapter can derive one from the target plus the ownership
                epoch it reads).
        """
        items = list(tasks)
        if not items:
            raise ValueError("Cannot register empty task list")
        if not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if self._episode_id and self._episode_id != episode_id and self._episode_open:
            raise ValueError(
                f"Negotiator is mid-episode {self._episode_id!r}; call reset() before starting "
                f"{episode_id!r} — otherwise beliefs from two attempts interleave"
            )
        if episode_id in self._published_episode_ids:
            raise ValueError(
                f"episode_id {episode_id!r} was already used by agent {self.agent_id} for an attempt "
                f"that published beliefs; mint a fresh id (e.g. target + ownership epoch) before "
                f"retrying, or beliefs from the previous attempt stay admissible"
            )

        by_id: Dict[str, TaskInfo] = {}
        for t in items:
            if not t.task_id or not t.target_id:
                raise ValueError(f"task_id and target_id must be non-empty: {t!r}")
            if t.task_id in by_id:
                raise ValueError(f"Duplicate task IDs detected: {t.task_id}")
            by_id[t.task_id] = t

        self._tasks = by_id
        self._episode_id = episode_id
        self.reset()
        _logger.info("Registered %d task(s) for episode %s: %s",
                     len(by_id), episode_id, sorted(by_id))

    def reset(self) -> None:
        """Clear all assignment and consensus state for the next independent attempt.

        Keeps the declared task set, the injected calculator and the episode bookkeeping
        (``_published_episode_ids``) that fences a replay; everything else is rebuilt, and the stats
        object is replaced rather than zeroed so a previous episode's counters cannot leak into a new
        tally. Clearing ``_episode_open`` is what makes ``register_tasks`` accept the next attempt's
        episode id, because ``reset`` also restores ``RUNNING`` and a status test alone could never
        tell "in progress" from "reset and idle".
        """
        self._bundle = []
        self._own_bids = {}
        self._winning_bids = {tid: 0.0 for tid in self._tasks}
        self._winning_agents = {tid: "" for tid in self._tasks}
        self._peer_bids = {}
        self._peer_winners = {}
        self._peer_agreement = {}
        self._seen_report_ids = set()
        self._last_seen_round = {}
        self._last_belief = None
        self._current_round = 0
        self._stable_rounds = 0
        self._status = ConsensusStatus.RUNNING
        self._episode_open = False
        self.stats = ConsensusStats()

    # ------------------------------------------------------------------ properties (compat surface)

    @property
    def bundle(self) -> List[str]:
        return list(self._bundle)

    @property
    def path(self) -> List[str]:
        """Execution path; identical to the bundle at ``L = 1`` (kept for interface compatibility)."""
        return list(self._bundle)

    @property
    def winning_bids(self) -> Dict[str, float]:
        return dict(self._winning_bids)

    @property
    def winning_agents(self) -> Dict[str, str]:
        return dict(self._winning_agents)

    @property
    def convergence_status(self) -> ConsensusStatus:
        return self._status

    @property
    def status(self) -> ConsensusStatus:
        return self._status

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def task_ids(self) -> List[str]:
        return sorted(self._tasks)

    @property
    def target_ids(self) -> Dict[str, str]:
        """Task to target declaration for this episode (REQ-206: carried, never inferred)."""
        return {tid: t.target_id for tid, t in self._tasks.items()}

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ------------------------------------------------------------------ phase 1: bundle construction

    def compute_bid(self, task_id: str) -> Optional[float]:
        """This agent's bid for ``task_id``, or ``None`` when it must abstain.

        ``None`` means "no bid exists for this task" and the task is left out of the belief. There
        is no fallback score: an absent telemetry input, a missing calculator or a failing
        computation all produce abstention, never a fabricated utility.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not registered")
        if self._bid_calculator is None or self._battery_pct is None:
            self._record_abstention(task_id)
            _logger.debug("Abstain on %s: no bid calculator or telemetry supplied", task_id)
            return None

        task = self._tasks[task_id]
        try:
            utility, _, _, _ = self._bid_calculator.compute_receiver_utility(
                point_camera=_as_vector(task.position),
                battery_pct=self._battery_pct,
            )
        except Exception as exc:  # abstain, never invent
            self._record_abstention(task_id)
            _logger.warning("Bid computation failed for task %s; abstaining: %s", task_id, exc)
            return None

        bid = _finite_bid(utility, self.utility_precision)
        if bid is None:
            self._record_abstention(task_id)
            _logger.warning("Bid for task %s was non-finite or out of [0, 1]; abstaining", task_id)
        return bid

    def _record_abstention(self, task_id: str) -> None:
        """Tally a task this agent declined to bid on, once per episode.

        Deliberately not memoised as "no bid": telemetry can arrive after the first round, and a
        cached abstention would suppress re-pricing for the rest of the episode.
        """
        if task_id not in self.stats.abstained_tasks:
            self.stats.abstained_tasks.append(task_id)

    def build_bundle(self) -> bool:
        """Greedy single-slot fill: take the best task where this agent strictly outbids the record.

        Returns:
            True if the bundle changed.
        """
        if len(self._bundle) >= self.max_bundle_size:
            return False

        best_task: Optional[str] = None
        best_bid = -1.0
        for task_id in sorted(self._tasks):
            if task_id in self._bundle:
                continue
            if task_id in self._own_bids:
                bid = self._own_bids[task_id]
            else:
                # Explicit lookup: ``dict.get(k, compute())`` would evaluate the calculator on every
                # call and re-tally abstentions for tasks already priced.
                bid = self.compute_bid(task_id)
            if bid is None:
                continue
            recorded_bid = self._winning_bids.get(task_id, 0.0)
            recorded_winner = self._winning_agents.get(task_id, "")
            if bid > recorded_bid + self.bid_epsilon:
                take = True
            else:
                # A peer claim can be adopted before this agent has priced anything, because a belief
                # may arrive before its first round and ``_merge`` takes a claim for an empty local
                # slot. A tied local bid then fails the strict-improvement gate and can never displace
                # that adopted claim, so the episode winner was decided by which agent published
                # first. The documented tie rule must be applied where the record is set, not only
                # where a peer claim arrives.
                #
                # ``recorded_winner`` must be non-empty: a zero bid on an unassigned task is not a
                # claim, and the belief self-consistency check in ``_admit`` rejects one.
                take = bool(recorded_winner) and recorded_winner != self.agent_id and \
                    self._prefer(self.agent_id, bid, recorded_winner, recorded_bid)
            if take and bid > best_bid:
                best_task, best_bid = task_id, bid

        if best_task is None:
            return False

        self._bundle.append(best_task)
        self._own_bids[best_task] = best_bid
        self._winning_bids[best_task] = best_bid
        self._winning_agents[best_task] = self.agent_id
        _logger.debug("Agent %s took task %s (bid=%.4f)", self.agent_id, best_task, best_bid)
        return True

    # ------------------------------------------------------------------ phase 2: consensus

    def create_consensus_message(self) -> ConsensusMessage:
        """Build the belief for this round and count one publish-call against the episode."""
        msg = self._build_belief()
        self.stats.beliefs_published += 1
        if self.stats.started_mono is None:
            self.stats.started_mono = msg.timestamp
        self._last_belief = msg
        self._published_episode_ids.add(self._episode_id)
        return msg

    def _build_belief(self) -> ConsensusMessage:
        """Build the current belief without touching publish counters."""
        return ConsensusMessage(
            sender_agent_id=self.agent_id,
            episode_id=self._episode_id,
            round_number=self._current_round,
            winning_bids=dict(self._winning_bids),
            winning_agents=dict(self._winning_agents),
            bundle=list(self._bundle),
            task_ids=self.task_ids,
            target_ids=self.target_ids,
            ownership_epoch=self.ownership_epoch,
            agreement_established=(self._status is ConsensusStatus.AGREED),
            timestamp=self._time(),
            belief_id=f"{self._episode_id}:{self.agent_id}:{self._current_round}",
        )

    def process_consensus_message(self, msg: ConsensusMessage) -> bool:
        """Validate then merge one inbound belief.

        Never raises for a protocol-level problem: a bad message is classified, counted and
        dropped, because an exception here would propagate out of the caller's receive path.

        Returns:
            True if local beliefs changed.
        """
        changed, _ = self._accept(msg)
        return changed

    def _accept(self, msg: ConsensusMessage) -> Tuple[bool, Optional[RejectionClass]]:
        changed, rej = self._admit(msg)
        if rej is not None:
            self.stats.reject(rej)
            _logger.debug("Agent %s dropped belief %s: %s",
                          self.agent_id, _describe(msg), rej.name)
        return changed, rej

    def _admit(self, msg: ConsensusMessage) -> Tuple[bool, Optional[RejectionClass]]:
        if self._status is not ConsensusStatus.RUNNING:
            return False, RejectionClass.POST_TERMINAL
        if not isinstance(msg, ConsensusMessage):
            return False, RejectionClass.MALFORMED
        if msg.schema_version != BELIEF_SCHEMA_VERSION:
            return False, RejectionClass.UNSUPPORTED_SCHEMA
        if not msg.sender_agent_id or not msg.episode_id:
            return False, RejectionClass.MALFORMED
        if msg.sender_agent_id == self.agent_id:
            return False, RejectionClass.SELF
        if msg.sender_agent_id not in self.participants:
            return False, RejectionClass.UNKNOWN_SENDER
        if msg.episode_id != self._episode_id:
            return False, RejectionClass.FOREIGN_EPISODE
        # Epoch binding (REQ-110): a belief is valid only for the ownership epoch its episode
        # started in, so a belief from before an abort-and-reclaim (e -> e+2) or from a restarted
        # adapter cannot be merged into this episode. Classified STALE, not a new rejection class:
        # the contract names it stale, and a caller that needs the distinction has the log line.
        if isinstance(msg.ownership_epoch, bool) or not isinstance(msg.ownership_epoch, int):
            return False, RejectionClass.MALFORMED
        if not isinstance(msg.agreement_established, bool):
            # A truthy int/str would silently read as "this peer agreed", so the report field is
            # type-checked rather than coerced.
            return False, RejectionClass.MALFORMED
        if msg.ownership_epoch != self.ownership_epoch:
            return False, RejectionClass.STALE
        if msg.round_number < 0:
            return False, RejectionClass.MALFORMED

        last = self._last_seen_round.get(msg.sender_agent_id, -1)
        if msg.round_number == last:
            return False, RejectionClass.DUPLICATE
        if msg.round_number < last:
            return False, RejectionClass.STALE

        if sorted(msg.task_ids) != sorted(self._tasks):
            return False, RejectionClass.TARGET_MISMATCH
        for task_id, target_id in msg.target_ids.items():
            known = self._tasks.get(task_id)
            if known is None or known.target_id != target_id:
                return False, RejectionClass.TARGET_MISMATCH

        # Vector self-consistency: a claim is (winner, bid > 0); an unassigned task is ("", 0).
        # Checked here so an inconsistent belief is classified and counted rather than merged partially.
        for task_id in self._tasks:
            bid = msg.winning_bids.get(task_id, 0.0)
            winner = msg.winning_agents.get(task_id, "")
            if not _valid_bid_value(bid):
                return False, RejectionClass.MALFORMED
            if winner and not float(bid) > 0.0:
                return False, RejectionClass.MALFORMED
            if not winner and float(bid) != 0.0:
                return False, RejectionClass.MALFORMED

        self._last_seen_round[msg.sender_agent_id] = msg.round_number
        self.stats.beliefs_delivered += 1
        return self._merge(msg), None

    def _merge(self, msg: ConsensusMessage) -> bool:
        """Merge peer vectors with the order-independent winner rule.

        For each task, the surviving claim is the better of the two under :meth:`_prefer` — a rule
        that depends only on the pair of claims, never on arrival order or on which claim is local.
        The peer's reported vectors are also retained verbatim for :meth:`check_agreement`.
        """
        changed = False
        for task_id in self._tasks:
            peer_bid = float(msg.winning_bids.get(task_id, 0.0))
            peer_winner = msg.winning_agents.get(task_id, "")
            mine_bid = self._winning_bids.get(task_id, 0.0)
            mine_winner = self._winning_agents.get(task_id, "")

            if not peer_winner:
                take = False
            elif not mine_winner:
                take = True
            else:
                take = self._prefer(peer_winner, peer_bid, mine_winner, mine_bid)

            if take:
                self._winning_bids[task_id] = peer_bid
                self._winning_agents[task_id] = peer_winner
                changed = True
                if task_id in self._bundle and peer_winner != self.agent_id:
                    self._release_task(task_id)

        self._peer_bids[msg.sender_agent_id] = {
            tid: float(msg.winning_bids.get(tid, 0.0)) for tid in self._tasks
        }
        self._peer_winners[msg.sender_agent_id] = {
            tid: str(msg.winning_agents.get(tid, "")) for tid in self._tasks
        }
        return changed

    def _prefer(self, a_id: str, a_bid: float, b_id: str, b_bid: float) -> bool:
        """True if claim ``(a_id, a_bid)`` should displace ``(b_id, b_bid)``.

        Symmetric and total: strictly higher bid wins; on an equal bid at the comparison precision
        the lexicographically smaller agent id wins. Because the rule depends only on the two
        claims and not on which one is "mine" or on arrival order, both agents reach the same
        winner — the defect this replaces left the tie branch unreachable and allowed two agents to
        each keep the task.
        """
        qa = round(float(a_bid), self.utility_precision)
        qb = round(float(b_bid), self.utility_precision)
        if qa != qb:
            return qa > qb
        return a_id < b_id

    def _release_task(self, task_id: str) -> None:
        """Single-task release. At ``L = 1`` there is no cascade, so none is ported."""
        if task_id in self._bundle:
            self._bundle.remove(task_id)
        self._own_bids.pop(task_id, None)
        if self._winning_agents.get(task_id) == self.agent_id:
            self._winning_agents[task_id] = ""
            self._winning_bids[task_id] = 0.0
        _logger.debug("Agent %s released task %s", self.agent_id, task_id)

    # ------------------------------------------------------------------ agreement predicate

    def check_agreement(self) -> bool:
        """Mutual-agreement predicate. Diagnostic; does not set status by itself.

        Satisfied only when every declared participant other than this agent has delivered at least
        one admitted belief **and** the winning bid **and** winning agent it reported equal this
        agent's vectors on every task. Two rules follow from that definition and are the reason it
        is not implemented as a round-number comparison:

        * A participant that has not reported is **not** counted as agreeing (silence is not
          assent), so a peer whose beliefs never arrive cannot by itself produce a false ``AGREED``.
        * Round numbers are per-process and not synchronised across agents, so they may legitimately
          differ; agreement is a property of the belief *contents*, which the merge rule makes
          order-independent.

        **What this cannot guarantee.** It is a local predicate over what peers last *told us* they
        believe — the standard CBAA consensus test — and it does not and cannot prove that a peer
        performed the same check, or that the peer's own view is current:

        * An agent that receives from everyone but is itself never heard satisfies this predicate on
          its own viewpoint while its peers stay non-converged. **This predicate cannot detect that;
          :meth:`check_distributed_agreement` can,** because it additionally requires each peer to
          have *reported* agreement on a belief it received from us. An adapter that calls only this
          method still has the blind spot, which is why design decision D3 puts the commit gate on
          the other predicate. Note what the reports do **not** provide: they are not a quorum, not a
          lease and not a failure detector — a silence-shaped hole is closed only for peers that are
          talking, and a peer that reports ``AGREED`` on a stale-but-matching snapshot still reports
          ``True``.
        * Under message loss a peer's latest belief may be older than ours, so agreement can be
          declared on a stale-but-matching snapshot; the epoch field bounds this across epochs, not
          within one.
        * ``AGREED`` is therefore evidence *about this agent's view of the episode*, never a
          distributed commitment. Whether the episode may be acted on is a decision the commit path
          owns, and the commit path today contains a measured dual-ownership defect
          (``EXP13_POST_IMPLEMENTATION_SAFETY_AUDIT.md`` §3.3, re-confirmed at
          ``EXP13_SAFETY_FINDINGS_20261020_210721.md``) unrelated to this module.
        """
        if not self._tasks:
            return False
        for other in self.participants:
            if other == self.agent_id:
                continue
            peer_bids = self._peer_bids.get(other)
            peer_winners = self._peer_winners.get(other)
            if peer_bids is None or peer_winners is None:
                return False
            for task_id in self._tasks:
                if round(peer_bids.get(task_id, 0.0), self.utility_precision) != \
                        round(self._winning_bids.get(task_id, 0.0), self.utility_precision):
                    return False
                if peer_winners.get(task_id, "") != self._winning_agents.get(task_id, ""):
                    return False
        return True

    def ingest_agreement_report(self, msg: ConsensusMessage) -> bool:
        """Record a peer's own agreement result, without merging its vectors.

        This is the acknowledgement leg of design decision D3, and it is deliberately narrow. An
        adapter calls it for every belief whose sender declared ``agreement_established``, whether or
        not this episode has terminated yet — a peer's terminating belief is normally in flight while
        this agent is still running its last round, and merging it without recording the declaration
        would lose the report. Nothing else changes: the merge rule, the round watermarks and the
        terminal status are untouched, and no extra message is published, because the terminating
        belief already carries the flag.

        Validation mirrors :meth:`_admit` minus the two gates that do not apply to a report: the
        round watermark (a report is deduplicated by ``belief_id``, not by round, since it is read
        after rounds stopped) and the vector self-consistency sweep (a report is not merged, so an
        inconsistent vector cannot corrupt local state through this path).

        Returns:
            True when a new report was recorded. False for anything refused or already seen; the
            reason is tallied in :attr:`ConsensusStats.rejected`, never raised, so a malformed
            inbound payload cannot break the caller's receive path.
        """
        if self._status not in (ConsensusStatus.RUNNING, ConsensusStatus.AGREED):
            # Only RUNNING and AGREED can carry a report. A bounded-out episode (MAX_ROUNDS, TIMEOUT,
            # LOCAL_STABLE, NO_WINNER, DUAL_CLAIM, ABORTED) is already disqualified, so a report from
            # it cannot arm anything and is tallied as POST_TERMINAL instead. Accepting a report while
            # still RUNNING is necessary because a peer's *terminating* belief routinely arrives one
            # round before this agent terminates: refusing it would throw the flag away and leave the
            # commit gate permanently un-openable. Recording early does not accelerate anything —
            # check_distributed_agreement() still requires this agent's own status to be AGREED.
            self.stats.reject(RejectionClass.POST_TERMINAL)
            return False
        if not isinstance(msg, ConsensusMessage):
            self.stats.reject(RejectionClass.MALFORMED)
            return False
        if msg.schema_version != BELIEF_SCHEMA_VERSION:
            self.stats.reject(RejectionClass.UNSUPPORTED_SCHEMA)
            return False
        if not msg.sender_agent_id or not msg.episode_id or not msg.belief_id:
            self.stats.reject(RejectionClass.MALFORMED)
            return False
        if msg.sender_agent_id == self.agent_id:
            self.stats.reject(RejectionClass.SELF)
            return False
        if msg.sender_agent_id not in self.participants:
            self.stats.reject(RejectionClass.UNKNOWN_SENDER)
            return False
        if msg.episode_id != self._episode_id:
            self.stats.reject(RejectionClass.FOREIGN_EPISODE)
            return False
        if isinstance(msg.ownership_epoch, bool) or not isinstance(msg.ownership_epoch, int):
            self.stats.reject(RejectionClass.MALFORMED)
            return False
        if msg.ownership_epoch != self.ownership_epoch:
            self.stats.reject(RejectionClass.STALE)
            return False
        if not isinstance(msg.agreement_established, bool):
            self.stats.reject(RejectionClass.MALFORMED)
            return False
        if msg.belief_id in self._seen_report_ids:
            self.stats.reject(RejectionClass.DUPLICATE)
            return False

        self._seen_report_ids.add(msg.belief_id)
        self._peer_agreement[msg.sender_agent_id] = msg.agreement_established
        self.stats.agreement_reports_accepted += 1
        _logger.debug("Agent %s recorded agreement report from %s: established=%s (%s)",
                      self.agent_id, msg.sender_agent_id, msg.agreement_established, msg.key())
        return True

    def check_distributed_agreement(self) -> bool:
        """Commit gate: this agent agreed **and** every declared peer says it agreed.

        Stronger than :meth:`check_agreement` in exactly one respect, and it is the predicate an
        adapter must consult before acting on an allocation: content agreement alone can be
        satisfied by an agent that everyone listens to and nobody is heard by.

        Requires, for every declared participant other than this agent, an admitted report with
        ``agreement_established == True`` bound to this episode and this ownership epoch. A peer that
        is silent is **not** treated as agreeing, so the gate fails safe: it can delay or veto a
        commit, never accelerate one. A peer that reports ``False`` holds the gate open until it
        either reports ``True`` or the episode's bound expires.

        **What this still is not.** It is two rounds of belief traffic, not a distributed-commit
        primitive. It does not survive a peer that agrees then aborts unilaterally, and it is not a
        proof that any peer reached a decision *about this agent* — it says each reported agreement
        under the same ``(episode_id, ownership_epoch)`` pair. Ownership itself is still settled by
        the confirm/ACK exchange, never by this predicate (design decision D3, level 4).
        """
        if self._status is not ConsensusStatus.AGREED:
            return False
        for other in self.participants:
            if other == self.agent_id:
                continue
            if self._peer_agreement.get(other) is not True:
                return False
        return True

    def peers_reported_agreement(self) -> List[str]:
        """Declared peers whose recorded report says ``True`` — for logs and for the commit window."""
        return sorted(p for p in self.participants
                      if p != self.agent_id and self._peer_agreement.get(p) is True)

    def _find_dual_claim(self) -> Optional[str]:
        """Check the two ownership invariants this agent can actually observe. Returns a violation
        description, or ``None`` when both hold.

        Checking for two values against one task id in ``winning_agents`` would be dead code — a dict
        cannot hold two winners per key. These checks are reachable and are the ones that matter for
        handoff safety:

        1. **Self-contradiction**: this agent keeps a task in its bundle while its own belief records
           a different agent as that task's winner. The earlier revision's tie handling made exactly
           this state possible and still reported convergence.
        2. **Target-level dual ownership**: this agent bundles a task whose ``target_id`` is recorded
           as won by another agent. One target must have one owner even when several task slots refer
           to the same target, which is why ``target_id`` is carried explicitly (REQ-206).
        """
        for task_id in self._bundle:
            winner = self._winning_agents.get(task_id, "")
            if winner and winner != self.agent_id:
                return (f"agent {self.agent_id} still bundles {task_id} but believes "
                        f"{winner} wins it")
            target_id = self._tasks[task_id].target_id
            for other_task, other_winner in self._winning_agents.items():
                if other_winner and other_winner != self.agent_id and \
                        other_task in self._tasks and \
                        self._tasks[other_task].target_id == target_id:
                    return (f"target {target_id} bundled by {self.agent_id} via {task_id} "
                            f"and won by {other_winner} via {other_task}")
        return None

    def abort(self, reason: str) -> None:
        """Cancel the episode from outside (e.g. the handoff deadline passed while negotiating).

        Terminal and non-converged: an aborted episode must never be read as a successful consensus,
        and its convergence time stays ``None``.
        """
        if self._status is ConsensusStatus.RUNNING:
            self._finish(ConsensusStatus.ABORTED, reason or "aborted")

    # ------------------------------------------------------------------ round driver

    def run_round(self, incoming_messages: Sequence[ConsensusMessage]) -> Tuple[bool, ConsensusMessage]:
        """Execute one round: price the tasks, ingest peer beliefs, evaluate termination.

        **Phase 1 runs before ingestion, on purpose.** Inverting these two steps is what made the
        episode winner depend on which agent published first: an agent that ingested a peer's claim
        while its own ``winning_agents`` entry was still empty took that claim wholesale (no local
        claim existed to compare against), and its own equal bid was then suppressed by the
        positive-improvement gate, so the tie rule in :meth:`_prefer` was never consulted. Pricing
        first records the local bid, which turns the same merge into a symmetric comparison and makes
        the outcome a function of the claims rather than of the schedule.

        Unlike the earlier revision, a terminated negotiator does **not** raise — it absorbs and
        tallies, so a late message cannot crash the caller's receive path.

        Publish accounting is deliberately asymmetric: a round that completes emits a belief and is
        counted in :attr:`ConsensusStats.beliefs_published`, while a censored round (round bound,
        deadline, or an already-terminal episode) emits nothing and returns the last belief
        unchanged. Otherwise a failed episode would inflate the network-message tally.

        Returns:
            ``(changed_this_round, outgoing_belief)``. ``changed_this_round`` is True when this
            agent's own bundle or winning vector moved **or** an accepted peer belief moved it, so
            it feeds the LOCAL_STABLE counter rather than answering "did a peer do anything". Round
            1 is therefore always ``True`` — the agent has just recorded its own bid.
        """
        if self._status is not ConsensusStatus.RUNNING:
            self.stats.reject(RejectionClass.POST_TERMINAL)
            return False, self._last_belief if self._last_belief is not None else self._build_belief()

        if self.stats.started_mono is None:
            self.stats.started_mono = self._time()

        if self._deadline_exceeded():
            self._finish(ConsensusStatus.TIMEOUT, "declared monotonic deadline exceeded before round completion")
            return False, self._last_belief if self._last_belief is not None else self._build_belief()

        if self._current_round >= self.max_rounds:
            self._finish(ConsensusStatus.MAX_ROUNDS, f"round bound {self.max_rounds} reached without agreement")
            return False, self._last_belief if self._last_belief is not None else self._build_belief()

        self._current_round += 1
        self._episode_open = True
        self.stats.rounds_executed = self._current_round

        changed = self.build_bundle()

        for msg in incoming_messages:
            round_changed, _ = self._accept(msg)
            changed = changed or round_changed

        if changed:
            self._stable_rounds = 0
        else:
            self._stable_rounds += 1

        conflict = self._find_dual_claim()
        if conflict is not None:
            self._finish(ConsensusStatus.DUAL_CLAIM, f"safety invariant violated: {conflict}")
        elif self.check_agreement():
            if self._current_round < MIN_AGREEMENT_ROUND:
                _logger.debug("Agent %s: belief match at round %d withheld until round %d "
                              "(AGREEMENT_REACHED clause iv)",
                              self.agent_id, self._current_round, MIN_AGREEMENT_ROUND)
            else:
                assigned = [t for t, w in self._winning_agents.items() if w]
                if assigned:
                    self._finish(ConsensusStatus.AGREED,
                                 "every declared participant reported vectors matching this agent's")
                else:
                    self._finish(ConsensusStatus.NO_WINNER, "agreement reached with every task unassigned")
        elif self._stable_rounds >= self.convergence_threshold:
            self._finish(ConsensusStatus.LOCAL_STABLE,
                         f"{self._stable_rounds} unchanged rounds without a mutual-agreement match")

        return changed, self.create_consensus_message()

    def _deadline_exceeded(self) -> bool:
        if self.deadline_s is None or self.stats.started_mono is None:
            return False
        return (self._time() - self.stats.started_mono) > self.deadline_s

    def _finish(self, status: ConsensusStatus, reason: str) -> None:
        self._status = status
        self.stats.outcome = status
        self.stats.outcome_reason = reason
        if status is ConsensusStatus.AGREED:
            self.stats.agreed_mono = self._time()
        if status is ConsensusStatus.DUAL_CLAIM:
            level = logging.ERROR
        elif status in (ConsensusStatus.AGREED, ConsensusStatus.NO_WINNER):
            level = logging.INFO
        else:
            level = logging.WARNING
        _logger.log(level, "Agent %s episode %s -> %s: %s (round %d)",
                    self.agent_id, self._episode_id, status.name, reason, self._current_round)

    # ------------------------------------------------------------------ results

    def get_assignment(self) -> Dict[str, str]:
        """Task to winning agent for tasks some agent claimed; unassigned tasks are omitted."""
        return {tid: w for tid, w in self._winning_agents.items() if w}

    def get_target_assignment(self) -> Dict[str, str]:
        """Target to winning agent — the advisory output an adapter would use to address a request."""
        return {self._tasks[tid].target_id: w for tid, w in self._winning_agents.items()
                if w and tid in self._tasks}

    def is_converged(self) -> bool:
        """True **only** for :attr:`ConsensusStatus.AGREED`.

        The earlier revision returned True for any non-running state, which made ``MAX_ROUNDS`` and
        ``TIMEOUT`` count as convergence. Callers that need "is this episode over" must use
        :meth:`is_terminal`.
        """
        return self._status is ConsensusStatus.AGREED

    def is_terminal(self) -> bool:
        return self._status is not ConsensusStatus.RUNNING

    def convergence_time_s(self) -> Optional[float]:
        """Monotonic single-process elapsed time from first belief to agreement, else ``None``."""
        return self.stats.convergence_time_s

    def metrics(self) -> Dict[str, object]:
        """Flat, unit-laboured snapshot for a trial record."""
        return {
            "protocol_id": self.PROTOCOL_ID,
            "algorithm_variant": self.ALGORITHM_VARIANT,
            "agent_id": self.agent_id,
            "episode_id": self._episode_id,
            "participants": list(self.participants),
            "ownership_epoch": self.ownership_epoch,
            "target_ids": sorted({t.target_id for t in self._tasks.values()}),
            "task_ids": sorted(self._tasks),
            "status": self._status.name,
            "outcome_reason": self.stats.outcome_reason,
            "rounds_executed": self.stats.rounds_executed,
            "peers_observed": sum(1 for p in self.participants
                                  if p != self.agent_id and p in self._peer_bids),
            "peers_declared": len(self.participants) - 1,
            "peers_reported_agreement": self.peers_reported_agreement(),
            "agreement_reports_recorded": self.stats.agreement_reports_accepted,
            "distributed_agreement": self.check_distributed_agreement(),
            "beliefs_published": self.stats.beliefs_published,
            "beliefs_delivered": self.stats.beliefs_delivered,
            "rejected_total": self.stats.total_rejected,
            "rejected_by_class": {k.name: v for k, v in sorted(self.stats.rejected.items(), key=lambda kv: kv[0].name)},
            "convergence_time_s": self.stats.convergence_time_s,
            "clock_basis": "monotonic-single-process" if self.stats.convergence_time_s is not None else "not_applicable",
            "assignment": self.get_assignment(),
        }

    def __repr__(self) -> str:
        return (f"CBBANegotiator(agent={self.agent_id!r}, episode={self._episode_id!r}, "
                f"round={self._current_round}/{self.max_rounds}, status={self._status.name}, "
                f"bundle={self._bundle})")


def _as_vector(position: Sequence[float]):
    """Return the position as a float array, using the caller's numpy if it is already loaded."""
    np = _numpy()
    values = [float(c) for c in position]
    if np is not None:
        return np.array(values, dtype=np.float64)
    return values


def _numpy():
    try:
        import numpy  # optional: only needed when the injected calculator expects an array
    except ImportError:
        return None
    return numpy


def _describe(msg: object) -> str:
    """Identify a payload for a log line without assuming it is a well-formed belief.

    Called eagerly (an argument is evaluated before the log call), so a foreign payload — e.g. a
    plain dict from an adapter that never built a ConsensusMessage — must not raise here.
    """
    if isinstance(msg, ConsensusMessage):
        return msg.key()
    return f"<{type(msg).__name__}>"


def _valid_bid_value(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    v = float(value)
    return v == v and abs(v) != float("inf") and 0.0 <= v <= 1.0


def _finite_bid(value: object, precision: int) -> Optional[float]:
    if not _valid_bid_value(value):
        return None
    return round(float(value), precision)
