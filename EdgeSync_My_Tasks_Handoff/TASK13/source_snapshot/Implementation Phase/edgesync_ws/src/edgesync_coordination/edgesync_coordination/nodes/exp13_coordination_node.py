"""Experiment 13 coordination node — the runtime selection path for the CBAA arm.

WHAT THIS FILE IS FOR
=====================
``coordination_node.py`` is the deployed one-round Request/Response baseline and stays byte-identical
(design decision D6/D7). This subclass adds exactly four things and no others:

1. a ``negotiation_protocol`` parameter that selects between ``cnp`` (the baseline, and the default)
   and ``cbba`` (this experiment);
2. the belief transport: a ``ConsensusBelief`` publisher, subscriber and round timer wired to
   :class:`Exp13CbbaAdapter`, plus the commit decision applied to the FSM the baseline already owns;
3. the admission guards recorded as M0/M1/M2 in ``EXP13_SAFETY_FINDINGS_*.md``, which the baseline
   does not perform;
4. the Exp-7 signature intake: a ``ReidSignature`` subscriber that installs the reference embedding
   through the baseline's own ``set_reid_signature()`` (which has no caller in the baseline), so a
   ``cbba`` trigger can carry a real embedding while a ``cnp`` one still carries the zeros.

Everything else — telemetry, TF, the eligibility gate, the utility engine, the watchdog, the
receiver-side request handling, the confirm/ACK leg — is inherited untouched. ``UtilityEngine`` is
the baseline's own instance, so the ablation changes the allocator and nothing else.

WHAT SELECTION ACTUALLY CHANGES (read this before quoting a result)
===================================================================
Under ``cbba`` the *owner's release* is decided by consensus plus the peers' agreement reports, not by
which candidate answered first. The candidate-side contract is **not** changed: an eligible peer still
answers the trigger with ACCEPT immediately, exactly as the baseline does, because deferring that
answer would alter a timing behaviour that Experiments 12/14 and the deployed stack rely on. So an
awarded candidate proceeds as it already has, and a non-awarded one simply is not handed the target.
Gating the candidate's *acquisition* on consensus would require a deferred-acceptance protocol
change; that change was considered and deliberately not taken in this pass.

SAFETY STATE
============
This node cannot make ownership transfer safe. ``EXP13_SAFETY_FINDINGS_*.md`` records the measured
dual-OWNER path below the commit seam and that ``ReceiverState.AWAITING_ACK`` has no production
producer, so no configuration of this node completes an ownership transfer in the current tree. The
guards here reject more than the baseline does; they do not by themselves prove safety, and they have
not been executed against a ROS 2 graph on this host.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from edgesync_msgs.msg import ConsensusBelief, HandoffConfirmAck, HandoffHeader, HandoffResponse, \
    OwnershipEvent, ReidSignature

from ..fsm.ownership_fsm import OwnershipRole
from ..negotiation.exp13_cbba_adapter import (
    BASELINE_PROTOCOL_ID,
    PROTOCOL_ID,
    CommitAction,
    CommitDecision,
    Exp13CbbaAdapter,
    ProtocolConfigError,
    resolve_negotiation_protocol,
)
from .coordination_node import CoordinationNode

BELIEF_TOPIC_SUFFIX = 'consensus_belief'

# Where Exp-7 puts the smoothed target signature, and how old one may be before this node refuses to
# put it on the wire. 2000 ms is the SignatureSmoother's own documented flush horizon (3 frames at
# 2 Hz = 1.5 s) plus one frame of slack; anything older than that is a description of a target the
# producer has already given up on.
REID_SIGNATURE_TOPIC_DEFAULT = '/edgesync/identity/reid_signature'
REID_SIGNATURE_MAX_AGE_MS_DEFAULT = 2000.0
# How far a producer's stamp may sit ahead of this node's own clock before the signature is refused.
# The field is the producer's clock domain, so it is not comparable across hosts without a time error
# model; this tolerance is the only cross-host assumption the arm makes, and it is a refusal
# threshold, never a freshness calculation.
REID_SIGNATURE_CLOCK_SKEW_TOLERANCE_NS = 100_000_000

RESPONSE_REASON_NAMES = {0: 'OK', 1: 'BUSY_CPU', 2: 'LOW_BATTERY', 3: 'NAV_UNAVAILABLE',
                         4: 'LINK_STALE', 5: 'TIME_INSUFFICIENT', 6: 'OUT_OF_REACH'}


class Exp13CoordinationNode(CoordinationNode):
    """Baseline coordination node with a selectable CBAA (L = 1) negotiation arm."""

    def __init__(self, fsm: Optional[Any] = None, node_name: str = 'exp13_coordination_node'):
        # Every attribute an override reads is installed *before* super().__init__() runs, because the
        # parent constructor publishes its initial ownership/heartbeat messages and a subscription
        # callback can be delivered re-entrantly (intra-process comms, a concurrent executor, and the
        # synchronous fake middleware this arm is tested with). An override that raised AttributeError
        # there would abort node construction, so the guards must have something to read.
        self.adapter: Optional[Exp13CbbaAdapter] = None
        self.pub_belief: Dict[str, Any] = {}
        self.sub_belief = None
        self.peers: list = []
        self.timer_round = None
        # Seconds, taken from belief_round_period_ms when the adapter is built; 0.0 is the control
        # arm, which has no round driver.
        self._round_period_s = 0.0
        self._exp13_episode_target = ''
        self._exp13_heartbeat_seen = False
        self.exp_id = ''
        self.negotiation_protocol = BASELINE_PROTOCOL_ID
        self.sub_signature = None
        self._exp13_sig_state: Optional[Dict[str, Any]] = None
        self._exp13_sig_max_age_ms = 0.0
        self._exp13_sig_required = False
        self._exp13_sig_counts = {
            'received': 0, 'applied': 0, 'invalidated': 0, 'dropped_before_broadcast': 0,
            'refused_foreign_sender': 0, 'refused_dimension': 0, 'refused_invalid': 0,
            'refused_target_mismatch': 0, 'refused_out_of_order': 0, 'refused_future_stamp': 0,
        }

        super().__init__(fsm=fsm, node_name=node_name)

        # Declared here, after the parent has built the node: rclpy cannot declare on a Node that does
        # not exist yet, which is why the selector is resolved below and not before super().
        self.declare_parameter('negotiation_protocol', BASELINE_PROTOCOL_ID)
        self.declare_parameter('exp_id', '')
        self.declare_parameter('participant_ids', ['ugv_1', 'ugv_2'])
        self.declare_parameter('swarm_participant_count', 2)
        self.declare_parameter('target_count', 1)
        self.declare_parameter('max_bundle_size', 1)
        self.declare_parameter('max_rounds', 4)
        self.declare_parameter('convergence_threshold', 2)
        self.declare_parameter('bid_epsilon', 1e-6)
        self.declare_parameter('utility_precision', 4)
        self.declare_parameter('deadline_s', 1.0)
        self.declare_parameter('belief_round_period_ms', 250)
        self.declare_parameter('commit_grace_ms', 500)
        self.declare_parameter('require_distributed_agreement', True)
        # ES-EXP07 <-> ES-EXP13: the reference embedding a handoff must carry. Read-only intake —
        # the arm subscribes to Exp-7's signature and installs it through the inherited setter, so
        # coordination_node.py itself is not touched and the control arm never subscribes at all.
        self.declare_parameter('reid_signature_topic', REID_SIGNATURE_TOPIC_DEFAULT)
        self.declare_parameter('reid_signature_max_age_ms', REID_SIGNATURE_MAX_AGE_MS_DEFAULT)
        self.declare_parameter('require_valid_reid_signature', False)

        self.exp_id = self.get_parameter('exp_id').get_parameter_value().string_value

        # Selector first, and it raises rather than defaulting: a typo in an experiment configuration
        # must not quietly run a different experiment (brief D6).
        raw_protocol = self.get_parameter('negotiation_protocol').get_parameter_value().string_value
        try:
            self.negotiation_protocol = resolve_negotiation_protocol(raw_protocol)
        except ProtocolConfigError as exc:
            self.get_logger().fatal(f'[EXP13] refusing to start: {exc}')
            raise

        if self.negotiation_protocol == PROTOCOL_ID:
            self._build_adapter()
        else:
            self.get_logger().info(
                f'[EXP13] node={node_name} robot_id={self.robot_id} negotiation_protocol='
                f'{BASELINE_PROTOCOL_ID}: running the inherited one-round baseline path. No belief '
                f'topic, no round timer, no adapter — this is the control arm.'
            )

    # ------------------------------------------------------------------ construction

    def _param_config(self) -> Dict[str, Any]:
        """The Exp-13 parameter block, read off this node's own declared parameters.

        Read from the node and not from the YAML file so that what the adapter uses is exactly what
        this process was configured with — including anything overridden on the command line.
        """
        gp = self.get_parameter
        deadline = gp('deadline_s').get_parameter_value().double_value
        return {
            'max_bundle_size': int(gp('max_bundle_size').get_parameter_value().integer_value),
            'max_rounds': int(gp('max_rounds').get_parameter_value().integer_value),
            'convergence_threshold': int(gp('convergence_threshold').get_parameter_value().integer_value),
            'bid_epsilon': float(gp('bid_epsilon').get_parameter_value().double_value),
            'utility_precision': int(gp('utility_precision').get_parameter_value().integer_value),
            'swarm_participant_count': int(gp('swarm_participant_count').get_parameter_value().integer_value),
            'target_count': int(gp('target_count').get_parameter_value().integer_value),
            # A non-positive deadline means "bounded only by max_rounds", which is what the YAML
            # documents; a double parameter cannot carry None, so the sentinel is converted here.
            'deadline_s': None if deadline <= 0.0 else deadline,
            'belief_round_period_ms': int(gp('belief_round_period_ms').get_parameter_value().integer_value),
            'commit_grace_ms': int(gp('commit_grace_ms').get_parameter_value().integer_value),
            'require_distributed_agreement': bool(
                gp('require_distributed_agreement').get_parameter_value().bool_value),
        }

    def _build_adapter(self) -> None:
        config = self._param_config()
        participants = [str(p) for p in
                        self.get_parameter('participant_ids').get_parameter_value().string_array_value]
        if self.robot_id not in participants:
            participants = [self.robot_id] + participants
            self.get_logger().warn(
                f'[EXP13] participant_ids did not include this node ({self.robot_id}); it was added, '
                f'and the set is now {participants}. A launch that names a participant set without '
                f'its own robot is a configuration error, not something to run silently.')
        self.adapter = Exp13CbbaAdapter(
            agent_id=self.robot_id,
            participant_ids=participants,
            config=config,
            # The baseline's own scorer: the ablation must not change how utility is computed.
            utility_engine=self.utility,
            publish_belief=self._publish_belief,
            stamp_fn=lambda: int(self.get_clock().now().nanoseconds),
        )

        qos = self._belief_qos()
        # Directed, exactly like the baseline's negotiation publishers: publish to each peer's topic,
        # subscribe on our own (coordination_node.py:160-207). A self-named publisher would only ever
        # connect under robot_bringup's /edgesync_swarm remap and would be dead in the Tier-1
        # simulation launch, where /ugv_1/... and /ugv_2/... are distinct absolute topics.
        self.peers = [p for p in self.adapter.participants if p != self.robot_id]
        self.pub_belief = {
            peer: self.create_publisher(
                ConsensusBelief, f'/{peer}/edgesync/{BELIEF_TOPIC_SUFFIX}', qos)
            for peer in self.peers
        }
        self.sub_belief = self.create_subscription(
            ConsensusBelief, f'/{self.robot_id}/edgesync/{BELIEF_TOPIC_SUFFIX}',
            self._on_belief_msg, qos)

        period_s = max(0.001, config['belief_round_period_ms'] / 1000.0)
        self._round_period_s = period_s
        self._start_round_timer()
        self._wire_signature_intake()

        self.get_logger().info(
            f'[EXP13] negotiation_protocol={PROTOCOL_ID} armed: agent={self.robot_id} '
            f'participants={participants} max_rounds={config["max_rounds"]} '
            f'round_period_ms={config["belief_round_period_ms"]} '
            f'commit_grace_ms={config["commit_grace_ms"]} '
            f'require_distributed_agreement={config["require_distributed_agreement"]} '
            f'exp_id={self.exp_id or "(unset)"} subscribe=/{self.robot_id}/edgesync/'
            f'{BELIEF_TOPIC_SUFFIX} publish={self.peers} '
            f'(one belief per round per peer, so a 2-participant episode counts 1 publish per '
            f'round). Algorithm is CBAA at bundle length L = 1, not full CBBA.'
        )

    @staticmethod
    def _belief_qos():
        """RELIABLE, KEEP_LAST(10): a lost belief is a non-converged episode, so do not risk it.

        ``BEST_EFFORT`` is what the baseline uses for 10 Hz heartbeats, where dropping stale samples
        is correct. Consensus is the opposite case — every vector has to be seen at least once.
        """
        return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=10)

    # ------------------------------------------------------------------ transport

    def _publish_belief(self, payload: Dict[str, Any]) -> None:
        """Wire-shaped dict → ``ConsensusBelief``, sent to every declared peer. Mechanical here.

        One dict becomes one message object shared by all publishers: the peers must see the *same*
        belief, and a per-peer copy would let a bug give two peers different vectors under one
        ``belief_id`` — which the negotiator would then deduplicate and drop.
        """
        h = payload['header']
        msg = ConsensusBelief()
        msg.header.protocol_version = int(h['protocol_version'])
        msg.header.message_type = int(h['message_type'])
        msg.header.sender_id = str(h['sender_id'])
        msg.header.receiver_id = str(h['receiver_id'])
        msg.header.transaction_id = str(h['transaction_id'])
        msg.header.target_id = str(h['target_id'])
        msg.header.ownership_epoch = int(h['ownership_epoch'])
        # header.sequence is the round number, set by the adapter — the node adds no counter of its
        # own, because two counters disagreeing is a second, competing ordering story.
        msg.header.sequence = int(h['sequence'])
        msg.header.stamp_utc_ns = int(h['stamp_utc_ns'])
        msg.schema_version = int(payload['schema_version'])
        msg.belief_id = str(payload['belief_id'])
        msg.round_number = int(payload['round_number'])
        msg.task_ids = [str(t) for t in payload['task_ids']]
        msg.target_ids = [str(t) for t in payload['target_ids']]
        msg.winning_bids = [float(b) for b in payload['winning_bids']]
        msg.winning_agents = [str(w) for w in payload['winning_agents']]
        msg.bundle = [str(b) for b in payload['bundle']]
        msg.agreement_established = bool(payload['agreement_established'])
        for pub in self.pub_belief.values():
            pub.publish(msg)

    def _belief_to_mapping(self, msg: ConsensusBelief) -> Dict[str, Any]:
        """``ConsensusBelief`` → the dict shape the adapter already validates.

        Explicit, field by field, because a generated ``get_fields_and_field_types()`` loop would
        coerce ``float32`` against ``float64`` and array shapes invisibly. All validation belongs to
        ``belief_from_mapping`` and nowhere else.
        """
        return {
            'header': {
                'protocol_version': int(msg.header.protocol_version),
                'message_type': int(msg.header.message_type),
                'sender_id': str(msg.header.sender_id),
                'receiver_id': str(msg.header.receiver_id),
                'transaction_id': str(msg.header.transaction_id),
                'target_id': str(msg.header.target_id),
                'ownership_epoch': int(msg.header.ownership_epoch),
                'sequence': int(msg.header.sequence),
                'stamp_utc_ns': int(msg.header.stamp_utc_ns),
            },
            'schema_version': int(msg.schema_version),
            'belief_id': str(msg.belief_id),
            'round_number': int(msg.round_number),
            'task_ids': [str(t) for t in msg.task_ids],
            'target_ids': [str(t) for t in msg.target_ids],
            'winning_bids': [float(b) for b in msg.winning_bids],
            'winning_agents': [str(w) for w in msg.winning_agents],
            'bundle': [str(b) for b in msg.bundle],
            'agreement_established': bool(msg.agreement_established),
        }

    def _on_belief_msg(self, msg: ConsensusBelief) -> None:
        if self.adapter is None:
            return
        if msg.header.sender_id == self.robot_id:
            # Our own copy, echoed back by the shared topic. The negotiator would classify it SELF;
            # dropping it here keeps that tally about peers.
            return
        applied, note = self.adapter.on_belief(self._belief_to_mapping(msg))
        if not applied:
            self.get_logger().debug(f'[EXP13] belief from {msg.header.sender_id} '
                                    f'round {msg.round_number} not applied: {note}')

    # ------------------------------------------------------------------ episode driving

    def _drive_round(self) -> None:
        """One timer tick = one consensus round, then one look at the commit decision."""
        if self.adapter is None:
            return
        # No episode yet. The arm is built before the first trigger, and driving an empty adapter would
        # hand back a closing ABORT/STANDBY decision — which would stop this very timer and, for a node
        # that starts as OWNER, run the recovery path and bump the epoch with no attempt behind it.
        #
        # A terminal episode is refused for the same reason one tick later than that: re-applying the
        # cached verdict would run ``abort_to_recovery()`` a second time and advance the epoch twice for
        # one failed attempt. The timer is stopped when the episode closes, so this is the guard for a
        # tick already in flight — and for a driver invoked by hand, which is how tests re-drive it.
        if self.adapter.negotiator is None or self.adapter.negotiator.is_terminal():
            return
        self.adapter.set_telemetry(self._live_battery_pct())
        self.adapter.run_round()
        decision = self.adapter.arm_commit(self._is_owner())
        if decision.closes_episode:
            self._stop_round_timer()
            self._apply_commit(decision)

    def _is_owner(self) -> bool:
        return self.fsm.role == OwnershipRole.OWNER

    def _live_battery_pct(self) -> Optional[float]:
        """The node's battery, or ``None`` until an onboard heartbeat has actually produced one.

        ``_my_telemetry.battery_pct`` starts at the declared ``battery_pct`` parameter (the baseline's
        own 85.0 stand-in, ``coordination_node.py:62``). DEF-01 says a stand-in must not decide an
        allocation, so bidding is enabled only once a StateHeartbeat has been seen. Until then every
        agent abstains and the episode ends NO_WINNER — a visible, honest failure, not a fake bid.
        """
        if not self._exp13_heartbeat_seen:
            return None
        return float(self._my_telemetry.battery_pct)

    def _stop_round_timer(self) -> None:
        if self.timer_round is not None:
            self.timer_round.cancel()
            self.destroy_timer(self.timer_round)
            self.timer_round = None

    def _start_round_timer(self) -> None:
        """Arm (or re-arm) the round driver for the episode that is now open.

        ``_stop_round_timer`` destroys the timer, so an episode that closes leaves the node with no
        driver at all: the next trigger would open a fresh episode whose rounds nobody ran, and the
        owner's FSM would sit in ``NEGOTIATING`` until the watchdog reclaimed it. Re-arming here is what
        makes a second handoff attempt in the same process negotiate at all.
        """
        if self.adapter is None or self._round_period_s <= 0.0:
            return
        self._stop_round_timer()
        self.timer_round = self.create_timer(self._round_period_s, self._drive_round)

    # ------------------------------------------------------------------ the commit seam

    def _apply_commit(self, decision: CommitDecision) -> None:
        """Map a :class:`CommitDecision` onto the baseline FSM. ``COMMIT`` is the only transfer."""
        action = decision.action
        self.get_logger().info(
            f'[EXP13] episode {decision.episode_id} action={action.name} '
            f'winner={decision.receiver_id or "-"} rounds={decision.rounds} '
            f'beliefs={decision.beliefs_published} :: {decision.reason}')

        if decision.closes_episode:
            # Invariant, not a caller's courtesy: whatever applied a closing decision must also stop
            # the driver, or the next tick re-applies the same verdict to a transaction that has moved
            # on. ``_drive_round`` relied on doing this itself.
            self._stop_round_timer()
            # ...and the episode itself. The adapter refuses to open a second episode while the first
            # is RUNNING, so a closing decision that left it open would make every later handoff
            # trigger in this process fail here and negotiate nothing. ``CBBANegotiator.abort()`` is a
            # no-op unless the status is still RUNNING, so a verdict that came out of ``run_round``
            # keeps the status and reason it already earned.
            if self.adapter is not None:
                self.adapter.abort(f"closing decision applied: {action.name}")

        if action is CommitAction.WAIT:
            return

        if action in (CommitAction.COMMIT, CommitAction.ACQUIRE):
            # M4: a consensus verdict is only authoritative for the transaction the FSM is actually
            # in. A superseded or already-closed episode must not be able to hand the target over or
            # tell a standby to start acquisition.
            active = self.fsm.active_transaction_id or ''
            if decision.episode_id != active:
                self.get_logger().error(
                    f'[EXP13] refusing to apply {action.name}: episode {decision.episode_id!r} is not '
                    f'the FSM\'s active transaction {active!r}. No transfer, no acquisition.')
                self._close_attempt_without_transfer(
                    f'episode/transaction mismatch ({decision.episode_id} != {active})')
                return

        if action is CommitAction.COMMIT:
            # The three lines the baseline's selection step performs
            # (coordination_node.py:542/551/554), reached through consensus instead of through the
            # first ACCEPT to arrive.
            self.selected_receiver_id = decision.receiver_id
            self.fsm.on_peer_accepted()
            self.watchdog.arm()
            self.get_logger().info(
                f'[EXP13] COMMIT: {self.robot_id} will hand {self._exp13_episode_target} to '
                f'{decision.receiver_id}; sender FSM committed and watchdog armed.')
            return

        if action is CommitAction.ACQUIRE:
            self.get_logger().info(
                f'[EXP13] ACQUIRE: consensus awarded {self.robot_id} the target; the receiver-side '
                f'acquisition and confirm legs stay exactly as the baseline runs them.')
            return

        if action is CommitAction.RETAIN:
            self._close_attempt_without_transfer('RETAIN: this agent holds the best claim')
            return

        if action is CommitAction.UNSAFE:
            # Every participant reports a dual claim, but only the one holding the target may run the
            # recovery path: abort_to_recovery() sets role = OWNER (ownership_fsm.py:126-135), so
            # letting a standby robot run it would *create* the second owner the violation is about.
            self._close_attempt_without_transfer(f'UNSAFE: {decision.reason}')
            self.get_logger().error(
                f'[EXP13] UNSAFE — dual claim reported by the negotiator: {decision.reason}. No '
                f'transfer performed. Record this episode as a safety violation, not a timeout.')
            return

        if action is CommitAction.ABORT:
            self._close_attempt_without_transfer(decision.reason)
            return

        # STANDBY: neither owner nor winner, so nothing is asked of this node.
        self.get_logger().info(f'[EXP13] STANDBY: {decision.reason}')

    def _on_watchdog_timeout(self) -> None:
        """End the consensus episode *before* the baseline reclaims the target.

        The cutoff reclaims ownership at epoch e+2 and returns the sender FSM to solo tracking
        (``coordination_node.py:289-294``) — a path that exists precisely because a candidate went
        silent. Under ``cbba`` the round driver is still ticking when that happens, so without this
        the node would keep publishing beliefs for a transaction it has just given up on, and would
        keep a RUNNING episode that blocks the next handoff trigger. M4 is what stops a late verdict
        from transferring anything, and a last line of defence is not a plan.

        The closing decision is deliberately *not* applied here: ``abort_to_recovery()`` in the
        inherited handler owns the epoch, and running both would advance it twice for one failure.
        """
        if self.adapter is None:
            return super()._on_watchdog_timeout()
        episode = self.adapter.episode_id
        self._stop_round_timer()
        if episode:
            self.adapter.abort("watchdog cutoff: candidate silent", is_owner=self._is_owner())
            self.negotiation.close_transaction(episode, "watchdog cutoff")
            self.get_logger().warn(
                f'[EXP13] episode {episode} aborted by the watchdog cutoff; its transaction is closed '
                f'and no belief or verdict for it can commit from here on.')
        return super()._on_watchdog_timeout()

    def _close_attempt_without_transfer(self, why: str) -> None:
        """Return the sender FSM to solo tracking using the baseline's own recovery path.

        ``broadcast_handoff_request`` (inherited) moved the FSM to NEGOTIATING, and leaving it there
        means the watchdog reads the silence as a dead candidate. ``abort_to_recovery`` followed by
        ``recovery_to_solo_tracking`` is the baseline's existing exit from a failed attempt: it
        reclaims at epoch e+2, which is also what fences this attempt's beliefs. That epoch bump is
        why a RETAIN is not free — the cost is bookkeeping, not ownership.
        """
        if self.fsm.role is not OwnershipRole.OWNER:
            self.get_logger().error(
                f'[EXP13] refusing the recovery path for a node in role '
                f'{self.fsm.role.name} ({why}): abort_to_recovery() declares the caller OWNER at '
                f'e+2, and a robot that never held the target must not get to say it does.')
            return
        epoch_before = int(self.fsm.epoch_manager.current_epoch)
        reclaim = self.fsm.abort_to_recovery(f'EXP13 no-transfer: {why}')
        self.fsm.recovery_to_solo_tracking()
        self.selected_receiver_id = None
        if self.adapter is not None and self.negotiation is not None:
            txn = self._exp13_episode_txn()
            if txn:
                self.negotiation.close_transaction(txn, f'no-transfer: {why[:80]}')
        self.get_logger().warn(
            f'[EXP13] attempt closed without transfer ({why}); epoch {epoch_before} -> {reclaim}, '
            f'solo tracking resumed.')

    def _exp13_episode_txn(self) -> str:
        return (self.adapter.episode_id if self.adapter is not None else '') or ''

    # ------------------------------------------------------------------ Re-ID signature intake

    def _wire_signature_intake(self) -> None:
        """Subscribe to Exp-7's ReidSignature so the trigger can carry a real reference embedding.

        The baseline initialises ``_latest_reid_signature`` to zeros and never updates it, because
        nothing calls ``set_reid_signature()``. The consumer lives here rather than in
        ``coordination_node.py`` so the default ``cnp`` arm stays byte-identical to the deployed node;
        under ``cbba`` the inherited parent line ``coordination_node.py:476`` then publishes whatever
        this arm last validated.
        """
        topic = self.get_parameter('reid_signature_topic').get_parameter_value().string_value
        self._exp13_sig_max_age_ms = float(
            self.get_parameter('reid_signature_max_age_ms').get_parameter_value().double_value)
        self._exp13_sig_required = bool(
            self.get_parameter('require_valid_reid_signature').get_parameter_value().bool_value)
        self.sub_signature = self.create_subscription(
            ReidSignature, topic, self._on_reid_signature, 5)
        self.get_logger().info(
            f'[EXP13] Re-ID signature intake armed on {topic!r}; a trigger carries a signature only '
            f'while one is fresh (max age '
            f'{self._exp13_sig_max_age_ms:.0f} ms) and '
            f'require_valid_reid_signature={self._exp13_sig_required}.')

    def _on_reid_signature(self, msg: Any) -> None:
        """Admit, hold and install one signature — or drop the one already held."""
        self._exp13_sig_counts['received'] += 1
        sender = str(getattr(msg, 'sender_id', '') or '')
        if sender and sender != self.robot_id:
            # A peer's view of its own target is not this robot's reference embedding. Adopting it is
            # the identity-shaped twin of the heartbeat-poisoning defect recorded as S-9.
            self._exp13_sig_counts['refused_foreign_sender'] += 1
            self.get_logger().warn(
                f'[EXP13] ReidSignature from {sender!r} refused: this node is {self.robot_id!r} and '
                f'will not hand over a target using an embedding it never observed.')
            return

        if not bool(getattr(msg, 'valid', False)):
            self._exp13_sig_state = None
            self.set_reid_signature(np.zeros(512, dtype=np.float32))
            self._exp13_sig_counts['invalidated'] += 1
            self.get_logger().info(
                '[EXP13] signature invalidated by its producer; the next trigger carries no reference '
                'embedding, so the receiver will refuse the handoff rather than guess.')
            return

        vector = np.asarray(getattr(msg, 'vector', []), dtype=np.float32).reshape(-1)
        if vector.size != 512:
            self._exp13_sig_counts['refused_dimension'] += 1
            self.get_logger().error(
                f'[EXP13] ReidSignature refused: expected 512 floats, got {vector.size}. The contract '
                f'is float32[512]; a different producer must not be silently truncated or padded.')
            return
        if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-6:
            self._exp13_sig_counts['refused_invalid'] += 1
            self.get_logger().error(
                f'[EXP13] ReidSignature flagged valid but is not usable (finite='
                f'{bool(np.all(np.isfinite(vector)))}, norm='
                f'{float(np.linalg.norm(vector)):.6g}); refused instead of scored.')
            return

        stamp_ns = int(getattr(msg, 'stamp_utc_ns', 0) or 0)
        received_ns = self.get_clock().now().nanoseconds
        target = str(getattr(msg, 'target_id', '') or '')
        active = str(getattr(self.fsm, 'active_target_id', '') or '')
        if stamp_ns > received_ns + REID_SIGNATURE_CLOCK_SKEW_TOLERANCE_NS:
            # The stamp arrives in the producer's clock domain, so "later than my own now" is not
            # proof of a newer observation: it is what an unsynchronised peer, a driver in a different
            # time base, or use_sim_time on one host and not the other looks like. Ageing is measured
            # locally (see `received_ns` below) precisely so this cannot silently make a signature
            # immortal, and an implausible stamp is refused rather than trusted.
            self._exp13_sig_counts['refused_future_stamp'] += 1
            self.get_logger().warn(
                f'[EXP13] ReidSignature stamped {stamp_ns} is more than '
                f'{REID_SIGNATURE_CLOCK_SKEW_TOLERANCE_NS / 1e6:.0f} ms ahead of this node\'s clock '
                f'({received_ns}); refused. Two hosts with no shared time error model cannot decide '
                f'freshness from a timestamp, so the arm refuses instead of guessing who is ahead.')
            return
        if target and active and target != active:
            self._exp13_sig_counts['refused_target_mismatch'] += 1
            self.get_logger().warn(
                f'[EXP13] ReidSignature describes {target!r} while this node is negotiating '
                f'{active!r}; refused, because attaching one target\'s identity to another '
                f'target\'s handoff is exactly the stale association this contract exists to stop.')
            return
        held = self._exp13_sig_state
        if held is not None and stamp_ns and stamp_ns < held['stamp_ns']:
            # DDS gives no ordering guarantee between two publications of the same topic, so an old
            # observation can arrive last. Accepting it would silently replace a fresh identity with
            # the one the producer has already moved past.
            self._exp13_sig_counts['refused_out_of_order'] += 1
            self.get_logger().warn(
                f'[EXP13] ReidSignature stamped {stamp_ns} arrived after one stamped '
                f'{held["stamp_ns"]}; the older observation is dropped, because a handoff must be '
                f'authorised by what the camera saw most recently.')
            return
        self._exp13_sig_state = {
            'stamp_ns': stamp_ns,
            'received_ns': received_ns,
            'target_id': target,
            'model_id': str(getattr(msg, 'model_id', '') or ''),
            # 0.0 on this field means "the producer measured nothing", not "quality is zero". Kept as
            # None so no later gate can read an unmeasured field as a low score and start trusting or
            # rejecting a signature on it. Nothing in this arm decides anything from quality today,
            # and that is asserted by test rather than assumed.
            'quality': (float(getattr(msg, 'quality', 0.0) or 0.0)
                        or None),
        }
        self.set_reid_signature(vector)
        self._exp13_sig_counts['applied'] += 1

    def _exp13_signature_usable(self) -> Tuple[bool, str]:
        """Is the held signature still fit to authorise this handoff? Never fabricate one."""
        state = self._exp13_sig_state
        if state is None:
            return False, 'none'
        target = str(getattr(self.fsm, 'active_target_id', '') or '')
        if state['target_id'] and target and state['target_id'] != target:
            return False, f"target_mismatch({state['target_id']}!={target})"
        # Two ages are available and both are kept. The producer's stamp is the observation time, so
        # it catches a replayed crop that only now arrives; local arrival time is the only measure
        # that survives a clock-domain difference between hosts. A signature is dropped when either
        # one exceeds the limit, so neither measure can be argued past: a producer whose clock runs
        # ahead used to make a signature never expire, because the stamp age went negative.
        if self._exp13_sig_max_age_ms > 0.0:
            now_ns = self.get_clock().now().nanoseconds
            ages = {}
            if state['stamp_ns']:
                stamp_age = (now_ns - state['stamp_ns']) / 1e6
                if stamp_age >= 0.0:
                    ages['stamp'] = stamp_age
            received_ns = state.get('received_ns')
            if received_ns:
                ages['arrival'] = (now_ns - received_ns) / 1e6
            if ages:
                worst = max(ages, key=lambda k: ages[k])
                if ages[worst] > self._exp13_sig_max_age_ms:
                    detail = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(ages.items()))
                    return False, (f'stale({worst}={ages[worst]:.0f}ms>'
                                   f'{self._exp13_sig_max_age_ms:.0f}ms; ages: {detail})')
        return True, 'fresh'

    def _exp13_drop_signature(self, why: str) -> None:
        self._exp13_sig_state = None
        self.set_reid_signature(np.zeros(512, dtype=np.float32))
        self._exp13_sig_counts['dropped_before_broadcast'] += 1
        self.get_logger().warn(
            f'[EXP13] trigger will carry no reference embedding ({why}); the receiver refuses rather '
            f'than verify against a stale or absent identity.')

    # ------------------------------------------------------------------ baseline seams overridden

    def broadcast_handoff_request(self, request_msg: Optional[Any] = None) -> None:
        """Under ``cbba``: publish the trigger as the baseline does, then negotiate over beliefs.

        The HandoffRequest is *not* an election in this arm — it is the event that starts an episode,
        carries the spatial prior and fixes the transaction id every participant must echo. Inheriting
        the parent's body keeps the message, the FSM transitions and the Re-ID signature population
        identical to the baseline; the only difference is that the response window no longer selects.
        """
        if self.adapter is None:
            return super().broadcast_handoff_request(request_msg)

        # ES-EXP07 <-> ES-EXP13: a trigger is the document Robot B verifies against, so this is the
        # last place a reference embedding can still be checked for belonging to *this* attempt.
        # Only the self-built path (request_msg is None, which is what coordination_node.py:277 uses)
        # is in scope: a caller-supplied message is published verbatim by the parent and this arm
        # does not rewrite another node's payload.
        if request_msg is None:
            usable, reason = self._exp13_signature_usable()
            if not usable:
                if self._exp13_sig_required:
                    self.get_logger().error(
                        f'[EXP13] handoff withheld: require_valid_reid_signature=true and no usable '
                        f'signature ({reason}). This node keeps the target; ownership is not '
                        f'negotiated for an identity nobody can confirm.')
                    return
                if reason != 'none':
                    self._exp13_drop_signature(reason)
        super().broadcast_handoff_request(request_msg)
        episode_id = self.fsm.active_transaction_id or ''
        target_id = self.fsm.active_target_id or ''
        self._exp13_episode_target = target_id
        if not episode_id:
            # No transaction id means no episode identity that a peer could echo, so there is nothing
            # honest to negotiate. Close the attempt through the baseline's own exit rather than mint
            # a private id the peer will never see.
            self.get_logger().error('[EXP13] FSM holds no active transaction id after the trigger was '
                                    'sent; not opening an episode. This node will not negotiate an '
                                    'identity that peers cannot echo.')
            self._stop_round_timer()
            return
        # M0, scoped to Exp-13: the transaction table only knows what a node registers. The baseline
        # never calls open_transaction in production, which is why admission checks there reject
        # every message; this arm opens the episode so the CONFIRM/ACK guards below have a real
        # transaction to validate against.
        self.negotiation.open_transaction(episode_id, target_id, self.peer_id)
        position, tto_safe_s = self._episode_geometry(request_msg)
        try:
            self.adapter.start_episode(
                target_id=target_id,
                ownership_epoch=int(self.fsm.epoch_manager.current_epoch),
                position=position,
                tto_safe_s=tto_safe_s,
                battery_pct=self._live_battery_pct(),
                # The initiator adopts the id it already put on the wire. Re-minting a second,
                # differently-spelled identity here is how two nodes end up negotiating different
                # transactions while both sides' vectors look perfectly valid (design decision D5).
                episode_id=episode_id,
            )
        except ProtocolConfigError as exc:
            self._stop_round_timer()
            # The adapter may still be negotiating the *previous* attempt, whose transaction the FSM
            # has already dropped (``trigger_candidate`` overwrote ``active_transaction_id`` before the
            # trigger was sent). Leaving that episode RUNNING would let it reach AGREED later and hand
            # over a target nobody is negotiating for, so it is closed here and now.
            stale = self.adapter.episode_id
            if stale and stale != episode_id:
                self.adapter.abort(f'superseded by transaction {episode_id}',
                                   is_owner=self._is_owner())
                self.get_logger().warn(
                    f'[EXP13] abandoned episode {stale!r}: the trigger for {episode_id!r} replaced it '
                    f'while it was still running. Peers that keep sending beliefs for {stale!r} will '
                    f'be refused as FOREIGN_EPISODE.')
            self.get_logger().error(f'[EXP13] could not open episode {episode_id!r}: {exc}. '
                                    f'Falling back to solo tracking.')
            # The request for this transaction is already on the wire, so a peer will answer it. Closing
            # the record here is what turns "we refused to negotiate" into a refusal the admission check
            # enforces instead of one that only the log remembers.
            self.negotiation.close_transaction(episode_id, "episode could not open")
            self._close_attempt_without_transfer(f'episode open failed: {exc}')
            return
        self._start_round_timer()
        self.get_logger().info(
            f'[EXP13] episode {episode_id} opened for target {target_id} at epoch '
            f'{self.fsm.epoch_manager.current_epoch}; beliefs now drive the allocation.')

    def _episode_geometry(self, request_msg: Optional[Any]) -> Tuple[list, float]:
        """Target position and safe TTO for the episode, from the same sources the parent used."""
        if request_msg is not None:
            pos = [float(c) for c in (list(getattr(request_msg, 'position', []) or [])[:3])]
            if len(pos) == 3:
                tto_ms = float(getattr(request_msg, 'tto_remaining_ms_at_send', 0.0) or 0.0)
                return pos, tto_ms / 1000.0
        tto = self._last_tto_estimate
        if tto is not None:
            traj = getattr(tto, 'predicted_trajectory', [])
            pos = [float(traj[0].x), float(traj[0].y), float(traj[0].z)] if traj else [0.0, 0.0, 0.0]
            return pos, float(tto.tto_safe_s)
        # No estimate at all: the geometry is unknown, so there is nothing honest to price. Zeros are
        # passed rather than a plausible-looking distance — every agent abstains and the episode ends
        # NO_WINNER instead of inventing an allocation.
        self.get_logger().warn('[EXP13] no TTO estimate available; bidding with zero geometry, so '
                               'this episode is expected to end NO_WINNER rather than invent a bid')
        return [0.0, 0.0, 0.0], 0.0

    def _on_handoff_request(self, msg) -> None:
        """Candidate side: the baseline gate answers, and the same trigger opens our episode."""
        if self.adapter is None:
            return super()._on_handoff_request(msg)

        super()._on_handoff_request(msg)
        if not self.fsm.active_transaction_id:
            return  # the parent's admission dropped it; there is no episode to open
        txn = self.fsm.active_transaction_id
        self.negotiation.open_transaction(txn, msg.header.target_id, msg.header.sender_id)
        if self.adapter.episode_id == txn:
            return  # already negotiating this exact attempt
        position = [float(c) for c in (list(getattr(msg, 'position', []) or []) + [0.0, 0.0, 0.0])[:3]]
        tto_safe_s = float(getattr(msg, 'tto_remaining_ms_at_send', 0.0) or 0.0) / 1000.0
        try:
            # The initiator's transaction id is echoed verbatim, never re-derived locally.
            self.adapter.start_episode(
                target_id=msg.header.target_id,
                ownership_epoch=int(msg.header.ownership_epoch),
                position=position,
                tto_safe_s=tto_safe_s,
                battery_pct=self._live_battery_pct(),
                episode_id=txn,
            )
            self._exp13_episode_target = msg.header.target_id
        except ProtocolConfigError as exc:
            self._stop_round_timer()
            self.get_logger().error(
                f'[EXP13] refusing episode {txn!r}: {exc}; not participating')
            return
        self._start_round_timer()

    def _on_handoff_response(self, msg: HandoffResponse) -> None:
        """Under ``cbba`` a response is a feasibility report, not a vote that selects a receiver."""
        if self.adapter is None:
            return super()._on_handoff_response(msg)

        sender = msg.header.sender_id
        if msg.decision == HandoffResponse.DECISION_REJECT:
            reason = RESPONSE_REASON_NAMES.get(int(msg.reason), f'UNKNOWN_{msg.reason}')
            self.adapter.mark_infeasible(sender, f'hard gate rejected ({reason})')
            self.get_logger().info(
                f'[EXP13] {sender} reported itself infeasible ({reason}); consensus may still award '
                f'it, and arm_commit() will then override the winner.')
        elif msg.decision == HandoffResponse.DECISION_ACCEPT:
            self.get_logger().info(
                f'[EXP13] {sender} ACCEPTed the trigger (utility={float(msg.utility):.4f}); no '
                f'receiver is selected from responses in this arm')
        if self._negotiation_window_open:
            self._collected_responses.append(msg)

    def _on_negotiation_timeout(self) -> None:
        """Baseline closes its window by picking the highest utility. Under ``cbba`` it picks nothing.

        Selection is the consensus result and the round timer owns the outcome. The window is still
        closed, so a late ACCEPT cannot be mistaken for a vote in an arm that ignores votes.
        """
        if self.adapter is None:
            return super()._on_negotiation_timeout()

        if self._neg_timer is not None:
            self._neg_timer.cancel()
            self.destroy_timer(self._neg_timer)
            self._neg_timer = None
        self._negotiation_window_open = False
        self.get_logger().info(
            f'[EXP13] response window closed with {len(self._collected_responses)} answer(s); no '
            f'receiver selected — the belief exchange decides. Episode: '
            f'{self.adapter.episode_id or "(none)"}')

    def _on_local_heartbeat(self, msg) -> None:
        """Adopt only *this* robot's heartbeat, then inherit the baseline's telemetry update.

        Recorded as M3. ``state_heartbeat`` is in ``robot_bringup.launch.py``'s swarm list, so both
        robots' heartbeats are remapped onto ``/edgesync_swarm/state_heartbeat`` and each coordination
        node's ``sub_local_heartbeat`` receives the *peer's* telemetry too. The baseline
        (``coordination_node.py:255-259``) writes whichever sample arrived last into
        ``_my_telemetry.cpu_load_pct`` / ``battery_pct`` / ``nav_status``, and that object is what the
        hard gate G(j) and the utility bid are evaluated on — so under the deployed launch a robot can
        reject a handoff because its *peer* was busy, and bid as if it had the peer's battery. The
        baseline handler is left as it is (Exp 12/14 depend on it); this arm ignores foreign senders
        and, in exchange, will not bid on the parameter-defaulted stand-in telemetry (see
        :meth:`_live_battery_pct`).

        Gated on the arm like every other override here, so the ``cnp`` control arm stays behaviourally
        identical to the deployed baseline node.
        """
        if self.adapter is None:
            return super()._on_local_heartbeat(msg)
        sender = str(msg.header.sender_id)
        if sender != self.robot_id:
            self.get_logger().debug(
                f'[EXP13] ignoring state_heartbeat from {sender}: local telemetry must not be '
                f'overwritten by a peer sample (M3)')
            return
        self._exp13_heartbeat_seen = True
        super()._on_local_heartbeat(msg)

    # ------------------------------------------------------------------ M1 / M2 guards

    def _on_handoff_confirm(self, msg) -> None:
        """M1: ACK only a CONFIRM the sender FSM actually accepted, and only for a known transaction.

        The baseline publishes ``HandoffConfirmAck`` unconditionally
        (``coordination_node.py:566-577``), even when ``on_confirm_received()`` refused the transition
        and returned the *unchanged* epoch — so a CONFIRM in a state that cannot confirm still gets an
        ACK, and the peer believes ownership moved. Here the refusal is honoured: no ACK, and the
        watchdog stays armed so its timeout reclaims at e+2 through the existing path.
        """
        if self.adapter is None:
            return super()._on_handoff_confirm(msg)

        epoch_before = int(self.fsm.epoch_manager.current_epoch)
        admitted, why = self.negotiation.admit_message(
            msg.header.protocol_version, msg.header.receiver_id, msg.header.transaction_id,
            msg.header.ownership_epoch, epoch_before)
        if not admitted:
            self.get_logger().warn(f'[EXP13] CONFIRM dropped by admission: {why}')
            return

        new_epoch = self.fsm.on_confirm_received()
        if int(new_epoch) <= epoch_before:
            self.get_logger().error(
                f'[EXP13] CONFIRM from {msg.header.sender_id} refused by the FSM (state '
                f'{self.fsm.sender_state.name}, epoch still {epoch_before}). Publishing NO ACK — a '
                f'refused confirm must not look accepted to the peer.')
            return

        self.watchdog.disarm()
        ack = HandoffConfirmAck()
        ack.header.protocol_version = self.proto_ver
        ack.header.message_type = HandoffHeader.MSG_TYPE_CONFIRM_ACK
        ack.header.sender_id = self.robot_id
        ack.header.receiver_id = msg.header.sender_id
        ack.header.transaction_id = msg.header.transaction_id
        ack.header.ownership_epoch = int(new_epoch)
        ack.new_epoch = int(new_epoch)
        self.negotiation.mark_confirmed(msg.header.transaction_id, int(new_epoch))
        self.pub_confirm_ack.publish(ack)
        self.get_logger().info(f'[EXP13] Published HANDOFF_CONFIRM_ACK with epoch {new_epoch}.')

    def _on_handoff_confirm_ack(self, msg) -> None:
        """M2: take ownership only on ACK(e+1) for *this* transaction, and nothing else.

        The baseline accepts whatever ``new_epoch`` the peer names (``coordination_node.py:580-596``)
        and never compares it against e+1 or checks the transaction, so a replayed or forged ACK moves
        the receiver's epoch to whatever the message says. Both comparisons are made here before the
        FSM is touched.
        """
        if self.adapter is None:
            return super()._on_handoff_confirm_ack(msg)

        expected = int(self.fsm.epoch_manager.current_epoch) + 1
        txn = self.fsm.active_transaction_id or msg.header.transaction_id
        if int(msg.new_epoch) != expected:
            self.get_logger().error(
                f'[EXP13] CONFIRM_ACK rejected: new_epoch={msg.new_epoch} is not e+1={expected} '
                f'(from {msg.header.sender_id}). Ownership NOT taken.')
            return
        if msg.header.transaction_id != txn:
            self.get_logger().error(
                f'[EXP13] CONFIRM_ACK rejected: transaction {msg.header.transaction_id!r} is not the '
                f'active transaction {txn!r}. Ownership NOT taken.')
            return

        self.fsm.on_confirm_ack_received(int(msg.new_epoch))
        self.get_logger().info(
            f'[EXP13] OWNERSHIP TRANSFERRED at epoch {msg.new_epoch} after both guards passed.')
        for i in range(1, 4):
            claim = OwnershipEvent()
            claim.header.protocol_version = self.proto_ver
            claim.header.message_type = HandoffHeader.MSG_TYPE_OWNERSHIP_EVENT
            claim.header.sender_id = self.robot_id
            claim.header.receiver_id = '*'
            claim.header.transaction_id = msg.header.transaction_id
            claim.header.ownership_epoch = int(msg.new_epoch)
            claim.role = OwnershipEvent.ROLE_OWNER
            claim.burst_index = i
            self.pub_claim.publish(claim)


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = Exp13CoordinationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
