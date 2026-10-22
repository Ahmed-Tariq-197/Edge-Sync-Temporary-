"""Experiment 13 node-level integration tests — what the selected arm actually wires up.

TEST CLASS: UNIT-LEVEL INTEGRATION against stubbed middleware. NOT a ROS 2 integration test.
================================================================================================
This host has no rclpy, no ``edgesync_msgs`` generated code, no DDS and no launch system; the probe
output that establishes that is recorded in ``EXP13_TEST_AND_VALIDATION_REPORT_*.md``. The middleware
scaffolding is therefore replaced by stubs *generated from the real ``.msg`` files on disk* — the same
technique ``edgesync_identity/tests/test_identity_node_transreid_integration.py`` already uses in this
repository, for the same reason.

What that buys, and what it does not:

* BUYS: the node's own wiring is executed. Parameter → selector → adapter → publishers, subscription
  and timer → FSM calls. Every field the node reads or writes must exist in the ``.msg`` definition,
  because the stub classes use ``__slots__`` derived from those files, so a renamed field is an
  ``AttributeError`` here rather than a build error nobody can produce on this host.
* BUYS: two nodes connected through an in-process topic registry, so the directed belief topics are
  exercised as a *pair* — if a publisher name and the peer's subscription name disagree, the belief
  never arrives.
* DOES NOT BUY: DDS discovery, QoS negotiation, real serialisation, message-size limits, the namespace
  or remap a launch file applies, executor behaviour, wall-clock timing, or anything about a robot.
  Delivery is synchronous on publish, which is stronger than a network. A pass here says the Python
  integration is coherent; it says nothing about whether a graph forms.

Nothing in this file is evidence about safety on hardware, and none of it is reported as ROS 2
integration in the deliverables.

Categories (Phase D of the Experiment 13 brief): N1 selector dispatch and the control arm, N2
parameters reaching the runtime, N3 participant rules (D4), N4 topic/QoS topology, N5 episode identity
(D5), N6 transaction admission M0, N7 the ACK leg M1, N8 ownership guards M2, N9 telemetry M3, N10 the
commit seam, N11 the response path, N12 two nodes over the topics, N13 build/launch wiring.
"""
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

from edgesync_coordination.fsm.ownership_fsm import OwnershipRole, ReceiverState, SenderState
from edgesync_coordination.negotiation.cbba_negotiator import RejectionClass
from edgesync_coordination.negotiation.exp13_cbba_adapter import (
    CommitAction,
    CommitDecision,
    ProtocolConfigError,
    belief_from_mapping,
)

PKG_ROOT = Path(__file__).resolve().parents[1]
WS_SRC = PKG_ROOT.parent
MSG_DIR = WS_SRC / "edgesync_msgs" / "msg"
SETUP_PY = PKG_ROOT / "setup.py"
CMAKE = WS_SRC / "edgesync_msgs" / "CMakeLists.txt"
EXP13_LAUNCH = WS_SRC / "edgesync_bringup" / "launch" / "exp13_bringup.launch.py"
BASE_LAUNCH = WS_SRC / "edgesync_bringup" / "launch" / "robot_bringup.launch.py"
BASE_NODE_PY = PKG_ROOT / "edgesync_coordination" / "nodes" / "coordination_node.py"
EXP13_NODE_PY = PKG_ROOT / "edgesync_coordination" / "nodes" / "exp13_coordination_node.py"

PRIMITIVES = {
    "bool": False, "byte": 0, "char": 0,
    "int8": 0, "uint8": 0, "int16": 0, "uint16": 0, "int32": 0, "uint32": 0,
    "int64": 0, "uint64": 0, "float32": 0.0, "float64": 0.0, "string": "",
}


# --------------------------------------------------------------------- stub middleware

def _parse_msg(path):
    """``.msg`` → (fields, constants), comments stripped.

    Built from what is on disk so that the stubs cannot silently accept a field the real generator
    would not have produced.
    """
    fields, constants = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#")[0].strip()
        if not body:
            continue
        if "=" in body:
            head, _, value = body.rpartition("=")
            name = head.split()[-1]
            value = value.strip()
            constants[name] = int(value, 0) if re.fullmatch(r"[+-]?(0[xX][0-9a-fA-F]+|\d+)", value) \
                else value
            continue
        parts = body.split()
        if len(parts) == 2:
            fields.append((parts[0], parts[1]))
    return fields, constants


def _build_message_classes():
    parsed = {p.stem: _parse_msg(p) for p in sorted(MSG_DIR.glob("*.msg"))}
    classes = {}

    class _StubMsg:
        __slots__ = ()
        _fields = ()

        def __init__(self):
            for ftype, fname in self._fields:
                base = ftype.split("/")[-1]
                if "[" in base:
                    setattr(self, fname, [])
                elif base in PRIMITIVES:
                    setattr(self, fname, PRIMITIVES[base])
                else:
                    setattr(self, fname, classes[base]() if base in classes else None)

        def __repr__(self):
            return f"<stub {type(self).__name__}>"

    for name, (fields, constants) in parsed.items():
        attrs = dict(constants)
        attrs.update({"_fields": fields, "__slots__": tuple(f[1] for f in fields),
                      "__repr__": _StubMsg.__repr__})
        classes[name] = types.new_class(name, (_StubMsg,), {},
                                        lambda ns, a=attrs: ns.update(a))
    return classes


class _RecordingLogger:
    """Stand-in for the rclpy logger that keeps every record so a test can assert on the refusal."""

    def __init__(self, name):
        self.name = name
        self.records = []

        for level in ("debug", "info", "warn", "error", "fatal"):
            def emit(msg, _level=level, *a, **k):
                self.records.append((_level, str(msg)))
            setattr(self, level, emit)
        self.warning = self.warn

    def text(self, level=None):
        return " || ".join(m for lv, m in self.records if level is None or lv == level)


class _FakeTime:
    def __init__(self, nanoseconds):
        self.nanoseconds = int(nanoseconds)


class _FakeClock:
    def __init__(self, world):
        self._world = world

    def now(self):
        return _FakeTime(self._world.clock_ns)


class FakePublisher:
    def __init__(self, world, msg_type, topic, qos):
        self.world = world
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.published = []

    def publish(self, msg):
        self.published.append(msg)
        self.world.publish(self.topic, msg)


class FakeTimer:
    def __init__(self, period, cb):
        self.period = period
        self.cb = cb
        self.cancelled = False
        self.destroyed = False

    def cancel(self):
        self.cancelled = True


class FakeWorld:
    """An in-process topic registry standing in for the DDS domain.

    Publish is synchronous and reaches every subscriber of that topic, including a node's own
    subscription to the same name — which is how the baseline's self-addressed heartbeat topic behaves
    inside one process, and what makes the M3 filter observable.
    """

    def __init__(self):
        self.topics = {}
        self.pub_log = []
        self.sub_log = []
        self.timer_log = []
        self.clock_ns = 1_000_000_000

    def subscribe(self, topic, cb):
        self.topics.setdefault(topic, []).append(cb)

    def publish(self, topic, msg):
        for cb in list(self.topics.get(topic, [])):
            cb(msg)

    def subscriber_count(self, topic):
        return len(self.topics.get(topic, []))

    def topics_of(self, needle):
        return sorted({p.topic for p in self.pub_log if needle in p.topic})

    def sub_topics(self, needle):
        return sorted({t for t, _, _, _ in self.sub_log if needle in t})


class _ParameterValue:
    def __init__(self, value):
        self._value = value

    @property
    def string_value(self):
        return str(self._value)

    @property
    def double_value(self):
        return float(self._value)

    @property
    def bool_value(self):
        return bool(self._value)

    @property
    def integer_value(self):
        return int(self._value)

    @property
    def string_array_value(self):
        return list(self._value)


class _Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_parameter_value(self):
        return _ParameterValue(self.value)


class _StubNode:
    """``rclpy.node.Node`` scaffolding only: parameters, endpoints, clock, logger, timers."""

    world = None
    param_overrides = {}

    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._params = {}
        self._logger = _RecordingLogger(name)
        self._timers = []

    def declare_parameter(self, name, default, *args, **kwargs):
        self._params[name] = _Parameter(name, _StubNode.param_overrides.get(name, default))

    def get_parameter(self, name):
        return self._params[name]

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return _FakeClock(self.world)

    def create_publisher(self, msg_type, topic, qos, *args, **kwargs):
        pub = FakePublisher(self.world, msg_type, topic, qos)
        self.world.pub_log.append(pub)
        return pub

    def create_subscription(self, msg_type, topic, cb, qos, *args, **kwargs):
        self.world.subscribe(topic, cb)
        self.world.sub_log.append((topic, cb, msg_type, qos))
        return ("subscription", topic, cb)

    def create_timer(self, period, cb, *args, **kwargs):
        timer = FakeTimer(period, cb)
        self._timers.append(timer)
        self.world.timer_log.append(timer)
        return timer

    def destroy_timer(self, timer):
        if timer in self._timers:
            self._timers.remove(timer)
        timer.destroyed = True

    def destroy_node(self):
        pass


class _QoSProfile:
    def __init__(self, reliability=None, history=None, depth=None, **kwargs):
        self.reliability = reliability
        self.history = history
        self.depth = depth


class _Policy:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _install_stubs():
    """Fake rclpy / tf2_ros / edgesync_msgs modules, plus the sys.modules state to restore."""
    classes = _build_message_classes()

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _StubNode
    rclpy.node = rclpy_node
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = _QoSProfile
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE=_Policy("RELIABLE"), BEST_EFFORT=_Policy("BEST_EFFORT"))
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=_Policy("KEEP_LAST"))
    rclpy.time = types.ModuleType("rclpy.time")
    rclpy.time.Time = lambda: _FakeTime(0)
    rclpy.duration = types.ModuleType("rclpy.duration")
    rclpy.duration.Duration = lambda seconds=0.0: types.SimpleNamespace(seconds=seconds)

    def _no_middleware(*args, **kwargs):
        raise RuntimeError("rclpy stub only — no ROS 2 middleware on this host")

    rclpy.init = rclpy.shutdown = rclpy.spin = _no_middleware
    rclpy.ok = lambda: False

    tf2 = types.ModuleType("tf2_ros")

    class _Buffer:
        def lookup_transform(self, *args, **kwargs):
            raise RuntimeError("no TF in a unit test: the node's documented offline fallback runs")

    class _Listener:
        def __init__(self, buffer_, node, *args, **kwargs):
            self.buffer = buffer_

    tf2.Buffer = _Buffer
    tf2.TransformListener = _Listener

    msgs_pkg = types.ModuleType("edgesync_msgs")
    msgs_pkg.__path__ = []
    msg_mod = types.ModuleType("edgesync_msgs.msg")
    for name, cls in classes.items():
        setattr(msg_mod, name, cls)
    msgs_pkg.msg = msg_mod

    modules = {"rclpy": rclpy, "rclpy.node": rclpy_node, "rclpy.qos": rclpy.qos,
               "rclpy.time": rclpy.time, "rclpy.duration": rclpy.duration, "tf2_ros": tf2,
               "edgesync_msgs": msgs_pkg, "edgesync_msgs.msg": msg_mod}
    saved = {key: sys.modules.get(key) for key in modules}
    sys.modules.update(modules)
    return classes, saved


def _import_arm(saved):
    """Import both node modules under the stubs, then put ``sys.modules`` back exactly as it was.

    Leaving the fakes installed would let this package's own ROS tests
    (``test_response_collector.py`` and friends) run against a fake middleware instead of reporting the
    genuine ``ModuleNotFoundError`` that this host produces.
    """
    from edgesync_coordination.nodes.exp13_coordination_node import (
        BELIEF_TOPIC_SUFFIX,
        Exp13CoordinationNode,
    )
    from edgesync_coordination.nodes.coordination_node import CoordinationNode
    for key, module in saved.items():
        if module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = module
    return Exp13CoordinationNode, CoordinationNode, BELIEF_TOPIC_SUFFIX


MSG, _SAVED = _install_stubs()
Exp13CoordinationNode, CoordinationNode, BELIEF_SUFFIX = _import_arm(_SAVED)
_UNSTALLED = all(key not in sys.modules for key in _SAVED)


# --------------------------------------------------------------------- fixtures and helpers

@pytest.fixture
def world():
    return FakeWorld()


def build_node(world, robot="ugv_1", peer="ugv_2", owner=True, protocol="cbba", **params):
    """Construct the arm against ``world``. Defaults match the deployed two-robot configuration.

    ``protocol=None`` leaves ``negotiation_protocol`` at whatever the node declares, which is how the
    untouched-default test reaches the runtime.
    """
    overrides = {"robot_id": robot, "peer_id": peer, "is_initial_owner": owner}
    if protocol is not None:
        overrides["negotiation_protocol"] = protocol
    overrides.update(params)
    _StubNode.world = world
    _StubNode.param_overrides = overrides
    return Exp13CoordinationNode()


def hb(sender, battery=85.0, cpu=20.0, nav=1, epoch=1):
    msg = MSG["StateHeartbeat"]()
    msg.header.protocol_version = 0x0203
    msg.header.sender_id = sender
    msg.header.receiver_id = "*"
    msg.header.ownership_epoch = epoch
    msg.header.message_type = MSG["HandoffHeader"].MSG_TYPE_STATE_HEARTBEAT
    msg.battery_pct = battery
    msg.cpu_load = cpu
    msg.nav_status = nav
    return msg


def request(txn="txn_exp13_1", target="target_0", sender="ugv_1", receiver="ugv_2", epoch=1,
            position=(5.0, 0.0, 1.5), tto_ms=3000.0, sigma_ms=0.0):
    req = MSG["HandoffRequest"]()
    req.header.protocol_version = 0x0203
    req.header.message_type = MSG["HandoffHeader"].MSG_TYPE_REQUEST
    req.header.sender_id = sender
    req.header.receiver_id = receiver
    req.header.transaction_id = txn
    req.header.target_id = target
    req.header.ownership_epoch = epoch
    req.position = list(position)
    req.tto_remaining_ms_at_send = int(tto_ms)
    req.tto_sigma_ms = float(sigma_ms)
    return req


def response(sender="ugv_2", decision=None, utility=0.5, reason=0, txn="txn_exp13_1", epoch=1):
    res = MSG["HandoffResponse"]()
    res.header.protocol_version = 0x0203
    res.header.message_type = MSG["HandoffHeader"].MSG_TYPE_RESPONSE
    res.header.sender_id = sender
    res.header.receiver_id = "ugv_1"
    res.header.transaction_id = txn
    res.header.ownership_epoch = epoch
    res.decision = MSG["HandoffResponse"].DECISION_ACCEPT if decision is None else decision
    res.reason = reason
    res.utility = utility
    return res


def confirm(sender="ugv_2", txn="txn_exp13_1", epoch=1):
    msg = MSG["HandoffConfirm"]()
    msg.header.protocol_version = 0x0203
    msg.header.message_type = MSG["HandoffHeader"].MSG_TYPE_CONFIRM
    msg.header.sender_id = sender
    msg.header.receiver_id = "ugv_1"
    msg.header.transaction_id = txn
    msg.header.ownership_epoch = epoch
    return msg


def belief(sender="ugv_2", txn="txn_exp13_1", rnd=1, epoch=1, agreement=False, schema=3,
           target="target_0", bid=0.7):
    """A peer-shaped belief, for feeding the node's own subscription."""
    msg = MSG["ConsensusBelief"]()
    msg.header.protocol_version = 0x0203
    msg.header.message_type = 9
    msg.header.sender_id = sender
    msg.header.receiver_id = "*"
    msg.header.transaction_id = txn
    msg.header.target_id = target
    msg.header.ownership_epoch = epoch
    msg.header.sequence = rnd
    msg.schema_version = schema
    msg.belief_id = f"{txn}:{sender}:{rnd}"
    msg.round_number = rnd
    msg.task_ids = ["task_0"]
    msg.target_ids = [target]
    msg.winning_bids = [bid]
    msg.winning_agents = [sender]
    msg.bundle = ["task_0"]
    msg.agreement_established = agreement
    return msg


def ack(sender="ugv_1", txn="txn_m2", new_epoch=2):
    msg = MSG["HandoffConfirmAck"]()
    msg.header.protocol_version = 0x0203
    msg.header.message_type = MSG["HandoffHeader"].MSG_TYPE_CONFIRM_ACK
    msg.header.sender_id = sender
    msg.header.receiver_id = "ugv_2"
    msg.header.transaction_id = txn
    msg.header.ownership_epoch = new_epoch
    msg.new_epoch = new_epoch
    return msg


def open_episode(node, battery=85.0, txn="txn_exp13_1", target="target_0", epoch=1, req=None):
    """Put an owner node into a running episode through its own trigger path."""
    node._on_local_heartbeat(hb(node.robot_id, battery=battery, epoch=epoch))
    node.broadcast_handoff_request(req or request(txn=txn, target=target, sender=node.robot_id,
                                                  receiver=node.peer_id, epoch=epoch))
    return node


def drive_until_closed(node, is_owner=True, rounds=8):
    for _ in range(rounds):
        node._drive_round()
        if node.adapter.arm_commit(is_owner).closes_episode:
            break
    return node.adapter.arm_commit(is_owner)


def drive_pair(a, b, rounds=8):
    """Run two nodes' round drivers in the interleaving two robots on one timer period produce.

    A round only sees the beliefs the peer published in an *earlier* round, so the schedule decides
    whether a mutual view is ever formed: driving one node to its round bound before the other has
    emitted anything is a partition, not a negotiation. The partition is a test of its own below.
    """
    for _ in range(rounds):
        if a.adapter.negotiator is not None and not a.adapter.negotiator.is_terminal():
            a._drive_round()
        if b.adapter.negotiator is not None and not b.adapter.negotiator.is_terminal():
            b._drive_round()
        if (a.adapter.negotiator is None or a.adapter.negotiator.is_terminal()) and \
           (b.adapter.negotiator is None or b.adapter.negotiator.is_terminal()):
            break


def receiver_in_awaiting_ack(node, txn="txn_m2", target="target_0"):
    """Walk the receiver FSM through its own methods to the state that waits for an ACK."""
    assert node.fsm.on_request_received(txn, target) is True
    assert node.fsm.accept_request() is True
    assert node.fsm.start_acquisition() is True
    assert node.fsm.target_locked() is True
    assert node.fsm.send_confirm() is True
    assert node.fsm.receiver_state is ReceiverState.AWAITING_ACK
    return node


def owner_in_verifying(node, txn="txn_exp13_1"):
    """An owner whose FSM is in the state that may legitimately accept a CONFIRM."""
    open_episode(node, txn=txn)
    assert node.fsm.on_peer_accepted() is True
    assert node.fsm.on_peer_verifying() is True
    assert node.fsm.sender_state is SenderState.IDENTITY_VERIFICATION
    return node


# ------------------------------------------------------------------ N0 the harness itself

def test_the_stubs_were_uninstalled_after_the_import():
    """A fake left in ``sys.modules`` would make the ROS tests on this host lie about their result."""
    assert _UNSTALLED
    for key in ("rclpy", "rclpy.node", "edgesync_msgs", "edgesync_msgs.msg", "tf2_ros"):
        assert sys.modules.get(key) is None, f"{key} is still stubbed"


def test_the_stub_messages_came_from_the_definitions_on_disk():
    assert set(MSG) >= {"HandoffHeader", "HandoffRequest", "HandoffResponse", "HandoffConfirm",
                        "HandoffConfirmAck", "OwnershipEvent", "StateHeartbeat", "TTOEstimate",
                        "ConsensusBelief"}
    assert MSG["ConsensusBelief"].__slots__[:3] == ("header", "schema_version", "belief_id")


def test_a_stub_message_refuses_a_field_the_definition_does_not_declare():
    msg = MSG["ConsensusBelief"]()
    with pytest.raises(AttributeError):
        msg.episode_id = "would-have-been-a-second-identity"


# ------------------------------------------------------------------ N1 selector dispatch

def test_the_cbba_selector_builds_the_adapter_and_the_belief_endpoints(world):
    node = build_node(world, protocol="cbba")
    assert node.negotiation_protocol == "cbba"
    assert node.adapter is not None and node.adapter.agent_id == "ugv_1"
    assert node.pub_belief and node.sub_belief is not None
    assert node.timer_round is not None


def test_the_cnp_selector_runs_the_inherited_baseline_with_no_belief_plumbing(world):
    node = build_node(world, protocol="cnp")
    assert node.adapter is None
    assert node.pub_belief == {} and node.sub_belief is None and node.timer_round is None
    assert world.topics_of(BELIEF_SUFFIX) == []
    assert world.sub_topics(BELIEF_SUFFIX) == []


def test_an_unconfigured_node_gets_the_declared_baseline_default(world):
    """The fallback is a *declared* value at the node, so an existing launch keeps the baseline."""
    node = build_node(world, protocol=None)
    assert node.get_parameter("negotiation_protocol").value == "cnp"
    assert node.adapter is None


def test_an_unknown_selector_refuses_to_start_and_never_reaches_the_baseline(world):
    with pytest.raises(ProtocolConfigError):
        build_node(world, protocol="cbba_v2")
    assert world.topics_of(BELIEF_SUFFIX) == []


@pytest.mark.parametrize("token", ["", "   ", "CBBA", "cnp2", "auction_cnp", "centralized_broker"])
def test_every_rejected_selector_value_raises_instead_of_redirecting(world, token):
    with pytest.raises(ProtocolConfigError):
        build_node(world, protocol=token)


def test_the_refusal_message_names_the_bad_value_and_what_is_supported(world):
    with pytest.raises(ProtocolConfigError) as exc:
        build_node(world, protocol="centralized_broker")
    text = str(exc.value)
    assert "centralized_broker" in text and "cbba" in text and "cnp" in text


def test_a_surrounding_space_in_the_configured_value_still_selects_the_requested_arm(world):
    assert build_node(world, protocol=" cbba ").adapter is not None


def test_the_control_arm_selects_from_responses_the_way_the_baseline_does(world):
    """Proof the two arms really differ at the selection step, not only in the logs."""
    cnp = build_node(world, protocol="cnp")
    cbba = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2")
    for node in (cnp, cbba):
        node._negotiation_window_open = False
        node._on_handoff_response(response(sender="ugv_2", utility=0.4))
    assert cnp.selected_receiver_id == "ugv_2"
    assert cbba.selected_receiver_id is None


# ------------------------------------------------------------------ N2 parameters reach runtime

def test_the_parameter_block_is_carried_until_an_episode_opens(world):
    node = build_node(world, protocol="cbba", max_rounds=7, convergence_threshold=3, utility_precision=5)
    assert node.adapter.negotiator is None, "no episode open yet, so there is no negotiator"
    assert node.adapter.max_rounds == 7
    assert node.adapter._negotiator_config["convergence_threshold"] == 3
    assert node.adapter._negotiator_config["utility_precision"] == 5


def test_the_parameters_take_effect_on_the_negotiator_the_moment_an_episode_opens(world):
    node = build_node(world, protocol="cbba", max_rounds=7, convergence_threshold=3)
    open_episode(node)
    assert node.adapter.negotiator.max_rounds == 7
    assert node.adapter.negotiator.convergence_threshold == 3


def test_a_non_positive_deadline_means_bounded_only_by_rounds(world):
    assert build_node(world, protocol="cbba", deadline_s=0.0).adapter._deadline_s() is None


def test_a_positive_deadline_reaches_the_negotiator(world):
    node = build_node(world, protocol="cbba", deadline_s=1.5)
    open_episode(node)
    assert node.adapter.negotiator.deadline_s == pytest.approx(1.5)


def test_require_distributed_agreement_false_is_honoured_by_the_adapter(world):
    assert build_node(world, protocol="cbba", require_distributed_agreement=False) \
        .adapter.require_distributed_agreement is False


def test_the_round_period_parameter_sets_the_driver_period(world):
    assert build_node(world, protocol="cbba", belief_round_period_ms=120).timer_round.period \
        == pytest.approx(0.12)


def test_the_baseline_utility_engine_is_the_one_the_adapter_bids_with(world):
    """The ablation may change the allocator and nothing else."""
    node = build_node(world, protocol="cbba")
    assert node.adapter._utility_engine is node.utility


def test_a_missing_required_key_raises_rather_than_defaulting_mid_experiment(world):
    """The node declares every key, so a hand-rolled config that drops one must still be caught."""
    from edgesync_coordination.negotiation.exp13_cbba_adapter import Exp13CbbaAdapter
    config = build_node(world, protocol="cbba")._param_config()
    config.pop("max_rounds")
    with pytest.raises(ProtocolConfigError):
        Exp13CbbaAdapter(agent_id="ugv_1", participant_ids=["ugv_1", "ugv_2"], config=config)


# ------------------------------------------------------------------ N3 participant rules (D4)

def test_a_participant_set_that_omits_this_node_is_repaired_and_reported(world):
    node = build_node(world, protocol="cbba", robot="ugv_3", peer="ugv_1",
                      participant_ids=["ugv_1", "ugv_2"], swarm_participant_count=3)
    assert node.adapter.participants == ["ugv_3", "ugv_1", "ugv_2"]
    assert "did not include this node" in node._logger.text("warn")


def test_repairing_the_participant_set_without_raising_the_count_refuses_loudly(world):
    with pytest.raises(ProtocolConfigError):
        build_node(world, protocol="cbba", robot="ugv_3", peer="ugv_1",
                   participant_ids=["ugv_1", "ugv_2"], swarm_participant_count=2)


def test_a_declared_participant_is_never_dropped_to_make_the_episode_fit(world):
    node = build_node(world, protocol="cbba", participant_ids=["ugv_1", "ugv_2", "ugv_3"],
                      swarm_participant_count=3, peer="ugv_2")
    assert node.adapter.participants == ["ugv_1", "ugv_2", "ugv_3"]
    assert sorted(node.pub_belief) == ["ugv_2", "ugv_3"]


def test_a_single_participant_set_is_refused_rather_than_running_a_vacuous_consensus(world):
    with pytest.raises(ProtocolConfigError):
        build_node(world, protocol="cbba", participant_ids=["ugv_1"], swarm_participant_count=1)


def test_a_multi_target_episode_is_refused_rather_than_allocating_one_of_n(world):
    node = build_node(world, protocol="cbba", target_count=3)
    open_episode(node)
    assert node.adapter.episode_id == ""
    assert "target_count" in node._logger.text("error")


# ------------------------------------------------------------------ N4 topic and QoS topology

def test_belief_publishers_are_directed_at_each_peer_and_the_subscription_is_self_addressed(world):
    build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2")
    assert world.topics_of(BELIEF_SUFFIX) == ["/ugv_2/edgesync/consensus_belief"]
    assert world.sub_topics(BELIEF_SUFFIX) == ["/ugv_1/edgesync/consensus_belief"]
    assert "/ugv_1/edgesync/consensus_belief" not in world.topics_of(BELIEF_SUFFIX), (
        "a self-addressed belief publisher is dead in the Tier-1 topology, where each robot owns its "
        "own absolute topic name")


def test_beliefs_are_published_reliable_with_a_history_of_ten(world):
    node = build_node(world, protocol="cbba")
    for pub in node.pub_belief.values():
        assert pub.qos.reliability.name == "RELIABLE"
        assert pub.qos.history.name == "KEEP_LAST" and pub.qos.depth == 10
    qos = [q for t, _, _, q in world.sub_log if t == "/ugv_1/edgesync/consensus_belief"][0]
    assert qos.reliability.name == "RELIABLE"


def test_the_baseline_endpoints_of_the_node_are_left_where_they_are(world):
    node = build_node(world, protocol="cbba")
    assert world.sub_topics("state_heartbeat") == ["/ugv_1/edgesync/state_heartbeat"]
    assert node.pub_req.topic == "/ugv_2/edgesync/handoff_request"
    assert node.pub_res.topic == "/ugv_2/edgesync/handoff_response"
    assert node.pub_confirm_ack.topic == "/ugv_2/edgesync/handoff_confirm_ack"
    assert node.pub_claim.topic == "/ugv_1/edgesync/ownership_event"


def test_a_published_belief_survives_the_message_definition_field_by_field(world):
    """Node → stub message → peer mapping → ``ConsensusMessage``. No field name is guessed."""
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node.adapter.run_round()
    published = next(iter(node.pub_belief.values())).published[0]
    received = belief_from_mapping(node._belief_to_mapping(published))
    assert received.sender_agent_id == "ugv_1"
    assert received.episode_id == node.adapter.episode_id == "txn_exp13_1"
    assert received.round_number == 1
    assert received.task_ids == ["task_0"]
    assert received.target_ids == {"task_0": "target_0"}
    assert received.schema_version == 3


def test_the_wire_sequence_is_the_round_number_with_no_second_counter(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node.adapter.run_round()
    node.adapter.run_round()
    msgs = next(iter(node.pub_belief.values())).published
    assert [m.header.sequence for m in msgs] == [m.round_number for m in msgs] == [1, 2]


# ------------------------------------------------------------------ N5 episode identity (D5)

def test_the_initiator_adopts_the_transaction_id_it_put_on_the_wire(world):
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_42")
    assert node.pub_req.published[-1].header.transaction_id == "txn_42"
    assert node.adapter.episode_id == "txn_42" == node.fsm.active_transaction_id
    assert node.adapter.metrics()["episode_id_kind"] == "opaque_transaction_id"


def test_a_canonical_transaction_id_is_validated_and_adopted(world):
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="target_0@e1#0")
    assert node.adapter.episode_id == "target_0@e1#0"
    assert node.adapter.metrics()["episode_id_kind"] == "canonical"


def test_a_canonical_id_naming_another_target_is_refused_instead_of_negotiating_along(world):
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="other_target@e1#0")
    assert node.adapter.episode_id == ""
    assert "could not open episode" in node._logger.text("error")


def test_a_trigger_that_reached_no_transaction_opens_no_episode(world):
    """A standby has nothing to initiate, and an id it invented privately is unechoable."""
    node = build_node(world, protocol="cbba", owner=False)
    node._on_local_heartbeat(hb("ugv_1"))
    node.broadcast_handoff_request(request())
    assert node.adapter.episode_id == ""
    assert "no active transaction id" in node._logger.text("error")


def test_the_candidate_echoes_the_initiators_id_and_epoch(world):
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False)
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_from_ugv_1"))
    assert cand.adapter.episode_id == owner.adapter.episode_id == "txn_from_ugv_1"
    assert cand.adapter.negotiator.ownership_epoch == 1
    assert cand.adapter.metrics()["episode_id_kind"] == "opaque_transaction_id"


def test_a_second_trigger_never_silently_replaces_a_running_episode(world):
    """One node, one episode: a superseded trigger is refused and closed, not quietly swapped.

    The FSM has already been moved to the new transaction by the inherited trigger, so the arm cannot
    simply ignore the request. What it must do is make the old episode terminal and hand the new
    transaction back to the caller as a failed attempt — after that, the same trigger succeeds.
    """
    owner = build_node(world, protocol="cbba", require_distributed_agreement=False)
    open_episode(owner, txn="txn_a")
    # A reclaim by any other owner of the attempt (watchdog, manual recovery) returns the sender FSM
    # to TRACKING while the episode is still RUNNING; the next trigger then arrives for txn_b.
    owner.fsm.sender_state = SenderState.TRACKING
    owner.fsm.role = OwnershipRole.OWNER
    open_episode(owner, txn="txn_b")
    assert "could not open episode 'txn_b'" in owner._logger.text("error")
    assert owner.adapter.episode_id == "txn_a", "the running episode was swapped out from under itself"
    assert owner.adapter.negotiator.is_terminal(), "the superseded episode must be terminal, not RUNNING"
    assert owner.fsm.sender_state is SenderState.TRACKING
    assert owner.fsm.epoch_manager.current_epoch == 3
    assert owner.selected_receiver_id is None
    # Now that nothing is running, the next attempt opens normally and re-arms its driver.
    open_episode(owner, txn="txn_c")
    assert owner.adapter.episode_id == "txn_c"
    assert owner.timer_round is not None


# ------------------------------------------------------------------ N6 M0 transaction admission

def test_the_trigger_registers_the_transaction_so_the_guards_have_something_to_validate(world):
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_m0")
    record = node.negotiation._table["txn_m0"]
    assert record.target_id == "target_0" and record.peer_id == "ugv_2"
    assert record.state.name == "OPEN"


def test_a_valid_confirm_is_not_blocked_by_the_new_admission_check(world):
    """Brief §6 in both directions: open the transaction, then let the legitimate CONFIRM through."""
    node = owner_in_verifying(build_node(world, protocol="cbba"), txn="txn_valid")
    node._on_handoff_confirm(confirm(txn="txn_valid"))
    assert len(node.pub_confirm_ack.published) == 1, (
        "M0 must not become a check that rejects every legitimate handoff")


def test_a_confirm_for_an_unknown_transaction_is_dropped_before_the_fsm(world):
    node = owner_in_verifying(build_node(world, protocol="cbba"), txn="txn_valid")
    node._on_handoff_confirm(confirm(txn="never_opened"))
    assert node.pub_confirm_ack.published == []
    assert node.fsm.sender_state is SenderState.IDENTITY_VERIFICATION
    assert "Unknown transaction" in node._logger.text("warn")


def test_the_candidate_registers_the_initiators_transaction_too(world):
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False)
    cand._on_local_heartbeat(hb("ugv_2"))
    owner.broadcast_handoff_request(request(txn="txn_pair"))
    assert cand.negotiation._table["txn_pair"].state.name == "OPEN"
    assert cand.negotiation._table["txn_pair"].peer_id == "ugv_1"


# ------------------------------------------------------------------ N7 M1 the ACK follows the verdict

def test_an_accepted_confirm_is_acked_at_e_plus_one_and_recorded(world):
    node = owner_in_verifying(build_node(world, protocol="cbba"), txn="txn_m1")
    node._on_handoff_confirm(confirm(txn="txn_m1"))
    sent = node.pub_confirm_ack.published[-1]
    assert sent.new_epoch == 2 == node.fsm.epoch_manager.current_epoch
    assert sent.header.transaction_id == "txn_m1" and sent.header.sender_id == "ugv_1"
    assert node.negotiation.get_stored_ack_epoch("txn_m1") == 2


def test_a_confirm_the_fsm_refuses_gets_no_ack(world):
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_m1")
    assert node.fsm.sender_state is SenderState.NEGOTIATING
    node._on_handoff_confirm(confirm(txn="txn_m1"))
    assert node.pub_confirm_ack.published == []
    assert node.fsm.epoch_manager.current_epoch == 1
    assert "Publishing NO ACK" in node._logger.text("error")


def test_a_refused_confirm_does_not_disarm_the_watchdog(world):
    """The refusal must leave the existing timeout path in charge, not silently park the attempt."""
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_m1")
    node.watchdog.arm()
    node._on_handoff_confirm(confirm(txn="txn_m1"))
    assert node.watchdog.is_armed is True


def test_the_control_arm_keeps_the_baselines_unconditional_ack(world):
    """Regression for §5: the deployed path is unchanged, ACK included."""
    node = build_node(world, protocol="cnp")
    node._on_handoff_confirm(confirm(txn="not_opened_either"))
    assert len(node.pub_confirm_ack.published) == 1, (
        "the baseline ACK is what Experiments 12/14 measured; only the cbba arm conditions it")


# ------------------------------------------------------------------ N8 M2 ownership only on e+1

def test_an_ack_at_e_plus_one_for_the_active_transaction_transfers_ownership(world):
    node = receiver_in_awaiting_ack(build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1",
                                               owner=False))
    node._on_handoff_confirm_ack(ack(txn="txn_m2", new_epoch=2))
    assert node.fsm.role is OwnershipRole.OWNER
    assert node.fsm.epoch_manager.current_epoch == 2


def test_ownership_is_announced_by_exactly_three_claims_at_the_new_epoch(world):
    node = receiver_in_awaiting_ack(build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1",
                                               owner=False))
    node._on_handoff_confirm_ack(ack(txn="txn_m2", new_epoch=2))
    claims = node.pub_claim.published
    assert [c.burst_index for c in claims] == [1, 2, 3]
    assert all(c.role == MSG["OwnershipEvent"].ROLE_OWNER and c.header.ownership_epoch == 2
               and c.header.transaction_id == "txn_m2" for c in claims)


@pytest.mark.parametrize("new_epoch,txn", [(3, "txn_m2"), (1, "txn_m2"), (2, "other_txn")])
def test_an_ack_that_is_not_e_plus_one_or_not_this_transaction_takes_nothing(world, new_epoch, txn):
    node = receiver_in_awaiting_ack(build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1",
                                               owner=False))
    node._on_handoff_confirm_ack(ack(txn=txn, new_epoch=new_epoch))
    assert node.fsm.role is OwnershipRole.CANDIDATE
    assert node.fsm.epoch_manager.current_epoch == 1
    assert node.pub_claim.published == []
    assert "Ownership NOT taken" in node._logger.text("error")


def test_a_replayed_ack_cannot_announce_ownership_twice(world):
    node = receiver_in_awaiting_ack(build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1",
                                               owner=False))
    msg = ack(txn="txn_m2", new_epoch=2)
    node._on_handoff_confirm_ack(msg)
    before = len(node.pub_claim.published)
    node._on_handoff_confirm_ack(msg)
    assert len(node.pub_claim.published) == before, "a replay must not re-announce ownership"
    assert node.fsm.epoch_manager.current_epoch == 2


def test_the_control_arm_still_takes_whatever_epoch_the_peer_names(world):
    """The baseline weakness is documented, not silently fixed in the shared path."""
    node = receiver_in_awaiting_ack(build_node(world, protocol="cnp", robot="ugv_2", peer="ugv_1",
                                               owner=False))
    node._on_handoff_confirm_ack(ack(txn="whatever", new_epoch=9))
    assert node.fsm.epoch_manager.current_epoch == 9


# ------------------------------------------------------------------ N9 M3 local telemetry

def test_a_peer_heartbeat_does_not_overwrite_local_telemetry(world):
    node = build_node(world, protocol="cbba")
    before = (node._my_telemetry.battery_pct, node._my_telemetry.cpu_load_pct)
    node._on_local_heartbeat(hb("ugv_2", battery=12.0, cpu=97.0))
    assert (node._my_telemetry.battery_pct, node._my_telemetry.cpu_load_pct) == before
    assert node._exp13_heartbeat_seen is False


def test_the_peers_heartbeat_still_reaches_the_shared_swarm_topic(world):
    """The topology that makes M3 a finding: one topic, two senders."""
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True)
    topic = "/ugv_1/edgesync/state_heartbeat"
    world.publish(topic, hb("ugv_2", battery=5.0))
    assert owner._my_telemetry.battery_pct == pytest.approx(85.0)


def test_own_heartbeat_is_adopted_and_enables_bidding(world):
    node = build_node(world, protocol="cbba")
    assert node._live_battery_pct() is None
    node._on_local_heartbeat(hb("ugv_1", battery=73.0))
    assert node._exp13_heartbeat_seen is True
    assert node._live_battery_pct() == pytest.approx(73.0)


def test_the_control_arm_keeps_the_baselines_last_writer_wins_telemetry(world):
    node = build_node(world, protocol="cnp")
    node._on_local_heartbeat(hb("ugv_2", battery=12.0))
    assert node._my_telemetry.battery_pct == pytest.approx(12.0), (
        "M3 is a finding about the shared launch topology; the shared handler is not edited here")


def test_bidding_uses_the_live_battery_rather_than_the_parameter_default(world):
    node = build_node(world, protocol="cbba", battery_pct=85.0)
    open_episode(node, battery=41.0)
    node.adapter.run_round()
    assert node.adapter._battery_pct == pytest.approx(41.0)


def test_an_episode_opened_without_telemetry_abstains_instead_of_bidding_on_the_default(world):
    node = build_node(world, protocol="cbba", battery_pct=85.0)
    node.broadcast_handoff_request(request())
    assert node.adapter._battery_pct is None
    payload = node.adapter.run_round()
    assert payload is None or payload["winning_bids"] == [0.0]


# ------------------------------------------------------------------ N10 the commit seam

def test_commit_is_the_only_action_that_names_a_receiver(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._apply_commit(CommitDecision(action=CommitAction.COMMIT, receiver_id="ugv_2",
                                      reason="test", episode_id="txn_exp13_1", is_owner=True))
    assert node.selected_receiver_id == "ugv_2"
    assert node.fsm.sender_state is SenderState.COMMITTED


def test_a_commit_arms_the_watchdog_that_the_baseline_arms_on_selection(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._apply_commit(CommitDecision(action=CommitAction.COMMIT, receiver_id="ugv_2",
                                      reason="test", episode_id="txn_exp13_1", is_owner=True))
    assert node.watchdog.is_armed is True


@pytest.mark.parametrize("action", [CommitAction.WAIT, CommitAction.STANDBY])
def test_actions_that_ask_nothing_of_the_owner_change_nothing(world, action):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    epoch = node.fsm.epoch_manager.current_epoch
    node._apply_commit(CommitDecision(action=action, reason="test", episode_id="txn_exp13_1"))
    assert node.fsm.epoch_manager.current_epoch == epoch
    assert node.selected_receiver_id is None
    assert node.fsm.sender_state is SenderState.NEGOTIATING


# ------------------------------------------------------------------ N10b M4 verdict / transaction match

def test_a_commit_aimed_at_another_transaction_does_not_hand_the_target_over(world):
    """M4: consensus over episode A cannot transfer the target the FSM is negotiating as episode B."""
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_real")
    claims_before = len(node.pub_claim.published)
    node._apply_commit(CommitDecision(action=CommitAction.COMMIT, receiver_id="ugv_2",
                                      reason="a stale episode reached AGREED",
                                      episode_id="txn_stale", is_owner=True))
    assert node.selected_receiver_id is None
    assert node.fsm.sender_state is not SenderState.COMMITTED
    assert node.watchdog.is_armed is False
    assert len(node.pub_claim.published) == claims_before, (
        "the initial-ownership claim is published at construction; nothing new may be announced here")
    assert "not the FSM's active transaction" in node._logger.text("error")
    # Refusing the verdict also ends the attempt, so the node does not sit in NEGOTIATING forever.
    assert node.fsm.sender_state is SenderState.TRACKING
    assert node.fsm.epoch_manager.current_epoch == 3


def test_an_acquire_aimed_at_another_transaction_starts_no_acquisition(world):
    node = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False)
    node._on_handoff_request(request(txn="txn_real"))
    assert node.fsm.active_transaction_id == "txn_real"
    receiver_state = node.fsm.receiver_state
    node._apply_commit(CommitDecision(action=CommitAction.ACQUIRE, receiver_id="ugv_2",
                                      reason="a stale episode reached AGREED",
                                      episode_id="txn_stale", is_owner=False))
    assert node.fsm.role is OwnershipRole.CANDIDATE
    assert node.fsm.receiver_state is receiver_state, "no acquisition leg was started"
    assert node.pub_claim.published == []
    assert node.fsm.epoch_manager.current_epoch == 1
    assert "not the FSM's active transaction" in node._logger.text("error")


def test_an_acquire_for_the_episode_the_fsm_is_in_is_accepted_without_a_claim(world):
    """ACQUIRE only *permits* the baseline's own acquisition legs; it never claims ownership itself."""
    node = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False)
    node._on_handoff_request(request(txn="txn_real"))
    receiver_state = node.fsm.receiver_state
    node._apply_commit(CommitDecision(action=CommitAction.ACQUIRE, receiver_id="ugv_2",
                                      reason="consensus awarded ugv_2",
                                      episode_id="txn_real", is_owner=False))
    assert node.fsm.role is OwnershipRole.CANDIDATE
    assert node.fsm.receiver_state is receiver_state
    assert node.pub_claim.published == []
    assert node.pub_confirm_ack.published == []
    assert node.fsm.epoch_manager.current_epoch == 1
    assert "consensus awarded ugv_2" in node._logger.text("info")


def test_a_tick_after_the_episode_closed_does_not_re_apply_the_verdict(world):
    """One failed attempt advances the epoch once, however many ticks arrive after it closed."""
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._apply_commit(CommitDecision(action=CommitAction.ABORT, reason="test", is_owner=True))
    epoch = node.fsm.epoch_manager.current_epoch
    node._drive_round()
    node._drive_round()
    assert node.fsm.epoch_manager.current_epoch == epoch, (
        "re-applying ABORT would run abort_to_recovery() again and bump the epoch twice")


@pytest.mark.parametrize("action", [CommitAction.ABORT, CommitAction.RETAIN, CommitAction.UNSAFE])
def test_a_non_owner_never_runs_the_recovery_path_even_when_the_episode_closes(world, action):
    """``abort_to_recovery()`` sets role = OWNER, so a standby must not be able to call it."""
    node = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False)
    node._on_local_heartbeat(hb("ugv_2"))
    node._apply_commit(CommitDecision(action=action, reason="test", is_owner=False))
    assert node.fsm.role is OwnershipRole.STANDBY
    assert node.fsm.epoch_manager.current_epoch == 1
    assert "refusing the recovery path" in node._logger.text("error")


@pytest.mark.parametrize("action", [CommitAction.ABORT, CommitAction.RETAIN])
def test_an_owner_closes_the_attempt_at_e_plus_2_and_resumes_solo_tracking(world, action):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._apply_commit(CommitDecision(action=action, reason="test", is_owner=True))
    assert node.fsm.epoch_manager.current_epoch == 3
    assert node.fsm.role is OwnershipRole.OWNER
    assert node.fsm.sender_state is SenderState.TRACKING
    assert node.selected_receiver_id is None
    assert node.negotiation._table["txn_exp13_1"].state.name == "CLOSED"


def test_a_unsafe_dual_claim_transfers_nothing_and_is_logged_as_a_violation(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._apply_commit(CommitDecision(action=CommitAction.UNSAFE,
                                      reason="two holders claimed task_0", is_owner=True))
    assert node.selected_receiver_id is None
    assert node.pub_confirm_ack.published == []
    assert "dual claim" in node._logger.text("error")
    assert node.fsm.epoch_manager.current_epoch == 3


def test_a_closing_decision_stops_the_round_driver(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    timer = node.timer_round
    node._apply_commit(CommitDecision(action=CommitAction.ABORT, reason="test", is_owner=True))
    assert node.timer_round is None and timer.cancelled and timer.destroyed


def test_the_driver_is_rearmed_for_the_next_attempt(world):
    """A closed episode destroys its timer; the next trigger must create a fresh one."""
    node = build_node(world, protocol="cbba", require_distributed_agreement=False)
    open_episode(node)
    node._apply_commit(CommitDecision(action=CommitAction.ABORT, reason="test", is_owner=True))
    assert node.timer_round is None
    open_episode(node, txn="txn_second")
    assert node.timer_round is not None and node.adapter.episode_id == "txn_second"


def test_a_watchdog_cutoff_ends_the_episode_it_is_abandoning(world):
    """The reclaim path is the baseline's; this arm only has to stop negotiating behind it."""
    node = build_node(world, protocol="cbba")
    open_episode(node, txn="txn_wd")
    timer = node.timer_round
    node.watchdog.arm()
    node._on_watchdog_timeout()
    assert node.adapter.negotiator.is_terminal(), \
        "a reclaimed attempt must not leave an episode that can still reach AGREED"
    assert node.timer_round is None and timer.cancelled and timer.destroyed
    assert node.negotiation._table["txn_wd"].state.name == "CLOSED"
    assert node.fsm.sender_state is SenderState.TRACKING
    assert node.fsm.epoch_manager.current_epoch == 3, "the baseline owns the e+2 reclaim, once"
    assert "aborted by the watchdog cutoff" in node._logger.text("warn")
    # A belief that arrives after the cutoff is still counted on the wire but never reaches the dead
    # episode: the post-terminal path records reports, it does not re-open the merge.
    delivered = node.adapter.negotiator.stats.beliefs_delivered
    world.publish(f"/{node.robot_id}/edgesync/{BELIEF_SUFFIX}", belief(sender="ugv_2", txn="txn_wd"))
    assert node.adapter.negotiator.stats.beliefs_delivered == delivered
    assert "not applied: POST_TERMINAL" in node._logger.text("debug")


def test_the_control_arm_keeps_the_watchdog_exactly_as_the_baseline_runs_it(world):
    node = build_node(world, protocol="cnp")
    node.fsm.trigger_candidate("target_0", "txn_cnp")
    node.fsm.send_request()
    node._on_watchdog_timeout()
    assert node.fsm.sender_state is SenderState.TRACKING
    assert node.fsm.epoch_manager.current_epoch == 3


def test_driving_with_no_episode_open_does_nothing_at_all(world):
    node = build_node(world, protocol="cbba")
    epoch = node.fsm.epoch_manager.current_epoch
    node._drive_round()
    node._drive_round()
    assert node.fsm.epoch_manager.current_epoch == epoch, (
        "an idle tick must not close a decision that was never earned")
    assert node.timer_round is not None


def _negotiate_without_acting(a, b, rounds=8):
    """Run both negotiators through the node's subscriptions, but never let a node act on a verdict.

    ``_drive_round`` applies a closing decision the moment it sees one, so a test that wants to observe
    the window between *consensus says COMMIT* and *the node commits* has to drive the adapters instead
    of the timers. Beliefs still cross the real topic names, because the publishers deliver through the
    fake world into each node's own ``_on_belief_msg``.
    """
    for _ in range(rounds):
        a.adapter.run_round()
        b.adapter.run_round()
        if a.adapter.negotiator.is_terminal() and b.adapter.negotiator.is_terminal():
            break


def test_a_veto_that_lands_after_the_computed_commit_still_stops_the_transfer(world):
    """The race the response leg can lose: consensus arms COMMIT, then ugv_2 reports it is infeasible.

    ``arm_commit`` caches its verdict, so without the downgrade in ``mark_infeasible`` the node would
    honour a COMMIT that a later, safer fact had already vetoed.
    """
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                       require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False,
                      require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    owner._on_local_heartbeat(hb("ugv_1", battery=60.0))
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_late"))
    _negotiate_without_acting(owner, cand)
    armed = owner.adapter.arm_commit(True)
    assert armed.action is CommitAction.COMMIT and armed.receiver_id == "ugv_2", str(armed)
    assert owner.fsm.sender_state is SenderState.NEGOTIATING, "the node has not acted yet"
    world.publish("/ugv_1/edgesync/handoff_response",
                  response(sender="ugv_2", txn="txn_late",
                           decision=MSG["HandoffResponse"].DECISION_REJECT,
                           reason=MSG["HandoffResponse"].REASON_LOW_BATTERY))
    downgraded = owner.adapter.arm_commit(True)
    assert downgraded.action is CommitAction.ABORT, str(downgraded)
    assert "late infeasibility report" in downgraded.reason
    owner._apply_commit(downgraded)
    assert owner.selected_receiver_id is None
    assert owner.fsm.sender_state is SenderState.TRACKING
    assert owner.watchdog.is_armed is False


def test_the_agreement_report_leg_is_what_arms_the_commit_at_the_node(world):
    """Design decision D3, executed through the node's own subscription.

    A peer belief that awards itself the target is enough for this agent's *vector* to converge, but it
    is not an agreement: the commit stays held in the grace window until a belief arrives carrying
    ``agreement_established``. Two beliefs, one difference, one different action.
    """
    node = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                      require_distributed_agreement=True, commit_grace_ms=100000,
                      max_rounds=6, convergence_threshold=1)
    node._on_local_heartbeat(hb("ugv_1", battery=30.0))
    open_episode(node, txn="txn_grace")
    world.publish(f"/ugv_1/edgesync/{BELIEF_SUFFIX}",
                  belief(sender="ugv_2", txn="txn_grace", rnd=1, bid=0.99))
    for _ in range(3):
        node.adapter.run_round()
    assert node.adapter.negotiator.get_assignment()["task_0"] == "ugv_2"
    assert node.adapter.negotiator.status.name == "AGREED"
    held = node.adapter.arm_commit(True)
    assert held.action is CommitAction.WAIT, str(held)
    assert "awaiting peer agreement reports" in held.reason
    assert node.selected_receiver_id is None
    # Same winner, now declared as an agreement.
    world.publish(f"/ugv_1/edgesync/{BELIEF_SUFFIX}",
                  belief(sender="ugv_2", txn="txn_grace", rnd=2, bid=0.99, agreement=True))
    armed = node.adapter.arm_commit(True)
    assert armed.action is CommitAction.COMMIT, str(armed)
    assert armed.receiver_id == "ugv_2"


# ------------------------------------------------------------------ N11 responses and the window

def test_a_reject_vetoes_that_agent_for_the_commit(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._negotiation_window_open = True
    node._on_handoff_response(response(sender="ugv_2",
                                       decision=MSG["HandoffResponse"].DECISION_REJECT,
                                       reason=MSG["HandoffResponse"].REASON_LOW_BATTERY))
    assert node.adapter._infeasible["ugv_2"].startswith("hard gate")
    assert "LOW_BATTERY" in node._logger.text("info")


def test_an_accept_is_reported_but_selects_no_receiver_in_this_arm(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._negotiation_window_open = True
    node._on_handoff_response(response(sender="ugv_2", utility=0.9))
    assert node.selected_receiver_id is None
    assert len(node._collected_responses) == 1
    assert "no receiver is selected from responses" in node._logger.text("info")


def test_the_response_window_closes_without_choosing(world):
    node = build_node(world, protocol="cbba")
    open_episode(node)
    node._negotiation_window_open = True
    node._on_handoff_response(response(sender="ugv_2", utility=0.9))
    node._on_negotiation_timeout()
    assert node._negotiation_window_open is False and node._neg_timer is None
    assert node.selected_receiver_id is None
    assert node.fsm.sender_state is SenderState.NEGOTIATING, (
        "closing the window must not abort the attempt that the beliefs are still resolving")


def test_the_control_arm_still_picks_the_highest_utility_answer(world):
    node = build_node(world, protocol="cnp")
    node.fsm.trigger_candidate("target_0", "txn_x")
    node.fsm.send_request()
    node._negotiation_window_open = True
    node._collected_responses = [response(sender="ugv_2", utility=0.2),
                                 response(sender="ugv_3", utility=0.8)]
    node._on_negotiation_timeout()
    assert node.selected_receiver_id == "ugv_3"


# ------------------------------------------------------------------ N12 two nodes over the topics

def test_two_nodes_exchange_beliefs_over_their_real_topic_names(world):
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                       require_distributed_agreement=False)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False,
                      require_distributed_agreement=False)
    owner._on_local_heartbeat(hb("ugv_1", battery=60.0))
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_pair"))
    assert owner.adapter.episode_id == cand.adapter.episode_id == "txn_pair"
    owner.adapter.run_round()
    cand.adapter.run_round()
    assert owner.adapter.counters.beliefs_received >= 1
    assert cand.adapter.counters.beliefs_received >= 1
    assert owner.adapter.negotiator.stats.beliefs_published == 1
    assert cand.adapter.negotiator.stats.beliefs_published == 1


def test_the_peer_with_the_live_higher_battery_wins_and_the_owner_hands_to_it(world):
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                       require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False,
                      require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    owner._on_local_heartbeat(hb("ugv_1", battery=60.0))
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_win"))
    drive_pair(owner, cand)
    decision = owner.adapter.arm_commit(True)
    assert decision.action is CommitAction.COMMIT, f"expected a hand-off, got {decision}"
    assert decision.receiver_id == "ugv_2"
    assert owner.selected_receiver_id == "ugv_2"
    assert cand.adapter.arm_commit(False).action is CommitAction.ACQUIRE


def test_the_awarded_candidate_sees_the_same_winner_from_its_own_view(world):
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                       require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False,
                      require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    owner._on_local_heartbeat(hb("ugv_1", battery=60.0))
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_view"))
    drive_pair(owner, cand)
    assert owner.adapter.negotiator.get_assignment()["task_0"] == \
        cand.adapter.negotiator.get_assignment()["task_0"] == "ugv_2"


def test_a_node_that_bounds_out_before_the_peer_has_emitted_anything_hands_over_nothing(world):
    """The schedule, not only the configuration, decides whether two views can meet.

    Driven alone to its round bound, the owner never receives a belief, so the episode ends
    LOCAL_STABLE/NO_WINNER and the arm releases nothing. That is the honest reading of "the peer was
    not heard from" — and the reason a locally computed winner is not a completed transfer (D3).
    """
    owner = build_node(world, protocol="cbba", robot="ugv_1", peer="ugv_2", owner=True,
                       require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    cand = build_node(world, protocol="cbba", robot="ugv_2", peer="ugv_1", owner=False,
                      require_distributed_agreement=False, max_rounds=6, convergence_threshold=1)
    owner._on_local_heartbeat(hb("ugv_1", battery=60.0))
    cand._on_local_heartbeat(hb("ugv_2", battery=95.0))
    owner.broadcast_handoff_request(request(txn="txn_starve"))
    decision = drive_until_closed(owner)
    assert decision.action is not CommitAction.COMMIT, str(decision)
    assert owner.selected_receiver_id is None
    assert owner.fsm.sender_state is SenderState.TRACKING
    assert owner.adapter.negotiator.stats.beliefs_delivered == 0


def test_a_lone_node_never_releases_its_target_to_itself(world):
    owner = build_node(world, protocol="cbba", require_distributed_agreement=False)
    open_episode(owner)
    decision = drive_until_closed(owner)
    assert decision.action in (CommitAction.RETAIN, CommitAction.ABORT), str(decision)
    assert owner.pub_confirm_ack.published == []
    assert owner.fsm.sender_state is SenderState.TRACKING


def test_a_one_sided_agreement_does_not_release_the_target(world):
    owner = build_node(world, protocol="cbba", require_distributed_agreement=True,
                       commit_grace_ms=0, max_rounds=6)
    open_episode(owner)
    decision = drive_until_closed(owner)
    assert decision.action is not CommitAction.COMMIT, str(decision)
    assert owner.selected_receiver_id is None
    assert owner.fsm.epoch_manager.current_epoch > 1, "closed without transfer"


def test_beliefs_for_a_different_episode_are_not_merged_into_the_running_one(world):
    owner = build_node(world, protocol="cbba")
    open_episode(owner, txn="txn_here")
    world.publish("/ugv_1/edgesync/consensus_belief", belief(sender="ugv_2", txn="txn_elsewhere"))
    owner.adapter.run_round()
    stats = owner.adapter.negotiator.stats
    assert stats.count(RejectionClass.FOREIGN_EPISODE) >= 1, (
        f"a foreign-episode belief was silently merged: {stats.rejected}")
    assert owner.adapter.negotiator.get_assignment()["task_0"] == "ugv_1", (
        "the intruder's claim must not move the local winner")


# ------------------------------------------------------------------ N13 build and launch wiring

def test_the_arm_is_installed_as_its_own_executable_next_to_the_baseline_one():
    text = SETUP_PY.read_text(encoding="utf-8")
    assert "exp13_coordination_node = edgesync_coordination.nodes.exp13_coordination_node:main" in text
    assert "coordination_node = edgesync_coordination.nodes.coordination_node:main" in text, (
        "the deployed entry point must survive the experiment")


def test_cmake_registers_the_new_message_without_changing_an_existing_one():
    text = CMAKE.read_text(encoding="utf-8")
    assert "msg/ConsensusBelief.msg" in text
    for existing in ("HandoffRequest", "HandoffResponse", "HandoffConfirm", "HandoffConfirmAck",
                     "OwnershipEvent", "StateHeartbeat", "HandoffHeader"):
        assert f"msg/{existing}.msg" in text


def test_the_launch_selects_the_exp13_executable_and_passes_the_selector_through():
    text = EXP13_LAUNCH.read_text(encoding="utf-8")
    assert "exp13_coordination_node" in text
    # DeclareLaunchArgument and the Node parameter are the two halves of one path: an argument nobody
    # reads is as dead as a parameter nobody declares.
    assert re.search(r"DeclareLaunchArgument\(\s*'negotiation_protocol'", text), \
        "the experiment launch declares no selector argument"
    assert re.search(r"LaunchConfiguration\('negotiation_protocol'\)", text), \
        "the argument is declared but never read"
    assert re.search(r"'negotiation_protocol':\s*protocol", text), \
        "the selector never reaches the node's parameters"
    assert re.search(r"executable='exp13_coordination_node'", text), \
        "the launch must run the Exp-13 node, not the baseline executable"
    assert re.search(r"'negotiation_protocol',\s*default_value='cbba'", text), \
        "the experiment launch must default to cbba"
    assert "exp13_cbba.yaml" in text, "the experiment configuration must actually be injected"
    assert "consensus_belief" in text, "the belief topic needs the same swarm remap as the handoff set"


def test_the_launch_declares_both_arms_of_the_comparison():
    text = EXP13_LAUNCH.read_text(encoding="utf-8")
    assert "cnp" in text and "cbba" in text


def test_the_baseline_bringup_launch_is_untouched_by_this_experiment():
    text = BASE_LAUNCH.read_text(encoding="utf-8")
    assert "cbba" not in text.lower() and "exp13" not in text.lower()
    assert "coordination_node" in text


def test_the_deployed_coordination_node_source_has_no_experiment_branch():
    text = BASE_NODE_PY.read_text(encoding="utf-8")
    assert "exp13" not in text.lower(), "the baseline must not know about the experiment"
    assert "cbba" not in text.lower()


def test_the_subclass_keeps_the_constructors_signature():
    """``main()`` and the launch construct the node positionally; a new required argument would break."""
    names = lambda fn: [p.name for p in inspect.signature(fn).parameters.values()]
    assert names(Exp13CoordinationNode.__init__) == names(CoordinationNode.__init__)


def test_the_arm_does_not_redeclare_any_parameter_the_baseline_owns():
    baseline = re.findall(r"declare_parameter\('(\w+)'", BASE_NODE_PY.read_text(encoding="utf-8"))
    own = re.findall(r"declare_parameter\('(\w+)'", EXP13_NODE_PY.read_text(encoding="utf-8"))
    assert set(own).isdisjoint(baseline), f"duplicated declaration: {sorted(set(own) & set(baseline))}"
