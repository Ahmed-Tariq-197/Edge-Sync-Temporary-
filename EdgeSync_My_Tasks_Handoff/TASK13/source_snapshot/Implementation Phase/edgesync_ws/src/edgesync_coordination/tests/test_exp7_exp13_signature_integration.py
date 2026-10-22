"""Experiment 7 <-> Experiment 13 integration tests: the Re-ID signature leg of a handoff.

TEST CLASS: UNIT-LEVEL INTEGRATION against stubbed middleware. NOT a ROS 2 integration test.
================================================================================================
Same method and same limits as ``test_exp13_node_integration.py``: this host has no rclpy, no
generated ``edgesync_msgs``, no DDS and no launch system, so the middleware scaffolding is stubbed and
the message classes are generated from the ``.msg`` files on disk. Two nodes are connected through one
in-process topic registry. What that buys is that the *contract and the wiring* are executed — a field
name that drifted, a topic that does not match, or a guard that never runs all fail here. What it does
not buy is DDS discovery, QoS negotiation, real serialisation, an executor, a camera driver or a
robot. Delivery is synchronous on publish, which is stronger than any network.

WHAT IS UNDER TEST
==================
The defect these tests exist for: ``HandoffRequest.reid_feature_vector`` left the sender as
``np.zeros(512)`` forever, because ``CoordinationNode.set_reid_signature()`` has no caller in the
baseline and ``IdentityNode`` published nothing. A receiver therefore could only ever answer
``INVALID_EMBEDDING: reference`` and cancel. The contract that closes the gap is
``edgesync_msgs/msg/ReidSignature.msg``; the producer is Exp-7's ``identity_node`` and the consumer is
Exp-13's arm, which installs the signature through the inherited setter so the baseline's own publish
line carries it.

Cases: I1 message contract on disk; I2 Exp-7 production is opt-in; I3 a crop becomes a smoothed
signature; I4 an unusable crop invalidates instead of publishing zeros; I5 the arm installs a fresh
signature and puts it on the trigger; I6 foreign sender refused; I7 wrong dimension and non-finite
refused; I8 valid-flagged-but-zero refused; I9 invalidation drops what is held; I10 a stale signature
is dropped before the trigger; I11 the require flag withholds the trigger entirely; I12 a signature
about a different target is refused; I13 an out-of-order older signature cannot replace a newer one;
I14 a duplicate is idempotent; I15 the control arm subscribes to nothing; I16 end-to-end crop ->
trigger, checked with Exp-7's own guard; I17 launch wiring; I18 a stamp further ahead than the clock
tolerance is refused; I19 a stamp inside the tolerance is still admitted; I20 a producer whose clock
runs ahead cannot make a signature immortal, because local arrival age also bounds the hold; I21
``quality`` never decides whether a handoff is authorised; I22 an unmeasured ``quality`` is held as
absent rather than as zero; I23 the ``.msg`` text states the producer clock domain and what an absent
``quality`` means.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_exp13_node_integration as H  # noqa: E402  reuses the Tier-N harness, deliberately

WS_SRC = H.WS_SRC
MSG_DIR = WS_SRC / "edgesync_msgs" / "msg"
SRV_DIR = WS_SRC / "edgesync_msgs" / "srv"
CMAKE = WS_SRC / "edgesync_msgs" / "CMakeLists.txt"
EXP13_LAUNCH = WS_SRC / "edgesync_bringup" / "launch" / "exp13_bringup.launch.py"

# This file deliberately spans two packages, and on this host neither is installed: `colcon build`
# would put both on the path via their entry points, so the source directories are added by hand.
# It changes nothing about what is under test — only how the modules are found.
for _pkg in ("edgesync_identity", "edgesync_coordination"):
    _path = str(WS_SRC / _pkg)
    if _path not in sys.path:
        sys.path.append(_path)

SIGNATURE_TOPIC = "/edgesync/identity/reid_signature"
SIG_FIELDS = [("string", "sender_id"), ("string", "target_id"), ("bool", "valid"),
              ("float32[512]", "vector"), ("float32", "quality"), ("uint64", "stamp_utc_ns"),
              ("string", "model_id")]
UNIT = [1.0 / math.sqrt(512.0)] * 512

# The Exp-7 node calls create_service; the coordination arm never does, so the shared Tier-N stub does
# not have it. Added here rather than in the harness file, which 101 other tests depend on.
if not hasattr(H._StubNode, "create_service"):
    def _create_service(self, srv_type, name, callback, *args, **kwargs):
        self.world.services.append((name, srv_type, callback))
        return ("service", name, callback)

    H._StubNode.create_service = _create_service
    _orig_stub_init = H._StubNode.__init__

    def _init_with_services(self, name, *args, **kwargs):
        _orig_stub_init(self, name, *args, **kwargs)
        self.world.services = getattr(self.world, "services", [])

    H._StubNode.__init__ = _init_with_services


class FakeExtractor:
    """A stand-in extractor with three behaviours: a real-looking vector, the zeros sentinel, failure.

    Deliberate: these tests are about what the nodes do with a vector, not about inference accuracy,
    which is measured offline in Exp-7. Using a fake here is the only honest option on a host where
    the checkpoint cannot be loaded by a test that is meant to take milliseconds.
    """

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = []

    def extract_embedding(self, image_crop, cam_label=None):
        self.calls.append((getattr(image_crop, "shape", None), cam_label))
        if self.mode == "zeros":
            return np.zeros(512, dtype=np.float32)
        if self.mode == "short":
            return np.ones(384, dtype=np.float32)
        if self.mode == "raises":
            raise RuntimeError("simulated inference failure")
        return np.asarray(UNIT, dtype=np.float32)


class _Header:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class _Image:
    """Duck-typed sensor_msgs/Image: 2x2 RGB, the geometry image_to_rgb() accepts.

    The default header is deliberately the same instant the harness clock reports. The producer
    stamps the crop header into ``ReidSignature.stamp_utc_ns`` and the consumer refuses anything
    meaningfully ahead of its own clock, so a crop from the future would make the end-to-end test
    exercise the skew guard instead of the wiring.
    """

    def __init__(self, stamp=None):
        self.height = 2
        self.width = 2
        self.encoding = "rgb8"
        self.step = 6
        self.data = bytes(range(12))
        self.header = stamp or _Header(1, 0)


def _install_sensor_msgs(saved):
    pkg = types.ModuleType("sensor_msgs")
    pkg.__path__ = []
    mod = types.ModuleType("sensor_msgs.msg")
    mod.Image = _Image
    pkg.msg = mod
    for key, module in (("sensor_msgs", pkg), ("sensor_msgs.msg", mod)):
        saved.setdefault(key, sys.modules.get(key))
        sys.modules[key] = module


def _install_srv(saved):
    pkg = sys.modules.get("edgesync_msgs")
    srv = types.ModuleType("edgesync_msgs.srv")
    srv.IdentityVerify = types.SimpleNamespace(Request=object, Response=object)
    pkg.srv = srv
    saved.setdefault("edgesync_msgs.srv", sys.modules.get("edgesync_msgs.srv"))
    sys.modules["edgesync_msgs.srv"] = srv


def _import_nodes(saved):
    """Import both node modules under the stubs, then restore sys.modules exactly."""
    from edgesync_identity.nodes.identity_node import IdentityNode
    from edgesync_coordination.nodes.exp13_coordination_node import Exp13CoordinationNode
    for key in sorted(saved, reverse=True):
        module = saved[key]
        if module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = module
    return IdentityNode, Exp13CoordinationNode


_pre_saved = {key: sys.modules.get(key) for key in ("rclpy", "rclpy.node", "edgesync_msgs")}
_classes_bootstrap, saved_bootstrap = H._install_stubs()
_install_srv(saved_bootstrap)
_install_sensor_msgs(saved_bootstrap)
IdentityNode, Exp13CoordinationNode = _import_nodes(saved_bootstrap)
assert all(sys.modules.get(key) is None for key in
           ("rclpy", "rclpy.node", "edgesync_msgs", "edgesync_msgs.srv", "sensor_msgs")), \
    "a fake left in sys.modules would make the rclpy-absence tests on this host lie"


# The classes are regenerated per install, so tests must receive the ones that are installed *right
# now* rather than the bootstrap set. Held in a module-level cell the fixtures can both write and read.
_INSTALLED = {}


@pytest.fixture(scope="module", autouse=True)
def middleware():
    """Install the stub middleware for the duration of this module only, then remove it."""
    classes, saved = H._install_stubs()
    _install_srv(saved)
    _install_sensor_msgs(saved)
    _INSTALLED["classes"] = classes
    yield classes
    _INSTALLED.clear()
    for key in sorted(saved, reverse=True):
        module = saved[key]
        if module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = module


@pytest.fixture
def classes(middleware):
    """The message classes generated from the on-disk .msg files for this module's stub install."""
    return _INSTALLED["classes"]


@pytest.fixture
def world():
    return H.FakeWorld()


def build_identity(world, extractor=None, **params):
    """Construct Exp-7's node under the stubs. No `classes` argument: the message type it needs, it
    imports itself, which is exactly the import path these tests want exercised."""
    overrides = {"robot_id": "ugv_1", "service_name": "/identity/verify"}
    overrides.update(params)
    H._StubNode.world = world
    H._StubNode.param_overrides = overrides
    return IdentityNode(extractor=extractor if extractor is not None else FakeExtractor())


def build_arm(world, robot="ugv_1", peer="ugv_2", **params):
    return H.build_node(world, robot=robot, peer=peer, protocol="cbba", **params)


def signature(classes, sender="ugv_1", target="", valid=True, vector=None, quality=0.0,
              stamp_ns=0, model="transreid"):
    msg = classes["ReidSignature"]()
    msg.sender_id = sender
    msg.target_id = target
    msg.valid = bool(valid)
    msg.vector = list(UNIT) if vector is None else [float(x) for x in vector]
    msg.quality = float(quality)
    msg.stamp_utc_ns = int(stamp_ns)
    msg.model_id = model
    return msg


def deliver(world, classes, **kwargs):
    world.publish(SIGNATURE_TOPIC, signature(classes, **kwargs))


def trigger(world, node):
    """Move the owner into TRACKING and run the production trigger path (no pre-built message)."""
    node._on_local_heartbeat(H.hb(node.robot_id, battery=85.0, epoch=1))
    node.broadcast_handoff_request()
    return node.pub_req.published[-1]


# --------------------------------------------------------------------------- I1 message contract

def test_the_signature_definition_on_disk_is_the_contract_both_sides_code_against():
    fields, constants = H._parse_msg(MSG_DIR / "ReidSignature.msg")
    assert fields == SIG_FIELDS, (
        "ReidSignature changed shape: the producer and the consumer are written against these seven "
        "fields, so a rename here must be matched in identity_node.py, the arm and this file")
    assert constants == {}, "constants in a data message would not be visible to either side"


def test_the_new_type_is_registered_without_disturbing_any_existing_one():
    text = CMAKE.read_text(encoding="utf-8")
    assert "msg/ReidSignature.msg" in text
    registered = [line.strip().strip('"()') for line in text.splitlines()
                  if line.strip().startswith('"msg/')]
    # 12 types were already deployed; this file must add exactly one and change no other line.
    assert len(registered) == len(set(registered)) == 13, registered
    for existing in ("HandoffRequest", "HandoffHeader", "ConsensusBelief", "StateHeartbeat"):
        assert f"msg/{existing}.msg" in text


def test_the_carrier_the_receiver_already_uses_still_declares_the_512_float_fields():
    """IdentityVerify.srv is frozen; this asserts it stayed compatible with the new producer."""
    body = (SRV_DIR / "IdentityVerify.srv").read_text(encoding="utf-8")
    request_part = body.split("---")[0]
    assert "float32[512] reference_feature_vector" in request_part
    assert "float32[512] candidate_feature_vector" in request_part


# --------------------------------------------------------------------------- I2-I4 Exp-7 side

def test_nothing_is_published_until_the_producer_is_asked_for(world, classes):
    node = build_identity(world)
    assert node.pub_signature is None
    assert [p.topic for p in world.pub_log if "reid_signature" in p.topic] == []
    assert "publish_reid_signature=false" in node._logger.text("info")


def test_a_target_crop_becomes_a_smoothed_signature_with_the_model_that_made_it(world, classes):
    node = build_identity(world, publish_reid_signature=True,
                         reid_target_crop_topic="/camera/target_crop",
                         reid_signature_topic=SIGNATURE_TOPIC,
                         reid_signature_target_id="target_0")
    world.publish("/camera/target_crop", _Image())

    published = [p for p in world.pub_log if p.topic == SIGNATURE_TOPIC][0].published
    assert len(published) == 1
    msg = published[0]
    assert msg.valid is True
    assert np.allclose(np.asarray(msg.vector, dtype=np.float32), np.asarray(UNIT), atol=1e-6)
    assert msg.sender_id == "ugv_1" and msg.target_id == "target_0"
    assert msg.stamp_utc_ns == 1_000_000_000, (
        "the stamp is the observation time, so a consumer can age the signature")
    assert msg.model_id == node.reid_model_id
    assert node.signature_publish_count == 1


def test_an_unusable_embedding_invalidates_instead_of_publishing_a_zero_vector(world, classes):
    node = build_identity(world, extractor=FakeExtractor("zeros"), publish_reid_signature=True,
                         reid_target_crop_topic="/camera/target_crop",
                         reid_signature_topic=SIGNATURE_TOPIC)
    world.publish("/camera/target_crop", _Image())

    msg = [p for p in world.pub_log if p.topic == SIGNATURE_TOPIC][0].published[0]
    assert msg.valid is False, (
        "zeros must arrive as an explicit invalidation; a consumer cannot tell a sentinel from a "
        "real embedding by looking at the numbers alone")
    assert list(msg.vector) == [0.0] * 512


def test_an_extractor_that_fails_produces_an_invalidation_and_not_a_silently_absent_signature(
        world, classes):
    node = build_identity(world, extractor=FakeExtractor("raises"), publish_reid_signature=True,
                         reid_target_crop_topic="/camera/target_crop",
                         reid_signature_topic=SIGNATURE_TOPIC)
    world.publish("/camera/target_crop", _Image())
    assert node.signature_refusal_count == 1
    assert [p for p in world.pub_log if p.topic == SIGNATURE_TOPIC][0].published[0].valid is False


def test_a_wrong_dimension_embedding_is_reported_not_resampled(world, classes):
    node = build_identity(world, extractor=FakeExtractor("short"), publish_reid_signature=True,
                         reid_target_crop_topic="/camera/target_crop",
                         reid_signature_topic=SIGNATURE_TOPIC)
    world.publish("/camera/target_crop", _Image())
    assert node.signature_refusal_count == 1
    assert "unusable embedding" in node._logger.text("error")


# --------------------------------------------------------------------------- I5-I8 arm intake

def test_a_fresh_signature_is_installed_and_reaches_the_trigger(world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns)
    assert np.allclose(node._latest_reid_signature, np.asarray(UNIT), atol=1e-7)
    assert node._exp13_sig_counts["applied"] == 1

    req = trigger(world, node)
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32),
                       np.asarray(UNIT), atol=1e-7), (
        "this is the line the whole contract exists for: the trigger carries the identity the "
        "receiver will verify against")


@pytest.mark.parametrize("kwargs,counter", [
    ({"sender": "ugv_2"}, "refused_foreign_sender"),
    ({"vector": [1.0] * 384}, "refused_dimension"),
    ({"vector": [float("nan")] * 512}, "refused_invalid"),
    ({"vector": [0.0] * 512}, "refused_invalid"),
])
def test_a_signature_that_cannot_be_trusted_never_reaches_the_trigger(world, classes, kwargs,
                                                                      counter):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns, **kwargs)
    assert node._exp13_sig_counts[counter] == 1
    assert np.allclose(node._latest_reid_signature, np.zeros(512))
    assert np.allclose(np.asarray(trigger(world, node).reid_feature_vector, dtype=np.float32),
                       np.zeros(512))


def test_a_peers_signature_is_not_adopted_as_this_robots_reference_embedding(world, classes):
    """The topic name is per-robot, but a mis-remapped launch must not be the only thing preventing it."""
    node = build_arm(world, robot="ugv_1")
    deliver(world, classes, sender="ugv_2", stamp_ns=world.clock_ns)
    assert "will not hand over a target using an embedding it never observed" in \
        node._logger.text("warn")
    assert node._exp13_sig_state is None


# --------------------------------------------------------------------------- I9-I12 lifecycle

def test_an_invalidation_from_the_producer_drops_what_is_held(world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns)
    assert node._exp13_sig_counts["applied"] == 1
    deliver(world, classes, valid=False, vector=[0.0] * 512, stamp_ns=world.clock_ns + 1)
    assert node._exp13_sig_counts["invalidated"] == 1
    assert node._exp13_sig_state is None
    assert np.allclose(node._latest_reid_signature, np.zeros(512))


def test_a_stale_signature_is_dropped_before_the_trigger_rather_than_negotiated_with(
        world, classes):
    node = build_arm(world, reid_signature_max_age_ms=500)
    deliver(world, classes, stamp_ns=world.clock_ns - 2_000_000_000)
    assert node._exp13_sig_counts["applied"] == 1, (
        "arrival is accepted; it is the moment of sending the trigger that must be fresh")
    req = trigger(world, node)
    assert node._exp13_sig_counts["dropped_before_broadcast"] == 1
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32), np.zeros(512))
    assert "stale" in node._logger.text("warn")


def test_a_signature_about_another_active_target_is_refused_on_arrival(world, classes):
    node = build_arm(world)
    node.fsm.trigger_candidate("target_0", "txn_exp13_1")
    deliver(world, classes, target="target_7", stamp_ns=world.clock_ns)
    assert node._exp13_sig_counts["refused_target_mismatch"] == 1
    assert node._exp13_sig_state is None


def test_requiring_a_valid_signature_withholds_the_trigger_instead_of_starting_a_blind_handoff(
        world, classes):
    node = build_arm(world, require_valid_reid_signature=True)
    node._on_local_heartbeat(H.hb(node.robot_id, battery=85.0, epoch=1))
    node.broadcast_handoff_request()
    assert node.pub_req.published == [], (
        "with the flag set, no request may leave: a handoff negotiated for an identity nobody can "
        "confirm is the failure this contract is meant to make impossible")
    assert "handoff withheld" in node._logger.text("error")
    assert node.adapter.episode_id == ""


def test_an_older_signature_that_arrives_last_does_not_replace_a_newer_one(world, classes):
    """DDS does not order two publications of one topic, so arrival order is not observation order.

    Fixture corrected in this pass. It used to stamp the newer observation a full second *ahead of
    this node's own clock*, which the arm now refuses as clock skew: a producer's stamp is only
    trusted up to the declared tolerance past "now". Time is advanced here instead, which is what a
    producer sharing the node's clock domain actually does.
    """
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns, model="old")
    world.clock_ns += 1_000_000_000
    newer = world.clock_ns
    deliver(world, classes, stamp_ns=newer, model="new")
    assert node._exp13_sig_state["stamp_ns"] == newer
    assert node._exp13_sig_state["model_id"] == "new"

    deliver(world, classes, stamp_ns=newer - 1_000_000, model="replayed_old")
    assert node._exp13_sig_counts["refused_out_of_order"] == 1, (
        "the brief's section 6 item 8 asks that reordered inputs cannot create a stale identity "
        "association; without this guard the older observation wins purely by arriving last")
    assert node._exp13_sig_state["stamp_ns"] == newer
    assert node._exp13_sig_state["model_id"] == "new"
    req = trigger(world, node)
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32), np.asarray(UNIT),
                       atol=1e-7)


def test_a_duplicate_signature_is_idempotent_and_not_counted_twice_as_state(world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns)
    deliver(world, classes, stamp_ns=world.clock_ns)
    assert node._exp13_sig_counts["received"] == 2
    assert node._exp13_sig_counts["applied"] == 2, (
        "re-applying the same vector is harmless; the state is one observation, not a queue")
    assert node._exp13_sig_counts["refused_out_of_order"] == 0
    assert node._exp13_sig_state["stamp_ns"] == world.clock_ns


# --------------------------------------------------------------------------- I13-I15 wiring

def test_the_control_arm_subscribes_to_nothing_new(world, classes):
    node = H.build_node(world, protocol="cnp")
    assert world.sub_topics("reid_signature") == []
    assert node.sub_signature is None
    assert np.allclose(node._latest_reid_signature, np.zeros(512)), (
        "the baseline arm must remain byte-for-byte the behaviour of coordination_node.py, zeros "
        "included; the fix is not smuggled into the control arm")


def test_the_end_to_end_leg_produces_a_trigger_the_receivers_own_guard_accepts(world, classes):
    """Exp-7 publishes, Exp-13 consumes, and Exp-7's guard is used to adjudicate the result."""
    from edgesync_identity.core.reid_input_adapter import is_invalid_embedding

    identity = build_identity(world, publish_reid_signature=True,
                             reid_target_crop_topic="/camera/target_crop",
                             reid_signature_topic=SIGNATURE_TOPIC)
    node = build_arm(world)
    assert world.sub_topics("reid_signature") == [SIGNATURE_TOPIC]

    world.publish("/camera/target_crop", _Image())
    assert identity.signature_publish_count == 1
    assert node._exp13_sig_counts["applied"] == 1

    req = trigger(world, node)
    vector = np.asarray(req.reid_feature_vector, dtype=np.float32)
    assert vector.shape == (512,)
    assert is_invalid_embedding(vector) is None, (
        "the reference embedding that reaches the wire must pass the same guard the receiver applies "
        "to it — otherwise the fix moved the failure instead of removing it")
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_the_experiment_launch_can_start_the_producer_on_the_same_topic(world):
    text = EXP13_LAUNCH.read_text(encoding="utf-8")
    assert "with_identity" in text and "IfCondition(with_identity)" in text
    assert "identity_node" in text
    assert "'publish_reid_signature': True" in text
    assert "exp7_transreid.yaml" in text, (
        "without the Exp-7 overlay the producer would run the mock and the trigger would carry a "
        "mock embedding, which is worse than carrying none")
    assert "'reid_signature_topic': signature_topic" in text, (
        "producer and consumer must name the same topic; a per-robot absolute name is what keeps a "
        "two-robot run on one host from crossing the streams")
    assert "reid_target_crop_topic': ''" in text, (
        "the launch must not pretend a crop source exists: nothing in this repository publishes one")


# ------------------------------------------------------------- I18-I23 clock and quality contract
# Phase 5.3/5.4. `stamp_utc_ns` is the producer's clock domain and `quality` has no measurement
# behind it. Both were previously load-bearing-looking: ageing compared a foreign stamp against this
# node's clock, and the consumer stored quality as though it were a score.

def test_a_stamp_further_ahead_than_the_clock_tolerance_is_refused_not_trusted(world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns + 200_000_000)
    assert node._exp13_sig_counts["refused_future_stamp"] == 1
    assert node._exp13_sig_state is None, (
        "a signature stamped 200 ms into this node's future is a clock disagreement, not a newer "
        "observation; accepting it would make the held signature expire backwards")
    assert "ahead of this node" in node._logger.text("warn")
    req = trigger(world, node)
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32), np.zeros(512))


def test_a_stamp_inside_the_tolerance_is_admitted_so_a_skewed_clock_does_not_blind_the_leg(
        world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns + 50_000_000)
    assert node._exp13_sig_counts["refused_future_stamp"] == 0
    assert node._exp13_sig_counts["applied"] == 1


def test_a_producer_running_ahead_cannot_make_a_signature_immortal(world, classes):
    """The hole the arrival clock closes: a negative stamp age used to read as permanently fresh.

    The numbers are chosen so this test fails if the arrival term is removed, which the mutation check
    found it did not at a wider margin. The skew is 90 ms — inside the 100 ms tolerance, so the stamp
    is admitted — and the clock then moves 550 ms. Arrival age is 550 ms, over the 500 ms limit; the
    producer's stamp age is 550 - 90 = 460 ms, under it. Only the local measure can drop this one.
    """
    node = build_arm(world, reid_signature_max_age_ms=500)
    deliver(world, classes, stamp_ns=world.clock_ns + 90_000_000)
    assert node._exp13_sig_state is not None

    world.clock_ns += 550_000_000
    req = trigger(world, node)
    assert node._exp13_sig_counts["dropped_before_broadcast"] == 1, (
        "local arrival age must bound the hold even when the producer's stamp still looks fresh")
    assert "stale" in node._logger.text("warn")
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32), np.zeros(512))


@pytest.mark.parametrize("quality", [0.0, 0.95])
def test_quality_never_decides_whether_a_signature_authorises_a_handoff(world, classes, quality):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns, quality=quality)
    assert node._exp13_sig_counts["applied"] == 1
    req = trigger(world, node)
    assert np.allclose(np.asarray(req.reid_feature_vector, dtype=np.float32),
                       np.asarray(UNIT), atol=1e-7), (
        "an unmeasured field must not become a gate: 0.0 is 'no measurement', and rejecting on it "
        "would silently disable the leg whenever the producer has no score to report")


def test_an_unmeasured_quality_is_held_as_absent_not_as_zero(world, classes):
    node = build_arm(world)
    deliver(world, classes, stamp_ns=world.clock_ns, quality=0.0)
    assert node._exp13_sig_state["quality"] is None, (
        "stored as None so a later consumer cannot mistake the absence of a measurement for a "
        "confidence of zero and act on it")
    deliver(world, classes, stamp_ns=world.clock_ns + 1_000_000, quality=0.4)
    assert node._exp13_sig_state["quality"] == pytest.approx(0.4)


def test_the_message_contract_states_the_producer_clock_and_the_absent_quality(world, classes):
    text = MSG_DIR.joinpath("ReidSignature.msg").read_text(encoding="utf-8")
    assert "PRODUCER's clock domain" in text, (
        "the field name says utc_ns; if the file no longer says whose clock it is, a reader will "
        "assume a shared epoch that nothing establishes")
    assert "NOT an identity-match confidence" in text and "measured nothing" in text
