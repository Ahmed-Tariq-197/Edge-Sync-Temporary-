"""ROS-level tests: IdentityNode selects its Re-ID backbone from the reid_backbone parameter."""
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("edgesync_msgs.srv")

from rclpy.parameter import Parameter  # noqa: E402

from edgesync_identity.models.mobilenetv2_trt_extractor import MobileNetV2TrtExtractor  # noqa: E402
from edgesync_identity.models.osnet_trt_extractor import OsnetTrtExtractor  # noqa: E402
from edgesync_identity.nodes.identity_node import IdentityNode  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def _node_with_backbone(name):
    overrides = [Parameter("reid_backbone", Parameter.Type.STRING, name)]
    return IdentityNode(parameter_overrides=overrides)


def test_default_backbone_is_osnet():
    node = IdentityNode()
    assert node.reid_backbone == "osnet"
    assert isinstance(node.extractor, OsnetTrtExtractor)
    node.destroy_node()


def test_parameter_selects_mobilenetv2():
    node = _node_with_backbone("mobilenetv2")
    assert node.reid_backbone == "mobilenetv2"
    assert isinstance(node.extractor, MobileNetV2TrtExtractor)
    node.destroy_node()


def test_injected_extractor_overrides_parameter():
    injected = OsnetTrtExtractor()
    node = IdentityNode(extractor=injected,
                        parameter_overrides=[Parameter("reid_backbone", Parameter.Type.STRING, "mobilenetv2")])
    assert node.extractor is injected
    node.destroy_node()
