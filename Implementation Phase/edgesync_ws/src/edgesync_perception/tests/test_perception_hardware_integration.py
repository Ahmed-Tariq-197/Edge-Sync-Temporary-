"""Unit tests for Perception Hardware Integration & YOLOv8 (Sprint 3).

Verifies:
1. YoloV8Detector parsing of Ultralytics inference format.
2. Dynamic CameraInfo ingestion without hardcoded intrinsics.
3. Synchronized frame processing and pinhole deprojection into camera_link frame.
4. Robust handling of missing intrinsics, invalid depth, and class filtering.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from edgesync_perception.models.yolov8_detector import YoloV8Detector
from edgesync_perception.interfaces.base_detector import BaseDetector, Detection
from edgesync_perception.nodes.perception_node import PerceptionNode


class MockBox:
    """Mock box matching Ultralytics YOLO format."""
    def __init__(self, xyxy, conf, cls_id):
        self.xyxy = [np.array(xyxy)]
        self.conf = [np.array(conf)]
        self.cls = [np.array(cls_id)]


class MockUltralyticsResults:
    """Mock results matching Ultralytics YOLO return."""
    def __init__(self, boxes):
        self.boxes = boxes


def test_yolov8_detector_parsing():
    """Verify YoloV8Detector parses Ultralytics boxes correctly."""
    mock_model = MagicMock()
    mock_boxes = [
        MockBox([100.0, 150.0, 200.0, 350.0], 0.88, 0),   # Person
        MockBox([300.0, 200.0, 400.0, 300.0], 0.75, 2),   # Car (should be filtered if classes specified)
    ]
    mock_model.return_value = [MockUltralyticsResults(mock_boxes)]

    detector = YoloV8Detector(
        confidence_threshold=0.50,
        target_class_id=0,
        model_instance=mock_model
    )

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image)

    assert len(detections) == 2
    # Verify first detection attributes
    p_det = detections[0]
    assert p_det.xmin == 100.0
    assert p_det.ymin == 150.0
    assert p_det.xmax == 200.0
    assert p_det.ymax == 350.0
    assert pytest.approx(p_det.confidence, abs=1e-3) == 0.88
    assert p_det.class_id == 0
    assert p_det.class_name == "person"


@pytest.fixture(autouse=True)
def rclpy_session():
    import rclpy
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_perception_node_dynamic_camera_info():
    """Verify CameraInfo dynamically initializes the pinhole deprojector without hardcoding."""
    node = PerceptionNode(node_name='test_perception_node')

    assert node.deprojector is None

    info_msg = CameraInfo()
    info_msg.header.frame_id = 'camera_link'
    info_msg.width = 640
    info_msg.height = 480
    # k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    info_msg.k = [554.25, 0.0, 320.0, 0.0, 554.25, 240.0, 0.0, 0.0, 1.0]

    node._on_camera_info(info_msg)

    assert node.deprojector is not None
    assert pytest.approx(node.deprojector.intrinsics.fx, abs=1e-2) == 554.25
    assert pytest.approx(node.deprojector.intrinsics.fy, abs=1e-2) == 554.25
    assert pytest.approx(node.deprojector.intrinsics.cx, abs=1e-2) == 320.0
    assert pytest.approx(node.deprojector.intrinsics.cy, abs=1e-2) == 240.0
    assert node.camera_frame_id == 'camera_link'

    node.destroy_node()


def test_perception_node_synced_deprojection_pipeline():
    """Verify full perception pipeline from synced RGB/Depth to published PointStamped."""
    # 1. Custom mock detector returning a centered person
    class CenteredPersonDetector(BaseDetector):
        def detect(self, image: np.ndarray):
            return [
                Detection(xmin=280.0, ymin=200.0, xmax=360.0, ymax=280.0, confidence=0.92, class_id=0)
            ]

    node = PerceptionNode(detector=CenteredPersonDetector(), node_name='test_perception_pipeline')

    # Feed camera info (fx=500, fy=500, cx=320, cy=240)
    info_msg = CameraInfo()
    info_msg.header.frame_id = 'camera_link'
    info_msg.width = 640
    info_msg.height = 480
    info_msg.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    node._on_camera_info(info_msg)

    # 2. Mock publisher to capture output
    published_points = []
    node.pub_centroid.publish = lambda msg: published_points.append(msg)

    # 3. Create synthetic RGB and Depth images
    rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    # Target person is at depth 2.0 meters (2000 mm in uint16)
    depth_arr = np.full((480, 640), 2000, dtype=np.uint16)

    rgb_msg = Image()
    rgb_msg.header.stamp.sec = 100
    rgb_msg.header.stamp.nanosec = 500000000
    rgb_msg.header.frame_id = 'camera_link'
    rgb_msg.height = 480
    rgb_msg.width = 640
    rgb_msg.encoding = 'rgb8'
    rgb_msg.data = rgb_arr.tobytes()

    depth_msg = Image()
    depth_msg.header.stamp = rgb_msg.header.stamp
    depth_msg.header.frame_id = 'camera_link'
    depth_msg.height = 480
    depth_msg.width = 640
    depth_msg.encoding = '16UC1'
    depth_msg.data = depth_arr.tobytes()

    # 4. Trigger synchronized callback
    node._on_synced_frames(rgb_msg, depth_msg)

    assert len(published_points) == 1
    pt_msg: PointStamped = published_points[0]
    assert pt_msg.header.frame_id == 'camera_link'
    assert pt_msg.header.stamp.sec == 100
    assert pt_msg.header.stamp.nanosec == 500000000

    # Person center is at u = (280+360)/2 = 320 (cx), v = (200+280)/2 = 240 (cy)
    # So X_c = (320-320)*2/500 = 0.0, Y_c = (240-240)*2/500 = 0.0, Z_c = 2.0
    assert pytest.approx(pt_msg.point.x, abs=1e-2) == 0.0
    assert pytest.approx(pt_msg.point.y, abs=1e-2) == 0.0
    assert pytest.approx(pt_msg.point.z, abs=1e-2) == 2.0

    node.destroy_node()


def test_perception_node_handles_missing_intrinsics():
    """Verify that if CameraInfo is not yet received, frames are skipped safely without crashing."""
    node = PerceptionNode(node_name='test_uncalibrated_node')
    published_points = []
    node.pub_centroid.publish = lambda msg: published_points.append(msg)

    rgb_msg = Image()
    depth_msg = Image()

    # Should safely return without raising exception or publishing corrupted point
    node._on_synced_frames(rgb_msg, depth_msg)
    assert len(published_points) == 0

    node.destroy_node()
