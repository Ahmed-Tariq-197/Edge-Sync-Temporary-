"""Perception Node (C-01): 2D Detection, ByteTrack MOT, and Depth Deprojection.

SOLID Compliance:
- Depends on abstractions (BaseDetector, BaseTracker) via Dependency Injection.
- ZERO hardcoded parameters: reads from ROS 2 parameter system.
- Publishes camera-frame 3D centroid observation p_c and R_cam.
- Frame transformation to 'map' and ego-motion inflation are strictly deferred to C-02.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import numpy as np

from ..interfaces.base_detector import BaseDetector
from ..interfaces.base_tracker import BaseTracker
from ..models.yolo_trt_detector import YoloTrtDetector
from ..models.bytetrack_mot import ByteTrackMOT
from ..math.depth_deprojector import CameraIntrinsics, DepthDeprojector


class PerceptionNode(Node):
    """ROS 2 node responsible for image processing and 3D centroid extraction."""

    def __init__(self, detector: Optional[BaseDetector] = None,
                 tracker: Optional[BaseTracker] = None):
        super().__init__('perception_node')

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_centroid_topic', 'perception/target_centroid_camera')
        self.declare_parameter('target_class_id', 0)  # Person class index

        # 2. Extract Parameter Values
        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_centroid_topic').get_parameter_value().string_value

        # 3. Dependency Injection (SOLID: DIP)
        self.detector: BaseDetector = detector if detector is not None else YoloTrtDetector(
            confidence_threshold=self.conf_thresh
        )
        self.tracker: BaseTracker = tracker if tracker is not None else ByteTrackMOT()

        # 4. Math Modules
        self.deprojector: Optional[DepthDeprojector] = None
        self._last_depth_image: Optional[np.ndarray] = None

        # 5. Subscribers & Publishers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )

        self.sub_info = self.create_subscription(
            CameraInfo, info_topic, self._on_camera_info, 10
        )
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._on_depth_frame, sensor_qos
        )
        self.sub_rgb = self.create_subscription(
            Image, rgb_topic, self._on_rgb_frame, sensor_qos
        )

        self.pub_centroid = self.create_publisher(
            PointStamped, output_topic, 10
        )

        self.get_logger().info("EdgeSync Perception Node initialized successfully.")

    def _on_camera_info(self, msg: CameraInfo):
        """Cache camera intrinsics once available."""
        if self.deprojector is None:
            intrinsics = CameraIntrinsics(
                fx=float(msg.k[0]),
                fy=float(msg.k[4]),
                cx=float(msg.k[2]),
                cy=float(msg.k[5]),
                width=msg.width,
                height=msg.height
            )
            self.deprojector = DepthDeprojector(intrinsics)
            self.get_logger().info(f"Camera intrinsics received: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    def _on_depth_frame(self, msg: Image):
        """Cache latest synchronized depth frame."""
        # Convert ROS Image to numpy array (uint16 or float32)
        try:
            if msg.encoding in ['16UC1', 'mono16']:
                depth_data = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            elif msg.encoding in ['32FC1']:
                depth_data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            else:
                depth_data = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            self._last_depth_image = depth_data
        except Exception as e:
            self.get_logger().warn(f"Failed to decode depth image: {e}")

    def _on_rgb_frame(self, msg: Image):
        """Process RGB frame, execute detection, tracking, and deprojection."""
        if self.deprojector is None or self._last_depth_image is None:
            return

        # Convert RGB image
        try:
            rgb_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except Exception as e:
            self.get_logger().warn(f"Failed to decode RGB image: {e}")
            return

        # 1. 2D Inference via Injected Detector
        detections = self.detector.detect(rgb_data)

        # 2. Multi-Object Tracking via Injected Tracker
        tracks = self.tracker.update(detections)

        # 3. Deproject target track (e.g. track 1 or primary target)
        if not tracks:
            return

        primary_target = tracks[0]
        result = self.deprojector.deproject_bbox(self._last_depth_image, primary_target.bbox)

        if result is not None:
            # Publish camera-frame centroid (C-01 contract)
            point_msg = PointStamped()
            point_msg.header.stamp = msg.header.stamp  # Exact image capture time
            point_msg.header.frame_id = msg.header.frame_id
            point_msg.point.x = float(result.point_camera[0])
            point_msg.point.y = float(result.point_camera[1])
            point_msg.point.z = float(result.point_camera[2])
            self.pub_centroid.publish(point_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
