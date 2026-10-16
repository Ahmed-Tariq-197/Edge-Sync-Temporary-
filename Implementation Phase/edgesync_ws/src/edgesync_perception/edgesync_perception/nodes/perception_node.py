"""Perception Node (C-01): Synchronized RGB-D Ingestion, YOLOv8 Person Detection, and 3D Deprojection.

Sprint 3 Hardware Integration:
- Uses message_filters.ApproximateTimeSynchronizer to align RGB and Depth image streams.
- Dynamically extracts pinhole camera intrinsics (fx, fy, cx, cy) from sensor_msgs/CameraInfo.
- Executes YOLOv8 inference (ultralytics format) to detect target person (class 0).
- Extracts median depth from bounding box ROI and deprojects to 3D centroid in camera_link frame.
- Publishes PointStamped to /{robot_id}/edgesync/perception/target_centroid_camera.
- SOLID Compliance: Dependency injection for detector and tracker, zero hardcoded parameters.
"""
from typing import Optional, List
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from ..interfaces.base_detector import BaseDetector, Detection
from ..interfaces.base_tracker import BaseTracker
from ..models.yolov8_detector import YoloV8Detector
from ..models.yolo_trt_detector import YoloTrtDetector
from ..models.bytetrack_mot import ByteTrackMOT
from ..math.depth_deprojector import CameraIntrinsics, DepthDeprojector

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False


class PerceptionNode(Node):
    """ROS 2 Node responsible for synchronized vision processing and 3D centroid extraction."""

    def __init__(self,
                 detector: Optional[BaseDetector] = None,
                 tracker: Optional[BaseTracker] = None,
                 node_name: str = 'perception_node'):
        super().__init__(node_name)

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'robot_1')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value

        # Default topics adhere to /{robot_id}/edgesync/ convention
        default_rgb = f'/{self.robot_id}/camera/color/image_raw'
        default_depth = f'/{self.robot_id}/camera/aligned_depth_to_color/image_raw'
        default_info = f'/{self.robot_id}/camera/color/camera_info'
        default_centroid = f'/{self.robot_id}/edgesync/perception/target_centroid_camera'

        self.declare_parameter('rgb_topic', default_rgb)
        self.declare_parameter('depth_topic', default_depth)
        self.declare_parameter('camera_info_topic', default_info)
        self.declare_parameter('output_centroid_topic', default_centroid)
        self.declare_parameter('camera_frame_id', 'camera_link')

        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('target_class_id', 0)  # COCO class 0: person
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('detector_type', 'yolov8')  # 'yolov8' or 'trt'
        self.declare_parameter('sync_slop_sec', 0.05)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('enable_tracking', True)

        # 2. Extract Parameter Values
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.output_centroid_topic = self.get_parameter('output_centroid_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value

        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.target_class_id = self.get_parameter('target_class_id').get_parameter_value().integer_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        detector_type = self.get_parameter('detector_type').get_parameter_value().string_value
        sync_slop = self.get_parameter('sync_slop_sec').get_parameter_value().double_value
        sync_queue = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        self.enable_tracking = self.get_parameter('enable_tracking').get_parameter_value().bool_value

        # 3. CV Bridge Initialization
        self.cv_bridge = CvBridge() if _HAS_CV_BRIDGE else None

        # 4. Dependency Injection (SOLID: DIP & OCP)
        if detector is not None:
            self.detector: BaseDetector = detector
        elif detector_type.lower() == 'trt':
            self.detector = YoloTrtDetector(
                engine_path=model_path,
                confidence_threshold=self.conf_thresh
            )
        else:
            self.detector = YoloV8Detector(
                model_path=model_path,
                confidence_threshold=self.conf_thresh,
                target_class_id=self.target_class_id
            )

        self.tracker: Optional[BaseTracker] = tracker if tracker is not None else (
            ByteTrackMOT() if self.enable_tracking else None
        )

        # 5. Intrinsics & Deprojector State
        self.deprojector: Optional[DepthDeprojector] = None
        self._intrinsics_calibrated: bool = False

        # 6. Subscribers & Publishers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=sync_queue
        )

        # Dynamic Camera Info Subscriber
        self.sub_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            10
        )

        # Synchronized RGB and Depth Subscribers using ApproximateTimeSynchronizer
        self.sub_rgb = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=sensor_qos
        )
        self.sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=sensor_qos
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=sync_queue,
            slop=sync_slop
        )
        self.sync.registerCallback(self._on_synced_frames)

        # Publisher for 3D centroid in camera_link frame
        self.pub_centroid = self.create_publisher(
            PointStamped,
            self.output_centroid_topic,
            10
        )

        self.get_logger().info(
            f"EdgeSync Perception Node initialized for robot '{self.robot_id}'. "
            f"Listening to RGB: {self.rgb_topic}, Depth: {self.depth_topic}, Info: {self.camera_info_topic}"
        )

    def _on_camera_info(self, msg: CameraInfo):
        """Dynamically extract pinhole camera intrinsics from CameraInfo."""
        # msg.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn(
                f"Received uncalibrated CameraInfo (fx={fx}, fy={fy}). Ignoring until calibrated.",
                throttle_duration_sec=5.0
            )
            return

        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=int(msg.width),
            height=int(msg.height)
        )

        self.deprojector = DepthDeprojector(intrinsics)
        self._intrinsics_calibrated = True

        # Use camera_link from message header if populated
        if msg.header.frame_id:
            self.camera_frame_id = msg.header.frame_id

        self.get_logger().info(
            f"Dynamic camera intrinsics configured: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, "
            f"frame_id='{self.camera_frame_id}'"
        )

    def _decode_rgb_image(self, msg: Image) -> Optional[np.ndarray]:
        """Decode sensor_msgs/Image to HxWx3 uint8 RGB array."""
        try:
            if self.cv_bridge is not None:
                return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Pure numpy decoding fallback
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ['rgb8', 'bgr8']:
                img = arr.reshape((msg.height, msg.width, 3))
                if msg.encoding == 'bgr8':
                    img = img[:, :, ::-1]
                return img
            elif msg.encoding in ['rgba8', 'bgra8']:
                img = arr.reshape((msg.height, msg.width, 4))
                if msg.encoding == 'bgra8':
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = img[:, :, :3]
                return img
            elif msg.encoding in ['mono8']:
                gray = arr.reshape((msg.height, msg.width))
                return np.stack([gray, gray, gray], axis=-1)
            else:
                return arr.reshape((msg.height, msg.width, -1))[:, :, :3]
        except Exception as e:
            self.get_logger().warn(f"Failed to decode RGB image: {e}")
            return None

    def _decode_depth_image(self, msg: Image) -> Optional[np.ndarray]:
        """Decode sensor_msgs/Image to HxW depth array (uint16 mm or float32 meters)."""
        try:
            if self.cv_bridge is not None:
                return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Pure numpy decoding fallback
            if msg.encoding in ['16UC1', 'mono16']:
                return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            elif msg.encoding in ['32FC1']:
                return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            else:
                return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        except Exception as e:
            self.get_logger().warn(f"Failed to decode Depth image: {e}")
            return None

    def _on_synced_frames(self, rgb_msg: Image, depth_msg: Image):
        """Synchronized callback processing aligned RGB and Depth frames."""
        # 1. Gate: Intrinsics must be loaded
        if self.deprojector is None:
            self.get_logger().info(
                "Awaiting CameraInfo intrinsics before deprojecting frames...",
                throttle_duration_sec=5.0
            )
            return

        # 2. Decode RGB and Depth images
        rgb_data = self._decode_rgb_image(rgb_msg)
        depth_data = self._decode_depth_image(depth_msg)

        if rgb_data is None or depth_data is None:
            return

        # 3. YOLOv8 2D Inference (target person extraction)
        detections = self.detector.detect(rgb_data)
        person_dets: List[Detection] = [
            d for d in detections
            if d.class_id == self.target_class_id and d.confidence >= self.conf_thresh
        ]

        if not person_dets:
            return

        # 4. Multi-Object Tracking or Primary Selection
        if self.tracker is not None:
            tracks = self.tracker.update(person_dets)
            if tracks:
                primary_bbox = tracks[0].bbox
            else:
                primary_bbox = max(person_dets, key=lambda d: d.confidence).bbox
        else:
            primary_bbox = max(person_dets, key=lambda d: d.confidence).bbox

        # 5. Pinhole Deprojection from Central Masked ROI
        result = self.deprojector.deproject_bbox(depth_data, primary_bbox)
        if result is None:
            # Depth invalid, NaN, or out of range
            return

        # 6. Publish 3D Centroid in camera_link frame
        point_msg = PointStamped()
        point_msg.header.stamp = rgb_msg.header.stamp
        point_msg.header.frame_id = self.camera_frame_id
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
