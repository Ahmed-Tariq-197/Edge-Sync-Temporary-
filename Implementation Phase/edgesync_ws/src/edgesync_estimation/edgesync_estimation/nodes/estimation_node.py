"""Estimation Node (C-02 & C-03): TF2 Coordinate Transformation, 3D Target EKF & Covariance Propagation.

Sprint 3 Hardware Integration:
- Uses tf2_ros Buffer and TransformListener to transform 3D observations from camera_link to map.
- Actively looks up transform at exact message timestamp stamp_utc_ns.
- Handles TransformException gracefully without crashing when TF tree is delayed.
- Inflates camera measurement noise with robot ego-motion uncertainty to compute R_total.
- Updates 6-DoF TargetEKF3D in the shared map frame.
- Publishes filtered global state to /{robot_id}/edgesync/target/state_3d.
"""
from typing import Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener, TransformException

from edgesync_msgs.msg import Target3DState, TTOEstimate
from ..interfaces.base_estimator import BaseStateEstimator
from ..interfaces.base_tto_predictor import BaseTTOPredictor
from ..core.target_ekf_3d import TargetEKF3D
from ..core.tto_calculator import AnalyticalTTOCalculator
from ..core.ego_covariance import EgoCovariancePropagator


class EstimationNode(Node):
    """ROS 2 Node for 3D Target State Estimation and TF2 Coordinate Transformation."""

    def __init__(self,
                 estimator: Optional[BaseStateEstimator] = None,
                 tto_predictor: Optional[BaseTTOPredictor] = None,
                 node_name: str = 'estimation_node'):
        super().__init__(node_name)

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'robot_1')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value

        default_centroid = f'/{self.robot_id}/edgesync/perception/target_centroid_camera'
        default_state = f'/{self.robot_id}/edgesync/target/state_3d'
        default_amcl = f'/{self.robot_id}/amcl_pose'

        self.declare_parameter('shared_map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('base_link_frame', f'{self.robot_id}/base_link')
        self.declare_parameter('centroid_topic', default_centroid)
        self.declare_parameter('output_state_topic', default_state)
        self.declare_parameter('amcl_pose_topic', default_amcl)
        self.declare_parameter('enable_ego_covariance', True)
        self.declare_parameter('ekf_process_accel_noise_q', 0.80)
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('target_id', 'target_human')

        self.declare_parameter('tto_hfov_deg', 60.0)
        self.declare_parameter('tto_vfov_deg', 49.5)
        self.declare_parameter('tto_min_range_m', 0.40)
        self.declare_parameter('tto_max_range_m', 5.50)

        # 2. Extract Parameters
        self.map_frame = self.get_parameter('shared_map_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_link_frame').get_parameter_value().string_value
        self.centroid_topic = self.get_parameter('centroid_topic').get_parameter_value().string_value
        self.output_state_topic = self.get_parameter('output_state_topic').get_parameter_value().string_value
        self.amcl_pose_topic = self.get_parameter('amcl_pose_topic').get_parameter_value().string_value
        self.enable_ego_cov = self.get_parameter('enable_ego_covariance').get_parameter_value().bool_value
        self.tf_timeout_sec = self.get_parameter('tf_timeout_sec').get_parameter_value().double_value
        self.target_id = self.get_parameter('target_id').get_parameter_value().string_value
        q_accel = self.get_parameter('ekf_process_accel_noise_q').get_parameter_value().double_value

        hfov = self.get_parameter('tto_hfov_deg').get_parameter_value().double_value
        vfov = self.get_parameter('tto_vfov_deg').get_parameter_value().double_value
        min_r = self.get_parameter('tto_min_range_m').get_parameter_value().double_value
        max_r = self.get_parameter('tto_max_range_m').get_parameter_value().double_value

        # 3. Dependency Injection (SOLID: DIP)
        self.estimator: BaseStateEstimator = estimator if estimator is not None else TargetEKF3D(
            target_id=self.target_id,
            shared_frame=self.map_frame,
            q_accel=q_accel
        )
        self.tto_predictor: BaseTTOPredictor = tto_predictor if tto_predictor is not None else AnalyticalTTOCalculator(
            hfov_deg=hfov, vfov_deg=vfov, min_range_m=min_r, max_range_m=max_r
        )

        # 4. TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 5. Subscribers & Publishers
        self.sub_centroid = self.create_subscription(
            PointStamped,
            self.centroid_topic,
            self._on_centroid_received,
            10
        )
        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self._on_amcl_pose,
            10
        )
        self.pub_state = self.create_publisher(
            Target3DState,
            self.output_state_topic,
            10
        )
        self.pub_tto = self.create_publisher(
            TTOEstimate,
            f'/{self.robot_id}/edgesync/tto/estimate',
            10
        )

        self._last_stamp_s: Optional[float] = None
        self._robot_pose_cov = np.diag([0.02, 0.02, 0.005])  # Safe prior [x, y, yaw]

        self.get_logger().info(
            f"EdgeSync Estimation Node initialized for robot '{self.robot_id}'. "
            f"Subscribing to: '{self.centroid_topic}', Publishing to: '{self.output_state_topic}'"
        )

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        """Update robot pose uncertainty dynamically from localization."""
        # 6x6 covariance in row-major order: index 0 is var(x), 7 is var(y), 35 is var(yaw)
        cov = msg.pose.covariance
        var_x = max(1e-4, float(cov[0]))
        var_y = max(1e-4, float(cov[7]))
        var_yaw = max(1e-4, float(cov[35]))
        self._robot_pose_cov = np.diag([var_x, var_y, var_yaw])

    def _on_centroid_received(self, msg: PointStamped):
        """Process centroid observation, transform to map, inflate ego noise, and update EKF."""
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        stamp_utc_ns = int(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        target_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        source_frame = msg.header.frame_id if msg.header.frame_id else self.camera_frame

        # 1. TF2 Coordinate Transform: camera_link -> map at exact stamp_utc_ns
        try:
            tf_cam_to_map = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                target_time,
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF2 lookup failure ({source_frame} -> {self.map_frame}) at {stamp_utc_ns} ns: {ex}",
                throttle_duration_sec=2.0
            )
            return
        except Exception as ex:
            self.get_logger().warn(
                f"Unexpected error looking up transform ({source_frame} -> {self.map_frame}): {ex}",
                throttle_duration_sec=2.0
            )
            return

        # 2. Extract translation and rotation quaternion
        tx = float(tf_cam_to_map.transform.translation.x)
        ty = float(tf_cam_to_map.transform.translation.y)
        tz = float(tf_cam_to_map.transform.translation.z)
        t_cam_to_map = np.array([tx, ty, tz], dtype=np.float64)

        qx = float(tf_cam_to_map.transform.rotation.x)
        qy = float(tf_cam_to_map.transform.rotation.y)
        qz = float(tf_cam_to_map.transform.rotation.z)
        qw = float(tf_cam_to_map.transform.rotation.w)

        # 3. Apply exact Hamilton vector quaternion rotation: v' = v + 2 * (u x (u x v + w * v))
        pt_cam = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        u_vec = np.array([qx, qy, qz], dtype=np.float64)
        cross_uv = np.cross(u_vec, pt_cam)
        v_rot = pt_cam + 2.0 * np.cross(u_vec, cross_uv + qw * pt_cam)
        pt_map = v_rot + t_cam_to_map

        # 4. Compute Time Delta dt and Run EKF Predict Step
        if self._last_stamp_s is None:
            dt = 0.033
        else:
            dt = max(0.001, stamp_sec - self._last_stamp_s)
        self._last_stamp_s = stamp_sec

        self.estimator.predict(dt)

        # 5. Compute Camera Measurement Covariance R_cam and Total Covariance R_total
        z_val = max(0.40, float(pt_cam[2]))
        var_x = ((z_val / 554.0) * 1.5) ** 2
        var_y = ((z_val / 554.0) * 1.5) ** 2
        var_z = (0.015 * (z_val ** 2) + 0.01) ** 2
        r_cam = np.diag([var_x, var_y, var_z])

        # Planar robot yaw in map frame extracted from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        robot_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        # Extrinsics from camera to base_link
        r_cam_to_base = np.eye(3, dtype=np.float64)
        t_cam_to_base = np.array([0.20, 0.0, 0.15], dtype=np.float64)

        r_total = EgoCovariancePropagator.propagate_covariance(
            point_camera=pt_cam,
            r_cam=r_cam,
            sigma_robot_pose=self._robot_pose_cov,
            r_cam_to_base=r_cam_to_base,
            t_cam_to_base=t_cam_to_base,
            robot_yaw=robot_yaw,
            enable_ego_propagation=self.enable_ego_cov
        )

        # 6. EKF Measurement Update in map frame
        state, cov = self.estimator.update(pt_map, r_total)

        # 7. Publish Filtered Global State
        out_msg = Target3DState()
        out_msg.target_id = self.target_id
        out_msg.frame_id = self.map_frame
        out_msg.position = [float(state[0]), float(state[1]), float(state[2])]
        out_msg.velocity = [float(state[3]), float(state[4]), float(state[5])]
        out_msg.full_covariance = cov.flatten().astype(np.float32).tolist()
        out_msg.confidence = float(np.clip(1.0 / (np.trace(cov[:3, :3]) + 0.1), 0.0, 1.0))
        out_msg.stamp_ns = stamp_utc_ns

        self.pub_state.publish(out_msg)

        # 8. Compute and Publish TTO Estimate (Interface I-04 / C-03 -> C-04)
        cam_pos_map = np.array([tx, ty, tz], dtype=np.float64)
        r_cam_map = np.array([
            [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
            [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
            [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
        ], dtype=np.float64)

        tto_res = self.tto_predictor.compute_tto(state, cov, cam_pos_map, r_cam_map)
        tto_msg = TTOEstimate()
        tto_msg.target_id = self.target_id
        tto_msg.frame_id = self.map_frame
        tto_msg.tto_hat_s = float(tto_res.tto_hat_s)
        tto_msg.sigma_tto_s = float(tto_res.sigma_tto_s)
        tto_msg.tto_safe_s = float(tto_res.tto_safe_s)
        tto_msg.exit_boundary = str(tto_res.exit_boundary)
        tto_msg.stamp_utc_ns = stamp_utc_ns
        for pt in tto_res.predicted_trajectory:
            tto_msg.predicted_trajectory.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
        self.pub_tto.publish(tto_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
