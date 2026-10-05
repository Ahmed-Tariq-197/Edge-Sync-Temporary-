"""Unit tests for EstimationNode TF2 & EKF Hardware Wiring (Sprint 3).

Verifies:
1. TF2 coordinate transform from camera_link to map at exact timestamp stamp_utc_ns.
2. EKF update with transformed global coordinates and inflated R_total.
3. Publication of Target3DState on /{robot_id}/edgesync/target/state_3d.
4. Graceful handling of TransformException without crashing.
5. Dynamic robot pose covariance updates from AMCL localization.
"""
import math
import numpy as np
import pytest

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, PoseWithCovarianceStamped
from edgesync_msgs.msg import Target3DState
from edgesync_estimation.nodes.estimation_node import EstimationNode


@pytest.fixture(autouse=True)
def rclpy_session():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_estimation_node_tf2_and_ekf_pipeline():
    """Verify TF2 lookup, map frame transform, EKF update, and state publication."""
    node = EstimationNode(node_name='test_estimation_pipeline')

    # Capture published Target3DState
    published_states = []
    node.pub_state.publish = lambda msg: published_states.append(msg)

    # 1. Populate TF2 buffer with known transform from camera_link to map
    tf_msg = TransformStamped()
    tf_msg.header.stamp.sec = 100
    tf_msg.header.stamp.nanosec = 0
    tf_msg.header.frame_id = 'map'
    tf_msg.child_frame_id = 'camera_link'

    # Translation: robot is at (5.0, 3.0, 0.0) in map
    tf_msg.transform.translation.x = 5.0
    tf_msg.transform.translation.y = 3.0
    tf_msg.transform.translation.z = 0.0

    # Rotation: Pure identity (0 deg yaw) for direct verification
    tf_msg.transform.rotation.x = 0.0
    tf_msg.transform.rotation.y = 0.0
    tf_msg.transform.rotation.z = 0.0
    tf_msg.transform.rotation.w = 1.0

    node.tf_buffer.set_transform(tf_msg, 'default_authority')

    # 2. Inject 3D Centroid in camera_link frame at (0.0, 0.0, 2.0)
    point_msg = PointStamped()
    point_msg.header.stamp = tf_msg.header.stamp
    point_msg.header.frame_id = 'camera_link'
    point_msg.point.x = 0.0
    point_msg.point.y = 0.0
    point_msg.point.z = 2.0

    node._on_centroid_received(point_msg)

    # 3. Assertions
    assert len(published_states) == 1
    state_msg: Target3DState = published_states[0]

    assert state_msg.target_id == 'target_human'
    assert state_msg.frame_id == 'map'
    assert state_msg.stamp_ns == 100_000_000_000

    # Global position should be camera_pos + target_offset = (5.0+0, 3.0+0, 0.0+2.0) = (5.0, 3.0, 2.0)
    assert pytest.approx(state_msg.position[0], abs=1e-3) == 5.0
    assert pytest.approx(state_msg.position[1], abs=1e-3) == 3.0
    assert pytest.approx(state_msg.position[2], abs=1e-3) == 2.0

    # Verify 36-element full covariance
    assert len(state_msg.full_covariance) == 36
    cov_mat = np.array(state_msg.full_covariance).reshape(6, 6)
    assert np.allclose(cov_mat, cov_mat.T, atol=1e-5)  # Symmetry
    assert state_msg.confidence > 0.0

    node.destroy_node()


def test_estimation_node_rotational_transform():
    """Verify quaternion rotation correctly rotates target point into map frame."""
    node = EstimationNode(node_name='test_rotational_transform')
    published_states = []
    node.pub_state.publish = lambda msg: published_states.append(msg)

    # TF: Robot at (0, 0, 0) rotated by 90 degrees around Z axis (yaw = +pi/2)
    tf_msg = TransformStamped()
    tf_msg.header.stamp.sec = 50
    tf_msg.header.stamp.nanosec = 0
    tf_msg.header.frame_id = 'map'
    tf_msg.child_frame_id = 'camera_link'
    tf_msg.transform.translation.x = 0.0
    tf_msg.transform.translation.y = 0.0
    tf_msg.transform.translation.z = 0.0

    # Quaternion for +90 deg rotation around Z: [0, 0, sin(45 deg), cos(45 deg)]
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    tf_msg.transform.rotation.x = 0.0
    tf_msg.transform.rotation.y = 0.0
    tf_msg.transform.rotation.z = s
    tf_msg.transform.rotation.w = c

    node.tf_buffer.set_transform(tf_msg, 'default_authority')

    # Point in camera_link: (1.0, 0.0, 0.0) -> rotated by +90 deg becomes (0.0, 1.0, 0.0)
    point_msg = PointStamped()
    point_msg.header.stamp = tf_msg.header.stamp
    point_msg.header.frame_id = 'camera_link'
    point_msg.point.x = 1.0
    point_msg.point.y = 0.0
    point_msg.point.z = 0.0

    node._on_centroid_received(point_msg)

    assert len(published_states) == 1
    state_msg = published_states[0]
    assert pytest.approx(state_msg.position[0], abs=1e-3) == 0.0
    assert pytest.approx(state_msg.position[1], abs=1e-3) == 1.0
    assert pytest.approx(state_msg.position[2], abs=1e-3) == 0.0

    node.destroy_node()


def test_estimation_node_handles_transform_exception_gracefully():
    """Verify node handles missing/delayed TF transform without crashing."""
    node = EstimationNode(node_name='test_delayed_tf_node')
    published_states = []
    node.pub_state.publish = lambda msg: published_states.append(msg)

    # Point arrives at stamp t=200s, but NO transform is in the buffer
    point_msg = PointStamped()
    point_msg.header.stamp.sec = 200
    point_msg.header.stamp.nanosec = 0
    point_msg.header.frame_id = 'camera_link'
    point_msg.point.x = 1.0
    point_msg.point.y = 2.0
    point_msg.point.z = 3.0

    # Must execute and catch TransformException gracefully without throwing
    node._on_centroid_received(point_msg)

    # Nothing should be published on TF failure
    assert len(published_states) == 0

    node.destroy_node()


def test_estimation_node_amcl_pose_covariance_update():
    """Verify dynamic robot pose covariance update from AMCL."""
    node = EstimationNode(node_name='test_amcl_cov_node')

    pose_msg = PoseWithCovarianceStamped()
    # 6x6 covariance (36 elements): var(x)=0.05 at idx 0, var(y)=0.08 at idx 7, var(yaw)=0.01 at idx 35
    cov = [0.0] * 36
    cov[0] = 0.05
    cov[7] = 0.08
    cov[35] = 0.01
    pose_msg.pose.covariance = cov

    node._on_amcl_pose(pose_msg)

    assert pytest.approx(node._robot_pose_cov[0, 0], abs=1e-4) == 0.05
    assert pytest.approx(node._robot_pose_cov[1, 1], abs=1e-4) == 0.08
    assert pytest.approx(node._robot_pose_cov[2, 2], abs=1e-4) == 0.01

    node.destroy_node()
