"""Simulation Launch File for Dual JetAuto UGVs (Tier 1 Simulation)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    pkg_bringup = get_package_share_directory('edgesync_bringup')
    timing_params = os.path.join(pkg_bringup, 'config', 'timing_budgets.yaml')
    algo_params = os.path.join(pkg_bringup, 'config', 'algorithm_thresholds.yaml')
    hw_params = os.path.join(pkg_bringup, 'config', 'hardware_params.yaml')

    # Robot 1 (ugv_1 - Initial Owner)
    robot_1_nodes = GroupAction([
        PushRosNamespace('ugv_1'),
        Node(
            package='edgesync_coordination',
            executable='coordination_node',
            name='coordination_node',
            parameters=[timing_params, algo_params, {'robot_id': 'ugv_1', 'peer_id': 'ugv_2', 'is_initial_owner': True}],
            output='screen'
        ),
        Node(
            package='edgesync_estimation',
            executable='estimation_node',
            name='estimation_node',
            parameters=[timing_params, algo_params, hw_params],
            output='screen'
        ),
        Node(
            package='edgesync_identity',
            executable='identity_node',
            name='identity_node',
            parameters=[algo_params],
            output='screen'
        ),
    ])

    # Robot 2 (ugv_2 - Initial Standby / Receiver)
    robot_2_nodes = GroupAction([
        PushRosNamespace('ugv_2'),
        Node(
            package='edgesync_coordination',
            executable='coordination_node',
            name='coordination_node',
            parameters=[timing_params, algo_params, {'robot_id': 'ugv_2', 'peer_id': 'ugv_1', 'is_initial_owner': False}],
            output='screen'
        ),
        Node(
            package='edgesync_acquisition',
            executable='liveness_publisher',
            name='liveness_publisher',
            parameters=[{'robot_id': 'ugv_2', 'peer_id': 'ugv_1'}],
            output='screen'
        ),
        Node(
            package='edgesync_identity',
            executable='identity_node',
            name='identity_node',
            parameters=[algo_params],
            output='screen'
        ),
    ])

    return LaunchDescription([
        robot_1_nodes,
        robot_2_nodes,
    ])
