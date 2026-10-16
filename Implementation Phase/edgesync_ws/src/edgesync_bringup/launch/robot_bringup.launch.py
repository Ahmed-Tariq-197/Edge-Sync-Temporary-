"""Master Bringup Launch File for Single JetAuto UGV in EdgeSync (Sprint 6)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('edgesync_bringup')

    timing_params = os.path.join(pkg_bringup, 'config', 'timing_budgets.yaml')
    algo_params = os.path.join(pkg_bringup, 'config', 'algorithm_thresholds.yaml')
    hw_params = os.path.join(pkg_bringup, 'config', 'hw_params.yaml')
    coordination_params = os.path.join(pkg_bringup, 'config', 'coordination_params.yaml')

    arg_robot_id = DeclareLaunchArgument('robot_id', default_value='ugv_1', description='Robot namespace/ID')
    arg_peer_id = DeclareLaunchArgument('peer_id', default_value='ugv_2', description='Peer UGV ID')
    arg_is_owner = DeclareLaunchArgument('is_initial_owner', default_value='true', description='Is initial target owner')

    robot_id = LaunchConfiguration('robot_id')
    peer_id = LaunchConfiguration('peer_id')
    is_owner = LaunchConfiguration('is_initial_owner')

    node_namespace = [LaunchConfiguration('robot_id'), '/edgesync']

    swarm_topics = [
        'handoff_request',
        'handoff_response',
        'handoff_confirm',
        'handoff_confirm_ack',
        'handoff_cancel',
        'ownership_event',
        'state_heartbeat',
    ]

    global_remappings = [
        (topic, f'/edgesync_swarm/{topic}') for topic in swarm_topics
    ]
    for topic in swarm_topics:
        global_remappings.append((['/', robot_id, f'/edgesync/{topic}'], f'/edgesync_swarm/{topic}'))
        global_remappings.append((['/', peer_id, f'/edgesync/{topic}'], f'/edgesync_swarm/{topic}'))

    node_perception = Node(
        package='edgesync_perception',
        executable='perception_node',
        name='perception_node',
        namespace=node_namespace,
        parameters=[hw_params, algo_params],
        output='screen'
    )

    node_estimation = Node(
        package='edgesync_estimation',
        executable='estimation_node',
        name='estimation_node',
        namespace=node_namespace,
        parameters=[
            timing_params, algo_params, hw_params,
            {'robot_id': robot_id}
        ],
        output='screen'
    )

    node_identity = Node(
        package='edgesync_identity',
        executable='identity_node',
        name='identity_node',
        namespace=node_namespace,
        parameters=[
            algo_params,
            {'robot_id': robot_id}
        ],
        output='screen'
    )

    node_coordination = Node(
        package='edgesync_coordination',
        executable='coordination_node',
        name='coordination_node',
        namespace=node_namespace,
        remappings=global_remappings,
        parameters=[
            coordination_params, timing_params, algo_params, hw_params,
            {'robot_id': robot_id, 'peer_id': peer_id, 'is_initial_owner': is_owner}
        ],
        output='screen'
    )

    node_target_acquisition = Node(
        package='edgesync_acquisition',
        executable='target_acquisition_node',
        name='target_acquisition_node',
        namespace=node_namespace,
        remappings=global_remappings,
        parameters=[
            timing_params, algo_params, hw_params,
            {'robot_id': robot_id, 'peer_id': peer_id}
        ],
        output='screen'
    )

    node_safety = Node(
        package='edgesync_navigation',
        executable='safety_authority_node',
        name='safety_authority_node',
        namespace=node_namespace,
        remappings=global_remappings,
        parameters=[
            timing_params,
            hw_params,
            {'robot_id': robot_id, 'peer_id': peer_id}
        ],
        output='screen'
    )

    node_monitoring = Node(
        package='edgesync_observability',
        executable='monitoring_node',
        name='monitoring_node',
        namespace=node_namespace,
        remappings=global_remappings,
        parameters=[
            timing_params,
            {'robot_id': robot_id, 'peer_id': peer_id}
        ],
        output='screen'
    )

    return LaunchDescription([
        arg_robot_id,
        arg_peer_id,
        arg_is_owner,
        node_perception,
        node_estimation,
        node_identity,
        node_coordination,
        node_target_acquisition,
        node_safety,
        node_monitoring,
    ])
