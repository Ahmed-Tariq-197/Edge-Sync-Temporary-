"""Master Bringup Launch File for Single JetAuto UGV in EdgeSync (Sprint 6)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# TR-01: Re-ID launch defaults are read from algorithm_thresholds.yaml (edgesync_bringup pkg).
# ES-EXP07 §4.2: an optional experiment overlay (config/exp7_transreid.yaml) is layered on top.
from edgesync_bringup.reid_launch_defaults import (
    coerce_camera_id, describe_reid_sources, known_backbone_ids, resolve_reid_config,
    select_experiment_config)


def generate_launch_description():
    pkg_bringup = get_package_share_directory('edgesync_bringup')

    timing_params = os.path.join(pkg_bringup, 'config', 'timing_budgets.yaml')
    algo_params = os.path.join(pkg_bringup, 'config', 'algorithm_thresholds.yaml')
    hw_params = os.path.join(pkg_bringup, 'config', 'hw_params.yaml')
    coordination_params = os.path.join(pkg_bringup, 'config', 'coordination_params.yaml')

    arg_robot_id = DeclareLaunchArgument('robot_id', default_value='ugv_1', description='Robot namespace/ID')
    arg_peer_id = DeclareLaunchArgument('peer_id', default_value='ugv_2', description='Peer UGV ID')
    arg_is_owner = DeclareLaunchArgument('is_initial_owner', default_value='true', description='Is initial target owner')
    # Gate 1: Experiment 7 — Re-ID model selection (ES-EXP07-DCP-REV10 §8.1)
    # TR-01: these defaults used to be literals ('osnet_mock', ''). Because the parameter
    # dictionary below is placed after algo_params in parameters=[...], the literals silently
    # overwrote the model selection and asset paths configured in algorithm_thresholds.yaml,
    # so the stack could never launch the TransReID arm. Each default is now read from that YAML;
    # an explicit reid_model_type:=... on the command line still overrides it.
    # ES-EXP07 §4.2: the same read path layers an experiment file on top of the shared one. The
    # overlay is selected with EDGESYNC_EXPERIMENT_CONFIG=<path> rather than a launch argument,
    # because this function runs before the CLI is parsed: an argument would either be invisible
    # to the resolver or compete with the dictionary below for precedence. See
    # reid_launch_defaults.select_experiment_config.
    experiment_config, experiment_origin = select_experiment_config(algo_params)
    reid = resolve_reid_config(algo_params, experiment_config)
    reid_defaults = reid['values']
    print(describe_reid_sources(algo_params, experiment_config))
    print(f"ES-EXP07 §4.2 experiment overlay: path={experiment_config or '(none)'} "
          f"selected_by={experiment_origin} statuses={reid['statuses']}")
    if experiment_config and not reid['model_is_valid']:
        raise ValueError(
            f"[robot_bringup] {experiment_config} selects {reid['model_id']!r}, which is not a "
            f"registered Re-ID backbone. Registered: {', '.join(known_backbone_ids())}."
        )

    arg_reid_model = DeclareLaunchArgument(
        'reid_model_type', default_value=reid_defaults['reid_model_type'],
        description='Re-ID extractor: osnet_mock | osnet_trt | osnet_torch | transreid | '
                    'transreid_trt (default read from algorithm_thresholds.yaml '
                    '+ optional experiment overlay)')
    arg_reid_ckpt = DeclareLaunchArgument(
        'reid_checkpoint_path', default_value=reid_defaults['reid_checkpoint_path'],
        description='Path to TransReID .pth checkpoint')
    arg_reid_proj = DeclareLaunchArgument(
        'reid_projection_path', default_value=reid_defaults['reid_projection_path'],
        description='Path to SVD projection .npz matrix')
    arg_reid_engine = DeclareLaunchArgument(
        'reid_engine_path', default_value=reid_defaults['reid_engine_path'],
        description='Path to TensorRT .engine file')
    # ES-EXP07 §4.4: traceability id. A string parameter, so a launch substitution is type-safe.
    arg_exp_id = DeclareLaunchArgument(
        'exp_id', default_value=reid_defaults['exp_id'],
        description='Experiment identifier recorded in identity_node startup logs')

    robot_id = LaunchConfiguration('robot_id')
    peer_id = LaunchConfiguration('peer_id')
    is_owner = LaunchConfiguration('is_initial_owner')
    reid_model_type = LaunchConfiguration('reid_model_type')
    reid_checkpoint_path = LaunchConfiguration('reid_checkpoint_path')
    reid_projection_path = LaunchConfiguration('reid_projection_path')
    reid_engine_path = LaunchConfiguration('reid_engine_path')
    exp_id = LaunchConfiguration('exp_id')
    # ES-EXP07 §4.2.8: reid_camera_id is an INTEGER parameter on identity_node and a launch
    # substitution can only produce a string, so the configured value is coerced here and passed
    # as a Python int. Per-run changes go through the YAML, not a `:=` substitution.
    reid_camera_id = coerce_camera_id(reid_defaults['reid_camera_id'])

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
            {
                'robot_id': robot_id,
                'reid_model_type': reid_model_type,
                'reid_checkpoint_path': reid_checkpoint_path,
                'reid_projection_path': reid_projection_path,
                'reid_engine_path': reid_engine_path,
                'exp_id': exp_id,
                'reid_camera_id': reid_camera_id,
            }
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
        arg_reid_model,
        arg_reid_ckpt,
        arg_reid_proj,
        arg_reid_engine,
        arg_exp_id,
        node_perception,
        node_estimation,
        node_identity,
        node_coordination,
        node_target_acquisition,
        node_safety,
        node_monitoring,
    ])
