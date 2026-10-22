"""Experiment 13 bringup — the documented runtime selection path for the CBAA (L = 1) arm.

WHY THIS FILE EXISTS
====================
``robot_bringup.launch.py`` is the deployed stack and stays byte-identical: it launches the baseline
``coordination_node`` executable and passes four YAML files to it, none of which is
``config/exp13_cbba.yaml``. Design decision D6 requires a selector that reaches a running node, and
the smallest way to do that without editing a shared launch file is a dedicated launcher for the
experiment. This file is that launcher. It selects the *executable*, so a ``cnp`` run and a ``cbba``
run differ by exactly one line of launch configuration plus one parameter.

WHAT IT LAUNCHES, AND WHAT IT DELIBERATELY DOES NOT
===================================================
Per robot: ``exp13_coordination_node`` and ``monitoring_node``; ``identity_node`` only when
``with_identity:=true``.

``monitoring_node`` is not decoration. ``edgesync_coordination`` bids on the battery value in
``_my_telemetry``, and the Exp-13 node refuses to bid at all until it has seen a ``StateHeartbeat``
carrying *its own* ``sender_id`` (``exp13_coordination_node.py`` :meth:`_live_battery_pct`). That
heartbeat is published by ``monitoring_node`` (``edgesync_observability/nodes/monitoring_node.py:75``,
1 Hz by default). Launch this file without it and every episode ends ``NO_WINNER`` — which is the
honest outcome, not a bug, but not an experiment either.

Not launched: ``perception_node``, ``estimation_node``, ``target_acquisition_node``,
``safety_authority_node``; ``identity_node`` is not launched unless ``with_identity:=true``.
Consequences that must be carried into any claim made from a run of this file:
  * No ``estimation_node`` means no autonomous ``TTOEstimate`` stream, so the handoff trigger has to be
    injected (see TRIGGERING AN EPISODE). Nothing here measures the full stack.
  * No ``target_acquisition_node`` means no ``AcqStatus``, so the anti-deadlock watchdog is never fed
    after it is armed: a committed owner times out at 300 ms and reclaims at e+2. That is the watchdog
    doing its job, and it is recorded as such rather than as a transfer.
  * With ``with_identity:=false`` (the default) nothing produces a ``ReidSignature``, so the trigger's
    ``reid_feature_vector`` stays the zeros the baseline initialises it with. That is the defect the
    ES-EXP07 <-> ES-EXP13 contract exists to close, and it is not closed by a run of this file that
    leaves the switch off.
  * With ``with_identity:=true`` the owner's identity node publishes the smoothed signature the arm
    installs, but no node in this repository publishes a target *crop* (``perception_node`` keeps its
    bounding box in-process and publishes only a 3D centroid), so ``reid_target_crop_topic`` is left
    empty here: the signature leg is live, the observation leg still needs an image source. Point
    ``reid_target_crop_topic`` at any ``sensor_msgs/Image`` topic carrying the tracked crop and the
    chain completes without changing this file again.

TRIGGERING AN EPISODE
=====================
``CoordinationNode._on_tto_estimate`` starts a handoff when ``tto_safe_s`` falls below
``handoff_trigger_threshold_s`` (2.5 s) and the node is OWNER in TRACKING. With no estimation node
running, publish the trigger yourself on the owner's own topic (it is *not* in the swarm remap list):

    ros2 topic pub -1 /ugv_1/edgesync/tto/estimate edgesync_msgs/msg/TTOEstimate \\
      "{tto_safe_s: 2.0, sigma_tto_s: 0.05, frame_id: map, predicted_trajectory: []}"

RUNNING THE TWO ARMS
====================
Two terminals on one host (Tier-U), or one per Jetson (Tier-R), then the trigger above:

    ros2 launch edgesync_bringup exp13_bringup.launch.py robot_id:=ugv_1 peer_id:=ugv_2 \\
        is_initial_owner:=true  negotiation_protocol:=cbba
    ros2 launch edgesync_bringup exp13_bringup.launch.py robot_id:=ugv_2 peer_id:=ugv_1 \\
        is_initial_owner:=false negotiation_protocol:=cbba

``negotiation_protocol:=cnp`` on both sides runs the control arm in the *same* node: no belief
publisher, no round timer, no adapter, and the inherited one-round selection. An unrecognised value
makes the node log a fatal line and raise during construction — it does not fall back to ``cnp``.

STATUS OF THIS FILE
===================
Written and read back; ``launch``/``launch_ros`` are not installed on this host, so it has never been
executed. Everything above is a claim about code paths, not about a graph. See
``EXP13_TEST_AND_VALIDATION_REPORT_*.md`` for which layers were actually tested.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The baseline's swarm list (robot_bringup.launch.py:84-92) plus the Exp-13 belief topic. Both robots
# publish beliefs on /<peer>/edgesync/consensus_belief and subscribe on /<self>/..., which connect
# either way; remapping them onto one shared topic as well is what makes `ros2 topic echo` show a
# single consensus stream for the run.
SWARM_TOPICS = [
    'handoff_request',
    'handoff_response',
    'handoff_confirm',
    'handoff_confirm_ack',
    'handoff_cancel',
    'ownership_event',
    'state_heartbeat',
    'consensus_belief',
]


def generate_launch_description():
    pkg_bringup = get_package_share_directory('edgesync_bringup')

    coordination_params = os.path.join(pkg_bringup, 'config', 'coordination_params.yaml')
    timing_params = os.path.join(pkg_bringup, 'config', 'timing_budgets.yaml')
    algo_params = os.path.join(pkg_bringup, 'config', 'algorithm_thresholds.yaml')
    hw_params = os.path.join(pkg_bringup, 'config', 'hw_params.yaml')
    exp13_params = os.path.join(pkg_bringup, 'config', 'exp13_cbba.yaml')
    exp7_params = os.path.join(pkg_bringup, 'config', 'exp7_transreid.yaml')

    arg_robot_id = DeclareLaunchArgument(
        'robot_id', default_value='ugv_1', description='This UGV id (topic namespace and agent id)')
    arg_peer_id = DeclareLaunchArgument(
        'peer_id', default_value='ugv_2',
        description='Peer UGV id. Beliefs are published to its consensus_belief topic.')
    arg_is_owner = DeclareLaunchArgument(
        'is_initial_owner', default_value='true', description='Does this UGV start as target owner?')
    # The selector lives here as a launch argument, which is what makes it a runtime selection path:
    # it is passed into the node's declared `negotiation_protocol` parameter by name. The default is
    # `cbba` because this file is the Exp-13 launcher; the node's own parameter default is `cnp`, so
    # a node started any other way still runs the baseline.
    arg_protocol = DeclareLaunchArgument(
        'negotiation_protocol', default_value='cbba',
        description="Allocator: 'cnp' (inherited one-round baseline, the control arm) or 'cbba' "
                    "(CBAA at bundle length L = 1). Anything else makes the node refuse to start.")
    arg_exp_id = DeclareLaunchArgument(
        'exp_id', default_value='EXP13',
        description='Traceability id recorded in the node startup log')
    # ES-EXP07 <-> ES-EXP13: the reference-embedding leg. Off by default so that a run of this file
    # stays comparable with every run made before the signature existed; on, it gives the owner an
    # identity_node that publishes ReidSignature and hands the arm a real embedding to put on the
    # trigger. It costs a model load, so it is a decision, not a default.
    arg_with_identity = DeclareLaunchArgument(
        'with_identity', default_value='false',
        description="Launch this robot's identity_node and let it publish ReidSignature "
                    "(requires the Exp-7 checkpoint and projection assets in package share)")

    robot_id = LaunchConfiguration('robot_id')
    peer_id = LaunchConfiguration('peer_id')
    is_owner = LaunchConfiguration('is_initial_owner')
    protocol = LaunchConfiguration('negotiation_protocol')
    exp_id = LaunchConfiguration('exp_id')
    with_identity = LaunchConfiguration('with_identity')

    # Per-robot, absolute, self-addressed — the same shape as the belief topic. A shared swarm topic
    # here would put a peer's embedding on this robot's input; the arm refuses a foreign sender_id,
    # but a name that cannot cross is better than a guard that has to catch the crossing.
    signature_topic = [robot_id, '/edgesync/identity/reid_signature']

    node_namespace = [robot_id, '/edgesync']

    remappings = [(['/', robot_id, f'/edgesync/{topic}'], f'/edgesync_swarm/{topic}')
                  for topic in SWARM_TOPICS]
    remappings += [(['/', peer_id, f'/edgesync/{topic}'], f'/edgesync_swarm/{topic}')
                   for topic in SWARM_TOPICS]

    node_coordination = Node(
        package='edgesync_coordination',
        executable='exp13_coordination_node',
        name='exp13_coordination_node',
        namespace=node_namespace,
        remappings=remappings,
        parameters=[
            coordination_params, timing_params, algo_params, hw_params, exp13_params,
            {
                'robot_id': robot_id,
                'peer_id': peer_id,
                'is_initial_owner': is_owner,
                'negotiation_protocol': protocol,
                'exp_id': exp_id,
                'reid_signature_topic': signature_topic,
            },
        ],
        output='screen',
    )

    node_monitoring = Node(
        package='edgesync_observability',
        executable='monitoring_node',
        name='monitoring_node',
        namespace=node_namespace,
        remappings=remappings,
        parameters=[
            timing_params, hw_params,
            # monitoring_node declares robot_id/peer_id only; no exp_id is passed because it does not
            # declare one, and a key a node does not declare is silently dropped rather than logged.
            {'robot_id': robot_id, 'peer_id': peer_id},
        ],
        output='screen',
    )

    node_identity = Node(
        package='edgesync_identity',
        executable='identity_node',
        name='identity_node',
        namespace=node_namespace,
        remappings=remappings,
        parameters=[
            # exp7_transreid.yaml is the Exp-7 overlay: it selects reid_model_type=transreid and the
            # approved checkpoint/projection, so a signature produced under this launch is a TransReID
            # signature and not a mock. Passed as a parameter file because identity_node declares these
            # keys; robot_bringup.launch.py resolves the same file through its own resolver instead.
            exp7_params,
            {
                'robot_id': robot_id,
                'exp_id': exp_id,
                'publish_reid_signature': True,
                'reid_signature_topic': signature_topic,
                # Empty on purpose. No node in this repository publishes a target crop:
                # perception_node computes a bounding box, deprojects it and publishes only the 3D
                # centroid, so there is nothing to subscribe to. With this unset the signature leg is
                # live but starved, and identity_node says so in a WARN at startup.
                'reid_target_crop_topic': '',
            },
        ],
        output='screen',
        condition=IfCondition(with_identity),
    )

    return LaunchDescription([
        arg_robot_id,
        arg_peer_id,
        arg_is_owner,
        arg_protocol,
        arg_exp_id,
        arg_with_identity,
        node_coordination,
        node_monitoring,
        node_identity,
    ])
