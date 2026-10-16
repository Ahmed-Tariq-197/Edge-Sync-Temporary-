"""Launch the identity node for a Task 6 ablation arm (OSNet baseline or MobileNetV2)."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

BACKBONE_ARGUMENT = 'backbone'
DEFAULT_BACKBONE = 'osnet'
ENGINE_ARGUMENT = 'engine_path'
ROBOT_ARGUMENT = 'robot_id'
DEFAULT_ROBOT_ID = 'ugv_1'
CONFIG_PREFIX = 'ablation_task6_'
CONFIG_SUFFIX = '.yaml'


def generate_launch_description():
    pkg_bringup = get_package_share_directory('edgesync_bringup')

    arg_backbone = DeclareLaunchArgument(
        BACKBONE_ARGUMENT, default_value=DEFAULT_BACKBONE,
        description='Re-ID backbone arm: osnet | mobilenetv2')
    arg_engine = DeclareLaunchArgument(
        ENGINE_ARGUMENT, default_value='',
        description='Absolute path to the serialized TensorRT engine for the selected backbone')
    arg_robot = DeclareLaunchArgument(
        ROBOT_ARGUMENT, default_value=DEFAULT_ROBOT_ID, description='Robot namespace/ID')

    arm_config = PathJoinSubstitution([
        pkg_bringup, 'config',
        [CONFIG_PREFIX, LaunchConfiguration(BACKBONE_ARGUMENT), CONFIG_SUFFIX],
    ])

    node_identity = Node(
        package='edgesync_identity',
        executable='identity_node',
        name='identity_node',
        namespace=[LaunchConfiguration(ROBOT_ARGUMENT), '/edgesync'],
        parameters=[
            arm_config,
            {'robot_id': ParameterValue(LaunchConfiguration(ROBOT_ARGUMENT), value_type=str),
             'engine_path': ParameterValue(LaunchConfiguration(ENGINE_ARGUMENT), value_type=str)},
        ],
        output='screen',
    )

    return LaunchDescription([arg_backbone, arg_engine, arg_robot, node_identity])
