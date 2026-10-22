"""Unit tests for Sprint 6 Unified Bringup Launch (edgesync_bringup).

Verifies:
1. Launch Argument Declarations:
   - robot_id, peer_id, is_initial_owner.
2. Instantiation of ALL 5 Core Nodes:
   - perception_node (edgesync_perception)
   - estimation_node (edgesync_estimation)
   - coordination_node (edgesync_coordination)
   - identity_node (edgesync_identity)
   - target_acquisition_node (edgesync_acquisition)
3. Dynamic Namespacing:
   - Every node is dynamically assigned namespace=[LaunchConfiguration('robot_id'), '/edgesync'].
4. YAML Parameter Injection:
   - coordination_params.yaml, hw_params.yaml, timing_budgets.yaml, algorithm_thresholds.yaml
     are correctly injected into the node parameter lists.
"""
import os
import importlib.util
import pytest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_launch_module():
    launch_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'launch', 'robot_bringup.launch.py')
    )
    spec = importlib.util.spec_from_file_location("robot_bringup_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launch_desc() -> LaunchDescription:
    module = _load_launch_module()
    return module.generate_launch_description()


def test_robot_bringup_launch_arguments(launch_desc):
    """Verify launch argument declarations and defaults."""
    args = [action for action in launch_desc.entities if isinstance(action, DeclareLaunchArgument)]
    arg_names = {arg.name: arg for arg in args}

    assert 'robot_id' in arg_names
    assert 'peer_id' in arg_names
    assert 'is_initial_owner' in arg_names

    assert arg_names['robot_id'].default_value[0].text == 'ugv_1'
    assert arg_names['peer_id'].default_value[0].text == 'ugv_2'
    assert arg_names['is_initial_owner'].default_value[0].text == 'true'


def test_robot_bringup_instantiates_all_core_nodes(launch_desc):
    """Verify that perception, estimation, coordination, identity, and target_acquisition are present."""
    nodes = [action for action in launch_desc.entities if isinstance(action, Node)]
    node_map = {n.node_executable: n for n in nodes}

    # Verify all 7 system nodes
    required_executables = [
        ('perception_node', 'edgesync_perception'),
        ('estimation_node', 'edgesync_estimation'),
        ('coordination_node', 'edgesync_coordination'),
        ('identity_node', 'edgesync_identity'),
        ('target_acquisition_node', 'edgesync_acquisition'),
        ('safety_authority_node', 'edgesync_navigation'),
        ('monitoring_node', 'edgesync_observability'),
    ]

    for exec_name, pkg_name in required_executables:
        assert exec_name in node_map, f"Missing required node executable: {exec_name}"
        assert node_map[exec_name].node_package == pkg_name, f"Package mismatch for {exec_name}"


def test_robot_bringup_dynamic_namespacing(launch_desc):
    """Verify every node is namespaced under [LaunchConfiguration('robot_id'), '/edgesync']."""
    nodes = [action for action in launch_desc.entities if isinstance(action, Node)]

    for node in nodes:
        ns = node._Node__node_namespace
        assert ns is not None, f"Node {node.node_executable} has no namespace"
        assert isinstance(ns, list), f"Node {node.node_executable} namespace is not a list substitution"

        # Check list elements: first element is LaunchConfiguration('robot_id'), second is '/edgesync'
        first_sub = ns[0]
        assert isinstance(first_sub, LaunchConfiguration), f"First namespace substitution not LaunchConfiguration in {node.node_executable}"
        assert first_sub.variable_name[0].text == 'robot_id'
        assert ns[1] == '/edgesync'


def test_robot_bringup_parameter_injection(launch_desc):
    """Verify that YAML parameter files and configuration dictionaries are injected."""
    nodes = [action for action in launch_desc.entities if isinstance(action, Node)]
    node_map = {n.node_executable: n for n in nodes}

    # Coordination node must receive coordination_params, timing_params, algo_params, hw_params, and dict
    coord_params = node_map['coordination_node']._Node__parameters
    assert len(coord_params) >= 5

    param_reprs = []
    for p in coord_params:
        if hasattr(p, 'param_file'):
            for s in p.param_file:
                param_reprs.append(getattr(s, 'text', str(s)))
        else:
            param_reprs.append(str(p))
    param_str = ' '.join(param_reprs)

    assert 'coordination_params.yaml' in param_str
    assert 'timing_budgets.yaml' in param_str
    assert 'algorithm_thresholds.yaml' in param_str
    assert 'hw_params.yaml' in param_str or 'hardware_params.yaml' in param_str

    # Target acquisition node must receive timing_params, algo_params, hw_params
    acq_params = node_map['target_acquisition_node']._Node__parameters
    acq_reprs = []
    for p in acq_params:
        if hasattr(p, 'param_file'):
            for s in p.param_file:
                acq_reprs.append(getattr(s, 'text', str(s)))
        else:
            acq_reprs.append(str(p))
    acq_str = ' '.join(acq_reprs)

    assert 'timing_budgets.yaml' in acq_str
    assert 'hw_params.yaml' in acq_str or 'hardware_params.yaml' in acq_str


def test_robot_bringup_global_remappings(launch_desc):
    """Verify that global swarm remappings are applied to communication nodes but not hardware/local nodes."""
    nodes = [action for action in launch_desc.entities if isinstance(action, Node)]
    node_map = {n.node_executable: n for n in nodes}

    comm_nodes = [
        'coordination_node',
        'target_acquisition_node',
        'safety_authority_node',
        'monitoring_node',
    ]

    local_nodes = [
        'perception_node',
        'estimation_node',
        'identity_node',
    ]

    required_swarm_topics = [
        'handoff_request',
        'handoff_response',
        'handoff_confirm',
        'handoff_confirm_ack',
        'handoff_cancel',
        'ownership_event',
        'state_heartbeat',
    ]

    for node_name in comm_nodes:
        node = node_map[node_name]
        remappings = node._Node__remappings
        assert remappings is not None and len(remappings) > 0, f"{node_name} has no remappings"

        # Extract string representations of (from_topic, to_topic)
        parsed_rules = []
        for src_tuple, dst_tuple in remappings:
            src_str = ''.join(getattr(s, 'text', str(s)) for s in src_tuple)
            dst_str = ''.join(getattr(s, 'text', str(s)) for s in dst_tuple)
            parsed_rules.append((src_str, dst_str))

        # Check that all 7 required swarm topics are mapped to /edgesync_swarm/<topic>
        for topic in required_swarm_topics:
            target_dest = f'/edgesync_swarm/{topic}'
            matches = [dst for src, dst in parsed_rules if src == topic and dst == target_dest]
            assert len(matches) > 0, f"{node_name} missing global remapping for {topic} -> {target_dest}"

    for node_name in local_nodes:
        node = node_map[node_name]
        remappings = node._Node__remappings
        assert remappings is None or len(remappings) == 0, f"Local node {node_name} should not have remappings"

