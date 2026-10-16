"""Unit tests for the Task 6 identity ablation launch file and arm configs."""
import importlib.util
import os

import pytest

pytest.importorskip("launch_ros")
yaml = pytest.importorskip("yaml")

from launch.actions import DeclareLaunchArgument  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

BRINGUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARM_BACKBONES = ('osnet', 'mobilenetv2')


def _load_launch_module():
    path = os.path.join(BRINGUP_DIR, 'launch', 'task6_identity_ablation.launch.py')
    spec = importlib.util.spec_from_file_location('task6_identity_ablation_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_parameters(backbone):
    path = os.path.join(BRINGUP_DIR, 'config', f'ablation_task6_{backbone}.yaml')
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)['/**']['ros__parameters']


def test_launch_declares_arguments_and_identity_node():
    description = _load_launch_module().generate_launch_description()
    arguments = {a.name for a in description.entities if isinstance(a, DeclareLaunchArgument)}
    assert arguments == {'backbone', 'engine_path', 'robot_id'}
    nodes = [e for e in description.entities if isinstance(e, Node)]
    assert [n.node_executable for n in nodes] == ['identity_node']


def test_arms_differ_only_in_backbone():
    osnet, mobilenet = (_arm_parameters(b) for b in ARM_BACKBONES)
    assert osnet['reid_backbone'] == 'osnet'
    assert mobilenet['reid_backbone'] == 'mobilenetv2'
    rest_osnet = {k: v for k, v in osnet.items() if k != 'reid_backbone'}
    rest_mobilenet = {k: v for k, v in mobilenet.items() if k != 'reid_backbone'}
    assert rest_osnet == rest_mobilenet


def test_arm_configs_use_wildcard_node_key():
    for backbone in ARM_BACKBONES:
        path = os.path.join(BRINGUP_DIR, 'config', f'ablation_task6_{backbone}.yaml')
        with open(path, encoding='utf-8') as handle:
            assert '/**' in yaml.safe_load(handle)
