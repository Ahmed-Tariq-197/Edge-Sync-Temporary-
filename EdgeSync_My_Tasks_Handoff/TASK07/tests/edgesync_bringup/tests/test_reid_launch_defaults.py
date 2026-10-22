"""TR-01 unit tests: Re-ID launch defaults must come from the parameter YAML.

Test class: UNIT (pure Python, no ROS 2 runtime). These prove the helper that
``robot_bringup.launch.py`` now uses returns the configured TransReID selection instead of the
old literal ``osnet_mock``/empty defaults. The launch file itself cannot be executed on this
host (``launch``/``launch_ros`` are not installed), so launch-time precedence is recorded as
BLOCKED — Pending ROS 2 runtime, with the same assertions re-checked by
``tests/test_robot_bringup_launch.py`` on the target.
"""
import os

from edgesync_bringup.reid_launch_defaults import (
    HARDCODED_FALLBACKS, REID_LAUNCH_KEYS, describe_reid_sources,
    read_reid_parameters, resolve_reid_launch_defaults)

REAL_YAML = os.path.join(os.path.dirname(__file__), '..', 'config',
                         'algorithm_thresholds.yaml')


def write_yaml(tmp_path, text, name='params.yaml'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


class TestAgainstTheShippedConfig:
    def test_real_config_selects_transreid(self):
        assert os.path.isfile(REAL_YAML), REAL_YAML
        defaults = resolve_reid_launch_defaults(REAL_YAML)
        assert defaults['reid_model_type'] == 'transreid'

    def test_real_config_supplies_checkpoint_and_projection(self):
        defaults = resolve_reid_launch_defaults(REAL_YAML)
        assert defaults['reid_checkpoint_path'] == 'models/transreid/transformer_120.pth'
        assert defaults['reid_projection_path'].endswith('svd_projection_768_to_512_real_v2.npz')

    def test_old_literal_would_have_been_overwritten(self):
        """The pre-TR-01 launch default was the literal 'osnet_mock'; the YAML disagrees."""
        defaults = resolve_reid_launch_defaults(REAL_YAML)
        assert defaults['reid_model_type'] != HARDCODED_FALLBACKS['reid_model_type']

    def test_shipped_assets_exist_in_the_package_tree(self):
        """The YAML paths are relative to the installed share dir, so check them at the source root."""
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        defaults = resolve_reid_launch_defaults(REAL_YAML)
        for key in ('reid_checkpoint_path', 'reid_projection_path'):
            assert os.path.isfile(os.path.join(pkg_root, defaults[key])), key

    def test_describe_reports_yaml_origin_for_every_configured_key(self):
        summary = describe_reid_sources(REAL_YAML)
        assert 'reid_model_type=\'transreid\'(yaml)' in summary
        assert '(yaml)' in summary and 'reid_engine_path' in summary


class TestFallbacksAndMalformedInput:
    def test_missing_file_returns_previous_literal_defaults(self):
        defaults = resolve_reid_launch_defaults(os.path.join('no', 'such', 'file.yaml'))
        assert defaults == HARDCODED_FALLBACKS

    def test_malformed_yaml_returns_fallbacks(self, tmp_path):
        path = write_yaml(tmp_path, 'reid_model_type: [unclosed')
        assert read_reid_parameters(path) == {}
        assert resolve_reid_launch_defaults(path) == HARDCODED_FALLBACKS

    def test_non_mapping_document_returns_fallbacks(self, tmp_path):
        path = write_yaml(tmp_path, '- just\n- a\n- list\n')
        assert read_reid_parameters(path) == {}

    def test_yaml_without_reid_keys_returns_fallbacks(self, tmp_path):
        path = write_yaml(tmp_path, 'algorithm_thresholds:\n  ros__parameters:\n    trigger_mode: "reactive"\n')
        assert resolve_reid_launch_defaults(path) == HARDCODED_FALLBACKS

    def test_null_values_are_not_stringified(self, tmp_path):
        path = write_yaml(tmp_path,
                          'algorithm_thresholds:\n  ros__parameters:\n    reid_model_type:\n')
        assert read_reid_parameters(path) == {}


class TestParsingShapes:
    def test_nested_namespace_shape(self, tmp_path):
        path = write_yaml(tmp_path,
                          'algorithm_thresholds:\n'
                          '  ros__parameters:\n'
                          '    reid_model_type: "transreid"\n'
                          '    reid_checkpoint_path: "models/transreid/a.pth"\n')
        found = read_reid_parameters(path)
        assert found == {'reid_model_type': 'transreid',
                         'reid_checkpoint_path': 'models/transreid/a.pth'}

    def test_flat_ros_parameters_root(self, tmp_path):
        path = write_yaml(tmp_path,
                          'ros__parameters:\n'
                          '  reid_model_type: "osnet_torch"\n'
                          '  reid_engine_path: " e.trt "\n')
        found = read_reid_parameters(path)
        assert found['reid_model_type'] == 'osnet_torch'
        assert found['reid_engine_path'] == 'e.trt'

    def test_non_string_scalars_become_launch_strings(self, tmp_path):
        path = write_yaml(tmp_path,
                          'algorithm_thresholds:\n'
                          '  ros__parameters:\n'
                          '    reid_model_type: 3\n'
                          '    reid_camera_id: 2\n')
        assert read_reid_parameters(path)['reid_model_type'] == '3'

    def test_only_declared_keys_are_read(self, tmp_path):
        path = write_yaml(tmp_path,
                          'algorithm_thresholds:\n'
                          '  ros__parameters:\n'
                          '    reid_similarity_threshold: 0.70\n'
                          '    reid_model_type: "transreid"\n')
        found = read_reid_parameters(path)
        assert set(found) <= set(REID_LAUNCH_KEYS)
        assert 'reid_similarity_threshold' not in found

    def test_first_matching_section_wins(self, tmp_path):
        path = write_yaml(tmp_path,
                          'a:\n  ros__parameters:\n    reid_model_type: "transreid"\n'
                          'b:\n  ros__parameters:\n    reid_model_type: "osnet_mock"\n')
        assert read_reid_parameters(path)['reid_model_type'] == 'transreid'


class TestExplicitCommandLineStillWins:
    def test_launch_default_is_a_string_ready_for_declare_launch_argument(self):
        """DeclareLaunchArgument(default_value=...) needs a str; the helper guarantees it."""
        defaults = resolve_reid_launch_defaults(REAL_YAML)
        for key in REID_LAUNCH_KEYS:
            assert isinstance(defaults[key], str), key
            # A CLI 'reid_model_type:=osnet_mock' overrides this default at launch time,
            # which is the precedence TR-01 is meant to keep.
            assert defaults[key] == defaults[key].strip()
