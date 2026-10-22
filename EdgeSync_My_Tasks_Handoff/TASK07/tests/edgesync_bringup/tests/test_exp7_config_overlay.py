"""ES-EXP07 §4.2/§4.3/§4.4/§4.5 unit tests for the experiment-configuration overlay.

Test class: UNIT (pure Python, no ROS 2 runtime). They pin the behaviour that
``robot_bringup.launch.py`` relies on: shared config first, experiment overlay second, a
requested-but-broken overlay never treated as loaded, and the resolved model id being the one the
launch dictionary injects. The launch file itself is exercised in
``tests/test_robot_bringup_launch.py`` on a host that has ``launch``/``launch_ros``; that part is
recorded as PENDING — no ROS 2 runtime in this workspace.
"""
import os

import pytest

from edgesync_bringup.reid_launch_defaults import (
    DEFAULT_EXP_ID, EXPERIMENT_CONFIG_ENV_VAR, HARDCODED_FALLBACKS, MODEL_ALIAS_KEY,
    REID_LAUNCH_KEYS, STATIC_BACKBONE_IDS, coerce_camera_id, describe_reid_sources,
    known_backbone_ids, read_reid_parameters, resolve_reid_config,
    resolve_reid_launch_defaults, select_experiment_config)

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SHARED = os.path.join(PKG_ROOT, 'config', 'algorithm_thresholds.yaml')
EXPERIMENT = os.path.join(PKG_ROOT, 'config', 'exp7_transreid.yaml')


def write(tmp_path, text, name='exp.yaml'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


def overlay(**params):
    lines = ['/**:', '  ros__parameters:']
    lines += [f'    {key}: {value}' for key, value in params.items()]
    return '\n'.join(lines) + '\n'


class TestShippedFiles:
    def test_experiment_file_exists_where_bringup_looks_for_it(self):
        assert os.path.isfile(EXPERIMENT), EXPERIMENT

    def test_experiment_file_selects_the_real_transreid_arm(self):
        record = resolve_reid_config(SHARED, EXPERIMENT)
        assert record['model_id'] == 'transreid'
        assert record['model_source'] == 'canonical:experiment'
        assert record['model_is_valid'] is True
        assert record['is_mock'] is False

    def test_experiment_file_identifies_the_experiment(self):
        assert resolve_reid_config(SHARED, EXPERIMENT)['values']['exp_id'] == DEFAULT_EXP_ID

    def test_experiment_file_is_a_valid_ros2_parameter_document(self):
        import yaml

        with open(EXPERIMENT, encoding='utf-8') as handle:
            document = yaml.safe_load(handle)
        assert list(document) == ['/**'], sorted(document)
        params = document['/**']['ros__parameters']
        assert isinstance(params, dict)
        # §4.2.8: written as a YAML integer so the typed value survives to the INTEGER parameter.
        assert isinstance(params['reid_camera_id'], int)

    def test_experiment_file_only_uses_names_the_node_declares(self):
        import yaml

        with open(EXPERIMENT, encoding='utf-8') as handle:
            params = yaml.safe_load(handle)['/**']['ros__parameters']
        declared = {'robot_id', 'similarity_threshold', 'smoothing_alpha', 'enable_spatial_gating',
                    'service_name', 'reid_model_type', 'reid_checkpoint_path',
                    'reid_projection_path', 'reid_engine_path', 'reid_camera_id', 'exp_id',
                    'engine_path', MODEL_ALIAS_KEY}
        assert set(params) <= declared, sorted(set(params) - declared)

    def test_experiment_file_does_not_point_at_a_missing_engine(self):
        record = resolve_reid_config(SHARED, EXPERIMENT)
        assert record['values']['reid_engine_path'] == ''

    def test_reused_assets_exist_in_the_package_tree(self):
        defaults = resolve_reid_launch_defaults(SHARED, EXPERIMENT)
        for key in ('reid_checkpoint_path', 'reid_projection_path'):
            assert os.path.isfile(os.path.join(PKG_ROOT, defaults[key])), key


class TestOverlayPrecedence:
    def test_experiment_overrides_only_the_keys_it_defines(self, tmp_path):
        path = write(tmp_path, overlay(exp_id='"EXP07-armB"'))
        record = resolve_reid_config(SHARED, path)
        assert record['values']['exp_id'] == 'EXP07-armB'
        assert record['origins']['exp_id'] == 'experiment'
        assert record['values']['reid_checkpoint_path'] \
            == resolve_reid_launch_defaults(SHARED)['reid_checkpoint_path']
        assert record['origins']['reid_checkpoint_path'] == 'shared'

    def test_shared_config_alone_still_resolves(self):
        record = resolve_reid_config(SHARED)
        assert record['model_id'] == 'transreid'
        assert record['model_source'] == 'canonical:shared'

    def test_missing_optional_experiment_keeps_shared_behaviour(self):
        assert (resolve_reid_launch_defaults(SHARED)
                == resolve_reid_launch_defaults(SHARED, ''))

    def test_requested_but_missing_experiment_raises_instead_of_falling_back(self, tmp_path):
        absent = os.path.join(str(tmp_path), 'no_such_experiment.yaml')
        with pytest.raises(FileNotFoundError, match='Refusing to continue'):
            resolve_reid_config(SHARED, absent)

    def test_malformed_requested_experiment_raises(self, tmp_path):
        path = write(tmp_path, overlay(reid_model_type='"transreid"') + '  [unclosed\n')
        with pytest.raises(ValueError, match='malformed YAML'):
            resolve_reid_config(SHARED, path)

    def test_unreadable_shared_file_is_reported_not_hidden(self, tmp_path):
        path = write(tmp_path, overlay(reid_model_type='"transreid"'))
        record = resolve_reid_config(os.path.join(str(tmp_path), 'absent_shared.yaml'), path)
        assert record['values']['reid_model_type'] == 'transreid'
        assert list(record['statuses'].values()) == ['absent', 'loaded']


class TestAliasAndValidation:
    def test_alias_is_used_when_the_canonical_key_is_absent(self, tmp_path):
        path = write(tmp_path, overlay(reid_model='"transreid"'))
        record = resolve_reid_config(os.path.join(str(tmp_path), 'no_shared.yaml'), path)
        assert record['model_id'] == 'transreid'
        assert record['model_source'] == 'alias:experiment'
        assert record['values']['reid_model_type'] == 'transreid'

    def test_explicit_canonical_beats_the_alias_in_the_same_file(self, tmp_path):
        path = write(tmp_path, overlay(reid_model_type='"transreid"', reid_model='"osnet_torch"'))
        record = resolve_reid_config(os.path.join(str(tmp_path), 'no_shared.yaml'), path)
        assert record['model_id'] == 'transreid'
        assert record['conflicts'] == [('experiment', 'transreid', 'osnet_torch')]

    def test_agreeing_alias_and_canonical_produce_no_conflict(self, tmp_path):
        path = write(tmp_path, overlay(reid_model_type='"TransReID"', reid_model='"transreid"'))
        record = resolve_reid_config(os.path.join(str(tmp_path), 'no_shared.yaml'), path)
        assert record['conflicts'] == []
        assert record['model_id'] == 'transreid'

    def test_experiment_layer_beats_the_shared_alias(self, tmp_path):
        shared = write(tmp_path, overlay(reid_model='"osnet_torch"'), name='shared.yaml')
        experiment = write(tmp_path, overlay(reid_model_type='"transreid"'), name='exp.yaml')
        assert resolve_reid_config(shared, experiment)['model_source'] == 'canonical:experiment'

    @pytest.mark.parametrize('model_id', list(STATIC_BACKBONE_IDS))
    def test_every_registered_id_resolves_as_valid(self, tmp_path, model_id):
        record = resolve_reid_config(
            os.path.join(str(tmp_path), 'no_shared.yaml'),
            write(tmp_path, overlay(reid_model_type=f'"{model_id}"')))
        assert record['model_is_valid'] is True
        assert record['model_id'] == model_id

    def test_unknown_id_is_reported_invalid_and_described(self, tmp_path):
        record = resolve_reid_config(
            os.path.join(str(tmp_path), 'no_shared.yaml'),
            write(tmp_path, overlay(reid_model_type='"transreidx"')))
        assert record['model_is_valid'] is False
        assert 'valid=False' in describe_reid_config(record)


def describe_reid_config(record):
    """Re-render a record the way describe_reid_sources does, for assertion convenience."""
    return f"model_id={record['model_id']!r} valid={record['model_is_valid']}"


class TestTypingAndCoercion:
    def test_camera_id_from_the_experiment_file_coerces_to_int(self, tmp_path):
        record = resolve_reid_config(
            os.path.join(str(tmp_path), 'no_shared.yaml'), write(tmp_path, overlay(reid_camera_id='4')))
        assert coerce_camera_id(record['values']['reid_camera_id']) == 4

    def test_non_integer_camera_id_is_recorded_as_an_error(self, tmp_path):
        record = resolve_reid_config(
            os.path.join(str(tmp_path), 'no_shared.yaml'),
            write(tmp_path, overlay(reid_camera_id='"two"')))
        assert 'reid_camera_id' in record['errors']
        assert 'must be an integer' in record['errors']['reid_camera_id']

    def test_negative_camera_id_is_rejected(self):
        with pytest.raises(ValueError, match='>= 0'):
            coerce_camera_id(-1)

    def test_camera_id_is_never_left_as_a_launch_string(self):
        # The launch dictionary must be able to carry a real int for this key.
        value = resolve_reid_config(SHARED, EXPERIMENT)['values']['reid_camera_id']
        assert isinstance(coerce_camera_id(value), int)


class TestSilentMockSignals:
    def test_nothing_configured_warns_about_the_mock(self, tmp_path):
        # A file that loads cleanly but declares no Re-ID key: this isolates the §4.5 guard from
        # the separate "layer unreadable" warning.
        shared = write(tmp_path, overlay(similarity_threshold=0.7), name='shared.yaml')
        summary = describe_reid_sources(shared)
        assert "WARNING: no Re-ID model was configured" in summary
        assert "'osnet_mock'" in summary and 'NOT scientific' in summary

    def test_configured_real_model_does_not_warn(self):
        summary = describe_reid_sources(SHARED, EXPERIMENT)
        assert 'WARNING' not in summary, summary
        assert "reid_model_type='transreid'(experiment)" in summary

    def test_explicit_mock_is_reported_as_a_note(self, tmp_path):
        # Both layers load cleanly and neither is unreadable, so any WARNING in the summary would be
        # the §4.5 silent-mock guard rather than a layer-status diagnostic.
        shared = write(tmp_path, overlay(similarity_threshold=0.7, reid_camera_id=2),
                       name='shared.yaml')
        path = write(tmp_path, overlay(reid_model_type='"osnet_mock"'))
        summary = describe_reid_sources(shared, path)
        assert 'WARNING' not in summary, summary
        assert 'NOTE:' in summary and "'osnet_mock' explicitly selected" in summary, summary
        assert "model_id='osnet_mock' source=canonical:experiment" in summary, summary
        assert "no Re-ID model was configured" not in summary, summary

    def test_resolver_labels_the_shared_yaml_as_yaml_for_the_tr01_contract(self):
        assert "reid_model_type='transreid'(yaml)" in describe_reid_sources(SHARED)

    def test_fallbacks_still_match_the_pre_task7_literals(self, tmp_path):
        assert (resolve_reid_launch_defaults(os.path.join(str(tmp_path), 'gone.yaml'))
                == HARDCODED_FALLBACKS)
        assert read_reid_parameters(os.path.join(str(tmp_path), 'gone.yaml')) == {}


class TestRegistryAgreement:
    def test_static_mirror_equals_the_real_registry(self):
        assert known_backbone_ids() == STATIC_BACKBONE_IDS

    def test_fallbacks_cover_every_launch_key(self):
        assert set(HARDCODED_FALLBACKS) == set(REID_LAUNCH_KEYS) | {MODEL_ALIAS_KEY}


class TestOverlaySelector:
    def test_no_override_selects_no_overlay(self):
        assert select_experiment_config(SHARED, '') == ('', 'none')

    def test_absolute_override_is_returned_unchanged(self):
        assert select_experiment_config(SHARED, EXPERIMENT) == (EXPERIMENT, 'override')

    def test_relative_override_is_resolved_against_the_config_directory(self):
        path, origin = select_experiment_config(SHARED, 'exp7_transreid.yaml')
        assert origin == 'override' and os.path.normpath(path) == os.path.normpath(EXPERIMENT)

    def test_relative_override_also_resolves_against_the_package_directory(self):
        path, _origin = select_experiment_config(SHARED,
                                                 os.path.join('config', 'exp7_transreid.yaml'))
        assert os.path.normpath(path) == os.path.normpath(EXPERIMENT)

    def test_unmatched_relative_override_keeps_the_config_form_for_the_error(self, tmp_path):
        shared = write(tmp_path, overlay(reid_model_type='"transreid"'), name='shared.yaml')
        path, origin = select_experiment_config(shared, 'missing.yaml')
        assert origin == 'override'
        assert os.path.normpath(path) == os.path.normpath(os.path.join(str(tmp_path),
                                                                       'missing.yaml'))

    def test_environment_variable_is_the_documented_selector(self, monkeypatch):
        monkeypatch.delenv(EXPERIMENT_CONFIG_ENV_VAR, raising=False)
        assert select_experiment_config(SHARED) == ('', 'none')
        monkeypatch.setenv(EXPERIMENT_CONFIG_ENV_VAR, EXPERIMENT)
        assert select_experiment_config(SHARED) == (EXPERIMENT, 'override')

    def test_explicit_override_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(EXPERIMENT_CONFIG_ENV_VAR, '/somewhere/else.yaml')
        assert select_experiment_config(SHARED, EXPERIMENT)[0] == EXPERIMENT
