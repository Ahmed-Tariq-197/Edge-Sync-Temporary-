"""ES-EXP07 §4.3/§4.4/§4.5 unit tests for IdentityNode model selection and traceability.

Test class: UNIT. These reuse the stub-ROS harness of
``test_identity_node_transreid_integration.py`` (that file documents why the stubs are not
integration evidence), so what is asserted here is IdentityNode's own parameter resolution and
logging — never real middleware behaviour.

The extractor is injected in every case except the unknown-id case: this file is about which id
the node resolves and how loudly it reports the fallback, not about loading weights.
"""
import logging

import pytest

import test_identity_node_transreid_integration as harness

IdentityNode = harness.IdentityNode
MODEL_LOG_PREFIX = 'ES-EXP07 §4.5 SILENT-MOCK GUARD'


def build_node(overrides=None, extractor=None):
    return harness.make_node(overrides, extractor or harness._RecordingExtractor(
        harness._unit_vector(0)))


@pytest.fixture
def logs():
    """Capture the stub node's logger records for one construction."""
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(f'{record.levelname}:{record.getMessage()}')

    handler = _Capture()
    logger = logging.getLogger('stub.identity_node')
    previous = logger.handlers
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    try:
        yield captured
    finally:
        logger.handlers = previous


class TestCanonicalAndAliasPrecedence:
    def test_canonical_parameter_wins_when_both_are_supplied(self):
        node = build_node({'reid_model_type': 'transreid', 'reid_model': 'osnet_torch'})
        assert node.reid_model_id == 'transreid'
        assert node._model_id_source == 'explicit:reid_model_type'

    def test_alias_is_used_when_canonical_is_unset(self):
        node = build_node({'reid_model': 'transreid'})
        assert node.reid_model_id == 'transreid'
        assert node._model_id_source == 'explicit:reid_model'

    def test_agreeing_values_do_not_warn(self, logs):
        recorder = harness._RecordingExtractor(harness._unit_vector(0))
        build_node({'reid_model_type': 'transreid', 'reid_model': ' TransReID '}, recorder)
        assert not [line for line in logs if line.startswith('WARNING')]

    def test_conflicting_values_warn_and_stay_deterministic(self, logs):
        node = build_node({'reid_model_type': 'transreid', 'reid_model': 'osnet_torch'})
        warnings = [line for line in logs if line.startswith('WARNING')]
        assert node.reid_model_id == 'transreid'
        assert len(warnings) == 1
        assert 'conflicting model ids' in warnings[0]
        assert 'reid_model_type' in warnings[0] and 'reid_model' in warnings[0]

    @pytest.mark.parametrize('value', ['osnet_mock', 'osnet_trt', 'osnet_torch',
                                       'transreid', 'transreid_trt'])
    def test_every_registered_id_is_accepted(self, value):
        node = build_node({'reid_model_type': value})
        assert node.reid_model_id == value

    def test_unknown_id_is_rejected_before_the_extractor_is_built(self):
        with pytest.raises(ValueError, match='Unknown model_type'):
            build_node({'reid_model_type': 'transreidx'})

    def test_rejection_message_lists_the_registered_choices(self):
        with pytest.raises(ValueError) as excinfo:
            build_node({'reid_model': 'visionreid'})
        for backbone in ('osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt'):
            assert backbone in str(excinfo.value)


class TestSilentMockFallbackGuard:
    def test_nothing_configured_falls_back_to_the_registry_default(self):
        node = build_node({})
        assert node.reid_model_id == 'osnet_mock'
        assert node._model_id_source == 'default:node'

    def test_implicit_fallback_logs_a_warning(self, logs):
        build_node({})
        guard = [line for line in logs if MODEL_LOG_PREFIX in line]
        assert len(guard) == 1 and guard[0].startswith('WARNING')

    def test_warning_refuses_to_call_the_mock_scientific(self, logs):
        build_node({})
        line = next(l for l in logs if MODEL_LOG_PREFIX in l)
        assert 'NOT scientific' in line

    def test_explicit_mock_selection_is_a_note_not_a_warning(self, logs):
        node = build_node({'reid_model_type': 'osnet_mock'})
        assert node.reid_model_id == 'osnet_mock'
        assert not [line for line in logs if MODEL_LOG_PREFIX in line and line.startswith('WARNING')]

    def test_real_model_selection_never_triggers_the_guard(self, logs):
        build_node({'reid_model_type': 'transreid'})
        assert not [line for line in logs if MODEL_LOG_PREFIX in line]


class TestExperimentTraceability:
    def test_exp_id_defaults_to_exp07(self):
        assert build_node({}).exp_id == 'EXP07'

    def test_exp_id_override_is_kept(self):
        assert build_node({'exp_id': 'EXP07-armB'}).exp_id == 'EXP07-armB'

    def test_metadata_line_records_id_model_and_origin(self, logs):
        build_node({'reid_model_type': 'transreid'})
        meta = [line for line in logs if 'EXP-METADATA' in line]
        assert len(meta) == 1
        assert 'exp_id=EXP07' in meta[0]
        assert 'reid_model=transreid' in meta[0]
        assert 'model_source=explicit:reid_model_type' in meta[0]

    def test_metadata_line_survives_the_fallback_case(self, logs):
        build_node({})
        meta = next(line for line in logs if 'EXP-METADATA' in line)
        assert 'reid_model=osnet_mock' in meta and 'model_source=default:node' in meta

    def test_camera_id_is_still_read_as_an_integer(self):
        node = build_node({'reid_camera_id': 2})
        assert node.reid_camera_id == 2 and isinstance(node.reid_camera_id, int)


class TestFrozenContractUntouched:
    def test_service_and_verification_path_still_work(self):
        reference = harness._unit_vector(0)
        node = build_node({}, harness._RecordingExtractor(reference))
        response = harness.run_verify(node, harness._StubRequest(
            'txn-1', reference=reference,
            crop=harness._StubImage(bytes([255, 0, 0, 0, 255, 0]), 1, 2, 'rgb8')))
        assert response.match is True
        assert pytest.approx(response.similarity, abs=1e-6) == 1.0

    def test_node_declares_the_new_parameters(self):
        node = build_node({})
        # declare_parameter() is the only place ROS 2 accepts an override from a parameter file.
        assert node._params['reid_model'].get_parameter_value().string_value == ''
        assert node._params['exp_id'].get_parameter_value().string_value == 'EXP07'
        assert node._params['reid_model_type'].get_parameter_value().string_value == ''
