"""ES-EXP07 §4.1 — focused tests for the registry-backed extractor factory.

Test class: UNIT. No TensorRT, no engine, no Jetson and no ROS 2 runtime exist on the host where
these were written, so nothing here is real inference or integration evidence. What they pin is
the plugin contract required by the Team Execution Guide rule #0: identifiers are declared
constants, the registry is the single source of truth, dispatch reaches the implementation the
registry declares, unknown identifiers are rejected before construction, and an unavailable
optional dependency never degrades into the ``osnet_mock`` pseudo-embedding.

Real TransReID construction (which loads the 414 MB checkpoint) runs only under ES07_HEAVY, the
same gate the rest of the Task 7 suite uses.
"""
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from edgesync_identity.interfaces.base_identity_extractor import BaseIdentityExtractor
from edgesync_identity.models import extractor_factory as F

BRINGUP_MODELS = os.path.abspath(os.path.join(
    pkg_dir, '..', 'edgesync_bringup', 'models', 'transreid'))
REAL_CKPT = os.path.join(BRINGUP_MODELS, 'transformer_120.pth')
REAL_PROJ = os.path.join(BRINGUP_MODELS, 'svd_projection_768_to_512_real_v2.npz')

EXPECTED_IDS = ('osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt')


def _stub_class_init(monkeypatch, module_name, class_name):
    """Replace a heavy extractor __init__ with a recorder; returns the class and its calls."""
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    def fake_init(self, *args, **kwargs):
        self._stub_args = args
        self._stub_kwargs = kwargs

    monkeypatch.setattr(klass, '__init__', fake_init)
    return klass


def _real_proj(tmp_path):
    proj = tmp_path / 'proj.npz'
    np.savez(proj, projection=np.eye(768, 512, dtype=np.float32),
             mean=np.zeros(768, dtype=np.float32))
    return str(proj)


class TestIdentifiersAreDeclaredConstants:
    def test_constants_match_the_registry_exactly(self):
        """Rule #0: the declared constants ARE the registry keys — no loose literals."""
        constants = {
            F.BACKBONE_ID_OSNET_MOCK, F.BACKBONE_ID_OSNET_TRT, F.BACKBONE_ID_OSNET_TORCH,
            F.BACKBONE_ID_TRANSREID, F.BACKBONE_ID_TRANSREID_TRT,
        }
        assert constants == set(F.BACKBONE_REGISTRY) == set(EXPECTED_IDS)

    def test_registry_declaration_order_is_deterministic(self):
        assert F.registered_backbone_ids() == EXPECTED_IDS
        assert tuple(F.get_registered_backbones()) == EXPECTED_IDS

    def test_default_identifier_is_the_documented_mock(self):
        assert F.DEFAULT_BACKBONE_ID == F.BACKBONE_ID_OSNET_MOCK
        assert F.MOCK_BACKBONE_IDS == (F.BACKBONE_ID_OSNET_MOCK,)

    def test_no_builder_body_hides_an_unregistered_identifier(self):
        """Every implementation string must import and be the class the id is expected to yield."""
        for backbone_id, spec in F.BACKBONE_REGISTRY.items():
            resolved = F._resolve_registered_class(spec)
            assert issubclass(resolved, BaseIdentityExtractor), backbone_id
            assert spec.implementation.endswith('.' + resolved.__name__), backbone_id


class TestIntrospection:
    def test_get_registered_backbones_maps_id_to_implementation(self):
        registered = F.get_registered_backbones()
        assert registered['transreid'].endswith('.TransReIdExtractor')
        assert registered['transreid_trt'].endswith('.TransReIdTrtExtractor')
        assert registered['osnet_mock'].endswith('.OsnetTrtExtractor')

    def test_returned_mapping_is_a_copy_and_cannot_mutate_the_registry(self):
        registered = F.get_registered_backbones()
        registered['invented'] = 'x.Y'
        assert 'invented' not in F.get_registered_backbones()

    def test_is_registered_and_normalize_agree(self):
        assert F.is_registered_backbone('  TransReID ') is True
        assert F.is_registered_backbone('transreid_trt') is True
        assert F.is_registered_backbone('cbba') is False
        assert F.normalize_backbone_id(' OSNet_Mock ') == 'osnet_mock'


class TestDispatchReachesTheRegisteredImplementation:
    def test_osnet_mock_preserves_the_ci_safe_construction(self, monkeypatch):
        klass = _stub_class_init(monkeypatch, 'edgesync_identity.models.osnet_trt_extractor',
                                 'OsnetTrtExtractor')
        made = F.create_extractor('osnet_mock')
        assert type(made) is klass
        assert made._stub_kwargs.get('engine_path') is None

    def test_osnet_trt_passes_the_engine_path(self, monkeypatch, tmp_path):
        klass = _stub_class_init(monkeypatch, 'edgesync_identity.models.osnet_trt_extractor',
                                 'OsnetTrtExtractor')
        engine = tmp_path / 'orin.engine'
        engine.write_bytes(b'\x00' * 16)
        made = F.create_extractor('osnet_trt', engine_path=str(engine))
        assert type(made) is klass
        assert made._stub_kwargs['engine_path'] == str(engine)

    def test_osnet_torch_reaches_the_torch_arm(self, monkeypatch, tmp_path):
        klass = _stub_class_init(monkeypatch, 'edgesync_identity.models.osnet_torch_extractor',
                                 'OsnetTorchExtractor')
        ckpt = tmp_path / 'osnet.pth'
        ckpt.write_bytes(b'\x00' * 8)
        made = F.create_extractor('osnet_torch', checkpoint_path=str(ckpt))
        assert type(made) is klass

    def test_transreid_and_transreid_trt_are_distinct_implementations(self, monkeypatch, tmp_path):
        torch_arm = _stub_class_init(monkeypatch, 'edgesync_identity.models.transreid_extractor',
                                     'TransReIdExtractor')
        trt_arm = _stub_class_init(monkeypatch, 'edgesync_identity.models.transreid_trt_extractor',
                                   'TransReIdTrtExtractor')
        ckpt = tmp_path / 'transformer_120.pth'
        ckpt.write_bytes(b'\x00' * 8)
        engine = tmp_path / 'orin.engine'
        engine.write_bytes(b'\x00' * 8)
        proj = _real_proj(tmp_path)

        a = F.create_extractor('transreid', checkpoint_path=str(ckpt), projection_path=proj)
        b = F.create_extractor('transreid_trt', engine_path=str(engine), projection_path=proj)
        assert type(a) is torch_arm and type(b) is trt_arm
        assert type(a) is not type(b)

    def test_case_and_whitespace_are_normalized_before_lookup(self, monkeypatch):
        klass = _stub_class_init(monkeypatch, 'edgesync_identity.models.osnet_trt_extractor',
                                 'OsnetTrtExtractor')
        assert type(F.create_extractor(' OSNET_MOCK ')) is klass


class TestUnknownIdentifierRejectedBeforeConstruction:
    def test_message_names_the_value_and_every_registered_choice(self):
        with pytest.raises(ValueError, match='Unknown model_type'):
            F.create_extractor('transreid_trt_v9')
        try:
            F.create_extractor('invented_backbone')
        except ValueError as exc:
            text = str(exc)
            for backbone_id in EXPECTED_IDS:
                assert backbone_id in text
            assert 'invented_backbone' in text
        else:
            pytest.fail('unknown identifier did not raise')

    def test_empty_and_none_are_rejected_not_defaulted(self):
        for bad in ('', '   ', None):
            with pytest.raises(ValueError):
                F.create_extractor(bad)

    def test_construction_never_started_for_an_unknown_id(self, monkeypatch):
        calls = []
        real_init = F.OsnetTrtExtractor.__init__

        def spy(self, *a, **k):
            calls.append((a, k))
            return real_init(self, *a, **k)

        monkeypatch.setattr(F.OsnetTrtExtractor, '__init__', spy)
        with pytest.raises(ValueError):
            F.create_extractor('osnet_mockk')
        assert calls == []


class TestOptionalDependencyNeverDegradesToMock:
    def test_importerror_from_a_builder_becomes_runtimeerror(self, monkeypatch):
        def boom(**_kwargs):
            raise ImportError('No module named tensorrt')

        spec = F.BACKBONE_REGISTRY[F.BACKBONE_ID_TRANSREID_TRT]
        monkeypatch.setitem(F.BACKBONE_REGISTRY, F.BACKBONE_ID_TRANSREID_TRT,
                            F.BackboneSpec(spec.backbone_id, spec.implementation,
                                           spec.description, boom))
        with pytest.raises(RuntimeError, match=r"will NOT fall back to 'osnet_mock'"):
            F.create_extractor('transreid_trt', engine_path='x.engine', projection_path='p.npz')

    def test_real_transreid_trt_on_a_host_without_tensorrt_raises_not_mocks(self):
        """Off-target this host has no engine: the arm must fail, never return a pseudo-embedding."""
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            F.create_extractor('transreid_trt', engine_path='/no/such.engine',
                               projection_path='/no/such.npz')
        assert not importlib.util.find_spec('tensorrt'), 'host unexpectedly has TensorRT'

    def test_factory_import_needs_no_optional_dependency(self):
        """A minimal interpreter must import the factory even without tensorrt/pycuda present."""
        code = ('import sys;'
                f'sys.path.insert(0, {pkg_dir!r});'
                'import edgesync_identity.models.extractor_factory as f;'
                'print(len(f.registered_backbone_ids()))')
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0, done.stderr[-400:]
        assert done.stdout.strip().endswith('5')
        assert 'tensorrt' not in done.stdout


class TestVerificationIsNotTautological:
    def test_wrong_concrete_class_is_rejected(self, monkeypatch):
        """A builder returning an object outside the contract must fail the registry check."""
        entry = F.BACKBONE_REGISTRY[F.BACKBONE_ID_OSNET_MOCK]

        def returns_a_plain_object(_engine_path=''):
            return object()

        monkeypatch.setitem(F.BACKBONE_REGISTRY, F.BACKBONE_ID_OSNET_MOCK,
                            F.BackboneSpec(entry.backbone_id, entry.implementation,
                                           entry.description, returns_a_plain_object))
        with pytest.raises(TypeError, match='does not implement BaseIdentityExtractor'):
            F.create_extractor('osnet_mock')

    def test_subclass_of_the_same_contract_is_rejected(self, monkeypatch):
        """Type identity is required, so a look-alike subclass cannot impersonate an arm."""
        class Impostor(F.OsnetTrtExtractor):
            def __init__(self, *args, **kwargs):
                pass

        def build(_engine_path=''):
            return Impostor()

        entry = F.BACKBONE_REGISTRY[F.BACKBONE_ID_OSNET_MOCK]
        monkeypatch.setitem(F.BACKBONE_REGISTRY, F.BACKBONE_ID_OSNET_MOCK,
                            F.BackboneSpec(entry.backbone_id, entry.implementation,
                                           entry.description, build))
        with pytest.raises(TypeError, match='declared'):
            F.create_extractor('osnet_mock')

    def test_builder_argument_names_are_derived_not_hardcoded(self):
        assert F._BUILDER_ARGS[F.BACKBONE_ID_OSNET_MOCK] == ()
        assert F._BUILDER_ARGS[F.BACKBONE_ID_OSNET_TRT] == ('engine_path',)
        assert F._BUILDER_ARGS[F.BACKBONE_ID_TRANSREID] == ('checkpoint_path', 'projection_path')


class TestFrozenContractPreserved:
    def test_create_extractor_signature_is_unchanged(self):
        params = list(inspect.signature(F.create_extractor).parameters)
        assert params == ['model_type', 'checkpoint_path', 'projection_path', 'engine_path']

    @pytest.mark.skipif(not (os.path.isfile(REAL_CKPT) and os.path.isfile(REAL_PROJ)),
                        reason='shipped TransReID assets absent')
    @pytest.mark.skipif(not os.environ.get('ES07_HEAVY'),
                        reason='loads the 414 MB checkpoint; set ES07_HEAVY=1')
    def test_registered_transreid_arm_performs_real_inference(self):
        made = F.create_extractor('transreid', checkpoint_path=REAL_CKPT,
                                  projection_path=REAL_PROJ)
        crop = (np.random.default_rng(7).random((128, 256, 3)) * 255).astype('uint8')
        vec = np.asarray(made.extract_embedding(crop))
        assert vec.shape == (512,)
        assert vec.dtype == np.float32
        assert np.isfinite(vec).all()
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
