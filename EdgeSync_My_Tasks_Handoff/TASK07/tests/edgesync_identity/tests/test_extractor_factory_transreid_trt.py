"""TR-10 wiring — focused regression tests for the ``transreid_trt`` extractor-factory branch.

Charter §6 step 9 requires the *production* path to be able to load the intended engine together
with the intended preprocessing and projection. The production construction point is
``identity_node.py`` → ``create_extractor(model_type=…, engine_path=…, projection_path=…)``, so
without a ``transreid_trt`` arm in the factory the TR-10 extractor is unreachable from the node no
matter how correct the class is. These tests are UNIT tests (charter §9): no TensorRT, no engine and
no Jetson exist on this host, so nothing here is real inference; what they pin is the dispatch, the
asset plumbing and — per charter §3C — that adding the branch did not change the frozen OSNet arms
or the Torch TransReID arm.
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from edgesync_identity.models.extractor_factory import create_extractor  # noqa: E402
from edgesync_identity.models.osnet_trt_extractor import OsnetTrtExtractor  # noqa: E402

ENGINE = '/tmp/does_not_exist_orin.engine'


def _assets(tmp_path):
    """A real dummy engine file and a real (768,512) projection, both on disk."""
    engine = tmp_path / 'orin.engine'
    engine.write_bytes(b'\x00' * 16)
    proj = tmp_path / 'proj.npz'
    np.savez(proj, projection=np.eye(768, 512, dtype=np.float32),
             mean=np.zeros(768, dtype=np.float32))
    return str(engine), str(proj)


class TestTransreidTrtDispatch:
    def test_reaches_the_trt_extractor_class(self, monkeypatch):
        """The branch must construct TransReIdTrtExtractor, not the Torch arm."""
        from edgesync_identity.models import transreid_trt_extractor as trt_module

        calls = {}

        def fake_init(self, engine_path, projection_path, input_shape=(256, 128),
                      use_unified_memory=True):
            calls['engine_path'] = engine_path
            calls['projection_path'] = projection_path

        monkeypatch.setattr(trt_module.TransReIdTrtExtractor, '__init__', fake_init)
        extractor = create_extractor('transreid_trt', engine_path=ENGINE,
                                     projection_path='p.npz')
        assert isinstance(extractor, trt_module.TransReIdTrtExtractor)
        assert calls == {'engine_path': ENGINE, 'projection_path': 'p.npz'}

    def test_missing_engine_fails_fast_with_the_build_script_named(self, tmp_path):
        """§5: an absent engine is an error, never a mock embedding."""
        _, projection = _assets(tmp_path)
        with pytest.raises(FileNotFoundError, match='TensorRT engine not found'):
            create_extractor('transreid_trt', engine_path=str(tmp_path / 'absent.engine'),
                             projection_path=projection)

    def test_empty_engine_path_is_not_treated_as_valid(self, tmp_path):
        _, projection = _assets(tmp_path)
        with pytest.raises(FileNotFoundError):
            create_extractor('transreid_trt', engine_path='', projection_path=projection)

    def test_missing_projection_fails_fast(self, tmp_path):
        """The engine stops at 768-D; without the fitted projection no 512-D output exists."""
        engine, _ = _assets(tmp_path)
        with pytest.raises(FileNotFoundError, match='SVD projection not found'):
            create_extractor('transreid_trt', engine_path=engine,
                             projection_path=str(tmp_path / 'absent.npz'))

    def test_host_without_tensorrt_raises_instead_of_mocking(self, tmp_path):
        """Every asset present, no tensorrt → RuntimeError mentioning the blocker."""
        if 'tensorrt' in sys.modules:
            pytest.skip('a real tensorrt is importable here; the no-mock branch is not the one '
                        'being exercised')
        engine, projection = _assets(tmp_path)
        with pytest.raises(RuntimeError, match='tensorrt is not importable'):
            create_extractor('transreid_trt', engine_path=engine, projection_path=projection)

    def test_accepted_and_documented_model_types_agree(self):
        """The docstring's option list cannot drift from the values the factory accepts."""
        from edgesync_identity.models import extractor_factory as factory_module

        documented = set()
        for line in inspect.getsource(factory_module).splitlines():
            stripped = line.strip()
            if stripped.startswith('- "'):
                documented.add(stripped.split('"')[1])
        accepted = {'osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt'}
        assert documented == accepted

    def test_unknown_type_message_lists_every_supported_value(self):
        with pytest.raises(ValueError) as excinfo:
            create_extractor('transreid_trt_v9')
        message = str(excinfo.value)
        for name in ('osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt'):
            assert name in message
        assert 'Unknown model_type' in message

    def test_model_type_is_stripped_and_case_insensitive(self, tmp_path):
        _, projection = _assets(tmp_path)
        with pytest.raises(FileNotFoundError, match='TensorRT engine not found'):
            create_extractor('  TransReID_TRT ', engine_path=ENGINE, projection_path=projection)


class TestNodeContractCompatibility:
    """identity_node probes these two things about whatever extractor it is handed."""

    def test_extract_embedding_accepts_the_cam_label_kwarg(self):
        """TR-03 routes SIE through ``inspect.signature``, so the kwarg must be declared."""
        from edgesync_identity.models.transreid_trt_extractor import TransReIdTrtExtractor

        assert 'cam_label' in inspect.signature(
            TransReIdTrtExtractor.extract_embedding).parameters

    def test_inherits_the_frozen_512_dimension(self):
        from edgesync_identity.interfaces.base_identity_extractor import BaseIdentityExtractor
        from edgesync_identity.models.transreid_trt_extractor import TransReIdTrtExtractor

        assert issubclass(TransReIdTrtExtractor, BaseIdentityExtractor)
        assert TransReIdTrtExtractor.EMBEDDING_DIM == 512


class TestFrozenArmsUnchanged:
    """§3C step 5: adding a branch must not move the behaviour of the existing arms."""

    def test_osnet_mock_still_builds_an_engineless_osnet(self):
        extractor = create_extractor('osnet_mock')
        assert isinstance(extractor, OsnetTrtExtractor)
        assert extractor.engine_path is None
        assert extractor._trt_context is None

    def test_osnet_trt_still_requires_its_engine(self):
        with pytest.raises(FileNotFoundError, match='osnet_trt requires a valid TensorRT engine'):
            create_extractor('osnet_trt', engine_path='/nonexistent/path.engine')

    def test_osnet_torch_still_requires_its_checkpoint(self):
        with pytest.raises(FileNotFoundError, match='osnet_torch requires a valid checkpoint'):
            create_extractor('osnet_torch', checkpoint_path='/nonexistent/osnet.pth')

    def test_transreid_torch_arm_still_requires_both_assets(self):
        with pytest.raises(FileNotFoundError, match='transreid requires a valid checkpoint'):
            create_extractor('transreid', checkpoint_path='/nonexistent/model.pth',
                             projection_path='/nonexistent/proj.npz')

    def test_importing_the_factory_does_not_pull_in_tensorrt_code(self):
        """Lazy import: a host with no TRT must still be able to select osnet_mock.

        Asserted in a fresh interpreter instead of by matching source strings. ES-EXP07 §4.1 moved
        dispatch into ``BACKBONE_REGISTRY``, whose builders resolve the class through
        ``importlib``, so the literal ``from ..models.transreid_trt_extractor import`` this test
        once searched for no longer exists anywhere. What the check has always meant is that the
        TRT module never enters ``sys.modules`` unless the TRT arm is asked for — which is what is
        asserted below, together with the construction actually succeeding.
        """
        deferred = ('tensorrt', 'pycuda', 'edgesync_identity.models.transreid_trt_extractor')
        probe = (
            'import sys;'
            'import edgesync_identity.models.extractor_factory as f;'
            'extractor = f.create_extractor("osnet_mock");'
            'loaded = [m for m in %r if m in sys.modules];'
            'print(loaded, type(extractor).__name__)' % (deferred,)
        )
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            [pkg_dir] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                                env=env, cwd=pkg_dir)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith('[]'), result.stdout
        assert 'OsnetTrtExtractor' in result.stdout, result.stdout


class TestConfiguredValueListAgreement:
    def test_bringup_yaml_documents_the_same_option_set(self):
        """The shipped config's inline option list is the operator's only documentation."""
        yaml_path = os.path.abspath(os.path.join(
            pkg_dir, '..', 'edgesync_bringup', 'config', 'algorithm_thresholds.yaml'))
        if not os.path.isfile(yaml_path):
            pytest.skip('edgesync_bringup not present alongside edgesync_identity in this '
                        'workspace layout')
        with open(yaml_path, encoding='utf-8') as handle:
            text = handle.read()
        options_line = next(line for line in text.splitlines()
                            if 'reid_model_type:' in line and 'Options:' in line)
        for name in ('osnet_mock', 'osnet_trt', 'osnet_torch', 'transreid', 'transreid_trt'):
            assert f'"{name}"' in options_line
        # The comment edit must not have changed the value actually selected.
        import yaml as _yaml

        loaded = _yaml.safe_load(text)['algorithm_thresholds']['ros__parameters']
        assert loaded['reid_model_type'] == 'transreid'
        assert loaded['reid_engine_path'] == ''
