"""TransReID TensorRT feature extractor (TR-10) — production deployment arm.

Charter §6 steps 6-10 describe the path this class implements: build the engine **in the actual
target environment** (Jetson Orin Nano), load it resident, run real inference, apply the same
documented post-graph contract as the Torch arm, and verify the production path loads the intended
engine plus the intended preprocessing and projection.

Deliberate divergences from ``osnet_trt_extractor.py`` (inspected, not modified — charter §3B):

  * NO silent mock. ``OsnetTrtExtractor`` falls back to ``np.tile(crop RGB means)`` whenever the
    engine fails to load, and that vector is unit-normalised and returned like a real embedding.
    Here, a missing ``tensorrt``/``pycuda`` module or an unloadable engine raises at construction.
    Reason: charter §5 forbids replacing failed inference with synthetic or fabricated outputs, and
    a deterministic-looking pseudo-embedding is exactly that. Callers therefore get a loud
    construction failure instead of a plausible wrong feature.
  * The engine ends at the post-BNNeck 768-D feature. The fitted SVD 768->512 projection stays on
    the host (``interfaces/base_identity_extractor.py`` output contract is float32[512], unit-norm
    or the zeros sentinel). Baking a fitted artefact into a binary engine would fuse two assets
    with different provenance and make the projection unauditable (charter §5).
  * Preprocessing is the TransReID contract (mean=std=[0.5,0.5,0.5]), not the ImageNet stats the
    OSNet arm uses. The two arms are not interchangeable at the input boundary, which is why the
    engine cannot be reused across models.

Status of this file on the development host: importable and unit-tested for its contract and
arithmetic paths, but the engine itself has never been built or executed here — there is no
TensorRT, no trtexec and no JetPack on this machine. Every hardware-dependent claim stays
``BLOCKED — Pending Jetson Orin Nano access``.

Traces: ISSUE-01 (non-finite guard), ISSUE-07 (genuine model only), ISSUE-08 (fail-fast assets).
"""
import logging
import os
import time
from typing import Optional

import numpy as np

from ..interfaces.base_identity_extractor import BaseIdentityExtractor

logger = logging.getLogger(__name__)

# The wire contract of the exported graph (TR-09): one float32 NCHW image, one 768-D feature.
IMAGE_BINDING = 'image'
CAM_BINDING = 'cam_label'
FEATURE_BINDING = 'feature_768'
FEATURE_DIM = 768


class TransReIdTrtExtractor(BaseIdentityExtractor):
    """TransReID ViT-Base/16 embedding extractor accelerated by a TensorRT engine.

    Args:
        engine_path: Serialized engine built on the target device (REQUIRED).
        projection_path: SVD 768->512 .npz fitted artefact (REQUIRED, same file the Torch arm uses).
        input_shape: (height, width) the engine was built for; the contract is (256, 128).
        use_unified_memory: Allocate the device buffers in managed memory, the usual Jetson
            configuration on a 4 GB unified-memory Orin Nano.
    """

    #: Post-BNNeck feature width the engine must produce; a 512-output engine means someone
    #: baked the projection into it and this extractor would then double-project.
    FEATURE_DIM = FEATURE_DIM

    def __init__(self, engine_path: str, projection_path: str,
                 input_shape: tuple = (256, 128),
                 use_unified_memory: bool = True):
        # ── TR-02 observability, before any loading, so load-time state is recordable ──
        self.last_failure: Optional[dict] = None
        self.failure_counts: dict = {}
        self.embedding_count: int = 0
        self.engine_metadata: Optional[dict] = None

        # ── ISSUE-08: fail-fast on both assets ──
        if not engine_path or not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f'[TransReIdTrtExtractor] TensorRT engine not found: {engine_path!r}. Build it on '
                f'the target device with scripts/build_transreid_trt_engine.py; an engine is not '
                f'portable across GPU architectures.')
        if not projection_path or not os.path.isfile(projection_path):
            raise FileNotFoundError(
                f'[TransReIdTrtExtractor] SVD projection not found: {projection_path!r}. '
                f'This arm produces float32[512] only by projecting a 768-D engine output.')

        self.engine_path = os.path.abspath(engine_path)
        self.projection_path = os.path.abspath(projection_path)
        self.input_height, self.input_width = input_shape
        self.use_unified_memory = use_unified_memory

        self._load_svd_projection()
        self._load_engine()

    # ────────────────────────────── assets ──────────────────────────────

    def _load_svd_projection(self):
        data = np.load(self.projection_path)
        self._svd_projection = data['projection'].astype(np.float32)
        self._svd_mean = data['mean'].astype(np.float32)
        if self._svd_projection.shape != (self.FEATURE_DIM, 512):
            raise ValueError(f'SVD projection shape mismatch: expected ({self.FEATURE_DIM}, 512), '
                             f'got {self._svd_projection.shape}')
        if self._svd_mean.shape != (self.FEATURE_DIM,):
            raise ValueError(f'SVD mean shape mismatch: expected ({self.FEATURE_DIM},), '
                             f'got {self._svd_mean.shape}')

    def _load_engine(self):
        """Deserialize the engine resident in (unified) memory and read its build settings.

        Raises RuntimeError — never falls back to a mock — when TensorRT is unavailable or the
        engine cannot be loaded, so a caller cannot mistake this arm for working inference.
        """
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                f'tensorrt is not importable on this host ({exc}). TR-10/TR-11 are BLOCKED — '
                f'Pending Jetson Orin Nano access; this class has not been exercised against a '
                f'real engine anywhere.') from exc

        try:
            import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                f'pycuda is not importable on this host ({exc}). Install it on the target, or '
                f'adapt _execute() to the cuda-python bindings available there.') from exc

        self._trt = trt
        self._cuda = cuda
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as handle, trt.Runtime(trt_logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(
                f'engine deserialisation returned None for {self.engine_path} — the file was '
                f'built by a different TensorRT major version, for a different GPU, or is '
                f'truncated. Refusing to continue: no substitute inference exists.')

        self._context = self._engine.create_execution_context()
        self._bindings = self._read_bindings()
        self.has_cam_input = CAM_BINDING in self._bindings['inputs']
        self._allocate_buffers()
        self._warmup()

    def _read_bindings(self):
        """Collect IO names, dtypes and shapes across the TRT 8.x and 10.x binding APIs."""
        trt = self._trt
        engine, context = self._engine, self._context
        inputs, outputs = {}, {}
        self._binding_order = []
        dtype_names = {trt.float32: 'float32', trt.float16: 'float16', trt.int8: 'int8',
                       trt.int32: 'int32', trt.bool: 'bool', trt.int64: 'int64'}

        def record(name, is_input, dtype, shape):
            target = inputs if is_input else outputs
            target[name] = {'dtype': dtype_names.get(dtype, str(dtype)), 'shape': list(shape)}
            self._binding_order.append(name)

        if hasattr(engine, 'num_io_tensors'):                      # TRT >= 10
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                dtype = engine.get_tensor_dtype(name)
                shape = tuple(int(d) for d in (context.get_tensor_shape(name) if context is not None
                                               else engine.get_tensor_shape(name)))
                if is_input and context is not None and -1 in shape:
                    # TR-09 exports with only the batch axis dynamic; production is batch 1.
                    shape = tuple(1 if d == -1 else d for d in shape)
                if is_input and context is not None:
                    context.set_input_shape(name, shape)
                record(name, is_input, dtype, shape)
            if context is not None:
                # Output shapes resolve only after the input shapes are fixed.
                for name in outputs:
                    outputs[name]['shape'] = [int(d) for d in context.get_tensor_shape(name)]
        else:                                                       # TRT 8.x
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                shape = (context.get_binding_shape(i) if context is not None
                         else engine.get_binding_shape(i))
                record(name, engine.binding_is_input(i), dtype, shape)

        self.engine_metadata = {
            'engine_path': self.engine_path,
            'engine_bytes': os.path.getsize(self.engine_path),
            'tensorrt_version': trt.__version__,
            'inputs': inputs,
            'outputs': outputs,
            'binding_order': list(self._binding_order),
            'input_shape': [1, 3, self.input_height, self.input_width],
            'uses_unified_memory': self.use_unified_memory,
            'name': getattr(engine, 'name', ''),
            'device_memory_bytes': int(engine.get_device_memory_size_v3().total)
                                    if hasattr(engine, 'get_device_memory_size_v3') else None,
            'api': 'num_io_tensors (TRT>=10)' if hasattr(engine, 'num_io_tensors')
                   else 'num_bindings (TRT 8.x)',
        }
        return {'inputs': inputs, 'outputs': outputs}

    def _validate_engine_contract(self):
        """The engine must expose the graph TR-09 exported and validated — nothing else."""
        outs = self._bindings['outputs']
        if list(outs) != [FEATURE_BINDING]:
            raise RuntimeError(f'engine outputs {list(outs)}, expected exactly [{FEATURE_BINDING}]. '
                               f'Refusing to guess which tensor is the feature.')
        if outs[FEATURE_BINDING]['shape'][-1] != self.FEATURE_DIM:
            raise RuntimeError(
                f"engine feature width {outs[FEATURE_BINDING]['shape'][-1]} != {self.FEATURE_DIM}. "
                f"A 512-wide output means the fitted SVD projection was baked into the engine, "
                f"which would be applied twice here.")
        ins = self._bindings['inputs']
        if IMAGE_BINDING not in ins:
            raise RuntimeError(f'engine inputs {list(ins)} lack the {IMAGE_BINDING!r} image binding')
        if list(ins[IMAGE_BINDING]['shape'][1:]) != [3, self.input_height, self.input_width]:
            raise RuntimeError(
                f"engine image shape {ins[IMAGE_BINDING]['shape']} disagrees with the declared "
                f"contract 1x3x{self.input_height}x{self.input_width}")
        unexpected = [n for n in ins if n not in (IMAGE_BINDING, CAM_BINDING)]
        if unexpected:
            raise RuntimeError(f'engine exposes unexpected inputs {unexpected}')

    def _allocate_buffers(self):
        """Allocate host+device buffers for every binding; production batch is fixed at 1."""
        self._stream = self._cuda.Stream()
        self._addresses = {}
        self._host = {
            IMAGE_BINDING: np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32),
            FEATURE_BINDING: np.empty((1, self.FEATURE_DIM), dtype=np.float32),
        }
        if self.has_cam_input:
            self._host[CAM_BINDING] = np.empty((1,), dtype=np.int64)
        for name, host in self._host.items():
            self._addresses[name] = self._cuda.mem_alloc(host.nbytes)

    def _warmup(self):
        """One real forward before timing or serving, so the first crop is not an outlier."""
        self._validate_engine_contract()
        self._host[IMAGE_BINDING][...] = 0.0
        if self.has_cam_input:
            self._host[CAM_BINDING][...] = 0
        self._run_engine()
        self.engine_metadata['warmed_up'] = True

    # ────────────────────────────── inference ──────────────────────────────

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize 256x128, /255, (x-0.5)/0.5, HWC->CHW, batch 1 — the deployed TransReID contract.

        Identical arithmetic to ``TransReIdExtractor.preprocess``; returning a numpy array instead
        of a torch tensor is the only difference, since the engine consumes device memory.
        """
        try:
            import cv2
            resized = cv2.resize(crop, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
        except ImportError:
            h, w = crop.shape[:2]
            y_idx = np.linspace(0, h - 1, self.input_height).astype(int)
            x_idx = np.linspace(0, w - 1, self.input_width).astype(int)
            resized = crop[y_idx[:, None], x_idx]

        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.5, 0.5, 0.5], dtype=np.float32)) \
            / np.array([0.5, 0.5, 0.5], dtype=np.float32)
        chw = np.transpose(normalized, (2, 0, 1))
        return np.ascontiguousarray(chw[None, ...])

    def _run_engine(self) -> np.ndarray:
        """H2D the staged buffers, execute, synchronise, D2H the feature. Returns 768-D float32."""
        for name in (IMAGE_BINDING, CAM_BINDING):
            host = self._host.get(name)
            if host is not None:
                self._addresses[name].copy_from_host(host.reshape(-1))

        addresses = [int(self._addresses[n]) for n in self._binding_order]
        if self.has_cam_input and CAM_BINDING not in self._binding_order:
            raise RuntimeError('cam_label buffer exists but the engine has no such binding')

        if hasattr(self._context, 'execute_async_v3'):        # TRT >= 10
            for name, address in zip(self._binding_order, addresses):
                self._context.set_tensor_address(name, address)
            ok = self._context.execute_async_v3(self._stream.handle)
        else:                                                  # TRT 8.x
            ok = self._context.execute_v2(addresses)
        if not ok:
            raise RuntimeError('TensorRT execution returned False (shape or memory violation)')

        self._stream.synchronize()
        self._addresses[FEATURE_BINDING].copy_to_host(
            self._host[FEATURE_BINDING].reshape(-1))
        return self._host[FEATURE_BINDING].reshape(-1).copy()

    def _execute(self, image_nhwc: np.ndarray, cam_label: Optional[int]) -> np.ndarray:
        """Run one engine invocation and return the 768-D post-BNNeck feature."""
        if cam_label is not None and not self.has_cam_input:
            raise ValueError('cam_label supplied to a camera-agnostic engine (no cam_label input)')
        if self.has_cam_input and cam_label is None:
            raise ValueError('this engine takes a cam_label input; none supplied')

        self._host[IMAGE_BINDING][...] = image_nhwc
        if self.has_cam_input:
            self._host[CAM_BINDING][0] = int(cam_label)
        return self._run_engine()

    # ────────────────────────────── contract ──────────────────────────────

    def _sentinel(self, stage: str, reason: str) -> np.ndarray:
        """Contract-mandated invalid vector, attributed (TR-02).

        ``interfaces/base_identity_extractor.py`` is frozen: implementations MUST return
        ``zeros(512)`` for malformed input and MUST NOT raise to callers. The node refuses to
        score whatever follows one of these.
        """
        self.last_failure = {'stage': stage, 'reason': reason,
                             'monotonic_s': time.monotonic(), 'engine': self.engine_path}
        self.failure_counts[stage] = self.failure_counts.get(stage, 0) + 1
        logger.error(f'[TransReIdTrtExtractor] invalid embedding at stage {stage!r}: {reason}')
        return np.zeros(512, dtype=np.float32)

    def extract_embedding(self, image_crop: np.ndarray,
                          cam_label: Optional[int] = None) -> np.ndarray:
        """float32[512], unit L2 norm, or the zeros sentinel — the frozen wire contract.

        Pipeline: validate crop → contract preprocessing → TensorRT engine → 768-D post-BNNeck
        feature → centre → SVD 768->512 → finite check → L2 normalise.
        """
        if image_crop is None or image_crop.size == 0:
            return self._sentinel('input_empty', 'crop is None or has zero elements')
        if image_crop.ndim != 3 or image_crop.shape[2] != 3:
            return self._sentinel('input_shape',
                                  f'expected HxWx3 RGB crop, got shape {image_crop.shape}')
        if image_crop.shape[0] < 16 or image_crop.shape[1] < 16:
            return self._sentinel('input_too_small',
                                  f'crop {image_crop.shape[:2]} is smaller than the 16px patch grid')
        if cam_label is not None and not 0 <= int(cam_label) <= 5:
            return self._sentinel('cam_label_out_of_range',
                                  f'cam_label {cam_label} outside the 0-5 camera range')

        try:
            x = np.asarray(self._execute(self.preprocess(image_crop), cam_label),
                           dtype=np.float32).reshape(-1)
            if x.shape[0] != self.FEATURE_DIM:
                return self._sentinel('engine_shape_mismatch',
                                      f'engine returned {x.shape[0]} values, expected '
                                      f'{self.FEATURE_DIM}')
            if not np.all(np.isfinite(x)):
                return self._sentinel('backbone_non_finite', 'engine output contains non-finite values')

            z_proj = (x - self._svd_mean) @ self._svd_projection
            if not np.all(np.isfinite(z_proj)):
                return self._sentinel('projection_non_finite',
                                      'SVD projection produced non-finite values')
            norm = float(np.linalg.norm(z_proj).item())
            if norm <= 1e-6:
                return self._sentinel('projection_degenerate',
                                      f'projected feature norm {norm:.3e} <= 1e-6')

            self.embedding_count += 1
            self.last_failure = None
            return (z_proj / norm).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 — the contract forbids raising to callers
            return self._sentinel('inference_exception', f'{type(exc).__name__}: {exc}')
