"""Extractor Factory: Registry-backed runtime selection of Re-ID feature extractors.

SOLID: DIP & OCP. Gate 1 + Gate 3 — ES-EXP07-DCP-REV10 §8.1, and the Task 7 plugin-registry
requirement (ES-EXP07 close-out §4.1).

Registered backbone identifiers (``BACKBONE_ID_*`` constants, never loose literals at call sites):
    - "osnet_mock"    -> OsnetTrtExtractor with NO engine (CPU-only deterministic pseudo-embedding).
    - "osnet_trt"     -> OsnetTrtExtractor with a mandatory TensorRT engine path (fail-fast).
    - "osnet_torch"   -> OsnetTorchExtractor with native PyTorch inference (Gate 3 baseline).
    - "transreid"     -> TransReIdExtractor with mandatory checkpoint + projection (fail-fast).
    - "transreid_trt" -> TransReIdTrtExtractor with mandatory engine + host-side projection (TR-10).

``BACKBONE_REGISTRY`` is the single source of truth: it maps each identifier to the concrete
implementation it must produce plus a builder that performs the asset validation. An unknown
identifier is rejected before any extractor is constructed.

Optional heavy dependencies (``torch``, ``tensorrt``, ``pycuda``) are imported by the builders, not
at module import, so importing this factory stays possible on a CPU-only host and no import cycle
is introduced: the builders import extractor modules from outside this module's own import graph.

Traces: ISSUE-07 (benchmark guard), ISSUE-08 (fail-fast on missing assets), ES-EXP07 §4.1.
"""
import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..models.osnet_trt_extractor import OsnetTrtExtractor

# ── Plugin identifiers (Rule #0: declared constants, not scattered string literals) ──
BACKBONE_ID_OSNET_MOCK = 'osnet_mock'
BACKBONE_ID_OSNET_TRT = 'osnet_trt'
BACKBONE_ID_OSNET_TORCH = 'osnet_torch'
BACKBONE_ID_TRANSREID = 'transreid'
BACKBONE_ID_TRANSREID_TRT = 'transreid_trt'

#: Identifier used when nothing is configured. Selecting it implicitly is reported by
#: ``identity_node`` as a fallback, never as a real Re-ID model (ES-EXP07 §4.5).
DEFAULT_BACKBONE_ID = BACKBONE_ID_OSNET_MOCK

#: Identifiers whose embeddings are deterministic stand-ins, not model output.
MOCK_BACKBONE_IDS = (BACKBONE_ID_OSNET_MOCK,)


@dataclass(frozen=True)
class BackboneSpec:
    """One registered extractor implementation and the code that builds it.

    Attributes:
        backbone_id: Registry key, already normalized (lower-case, stripped).
        implementation: ``module.ClassName`` this id MUST produce.
        description: Human-readable summary for introspection and logs.
        builder: Callable performing asset validation and instantiation. Receives the keyword
            arguments it declares; the factory passes only the ones a builder asks for.
    """

    backbone_id: str
    implementation: str
    description: str
    builder: Callable[..., BaseIdentityExtractor]

    @property
    def module_path(self) -> str:
        return self.implementation.rsplit('.', 1)[0]

    @property
    def class_name(self) -> str:
        return self.implementation.rsplit('.', 1)[1]

    @property
    def is_mock(self) -> bool:
        return self.backbone_id in MOCK_BACKBONE_IDS


# ── Fail-fast asset checks (ISSUE-08). Messages are part of the recorded test contract. ──

def _require_file(path: str, kind: str, backbone_id: str, hint: str) -> str:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ExtractorFactory] {backbone_id} requires a valid {kind} file. "
            f"Path provided: '{path}' — file not found. {hint}"
        )
    return path


def _build_osnet_mock() -> BaseIdentityExtractor:
    # CI-safe mock: no engine loaded, deterministic pseudo-embeddings from RGB stats.
    # ISSUE-07 benchmark guard: results from this extractor MUST NOT be reported
    # as scientific Re-ID accuracy numbers.
    return OsnetTrtExtractor(engine_path=None)


def _build_osnet_trt(engine_path: str) -> BaseIdentityExtractor:
    _require_file(
        engine_path, 'TensorRT engine', BACKBONE_ID_OSNET_TRT,
        'Build the engine with trtexec or provide the correct path via '
        "'reid_engine_path' parameter."
    )
    return OsnetTrtExtractor(engine_path=engine_path)


def _build_osnet_torch(checkpoint_path: str) -> BaseIdentityExtractor:
    _require_file(
        checkpoint_path, 'checkpoint', BACKBONE_ID_OSNET_TORCH,
        'Download osnet_x1_0_market1501.pth and set ' "'reid_checkpoint_path' parameter."
    )
    module = importlib.import_module('edgesync_identity.models.osnet_torch_extractor')
    return module.OsnetTorchExtractor(checkpoint_path=checkpoint_path)


def _build_transreid(checkpoint_path: str, projection_path: str) -> BaseIdentityExtractor:
    # TransReID ViT-Base/16 with Centered Truncated-SVD projection: both assets are mandatory.
    _require_file(
        checkpoint_path, 'checkpoint', BACKBONE_ID_TRANSREID,
        'Download transformer_120.pth and set ' "'reid_checkpoint_path' parameter."
    )
    _require_file(
        projection_path, 'SVD projection matrix', BACKBONE_ID_TRANSREID,
        'Run fit_svd_projection.py and set ' "'reid_projection_path' parameter."
    )
    module = importlib.import_module('edgesync_identity.models.transreid_extractor')
    return module.TransReIdExtractor(checkpoint_path=checkpoint_path,
                                     projection_path=projection_path)


def _build_transreid_trt(engine_path: str, projection_path: str) -> BaseIdentityExtractor:
    # TR-10: TransReID feature engine + the same host-side SVD projection the Torch arm uses.
    # Fail-fast lives inside TransReIdTrtExtractor.__init__ so the message names the build script,
    # and a host without tensorrt raises there rather than silently degrading to a pseudo-embedding.
    module = importlib.import_module('edgesync_identity.models.transreid_trt_extractor')
    return module.TransReIdTrtExtractor(engine_path=engine_path,
                                       projection_path=projection_path)


#: Single source of truth for supported extractors. Declaration order is deterministic.
BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    spec.backbone_id: spec
    for spec in (
        BackboneSpec(
            backbone_id=BACKBONE_ID_OSNET_MOCK,
            implementation='edgesync_identity.models.osnet_trt_extractor.OsnetTrtExtractor',
            description='CPU-only deterministic pseudo-embedding; NOT scientific Re-ID output.',
            builder=_build_osnet_mock,
        ),
        BackboneSpec(
            backbone_id=BACKBONE_ID_OSNET_TRT,
            implementation='edgesync_identity.models.osnet_trt_extractor.OsnetTrtExtractor',
            description='OSNet x1.0 via a mandatory serialized TensorRT engine.',
            builder=_build_osnet_trt,
        ),
        BackboneSpec(
            backbone_id=BACKBONE_ID_OSNET_TORCH,
            implementation='edgesync_identity.models.osnet_torch_extractor.OsnetTorchExtractor',
            description='OSNet x1.0 native PyTorch inference (Gate 3 baseline arm).',
            builder=_build_osnet_torch,
        ),
        BackboneSpec(
            backbone_id=BACKBONE_ID_TRANSREID,
            implementation='edgesync_identity.models.transreid_extractor.TransReIdExtractor',
            description='TransReID ViT-B/16 + centered truncated-SVD 768->512 projection.',
            builder=_build_transreid,
        ),
        BackboneSpec(
            backbone_id=BACKBONE_ID_TRANSREID_TRT,
            implementation='edgesync_identity.models.transreid_trt_extractor.TransReIdTrtExtractor',
            description='TransReID ViT-B/16 TensorRT feature engine + host-side SVD projection.',
            builder=_build_transreid_trt,
        ),
    )
}

#: Keyword arguments each builder accepts, resolved once from its signature.
_BUILDER_ARGS: Dict[str, Tuple[str, ...]] = {
    backbone_id: tuple(_spec.builder.__code__.co_varnames[:_spec.builder.__code__.co_argcount])
    for backbone_id, _spec in BACKBONE_REGISTRY.items()
}


def normalize_backbone_id(model_type: str) -> str:
    """Return the registry-normalized form of ``model_type`` without validating it."""
    return str(model_type or '').strip().lower()


def is_registered_backbone(model_type: str) -> bool:
    """True when ``model_type`` names a registered backbone after normalization."""
    return normalize_backbone_id(model_type) in BACKBONE_REGISTRY


def get_registered_backbones() -> Dict[str, str]:
    """Return ``backbone_id -> module.ClassName`` for every registered extractor.

    Insertion order is the registry declaration order, so the representation is deterministic and
    safe to assert on in tests and to print in logs.
    """
    return {backbone_id: spec.implementation for backbone_id, spec in BACKBONE_REGISTRY.items()}


def registered_backbone_ids() -> Tuple[str, ...]:
    """Return the registered identifiers in deterministic declaration order."""
    return tuple(BACKBONE_REGISTRY)


def validate_backbone_id(model_type: str) -> str:
    """Normalize ``model_type`` and reject anything not present in ``BACKBONE_REGISTRY``.

    Raises:
        ValueError: naming the rejected value and every registered identifier.
    """
    normalized = normalize_backbone_id(model_type)
    if normalized not in BACKBONE_REGISTRY:
        raise ValueError(
            f"[ExtractorFactory] Unknown model_type '{model_type}'. "
            f"Supported values: {', '.join(repr(b) for b in BACKBONE_REGISTRY)}."
        )
    return normalized


def _resolve_registered_class(spec: BackboneSpec) -> type:
    """Import and return the class this registry entry declares."""
    return getattr(importlib.import_module(spec.module_path), spec.class_name)


def _verify_against_registry(instance: Any, spec: BackboneSpec) -> None:
    """Assert the produced object really is the registered implementation and honoring the contract.

    This is not a restatement of the dispatch: the object produced by the builder is compared
    against the class obtained by resolving the registry entry's ``module.ClassName`` string
    independently, and against the frozen ``BaseIdentityExtractor`` contract.
    """
    if not isinstance(instance, BaseIdentityExtractor):
        raise TypeError(
            f"[ExtractorFactory] backbone '{spec.backbone_id}' produced "
            f"{type(instance).__name__}, which does not implement BaseIdentityExtractor."
        )
    registered = _resolve_registered_class(spec)
    if type(instance) is not registered:
        raise TypeError(
            f"[ExtractorFactory] backbone '{spec.backbone_id}' declared "
            f"{spec.implementation} but produced {type(instance).__module__}."
            f"{type(instance).__name__}."
        )


def create_extractor(model_type: str,
                     checkpoint_path: str = "",
                     projection_path: str = "",
                     engine_path: str = "") -> BaseIdentityExtractor:
    """Instantiate the extractor registered under ``model_type``.

    Args:
        model_type: A registered backbone identifier (see ``get_registered_backbones``).
        checkpoint_path: Model weights (.pth). Required for "osnet_torch" and "transreid".
        projection_path: SVD projection matrix (.npz). Required for "transreid" and
            "transreid_trt" — the engine stops at the 768-D feature so the fitted projection
            stays host-side and its provenance remains auditable.
        engine_path: Serialized TensorRT engine (.engine). Required for "osnet_trt" and
            "transreid_trt".

    Returns:
        BaseIdentityExtractor: Instance of the registered implementation.

    Raises:
        ValueError: If ``model_type`` is not registered.
        FileNotFoundError: If a required checkpoint, projection or engine file is missing.
        RuntimeError: If a TensorRT backbone is requested on a host without TensorRT.
        TypeError: If the produced object is not the registered implementation.
    """
    backbone_id = validate_backbone_id(model_type)
    spec = BACKBONE_REGISTRY[backbone_id]

    available = {'checkpoint_path': checkpoint_path,
                 'projection_path': projection_path,
                 'engine_path': engine_path}
    try:
        instance = spec.builder(**{name: available[name] for name in _BUILDER_ARGS[backbone_id]})
    except ImportError as exc:  # optional dependency absent: never degrade to a mock
        raise RuntimeError(
            f"[ExtractorFactory] backbone '{backbone_id}' needs a dependency that is not "
            f"installed on this host: {exc}. It will NOT fall back to '{DEFAULT_BACKBONE_ID}'."
        ) from exc

    _verify_against_registry(instance, spec)
    return instance
