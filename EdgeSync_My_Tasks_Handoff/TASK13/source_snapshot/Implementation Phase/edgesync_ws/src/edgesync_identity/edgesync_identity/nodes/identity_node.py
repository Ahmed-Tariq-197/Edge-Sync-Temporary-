"""Identity Node (C-07): Resident Re-ID Engine & Candidate Verification Service.

Gate 1 — ES-EXP07-DCP-REV10:
  - ISSUE-02: Production callback routes through IdentityVerifier.verify_candidate()
    with mandatory isfinite guard and spatial Mahalanobis gating.
  - ISSUE-07: Extractor injected via ExtractorFactory (model_type parameter).
  - ISSUE-10: Parameter keys harmonized with algorithm_thresholds.yaml
    (reid_similarity_threshold → similarity_threshold, etc.).
  - ISSUE-12: Transaction-keyed candidate cache with 2.0 s TTL and event purge
    subscribers for /handoff/confirm and /handoff/cancel.

Task 7 TransReID integration fixes (ES-EXP07 execution, see ES-EXP07-EXECUTION-AND-CHANGE-LOG.md):
  - TR-02: an all-zero/non-finite embedding is refused with an explicit INVALID_EMBEDDING
    reason instead of being scored as a low-similarity mismatch.
  - TR-03: reid_camera_id (1-based, 0 = unset) feeds the TransReID SIE camera label; the
    frozen IdentityVerify.srv is unchanged.
  - TR-04: candidate_crop is decoded through explicit encoding handling (bgr8/rgb8),
    unsupported encodings are rejected with a diagnostic.
  - TR-05: checkpoint/projection/engine paths are resolved without depending on the CWD.
  - ES-EXP07 §4.3: reid_model_type stays canonical; reid_model is accepted as an alias with
    documented precedence (explicit canonical > explicit alias > registry default).
  - ES-EXP07 §4.4: exp_id (default EXP07) is declared, read and logged as EXP-METADATA.
  - ES-EXP07 §4.5: falling back to osnet_mock because nothing configured a model logs a WARN;
    requesting the mock explicitly logs a NOTE instead. The srv/msg contract is unchanged.
  - ES-EXP07 <-> ES-EXP13: optional ReidSignature production. IdentityNode now publishes the
    smoothed target signature (publish_reid_signature, reid_signature_topic) and can take the crop
    that feeds it from a sensor_msgs/Image topic (reid_target_crop_topic). Both default to off, so
    the frozen IdentityVerify.srv and every existing path are unchanged.

SOLID & Architectural Mandates:
- Resident engine loaded at startup in unified memory (zero engine-load delay).
- Serves IdentityVerify.srv (I-10) for candidate cross-verification.
- Maintains continuous 2 Hz smoothed target signature buffer.
- ZERO magic numbers: all thresholds and topic/service names injected from YAML.
"""
from typing import Optional, Dict
import inspect
import os
import time
import rclpy
from rclpy.node import Node
import numpy as np

from edgesync_msgs.srv import IdentityVerify
from edgesync_msgs.msg import HandoffConfirm, HandoffCancel
from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..models.extractor_factory import (
    DEFAULT_BACKBONE_ID, MOCK_BACKBONE_IDS, create_extractor, normalize_backbone_id,
    registered_backbone_ids, validate_backbone_id)
from ..core.signature_smoother import SignatureSmoother
from ..core.identity_verifier import IdentityVerifier
from ..core.reid_input_adapter import (
    image_to_rgb, is_invalid_embedding, resolve_asset_path, resolve_camera_label)


class _CacheEntry:
    """Transaction-scoped candidate vector cache entry with TTL (ISSUE-12)."""
    __slots__ = ('vector', 'timestamp')

    def __init__(self, vector: np.ndarray):
        self.vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.timestamp = time.monotonic()


class IdentityNode(Node):
    """ROS 2 Node providing resident Re-ID embedding and verification services."""

    # Maximum cache entry lifetime (seconds) — ISSUE-12
    CACHE_TTL_S: float = 2.0

    # ES-EXP07 §4.3: sentinel for "this model id parameter was not supplied".
    MODEL_ID_UNSET: str = ''

    def __init__(self, extractor: Optional[BaseIdentityExtractor] = None):
        super().__init__('identity_node')

        # ── 1. Parameter Declarations (Zero Hardcoding — ISSUE-10) ──
        self.declare_parameter('robot_id', 'ugv_2')
        self.declare_parameter('similarity_threshold', 0.70)
        self.declare_parameter('smoothing_alpha', 0.20)
        self.declare_parameter('enable_spatial_gating', True)
        self.declare_parameter('service_name', '/identity/verify')
        # Gate 1 model selection parameters
        # ES-EXP07 §4.3/§4.5: '' is the "caller supplied nothing" sentinel. Declaring the mock as
        # the default made an accidental mock indistinguishable from a deliberate one, so the
        # default is unresolved here and _resolve_model_id() applies the documented precedence.
        self.declare_parameter('reid_model_type', self.MODEL_ID_UNSET)
        self.declare_parameter('reid_model', self.MODEL_ID_UNSET)  # ES-EXP07 §4.3: legacy alias
        self.declare_parameter('reid_checkpoint_path', '')
        self.declare_parameter('reid_projection_path', '')
        self.declare_parameter('reid_engine_path', '')
        # ES-EXP07 §4.4: experiment traceability id; overridable by exp7_transreid.yaml/launch.
        self.declare_parameter('exp_id', 'EXP07')
        # TR-03: 1-based camera id used for TransReID Side Information Embedding (SIE).
        # 0 (default) keeps inference camera-agnostic, i.e. exactly the pre-TR-03 behavior.
        self.declare_parameter('reid_camera_id', 0)
        # ES-EXP07 <-> ES-EXP13: the reference embedding a handoff carries had no carrier. These keys
        # turn IdentityNode into the producer of ReidSignature. Both are opt-in: with the defaults the
        # node behaves exactly as before, so an experiment that has not wired a crop source cannot
        # start publishing embeddings nobody asked for.
        self.declare_parameter('publish_reid_signature', False)
        self.declare_parameter('reid_signature_topic', '/edgesync/identity/reid_signature')
        self.declare_parameter('reid_target_crop_topic', '')
        self.declare_parameter('reid_signature_target_id', '')
        # Legacy parameter support for backward-compatible YAML consumption
        self.declare_parameter('engine_path', '')

        # ── 2. Extract Parameters ──
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.similarity_threshold = self.get_parameter('similarity_threshold').get_parameter_value().double_value
        alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        spatial_gating = self.get_parameter('enable_spatial_gating').get_parameter_value().bool_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        self.exp_id = (self.get_parameter('exp_id').get_parameter_value().string_value.strip()
                       or 'EXP07')
        model_type, self._model_id_source = self._resolve_model_id(
            self.get_parameter('reid_model_type').get_parameter_value().string_value,
            self.get_parameter('reid_model').get_parameter_value().string_value,
        )
        self.reid_model_id = model_type
        checkpoint_path = self.get_parameter('reid_checkpoint_path').get_parameter_value().string_value
        projection_path = self.get_parameter('reid_projection_path').get_parameter_value().string_value
        engine_path = self.get_parameter('reid_engine_path').get_parameter_value().string_value
        if not engine_path:
            engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.reid_camera_id = self.get_parameter('reid_camera_id').get_parameter_value().integer_value

        # ── 2b. TR-05: resolve model assets without depending on the process CWD ──
        checkpoint_path, ck_attempts = resolve_asset_path(checkpoint_path)
        projection_path, proj_attempts = resolve_asset_path(projection_path)
        engine_path, engine_attempts = resolve_asset_path(engine_path)
        self._asset_paths = {
            'checkpoint': checkpoint_path, 'projection': projection_path, 'engine': engine_path}
        self._asset_attempts = {
            'checkpoint': ck_attempts, 'projection': proj_attempts, 'engine': engine_attempts}
        # On-target verification needs to see which file was actually loaded, not just the request.
        for label, resolved in sorted(self._asset_paths.items()):
            if resolved:
                self.get_logger().info(
                    f"TR-05 reid_{label}_path resolved to '{resolved}' "
                    f"(exists={os.path.isfile(resolved)}, "
                    f"bytes={os.path.getsize(resolved) if os.path.isfile(resolved) else 0}, "
                    f"attempted={list(self._asset_attempts[label])})"
                )

        # ── 2c. TR-03: camera conditioning input for extractors that support SIE ──
        self._cam_label, cam_reason = resolve_camera_label(self.reid_camera_id)

        # ── 3. Dependency Injection via Factory (SOLID: DIP — ISSUE-07) ──
        if extractor is not None:
            self.extractor: BaseIdentityExtractor = extractor
        else:
            self.extractor = create_extractor(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                projection_path=projection_path,
                engine_path=engine_path,
            )
        self._extractor_accepts_cam_label = (
            'cam_label' in inspect.signature(self.extractor.extract_embedding).parameters)
        self._log_model_selection()
        self.get_logger().info(
            f"Extractor created: model_type='{model_type}', "
            f"class={type(self.extractor).__name__}, "
            f"camera_conditioning={'supported' if self._extractor_accepts_cam_label else 'not supported'} "
            f"({cam_reason})"
        )

        # ── 4. Core Algorithms ──
        self.smoother = SignatureSmoother(alpha=alpha, embedding_dim=512)
        self.verifier = IdentityVerifier(
            similarity_threshold=self.similarity_threshold,
            enable_spatial_gating=spatial_gating,
        )

        # ── 5. Transaction-Keyed Candidate Cache (ISSUE-12) ──
        self._candidate_cache: Dict[str, _CacheEntry] = {}

        # TR-02: counts verification requests refused because no valid embedding existed
        self._invalid_frame_count: int = 0

        # Legacy single-slot fallback for backward compatibility with tests
        self._stored_candidate_vector: Optional[np.ndarray] = None
        self._live_target_vector: Optional[np.ndarray] = None

        # ── 6. Service Server (I-10 Interface) ──
        self.srv_verify = self.create_service(
            IdentityVerify, service_name, self._handle_identity_verify
        )

        # ── 7. Event Purge Subscribers (ISSUE-12) ──
        # Subscribe to handoff confirm/cancel to clean up cached transaction entries.
        try:
            self._sub_confirm = self.create_subscription(
                HandoffConfirm, '/edgesync_swarm/handoff_confirm',
                self._on_handoff_confirm, 10
            )
            self._sub_cancel = self.create_subscription(
                HandoffCancel, '/edgesync_swarm/handoff_cancel',
                self._on_handoff_cancel, 10
            )
        except Exception:
            # If message types are not available (e.g., in unit tests), skip subscriptions
            self._sub_confirm = None
            self._sub_cancel = None

        # ── 8. Target Signature Production (ES-EXP07 <-> ES-EXP13) ──
        self._wire_signature_production()

        self.get_logger().info(
            f"EdgeSync Identity Node initialized: service={service_name}, "
            f"threshold={self.similarity_threshold:.2f}, model={model_type}"
        )
        # ES-EXP07 §4.4: single machine-readable line that ties every downstream log/result to
        # the experiment it came from. Parse it with key=value, no message-definition change.
        self.get_logger().info(
            f"EXP-METADATA exp_id={self.exp_id} reid_model={self.reid_model_id} "
            f"model_source={self._model_id_source} robot_id={self.robot_id} "
            f"camera_id={self.reid_camera_id}"
        )

    # ── ES-EXP07 §4.3/§4.5: model-id resolution and mock-fallback signalling ──

    def _resolve_model_id(self, canonical: str,
                          alias: str) -> tuple:
        """Return ``(model_id, origin)`` for the requested Re-ID backbone.

        Precedence (ES-EXP07 §1.3): explicit ``reid_model_type`` > explicit ``reid_model``
        alias > the registry default. A value equal to :attr:`MODEL_ID_UNSET` counts as
        "not supplied"; anything else is an explicit request and is validated against the
        registry, so an unknown id fails here with the registered choices listed.
        """
        canonical = (canonical or '').strip()
        alias = (alias or '').strip()
        if canonical and alias:
            canonical_id = validate_backbone_id(normalize_backbone_id(canonical))
            alias_id = validate_backbone_id(normalize_backbone_id(alias))
            if canonical_id != alias_id:
                self.get_logger().warn(
                    f"ES-EXP07 §4.3 conflicting model ids: reid_model_type='{canonical}' and "
                    f"reid_model='{alias}' resolve to '{canonical_id}' and '{alias_id}'. "
                    f"Using the canonical 'reid_model_type' value '{canonical_id}'; fix the "
                    f"configuration to silence this warning."
                )
            return canonical_id, 'explicit:reid_model_type'
        if canonical:
            return (validate_backbone_id(normalize_backbone_id(canonical)),
                    'explicit:reid_model_type')
        if alias:
            return validate_backbone_id(normalize_backbone_id(alias)), 'explicit:reid_model'
        return DEFAULT_BACKBONE_ID, 'default:node'

    def _log_model_selection(self) -> None:
        """Make implicit mock selection impossible to miss (ES-EXP07 §4.5)."""
        if self.reid_model_id not in MOCK_BACKBONE_IDS:
            self.get_logger().info(
                f"ES-EXP07 §4.3 Re-ID backbone '{self.reid_model_id}' (origin: {self._model_id_source})"
            )
            return
        if self._model_id_source == 'default:node':
            self.get_logger().warn(
                f"ES-EXP07 §4.5 SILENT-MOCK GUARD: no Re-ID model was configured, so this node "
                f"fell back to '{self.reid_model_id}', which returns deterministic "
                f"pseudo-embeddings. Results are NOT scientific Re-ID accuracy numbers. "
                f"Set reid_model_type (registered: {', '.join(registered_backbone_ids())}) "
                f"explicitly."
            )
        else:
            self.get_logger().info(
                f"ES-EXP07 §4.5 '{self.reid_model_id}' was explicitly requested via "
                f"{self._model_id_source}; it is a mock and must not be reported as a real "
                f"Re-ID model."
            )

    # ── Target Signature Production (ES-EXP07 <-> ES-EXP13) ──

    def _wire_signature_production(self) -> None:
        """Build the ReidSignature publisher and, if configured, the target-crop subscriber.

        Everything here is off by default. The reason it exists at all is that the smoothed signature
        this node already maintains had no way to reach a handoff: ``store_candidate_vector`` and
        ``set_live_target_vector`` are never called by production code, and the coordination node's
        ``set_reid_signature()`` has no caller either, so ``HandoffRequest.reid_feature_vector`` left
        as the zeros it was initialised with.
        """
        self.pub_signature = None
        self._sub_crop = None
        self._reid_signature_type = None
        self.signature_publish_count = 0
        self.signature_refusal_count = 0
        self._last_signature_ns = 0
        self._signature_target_id = ''

        if not self.get_parameter('publish_reid_signature').get_parameter_value().bool_value:
            self.get_logger().info(
                "[EXP07] publish_reid_signature=false: this node publishes no ReidSignature, so a "
                "handoff request carries whatever signature the coordination node already holds "
                "(nothing, on a fresh node).")
            return

        try:
            from edgesync_msgs.msg import ReidSignature
        except Exception as exc:
            self.get_logger().error(
                f"[EXP07] cannot publish signatures: edgesync_msgs.msg.ReidSignature is unavailable "
                f"({type(exc).__name__}: {exc}). The node keeps serving /identity/verify.")
            return

        self._reid_signature_type = ReidSignature
        topic = self.get_parameter('reid_signature_topic').get_parameter_value().string_value
        self._signature_topic = topic
        self.pub_signature = self.create_publisher(ReidSignature, topic, 5)

        crop_topic = self.get_parameter('reid_target_crop_topic').get_parameter_value().string_value
        if not crop_topic:
            self.get_logger().warn(
                f"[EXP07] signature publisher live on '{topic}' but reid_target_crop_topic is unset: "
                "no observation feeds the extractor, so this node can only ever publish an "
                "invalidation. Point reid_target_crop_topic at a sensor_msgs/Image target crop to "
                "produce embeddings.")
            return
        try:
            from sensor_msgs.msg import Image
        except Exception as exc:
            self.get_logger().error(
                f"[EXP07] cannot subscribe for crops: sensor_msgs.msg.Image is unavailable "
                f"({type(exc).__name__}: {exc}).")
            return

        self._signature_target_id = (
            self.get_parameter('reid_signature_target_id').get_parameter_value().string_value or '')
        self._sub_crop = self.create_subscription(
            Image, crop_topic, self._on_target_crop, 5)
        self.get_logger().info(
            f"[EXP07] signature production armed: crop topic='{crop_topic}' "
            f"signature topic='{topic}' target_id='{self._signature_target_id or '(unset)'}' "
            f"model_id='{self.reid_model_id}' camera_id={self.reid_camera_id} "
            f"exp_id={self.exp_id}")

    @staticmethod
    def _crop_stamp_ns(msg) -> int:
        """Observation time of a crop, falling back to the node clock when the header is absent."""
        stamp = getattr(msg, 'header', None)
        sec = getattr(stamp, 'sec', 0) if stamp is not None else 0
        nsec = getattr(stamp, 'nanosec', 0) if stamp is not None else 0
        if sec or nsec:
            return int(sec) * 1_000_000_000 + int(nsec)
        return 0

    def _on_target_crop(self, msg) -> None:
        """Extract, smooth and publish the current owner's reference signature for one crop."""
        stamp_ns = self._crop_stamp_ns(msg)
        try:
            crop = image_to_rgb(msg)
        except ValueError as exc:
            self.signature_refusal_count += 1
            self.get_logger().warning(f"[EXP07] target crop rejected: {exc}")
            return

        try:
            if self._cam_label is not None and self._extractor_accepts_cam_label:
                emb = self.extractor.extract_embedding(crop, cam_label=self._cam_label)
            else:
                emb = self.extractor.extract_embedding(crop)
        except Exception as exc:
            self.signature_refusal_count += 1
            self.get_logger().error(
                f"[EXP07] extractor failed on the target crop: {type(exc).__name__}: {exc}")
            self._publish_signature(None, stamp_ns, valid=False)
            return

        try:
            smoothed = self.smoother.update(np.asarray(emb, dtype=np.float32).reshape(-1))
        except ValueError as exc:
            self.signature_refusal_count += 1
            self.get_logger().error(f"[EXP07] extractor returned an unusable embedding: {exc}")
            self._publish_signature(None, stamp_ns, valid=False)
            return

        guard = is_invalid_embedding(smoothed)
        if guard is not None:
            # The smoother flushes to its zeros sentinel after STREAK_FLUSH_THRESHOLD invalid frames.
            # Saying so out loud is what lets a consumer drop a signature it already holds instead of
            # going on handing over a target it stopped being able to see.
            self._publish_signature(None, stamp_ns, valid=False)
            self.get_logger().warning(f"[EXP07] no signature for this crop: {guard}")
            return

        self.set_live_target_vector(smoothed)
        self._publish_signature(smoothed, stamp_ns, valid=True)

    def _publish_signature(self, vector: Optional[np.ndarray], stamp_ns: int, valid: bool,
                           quality: float = 0.0) -> None:
        if self.pub_signature is None or self._reid_signature_type is None:
            return
        msg = self._reid_signature_type()
        msg.sender_id = self.robot_id
        msg.target_id = self._signature_target_id
        msg.valid = bool(valid)
        values = [float(x) for x in vector] if valid and vector is not None else [0.0] * 512
        msg.vector = values
        msg.quality = float(quality)
        # Every call site passes the default 0.0, because nothing in Exp-7 measures a quality today.
        # That is the documented "unavailable" value in ReidSignature.msg, not a score of zero: the
        # smoother's frame count and the crop area are not confidence measures, so publishing one
        # would invent evidence a receiver cannot interpret.
        msg.stamp_utc_ns = int(stamp_ns or self.get_clock().now().nanoseconds)
        msg.model_id = self.reid_model_id
        self.pub_signature.publish(msg)
        self.signature_publish_count += 1
        self._last_signature_ns = int(msg.stamp_utc_ns)

    # ── Cache Management (ISSUE-12) ──

    def store_candidate_vector(self, vector: np.ndarray,
                               transaction_id: str = "") -> None:
        """Store reference candidate vector received from HandoffRequest (Robot A).

        Args:
            vector: 512-dim feature vector from Robot A.
            transaction_id: Transaction ID for scoped storage. Empty string uses legacy slot.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if transaction_id:
            self._candidate_cache[transaction_id] = _CacheEntry(vec)
            self.get_logger().debug(
                f"Candidate vector cached for txn '{transaction_id}'"
            )
        else:
            self._stored_candidate_vector = vec
            self.get_logger().info("Candidate feature vector stored (legacy slot).")

    def _resolve_cached_vector(self, transaction_id: str) -> Optional[np.ndarray]:
        """Resolve cached reference vector for a transaction, respecting TTL.

        Args:
            transaction_id: The transaction ID to look up.

        Returns:
            Cached vector if found and within TTL, else None.
        """
        # Purge expired entries opportunistically
        self._purge_expired_cache()

        if transaction_id and transaction_id in self._candidate_cache:
            entry = self._candidate_cache[transaction_id]
            if (time.monotonic() - entry.timestamp) <= self.CACHE_TTL_S:
                return entry.vector.copy()
            else:
                del self._candidate_cache[transaction_id]
                return None

        # Legacy single-slot fallback
        return self._stored_candidate_vector.copy() if self._stored_candidate_vector is not None else None

    def _purge_expired_cache(self) -> None:
        """Remove all cache entries older than CACHE_TTL_S."""
        now = time.monotonic()
        expired = [k for k, v in self._candidate_cache.items()
                   if (now - v.timestamp) > self.CACHE_TTL_S]
        for k in expired:
            del self._candidate_cache[k]

    def _on_handoff_confirm(self, msg) -> None:
        """Event purge: remove cache entry on handoff confirmation."""
        txn_id = ""
        if hasattr(msg, 'header') and hasattr(msg.header, 'transaction_id'):
            txn_id = msg.header.transaction_id
        elif hasattr(msg, 'transaction_id'):
            txn_id = msg.transaction_id
        if txn_id and txn_id in self._candidate_cache:
            del self._candidate_cache[txn_id]
            self.get_logger().debug(f"Cache purged on confirm: txn '{txn_id}'")

    def _on_handoff_cancel(self, msg) -> None:
        """Event purge: remove cache entry on handoff cancellation."""
        txn_id = ""
        if hasattr(msg, 'header') and hasattr(msg.header, 'transaction_id'):
            txn_id = msg.header.transaction_id
        elif hasattr(msg, 'transaction_id'):
            txn_id = msg.transaction_id
        if txn_id and txn_id in self._candidate_cache:
            del self._candidate_cache[txn_id]
            self.get_logger().debug(f"Cache purged on cancel: txn '{txn_id}'")

    # ── Live Target Vector ──

    def set_live_target_vector(self, vector: np.ndarray) -> None:
        """Set live target vector from local receiver tracking/perception."""
        self._live_target_vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    # ── Static Cosine Similarity (retained for backward compatibility with tests) ──

    @staticmethod
    def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Compute exact cosine similarity with NumPy 2.x safe scalar conversion.

        Cosine similarity = (u . v) / (||u|| * ||v||)
        """
        u_flat = np.asarray(u, dtype=np.float32).reshape(-1)
        v_flat = np.asarray(v, dtype=np.float32).reshape(-1)

        norm_u = float(np.linalg.norm(u_flat).item())
        norm_v = float(np.linalg.norm(v_flat).item())

        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0

        dot_val = float(np.dot(u_flat, v_flat).item())
        cos_sim = dot_val / (norm_u * norm_v)
        return float(np.clip(cos_sim, -1.0, 1.0))

    # ── Service Handler (ISSUE-02: routes through IdentityVerifier) ──

    def _handle_identity_verify(self, request: IdentityVerify.Request,
                                response: IdentityVerify.Response) -> IdentityVerify.Response:
        """Handle Candidate Re-ID verification service request.

        ISSUE-02: This handler now routes ALL verification through
        self.verifier.verify_candidate(), which enforces:
          - np.all(np.isfinite()) pre-filter (ISSUE-01)
          - Cosine similarity threshold comparison
          - Optional spatial Mahalanobis gating (FR-53)
        """
        txn_id = getattr(request, 'transaction_id', '')

        # ── 1. Resolve Reference Vector (from Robot A's HandoffRequest) ──
        ref_arr = np.asarray(request.reference_feature_vector, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(ref_arr).item()) > 1e-6:
            reference_vec = ref_arr
            reference_source = 'request_vector'
        else:
            # Fallback to transaction-scoped cache or legacy slot
            cached = self._resolve_cached_vector(txn_id)
            if cached is not None:
                reference_vec = cached
                reference_source = 'cache'
            else:
                reference_vec = np.zeros(512, dtype=np.float32)
                reference_source = 'none'

        # ── 2. Resolve Candidate Vector (from local camera crop or direct vector) ──
        candidate_vec = None
        candidate_source = 'none'
        invalid_reason = None

        if hasattr(request, 'candidate_feature_vector'):
            cand_arr = np.asarray(request.candidate_feature_vector, dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(cand_arr).item()) > 1e-6:
                candidate_vec = cand_arr
                candidate_source = 'request_vector'

        if candidate_vec is None and hasattr(request, 'candidate_crop') and len(request.candidate_crop.data) > 0:
            # TR-04: explicit encoding handling instead of a blind HxWx3 reshape.
            try:
                crop_np = image_to_rgb(request.candidate_crop)
            except ValueError as e:
                invalid_reason = f"INVALID_INPUT: {e}"
                self.get_logger().error(f"Rejected candidate crop for txn={txn_id}: {e}")
            else:
                # TR-03: pass the configured camera id to extractors that support SIE.
                if self._cam_label is not None and self._extractor_accepts_cam_label:
                    emb = self.extractor.extract_embedding(crop_np, cam_label=self._cam_label)
                else:
                    emb = self.extractor.extract_embedding(crop_np)
                candidate_vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                candidate_source = 'candidate_crop'

        if candidate_vec is None and self._live_target_vector is not None:
            candidate_vec = self._live_target_vector.copy()
            candidate_source = 'live_target'

        if candidate_vec is None:
            candidate_vec = np.zeros(512, dtype=np.float32)
            invalid_reason = invalid_reason or "INVALID_INPUT: no candidate image or vector supplied"

        # ── 2b. TR-02: an invalid embedding must not be scored as a low-similarity match ──
        if invalid_reason is None:
            ref_guard = is_invalid_embedding(reference_vec)
            if ref_guard is not None:
                invalid_reason = f"INVALID_EMBEDDING: reference: {ref_guard}"
        if invalid_reason is None:
            guard = is_invalid_embedding(candidate_vec)
            if guard is not None:
                invalid_reason = f"INVALID_EMBEDDING: candidate: {guard}"

        if invalid_reason is not None:
            self._invalid_frame_count += 1
            response.match = False
            response.similarity = 0.0
            if hasattr(response, 'match_confirmed'):
                response.match_confirmed = False
            if hasattr(response, 'similarity_score'):
                response.similarity_score = 0.0
            response.message = (
                f"{invalid_reason} "
                f"(candidate_source={candidate_source}, ref_source={reference_source}, txn={txn_id})"
            )
            self.get_logger().warning(
                f"Identity verification refused without scoring: {response.message}"
            )
            return response

        # ── 3. Route through IdentityVerifier (ISSUE-02) ──
        import time as _time
        _t0 = _time.monotonic()
        match_confirmed, similarity_score, reason = self.verifier.verify_candidate(
            candidate_embedding=candidate_vec,
            reference_embedding=reference_vec,
        )
        dt_id_ms = (_time.monotonic() - _t0) * 1000.0

        # ── 4. Reid Quality Metrics (Phase C Observability) ──
        cand_norm = float(np.linalg.norm(candidate_vec).item())
        ref_norm = float(np.linalg.norm(reference_vec).item())
        cand_finite = bool(np.all(np.isfinite(candidate_vec)))
        ref_finite = bool(np.all(np.isfinite(reference_vec)))
        smoother_streak = getattr(self.smoother, '_zero_streak', -1)

        # ── 5. Pack Response ──
        response.match = match_confirmed
        response.similarity = float(similarity_score)
        if hasattr(response, 'match_confirmed'):
            response.match_confirmed = match_confirmed
        if hasattr(response, 'similarity_score'):
            response.similarity_score = float(similarity_score)
        response.message = reason

        if match_confirmed:
            self.get_logger().info(
                f"Identity MATCH CONFIRMED (txn={txn_id}, "
                f"score={similarity_score:.4f}, dt_id_ms={dt_id_ms:.1f}, "
                f"cand_norm={cand_norm:.4f}, ref_norm={ref_norm:.4f}, "
                f"smoother_streak={smoother_streak})"
            )
        else:
            self.get_logger().warn(
                f"Identity MISMATCH (txn={txn_id}, "
                f"score={similarity_score:.4f}, dt_id_ms={dt_id_ms:.1f}, "
                f"reason='{reason}', cand_finite={cand_finite}, ref_finite={ref_finite}, "
                f"smoother_streak={smoother_streak})"
            )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = IdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
