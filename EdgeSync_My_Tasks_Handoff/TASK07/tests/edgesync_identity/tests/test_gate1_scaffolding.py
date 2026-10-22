"""Gate 1 Unit Tests: Smoother Streak/Flush, NaN/Inf Rejection, Factory Fail-Fast.

ES-EXP07-DCP-REV10 — Step 4 (Tests):
  UNIT-06: Smoother streak tracking and frame-3 flush behavior.
  UNIT-07: IdentityVerifier NaN/Inf rejection guard (ISSUE-01).
  UNIT-08: ExtractorFactory fail-fast on missing assets (ISSUE-07, ISSUE-08).
  UNIT-09: TransReID shape + norm contract (deferred to Gate 2 — placeholder).

These tests do NOT require ROS 2, CUDA, or model weights. They execute on any
Python 3.10+ host with NumPy and pytest.
"""
import os
import tempfile
import pytest
import numpy as np

from edgesync_identity.core.signature_smoother import SignatureSmoother
from edgesync_identity.core.identity_verifier import IdentityVerifier
from edgesync_identity.models.extractor_factory import create_extractor
from edgesync_identity.interfaces.base_identity_extractor import BaseIdentityExtractor


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT-06: Smoother Streak Tracking and Frame-3 Flush
# Traces: ISSUE-03, ISSUE-04 (smoother timeline harmonization)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmootherStreakFlush:
    """UNIT-06: Verify EMA is decoupled from health; streak triggers flush at frame 3."""

    def _make_valid_embedding(self, seed: int = 0) -> np.ndarray:
        """Helper: generate a random unit-norm float32[512] vector."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_valid_embedding_resets_streak(self):
        """Valid embeddings must reset zero_streak_count to 0."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=1)

        result = smoother.update(v1)
        assert result.shape == (512,)
        assert pytest.approx(np.linalg.norm(result), abs=1e-5) == 1.0
        assert smoother.zero_streak_count == 0

    def test_single_zero_frame_holds_previous(self):
        """First invalid frame (streak=1) holds previous valid signature."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=10)
        sig_valid = smoother.update(v1)

        # Feed zero vector (invalid)
        sig_held = smoother.update(np.zeros(512, dtype=np.float32))

        assert smoother.zero_streak_count == 1
        assert np.allclose(sig_held, sig_valid, atol=1e-5)
        assert smoother.get_signature() is not None  # Buffer NOT flushed

    def test_two_zero_frames_holds_previous(self):
        """Second consecutive invalid frame (streak=2) still holds previous."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=20)
        sig_valid = smoother.update(v1)

        smoother.update(np.zeros(512, dtype=np.float32))  # streak=1
        sig_held = smoother.update(np.zeros(512, dtype=np.float32))  # streak=2

        assert smoother.zero_streak_count == 2
        assert np.allclose(sig_held, sig_valid, atol=1e-5)
        assert smoother.get_signature() is not None  # Still holding

    def test_three_zero_frames_flushes_buffer(self):
        """Third consecutive invalid frame (streak=3, t=1.5s) MUST flush buffer."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=30)
        smoother.update(v1)

        smoother.update(np.zeros(512, dtype=np.float32))  # streak=1
        smoother.update(np.zeros(512, dtype=np.float32))  # streak=2
        sig_flushed = smoother.update(np.zeros(512, dtype=np.float32))  # streak=3

        assert smoother.zero_streak_count == 3
        assert np.allclose(sig_flushed, np.zeros(512), atol=1e-8)
        assert smoother.get_signature() is None  # Buffer flushed to None
        assert smoother.is_flushed is True

    def test_valid_frame_after_flush_recovers(self):
        """Valid embedding after flush re-initializes the smoother."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=40)
        smoother.update(v1)

        # Flush
        for _ in range(3):
            smoother.update(np.zeros(512, dtype=np.float32))
        assert smoother.is_flushed is True

        # Recovery
        v2 = self._make_valid_embedding(seed=41)
        sig_recovered = smoother.update(v2)

        assert smoother.zero_streak_count == 0
        assert pytest.approx(np.linalg.norm(sig_recovered), abs=1e-5) == 1.0
        assert smoother.get_signature() is not None

    def test_ema_never_blends_with_zeros(self):
        """Key invariant: EMA math MUST NOT blend zeros into the running average.

        f_target(t) ≠ alpha * 0 + (1-alpha) * f_prev
        After invalid frames (within hold window), the signature should be
        exactly the previous valid signature, NOT a diluted version.
        """
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=50)
        sig_after_v1 = smoother.update(v1)

        # Feed one zero frame (streak=1, held)
        sig_after_zero = smoother.update(np.zeros(512, dtype=np.float32))

        # The held signature must be IDENTICAL to the last valid, not blended
        assert np.allclose(sig_after_zero, sig_after_v1, atol=1e-6)

    def test_nan_input_increments_streak(self):
        """NaN embedding is treated as invalid and increments streak."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=60)
        smoother.update(v1)

        nan_vec = np.full(512, np.nan, dtype=np.float32)
        smoother.update(nan_vec)
        assert smoother.zero_streak_count == 1

    def test_inf_input_increments_streak(self):
        """Inf embedding is treated as invalid and increments streak."""
        smoother = SignatureSmoother(alpha=0.20)
        v1 = self._make_valid_embedding(seed=70)
        smoother.update(v1)

        inf_vec = np.full(512, np.inf, dtype=np.float32)
        smoother.update(inf_vec)
        assert smoother.zero_streak_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT-07: IdentityVerifier NaN/Inf Rejection Guard
# Traces: ISSUE-01 (P0 safety — nan < 0.70 → False bypass)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifierNanInfRejection:
    """UNIT-07: NaN/Inf embeddings MUST be rejected with (False, 0.0, reason)."""

    def _make_valid_embedding(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_nan_candidate_rejected(self):
        """NaN candidate embedding MUST be rejected, not falsely confirmed."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        ref = self._make_valid_embedding(seed=100)
        nan_cand = np.full(512, np.nan, dtype=np.float32)

        matched, sim, reason = verifier.verify_candidate(nan_cand, ref)
        assert matched is False
        assert sim == 0.0
        assert "non-finite" in reason.lower() or "nan" in reason.lower()

    def test_nan_reference_rejected(self):
        """NaN reference embedding MUST be rejected."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        cand = self._make_valid_embedding(seed=101)
        nan_ref = np.full(512, np.nan, dtype=np.float32)

        matched, sim, reason = verifier.verify_candidate(cand, nan_ref)
        assert matched is False
        assert sim == 0.0
        assert "non-finite" in reason.lower() or "nan" in reason.lower()

    def test_inf_candidate_rejected(self):
        """Inf candidate embedding MUST be rejected."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        ref = self._make_valid_embedding(seed=102)
        inf_cand = np.full(512, np.inf, dtype=np.float32)

        matched, sim, reason = verifier.verify_candidate(inf_cand, ref)
        assert matched is False
        assert sim == 0.0

    def test_neg_inf_candidate_rejected(self):
        """-Inf candidate embedding MUST be rejected."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        ref = self._make_valid_embedding(seed=103)
        neg_inf = np.full(512, -np.inf, dtype=np.float32)

        matched, sim, reason = verifier.verify_candidate(neg_inf, ref)
        assert matched is False
        assert sim == 0.0

    def test_mixed_nan_in_candidate_rejected(self):
        """Even a single NaN element in an otherwise valid vector MUST trigger rejection."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        ref = self._make_valid_embedding(seed=104)
        cand = self._make_valid_embedding(seed=105)
        cand[256] = np.nan  # Inject single NaN

        matched, sim, reason = verifier.verify_candidate(cand, ref)
        assert matched is False
        assert sim == 0.0

    def test_valid_vectors_still_work(self):
        """Sanity: valid unit-norm vectors that match should still be confirmed."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        ref = self._make_valid_embedding(seed=110)
        # Near-identical query
        noise = np.random.default_rng(111).normal(0, 0.01, 512).astype(np.float32)
        query = ref + noise
        query /= np.linalg.norm(query)

        matched, sim, reason = verifier.verify_candidate(query, ref)
        assert matched is True
        assert sim >= 0.70
        assert "confirmed" in reason.lower()

    def test_zeros_vectors_rejected_below_threshold(self):
        """Two zero vectors → similarity 0.0 → below threshold → rejected (not NaN)."""
        verifier = IdentityVerifier(similarity_threshold=0.70)
        z = np.zeros(512, dtype=np.float32)

        matched, sim, reason = verifier.verify_candidate(z, z)
        assert matched is False
        # Zeros are finite, so they pass the finite guard,
        # but cosine sim of zeros → 0.0 < 0.70 → rejected on threshold.
        assert sim == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT-08: ExtractorFactory Fail-Fast
# Traces: ISSUE-07 (benchmark guard), ISSUE-08 (fail-fast on missing assets)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractorFactoryFailFast:
    """UNIT-08: Factory must fail-fast with FileNotFoundError on missing assets."""

    def test_osnet_mock_creates_successfully(self):
        """osnet_mock requires no external files and should always succeed."""
        extractor = create_extractor("osnet_mock")
        assert isinstance(extractor, BaseIdentityExtractor)

    def test_osnet_mock_produces_512d_output(self):
        """osnet_mock extractor must produce valid float32[512] output."""
        extractor = create_extractor("osnet_mock")
        # Feed a small synthetic crop
        crop = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
        emb = extractor.extract_embedding(crop)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32

    def test_osnet_trt_missing_engine_raises(self):
        """osnet_trt with non-existent engine path MUST raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="osnet_trt"):
            create_extractor("osnet_trt", engine_path="/nonexistent/path.engine")

    def test_osnet_trt_empty_engine_raises(self):
        """osnet_trt with empty engine path MUST raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="osnet_trt"):
            create_extractor("osnet_trt", engine_path="")

    def test_transreid_missing_checkpoint_raises(self):
        """transreid with missing checkpoint MUST raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="transreid.*checkpoint"):
            create_extractor("transreid", checkpoint_path="/nonexistent/model.pth",
                           projection_path="/nonexistent/proj.npz")

    def test_transreid_missing_projection_raises(self):
        """transreid with valid checkpoint but missing projection MUST raise FileNotFoundError."""
        # Create a temporary file to act as a valid checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"fake_checkpoint_data")
            ckpt_path = f.name
        try:
            with pytest.raises(FileNotFoundError, match="transreid.*projection"):
                create_extractor("transreid",
                               checkpoint_path=ckpt_path,
                               projection_path="/nonexistent/proj.npz")
        finally:
            os.unlink(ckpt_path)

    def test_unknown_model_type_raises_valueerror(self):
        """Unknown model_type MUST raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_extractor("invalid_model_name")

    def test_factory_case_insensitive(self):
        """Factory should handle case-insensitive model_type input."""
        extractor = create_extractor("OSNET_MOCK")
        assert isinstance(extractor, BaseIdentityExtractor)

    def test_factory_strips_whitespace(self):
        """Factory should strip leading/trailing whitespace from model_type."""
        extractor = create_extractor("  osnet_mock  ")
        assert isinstance(extractor, BaseIdentityExtractor)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT-09: TransReID Shape + Norm (Gate 2 — requires synthetic or real fixtures)
# Traces: Section 9.1 mathematical pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_transreid_assets():
    """Resolve TransReID checkpoint and projection paths.

    Priority order:
      1. Environment variables TRANSREID_CHECKPOINT_PATH / TRANSREID_PROJECTION_PATH
      2. Auto-discovered fixtures in tests/fixtures/ (generated by generate_test_fixtures.py)

    Returns:
        (checkpoint_path, projection_path) or (None, None) if unavailable.
    """
    ckpt = os.environ.get("TRANSREID_CHECKPOINT_PATH", "")
    proj = os.environ.get("TRANSREID_PROJECTION_PATH", "")

    if ckpt and os.path.isfile(ckpt) and proj and os.path.isfile(proj):
        return ckpt, proj

    # Auto-discover from tests/fixtures/
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    auto_ckpt = os.path.join(fixtures_dir, 'synthetic_transreid.pth')
    auto_proj = os.path.join(fixtures_dir, 'synthetic_svd_768_to_512.npz')

    if os.path.isfile(auto_ckpt) and os.path.isfile(auto_proj):
        return auto_ckpt, auto_proj

    return None, None


@pytest.fixture(scope="session")
def transreid_extractor():
    """Session-scoped fixture: load TransReID extractor once for all UNIT-09 tests.

    Skips the entire test class if fixtures are not available.
    """
    ckpt, proj = _resolve_transreid_assets()
    if ckpt is None or proj is None:
        pytest.skip(
            "TransReID assets not available. "
            "Run: python scripts/generate_test_fixtures.py "
            "OR set TRANSREID_CHECKPOINT_PATH / TRANSREID_PROJECTION_PATH"
        )

    extractor = create_extractor("transreid",
                                  checkpoint_path=ckpt,
                                  projection_path=proj)
    return extractor


class TestTransReIDShapeNorm:
    """UNIT-09: TransReID extractor output shape, norm, and sentinel contract.

    These tests validate the full Section 9.1 mathematical pipeline:
      Resize → ImageNet Norm → ViT-Base/16 → BNNeck → Center → SVD Proj → L2 Norm

    Tests auto-discover fixtures from tests/fixtures/ (generated by
    generate_test_fixtures.py) or use TRANSREID_CHECKPOINT_PATH /
    TRANSREID_PROJECTION_PATH environment variables.
    """

    def test_transreid_output_shape_512(self, transreid_extractor):
        """TransReID extractor MUST produce exactly float32[512]."""
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        emb = transreid_extractor.extract_embedding(crop)
        assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"

    def test_transreid_output_unit_norm(self, transreid_extractor):
        """TransReID valid output MUST have ||v||_2 = 1.0 ± 1e-4."""
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        emb = transreid_extractor.extract_embedding(crop)
        norm = float(np.linalg.norm(emb).item())
        assert pytest.approx(norm, abs=1e-4) == 1.0, f"Norm {norm} not unit"

    def test_transreid_output_finite(self, transreid_extractor):
        """TransReID output MUST contain only finite values (ISSUE-01)."""
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        emb = transreid_extractor.extract_embedding(crop)
        assert np.all(np.isfinite(emb)), "Output contains NaN or Inf"

    def test_transreid_empty_crop_returns_zeros(self, transreid_extractor):
        """Empty crop → zeros(512) sentinel."""
        emb = transreid_extractor.extract_embedding(np.array([], dtype=np.uint8))
        assert np.allclose(emb, np.zeros(512)), "Empty crop should return zeros"

    def test_transreid_none_crop_returns_zeros(self, transreid_extractor):
        """None crop → zeros(512) sentinel."""
        emb = transreid_extractor.extract_embedding(None)
        assert np.allclose(emb, np.zeros(512)), "None crop should return zeros"

    def test_transreid_tiny_crop_returns_zeros(self, transreid_extractor):
        """Crop smaller than minimum (16x16) → zeros(512) sentinel."""
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        emb = transreid_extractor.extract_embedding(tiny)
        assert np.allclose(emb, np.zeros(512)), "Tiny crop should return zeros"

    def test_transreid_different_crops_different_embeddings(self, transreid_extractor):
        """Different image crops MUST produce different embeddings."""
        crop_a = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        crop_b = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
        emb_a = transreid_extractor.extract_embedding(crop_a)
        emb_b = transreid_extractor.extract_embedding(crop_b)
        # Both should be unit-norm
        assert abs(float(np.linalg.norm(emb_a)) - 1.0) < 1e-4
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-4
        # They should not be identical
        assert not np.allclose(emb_a, emb_b, atol=1e-3), \
            "Different crops produced identical embeddings"

    def test_transreid_reproducible_on_same_input(self, transreid_extractor):
        """Same input crop MUST produce the same embedding (deterministic)."""
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        emb1 = transreid_extractor.extract_embedding(crop)
        emb2 = transreid_extractor.extract_embedding(crop)
        assert np.allclose(emb1, emb2, atol=1e-5), "Same input produced different outputs"
