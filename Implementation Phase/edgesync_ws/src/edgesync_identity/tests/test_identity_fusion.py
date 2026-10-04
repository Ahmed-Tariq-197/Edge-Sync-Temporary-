"""Unit tests for Re-ID identity verification and multi-cue fusion (Quality Gate 5).

Verifies:
1. Cosine similarity numerical properties (range [-1, 1], unit norm invariance).
2. Signature smoother EMA update and unit norm preservation.
3. Multi-cue verification: rejection of appearance distractors (< 0.70).
4. Spatial Mahalanobis gating rejection of out-of-bounds candidates.
"""
import pytest
import numpy as np
from edgesync_identity.core.signature_smoother import SignatureSmoother
from edgesync_identity.core.identity_verifier import IdentityVerifier


def test_cosine_similarity_properties():
    # Identical vectors -> similarity = 1.0
    v1 = np.random.randn(512).astype(np.float32)
    assert pytest.approx(IdentityVerifier.compute_cosine_similarity(v1, v1), abs=1e-5) == 1.0

    # Orthogonal vectors -> similarity = 0.0
    v_a = np.zeros(512, dtype=np.float32)
    v_b = np.zeros(512, dtype=np.float32)
    v_a[0] = 1.0
    v_b[1] = 1.0
    assert pytest.approx(IdentityVerifier.compute_cosine_similarity(v_a, v_b), abs=1e-5) == 0.0

    # Opposite vectors -> similarity = -1.0
    assert pytest.approx(IdentityVerifier.compute_cosine_similarity(v1, -v1), abs=1e-5) == -1.0


def test_signature_smoother_ema_and_normalization():
    smoother = SignatureSmoother(alpha=0.20, embedding_dim=512)

    v1 = np.random.randn(512).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    sig1 = smoother.update(v1)
    assert sig1.shape == (512,)
    assert pytest.approx(np.linalg.norm(sig1), abs=1e-5) == 1.0
    assert np.allclose(sig1, v1, atol=1e-5)  # First update equals v1

    # Second update
    v2 = np.random.randn(512).astype(np.float32)
    v2 = v2 / np.linalg.norm(v2)

    sig2 = smoother.update(v2)
    assert pytest.approx(np.linalg.norm(sig2), abs=1e-5) == 1.0
    assert smoother.update_count == 2


def test_identity_verifier_distractor_rejection():
    verifier = IdentityVerifier(similarity_threshold=0.70)

    target_ref = np.random.randn(512).astype(np.float32)
    target_ref = target_ref / np.linalg.norm(target_ref)

    # Distractor with low similarity (~0.1)
    distractor = np.random.randn(512).astype(np.float32)
    distractor = distractor / np.linalg.norm(distractor)

    matched, sim, reason = verifier.verify_candidate(distractor, target_ref)
    assert matched is False
    assert sim < 0.70
    assert "below threshold" in reason


def test_identity_verifier_true_match():
    verifier = IdentityVerifier(similarity_threshold=0.70)

    target_ref = np.random.randn(512).astype(np.float32)
    target_ref = target_ref / np.linalg.norm(target_ref)

    # Slightly perturbed query (cosine sim ~ 0.95)
    noise = np.random.normal(0, 0.1, 512).astype(np.float32)
    query = target_ref + noise
    query = query / np.linalg.norm(query)

    matched, sim, reason = verifier.verify_candidate(query, target_ref)
    assert matched is True
    assert sim >= 0.70
    assert "confirmed" in reason


def test_spatial_mahalanobis_gating_rejection():
    verifier = IdentityVerifier(similarity_threshold=0.70, mahalanobis_gate_threshold=9.21, enable_spatial_gating=True)

    target_ref = np.random.randn(512).astype(np.float32)
    target_ref = target_ref / np.linalg.norm(target_ref)

    # Appearance matches perfectly
    query = target_ref.copy()

    # But spatial position is 3 meters away from predicted target position
    pos_prior = np.array([1.0, 1.0, 0.0])
    pos_candidate = np.array([4.0, 1.0, 0.0])  # 3m discrepancy

    cov_prior = np.eye(3) * 0.05
    cov_cand = np.eye(3) * 0.05

    matched, sim, reason = verifier.verify_candidate(
        candidate_embedding=query,
        reference_embedding=target_ref,
        pos_candidate_map=pos_candidate,
        pos_prior_map=pos_prior,
        cov_prior_map=cov_prior,
        cov_candidate_map=cov_cand
    )

    assert matched is False
    assert "Spatial gate failed" in reason
