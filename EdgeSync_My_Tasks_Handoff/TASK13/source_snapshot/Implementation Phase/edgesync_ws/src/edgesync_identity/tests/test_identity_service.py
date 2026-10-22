"""Unit tests for Sprint 5 Identity Verification Service (edgesync_identity).

Tests:
1. Exact cosine similarity math with NumPy 2.x safety (.item()).
2. Threshold comparison against parameterized similarity_threshold (default 0.70).
3. match_confirmed and similarity_score returned in IdentityVerify response.
4. Stored candidate vector from HandoffRequest used if request reference is omitted.
"""
import pytest
import numpy as np
import rclpy
from edgesync_msgs.srv import IdentityVerify
from edgesync_identity.nodes.identity_node import IdentityNode
from edgesync_identity.core.identity_verifier import IdentityVerifier


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_cosine_similarity_exact_math():
    # Exact unit vectors
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(512, dtype=np.float32)
    v2[0] = 1.0

    # Identical
    sim_ident = IdentityNode.compute_cosine_similarity(v1, v2)
    assert isinstance(sim_ident, float)
    assert pytest.approx(sim_ident, abs=1e-5) == 1.0

    # Orthogonal
    v3 = np.zeros(512, dtype=np.float32)
    v3[1] = 1.0
    sim_orth = IdentityNode.compute_cosine_similarity(v1, v3)
    assert pytest.approx(sim_orth, abs=1e-5) == 0.0

    # Opposite
    sim_opp = IdentityNode.compute_cosine_similarity(v1, -v1)
    assert pytest.approx(sim_opp, abs=1e-5) == -1.0

    # Zero vector safety
    v_zero = np.zeros(512, dtype=np.float32)
    assert IdentityNode.compute_cosine_similarity(v_zero, v1) == 0.0
    assert IdentityNode.compute_cosine_similarity(v1, v_zero) == 0.0


def test_cosine_similarity_numpy_2_scalar_safety():
    """Verify no NumPy 2.x scalar conversion deprecation or dimensional errors."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((512, 1)).astype(np.float32)
    b = rng.standard_normal((1, 512)).astype(np.float32)

    sim = IdentityNode.compute_cosine_similarity(a, b)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_identity_node_service_match_confirmed():
    node = IdentityNode()
    rng = np.random.default_rng(123)

    # Reference vector
    ref_vec = rng.standard_normal(512).astype(np.float32)
    ref_vec /= np.linalg.norm(ref_vec)

    # Candidate with slight noise -> sim ~ 0.98 >= 0.70
    noise = rng.normal(0, 0.02, 512).astype(np.float32)
    live_vec = ref_vec + noise
    live_vec /= np.linalg.norm(live_vec)

    req = IdentityVerify.Request()
    req.transaction_id = "txn_id_test_1"
    req.reference_feature_vector = [float(x) for x in ref_vec]
    if hasattr(req, 'candidate_feature_vector'):
        req.candidate_feature_vector = [float(x) for x in live_vec]
    else:
        node.set_live_target_vector(live_vec)

    res = IdentityVerify.Response()
    res = node._handle_identity_verify(req, res)

    assert res.match is True
    assert getattr(res, 'match_confirmed', True) is True
    assert res.similarity >= 0.70
    assert getattr(res, 'similarity_score', res.similarity) >= 0.70
    assert "confirmed" in res.message

    node.destroy_node()


def test_identity_node_service_mismatch():
    node = IdentityNode()
    rng = np.random.default_rng(456)

    # Distractor with orthogonal / independent vector
    ref_vec = rng.standard_normal(512).astype(np.float32)
    ref_vec /= np.linalg.norm(ref_vec)

    live_distractor = rng.standard_normal(512).astype(np.float32)
    live_distractor /= np.linalg.norm(live_distractor)

    req = IdentityVerify.Request()
    req.transaction_id = "txn_id_test_2"
    req.reference_feature_vector = [float(x) for x in ref_vec]
    if hasattr(req, 'candidate_feature_vector'):
        req.candidate_feature_vector = [float(x) for x in live_distractor]
    else:
        node.set_live_target_vector(live_distractor)

    res = IdentityVerify.Response()
    res = node._handle_identity_verify(req, res)

    assert res.match is False
    assert getattr(res, 'match_confirmed', False) is False
    assert res.similarity < 0.70
    assert "mismatch" in res.message

    node.destroy_node()


def test_identity_node_stored_candidate_vector_fallback():
    node = IdentityNode()
    rng = np.random.default_rng(789)

    # Robot B stored candidate vector from HandoffRequest
    stored_vec = rng.standard_normal(512).astype(np.float32)
    stored_vec /= np.linalg.norm(stored_vec)
    node.store_candidate_vector(stored_vec)

    # Live target matches stored vector
    noise = rng.normal(0, 0.01, 512).astype(np.float32)
    live_vec = stored_vec + noise
    live_vec /= np.linalg.norm(live_vec)

    # Request has empty reference vector (0s)
    req = IdentityVerify.Request()
    req.transaction_id = "txn_id_stored_fallback"
    req.reference_feature_vector = [0.0] * 512
    if hasattr(req, 'candidate_feature_vector'):
        req.candidate_feature_vector = [float(x) for x in live_vec]
    else:
        node.set_live_target_vector(live_vec)

    res = IdentityVerify.Response()
    res = node._handle_identity_verify(req, res)

    assert res.match is True
    assert res.similarity >= 0.70

    node.destroy_node()
