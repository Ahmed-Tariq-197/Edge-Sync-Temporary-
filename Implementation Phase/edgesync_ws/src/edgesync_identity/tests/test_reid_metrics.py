"""Unit tests for Re-ID evaluation metrics (Task 6)."""
import math

import numpy as np
import pytest

from edgesync_identity.ablation.ablation_config import AblationConfig, JUNK_IDENTITY
from edgesync_identity.ablation.reid_metrics import (
    NOT_EVALUATED, ReidEvaluator, SimilarityHistogram, wilson_interval,
)

LOW_HISTOGRAM_BINS = 100
TOLERANCE = 1e-6


def unit_vector_with_similarity(similarity: float) -> np.ndarray:
    """2-D unit vector whose cosine similarity to (1, 0) equals `similarity`."""
    angle = math.acos(similarity)
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def build_gallery(similarities, pids, cams):
    emb = np.stack([unit_vector_with_similarity(s) for s in similarities])
    return emb, np.array(pids), np.array(cams)


QUERY = np.array([[1.0, 0.0]], dtype=np.float32)


def test_rank1_and_average_precision_follow_market_protocol():
    # Same identity + same camera (excluded), junk (excluded), then hit, miss, hit.
    gallery, g_pid, g_cam = build_gallery(
        [0.99, 0.95, 0.90, 0.80, 0.50], [1, JUNK_IDENTITY, 1, 2, 1], [1, 2, 2, 2, 3])
    result = ReidEvaluator(AblationConfig(robot_a_cameras=(1,), robot_b_cameras=(2,))).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.num_queries == 1
    assert result.cmc[1] == 1.0
    assert result.mean_ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=TOLERANCE)
    assert result.top1_correct.tolist() == [1]


def test_rank1_miss_when_impostor_ranks_first():
    gallery, g_pid, g_cam = build_gallery([0.9, 0.6], [2, 1], [2, 2])
    result = ReidEvaluator(AblationConfig(robot_a_cameras=(1,), robot_b_cameras=(2,))).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.cmc[1] == 0.0
    assert result.cmc[5] == 1.0
    assert result.mean_ap == pytest.approx(0.5, abs=TOLERANCE)


def test_query_without_any_positive_is_skipped():
    gallery, g_pid, g_cam = build_gallery([0.9], [2], [2])
    result = ReidEvaluator(AblationConfig(robot_a_cameras=(1,), robot_b_cameras=(2,))).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.num_queries == 0
    assert result.top1_correct.tolist() == [NOT_EVALUATED]


def test_cross_robot_false_handoff_and_pair_rates():
    # Robot A query; robot B gallery: one genuine (0.90) and two impostors (0.80, 0.30).
    config = AblationConfig(similarity_threshold=0.70, robot_a_cameras=(1,), robot_b_cameras=(2,))
    gallery, g_pid, g_cam = build_gallery([0.90, 0.80, 0.30], [1, 2, 3], [2, 2, 2])
    result = ReidEvaluator(config).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.cross_robot_queries == 1
    assert result.false_handoff_count == 0
    assert result.correct_handoff_count == 1
    assert result.cross_robot_fmr == pytest.approx(0.5, abs=TOLERANCE)
    assert result.cross_robot_tar == pytest.approx(1.0, abs=TOLERANCE)

    # Impostor is now the best candidate above threshold: a false handoff.
    gallery, g_pid, g_cam = build_gallery([0.95, 0.60], [2, 1], [2, 2])
    result = ReidEvaluator(config).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.false_handoff_count == 1
    assert result.false_handoff.tolist() == [1]
    assert result.false_handoff_rate == 1.0


def test_best_wrong_candidate_below_threshold_is_not_a_false_handoff():
    config = AblationConfig(similarity_threshold=0.70, robot_a_cameras=(1,), robot_b_cameras=(2,))
    gallery, g_pid, g_cam = build_gallery([0.50, 0.40], [2, 1], [2, 2])
    result = ReidEvaluator(config).evaluate(
        QUERY, np.array([1]), np.array([1]), gallery, g_pid, g_cam)
    assert result.false_handoff.tolist() == [0]
    assert result.correct_handoff_count == 0


def test_query_from_unassigned_camera_skips_cross_robot_metrics():
    config = AblationConfig(robot_a_cameras=(1,), robot_b_cameras=(2,))
    gallery, g_pid, g_cam = build_gallery([0.9, 0.8], [1, 2], [2, 2])
    result = ReidEvaluator(config).evaluate(
        QUERY, np.array([1]), np.array([9]), gallery, g_pid, g_cam)
    assert result.cross_robot_queries == 0
    assert result.false_handoff.tolist() == [NOT_EVALUATED]


def test_histogram_rates_threshold_and_eer():
    hist = SimilarityHistogram(LOW_HISTOGRAM_BINS)
    hist.add(np.array([0.9, 0.8, 0.7, 0.2]), np.array([0.6, 0.1, -0.3, 0.75]))
    assert hist.true_accept_rate(0.5) == pytest.approx(0.75, abs=TOLERANCE)
    assert hist.false_match_rate(0.5) == pytest.approx(0.5, abs=TOLERANCE)
    threshold, fmr = hist.threshold_at_true_accept_rate(0.75)
    assert 0.2 < threshold <= 0.7
    assert 0.0 <= fmr <= 1.0
    eer, eer_threshold = hist.equal_error_rate()
    assert 0.0 <= eer <= 0.5
    assert -1.0 <= eer_threshold <= 1.0


def test_histogram_clips_float_overshoot_instead_of_dropping():
    hist = SimilarityHistogram(LOW_HISTOGRAM_BINS)
    hist.add(np.array([1.0000001]), np.array([-1.0000001]))
    assert hist.genuine.sum() == 1
    assert hist.impostor.sum() == 1


def test_wilson_interval_known_values():
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert high == pytest.approx(0.0713, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
