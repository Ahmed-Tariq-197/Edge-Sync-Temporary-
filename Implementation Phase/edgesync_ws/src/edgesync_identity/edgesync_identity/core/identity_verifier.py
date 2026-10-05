"""Identity verification and multi-cue fusion (Math/Protocol Spec v2.3 §1.2 & §5).

Combines:
1. Deep appearance cosine similarity (OSNet 512-dim embedding).
2. Spatial Mahalanobis gating against transferred target state prior.
"""
from typing import Tuple, Optional
import numpy as np


class IdentityVerifier:
    """Verifies candidate target identity using visual Re-ID and spatial multi-cue fusion."""

    def __init__(self,
                 similarity_threshold: float = 0.70,
                 mahalanobis_gate_threshold: float = 9.21,  # Chi-square 3-DoF at 99%
                 enable_spatial_gating: bool = True):
        """Initialize verifier.

        Args:
            similarity_threshold: Minimum cosine similarity required for appearance match (0.70).
            mahalanobis_gate_threshold: Squared Mahalanobis distance cutoff (9.21 for 99% confidence in 3D).
            enable_spatial_gating: True to enforce multi-cue spatial gating (FR-53).
        """
        self.similarity_threshold = similarity_threshold
        self.mahalanobis_gate_threshold = mahalanobis_gate_threshold
        self.enable_spatial_gating = enable_spatial_gating

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        a = np.asarray(vec_a, dtype=np.float32).flatten()
        b = np.asarray(vec_b, dtype=np.float32).flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0

        # Cosine similarity for unit vectors simplifies to dot product
        cos_sim = float(np.dot(a / norm_a, b / norm_b))
        return float(np.clip(cos_sim, -1.0, 1.0))

    @staticmethod
    def compute_mahalanobis_distance(pos_candidate: np.ndarray,
                                     pos_prior: np.ndarray,
                                     cov_combined: np.ndarray) -> float:
        """Calculate squared Mahalanobis distance: (p_c - p_0)^T * S^-1 * (p_c - p_0)."""
        diff = (pos_candidate - pos_prior).reshape((3, 1))
        # Ensure positive-definiteness with regularized inversion
        s_reg = cov_combined + np.eye(3) * 1e-5
        inv_s = np.linalg.inv(s_reg)
        d2 = float((diff.T @ inv_s @ diff).item())
        return max(0.0, d2)

    def verify_candidate(self,
                         candidate_embedding: np.ndarray,
                         reference_embedding: np.ndarray,
                         pos_candidate_map: Optional[np.ndarray] = None,
                         pos_prior_map: Optional[np.ndarray] = None,
                         cov_prior_map: Optional[np.ndarray] = None,
                         cov_candidate_map: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        """Execute full multi-cue verification on a candidate target.

        Args:
            candidate_embedding: 512-dim embedding from local camera crop.
            reference_embedding: 512-dim transferred embedding from Robot A.
            pos_candidate_map: Candidate 3D position in map frame [x, y, z].
            pos_prior_map: Predicted target 3D position in map frame from Robot A.
            cov_prior_map: Transferred target position covariance (3x3).
            cov_candidate_map: Candidate observation covariance (3x3).

        Returns:
            Tuple of (is_matched: bool, similarity: float, reason: str).
        """
        # 1. Appearance Cosine Similarity Check
        sim = self.compute_cosine_similarity(candidate_embedding, reference_embedding)
        if sim < self.similarity_threshold:
            return False, sim, f"Appearance similarity ({sim:.3f}) below threshold ({self.similarity_threshold:.2f})"

        # 2. Spatial Multi-Cue Gating Check (if coordinates provided and enabled)
        if self.enable_spatial_gating and (pos_candidate_map is not None and pos_prior_map is not None):
            if cov_prior_map is None:
                cov_prior_map = np.eye(3) * 0.25
            if cov_candidate_map is None:
                cov_candidate_map = np.eye(3) * 0.10

            s_matrix = cov_prior_map[:3, :3] + cov_candidate_map[:3, :3]
            d2 = self.compute_mahalanobis_distance(pos_candidate_map, pos_prior_map, s_matrix)

            if d2 > self.mahalanobis_gate_threshold:
                return False, sim, f"Spatial gate failed: Mahalanobis d^2 ({d2:.2f}) > cutoff ({self.mahalanobis_gate_threshold:.2f})"

        return True, sim, "Identity match confirmed"
