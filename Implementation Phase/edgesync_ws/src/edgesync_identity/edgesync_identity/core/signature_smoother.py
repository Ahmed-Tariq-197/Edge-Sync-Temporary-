"""Exponential Moving Average (EMA) buffer for continuous target Re-ID signature smoothing.

Math/Protocol Spec v2.3 §5.1 & WP-5:
- Updates signature at ~2 Hz: f_target(t) = alpha * f_new + (1 - alpha) * f_target(t-1)
- Unit L2-normalization enforced at every update step.
- Ensures pre-warmed, resident signature is immediately available during handoff window.
"""
from typing import Optional
import numpy as np


class SignatureSmoother:
    """Maintains an EMA smoothed 512-dim Re-ID embedding buffer."""

    def __init__(self, alpha: float = 0.20, embedding_dim: int = 512):
        """Initialize smoother.

        Args:
            alpha: Weight for newly observed embedding (default 0.20).
            embedding_dim: Dimensionality of feature vector (OSNet = 512).
        """
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.embedding_dim = embedding_dim
        self._smoothed_signature: Optional[np.ndarray] = None
        self._update_count: int = 0

    def update(self, new_embedding: np.ndarray) -> np.ndarray:
        """Incorporate new embedding into smoothed signature."""
        emb = np.asarray(new_embedding, dtype=np.float32).flatten()
        if emb.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {emb.shape[0]}")

        # Ensure input is L2-normalized
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            return self.get_signature() if self._smoothed_signature is not None else np.zeros(self.embedding_dim, dtype=np.float32)

        if self._smoothed_signature is None:
            self._smoothed_signature = emb.copy()
        else:
            # Exponential Moving Average: f_target = alpha * f_new + (1 - alpha) * f_prev
            smoothed = self.alpha * emb + (1.0 - self.alpha) * self._smoothed_signature
            # Re-normalize to unit length
            self._smoothed_signature = smoothed / np.linalg.norm(smoothed)

        self._update_count += 1
        return self._smoothed_signature.copy()

    def get_signature(self) -> Optional[np.ndarray]:
        """Return the current smoothed signature or None if uninitialized."""
        if self._smoothed_signature is None:
            return None
        return self._smoothed_signature.copy()

    def reset(self):
        """Reset buffer state."""
        self._smoothed_signature = None
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count
