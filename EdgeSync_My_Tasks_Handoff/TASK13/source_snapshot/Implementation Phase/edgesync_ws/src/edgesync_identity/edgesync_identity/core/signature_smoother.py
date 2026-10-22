"""Exponential Moving Average (EMA) buffer for continuous target Re-ID signature smoothing.

Gate 1 — ES-EXP07-DCP-REV10, Section 11:
  Mathematical EMA state update is DECOUPLED from temporal health tracking.

  Timeline at 2 Hz (0.5 s / frame):
    - Valid embedding  → EMA update, streak resets to 0.
    - Invalid Frame 1 (t = 0.5 s) → streak = 1, HOLD previous signature.
    - Invalid Frame 2 (t = 1.0 s) → streak = 2, HOLD previous signature (max hold ≤ 1.0 s).
    - Invalid Frame 3 (t = 1.5 s) → streak = 3, FLUSH buffer to None, output zeros(512).

  Key invariants:
    - The mathematical EMA vector is NEVER blended with zeros:
      f_target(t) ≠ alpha * 0 + (1-alpha) * f_prev
    - Invalid inputs increment streak counter ONLY — they never enter EMA math.
    - Maximum hold window: 2 frames (1.0 s at 2 Hz).
    - Flush threshold: 3 consecutive invalid frames (1.5 s elapsed).

Traces: ISSUE-03, ISSUE-04 (smoother timeline harmonization).
"""
from typing import Optional
import numpy as np


class SignatureSmoother:
    """Maintains an EMA smoothed 512-dim Re-ID embedding buffer with health tracking."""

    # Maximum consecutive invalid frames before flush (3 frames = 1.5 s at 2 Hz)
    STREAK_FLUSH_THRESHOLD: int = 3
    # Maximum number of frames the smoother will hold a stale signature (2 frames = 1.0 s)
    MAX_HOLD_FRAMES: int = 2

    def __init__(self, alpha: float = 0.20, embedding_dim: int = 512):
        """Initialize smoother.

        Args:
            alpha: Weight for newly observed embedding (default 0.20).
            embedding_dim: Dimensionality of feature vector (512 for both OSNet and projected TransReID).
        """
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.embedding_dim = embedding_dim
        self._smoothed_signature: Optional[np.ndarray] = None
        self._update_count: int = 0
        # ISSUE-03 / ISSUE-04: Temporal health tracking — decoupled from EMA math
        self._zero_streak_count: int = 0

    def update(self, new_embedding: np.ndarray) -> np.ndarray:
        """Incorporate new embedding into smoothed signature with health tracking.

        Valid embedding (||emb||_2 > 1e-6, all finite):
            → EMA update, streak reset to 0.
        Invalid embedding (zero-norm, non-finite, wrong shape):
            → Streak incremented. If streak < FLUSH_THRESHOLD: hold previous.
              If streak >= FLUSH_THRESHOLD: flush buffer, return zeros.

        Args:
            new_embedding: 512-dim float32 embedding from extractor.

        Returns:
            np.ndarray: Current smoothed signature (unit-normalized) or zeros(512) sentinel.
        """
        emb = np.asarray(new_embedding, dtype=np.float32).flatten()
        if emb.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {emb.shape[0]}")

        # ── Determine if this frame is valid ──
        is_valid = True

        # Non-finite rejection (ISSUE-01 downstream safety)
        if not np.all(np.isfinite(emb)):
            is_valid = False

        # Zero-norm detection (sentinel input)
        norm = float(np.linalg.norm(emb).item()) if is_valid else 0.0
        if norm < 1e-6:
            is_valid = False

        # ── INVALID FRAME PATH ──
        if not is_valid:
            self._zero_streak_count += 1

            # Streak >= FLUSH_THRESHOLD (frame 3, t = 1.5 s): flush buffer entirely
            if self._zero_streak_count >= self.STREAK_FLUSH_THRESHOLD:
                self._smoothed_signature = None
                return np.zeros(self.embedding_dim, dtype=np.float32)

            # Streak < FLUSH_THRESHOLD (frames 1-2): hold previous valid signature
            if self._smoothed_signature is not None:
                return self._smoothed_signature.copy()
            else:
                return np.zeros(self.embedding_dim, dtype=np.float32)

        # ── VALID FRAME PATH ──
        # Reset streak on valid observation
        self._zero_streak_count = 0

        # L2-normalize input
        emb = emb / norm

        if self._smoothed_signature is None:
            # First valid embedding: initialize directly
            self._smoothed_signature = emb.copy()
        else:
            # Exponential Moving Average: f_target = alpha * f_new + (1 - alpha) * f_prev
            smoothed = self.alpha * emb + (1.0 - self.alpha) * self._smoothed_signature
            # Re-normalize to unit length
            smoothed_norm = float(np.linalg.norm(smoothed).item())
            if smoothed_norm > 1e-6:
                self._smoothed_signature = smoothed / smoothed_norm
            else:
                # Degenerate case: opposing vectors canceled out
                self._smoothed_signature = emb.copy()

        self._update_count += 1
        return self._smoothed_signature.copy()

    def get_signature(self) -> Optional[np.ndarray]:
        """Return the current smoothed signature or None if uninitialized/flushed."""
        if self._smoothed_signature is None:
            return None
        return self._smoothed_signature.copy()

    def reset(self):
        """Reset buffer state (full reinitialization)."""
        self._smoothed_signature = None
        self._update_count = 0
        self._zero_streak_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def zero_streak_count(self) -> int:
        """Current count of consecutive invalid frames (ISSUE-03 health tracking)."""
        return self._zero_streak_count

    @property
    def is_flushed(self) -> bool:
        """True if the smoother has been flushed due to streak >= FLUSH_THRESHOLD."""
        return (self._zero_streak_count >= self.STREAK_FLUSH_THRESHOLD
                and self._smoothed_signature is None)
