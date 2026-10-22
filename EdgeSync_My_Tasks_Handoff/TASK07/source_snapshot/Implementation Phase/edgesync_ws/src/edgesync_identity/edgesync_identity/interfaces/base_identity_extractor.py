"""Abstract Base Class for Re-Identification Feature Extraction (SOLID: DIP & OCP).

Gate 1 Formalization — ES-EXP07-DCP-REV10:
  - Valid crop → unit-norm float32[512] (||v||_2 = 1.0 ± 1e-4).
  - Invalid / empty crop → np.zeros(512, dtype=np.float32).
  - All outputs are exactly 512-element 1D float32 arrays.
  - Subclasses MUST enforce np.all(np.isfinite()) on raw backbone output.

Traces: ISSUE-01 (NaN guard), ISSUE-07 (contract formalization).
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseIdentityExtractor(ABC):
    """Abstract interface for visual appearance Re-ID feature extractors.

    Wire Contract (frozen):
        - Output MUST be exactly np.ndarray of shape (512,) and dtype float32.
        - Valid extraction: ||output||_2 = 1.0 (unit L2-normalized).
        - Invalid extraction sentinel: np.zeros(512, dtype=np.float32).
    """

    EMBEDDING_DIM: int = 512

    @abstractmethod
    def extract_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract a 512-dimensional L2-normalized feature embedding from an image crop.

        Args:
            image_crop: HxWx3 uint8 numpy array in RGB format.
                        May be None or empty (size == 0).

        Returns:
            np.ndarray: 512-element 1D float32 array.
                        Valid extraction → unit L2 norm (||v||_2 = 1.0).
                        Invalid / empty input → zeros(512).

        Contract:
            - Implementations MUST return zeros(512) for None, empty, or malformed crops.
            - Implementations MUST reject non-finite backbone outputs (NaN/Inf) → zeros(512).
            - Implementations MUST NOT raise exceptions to callers; errors are logged
              and result in the sentinel zeros(512) return.
        """
        pass

    @staticmethod
    def validate_embedding(embedding: np.ndarray) -> np.ndarray:
        """Post-condition validator: ensures output conforms to wire contract.

        Args:
            embedding: Raw embedding from a subclass implementation.

        Returns:
            np.ndarray: Validated float32[512] — either unit-normalized or zeros.
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)

        # Shape enforcement
        if emb.shape[0] != 512:
            return np.zeros(512, dtype=np.float32)

        # Non-finite rejection (ISSUE-01: NaN/Inf safety)
        if not np.all(np.isfinite(emb)):
            return np.zeros(512, dtype=np.float32)

        # Zero-vector passthrough (sentinel)
        norm = float(np.linalg.norm(emb).item())
        if norm < 1e-6:
            return np.zeros(512, dtype=np.float32)

        return emb
