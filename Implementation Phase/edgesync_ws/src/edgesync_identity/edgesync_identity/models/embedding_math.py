"""Small numeric helpers shared by Re-ID extractors."""
import numpy as np

from .reid_constants import EMBEDDING_DIM, INPUT_CHANNELS, L2_NORM_EPSILON


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a float32 copy of the vector scaled to unit L2 norm (zero stays zero)."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat).item())
    if norm > L2_NORM_EPSILON:
        return (flat / norm).astype(np.float32)
    return flat.astype(np.float32)


def mock_embedding_from_crop(crop: np.ndarray, embedding_dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic stand-in embedding for hosts without a TensorRT runtime.

    Tiles the per-channel mean colour of the crop. It exists only so the pipeline
    can be exercised on a development machine; it carries no identity information.
    """
    if crop.ndim != 3 or crop.shape[2] != INPUT_CHANNELS:
        raise ValueError(f"Expected HxWx{INPUT_CHANNELS} crop, got shape {crop.shape}")
    channel_means = np.mean(crop, axis=(0, 1)).astype(np.float32)
    repeats = embedding_dim // INPUT_CHANNELS + 1
    return np.tile(channel_means, repeats)[:embedding_dim]
