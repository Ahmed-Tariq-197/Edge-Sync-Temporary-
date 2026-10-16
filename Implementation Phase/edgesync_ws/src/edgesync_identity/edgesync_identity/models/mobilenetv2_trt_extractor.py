"""MobileNetV2 TensorRT Re-ID feature extractor (Task 6 ablation arm).

Drop-in replacement for OsnetTrtExtractor behind BaseIdentityExtractor. The network
is MobileNetV2 with a 1280 -> 512 embedding head, exported to a static-shape engine,
so the 512-dim L2-normalised contract of the identity service is unchanged.
"""
import os
from typing import Optional

import numpy as np

from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..runtime.base_inference_runner import BaseInferenceRunner
from ..runtime.trt_engine_runner import TensorRTUnavailableError, TrtEngineRunner
from .embedding_math import l2_normalize, mock_embedding_from_crop
from .reid_constants import EMBEDDING_DIM
from .reid_preprocessor import ReidPreprocessor


class MobileNetV2TrtExtractor(BaseIdentityExtractor):
    """512-dimensional visual embedding extractor backed by a MobileNetV2 engine."""

    def __init__(self, engine_path: Optional[str] = None,
                 runner: Optional[BaseInferenceRunner] = None,
                 preprocessor: Optional[ReidPreprocessor] = None,
                 embedding_dim: int = EMBEDDING_DIM):
        self.engine_path = engine_path
        self.embedding_dim = embedding_dim
        self._preprocessor = preprocessor if preprocessor is not None else ReidPreprocessor()
        self._runner: Optional[BaseInferenceRunner] = runner
        if self._runner is None and engine_path:
            self._runner = self._build_runner(engine_path)

    @staticmethod
    def _build_runner(engine_path: str) -> Optional[BaseInferenceRunner]:
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"MobileNetV2 engine not found: {engine_path}")
        try:
            return TrtEngineRunner(engine_path)
        except TensorRTUnavailableError:
            return None

    @property
    def is_hardware_backed(self) -> bool:
        """True when a real inference runner is attached (False means mock embeddings)."""
        return self._runner is not None

    @property
    def runner(self) -> Optional[BaseInferenceRunner]:
        return self._runner

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        return self._preprocessor(crop)

    def extract_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        if image_crop is None or image_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self._runner is not None:
            raw = self._runner.infer(self.preprocess(image_crop)).reshape(-1)
            if raw.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Engine returned {raw.shape[0]} values, expected {self.embedding_dim}")
        else:
            raw = mock_embedding_from_crop(image_crop, self.embedding_dim)
        return l2_normalize(raw)
