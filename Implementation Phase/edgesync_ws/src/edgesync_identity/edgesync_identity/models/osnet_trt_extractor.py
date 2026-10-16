"""OSNet TensorRT Re-Identification Feature Extractor (SOLID: DIP & OCP).

Loads serialized TensorRT engine into unified memory on NVIDIA Jetson Orin Nano,
ensuring resident zero-load latency during active handoff windows.
"""
import os
from typing import Optional
import numpy as np
from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..runtime.base_inference_runner import BaseInferenceRunner
from ..runtime.trt_engine_runner import TensorRTUnavailableError, TrtEngineRunner


class OsnetTrtExtractor(BaseIdentityExtractor):
    """OSNet 512-dimensional visual embedding extractor accelerated by TensorRT."""

    def __init__(self, engine_path: Optional[str] = None,
                 input_shape: tuple[int, int] = (256, 128),
                 runner: Optional[BaseInferenceRunner] = None):
        """Initialize OSNet extractor.

        Args:
            engine_path: Path to serialized osnet_x1_0.engine.
            input_shape: Re-ID standard input resolution (height=256, width=128).
            runner: Optional pre-built inference runner (dependency injection).
        """
        self.engine_path = engine_path
        self.input_height, self.input_width = input_shape
        self._runner: Optional[BaseInferenceRunner] = runner

        if self._runner is None and engine_path and os.path.exists(engine_path):
            self._load_resident_engine()

    def _load_resident_engine(self):
        """Deserialize and pre-warm the TensorRT engine through the shared runner."""
        try:
            self._runner = TrtEngineRunner(self.engine_path)
        except TensorRTUnavailableError:
            self._runner = None

    @property
    def is_hardware_backed(self) -> bool:
        """True when a real inference runner is attached (False means mock embeddings)."""
        return self._runner is not None

    @property
    def runner(self) -> Optional[BaseInferenceRunner]:
        return self._runner

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize, normalize (ImageNet stats) and format to 1x3xHxW float32."""
        try:
            import cv2
            resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback for headless testing
            h, w = crop.shape[:2]
            y_indices = (np.linspace(0, h - 1, self.input_height)).astype(int)
            x_indices = (np.linspace(0, w - 1, self.input_width)).astype(int)
            resized = crop[y_indices[:, None], x_indices]

        # Normalize with ImageNet mean and std dev
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # HWC to CHW
        chw = np.transpose(normalized, (2, 0, 1))
        return np.ascontiguousarray(chw[None, ...])

    def _execute_trt(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference through the shared runner."""
        if self._runner is None:
            return np.zeros(512, dtype=np.float32)
        return self._runner.infer(input_tensor)

    def extract_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 512-dim visual embedding."""
        if image_crop is None or image_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        input_batch = self.preprocess(image_crop)

        if self._runner is not None:
            raw_embedding = self._execute_trt(input_batch)
        else:
            # Deterministic mock pseudo-embedding for testing / non-CUDA host
            # Uses crop statistics to produce repeatable feature vectors
            mean_vals = np.mean(image_crop, axis=(0, 1)).astype(np.float32)
            raw_embedding = np.tile(mean_vals, 512 // 3 + 1)[:512]

        norm = np.linalg.norm(raw_embedding)
        if norm > 1e-6:
            return (raw_embedding / norm).astype(np.float32)
        return raw_embedding.astype(np.float32)
