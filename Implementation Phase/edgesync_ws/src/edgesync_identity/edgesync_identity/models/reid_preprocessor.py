"""Crop preprocessing shared by Re-ID extractors (resize, ImageNet normalisation, NCHW)."""
import numpy as np

from .reid_constants import (
    IMAGENET_MEAN, IMAGENET_STD, INPUT_HEIGHT, INPUT_WIDTH, PIXEL_MAX_VALUE,
)


class ReidPreprocessor:
    """Converts an HxWx3 uint8 RGB crop to a 1x3xHxW float32 tensor."""

    def __init__(self, input_height: int = INPUT_HEIGHT, input_width: int = INPUT_WIDTH,
                 mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD):
        self.input_height = input_height
        self.input_width = input_width
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)

    def _resize(self, crop: np.ndarray) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(crop, (self.input_width, self.input_height),
                              interpolation=cv2.INTER_LINEAR)
        except ImportError:
            height, width = crop.shape[:2]
            rows = np.linspace(0, height - 1, self.input_height).astype(int)
            cols = np.linspace(0, width - 1, self.input_width).astype(int)
            return crop[rows[:, None], cols]

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        resized = self._resize(crop)
        scaled = resized.astype(np.float32) / PIXEL_MAX_VALUE
        normalized = (scaled - self._mean) / self._std
        chw = np.transpose(normalized, (2, 0, 1))
        return np.ascontiguousarray(chw[None, ...], dtype=np.float32)
