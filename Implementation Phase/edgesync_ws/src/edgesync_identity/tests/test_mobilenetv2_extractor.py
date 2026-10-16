"""Unit tests for MobileNetV2TrtExtractor (Task 6) using an injected fake runner."""
import numpy as np
import pytest

from edgesync_identity.models.mobilenetv2_trt_extractor import MobileNetV2TrtExtractor
from edgesync_identity.models.reid_constants import (
    EMBEDDING_DIM, INFERENCE_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
)
from edgesync_identity.models.reid_preprocessor import ReidPreprocessor
from edgesync_identity.runtime.base_inference_runner import BaseInferenceRunner

CROP_HEIGHT = 200
CROP_WIDTH = 90
RAW_SCALE = 7.5
WRONG_OUTPUT_DIM = EMBEDDING_DIM - 1
UNIT_NORM_TOLERANCE = 1e-5


class FakeRunner(BaseInferenceRunner):
    def __init__(self, output_dim: int = EMBEDDING_DIM):
        self.output_dim = output_dim
        self.calls = []

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        self.calls.append(input_tensor.shape)
        out = np.zeros(self.output_dim, dtype=np.float32)
        out[0] = RAW_SCALE
        return out


def make_crop(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (CROP_HEIGHT, CROP_WIDTH, INPUT_CHANNELS), dtype=np.uint8)


def test_preprocessor_output_contract():
    tensor = ReidPreprocessor()(make_crop())
    assert tensor.shape == (INFERENCE_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


def test_runner_output_is_l2_normalised_and_input_shape_is_correct():
    runner = FakeRunner()
    extractor = MobileNetV2TrtExtractor(runner=runner)
    embedding = extractor.extract_embedding(make_crop())
    assert extractor.is_hardware_backed is True
    assert runner.calls == [(INFERENCE_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]
    assert embedding.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=UNIT_NORM_TOLERANCE)
    assert embedding[0] == pytest.approx(1.0, abs=UNIT_NORM_TOLERANCE)


def test_wrong_engine_output_dimension_is_rejected():
    extractor = MobileNetV2TrtExtractor(runner=FakeRunner(WRONG_OUTPUT_DIM))
    with pytest.raises(ValueError):
        extractor.extract_embedding(make_crop())


@pytest.mark.parametrize("bad_crop", [None, np.zeros((0, 0, INPUT_CHANNELS), dtype=np.uint8)])
def test_empty_crop_returns_zero_vector_without_inference(bad_crop):
    runner = FakeRunner()
    embedding = MobileNetV2TrtExtractor(runner=runner).extract_embedding(bad_crop)
    assert embedding.shape == (EMBEDDING_DIM,)
    assert not embedding.any()
    assert runner.calls == []


def test_mock_mode_is_deterministic_and_flagged():
    extractor = MobileNetV2TrtExtractor()
    assert extractor.is_hardware_backed is False
    first = extractor.extract_embedding(make_crop(1))
    second = extractor.extract_embedding(make_crop(1))
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=UNIT_NORM_TOLERANCE)


def test_missing_engine_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        MobileNetV2TrtExtractor(engine_path=str(tmp_path / "missing.engine"))


def test_grayscale_crop_is_rejected_in_mock_mode():
    with pytest.raises(ValueError):
        MobileNetV2TrtExtractor().extract_embedding(np.ones((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8))
