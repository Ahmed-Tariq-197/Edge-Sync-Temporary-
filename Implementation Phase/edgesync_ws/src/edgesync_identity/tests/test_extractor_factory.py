"""Unit tests for the Re-ID extractor factory (Task 6)."""
import numpy as np
import pytest

from edgesync_identity.factory.extractor_factory import (
    BackboneType, ExtractorFactory, UnknownBackboneError,
)
from edgesync_identity.interfaces.base_identity_extractor import BaseIdentityExtractor
from edgesync_identity.models.mobilenetv2_trt_extractor import MobileNetV2TrtExtractor
from edgesync_identity.models.osnet_trt_extractor import OsnetTrtExtractor
from edgesync_identity.models.reid_constants import EMBEDDING_DIM

CROP_HEIGHT = 120
CROP_WIDTH = 60
PIXEL_VALUE = 90
UNIT_NORM_TOLERANCE = 1e-5


def make_crop() -> np.ndarray:
    return np.full((CROP_HEIGHT, CROP_WIDTH, 3), PIXEL_VALUE, dtype=np.uint8)


def test_factory_creates_requested_backbone_types():
    factory = ExtractorFactory()
    assert isinstance(factory.create(BackboneType.OSNET.value), OsnetTrtExtractor)
    assert isinstance(factory.create(BackboneType.MOBILENETV2.value), MobileNetV2TrtExtractor)
    assert isinstance(factory.create("  MobileNetV2 "), MobileNetV2TrtExtractor)


def test_factory_rejects_unknown_backbone_with_available_list():
    with pytest.raises(UnknownBackboneError) as info:
        ExtractorFactory().create("resnet50")
    assert "osnet" in str(info.value) and "mobilenetv2" in str(info.value)


def test_factory_is_open_for_extension():
    class DummyExtractor(BaseIdentityExtractor):
        def extract_embedding(self, image_crop):
            return np.ones(EMBEDDING_DIM, dtype=np.float32)

    factory = ExtractorFactory()
    factory.register("dummy", lambda path: DummyExtractor())
    assert "dummy" in factory.available()
    assert isinstance(factory.create("dummy"), DummyExtractor)


def test_factory_default_builders_are_independent_per_instance():
    first, second = ExtractorFactory(), ExtractorFactory()
    first.register("only_first", lambda path: OsnetTrtExtractor())
    assert "only_first" not in second.available()


@pytest.mark.parametrize("backbone", [b.value for b in BackboneType])
def test_every_backbone_honours_the_embedding_contract(backbone):
    extractor = ExtractorFactory().create(backbone)
    assert isinstance(extractor, BaseIdentityExtractor)
    embedding = extractor.extract_embedding(make_crop())
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=UNIT_NORM_TOLERANCE)
