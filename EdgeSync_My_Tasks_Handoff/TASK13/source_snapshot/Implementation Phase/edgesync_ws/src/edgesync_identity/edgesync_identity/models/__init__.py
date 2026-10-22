"""Identity models."""
from .osnet_trt_extractor import OsnetTrtExtractor
from .extractor_factory import create_extractor

__all__ = ['OsnetTrtExtractor', 'create_extractor']
