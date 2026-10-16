"""Factory for Re-ID extractors (SOLID: OCP). New backbones register a builder, nothing else changes."""
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from ..models.mobilenetv2_trt_extractor import MobileNetV2TrtExtractor
from ..models.osnet_trt_extractor import OsnetTrtExtractor

ExtractorBuilder = Callable[[Optional[str]], BaseIdentityExtractor]


class BackboneType(str, Enum):
    """Supported Re-ID backbones (values are the YAML / CLI identifiers)."""

    OSNET = "osnet"
    MOBILENETV2 = "mobilenetv2"


class UnknownBackboneError(ValueError):
    """Raised when a backbone name has no registered builder."""


class ExtractorFactory:
    """Creates BaseIdentityExtractor instances from a backbone identifier."""

    def __init__(self, builders: Optional[Dict[str, ExtractorBuilder]] = None):
        self._builders: Dict[str, ExtractorBuilder] = (
            dict(builders) if builders is not None else self.default_builders())

    @staticmethod
    def default_builders() -> Dict[str, ExtractorBuilder]:
        return {
            BackboneType.OSNET.value: lambda path: OsnetTrtExtractor(engine_path=path),
            BackboneType.MOBILENETV2.value: lambda path: MobileNetV2TrtExtractor(engine_path=path),
        }

    def register(self, name: str, builder: ExtractorBuilder) -> None:
        self._builders[name.strip().lower()] = builder

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders))

    def create(self, backbone: str, engine_path: Optional[str] = None) -> BaseIdentityExtractor:
        key = backbone.strip().lower()
        if key not in self._builders:
            raise UnknownBackboneError(
                f"Unknown Re-ID backbone '{backbone}'. Available: {', '.join(self.available())}")
        return self._builders[key](engine_path if engine_path else None)
