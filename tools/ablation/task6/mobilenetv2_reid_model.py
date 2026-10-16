"""MobileNetV2 Re-ID model: ImageNet-pretrained backbone plus a 512-dim embedding head.

The head maps the native 1280-dim pooled MobileNetV2 feature to the 512-dim embedding
required by IdentityVerify.srv, so the identity service contract stays unchanged.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_PACKAGE_ROOT = (REPO_ROOT / "Implementation Phase" / "edgesync_ws" / "src" / "edgesync_identity")
sys.path.insert(0, str(IDENTITY_PACKAGE_ROOT))

from edgesync_identity.models.reid_constants import (  # noqa: E402
    EMBEDDING_DIM, IMAGENET_MEAN, IMAGENET_STD, INFERENCE_BATCH_SIZE, INPUT_CHANNELS,
    INPUT_HEIGHT, INPUT_WIDTH,
)

MOBILENETV2_FEATURE_DIM = 1280
POOLED_SPATIAL_SIZE = 1

__all__ = ["EMBEDDING_DIM", "IMAGENET_MEAN", "IMAGENET_STD", "INFERENCE_BATCH_SIZE",
           "INPUT_CHANNELS", "INPUT_HEIGHT", "INPUT_WIDTH", "MobileNetV2Embedder",
           "ReidTrainingModel", "REPO_ROOT"]


class MobileNetV2Embedder(nn.Module):
    """Image batch (N,3,256,128) -> embedding batch (N,512) (not yet L2-normalised)."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.mobilenet_v2(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(POOLED_SPATIAL_SIZE)
        self.projection = nn.Linear(MOBILENETV2_FEATURE_DIM, embedding_dim, bias=False)
        self.norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(images)).flatten(1)
        return self.norm(self.projection(pooled))


class ReidTrainingModel(nn.Module):
    """Embedder plus an identity classifier used only during training."""

    def __init__(self, num_identities: int, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True):
        super().__init__()
        self.embedder = MobileNetV2Embedder(embedding_dim, pretrained)
        self.classifier = nn.Linear(embedding_dim, num_identities, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedder(images))
