"""TransReID ViT-Base/16 Re-Identification Feature Extractor with Centered Truncated-SVD.

Gate 2 — ES-EXP07-DCP-REV10, Section 9.1:
  Pipeline: Resize(256x128) → (x/255 − 0.5)/0.5 Norm → ViT-Base/16 → BNNeck (post-BN features)
            → Center(x - μ_SVD) → Truncated SVD Proj(768→512) → L2 Normalize → float32[512]

  Mathematical Contract:
    x ∈ ℝ^768  (post-BNNeck CLS token)
    x̃ = x - μ_SVD
    z_proj = x̃ᵀ · W_SVD ∈ ℝ^512
    f = z_proj / ||z_proj||_2  (if ||z_proj||_2 > 1e-6 and isfinite)
    f = 0_512                  (otherwise — sentinel)

Task 7 additions (see ES-EXP07-EXECUTION-AND-CHANGE-LOG.md):
  - TR-06: the checkpoint load result is validated instead of discarded.
  - TR-02: every zeros-sentinel path records which stage failed and why. The sentinel itself is
    kept because the frozen wire contract in interfaces/base_identity_extractor.py requires it
    and forbids raising to callers; observability, not a signature change, is what distinguishes
    "extraction failed" from "a genuine low-similarity match".

Traces: ISSUE-07 (genuine model), ISSUE-01 (isfinite guard), ISSUE-08 (fail-fast).
"""
import os
import time
import logging
from typing import Optional
import numpy as np
from ..interfaces.base_identity_extractor import BaseIdentityExtractor

logger = logging.getLogger(__name__)


class TransReIdExtractor(BaseIdentityExtractor):
    """TransReID ViT-Base/16 visual embedding extractor with SVD dimensionality reduction.

    Requires:
        - checkpoint_path: Path to transformer_120.pth (official Market-1501 pretrained).
        - projection_path: Path to svd_projection_768_to_512.npz (fitted by fit_svd_projection.py).
    """

    # Measured (docs/experiments/exp07/execution/evidence/transreid_probe2.json): the official
    # checkpoint carries these four Jigsaw-Patch-Module heads, which the feature-extraction model
    # (num_classes=0) does not own. They are the ONLY tolerated unexpected keys; anything else
    # means checkpoint/schema drift and must fail loudly rather than load partially.
    TOLERATED_UNEXPECTED_KEYS = frozenset({
        'classifier_1.weight', 'classifier_2.weight',
        'classifier_3.weight', 'classifier_4.weight',
    })

    def __init__(self,
                 checkpoint_path: str,
                 projection_path: str,
                 input_shape: tuple = (256, 128)):
        """Initialize TransReID extractor with fail-fast asset validation.

        Args:
            checkpoint_path: Path to TransReID .pth checkpoint (REQUIRED — ISSUE-08).
            projection_path: Path to SVD projection .npz file (REQUIRED — ISSUE-08).
            input_shape: Re-ID standard input resolution (height=256, width=128).

        Raises:
            FileNotFoundError: If checkpoint or projection file is missing (fail-fast).
            RuntimeError: If the checkpoint does not fully cover the feature model (TR-06).
        """
        # ── TR-02 observability, initialised before loading so load-time state is recordable ──
        self.last_failure: Optional[dict] = None
        self.failure_counts: dict = {}
        self.embedding_count: int = 0
        self.checkpoint_coverage: Optional[dict] = None

        # ── ISSUE-08: Fail-fast validation ──
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"[TransReIdExtractor] Checkpoint not found: '{checkpoint_path}'"
            )
        if not os.path.isfile(projection_path):
            raise FileNotFoundError(
                f"[TransReIdExtractor] SVD projection matrix not found: '{projection_path}'"
            )

        self.checkpoint_path = checkpoint_path
        self.projection_path = projection_path
        self.input_height, self.input_width = input_shape

        # Load SVD projection artifacts
        self._load_svd_projection()

        # Load TransReID model
        self._load_model()

    def _load_svd_projection(self):
        """Load pre-fitted Centered Truncated-SVD projection matrix and mean vector.

        Expected .npz contents:
            - 'projection': W_SVD ∈ ℝ^(768, 512)
            - 'mean': μ_SVD ∈ ℝ^(768,)
            - 'explained_variance_ratio' (optional): for audit logging
        """
        data = np.load(self.projection_path)

        self._svd_projection: np.ndarray = data['projection'].astype(np.float32)
        self._svd_mean: np.ndarray = data['mean'].astype(np.float32)

        assert self._svd_projection.shape == (768, 512), (
            f"SVD projection shape mismatch: expected (768, 512), "
            f"got {self._svd_projection.shape}"
        )
        assert self._svd_mean.shape == (768,), (
            f"SVD mean shape mismatch: expected (768,), got {self._svd_mean.shape}"
        )

        if 'explained_variance_ratio' in data:
            total_var = float(np.sum(data['explained_variance_ratio']))
            logger.info(
                f"SVD projection loaded: shape={self._svd_projection.shape}, "
                f"retained variance={total_var:.4f}"
            )

    def _load_model(self):
        """Load TransReID ViT-Base/16 checkpoint (post-BNNeck features).

        Strips 'head.*' and classification keys, keeping only backbone + BNNeck.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "[TransReIdExtractor] PyTorch is required for TransReID inference. "
                "Install with: pip install torch torchvision"
            ) from e

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build ViT-Base/16 backbone
        from .transreid_backbone import build_transreid_model
        self._model = build_transreid_model(
            img_size=(self.input_height, self.input_width),
            num_classes=0,  # Feature extraction only — no classification head
        )

        # Load checkpoint, stripping head.* keys
        state_dict = torch.load(self.checkpoint_path, map_location='cpu',
                                weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter out classification head parameters
        filtered = {k: v for k, v in state_dict.items()
                    if not k.startswith('head.') and not k.startswith('classifier.')}

        # ── TR-06: the load result is validated, not discarded ──
        result = self._model.load_state_dict(filtered, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        n_model_params = len(self._model.state_dict())

        if missing:
            raise RuntimeError(
                f"[TransReIdExtractor] checkpoint does not cover {len(missing)}/{n_model_params} "
                f"model parameters; missing keys include {missing[:8]}. Refusing to run with "
                f"randomly initialised weights. Checkpoint: '{self.checkpoint_path}'"
            )
        intolerable = sorted(set(unexpected) - set(self.TOLERATED_UNEXPECTED_KEYS))
        if intolerable:
            raise RuntimeError(
                f"[TransReIdExtractor] checkpoint carries {len(intolerable)} key(s) the feature "
                f"model does not define: {intolerable[:8]}. Expected only the JPM heads "
                f"{sorted(self.TOLERATED_UNEXPECTED_KEYS)}. Checkpoint: '{self.checkpoint_path}'"
            )

        self.checkpoint_coverage = {
            'checkpoint_keys': len(state_dict),
            'keys_after_filter': len(filtered),
            'dropped_by_filter': sorted(set(state_dict) - set(filtered)),
            'model_parameter_entries': n_model_params,
            'missing_keys': missing,
            'unexpected_keys': unexpected,
            'coverage_pct': 100.0,
            'torch_parameters': int(sum(p.numel() for p in self._model.parameters())),
        }

        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"TransReID model loaded: checkpoint='{self.checkpoint_path}', "
            f"device={self._device}, params={self.checkpoint_coverage['torch_parameters']:,}, "
            f"coverage=missing:{len(missing)}/unexpected:{len(unexpected)} "
            f"(all tolerated JPM heads), dropped_by_filter="
            f"{self.checkpoint_coverage['dropped_by_filter']}"
        )

    def _sentinel(self, stage: str, reason: str) -> np.ndarray:
        """Return the contract-mandated invalid vector, attributing the failure (TR-02).

        ``interfaces/base_identity_extractor.py`` is frozen: implementations MUST return
        ``zeros(512)`` for malformed input and MUST NOT raise to callers. This keeps that wire
        behavior while making the failure attributable, so a caller can distinguish
        "extraction failed at stage X" from a genuine low-similarity embedding. The node
        (identity_node.py, TR-02) refuses to score whatever arrives after one of these.
        """
        self.last_failure = {'stage': stage, 'reason': reason,
                             'monotonic_s': time.monotonic(),
                             'checkpoint': self.checkpoint_path}
        self.failure_counts[stage] = self.failure_counts.get(stage, 0) + 1
        logger.error(f"[TransReIdExtractor] invalid embedding at stage '{stage}': {reason}")
        return np.zeros(512, dtype=np.float32)

    def preprocess(self, crop: np.ndarray) -> 'torch.Tensor':
        """Resize, normalize ([0.5, 0.5, 0.5] stats) and format to 1x3xHxW float32 tensor.

        Args:
            crop: HxWx3 uint8 numpy array in RGB format.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, 3, H, W).
        """
        import torch

        try:
            import cv2
            resized = cv2.resize(crop, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback for headless testing
            h, w = crop.shape[:2]
            y_idx = np.linspace(0, h - 1, self.input_height).astype(int)
            x_idx = np.linspace(0, w - 1, self.input_width).astype(int)
            resized = crop[y_idx[:, None], x_idx]

        # Normalize with TransReID checkpoint mean and std ([0.5, 0.5, 0.5])
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normalized = (normalized - mean) / std

        # HWC → CHW → batch
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(chw[None, ...])).to(self._device)
        return tensor

    def extract_embedding(self, image_crop: np.ndarray,
                          cam_label: Optional[int] = None) -> np.ndarray:
        """Extract L2-normalized 512-dim embedding via TransReID + SVD.

        Pipeline:
            1. Validate input crop
            2. Preprocess (resize, [0.5, 0.5, 0.5] norm, CHW)
            3. ViT-Base/16 forward (with optional cam_label) → 768-D CLS token post-BNNeck
            4. Center: x̃ = x - μ_SVD (if μ_SVD is nonzero)
            5. Project: z = x̃ᵀ · W_SVD ∈ ℝ^512
            6. Finite check + L2 normalize → f ∈ ℝ^512

        Args:
            image_crop: HxWx3 uint8 numpy array in RGB format.
            cam_label: Optional camera index (0-5) for Side Information Embedding (SIE).
                       If None, inference proceeds camera-agnostic.

        Returns:
            np.ndarray: 512-element float32 array (unit norm or zeros sentinel).
        """
        import torch

        # ── 1. Input validation ──
        if image_crop is None or image_crop.size == 0:
            return self._sentinel('input_empty', 'crop is None or has zero elements')

        if image_crop.ndim != 3 or image_crop.shape[2] != 3:
            return self._sentinel('input_shape',
                                  f'expected HxWx3 uint8 RGB crop, got shape {image_crop.shape}')

        if image_crop.shape[0] < 16 or image_crop.shape[1] < 16:
            return self._sentinel('input_too_small',
                                  f'crop {image_crop.shape[:2]} is smaller than the 16px patch grid')

        try:
            # ── 2. Preprocess ──
            input_tensor = self.preprocess(image_crop)

            # ── 3. Forward pass → 768-D post-BNNeck features ──
            with torch.no_grad():
                cam_tensor = None
                if cam_label is not None:
                    cam_tensor = torch.tensor([cam_label], device=self._device, dtype=torch.long)
                features_768 = self._model(input_tensor, cam_label=cam_tensor)  # (1, 768)

            x = features_768.cpu().numpy().flatten().astype(np.float32)

            # ── 4. Finite check on backbone output (ISSUE-01) ──
            if not np.all(np.isfinite(x)):
                return self._sentinel('backbone_non_finite',
                                      'ViT/BNNeck output contains non-finite values')

            # ── 5. Center: x̃ = x - μ_SVD ──
            x_centered = x - self._svd_mean

            # ── 6. Project: z = x̃ᵀ · W_SVD ∈ ℝ^512 ──
            z_proj = x_centered @ self._svd_projection

            # ── 7. Finite check + L2 normalize ──
            if not np.all(np.isfinite(z_proj)):
                return self._sentinel('projection_non_finite',
                                      'SVD projection produced non-finite values')

            norm = float(np.linalg.norm(z_proj).item())
            if norm <= 1e-6:
                return self._sentinel('projection_degenerate',
                                      f'projected feature norm {norm:.3e} <= 1e-6')

            self.embedding_count += 1
            self.last_failure = None
            return (z_proj / norm).astype(np.float32)

        except Exception as e:
            return self._sentinel('inference_exception', f'{type(e).__name__}: {e}')
