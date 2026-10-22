#!/usr/bin/env python3
"""Fit Centered Truncated-SVD Projection Matrix (768-D → 512-D).

Gate 2 — ES-EXP07-DCP-REV10, Section 9.1:
  Loads the TransReID ViT-Base/16 model, processes Market-1501 SVD fitting split
  (IDs 0001-0650), fits a Centered Truncated SVD projection, verifies retained
  variance >= 95.0%, and saves the projection matrix and mean vector.

Usage:
    python fit_svd_projection.py \\
        --checkpoint /path/to/transformer_120.pth \\
        --market1501 /path/to/Market-1501/ \\
        --output svd_projection_768_to_512.npz \\
        --target-dim 512 \\
        --min-variance 0.95

Data Split Provenance (Section 6.3):
    - Model Pretraining: IDs 0001-0751 (official checkpoint training set)
    - SVD Fitting:       IDs 0001-0650 (this script)
    - Calibration:       IDs 0651-0751 (FMR/FNMR verification — NOT tuning τ)
    - Test Set:          IDs 0752-1501 (strictly frozen, evaluation only)

Traces: ISSUE-05 (pretraining provenance), ISSUE-07 (genuine weights required).
"""
import argparse
import os
import sys
import glob
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_identity_id(folder_name: str) -> int:
    """Extract integer identity ID from Market-1501 folder/filename convention.

    Market-1501 naming: <4-digit ID>_<camera>_<sequence>_<frame>.jpg
    Folder structure: bounding_box_train/<4-digit ID>_*.jpg
    """
    try:
        return int(folder_name[:4])
    except (ValueError, IndexError):
        return -1


def collect_fitting_images(market1501_root: str,
                           id_min: int = 1,
                           id_max: int = 650) -> list:
    """Collect image paths for SVD fitting split (IDs 0001-0650).

    Args:
        market1501_root: Root directory of Market-1501 dataset.
        id_min: Minimum identity ID (inclusive).
        id_max: Maximum identity ID (inclusive).

    Returns:
        List of image file paths within the fitting split.
    """
    train_dir = os.path.join(market1501_root, 'bounding_box_train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Market-1501 training directory not found: '{train_dir}'. "
            f"Expected structure: {market1501_root}/bounding_box_train/*.jpg"
        )

    all_images = sorted(glob.glob(os.path.join(train_dir, '*.jpg')))
    fitting_images = []
    for img_path in all_images:
        basename = os.path.basename(img_path)
        pid = parse_identity_id(basename)
        if id_min <= pid <= id_max:
            fitting_images.append(img_path)

    logger.info(
        f"SVD fitting split: {len(fitting_images)} images from "
        f"{len(set(parse_identity_id(os.path.basename(p)) for p in fitting_images))} "
        f"identities (IDs {id_min:04d}-{id_max:04d})"
    )
    return fitting_images


def extract_features(checkpoint_path: str,
                     image_paths: list,
                     batch_size: int = 32) -> np.ndarray:
    """Extract 768-D post-BNNeck features from all images using TransReID.

    Args:
        checkpoint_path: Path to TransReID .pth checkpoint.
        image_paths: List of image file paths.
        batch_size: Inference batch size.

    Returns:
        np.ndarray: Feature matrix of shape (N, 768).
    """
    import torch
    import cv2

    # Add the edgesync_identity package to path for model import
    pkg_root = Path(__file__).resolve().parent
    identity_pkg = pkg_root / 'edgesync_identity'
    if not identity_pkg.exists():
        # Try relative to the script location in the workspace
        identity_pkg = pkg_root.parent / 'edgesync_identity' / 'edgesync_identity'
    sys.path.insert(0, str(identity_pkg.parent))

    from edgesync_identity.models.transreid_backbone import build_transreid_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build model
    model = build_transreid_model(img_size=(256, 128), num_classes=0)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    filtered = {k: v for k, v in state_dict.items()
                if not k.startswith('head.') and not k.startswith('classifier.')}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # TransReID ViT-B/16 was trained with (x/255 - 0.5) / 0.5. ImageNet stats put
    # every fitted feature on the wrong manifold (ES-EXP07 root cause, E7-BUG-001).
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    all_features = []
    total = len(image_paths)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_imgs = []

        for img_path in image_paths[start_idx:end_idx]:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            batch_imgs.append(img)

        if not batch_imgs:
            continue

        batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
        with torch.no_grad():
            features = model(batch_tensor)  # (B, 768)
        all_features.append(features.cpu().numpy())

        if (start_idx // batch_size) % 50 == 0:
            logger.info(f"Extracted {end_idx}/{total} images...")

    feature_matrix = np.concatenate(all_features, axis=0).astype(np.float32)
    logger.info(f"Feature extraction complete: shape={feature_matrix.shape}")
    return feature_matrix


def fit_centered_truncated_svd(features: np.ndarray,
                                target_dim: int = 512,
                                min_variance: float = 0.95) -> dict:
    """Fit Centered Truncated SVD to reduce 768-D features to target_dim.

    Mathematical procedure:
        1. Compute mean μ = mean(X, axis=0) ∈ ℝ^768
        2. Center: X̃ = X - μ
        3. Compute SVD: X̃ = U · Σ · Vᵀ
        4. Take top-target_dim components: W = V[:, :target_dim] ∈ ℝ^(768, target_dim)
        5. Verify: retained variance = Σ[:target_dim]² / Σ_total² >= min_variance

    Args:
        features: Feature matrix of shape (N, 768).
        target_dim: Target dimensionality (512).
        min_variance: Minimum retained variance threshold (0.95 = 95%).

    Returns:
        dict with keys: 'projection', 'mean', 'explained_variance_ratio',
                        'singular_values', 'total_variance_ratio'

    Raises:
        ValueError: If retained variance is below min_variance.
    """
    N, D = features.shape
    logger.info(f"Fitting Centered SVD: {N} samples, {D}-D → {target_dim}-D")

    # 1. Compute mean
    mu = np.mean(features, axis=0).astype(np.float32)
    logger.info(f"Mean vector: shape={mu.shape}, norm={np.linalg.norm(mu):.4f}")

    # 2. Center
    X_centered = features - mu

    # 3. Full SVD (economy mode for efficiency)
    logger.info("Computing SVD (this may take a few minutes)...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 4. Take top-k components
    W = Vt[:target_dim, :].T.astype(np.float32)  # (768, target_dim)
    assert W.shape == (D, target_dim), f"Projection shape error: {W.shape}"

    # 5. Variance analysis
    total_var = np.sum(S ** 2)
    retained_var = np.sum(S[:target_dim] ** 2)
    total_ratio = retained_var / total_var
    per_component_ratio = (S[:target_dim] ** 2) / total_var

    logger.info(f"SVD complete:")
    logger.info(f"  Singular values: top-5 = {S[:5].tolist()}")
    logger.info(f"  Retained variance (top-{target_dim}): {total_ratio:.6f} ({total_ratio*100:.2f}%)")
    logger.info(f"  Total singular values: {len(S)}")

    if total_ratio < min_variance:
        raise ValueError(
            f"Retained variance {total_ratio:.4f} ({total_ratio*100:.2f}%) is below "
            f"minimum threshold {min_variance:.4f} ({min_variance*100:.2f}%). "
            f"Consider increasing target_dim or reviewing the feature distribution."
        )

    return {
        'projection': W,
        'mean': mu,
        'explained_variance_ratio': per_component_ratio.astype(np.float32),
        'singular_values': S[:target_dim].astype(np.float32),
        'total_variance_ratio': np.float32(total_ratio),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fit Centered Truncated-SVD Projection (768-D → 512-D) for TransReID.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Split Provenance:
  Model Pretraining: IDs 0001-0751 (official checkpoint training set)
  SVD Fitting:       IDs 0001-0650 (this script — 650 identities)
  Calibration:       IDs 0651-0751 (FMR/FNMR verification)
  Test Set:          IDs 0752-1501 (frozen evaluation)
        """
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to TransReID .pth checkpoint (transformer_120.pth)')
    parser.add_argument('--market1501', required=True,
                        help='Path to Market-1501 dataset root directory')
    parser.add_argument('--output', default='svd_projection_768_to_512.npz',
                        help='Output path for SVD projection .npz file')
    parser.add_argument('--target-dim', type=int, default=512,
                        help='Target embedding dimensionality (default: 512)')
    parser.add_argument('--min-variance', type=float, default=0.95,
                        help='Minimum retained variance threshold (default: 0.95 = 95%%)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Feature extraction batch size (default: 32)')
    parser.add_argument('--id-min', type=int, default=1,
                        help='Minimum identity ID for fitting split (default: 1)')
    parser.add_argument('--id-max', type=int, default=650,
                        help='Maximum identity ID for fitting split (default: 650)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}'")

    # Collect fitting images
    image_paths = collect_fitting_images(args.market1501, args.id_min, args.id_max)
    if len(image_paths) == 0:
        raise ValueError("No images found in fitting split. Check Market-1501 path and ID range.")

    # Extract features
    features = extract_features(args.checkpoint, image_paths, args.batch_size)

    # Fit SVD
    svd_result = fit_centered_truncated_svd(features, args.target_dim, args.min_variance)

    # Save
    np.savez_compressed(
        args.output,
        projection=svd_result['projection'],
        mean=svd_result['mean'],
        explained_variance_ratio=svd_result['explained_variance_ratio'],
        singular_values=svd_result['singular_values'],
        total_variance_ratio=svd_result['total_variance_ratio'],
    )

    logger.info(f"SVD projection saved to: {args.output}")
    logger.info(f"  Projection shape: {svd_result['projection'].shape}")
    logger.info(f"  Mean shape: {svd_result['mean'].shape}")
    logger.info(f"  Retained variance: {float(svd_result['total_variance_ratio']):.4f}")
    logger.info("Done. Proceed to Gate 3 (Offline Benchmarks) after verification.")


if __name__ == '__main__':
    main()
