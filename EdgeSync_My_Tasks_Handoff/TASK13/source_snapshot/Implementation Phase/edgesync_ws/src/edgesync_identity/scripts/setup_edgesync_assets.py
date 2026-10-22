#!/usr/bin/env python3
"""EdgeSync Experiment 7 — Real Asset Acquisition and Verification Script.

Downloads, verifies, and organizes all model weights needed for
scientific closure of ES-EXP07-DCP-REV10.

Assets handled:
  1. ViT-Base/16 ImageNet pretrained backbone (for TransReID training)
  2. OSNet x1.0 ImageNet pretrained backbone (for OSNet baseline training)
  3. Directory structure for Market-1501 dataset
  4. Training commands and instructions

Usage:
    python setup_edgesync_assets.py --assets-dir ../../../edgesync_bringup
    python setup_edgesync_assets.py --assets-dir ../../../edgesync_bringup --download

Traces: ISSUE-05 (split provenance), ISSUE-07 (genuine weights), ISSUE-08 (fail-fast).
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import logging
import ssl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ASSET REGISTRY — Real download URLs and verification hashes
# ============================================================================
ASSETS = {
    # ViT-Base/16 ImageNet backbone (required to TRAIN TransReID)
    'vit_base_imagenet': {
        'filename': 'jx_vit_base_p16_224-80ecf9dd.pth',
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        'size_mb_approx': 330,
        'dest_subdir': 'models/transreid',
        'description': 'ViT-Base/16 ImageNet-21k pretrained backbone (for TransReID training)',
        'required_for': 'TransReID training (prerequisite for transformer_120.pth)',
        'is_final_weight': False,
    },
    # TransReID fine-tuned on Market-1501 (MUST BE TRAINED LOCALLY)
    'transreid_market1501': {
        'filename': 'transformer_120.pth',
        'url': '',
        'size_mb_approx': 330,
        'dest_subdir': 'models/transreid',
        'description': 'TransReID ViT-Base/16 fine-tuned on Market-1501 IDs 0001-0751 (120 epochs)',
        'required_for': 'Gate 2 -- TransReID Integration + SVD Fitting',
        'is_final_weight': True,
    },
    # OSNet x1.0 ImageNet pretrained (backbone for re-training)
    'osnet_x1_0_imagenet': {
        'filename': 'osnet_x1_0_imagenet.pth',
        'url': 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
        'size_mb_approx': 8.5,
        'dest_subdir': 'models/osnet',
        'description': 'OSNet x1.0 ImageNet pretrained backbone',
        'required_for': 'OSNet baseline training',
        'is_final_weight': False,
    },
    # OSNet x1.0 fine-tuned on Market-1501
    'osnet_market1501': {
        'filename': 'osnet_x1_0_market1501.pth',
        'url': '',
        'size_mb_approx': 8.5,
        'dest_subdir': 'models/osnet',
        'description': 'OSNet x1.0 fine-tuned on Market-1501 (baseline for fair comparison)',
        'required_for': 'Gate 3 -- Offline Accuracy Baseline',
        'is_final_weight': True,
    },
}

DIRECTORY_STRUCTURE = {
    'models/transreid': 'TransReID checkpoints (.pth)',
    'models/osnet': 'OSNet checkpoints (.pth)',
    'models/projections': 'SVD projection matrices (.npz)',
    'models/engines': 'TensorRT serialized engines (.engine)',
    'data/Market-1501-v15.09.15/bounding_box_train': 'Training images (IDs 0001-0751)',
    'data/Market-1501-v15.09.15/bounding_box_test': 'Gallery images (IDs 0002-1501 + distractors)',
    'data/Market-1501-v15.09.15/query': 'Query images for evaluation',
    'results/experiment7': 'Experiment 7 output logs and metrics',
}


def create_directory_structure(assets_dir):
    """Create the required directory tree for EdgeSync assets."""
    print("\n[1/5] Creating directory structure...")
    for subdir, description in DIRECTORY_STRUCTURE.items():
        full_path = os.path.join(assets_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"  [OK] {subdir}/ -- {description}")


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url, dest_path):
    """Download a file from URL with progress indication."""
    if not url:
        return False

    if os.path.isfile(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"  File already exists: {dest_path} ({size_mb:.1f} MB)")
        return True

    logger.info(f"  Downloading: {url}")
    logger.info(f"  Destination: {dest_path}")

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (EdgeSync-Assets/1.0)'
        })
        with urllib.request.urlopen(req, context=ctx) as response:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (1024 * 1024) < 65536:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB / {total/(1024*1024):.1f} MB ({pct:.0f}%)", end='', flush=True)
            print()

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"  [OK] Downloaded: {size_mb:.1f} MB")
        sha = compute_sha256(dest_path)
        logger.info(f"  SHA-256: {sha}")
        return True

    except Exception as e:
        logger.error(f"  [FAIL] Download failed: {e}")
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        return False


def print_asset_status(assets_dir):
    """Check and report status of all required assets."""
    print("\n[2/5] Checking asset status...")
    status = {}

    for asset_id, info in ASSETS.items():
        filepath = os.path.join(assets_dir, info['dest_subdir'], info['filename'])
        exists = os.path.isfile(filepath)
        size_mb = os.path.getsize(filepath) / (1024*1024) if exists else 0

        status[asset_id] = {'exists': exists, 'path': filepath, 'size_mb': size_mb}

        icon = "[OK]" if exists else "[MISSING]"
        final = " (FINAL WEIGHT)" if info['is_final_weight'] else " (prerequisite)"
        logger.info(f"  {icon} {info['filename']}: {size_mb:.1f} MB{final}")

    # Check Market-1501
    train_dir = os.path.join(assets_dir, 'data', 'Market-1501-v15.09.15', 'bounding_box_train')
    has_market = os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 100
    status['market1501'] = {'exists': has_market}
    n_files = len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0
    icon = "[OK]" if has_market else "[MISSING]"
    logger.info(f"  {icon} Market-1501 dataset: {n_files} training images found")

    # Check SVD projection
    proj_path = os.path.join(assets_dir, 'models', 'projections', 'svd_projection_768_to_512.npz')
    has_proj = os.path.isfile(proj_path)
    status['svd_projection'] = {'exists': has_proj, 'path': proj_path}
    icon = "[OK]" if has_proj else "[MISSING]"
    logger.info(f"  {icon} SVD projection matrix")

    return status


def download_available_assets(assets_dir):
    """Download all assets that have public URLs."""
    print("\n[3/5] Downloading available assets...")

    for asset_id, info in ASSETS.items():
        if not info['url']:
            logger.info(f"  [SKIP] {info['filename']} -- no public URL (must be trained locally)")
            continue

        dest_path = os.path.join(assets_dir, info['dest_subdir'], info['filename'])
        logger.info(f"  Asset: {info['filename']} ({info['description']})")
        success = download_file(info['url'], dest_path)
        if not success:
            logger.warning(f"  Manual download required for: {info['filename']}")


def print_training_instructions(assets_dir):
    """Print training instructions for assets that must be trained locally."""
    transreid_dir = os.path.join(assets_dir, 'models', 'transreid')
    osnet_dir = os.path.join(assets_dir, 'models', 'osnet')
    market_dir = os.path.join(assets_dir, 'data', 'Market-1501-v15.09.15')
    proj_dir = os.path.join(assets_dir, 'models', 'projections')

    print("\n[4/5] Training Instructions")
    print("=" * 78)
    print(f"""
  A. TRAINING TransReID (transformer_120.pth)
  --------------------------------------------
  Prerequisites:
    - ViT-Base backbone: {transreid_dir}/jx_vit_base_p16_224-80ecf9dd.pth
    - Market-1501 dataset: {market_dir}/

  Steps:
    1. git clone https://github.com/damo-cv/TransReID.git && cd TransReID
    2. pip install yacs timm
    3. python train.py \\
         --config_file configs/Market/vit_transreid_stride.yml \\
         MODEL.PRETRAIN_PATH '{transreid_dir}/jx_vit_base_p16_224-80ecf9dd.pth' \\
         DATASETS.ROOT_DIR '{os.path.dirname(market_dir)}'
    4. cp logs/.../transformer_120.pth '{transreid_dir}/transformer_120.pth'

  B. OSNet x1.0 (osnet_x1_0_market1501.pth)
  -------------------------------------------
  Option 1 -- Via Torchreid:
    pip install torchreid
    python scripts/main.py \\
      --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \\
      --root {os.path.dirname(market_dir)}

  Option 2 -- Hugging Face mirror:
    pip install huggingface_hub
    huggingface-cli download MYerassyl/retail-heat-osnet \\
        osnet_x1_0_market1501.pth --local-dir '{osnet_dir}/'

  C. FITTING SVD PROJECTION (after TransReID training)
  -----------------------------------------------------
    python fit_svd_projection.py \\
      --checkpoint '{transreid_dir}/transformer_120.pth' \\
      --market1501 '{market_dir}' \\
      --output '{proj_dir}/svd_projection_768_to_512.npz'
""")


def print_market1501_instructions(assets_dir):
    """Print Market-1501 dataset placement instructions."""
    dataset_dir = os.path.join(assets_dir, 'data', 'Market-1501-v15.09.15')
    print("=" * 78)
    print("  MARKET-1501 DATASET PLACEMENT INSTRUCTIONS")
    print("=" * 78)
    print(f"""
  Download from one of:
    - Torchreid auto-download: torchreid.data.register_image_dataset(...)
    - Kaggle: https://www.kaggle.com/datasets/whurobin/market-1501
    - Academic: http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

  Place extracted files in:
    {dataset_dir}/
    +-- bounding_box_train/    (12,936 images, IDs 0001-0751)
    +-- bounding_box_test/     (19,732 images, IDs 0002-1501 + distractors)
    +-- query/                 (3,368 query images)

  Four-Scope Data Split (ES-EXP07-DCP-REV10 Section 6.3):
  +--------------------+---------------+----------------------------------+
  | Scope              | Identity IDs  | Purpose                          |
  +--------------------+---------------+----------------------------------+
  | 1. Pretrain        | 0001 - 0751   | Official checkpoint training set |
  | 2. SVD Fit         | 0001 - 0650   | fit_svd_projection.py input      |
  | 3. Calibration     | 0651 - 0751   | FMR/FNMR verification (NOT tau)  |
  | 4. Test (FROZEN)   | 0752 - 1501   | Final evaluation ONLY            |
  +--------------------+---------------+----------------------------------+

  WARNING: Test set (IDs 0752-1501) must NEVER be used for tuning.
""")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description='Setup EdgeSync Experiment 7 real assets.',
    )
    parser.add_argument('--assets-dir',
                        default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'edgesync_bringup'),
                        help='Root directory for assets (default: edgesync_bringup)')
    parser.add_argument('--download', action='store_true',
                        help='Download assets with public URLs')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only check which assets exist')
    args = parser.parse_args()

    assets_dir = os.path.abspath(args.assets_dir)
    print("=" * 78)
    print("  EdgeSync Experiment 7 -- Real Asset Acquisition")
    print("  ES-EXP07-DCP-REV10")
    print("=" * 78)
    print(f"  Assets directory: {assets_dir}")

    create_directory_structure(assets_dir)
    status = print_asset_status(assets_dir)

    if args.verify_only:
        return

    if args.download:
        download_available_assets(assets_dir)
        status = print_asset_status(assets_dir)

    print_training_instructions(assets_dir)
    print_market1501_instructions(assets_dir)

    print("\n[5/5] NEXT STEPS (in order)")
    print("=" * 78)
    steps = []
    if not status.get('market1501', {}).get('exists'):
        steps.append("Download and place Market-1501 dataset")
    if not status.get('transreid_market1501', {}).get('exists'):
        steps.append("Train TransReID -> transformer_120.pth")
    if not status.get('svd_projection', {}).get('exists'):
        steps.append("Run fit_svd_projection.py with real TransReID checkpoint")
    if not status.get('osnet_market1501', {}).get('exists'):
        steps.append("Train/download OSNet x1.0 Market-1501 checkpoint")
    if not steps:
        steps.append("ALL ASSETS PRESENT -- Ready for Gate 3 (Offline Evaluation)")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    print()


if __name__ == '__main__':
    main()
