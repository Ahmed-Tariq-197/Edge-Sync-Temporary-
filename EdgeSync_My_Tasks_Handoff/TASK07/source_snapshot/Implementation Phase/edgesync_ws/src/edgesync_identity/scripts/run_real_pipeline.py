#!/usr/bin/env python3
"""EdgeSync Experiment 7 — Complete Real-Weight Pipeline.

Performs all steps in a single script:
  1. Loads the REAL official TransReID checkpoint (with JPM architecture)
  2. Fits Centered Truncated-SVD on Market-1501 IDs 0001-0650
  3. Evaluates TransReID + OSNet on frozen test set (IDs 0752-1501)
  4. Measures per-image inference latency
  5. Outputs complete results

This script directly instantiates the official TransReID model using
the vit_base_patch16_224_TransReID architecture from the damo-cv repo,
adapted to match the real checkpoint's state dict format.
"""
import os
import sys
import glob
import time
import json
import logging

import numpy as np
import cv2
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
BRINGUP = r"c:\Users\PC\OneDrive\Pictures\EdgeSync-main (3)\EdgeSync-main\Implementation Phase\edgesync_ws\src\edgesync_bringup"
MARKET_ROOT = os.path.join(BRINGUP, "data", "Market-1501-v15.09.15")
TRANSREID_CKPT = os.path.join(BRINGUP, "models", "transreid", "transformer_120.pth")
OSNET_CKPT = os.path.join(BRINGUP, "models", "osnet", "osnet_x1_0_market1501.pth")
# Corrected-run artifacts are versioned so the baseline projection and baseline JSON that
# produced gate3_real_results.json remain intact and re-checkable.
SVD_OUTPUT = os.path.join(BRINGUP, "models", "transreid", "svd_projection_768_to_512_real_v2.npz")
RESULTS_OUTPUT = os.path.join(BRINGUP, "results", "experiment7", "gate3_real_results_v2_corrected.json")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (256, 128)  # H, W
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Official TransReID Market-1501 inference normalisation (configs/Market/vit_transreid_stride.yml)
TRANSREID_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
TRANSREID_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


# ============================================================================
# OFFICIAL TransReID ViT Architecture (matching the real checkpoint)
# ============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 128), patch_size=16, stride_size=12, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        # Overlapping patches: ((H - patch) / stride + 1) * ((W - patch) / stride + 1)
        h_patches = (img_size[0] - patch_size) // stride_size + 1  # (256-16)/12+1 = 21
        w_patches = (img_size[1] - patch_size) // stride_size + 1  # (128-16)/12+1 = 10
        self.num_patches = h_patches * w_patches  # 210
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransReIDBase(nn.Module):
    """ViT-Base backbone matching the official TransReID state dict structure.

    Keys: base.cls_token, base.pos_embed, base.sie_embed, base.patch_embed.*,
          base.blocks.0-11.*, base.norm.*, base.fc.*
    """
    def __init__(self, img_size=(256, 128), patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., sie_xishu=1.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, stride_size=12, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 21*10 = 210 for 256x128 with stride=12

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # Will be overridden
        self.sie_embed = nn.Parameter(torch.zeros(6, 1, embed_dim))  # 6 cameras in Market-1501
        self.sie_xishu = sie_xishu

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1000)  # placeholder, not used in inference

    def forward(self, x, cam_label=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Handle pos_embed size mismatch
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate positional embeddings
            pos_embed = self.pos_embed
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            num_patches_needed = x.shape[1] - 1
            # Simple: just truncate or pad
            if patch_pos.shape[1] >= num_patches_needed:
                patch_pos = patch_pos[:, :num_patches_needed, :]
            x = x + torch.cat([cls_pos, patch_pos], dim=1)
        else:
            x = x + self.pos_embed

        # SIE: skip in inference (set cam_label=0)
        if cam_label is not None:
            sie = self.sie_embed[cam_label].squeeze(1)  # (B, embed_dim)
            if sie.dim() == 2:
                sie = sie.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + self.sie_xishu * sie

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token: (B, 768)


class TransReIDWithJPM(nn.Module):
    """Full TransReID model with JPM (Jigsaw Patch Module), matching the real
    transformer_120.pth checkpoint structure.

    Global branch: base.* + bottleneck -> 768-D
    JPM provides additional local features via b1/b2 side branches.
    For Re-ID embedding, we use the global CLS token post-BNNeck.
    """
    def __init__(self, num_classes=751, img_size=(256, 128)):
        super().__init__()
        self.base = TransReIDBase(img_size=img_size)

        # Global BNNeck
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck.bias.requires_grad_(False)

        # JPM BNNecks (b1, b2 branches)
        self.bottleneck_1 = nn.BatchNorm1d(768)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_2 = nn.BatchNorm1d(768)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_3 = nn.BatchNorm1d(768)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_4 = nn.BatchNorm1d(768)
        self.bottleneck_4.bias.requires_grad_(False)

        # JPM side branches (b1 = last 1 block, b2 = last 1 block)
        self.b1 = nn.ModuleList([
            Block(768, 12, 4.0),
            nn.LayerNorm(768),
        ])
        self.b2 = nn.ModuleList([
            Block(768, 12, 4.0),
            nn.LayerNorm(768),
        ])

        # Classifiers
        self.classifier = nn.Linear(768, num_classes, bias=False)
        self.classifier_1 = nn.Linear(768, num_classes, bias=False)
        self.classifier_2 = nn.Linear(768, num_classes, bias=False)
        self.classifier_3 = nn.Linear(768, num_classes, bias=False)
        self.classifier_4 = nn.Linear(768, num_classes, bias=False)

    def forward(self, x, cam_label=None):
        """Forward pass returning the global 768-D BNNeck feature for Re-ID."""
        # Get all tokens from the ViT backbone
        B = x.shape[0]
        patch_tokens = self.base.patch_embed(x)
        cls_tokens = self.base.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)

        if tokens.shape[1] != self.base.pos_embed.shape[1]:
            pos_embed = self.base.pos_embed
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            num_needed = tokens.shape[1] - 1
            if patch_pos.shape[1] >= num_needed:
                patch_pos = patch_pos[:, :num_needed, :]
            tokens = tokens + torch.cat([cls_pos, patch_pos], dim=1)
        else:
            tokens = tokens + self.base.pos_embed

        # SIE camera embedding. The checkpoint ships a trained base.sie_embed of shape
        # (6, 1, 768); omitting it discards a parameter the model was trained with.
        if cam_label is not None:
            sie = self.base.sie_embed[cam_label]
            tokens = tokens + self.base.sie_xishu * sie.expand(-1, tokens.shape[1], -1)

        # Run through transformer blocks
        for blk in self.base.blocks:
            tokens = blk(tokens)

        tokens = self.base.norm(tokens)

        # Global CLS feature
        global_feat = tokens[:, 0]  # (B, 768)

        # BNNeck
        feat_bn = self.bottleneck(global_feat)

        return feat_bn  # 768-D post-BNNeck for Re-ID


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_image(img_bgr, size=(256, 128), mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Preprocess an image for model input. Returns (1, 3, H, W) tensor.

    Normalisation is model-specific: OSNet was trained with ImageNet statistics, the
    official TransReID Market-1501 config uses PIXEL_MEAN/PIXEL_STD = [0.5, 0.5, 0.5].
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]))  # (W, H)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).unsqueeze(0)


def parse_market1501_name(filepath):
    """Parse (pid, camid) from Market-1501 filename."""
    name = os.path.basename(filepath)
    try:
        pid = int(name[:4])
        camid = int(name[6]) if len(name) > 6 else 0  # c1s1_000001.jpg
        return pid, camid
    except (ValueError, IndexError):
        return -1, -1


# ============================================================================
# STEP 2: Real SVD Fitting
# ============================================================================

def step2_fit_svd(model, device):
    """Extract 768-D features from IDs 0001-0650 and fit SVD."""
    logger.info("=" * 70)
    logger.info("STEP 2: Real SVD Fitting on Market-1501 IDs 0001-0650")
    logger.info("=" * 70)

    train_dir = os.path.join(MARKET_ROOT, "bounding_box_train")
    all_imgs = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))

    # Filter IDs 0001-0650
    fitting_imgs = []
    for p in all_imgs:
        pid, _ = parse_market1501_name(p)
        if 1 <= pid <= 650:
            fitting_imgs.append(p)

    n_ids = len(set(parse_market1501_name(p)[0] for p in fitting_imgs))
    logger.info(f"SVD fitting split: {len(fitting_imgs)} images, {n_ids} identities")

    # Extract features. Must match the inference path exactly (same normalisation and the
    # same SIE camera embedding), otherwise the projection is fitted on a different feature
    # distribution from the one it is later applied to.
    model.eval()
    features = []
    with torch.no_grad():
        for i, img_path in enumerate(fitting_imgs):
            img = cv2.imread(img_path)
            if img is None:
                continue
            _, fit_camid = parse_market1501_name(img_path)
            tensor = preprocess_image(img, mean=TRANSREID_MEAN,
                                      std=TRANSREID_STD).to(device)
            cam_label = torch.tensor([fit_camid - 1], device=device, dtype=torch.long)
            feat = model(tensor, cam_label=cam_label).cpu().numpy().reshape(-1)
            features.append(feat)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Extracted {i+1}/{len(fitting_imgs)}...")

    X = np.stack(features).astype(np.float32)  # (N, 768)
    logger.info(f"Feature matrix: {X.shape}")

    # Centered SVD
    mu = X.mean(axis=0)
    X_centered = X - mu
    logger.info("Computing SVD...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Top-512 components
    target_dim = 512
    W = Vt[:target_dim, :].T.astype(np.float32)  # (768, 512)
    total_var = np.sum(S ** 2)
    retained_var = np.sum(S[:target_dim] ** 2)
    variance_ratio = retained_var / total_var

    logger.info(f"SVD Results:")
    logger.info(f"  Projection: {W.shape}")
    logger.info(f"  Mean: {mu.shape}, norm={np.linalg.norm(mu):.4f}")
    logger.info(f"  Retained variance (top-512): {variance_ratio:.6f} ({variance_ratio*100:.4f}%)")
    logger.info(f"  Top-5 singular values: {S[:5].tolist()}")

    if variance_ratio < 0.95:
        logger.error(f"FAIL: Retained variance {variance_ratio:.4f} < 0.95!")
        sys.exit(1)

    # Save
    os.makedirs(os.path.dirname(SVD_OUTPUT), exist_ok=True)
    np.savez_compressed(SVD_OUTPUT, projection=W, mean=mu.astype(np.float32),
                        total_variance_ratio=np.float32(variance_ratio),
                        singular_values=S[:target_dim].astype(np.float32))
    logger.info(f"SVD saved to: {SVD_OUTPUT}")

    return W, mu, variance_ratio


# ============================================================================
# STEP 3: Offline Accuracy Evaluation
# ============================================================================

def extract_all_features(model, image_list, device, svd_W=None, svd_mu=None, desc="",
                         mean=IMAGENET_MEAN, std=IMAGENET_STD, use_cam=False):
    """Extract features for a list of (path, pid, camid) tuples.

    If svd_W is provided, applies SVD projection to get 512-D.
    Otherwise returns raw features.
    """
    features = []
    pids = []
    camids = []

    model.eval()
    with torch.no_grad():
        for i, (img_path, pid, camid) in enumerate(image_list):
            img = cv2.imread(img_path)
            if img is None:
                continue
            tensor = preprocess_image(img, mean=mean, std=std).to(device)
            if use_cam:
                cam_label = torch.tensor([camid - 1], device=device, dtype=torch.long)
                feat = model(tensor, cam_label=cam_label).cpu().numpy().reshape(-1)
            else:
                feat = model(tensor).cpu().numpy().reshape(-1)

            if svd_W is not None and svd_mu is not None:
                # Apply SVD: center then project
                feat_centered = feat - svd_mu
                feat = feat_centered @ svd_W  # (512,)

            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm

            features.append(feat.astype(np.float32))
            pids.append(pid)
            camids.append(camid)

            if (i + 1) % 2000 == 0:
                logger.info(f"  {desc}: {i+1}/{len(image_list)}...")

    return np.stack(features), np.array(pids), np.array(camids)


def compute_cmc_map(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
    """Standard Market-1501 CMC + mAP evaluation."""
    # Cosine distance matrix
    dist = 1.0 - np.dot(qf, gf.T)  # (Q, G)

    num_q = len(q_pids)
    all_cmc = np.zeros(max_rank)
    all_AP = []
    num_valid = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(dist[q_idx])
        ordered_g_pids = g_pids[order]
        ordered_g_camids = g_camids[order]

        # Remove gallery entries with same pid AND same camid (standard protocol)
        keep = ~((ordered_g_pids == q_pid) & (ordered_g_camids == q_camid))
        # Also remove junk (-1)
        keep = keep & (ordered_g_pids != -1)

        matches = (ordered_g_pids[keep] == q_pid).astype(np.float32)

        if matches.sum() == 0:
            continue

        num_valid += 1

        # CMC
        cmc = (np.cumsum(matches) > 0).astype(np.float32)
        cmc_padded = np.zeros(max_rank)
        cmc_padded[:min(len(cmc), max_rank)] = cmc[:max_rank]
        if len(cmc) < max_rank:
            cmc_padded[len(cmc):] = 1.0 if cmc[-1] > 0 else 0.0
        all_cmc += cmc_padded

        # AP
        n_true = int(matches.sum())
        precisions = np.cumsum(matches) / (np.arange(len(matches)) + 1.0)
        ap = float(np.sum(precisions * matches) / n_true)
        all_AP.append(ap)

    if num_valid == 0:
        return np.zeros(max_rank), 0.0

    cmc = all_cmc / num_valid
    mAP = float(np.mean(all_AP))

    return cmc, mAP


def step3_evaluate(transreid_model, osnet_model, device, svd_W, svd_mu):
    """Evaluate both models on frozen test set (IDs 0752-1501)."""
    logger.info("=" * 70)
    logger.info("STEP 3: Offline Accuracy Evaluation (Frozen Test IDs 0752-1501)")
    logger.info("=" * 70)

    # Collect gallery and query
    test_dir = os.path.join(MARKET_ROOT, "bounding_box_test")
    query_dir = os.path.join(MARKET_ROOT, "query")

    gallery = []
    for p in sorted(glob.glob(os.path.join(test_dir, "*.jpg"))):
        pid, camid = parse_market1501_name(p)
        if pid >= 0:  # Keep all including distractors (pid=0, pid=-1)
            gallery.append((p, pid, camid))

    query = []
    for p in sorted(glob.glob(os.path.join(query_dir, "*.jpg"))):
        pid, camid = parse_market1501_name(p)
        if pid > 0:  # Skip junk queries
            query.append((p, pid, camid))

    logger.info(f"Gallery: {len(gallery)} images")
    logger.info(f"Query:   {len(query)} images")

    results = {}

    # ── TransReID Evaluation ──
    logger.info("\n--- Evaluating TransReID (ViT-Base + SVD 768->512) ---")
    gf_t, gp_t, gc_t = extract_all_features(
        transreid_model, gallery, device, svd_W, svd_mu, "TransReID gallery",
        mean=TRANSREID_MEAN, std=TRANSREID_STD, use_cam=True)
    qf_t, qp_t, qc_t = extract_all_features(
        transreid_model, query, device, svd_W, svd_mu, "TransReID query",
        mean=TRANSREID_MEAN, std=TRANSREID_STD, use_cam=True)
    cmc_t, mAP_t = compute_cmc_map(qf_t, qp_t, qc_t, gf_t, gp_t, gc_t)

    logger.info(f"TransReID Results:")
    logger.info(f"  Rank-1:  {cmc_t[0]*100:.2f}%")
    logger.info(f"  Rank-5:  {cmc_t[4]*100:.2f}%")
    logger.info(f"  Rank-10: {cmc_t[9]*100:.2f}%")
    logger.info(f"  mAP:     {mAP_t*100:.2f}%")

    results['transreid'] = {
        'rank1': float(cmc_t[0] * 100),
        'rank5': float(cmc_t[4] * 100),
        'rank10': float(cmc_t[9] * 100),
        'mAP': float(mAP_t * 100),
        'feature_dim': int(gf_t.shape[1]),
        'n_query': len(query),
        'n_gallery': len(gallery),
    }

    # ── OSNet Evaluation ──
    logger.info("\n--- Evaluating OSNet x1.0 (native 512-D) ---")
    gf_o, gp_o, gc_o = extract_all_features(osnet_model, gallery, device, desc="OSNet gallery")
    qf_o, qp_o, qc_o = extract_all_features(osnet_model, query, device, desc="OSNet query")
    cmc_o, mAP_o = compute_cmc_map(qf_o, qp_o, qc_o, gf_o, gp_o, gc_o)

    logger.info(f"OSNet Results:")
    logger.info(f"  Rank-1:  {cmc_o[0]*100:.2f}%")
    logger.info(f"  Rank-5:  {cmc_o[4]*100:.2f}%")
    logger.info(f"  Rank-10: {cmc_o[9]*100:.2f}%")
    logger.info(f"  mAP:     {mAP_o*100:.2f}%")

    results['osnet'] = {
        'rank1': float(cmc_o[0] * 100),
        'rank5': float(cmc_o[4] * 100),
        'rank10': float(cmc_o[9] * 100),
        'mAP': float(mAP_o * 100),
        'feature_dim': int(gf_o.shape[1]),
        'n_query': len(query),
        'n_gallery': len(gallery),
    }

    return results


# ============================================================================
# STEP 4: Latency Measurement
# ============================================================================

def step4_latency(transreid_model, osnet_model, device, svd_W, svd_mu):
    """Measure per-image inference latency."""
    logger.info("=" * 70)
    logger.info("STEP 4: Latency Measurement (Laptop -- NOT Jetson Orin Nano)")
    logger.info("=" * 70)

    dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    dummy_tr = preprocess_image(dummy_img, mean=TRANSREID_MEAN,
                               std=TRANSREID_STD).to(device)
    dummy_os = preprocess_image(dummy_img).to(device)
    dummy_cam = torch.tensor([0], device=device, dtype=torch.long)

    svd_W_torch = torch.from_numpy(svd_W).to(device) if svd_W is not None else None
    svd_mu_torch = torch.from_numpy(svd_mu).to(device) if svd_mu is not None else None

    results = {}
    has_cuda = device.type == 'cuda'

    for name, model, use_svd, dummy_tensor, cam_kw, ckpt in [
        ("TransReID", transreid_model, True, dummy_tr, {'cam_label': dummy_cam}, TRANSREID_CKPT),
        ("OSNet", osnet_model, False, dummy_os, {}, OSNET_CKPT),
    ]:
        model.eval()
        logger.info(f"\n--- {name} Latency ---")

        # Model size: parameter footprint and serialised artifact size are different numbers.
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_size_mb = param_bytes / (1024 * 1024)
        ckpt_size_mb = (os.path.getsize(ckpt) / (1024 * 1024)) if os.path.exists(ckpt) else None
        model_size_mb = param_size_mb

        # Warmup, then 3 independent trials of 200 timed iterations each.
        # CUDA events time the GPU work without the CPU-side synchronisation bias of
        # perf_counter, and repeating the trial is what makes a contended host visible:
        # a single 200-iteration run once reported 39.35 ms for a 13.9 ms model.
        N_WARMUP, N_TIMED, N_TRIALS = 50, 200, 3
        back_lat, total_lat, proj_lat, trial_means = [], [], [], []
        for trial in range(N_TRIALS):
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    feat = model(dummy_tensor, **cam_kw)
                    if use_svd and svd_W_torch is not None:
                        feat = (feat - svd_mu_torch) @ svd_W_torch
                    if has_cuda:
                        torch.cuda.synchronize()

            t_back, t_total, t_proj = [], [], []
            with torch.no_grad():
                for _ in range(N_TIMED):
                    if has_cuda:
                        e0 = torch.cuda.Event(True)
                        e1 = torch.cuda.Event(True)
                        e2 = torch.cuda.Event(True)
                        e0.record()
                        feat = model(dummy_tensor, **cam_kw)
                        e1.record()
                        if use_svd and svd_W_torch is not None:
                            feat = (feat - svd_mu_torch) @ svd_W_torch
                        e2.record()
                        torch.cuda.synchronize()
                        t_back.append(e0.elapsed_time(e1))
                        t_proj.append(e1.elapsed_time(e2))
                        t_total.append(e0.elapsed_time(e2))
                    else:
                        t0 = time.perf_counter()
                        feat = model(dummy_tensor, **cam_kw)
                        t1 = time.perf_counter()
                        if use_svd and svd_W_torch is not None:
                            feat = (feat - svd_mu_torch) @ svd_W_torch
                        t2 = time.perf_counter()
                        t_back.append((t1 - t0) * 1000.0)
                        t_proj.append((t2 - t1) * 1000.0)
                        t_total.append((t2 - t0) * 1000.0)
            back_lat += t_back
            total_lat += t_total
            proj_lat += t_proj
            trial_means.append(float(np.mean(t_total)))
            logger.info(f"  trial {trial}: total {np.mean(t_total):.3f} ms "
                        f"(backbone {np.mean(t_back):.3f}, projection {np.mean(t_proj):.3f})")

        lat = np.array(total_lat)
        back = np.array(back_lat)
        proj = np.array(proj_lat)
        spread = (max(trial_means) - min(trial_means)) / float(np.mean(trial_means)) * 100.0
        results[name.lower().replace(" ", "_")] = {
            'mean_ms': float(np.mean(lat)),
            'median_ms': float(np.median(lat)),
            'p95_ms': float(np.percentile(lat, 95)),
            'p99_ms': float(np.percentile(lat, 99)),
            'min_ms': float(np.min(lat)),
            'max_ms': float(np.max(lat)),
            'backbone_mean_ms': float(np.mean(back)),
            'backbone_median_ms': float(np.median(back)),
            'backbone_p95_ms': float(np.percentile(back, 95)),
            'projection_mean_ms': float(np.mean(proj)),
            'includes_svd_projection': bool(use_svd),
            'param_size_mb': float(param_size_mb),
            'checkpoint_file_size_mb': float(ckpt_size_mb) if ckpt_size_mb else None,
            'warmup_iterations': N_WARMUP,
            'timed_iterations': N_TIMED,
            'trials': N_TRIALS,
            'trial_mean_ms': trial_means,
            'trial_spread_pct': float(spread),
            'timing_primitive': 'cuda_events' if has_cuda else 'perf_counter',
            'batch_size': 1,
            'model_size_mb': float(model_size_mb),
            'device': str(device),
            'note': 'Laptop measurement (NOT Jetson Orin Nano 8GB)',
        }

        logger.info(f"  Model size:    {model_size_mb:.1f} MB")
        logger.info(f"  Mean latency:  {np.mean(lat):.2f} ms   (backbone {np.mean(back):.2f}, "
                    f"projection {np.mean(proj):.3f})")
        logger.info(f"  Median:        {np.median(lat):.2f} ms")
        logger.info(f"  P95:           {np.percentile(lat, 95):.2f} ms")
        logger.info(f"  P99:           {np.percentile(lat, 99):.2f} ms")
        logger.info(f"  Trial spread:  {spread:.1f} %")
        if spread > 15.0:
            logger.warning(f"  {name}: trial spread {spread:.1f}% -- the host was not idle; "
                           f"treat these timings as unverified and re-measure.")
        logger.info(f"  Device:        {device}")
        logger.info(f"  ** Laptop measurement (NOT Jetson Orin Nano 8GB) **")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  EdgeSync Experiment 7 -- Complete Real-Weight Pipeline")
    print("  ES-EXP07-DCP-REV10 -- Scientific Closure")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  TransReID checkpoint: {TRANSREID_CKPT}")
    print(f"  OSNet checkpoint:     {OSNET_CKPT}")
    print(f"  Market-1501:          {MARKET_ROOT}")
    print()

    # ── Load TransReID ──
    logger.info("Loading TransReID model...")
    transreid = TransReIDWithJPM(num_classes=751, img_size=IMG_SIZE)
    sd = torch.load(TRANSREID_CKPT, map_location='cpu', weights_only=False)
    # Load with strict=False to handle any key mismatches
    missing, unexpected = transreid.load_state_dict(sd, strict=False)
    logger.info(f"TransReID loaded: {sum(p.numel() for p in transreid.parameters()):,} params")
    if missing:
        logger.info(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    transreid.to(DEVICE)
    transreid.eval()

    # ── Load OSNet (using official torchreid osnet module) ──
    logger.info("Loading OSNet model (official torchreid architecture)...")
    import importlib.util
    osnet_spec = importlib.util.spec_from_file_location(
        'osnet_module',
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'torchreid', 'reid', 'models', 'osnet.py')
    )
    osnet_module = importlib.util.module_from_spec(osnet_spec)
    osnet_spec.loader.exec_module(osnet_module)
    osnet = osnet_module.osnet_x1_0(num_classes=751, pretrained=False)
    sd2 = torch.load(OSNET_CKPT, map_location='cpu', weights_only=False)
    if 'state_dict' in sd2:
        sd2 = sd2['state_dict']
    osnet.load_state_dict(sd2)
    logger.info(f"OSNet loaded: {sum(p.numel() for p in osnet.parameters()):,} params")
    osnet.to(DEVICE)
    osnet.eval()

    # ── STEP 2: SVD ──
    svd_W, svd_mu, variance = step2_fit_svd(transreid, DEVICE)

    # ── STEP 3: Accuracy ──
    accuracy_results = step3_evaluate(transreid, osnet, DEVICE, svd_W, svd_mu)

    # ── STEP 4: Latency ──
    latency_results = step4_latency(transreid, osnet, DEVICE, svd_W, svd_mu)

    # ── Save Results ──
    os.makedirs(os.path.dirname(RESULTS_OUTPUT), exist_ok=True)
    output = {
        'experiment': 'ES-EXP07',
        'evaluation_type': 'Offline Market-1501 (Real Weights)',
        'revision': 'v2-corrected-preprocessing',
        'pipeline_config': {
            'transreid_pixel_mean': TRANSREID_MEAN.tolist(),
            'transreid_pixel_std': TRANSREID_STD.tolist(),
            'transreid_sie_applied': True,
            'transreid_sie_xishu': float(transreid.base.sie_xishu),
            'osnet_pixel_mean': IMAGENET_MEAN.tolist(),
            'osnet_pixel_std': IMAGENET_STD.tolist(),
            'transreid_neck_feat': 'after (post-BNNeck)',
            'osnet_feature': 'native 512-D, no projection',
            'image_size_hw': list(IMG_SIZE),
            'latency_warmup_iterations': 50,
            'latency_timed_iterations': 200,
            'latency_trials': 3,
            'latency_timing_primitive': 'cuda_events',
        },
        'threshold_tau': 0.70,
        'threshold_tau_note': 'Reported Rank-1/mAP are threshold-free retrieval metrics; '
                              'tau is not applied anywhere in this script.',
        'svd_variance_retained': float(variance),
        'device': str(DEVICE),
        'accuracy': accuracy_results,
        'latency': latency_results,
        'note': 'All latency numbers are Laptop measurements, NOT Jetson Orin Nano.',
    }
    with open(RESULTS_OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {RESULTS_OUTPUT}")

    # ── Print Summary ──
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("  Latency: Laptop measurement (NOT Jetson Orin Nano 8GB)")
    print("=" * 70)
    print(f"\n  SVD Retained Variance: {variance*100:.4f}%")
    print()
    hdr = f"  {'Model':<20} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8} {'Dim':>5} {'Mean(ms)':>10} {'Med(ms)':>9} {'P95(ms)':>9} {'Size(MB)':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for model_key, model_name in [('transreid', 'TransReID+SVD'), ('osnet', 'OSNet x1.0')]:
        acc = accuracy_results[model_key]
        lat_key = model_key if model_key in latency_results else model_key.replace(' ', '_')
        lat = latency_results.get(lat_key, latency_results.get(model_key, {}))
        print(f"  {model_name:<20} {acc['rank1']:>7.2f}% {acc['rank5']:>7.2f}% {acc['mAP']:>7.2f}% {acc['feature_dim']:>5d} {lat.get('mean_ms',0):>9.2f} {lat.get('median_ms',0):>8.2f} {lat.get('p95_ms',0):>8.2f} {lat.get('model_size_mb',0):>9.1f}")

    print(f"\n  Query count:   {accuracy_results['transreid']['n_query']}")
    print(f"  Gallery count: {accuracy_results['transreid']['n_gallery']}")
    print()
    print("  ** ALL LATENCY VALUES ARE LAPTOP MEASUREMENTS **")
    print("  ** NOT representative of Jetson Orin Nano 8GB performance **")
    print("=" * 70)


if __name__ == '__main__':
    main()
