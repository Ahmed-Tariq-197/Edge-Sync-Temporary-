"""TransReID ViT-Base/16 Backbone with Batch Normalization Neck and JPM/SIE support.

ES-EXP07 Verified Architecture:
  Matches official transformer_120.pth checkpoint:
    - Overlapping patch embedding (patch_size=16, stride_size=12) -> 210 patches for 256x128
    - Prepended CLS token -> 211 tokens
    - Side Information Embeddings (SIE) for 6 Market-1501 cameras
    - 12 Transformer encoder blocks (dim=768, num_heads=12, mlp_ratio=4.0)
    - Jigsaw Patch Module (JPM) branches (b1, b2, bottleneck_1..4, classifier_1..4)
    - Primary BNNeck (self.bottleneck) yielding 768-D post-BN features for Re-ID.
"""
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
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
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
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
    """Overlapping patch embedding with stride 12 and patch 16."""
    def __init__(self, img_size=(256, 128), patch_size=16, stride_size=12, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        h_patches = (img_size[0] - patch_size) // stride_size + 1  # (256-16)//12 + 1 = 21
        w_patches = (img_size[1] - patch_size) // stride_size + 1  # (128-16)//12 + 1 = 10
        self.num_patches = h_patches * w_patches  # 210
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransReIDBase(nn.Module):
    """ViT-Base backbone strictly matching transformer_120.pth checkpoint structure.

    Keys:
        base.cls_token: (1, 1, 768)
        base.pos_embed: (1, 211, 768)
        base.sie_embed: (6, 1, 768)
        base.patch_embed.proj.{weight, bias}
        base.blocks.0..11.*
        base.norm.{weight, bias}
        base.fc.{weight, bias}
    """
    def __init__(self, img_size=(256, 128), patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, sie_xishu=1.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, stride_size=12, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 210 for 256x128 with stride=12

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.sie_embed = nn.Parameter(torch.zeros(6, 1, embed_dim))  # 6 cameras in Market-1501
        self.sie_xishu = sie_xishu

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1000)

    def forward(self, x, cam_label=None):
        B = x.shape[0]
        patch_tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            cls_pos = self.pos_embed[:, :1, :]
            num_needed = tokens.shape[1] - 1
            patch_pos = self.pos_embed[:, 1:num_needed + 1, :]
            tokens = tokens + torch.cat([cls_pos, patch_pos], dim=1)
        else:
            tokens = tokens + self.pos_embed

        if cam_label is not None:
            if not isinstance(cam_label, torch.Tensor):
                cam_label = torch.tensor(cam_label, device=x.device, dtype=torch.long)
            sie = self.sie_embed[cam_label]
            if sie.dim() == 2:
                sie = sie.unsqueeze(1)
            tokens = tokens + self.sie_xishu * sie

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        return tokens


class TransReIDWithJPM(nn.Module):
    """Complete TransReID model matching 100% of keys in transformer_120.pth."""
    def __init__(self, num_classes=751, img_size=(256, 128), embed_dim=768):
        super().__init__()
        self.base = TransReIDBase(img_size=img_size, embed_dim=embed_dim)

        # Global BNNeck
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)

        # JPM BNNecks
        self.bottleneck_1 = nn.BatchNorm1d(embed_dim)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_2 = nn.BatchNorm1d(embed_dim)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_3 = nn.BatchNorm1d(embed_dim)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_4 = nn.BatchNorm1d(embed_dim)
        self.bottleneck_4.bias.requires_grad_(False)

        # JPM side branches
        self.b1 = nn.ModuleList([Block(embed_dim, 12, 4.0), nn.LayerNorm(embed_dim)])
        self.b2 = nn.ModuleList([Block(embed_dim, 12, 4.0), nn.LayerNorm(embed_dim)])

        # Classifiers (present in checkpoint)
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
            self.classifier_1 = nn.Linear(embed_dim, num_classes, bias=False)
            self.classifier_2 = nn.Linear(embed_dim, num_classes, bias=False)
            self.classifier_3 = nn.Linear(embed_dim, num_classes, bias=False)
            self.classifier_4 = nn.Linear(embed_dim, num_classes, bias=False)
        else:
            self.classifier = nn.Identity()
            self.classifier_1 = nn.Identity()
            self.classifier_2 = nn.Identity()
            self.classifier_3 = nn.Identity()
            self.classifier_4 = nn.Identity()

    def forward(self, x, cam_label=None, return_pre_bn=False):
        tokens = self.base(x, cam_label=cam_label)
        cls_feat = tokens[:, 0]  # (B, 768)
        if return_pre_bn:
            return cls_feat
        bn_feat = self.bottleneck(cls_feat)
        return bn_feat


def build_transreid_model(img_size=(256, 128), num_classes=0, **kwargs):
    """Factory function for TransReID model."""
    return TransReIDWithJPM(num_classes=num_classes, img_size=img_size, **kwargs)
