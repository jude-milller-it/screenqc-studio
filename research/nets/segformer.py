"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
Paper: https://arxiv.org/abs/2105.15203

MiT (Mix Transformer) encoder + MLP decoder.
Supported variants: MiT-b0 ~ MiT-b5.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ==================== Overlap Patch Embedding ====================
class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding via strided convolution (no position encoding needed)."""

    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                          # B, C, H, W
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)          # B, N, C
        x = self.norm(x)                           # LayerNorm over C
        x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)  # B, C, H, W
        return x, h, w


# ==================== Efficient Self-Attention ====================
class EfficientSelfAttention(nn.Module):
    """
    SegFormer efficient self-attention.
    Spatial reduction is applied to INPUT before K,V projection (not after).
    Q has full resolution (N tokens), K,V have reduced resolution (N_r tokens).
    """

    def __init__(self, dim, num_heads, reduction_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if reduction_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        else:
            self.sr = nn.Identity()

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape

        # Q from full-resolution input
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Spatial reduction on x, then K,V projection from reduced x
        if self.reduction_ratio > 1:
            x_spatial = x.transpose(1, 2).reshape(b, c, h, w)
            x_spatial = self.sr(x_spatial)
            _, _, h_r, w_r = x_spatial.shape
            x_reduced = x_spatial.flatten(2).transpose(1, 2)  # B, N_r, C
        else:
            x_reduced = x

        kv = self.kv(x_reduced).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


# ==================== Mix-FFN ====================
class MixFFN(nn.Module):
    """FFN with 3x3 depthwise conv (replaces position encoding)."""

    def __init__(self, dim, expansion_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = self.fc1(x)
        # spatial reshape → depthwise conv → flatten
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==================== Transformer Block ====================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio, expansion_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion_ratio, dropout)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


# ==================== MiT Encoder ====================
class MiT(nn.Module):
    """
    Mix Transformer encoder.
    Config dicts per variant:
      mit_b0: embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], heads=[1, 2, 5, 8]
      mit_b1: embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], heads=[1, 2, 5, 8]
      mit_b2: embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], heads=[1, 2, 5, 8]
      mit_b3: embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3], heads=[1, 2, 5, 8]
      mit_b4: embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3], heads=[1, 2, 5, 8]
      mit_b5: embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3], heads=[1, 2, 5, 8]
    """

    def __init__(self, variant='mit_b2', in_channels=3, dropout=0.0):
        super().__init__()
        configs = {
            'mit_b0': dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
            'mit_b1': dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
            'mit_b2': dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
            'mit_b3': dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
            'mit_b4': dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
            'mit_b5': dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3],
                           heads=[1, 2, 5, 8], reduction_ratios=[8, 4, 2, 1]),
        }
        cfg = configs[variant]
        self.embed_dims = cfg['embed_dims']
        self.depths = cfg['depths']

        # patch embedding params per stage
        patch_cfgs = [
            dict(patch_size=7, stride=4, padding=3),    # stage 1: H/4, W/4
            dict(patch_size=3, stride=2, padding=1),    # stage 2: H/8, W/8
            dict(patch_size=3, stride=2, padding=1),    # stage 3: H/16, W/16
            dict(patch_size=3, stride=2, padding=1),    # stage 4: H/32, W/32
        ]

        # build stages
        self.stages = nn.ModuleList()
        prev_dim = in_channels
        for i in range(4):
            stage = nn.ModuleList()
            # overlap patch embed
            pe = OverlapPatchEmbed(prev_dim, cfg['embed_dims'][i],
                                   patch_size=patch_cfgs[i]['patch_size'],
                                   stride=patch_cfgs[i]['stride'],
                                   padding=patch_cfgs[i]['padding'],
                                   norm_layer=nn.LayerNorm)
            stage.append(pe)
            # transformer blocks
            for _ in range(cfg['depths'][i]):
                stage.append(TransformerBlock(
                    dim=cfg['embed_dims'][i],
                    num_heads=cfg['heads'][i],
                    reduction_ratio=cfg['reduction_ratios'][i],
                    dropout=dropout,
                ))
            self.stages.append(stage)
            prev_dim = cfg['embed_dims'][i]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for stage in self.stages:
            # first module is patch embed
            x, h, w = stage[0](x)
            # remaining are transformer blocks
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)  # B, N, C
            for block in stage[1:]:
                x = block(x, h, w)
            # reshape back to spatial for next stage
            x = x.transpose(1, 2).reshape(b, c, h, w)
            features.append(x)
        return features  # 4 scales: [H/4, H/8, H/16, H/32]


# ==================== MLP Decoder ====================
class MLPDecoder(nn.Module):
    """
    Lightweight MLP decoder.
    Takes features from all 4 MiT stages → project, upsample, concat → MLP → seg head.
    """

    def __init__(self, embed_dims, num_classes, decoder_dim=256, dropout=0.1):
        super().__init__()
        # linear projections per stage to decoder_dim
        self.linear_c = nn.ModuleList([
            nn.Linear(dim, decoder_dim) for dim in embed_dims
        ])
        # final MLP fusion
        self.linear_fuse = nn.Sequential(
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.seg_head = nn.Linear(decoder_dim, num_classes)

    def forward(self, features):
        # features: list of 4 tensors [B,C1,H/4,W/4], [B,C2,H/8,W/8], ...
        h, w = features[0].shape[2], features[0].shape[3]

        projected = []
        for i, f in enumerate(features):
            b, c, fh, fw = f.shape
            f_flat = f.flatten(2).transpose(1, 2)  # B, N, C
            f_proj = self.linear_c[i](f_flat)       # B, N, decoder_dim
            f_proj = f_proj.transpose(1, 2).reshape(b, -1, fh, fw)  # back to spatial
            # upsample to 1/4 resolution
            if fh != h or fw != w:
                f_proj = F.interpolate(f_proj, size=(h, w), mode='bilinear', align_corners=True)
            projected.append(f_proj)

        # concatenate along channel dim
        fused = torch.cat(projected, dim=1)  # B, decoder_dim*4, H/4, W/4

        # MLP fusion (spatial → channels → spatial)
        b, c, fh, fw = fused.shape
        fused_flat = fused.flatten(2).transpose(1, 2)  # B, N, C*4
        fused_flat = self.linear_fuse(fused_flat)       # B, N, decoder_dim
        out = self.seg_head(fused_flat)                  # B, N, num_classes
        out = out.transpose(1, 2).reshape(b, -1, fh, fw)  # B, num_classes, H/4, W/4

        return out


# ==================== Full SegFormer ====================
class SegFormer(nn.Module):
    """
    SegFormer semantic segmentation model.

    Args:
        num_classes: number of classes (including background)
        variant: MiT variant, one of 'mit_b0' ~ 'mit_b5'
        pretrained: whether to load ImageNet-1k pretrained weights (requires weights file)
        decoder_dim: MLP decoder hidden dimension
        dropout: dropout rate
    """

    def __init__(self, num_classes, variant='mit_b2', pretrained=False,
                 decoder_dim=256, dropout=0.1):
        super().__init__()
        self.variant = variant
        self.encoder = MiT(variant=variant, dropout=dropout)
        self.decoder = MLPDecoder(
            embed_dims=self.encoder.embed_dims,
            num_classes=num_classes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        """
        Load ImageNet-1k pretrained encoder weights.
        Auto-detects format: timm, NVIDIA official, bubliiiing, or raw.
        If mapping fails, prints diagnostics so the user can report the format.
        """
        import os, re
        # try multiple possible filenames
        candidates = [
            f'model_data/segformer_{self.variant}_imagenet.pth',
            f'model_data/segformer_b2_backbone_weights.pth',
        ]
        weight_path = None
        for c in candidates:
            if os.path.exists(c):
                weight_path = c
                break
        if weight_path is None:
            print(f"[SegFormer] [WARN] No pretrained weights found. Tried: {candidates}")
            print(f"[SegFormer] Training from scratch (recommend UnFreeze_Epoch >= 200)")
            return

        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict and isinstance(state_dict['model'], dict):
            # some checkpoints wrap everything in a 'model' key
            state_dict = state_dict['model']
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        our_keys = set(self.encoder.state_dict().keys())
        src_keys = list(state_dict.keys())

        # ---- Strategy 1: try direct match (no prefix stripping) ----
        direct_hits = sum(1 for k in src_keys if k in our_keys)
        # ---- Strategy 2: strip common prefixes ----
        for prefix in ['backbone.', 'encoder.', 'model.', 'module.']:
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            stripped_hits = sum(1 for k in stripped if k in our_keys)
            if stripped_hits > direct_hits:
                state_dict = stripped
                direct_hits = stripped_hits
                print(f"[SegFormer] 去除前缀 '{prefix}' → {direct_hits} 个 key 匹配")

        # ---- Strategy 3: timm format mapping ----
        # timm keys: patch_embed1.proj.weight, block1.0.attn.qkv.weight, norm1.weight
        is_timm = any(re.match(r'(patch_embed|block)\d', k) for k in state_dict.keys())
        if is_timm and direct_hits < 10:
            print(f"[SegFormer] 检测到 timm 格式，执行 key 映射...")
            mapped = self._map_timm_keys(state_dict)
            timm_hits = sum(1 for k in mapped if k in our_keys)
            if timm_hits > direct_hits:
                state_dict = mapped
                direct_hits = timm_hits
                print(f"[SegFormer] timm 映射后 → {direct_hits} 个 key 匹配")

        # ---- Strategy 4: fuzzy match by stage/block numbering ----
        if direct_hits < 10:
            print(f"[SegFormer] 直接匹配仅 {direct_hits} 个 key，尝试模糊匹配...")
            mapped = self._fuzzy_map_keys(state_dict, our_keys)
            fuzzy_hits = sum(1 for k in mapped if k in our_keys)
            if fuzzy_hits > direct_hits:
                state_dict = mapped
                direct_hits = fuzzy_hits
                print(f"[SegFormer] 模糊匹配后 → {direct_hits} 个 key 匹配")

        # ---- apply ----
        encoder_state = self.encoder.state_dict()
        matched = {k: v for k, v in state_dict.items()
                   if k in encoder_state and state_dict[k].shape == encoder_state[k].shape}
        missing, unexpected = self.encoder.load_state_dict(matched, strict=False)

        n_matched = len(matched)
        n_total = len(encoder_state)
        print(f"[SegFormer] Loaded pretrained: {n_matched}/{n_total} encoder keys ({100*n_matched//max(n_total,1)}%)")

        if n_matched < 5:
            # Diagnostic mode: print both sides so user can report
            print(f"\n[SegFormer] [WARN] 预训练权重几乎未加载！格式不匹配。")
            print(f"[SegFormer] === 权重文件前 15 个 key ===")
            for k in sorted(src_keys)[:15]:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'scalar'
                print(f"    {k}  {list(shape)}")
            print(f"[SegFormer] === 模型 encoder 前 15 个 key ===")
            for k in sorted(our_keys)[:15]:
                print(f"    {k}")
            print(f"[SegFormer] 请将以上输出发给我，我来修复映射。")
        elif n_matched < n_total * 0.5:
            print(f"[SegFormer]   Missing keys sample: {sorted(missing)[:5]}...")

    def _fuzzy_map_keys(self, src_dict, our_keys):
        """Fallback: try multiple numbering conventions (stage{N}, stages.{N})"""
        import re
        mapped = {}
        our_list = sorted(our_keys)

        # Build a mapping from (stage, block, component) → our_key
        our_parsed = {}
        for k in our_list:
            m = re.match(r'stages\.(\d+)\.(\d+)\.(.+)', k)
            if m:
                stage, block, comp = int(m.group(1)), int(m.group(2)), m.group(3)
                our_parsed[(stage, block, comp)] = k
            # stage-wise norms
            m = re.match(r'stages\.(\d+)\.0\.(.+)', k)
            if m:
                our_parsed[(int(m.group(1)), 0, m.group(2))] = k

        for k, v in src_dict.items():
            assigned = False
            # Try: stageX.Y.component  (bubliiiing style)
            m = re.match(r'stage(\d+)\.(\d+)\.(.+)', k)
            if m:
                stage, block, comp = int(m.group(1)) - 1, int(m.group(2)), m.group(3)
                target = our_parsed.get((stage, block, comp))
                if target:
                    mapped[target] = v
                    assigned = True
            # Try: blockX.Y.component (timm-like without patch_embed)
            if not assigned:
                m = re.match(r'block(\d+)\.(\d+)\.(.+)', k)
                if m:
                    stage, block, comp = int(m.group(1)) - 1, int(m.group(2)) + 1, m.group(3).replace('mlp.', 'ffn.')
                    target = our_parsed.get((stage, block, comp))
                    if target:
                        mapped[target] = v
                        assigned = True
            # Try: layers.X.Y.component
            if not assigned:
                m = re.match(r'layers\.(\d+)\.(\d+)\.(.+)', k)
                if m:
                    stage, block, comp = int(m.group(1)), int(m.group(2)) + 1, m.group(3)
                    target = our_parsed.get((stage, block, comp))
                    if target:
                        mapped[target] = v
                        assigned = True
            if not assigned:
                # pass through as-is
                mapped[k] = v
        return mapped

    def _map_timm_keys(self, timm_state):
        """
        Map timm mix_transformer keys to our MiT encoder structure.

        timm:  patch_embed{N}.proj.weight   →   stages.{N-1}.0.proj.weight
               block{N}.{i}.attn.q.weight    →   stages.{N-1}.{i+1}.attn.q.weight
               block{N}.{i}.mlp.fc1.weight   →   stages.{N-1}.{i+1}.ffn.fc1.weight
               norm{N}.weight                →   stages.{N-1}.0.norm.weight (inside patch_embed)
        """
        import re
        our_state = self.encoder.state_dict()
        mapped = {}

        for k, v in timm_state.items():
            new_k = None

            # -- patch_embed{N}.xxx → stages.{N-1}.0.xxx
            m = re.match(r'patch_embed(\d+)\.(.+)', k)
            if m:
                stage_idx = int(m.group(1)) - 1
                new_k = f'stages.{stage_idx}.0.{m.group(2)}'

            # -- block{N}.{i}.xxx → stages.{N-1}.{i+1}.xxx
            m = re.match(r'block(\d+)\.(\d+)\.(.+)', k)
            if m:
                stage_idx = int(m.group(1)) - 1
                block_idx = int(m.group(2)) + 1  # stage[0] is patch_embed
                sub = m.group(3)
                # timm uses 'mlp' → we use 'ffn'
                sub = sub.replace('mlp.', 'ffn.')
                # timm uses 'attn.qkv' → we use 'attn.kv' for merged key+value
                # (timm's efficient attention may split differently; handle both)
                new_k = f'stages.{stage_idx}.{block_idx}.{sub}'

            # -- norm{N}.xxx → stages.{N-1}.{block_N}.norm1/norm2.xxx
            # These are the final norms per stage in timm
            m = re.match(r'norm(\d+)\.(.+)', k)
            if m:
                stage_idx = int(m.group(1)) - 1
                # timm's norm per stage — we don't have an exact equivalent in our blocks
                # Skip: these are post-stage norm layers we don't need for segmentation
                pass

            if new_k is not None and new_k in our_state and our_state[new_k].shape == v.shape:
                mapped[new_k] = v

        return mapped

    def forward(self, x):
        _, _, h, w = x.shape
        features = self.encoder(x)
        out = self.decoder(features)
        # upsample to input resolution
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

    def freeze_backbone(self):
        """Freeze encoder for two-stage training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True


# ==================== Helper ====================
SEGFORMER_VARIANTS = ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']


def segformer_b0(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b0', pretrained=pretrained)


def segformer_b1(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b1', pretrained=pretrained)


def segformer_b2(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b2', pretrained=pretrained)


def segformer_b3(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b3', pretrained=pretrained)


def segformer_b4(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b4', pretrained=pretrained)


def segformer_b5(num_classes, pretrained=False):
    return SegFormer(num_classes, variant='mit_b5', pretrained=pretrained)


if __name__ == "__main__":
    # quick test
    model = SegFormer(num_classes=4, variant='mit_b2')
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")
