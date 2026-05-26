from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..backbone.dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4.,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4.,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, encoder_out, pos=None, ctx_pos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, y=encoder_out, xpos=pos, ypos=ctx_pos, use_reentrant=False)
            else:
                hidden = blk(hidden, y=encoder_out, xpos=pos, ypos=ctx_pos)
        out = self.linear_out(hidden)
        return out


class LinearPts3d (nn.Module):
    """ 
    Linear head
    Each token outputs: (self.patch_size // self.downsample_ratio)² points
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, downsample_ratio=1, points_per_axis=None):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        # Output points per token after downsampling
        points_per_token = (self.patch_size // downsample_ratio) ** 2 if points_per_axis is None else points_per_axis ** 2
        self.points_per_axis = self.patch_size // downsample_ratio if points_per_axis is None else points_per_axis
        self.proj = nn.Linear(dec_embed_dim, output_dim * points_per_token)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # Calculate upsampling factor
        upsample_factor = self.points_per_axis

        # Project tokens
        feat = self.proj(tokens)  # B, S, (output_dim * upsample_factor²)

        # Reshape to prepare for pixel_shuffle
        H_patches = int(H // self.patch_size)
        W_patches = int(W // self.patch_size)
        feat = feat.view(B, H_patches, W_patches, -1)
        feat = feat.permute(0, 3, 1, 2)  # B, C, H_patches, W_patches

        # Use pixel_shuffle to upsample
        feat = F.pixel_shuffle(feat, upsample_factor)  # B, output_dim, H//downsample_ratio, W//downsample_ratio

        return feat.permute(0, 2, 3, 1)
