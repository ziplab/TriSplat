import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from dataclasses import dataclass
from typing import Literal

from einops import rearrange

from src.geometry.camera_emb import get_intrinsic_embedding_new
from .dinov2.layers import Mlp, PatchEmbed
from ..layers.pos_embed import RoPE2D, PositionGetter
from ..layers.block import BlockRope
from ..layers.attention import FlashAttentionRope
from .dinov2.hub.backbones import dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


@dataclass
class BackboneLocalGlobalCfg:
    name: Literal["local_global"]
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token", "none"] = 'token'
    predict_intrinsics: bool = False
    use_pred_intrinsics_for_embed: bool = False
    pred_intrinsics_min_focal: float = 1e-6

class BackboneLocalGlobal(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            cfg: BackboneLocalGlobalCfg,
            d_in,
            pos_type='rope100',
            decoder_size='large',
            use_checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # ----------------------
        #        Encoder
        # ----------------------
        self.patch_size = 14
        self.encoder = dinov2_vitl14_reg(pretrained=False, patch_size=self.patch_size)
        del self.encoder.mask_token

        # ----------------------
        #        Intrinsic head
        # ----------------------
        self.predict_intrinsics = cfg.predict_intrinsics
        if cfg.predict_intrinsics:
            self.intrinsic_head = Mlp(1024, hidden_features=1024, out_features=2)  # output fx, fy

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):  # eg rope100
            if RoPE2D is None: raise ImportError(
                "Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4.0
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4.0
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4.0
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
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
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #     Intrinsic embedding
        # ----------------------
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.use_pred_intrinsics_for_embed = cfg.use_pred_intrinsics_for_embed
        self.pred_intrinsics_min_focal = cfg.pred_intrinsics_min_focal
        if self.intrinsics_embed_type == 'pixelwise':
            print("Use pixelwise intrinsics embedding.")
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
            self.intrinsics_embed_layer = PatchEmbed(patch_size=self.patch_size,
                                                     in_chans=self.intrinsics_embed_decoder_dim,
                                                     embed_dim=dec_embed_dim,
                                                     norm_layer=partial(nn.LayerNorm, eps=1e-6))

            # zero init
            nn.init.constant_(self.intrinsics_embed_layer.proj.weight, 0)
            nn.init.constant_(self.intrinsics_embed_layer.proj.bias, 0)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if self.training and self.use_checkpoint:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def forward(self, imgs, intrinsics=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape

        # encode by dinov2
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        x_low = hidden['x_low']

        intrinsic_pred = None
        if self.predict_intrinsics:
            x_norm_clstoken = hidden['x_norm_clstoken']  # (B*N, 1024)
            intrinsic_pred = self.intrinsic_head(x_norm_clstoken)
            intrinsic_pred = F.relu(intrinsic_pred)

            # use predicted intrinsics as condition
            if self.use_pred_intrinsics_for_embed:
                focal_pred = rearrange(intrinsic_pred, '(b v) d -> b v d', b=B, v=N)
                min_focal = self.pred_intrinsics_min_focal
                intrinsics = intrinsics.clone()
                fx_pred = focal_pred[:, :, 0].to(intrinsics.dtype)
                fy_pred = focal_pred[:, :, 1].to(intrinsics.dtype)
                fx_gt = intrinsics[:, :, 0, 0]
                fy_gt = intrinsics[:, :, 1, 1]
                valid_fx = torch.isfinite(fx_pred) & (fx_pred > min_focal)
                valid_fy = torch.isfinite(fy_pred) & (fy_pred > min_focal)
                intrinsics[:, :, 0, 0] = torch.where(valid_fx, fx_pred, fx_gt).clamp_min(min_focal)
                intrinsics[:, :, 1, 1] = torch.where(valid_fy, fy_pred, fy_gt).clamp_min(min_focal)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # add intrinsic embedding here
        if self.intrinsics_embed_type == 'pixelwise':
            intrinsic_emb = get_intrinsic_embedding_new(intrinsics, imgs, degree=self.intrinsics_embed_degree)
            intrinsic_emb = self.intrinsics_embed_layer(intrinsic_emb)
            hidden = hidden + intrinsic_emb
        else:
            raise NotImplementedError

        hidden, pos = self.decode(hidden, N, H, W)
        return hidden, pos, self.patch_start_idx, x_low, intrinsic_pred
