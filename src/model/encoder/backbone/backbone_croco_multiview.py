from copy import deepcopy

import torch
from einops import rearrange
from torch import nn

from .backbone_croco import BackboneCrocoCfg
from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params
from .croco.patch_embed import get_patch_embed
from ....geometry.camera_emb import get_intrinsic_embedding

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}


class AsymmetricCroCoMulti(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)
        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1]
            feat_current_ctx = generate_ctx_views(feat_current)
            # img1 side
            f1, _ = blk1(feat_current[:, 0].contiguous(), feat_current_ctx[:, 0].contiguous(), pose[:, 0].contiguous(), pos_ctx[:, 0].contiguous())
            f1 = f1.unsqueeze(1)
            # img2 side
            f2, _ = blk2(rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(pose[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"))
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v-1)
            # store the result
            final_output.append(torch.cat((f1, f2), dim=1))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        return final_output

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2)

        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            intrinsic_embedding_all = rearrange(intrinsic_embedding, "b v c -> (b v) c").unsqueeze(1)

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1)

        feat, pose, _ = self._encode_image(images_all, shape_all, intrinsic_embedding_all)

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pose = rearrange(pose, "(b v) l c -> b v l c", b=b, v=v)

        # step 2: decoder
        dec_feat = self._decoder(feat, pose)
        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :-1]

        return dec_feat, shape, images

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
