import random
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange

from .backbone.dinov2.layers import PatchEmbed
from .backbone.croco.misc import freeze_all_params
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import Gaussians
from ..types import Triangles as TrianglesOut
from .backbone import BackboneCfg, get_backbone
from .common.gaussians import matrix_to_quaternion
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .common.triangle_adapter import TriangleAdapter, TriangleAdapterCfg, Triangles
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .layers.transformer_head import TransformerDecoder, LinearPts3d, CrossAttentionDecoder
from .layers.camera_head import CameraHead
from ...loss.loss_pose import se3_inverse
from ...misc.schedule_sample import get_scheduled_sampling_epsilon

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GeometryNormalRefinerCfg:
    enabled: bool = True
    arch: Literal["cnn", "unet"] = "unet"
    output_mode: Literal["residual", "direct"] = "residual"
    detach_point_cloud: bool = True
    use_rgb: bool = True
    use_depth: bool = True
    hidden_dim: int = 64
    num_blocks: int = 3
    num_scales: int = 3
    kernel_size: int = 3
    residual_scale: float = 0.25


@dataclass
class GeometryNormalCfg:
    smooth_kernel: int = 3
    normal_smooth_kernel: int = 5
    normal_smooth_iterations: int = 1
    refiner: GeometryNormalRefinerCfg = field(default_factory=GeometryNormalRefinerCfg)
    anchor_rotation: bool = True
    flip_to_camera: bool = True
    eps: float = 1e-6


@dataclass
class EncoderTrisplatCfg:
    name: Literal["trisplat"]
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_surfaces: int
    geometry_normal: GeometryNormalCfg = field(default_factory=GeometryNormalCfg)
    input_mean: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0)
    input_std: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0)
    pretrained_weights: str = ""
    pose_free: bool = True

    use_checkpoint: bool = False
    freeze: str = 'none'

    gt_pose_sampling_decay_start_step: int = 1000
    gt_pose_sampling_decay_end_step: int = 5000
    gt_pose_final_sample_ratio: float = 0.9

    gaussian_downsample_ratio: int = 1
    gaussians_per_axis: int = 14
    upscale_token_ratio: int = 1

    use_triangle: bool = False
    triangle_adapter: Optional[TriangleAdapterCfg] = None



def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


def _normalize_vector(
    vector: Float[Tensor, "*batch 3"],
    eps: float,
) -> tuple[Float[Tensor, "*batch 3"], Bool[Tensor, "*batch"]]:
    norm = vector.norm(dim=-1, keepdim=True)
    finite_mask = torch.isfinite(vector).all(dim=-1, keepdim=True)
    valid_mask = finite_mask & (norm > eps)
    normalized = torch.where(valid_mask, vector / norm.clamp_min(eps), torch.zeros_like(vector))
    return normalized, valid_mask.squeeze(-1)


def _smooth_normal_field(
    normal_map: Float[Tensor, "batch 3 height width"],
    valid_mask: Bool[Tensor, "batch 1 height width"],
    kernel_size: int,
    iterations: int,
    eps: float,
) -> Float[Tensor, "batch 3 height width"]:
    kernel_size = max(int(kernel_size), 1)
    iterations = max(int(iterations), 0)
    if kernel_size <= 1 or iterations == 0:
        return normal_map
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    batch, _, height, width = normal_map.shape
    valid_mask = valid_mask.bool()
    valid_float = valid_mask.float()
    smoothed = F.normalize(normal_map, dim=1, eps=eps)

    for _ in range(iterations):
        unfolded = F.unfold(
            smoothed,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        ).view(batch, 3, kernel_size * kernel_size, height * width)
        center = smoothed.view(batch, 3, 1, height * width)
        neighbor_valid = F.unfold(
            valid_float,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        ).view(batch, 1, kernel_size * kernel_size, height * width)

        # Only aggregate neighbors with a similar orientation so large planar regions
        # get smoothed while depth/normal discontinuities stay sharp.
        similarity = (unfolded * center).sum(dim=1, keepdim=True).clamp_min(0.0)
        weights = similarity * neighbor_valid
        weight_sum = weights.sum(dim=2).clamp_min(eps)
        smoothed_candidate = (unfolded * weights).sum(dim=2) / weight_sum
        smoothed_candidate = smoothed_candidate.view(batch, 3, height, width)
        smoothed_candidate = F.normalize(smoothed_candidate, dim=1, eps=eps)
        smoothed = torch.where(valid_mask, smoothed_candidate, smoothed)

    return smoothed


def _prepare_teacher_normal(
    teacher_normal: Float[Tensor, "batch view 3 teacher_height teacher_width"],
    out_h: int,
    out_w: int,
    points_hw: Float[Tensor, "batch view height width 3"],
    flip_to_camera: bool,
    eps: float,
) -> tuple[
    Float[Tensor, "batch view height width 3"],
    Bool[Tensor, "batch view height width 1"],
]:
    b, v, _, _, _ = teacher_normal.shape
    teacher_flat = teacher_normal.reshape(b * v, 3, *teacher_normal.shape[-2:])
    teacher_flat = F.interpolate(
        teacher_flat,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    teacher_flat = torch.nan_to_num(teacher_flat, 0.0, 0.0, 0.0)
    teacher_flat = F.normalize(teacher_flat, dim=1, eps=eps)
    teacher_hw = rearrange(teacher_flat, "(b v) c h w -> b v h w c", b=b, v=v)

    if flip_to_camera:
        view_dot = (teacher_hw * points_hw).sum(dim=-1, keepdim=True)
        teacher_hw = torch.where(view_dot > 0, -teacher_hw, teacher_hw)

    teacher_valid = torch.isfinite(teacher_hw).all(dim=-1, keepdim=True)
    teacher_valid = teacher_valid & (teacher_hw.norm(dim=-1, keepdim=True) > eps)
    return teacher_hw, teacher_valid


def _get_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        groups = _get_group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class _ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_blocks: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        groups = _get_group_count(out_channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        ]
        layers.extend(_ResidualConvBlock(out_channels, kernel_size) for _ in range(max(int(num_blocks), 1)))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


class _UNetDecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int, num_blocks: int) -> None:
        super().__init__()
        self.stage = _ConvStage(
            in_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
        )

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
    ) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.stage(torch.cat([x, skip], dim=1))


class NormalRefinementHead(nn.Module):
    def __init__(
        self,
        cfg: GeometryNormalRefinerCfg,
        in_channels: int,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(cfg.hidden_dim), 8)
        kernel_size = max(int(cfg.kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        self.cfg = cfg
        self.use_checkpoint = use_checkpoint
        if cfg.arch == "cnn":
            groups = _get_group_count(hidden_dim)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                *[_ResidualConvBlock(hidden_dim, kernel_size) for _ in range(max(int(cfg.num_blocks), 1))]
            )
        elif cfg.arch == "unet":
            num_scales = max(int(cfg.num_scales), 2)
            channels = [hidden_dim * (2 ** scale_idx) for scale_idx in range(num_scales)]
            self.encoder_stages = nn.ModuleList()
            stage_in_channels = in_channels
            for stage_channels in channels:
                self.encoder_stages.append(
                    _ConvStage(
                        stage_in_channels,
                        stage_channels,
                        kernel_size=kernel_size,
                        num_blocks=cfg.num_blocks,
                    )
                )
                stage_in_channels = stage_channels
            self.bottleneck = _ConvStage(
                channels[-1],
                channels[-1],
                kernel_size=kernel_size,
                num_blocks=cfg.num_blocks,
            )
            self.decoder_stages = nn.ModuleList()
            decoder_in_channels = channels[-1]
            for skip_channels in reversed(channels[:-1]):
                self.decoder_stages.append(
                    _UNetDecoderStage(
                        decoder_in_channels,
                        skip_channels,
                        skip_channels,
                        kernel_size=kernel_size,
                        num_blocks=cfg.num_blocks,
                    )
                )
                decoder_in_channels = skip_channels
        else:
            raise ValueError(f"Unsupported normal refiner arch: {cfg.arch}")

        self.out = nn.Conv2d(hidden_dim, 3, kernel_size=kernel_size, padding=padding)
        if cfg.output_mode == "residual":
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _predict(self, features: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch 3 height width"]:
        if self.cfg.arch == "cnn":
            hidden = self.blocks(self.stem(features))
        elif self.cfg.arch == "unet":
            skips = []
            hidden = features
            for stage_idx, stage in enumerate(self.encoder_stages):
                hidden = stage(hidden)
                skips.append(hidden)
                if stage_idx < len(self.encoder_stages) - 1:
                    hidden = F.avg_pool2d(hidden, kernel_size=2, stride=2)
            hidden = self.bottleneck(hidden)
            for decoder_stage, skip in zip(self.decoder_stages, reversed(skips[:-1])):
                hidden = decoder_stage(hidden, skip)
        else:
            raise ValueError(f"Unsupported normal refiner arch: {self.cfg.arch}")
        return self.out(hidden)

    def forward(
        self,
        base_normal: Float[Tensor, "batch 3 height width"],
        raw_normal: Float[Tensor, "batch 3 height width"],
        valid_mask: Bool[Tensor, "batch 1 height width"],
        rgb: Float[Tensor, "batch 3 height width"] | None = None,
        depth: Float[Tensor, "batch 1 height width"] | None = None,
        eps: float = 1e-6,
    ) -> tuple[
        Float[Tensor, "batch 3 height width"],
        Float[Tensor, "batch 3 height width"],
    ]:
        inputs = [raw_normal, base_normal]
        if self.cfg.use_rgb and rgb is not None:
            inputs.append(rgb)
        if self.cfg.use_depth and depth is not None:
            inputs.append(depth)
        inputs.append(valid_mask.float())

        features = torch.cat(inputs, dim=1)
        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if features.is_cuda
            else nullcontext()
        )
        with autocast_context:
            if self.use_checkpoint and self.training:
                prediction = checkpoint(self._predict, features, use_reentrant=False)
            else:
                prediction = self._predict(features)
        prediction = prediction.float()
        if self.cfg.output_mode == "residual":
            refined = F.normalize(
                base_normal + self.cfg.residual_scale * prediction,
                dim=1,
                eps=eps,
            )
        elif self.cfg.output_mode == "direct":
            refined = F.normalize(prediction, dim=1, eps=eps)
        else:
            raise ValueError(f"Unsupported normal refiner output mode: {self.cfg.output_mode}")
        refined = torch.where(valid_mask, refined, base_normal)
        return refined, prediction


class EncoderTrisplat(Encoder[EncoderTrisplatCfg]):
    backbone: nn.Module

    def __init__(self, cfg: EncoderTrisplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3, use_checkpoint=cfg.use_checkpoint)

        self.pose_free = cfg.pose_free
        self.use_triangle = cfg.use_triangle
        self.normal_refiner: NormalRefinementHead | None = None

        if self.use_triangle:
            assert cfg.triangle_adapter is not None, "triangle_adapter config required when use_triangle=True"
            self.triangle_adapter = TriangleAdapter(cfg.triangle_adapter)
            self.raw_gs_dim = 1 + self.triangle_adapter.d_in
            if cfg.geometry_normal.refiner.enabled:
                refiner_in_channels = 3 + 3 + 1
                if cfg.geometry_normal.refiner.use_rgb:
                    refiner_in_channels += 3
                if cfg.geometry_normal.refiner.use_depth:
                    refiner_in_channels += 1
                self.normal_refiner = NormalRefinementHead(
                    cfg.geometry_normal.refiner,
                    in_channels=refiner_in_channels,
                    use_checkpoint=cfg.use_checkpoint,
                )
        else:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
            self.raw_gs_dim = 1 + self.gaussian_adapter.d_in

        self.patch_size = self.backbone.patch_size

        self.gaussian_downsample_ratio = cfg.gaussian_downsample_ratio
        self.gaussians_per_axis = cfg.gaussians_per_axis
        self.gaussians_per_axis = min(self.gaussians_per_axis, self.patch_size // self.gaussian_downsample_ratio)

        self.upscale_token_ratio = cfg.upscale_token_ratio
        self.head_pathch_size = self.patch_size // self.upscale_token_ratio
        self.position_getter = self.backbone.position_getter

        self.dec_embed_dim = 1024
        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.backbone.rope,
            use_checkpoint=cfg.use_checkpoint,
        )
        self.point_head = LinearPts3d(patch_size=self.patch_size / self.upscale_token_ratio, dec_embed_dim=1024, output_dim=3, downsample_ratio=self.gaussian_downsample_ratio, points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio)

        # ----------------------
        #     Primitive Parameters Decoder
        # ----------------------
        self.gaussian_decoder = deepcopy(self.point_decoder)
        self.gaussian_head = LinearPts3d(patch_size=self.patch_size / self.upscale_token_ratio, dec_embed_dim=1024, output_dim=self.raw_gs_dim, downsample_ratio=self.gaussian_downsample_ratio, points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.backbone.rope,
            use_checkpoint=cfg.use_checkpoint,
        )
        self.camera_head = CameraHead(dim=512)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rgb_embed = PatchEmbed(patch_size=self.patch_size // self.upscale_token_ratio, in_chans=3, embed_dim=2048, norm_layer=norm_layer)
        nn.init.constant_(self.rgb_embed.proj.weight, 0)
        nn.init.constant_(self.rgb_embed.proj.bias, 0)

        # freeze parameters
        self.set_freeze(cfg.freeze)

    def set_freeze(self, freeze):  # this is for use by downstream models
        if freeze == 'none':
            return

        head_modules = [
            self.point_decoder,
            self.point_head,
            self.gaussian_decoder,
            self.gaussian_head,
            self.rgb_embed,
        ]
        if self.normal_refiner is not None:
            head_modules.append(self.normal_refiner)

        to_be_frozen = {
            'none':     [],
            'encoder':     [self.backbone.encoder],
            'decoder':     [self.backbone.decoder, self.backbone.register_token, self.backbone.intrinsics_embed_layer],
            'encoder+decoder': [self.backbone],
            "heads": head_modules,
            'encoder+decoder+point_head': [self.backbone, *head_modules],
            'all': [self]
        }
        freeze_all_params(to_be_frozen[freeze])

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Union[Gaussians, TrianglesOut]:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Encode the context images.
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            hidden, pos, patch_start_idx, x_low, intrinsic_pred = self.backbone(context["image"], context["intrinsics"].clone())
            del x_low

            # hidden shape: (b*v, n, c), pos shape: (b*v, n, 2)
            if self.upscale_token_ratio > 1:
                hidden_aux_token = hidden[:, :patch_start_idx, :]
                hidden_img_token = hidden[:, patch_start_idx:, :]
                hidden_img_token = rearrange(hidden_img_token, "b (h w) c -> b c h w", h=patch_h, w=patch_w)
                hidden_img_token = F.interpolate(hidden_img_token, scale_factor=self.upscale_token_ratio, mode="bilinear", align_corners=False)
                hidden_img_token = rearrange(hidden_img_token, "b c h w -> b (h w) c")
                hidden_upsampled = torch.cat([hidden_aux_token, hidden_img_token], dim=1)

                pos_aux = pos[:, :patch_start_idx]
                pos_img = self.position_getter(b * v, patch_h * self.upscale_token_ratio, patch_w * self.upscale_token_ratio, device=device)
                pos_img = pos_img + 1 if patch_start_idx > 0 else pos_img
                pos_upsampled = torch.cat([pos_aux, pos_img], dim=1)
            else:
                hidden_upsampled = hidden
                pos_upsampled = pos

            rgb = rearrange(context['image'], 'b v c h w -> (b v) c h w')
            rgb_feat = self.rgb_embed(rgb)
            hidden_gaussian = hidden_upsampled.clone()
            hidden_gaussian[:, patch_start_idx:, :] = hidden_gaussian[:, patch_start_idx:, :] + rgb_feat

            point_hidden = self.point_decoder(hidden_upsampled, xpos=pos_upsampled)
            gaussian_hidden = self.gaussian_decoder(hidden_gaussian, xpos=pos_upsampled)
            camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast('cuda', enabled=False):
            out_h, out_w = patch_h * self.gaussians_per_axis, patch_w * self.gaussians_per_axis
            # local points
            point_head_dtype = self.point_head.proj.weight.dtype
            point_hidden = point_hidden.to(point_head_dtype)
            ret = self.point_head([point_hidden[:, patch_start_idx:]], (h, w)).float().reshape(b, v, out_h, out_w, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # primitive parameters
            gaussian_head_dtype = self.gaussian_head.proj.weight.dtype
            gaussian_hidden = gaussian_hidden.to(gaussian_head_dtype)
            gaussian_params = self.gaussian_head([gaussian_hidden[:, patch_start_idx:]], (h, w)).float().reshape(b, v, out_h, out_w, -1)

            gaussian_params = rearrange(gaussian_params, "b v h w d -> (b v) d h w").contiguous()

            # camera
            camera_hidden = camera_hidden.to(next(self.camera_head.parameters()).dtype)
            camera_poses = self.camera_head(camera_hidden[:, patch_start_idx:], patch_h, patch_w).reshape(b, v, 4, 4)  #  c2w

            # convert to the cooridinate system of the first view
            w2c_v1 = se3_inverse(camera_poses[:, 0])
            camera_poses = torch.einsum('bij, bnjk -> bnik', w2c_v1, camera_poses)

            pts_all = rearrange(local_points, "b v h w xyz -> (b v) xyz h w").contiguous()

        # judge if pts_all have 3 dimensions or 4 dimensions
        if pts_all.dim() == 4:
            pts_all = rearrange(pts_all, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            pts_all = rearrange(pts_all, "(b v) d l -> b v l d", b=b, v=v)

        # transform the pts into local coordinate system
        local_pts = pts_all.clone()  # b, v, l, 3

        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)  # depth in a unified coordinate system

        if gaussian_params.dim() == 4:
            raw_params = rearrange(gaussian_params, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            raw_params = rearrange(gaussian_params, "(b v) d l -> b v l d", b=b, v=v)
        raw_params = rearrange(raw_params, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = raw_params[..., 0].sigmoid().unsqueeze(-1)

        if self.pose_free:
            if self.training:
                prob_use_gt_pose = get_scheduled_sampling_epsilon(global_step,
                                                                  epsilon_end=self.cfg.gt_pose_final_sample_ratio,
                                                                  decay_start_step=self.cfg.gt_pose_sampling_decay_start_step,
                                                                  decay_end_step=self.cfg.gt_pose_sampling_decay_end_step, )
                if random.random() < prob_use_gt_pose:
                    c2w = context['extrinsics']
                else:
                    c2w = camera_poses
            else:
                c2w = camera_poses
        else:
            c2w = context['extrinsics']

        if visualization_dump is not None:
            visualization_dump['pred_camera_poses'] = camera_poses.contiguous()
            visualization_dump['c2w'] = camera_poses.contiguous() if self.pose_free else context['extrinsics']
            visualization_dump['intrinsic_pred'] = intrinsic_pred

        if self.use_triangle:
            return self._forward_triangle(
                b, v, h, w, out_h, out_w, patch_h,
                depths, densities, raw_params, c2w,
                context, local_pts, global_step,
                visualization_dump,
            )
        else:
            return self._forward_gaussian(
                b, v, h, w, out_h, out_w, patch_h,
                pts_all, depths, densities, raw_params, c2w,
                context,
                local_pts, global_step,
                visualization_dump,
            )

    def _forward_gaussian(
        self, b, v, h, w, out_h, out_w, patch_h,
        pts_all, depths, densities, raw_params, c2w,
        context,
        local_pts, global_step, visualization_dump,
    ) -> Gaussians:
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(raw_params[..., 1:], "b v r srf c -> b v r srf () c"),
            extrinsics=rearrange(c2w, "b v i j -> b v () () () i j"),
        )

        if visualization_dump is not None:
            vh, vw = patch_h * self.gaussians_per_axis, out_w
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=out_h, w=out_w
            ).contiguous()
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=out_h, w=out_w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=out_h, w=out_w
            )
            visualization_dump['local_pts'] = rearrange(
                local_pts.unsqueeze(-2), "b v (h w) srf xyz -> b v h w srf xyz", h=out_h, w=out_w
            )
            if visualization_dump.get("collect_geom_normal", False):
                (
                    _,
                    geom_normal_cam_pred,
                    geom_normal_cam_forward,
                    geom_normal_mask,
                    geom_rotation_xyzw,
                    geom_normal_cam_raw,
                    geom_normal_cam_base,
                ) = self._build_triangle_geometry_rotation(local_pts, context, out_h, out_w)
                visualization_dump["geom_normal_cam_raw"] = geom_normal_cam_raw.contiguous()
                visualization_dump["geom_normal_cam_base"] = geom_normal_cam_base.contiguous()
                visualization_dump["geom_normal_cam_pred"] = geom_normal_cam_pred.contiguous()
                visualization_dump["geom_normal_cam_forward"] = geom_normal_cam_forward.contiguous()
                visualization_dump["geom_normal_cam"] = geom_normal_cam_forward.contiguous()
                visualization_dump["geom_normal_mask"] = geom_normal_mask.contiguous()
                visualization_dump["geom_rotation_xyzw"] = geom_rotation_xyzw.contiguous()
                visualization_dump["use_triangle"] = False

        return Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
            rearrange(gaussians.rotations, "b v r srf spp d -> b (v r srf spp) d"),
            rearrange(gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.mapped_scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
        )

    def _build_triangle_geometry_rotation(
        self,
        local_pts: Float[Tensor, "batch view pixel 3"],
        context: dict,
        out_h: int,
        out_w: int,
    ) -> tuple[
        Float[Tensor, "batch view pixel 4"],
        Float[Tensor, "batch view 3 height width"],
        Float[Tensor, "batch view 3 height width"],
        Bool[Tensor, "batch view 1 height width"],
        Float[Tensor, "batch view 4 height width"],
        Float[Tensor, "batch view 3 height width"],
        Float[Tensor, "batch view 3 height width"],
    ]:
        cfg = self.cfg.geometry_normal
        refiner_cfg = cfg.refiner
        eps = cfg.eps
        b, v, _, _ = local_pts.shape

        normal_pts = local_pts.detach() if refiner_cfg.detach_point_cloud else local_pts
        point_map_raw = rearrange(normal_pts, "b v (h w) c -> (b v) c h w", h=out_h, w=out_w)
        point_map = point_map_raw
        kernel_size = max(int(cfg.smooth_kernel), 1)
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            point_map = F.avg_pool2d(
                point_map,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )

        padded_raw = F.pad(point_map_raw, (1, 1, 1, 1), mode="replicate")
        dx_raw = padded_raw[:, :, 1:-1, 2:] - padded_raw[:, :, 1:-1, :-2]
        dy_raw = padded_raw[:, :, 2:, 1:-1] - padded_raw[:, :, :-2, 1:-1]

        padded = F.pad(point_map, (1, 1, 1, 1), mode="replicate")
        dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
        dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]

        points_hw_raw = rearrange(point_map_raw, "(b v) c h w -> b v h w c", b=b, v=v)
        points_hw = rearrange(point_map, "(b v) c h w -> b v h w c", b=b, v=v)
        dx_raw = rearrange(dx_raw, "(b v) c h w -> b v h w c", b=b, v=v)
        dy_raw = rearrange(dy_raw, "(b v) c h w -> b v h w c", b=b, v=v)
        dx = rearrange(dx, "(b v) c h w -> b v h w c", b=b, v=v)
        dy = rearrange(dy, "(b v) c h w -> b v h w c", b=b, v=v)

        normal_raw, _ = _normalize_vector(torch.cross(dx_raw, dy_raw, dim=-1), eps)
        if cfg.flip_to_camera:
            view_dot_raw = (normal_raw * points_hw_raw).sum(dim=-1, keepdim=True)
            normal_raw = torch.where(view_dot_raw > 0, -normal_raw, normal_raw)

        normal, normal_valid = _normalize_vector(torch.cross(dx, dy, dim=-1), eps)
        if cfg.flip_to_camera:
            view_dot = (normal * points_hw).sum(dim=-1, keepdim=True)
            normal = torch.where(view_dot > 0, -normal, normal)

        edge_mask = torch.ones((1, 1, out_h, out_w), dtype=torch.bool, device=normal_valid.device)
        edge_mask[..., 0, :] = False
        edge_mask[..., -1, :] = False
        edge_mask[..., :, 0] = False
        edge_mask[..., :, -1] = False
        edge_mask = edge_mask.expand(b, v, -1, -1)
        refiner_valid_mask = normal_valid & edge_mask

        normal_smooth_kernel = max(int(cfg.normal_smooth_kernel), 1)
        normal_smooth_iterations = max(int(cfg.normal_smooth_iterations), 0)
        if normal_smooth_kernel > 1 and normal_smooth_iterations > 0:
            normal = rearrange(normal, "b v h w c -> (b v) c h w")
            normal = _smooth_normal_field(
                normal,
                rearrange(normal_valid, "b v h w -> (b v) () h w"),
                kernel_size=normal_smooth_kernel,
                iterations=normal_smooth_iterations,
                eps=eps,
            )
            normal = rearrange(normal, "(b v) c h w -> b v h w c", b=b, v=v)
        normal_base = normal

        if self.normal_refiner is not None:
            rgb = rearrange(context["image"], "b v c h w -> (b v) c h w").float()
            rgb = F.interpolate(rgb, size=(out_h, out_w), mode="bilinear", align_corners=False)
            base_normal = rearrange(normal_base, "b v h w c -> (b v) c h w")
            raw_normal = rearrange(normal_raw, "b v h w c -> (b v) c h w")
            valid_mask = rearrange(refiner_valid_mask, "b v h w -> (b v) () h w")
            depth = point_map[:, 2:3]
            refined_normal, _ = self.normal_refiner(
                base_normal=base_normal,
                raw_normal=raw_normal,
                valid_mask=valid_mask,
                rgb=rgb,
                depth=depth,
                eps=eps,
            )
            normal = rearrange(refined_normal, "(b v) c h w -> b v h w c", b=b, v=v)
            if cfg.flip_to_camera:
                view_dot = (normal * points_hw).sum(dim=-1, keepdim=True)
                normal = torch.where(view_dot > 0, -normal, normal)

        pred_normal = normal
        forward_normal = pred_normal

        teacher_normal = context.get("mono_normal")
        teacher_blend_alpha = float(context.get("mono_normal_blend_alpha", 0.0))
        if teacher_normal is not None and teacher_blend_alpha > 0.0:
            teacher_normal_hw, teacher_valid = _prepare_teacher_normal(
                teacher_normal=teacher_normal,
                out_h=out_h,
                out_w=out_w,
                points_hw=points_hw,
                flip_to_camera=cfg.flip_to_camera,
                eps=eps,
            )
            blend_mask = teacher_valid & rearrange(refiner_valid_mask, "b v h w -> b v h w ()")
            blended_normal = F.normalize(
                teacher_blend_alpha * teacher_normal_hw
                + (1.0 - teacher_blend_alpha) * pred_normal,
                dim=-1,
                eps=eps,
            )
            forward_normal = torch.where(blend_mask, blended_normal, pred_normal)
            if cfg.flip_to_camera:
                view_dot_forward = (forward_normal * points_hw).sum(dim=-1, keepdim=True)
                forward_normal = torch.where(view_dot_forward > 0, -forward_normal, forward_normal)

        tangent = dx - (dx * forward_normal).sum(dim=-1, keepdim=True) * forward_normal
        tangent, tangent_valid = _normalize_vector(tangent, eps)

        ref_x = torch.zeros_like(forward_normal)
        ref_x[..., 0] = 1.0
        tangent_ref = ref_x - (ref_x * forward_normal).sum(dim=-1, keepdim=True) * forward_normal
        tangent_ref, tangent_ref_valid = _normalize_vector(tangent_ref, eps)

        ref_y = torch.zeros_like(forward_normal)
        ref_y[..., 1] = 1.0
        tangent_ref_y = ref_y - (ref_y * forward_normal).sum(dim=-1, keepdim=True) * forward_normal
        tangent_ref_y, tangent_ref_y_valid = _normalize_vector(tangent_ref_y, eps)

        tangent = torch.where(tangent_valid[..., None], tangent, tangent_ref)
        tangent_valid = tangent_valid | tangent_ref_valid
        tangent = torch.where(tangent_valid[..., None], tangent, tangent_ref_y)
        tangent_valid = tangent_valid | tangent_ref_y_valid

        bitangent, bitangent_valid = _normalize_vector(torch.cross(forward_normal, tangent, dim=-1), eps)
        tangent, tangent_reproj_valid = _normalize_vector(torch.cross(bitangent, forward_normal, dim=-1), eps)

        valid_mask = normal_valid & tangent_valid & bitangent_valid & tangent_reproj_valid
        valid_mask = valid_mask & edge_mask

        rotation_matrix = torch.stack([tangent, bitangent, forward_normal], dim=-1)
        quat_xyzw = matrix_to_quaternion(rotation_matrix)
        quat_wxyz = torch.roll(quat_xyzw, shifts=1, dims=-1)
        quat_wxyz = rearrange(quat_wxyz, "b v h w c -> b v (h w) c")

        return (
            quat_wxyz,
            rearrange(pred_normal, "b v h w c -> b v c h w"),
            rearrange(forward_normal, "b v h w c -> b v c h w"),
            rearrange(valid_mask, "b v h w -> b v () h w"),
            rearrange(quat_xyzw, "b v h w c -> b v c h w"),
            rearrange(normal_raw, "b v h w c -> b v c h w"),
            rearrange(normal_base, "b v h w c -> b v c h w"),
        )

    def _forward_triangle(
        self, b, v, h, w, out_h, out_w, patch_h,
        depths, densities, raw_params, c2w,
        context, local_pts, global_step,
        visualization_dump,
    ) -> TrianglesOut:
        opacities = self.map_pdf_to_opacity(densities, global_step)

        # depths shape: (b, v, h*w, srf, 1) -> squeeze for triangle adapter
        depths_flat = depths.squeeze(-1).squeeze(-1)  # (b, v, h*w)
        opacities_flat = opacities.squeeze(-1).squeeze(-1)  # (b, v, h*w)
        raw_tri_params = raw_params[..., 1:]  # remove density channel
        raw_tri_flat = raw_tri_params.squeeze(-2).clone()  # (b, v, h*w, d_in)

        (
            geom_rotation_wxyz,
            geom_normal_cam_pred,
            geom_normal_cam_forward,
            geom_normal_mask,
            geom_rotation_xyzw,
            geom_normal_cam_raw,
            geom_normal_cam_base,
        ) = self._build_triangle_geometry_rotation(local_pts, context, out_h, out_w)
        if self.cfg.geometry_normal.anchor_rotation:
            raw_rotation = raw_tri_flat[..., 3:7]
            rotation_mask = rearrange(geom_normal_mask, "b v () h w -> b v (h w) 1")
            raw_tri_flat[..., 3:7] = torch.where(rotation_mask, geom_rotation_wxyz, raw_rotation)

        # Generate normalized pixel coordinates for the output grid
        coords_h = torch.linspace(0.5 / out_h, 1.0 - 0.5 / out_h, out_h, device=depths.device)
        coords_w = torch.linspace(0.5 / out_w, 1.0 - 0.5 / out_w, out_w, device=depths.device)
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        coordinates = torch.stack([grid_x, grid_y], dim=-1)  # (out_h, out_w, 2)
        coordinates = coordinates.reshape(out_h * out_w, 2)  # (h*w, 2)
        coordinates = coordinates.unsqueeze(0).unsqueeze(0).expand(b, v, -1, -1)  # (b, v, h*w, 2)

        # Expand extrinsics/intrinsics to match spatial dims
        c2w_expanded = rearrange(c2w, "b v i j -> b v () i j").expand(b, v, out_h * out_w, 4, 4)
        intrinsics = context['intrinsics']
        intrinsics_expanded = rearrange(intrinsics, "b v i j -> b v () i j").expand(b, v, out_h * out_w, 3, 3)

        triangles = self.triangle_adapter.forward(
            extrinsics=c2w_expanded,
            intrinsics=intrinsics_expanded,
            coordinates=coordinates,
            depths=depths_flat,
            opacities=opacities_flat,
            raw_triangles=raw_tri_flat,
            image_shape=(h, w),
            global_step=global_step,
        )

        geom_normal_world = rearrange(geom_normal_cam_forward, "b v c h w -> b v h w c")
        rotation_cam2world = c2w[:, :, None, None, :3, :3]
        geom_normal_world = torch.matmul(
            rotation_cam2world,
            geom_normal_world.unsqueeze(-1),
        ).squeeze(-1)
        geom_normal_world = F.normalize(
            geom_normal_world,
            dim=-1,
            eps=self.cfg.geometry_normal.eps,
        )
        geom_normal_valid = rearrange(geom_normal_mask, "b v () h w -> b v h w 1")
        geom_normal_world = torch.where(
            geom_normal_valid,
            geom_normal_world,
            torch.zeros_like(geom_normal_world),
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=out_h, w=out_w
            ).contiguous()
            visualization_dump['local_pts'] = rearrange(
                local_pts.unsqueeze(-2), "b v (h w) srf xyz -> b v h w srf xyz", h=out_h, w=out_w
            )
            visualization_dump["geom_normal_cam_raw"] = geom_normal_cam_raw.contiguous()
            visualization_dump["geom_normal_cam_base"] = geom_normal_cam_base.contiguous()
            visualization_dump["geom_normal_cam_pred"] = geom_normal_cam_pred.contiguous()
            visualization_dump["geom_normal_cam_forward"] = geom_normal_cam_forward.contiguous()
            visualization_dump["geom_normal_cam"] = geom_normal_cam_forward.contiguous()
            visualization_dump["geom_normal_mask"] = geom_normal_mask.contiguous()
            visualization_dump["geom_rotation_xyzw"] = geom_rotation_xyzw.contiguous()
            visualization_dump['use_triangle'] = True

        return TrianglesOut(
            vertices=rearrange(triangles.vertices, "b v n d1 d2 -> b (v n) d1 d2"),
            sigma=rearrange(triangles.sigma, "b v n d -> b (v n) d"),
            opacity=rearrange(triangles.opacity, "b v n d -> b (v n) d"),
            features=rearrange(triangles.features, "b v n d -> b (v n) d"),
            centers=rearrange(triangles.centers, "b v n d -> b (v n) d") if triangles.centers is not None else None,
            normals=rearrange(geom_normal_world, "b v h w c -> b (v h w) c"),
            scales=rearrange(triangles.scales, "b v n d -> b (v n) d") if triangles.scales is not None else None,
            mapped_scales=rearrange(triangles.mapped_scales, "b v n d -> b (v n) d") if triangles.mapped_scales is not None else None,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
