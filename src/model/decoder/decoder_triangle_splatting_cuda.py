import logging
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ..types import Triangles
from .cuda_triangle_splatting import render_triangle_cuda
from .decoder import Decoder, DecoderOutput

logger = logging.getLogger(__name__)


@dataclass
class DecoderTriangleSplattingCUDACfg:
    name: Literal["triangle_splatting_cuda"]
    background_color: list[float] | None = None
    opacity_temp_initial: float = 1.0
    opacity_temp_final: float = 25.0
    opacity_temp_warmup_steps: int = 5000
    alpha_floor_min: float = 0.0
    alpha_floor_warmup_steps: int = 0
    sh_degree: int = 0
    prune_opacity_threshold: float = 0.0

    def __post_init__(self):
        if self.background_color is None:
            self.background_color = [0.0, 0.0, 0.0]


class DecoderTriangleSplattingCUDA(Decoder[DecoderTriangleSplattingCUDACfg]):

    def __init__(
        self,
        cfg: DecoderTriangleSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        triangles: Triangles,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode=None,
        global_step: int = 0,
        debug_log_interval: int | None = None,
        **kwargs,
    ) -> DecoderOutput:
        return_triangle_visibility_mask = bool(kwargs.pop("return_triangle_visibility_mask", False))
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()
        triangles = Triangles(
            vertices=triangles.vertices.float(),
            sigma=triangles.sigma.float(),
            opacity=triangles.opacity.float(),
            features=triangles.features.float(),
            centers=None if triangles.centers is None else triangles.centers.float(),
            normals=None if triangles.normals is None else triangles.normals.float(),
            scales=None if triangles.scales is None else triangles.scales.float(),
            mapped_scales=None if triangles.mapped_scales is None else triangles.mapped_scales.float(),
            primitive_valid_mask=triangles.primitive_valid_mask,
        )

        if global_step >= self.cfg.opacity_temp_warmup_steps:
            current_temperature = self.cfg.opacity_temp_final
        else:
            progress = global_step / self.cfg.opacity_temp_warmup_steps
            current_temperature = self.cfg.opacity_temp_initial + (
                self.cfg.opacity_temp_final - self.cfg.opacity_temp_initial
            ) * progress

        render_opacity = triangles.opacity.float().clamp(min=1e-6, max=1 - 1e-6)
        render_opacity = torch.sigmoid(
            torch.logit(render_opacity) * current_temperature
        ).float()
        alpha_floor_active = (
            self.cfg.alpha_floor_min > 0
            and global_step < self.cfg.alpha_floor_warmup_steps
        )
        if alpha_floor_active:
            render_opacity = render_opacity.clamp_min(self.cfg.alpha_floor_min)

        should_log_debug = self.should_log_debug_stats(global_step, debug_log_interval)
        if should_log_debug:
            logger.info(
                "[decoder debug] step=%s triangle opacity_temperature=%.6g alpha_floor=%s",
                global_step,
                current_temperature,
                self.cfg.alpha_floor_min if alpha_floor_active else 0.0,
            )
            self.log_debug_tensor_stats("triangle/input_scales", triangles.mapped_scales, global_step)
            self.log_debug_tensor_stats("triangle/render_scales", triangles.scales, global_step)
            self.log_debug_tensor_stats("triangle/input_sigma", triangles.sigma, global_step)
            self.log_debug_tensor_stats("triangle/input_opacity", triangles.opacity, global_step)
            self.log_debug_tensor_stats(
                "triangle/render_input_opacity",
                render_opacity,
                global_step,
            )

        return render_triangle_cuda(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=image_shape,
            vertices=triangles.vertices,
            opacity=triangles.opacity,
            sigma=triangles.sigma,
            features=triangles.features,
            background_color=self.background_color,
            global_step=global_step,
            opacity_temp_initial=self.cfg.opacity_temp_initial,
            opacity_temp_final=self.cfg.opacity_temp_final,
            opacity_temp_warmup_steps=self.cfg.opacity_temp_warmup_steps,
            alpha_floor_min=self.cfg.alpha_floor_min,
            alpha_floor_warmup_steps=self.cfg.alpha_floor_warmup_steps,
            sh_degree=self.cfg.sh_degree,
            log_render_stats=should_log_debug,
            return_triangle_visibility_mask=return_triangle_visibility_mask,
        )
