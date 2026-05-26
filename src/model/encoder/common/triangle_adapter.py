from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays


@dataclass
class Triangles:
    """Intermediate triangle representation with variadic batch dims,
    matching the GaussianAdapter local Gaussians pattern."""
    vertices: Float[Tensor, "*batch 3 3"]
    sigma: Float[Tensor, "*batch 1"]
    opacity: Float[Tensor, "*batch 1"]
    features: Float[Tensor, "*batch _"]
    centers: Optional[Float[Tensor, "*batch 3"]] = None
    scales: Optional[Float[Tensor, "*batch 3"]] = None
    mapped_scales: Optional[Float[Tensor, "*batch 3"]] = None


def quaternion_apply(quaternion, v):
    w, x, y, z = quaternion.unbind(dim=-1)
    vx, vy, vz = v.unbind(dim=-1)

    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)

    out_x = vx + w * tx + (y * tz - z * ty)
    out_y = vy + w * ty + (z * tx - x * tz)
    out_z = vz + w * tz + (x * ty - y * tx)

    return torch.stack([out_x, out_y, out_z], dim=-1)


@dataclass
class TriangleAdapterCfg:
    triangle_scale_min: float
    triangle_scale_max: float
    sh_degree: int = 0
    sigma_scale_initial: float = 1.0
    sigma_scale_final: float = 0.1
    sigma_warmup_steps: int = 5000
    coverage_scale_boost: float = 0.0
    coverage_opacity_threshold: float = 0.0


class TriangleAdapter(nn.Module):
    cfg: TriangleAdapterCfg

    def __init__(self, cfg: TriangleAdapterCfg):
        super().__init__()
        self.cfg = cfg

        self.d_sh = (cfg.sh_degree + 1) ** 2
        self.sh_dim = 3 * self.d_sh

        self.register_buffer('canonical_triangle', torch.tensor([
            [0.0, 0.57735, 0.0],
            [-0.5, -0.28868, 0.0],
            [0.5, -0.28868, 0.0]
        ], dtype=torch.float32) * 4)

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    @property
    def d_in(self) -> int:
        return 3 + 4 + self.sh_dim + 1

    @staticmethod
    def _ensure_finite(name: str, tensor: Tensor) -> None:
        if torch.isfinite(tensor).all():
            return

        flat = tensor.detach().reshape(-1).float()
        finite_mask = torch.isfinite(flat)
        raise FloatingPointError(
            f"[triangle adapter] non-finite {name}: "
            f"shape={tuple(tensor.shape)} "
            f"finite={int(finite_mask.sum().item())}/{flat.numel()}"
        )

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_triangles: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        global_step: int = 0,
        eps: float = 1e-8,
    ) -> Triangles:
        device = extrinsics.device
        h, w = image_shape

        self._ensure_finite("depths", depths)
        self._ensure_finite("opacities", opacities)
        self._ensure_finite("raw_triangles", raw_triangles)

        scales, rotations, sh, sigma = raw_triangles.split(
            (3, 4, self.sh_dim, 1), dim=-1
        )

        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        centers = origins + directions * depths[..., None]

        scale_min = self.cfg.triangle_scale_min
        scale_max = self.cfg.triangle_scale_max
        mapped_scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        self._ensure_finite("mapped_scales", mapped_scales)

        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = mapped_scales * depths[..., None] * multiplier[..., None]
        self._ensure_finite("scales", scales)

        if self.cfg.coverage_scale_boost > 0 and self.cfg.coverage_opacity_threshold > 0:
            low_conf_ratio = (
                (self.cfg.coverage_opacity_threshold - opacities).clamp(min=0.0)
                / self.cfg.coverage_opacity_threshold
            )
            scales = scales * (1.0 + self.cfg.coverage_scale_boost * low_conf_ratio[..., None])
            self._ensure_finite("coverage_boosted_scales", scales)

        rotations = F.normalize(rotations, p=2, dim=-1)
        self._ensure_finite("rotations", rotations)

        c2w_rotations = extrinsics[..., :3, :3]

        sh = rearrange(sh, "... (d_sh xyz) -> ... d_sh xyz", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, self.d_sh, 3)) * self.sh_mask.unsqueeze(-1)

        if global_step >= self.cfg.sigma_warmup_steps:
            current_sigma_scale = self.cfg.sigma_scale_final
        else:
            progress = global_step / self.cfg.sigma_warmup_steps
            current_sigma_scale = self.cfg.sigma_scale_initial + \
                (self.cfg.sigma_scale_final - self.cfg.sigma_scale_initial) * progress

        sigma = torch.sigmoid(sigma) * current_sigma_scale + eps
        self._ensure_finite("sigma", sigma)

        batch_shape = centers.shape[:-1]
        self._ensure_finite("centers", centers)

        canonical = self.canonical_triangle.view(
            *([1] * len(batch_shape)), 3, 3
        ).expand(*batch_shape, 3, 3)

        vertices = canonical * scales.unsqueeze(-2)
        self._ensure_finite("vertices_local", vertices)

        flat_shape = batch_shape + (3,)
        vertices_flat = vertices.reshape(*flat_shape, 3)

        rotation_expanded = rotations.unsqueeze(-2).expand(*batch_shape, 3, 4)

        vertices_rotated = quaternion_apply(rotation_expanded, vertices_flat)
        self._ensure_finite("vertices_rotated", vertices_rotated)

        c2w_expanded = c2w_rotations.unsqueeze(-3).expand(*batch_shape, 3, 3, 3)
        vertices_world = torch.einsum('...ij,...j->...i', c2w_expanded, vertices_rotated)
        self._ensure_finite("vertices_world", vertices_world)

        vertices = vertices_world + centers.unsqueeze(-2)
        self._ensure_finite("vertices", vertices)

        opacity = opacities.unsqueeze(-1)
        self._ensure_finite("opacity", opacity)

        sh_flat = sh.flatten(start_dim=-2)
        self._ensure_finite("features", sh_flat)

        return Triangles(
            vertices=vertices,
            sigma=sigma,
            opacity=opacity,
            features=sh_flat,
            centers=centers,
            scales=scales,
            mapped_scales=mapped_scales,
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)
