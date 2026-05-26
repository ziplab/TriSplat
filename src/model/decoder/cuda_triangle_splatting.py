import logging
import math
import torch
from torch import Tensor
from jaxtyping import Float
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
import torch.nn.functional as F
from .decoder import DecoderOutput

logger = logging.getLogger(__name__)


VISIBLE_OPACITY_THRESHOLDS = (1e-3, 1e-2, 1e-1)


def depth_to_normal(depth, K):
    B, _, H, W = depth.shape

    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing='ij'
    )
    x = x.unsqueeze(0).expand(B, -1, -1).float()
    y = y.unsqueeze(0).expand(B, -1, -1).float()

    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    X = (x - cx) * depth.squeeze(1) / fx
    Y = -(y - cy) * depth.squeeze(1) / fy
    Z = depth.squeeze(1)

    XYZ = torch.stack([X, Y, Z], dim=1)

    padded_XYZ = F.pad(XYZ, (1, 1, 1, 1), mode='replicate')
    dX = padded_XYZ[:, :, 1:-1, 2:] - padded_XYZ[:, :, 1:-1, :-2]
    dY = padded_XYZ[:, :, 2:, 1:-1] - padded_XYZ[:, :, :-2, 1:-1]

    normal = torch.cross(dX, dY, dim=1)

    norm = torch.norm(normal, dim=1, keepdim=True)
    normal = normal / (norm + 1e-8)

    return normal


def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
    K: torch.Tensor,
    H: int,
    W: int,
    device: torch.device
) -> torch.Tensor:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    z_sign = 1.0

    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H

    P[0, 2] = (2.0 * cx / W) - 1.0
    P[1, 2] = (2.0 * cy / H) - 1.0

    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def render_triangle_cuda(
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    image_shape: tuple[int, int],
    vertices: Float[Tensor, "batch num_tris 3 3"],
    opacity: Float[Tensor, "batch num_tris 1"],
    sigma: Float[Tensor, "batch num_tris 1"],
    features: Float[Tensor, "batch num_tris c"],
    background_color: Float[Tensor, "3"],
    global_step: int = 0,
    opacity_temp_initial: float = 1.0,
    opacity_temp_final: float = 25.0,
    opacity_temp_warmup_steps: int = 5000,
    alpha_floor_min: float = 0.0,
    alpha_floor_warmup_steps: int = 0,
    sh_degree: int = 0,
    log_render_stats: bool = False,
    return_triangle_visibility_mask: bool = False,
) -> DecoderOutput:
    B, V, _, _ = extrinsics.shape
    H, W = image_shape
    device = vertices.device

    all_images_list = []
    all_rend_normals_list = []
    all_surf_normals_list = []
    all_depths_list = []
    all_opacities_list = []
    all_proj_matrices_list = []
    all_visibility_masks_list = []
    rendered_triangles_per_view = []
    rendered_triangles_unique_per_batch = []
    rendered_visible_opacity_per_view = {
        threshold: [] for threshold in VISIBLE_OPACITY_THRESHOLDS
    }
    culled_reason_counts: dict[int, int] = {}

    w2c = torch.linalg.inv(extrinsics.float())

    fx = intrinsics[..., 0, 0] * W
    fy = intrinsics[..., 1, 1] * H
    fov_x = 2 * torch.atan(W / (2 * fx))
    fov_y = 2 * torch.atan(H / (2 * fy))

    for b in range(B):
        batch_images = []
        batch_rend_normals = []
        batch_surf_normals = []
        batch_depths = []
        batch_opacities = []
        batch_proj_matrices = []
        batch_visible_masks = []

        cur_vertices = vertices[b].contiguous().view(-1)

        if global_step >= opacity_temp_warmup_steps:
            current_temperature = opacity_temp_final
        else:
            progress = global_step / opacity_temp_warmup_steps
            current_temperature = opacity_temp_initial + (opacity_temp_final - opacity_temp_initial) * progress

        opacity_clamped = opacity[b].clamp(min=1e-6, max=1-1e-6)
        opacity_logits = torch.logit(opacity_clamped)
        cur_opacity = torch.sigmoid(opacity_logits * current_temperature).float()
        if alpha_floor_min > 0 and global_step < alpha_floor_warmup_steps:
            cur_opacity = cur_opacity.clamp_min(alpha_floor_min)
        cur_sigma = sigma[b].float()

        d_sh = (sh_degree + 1) ** 2
        cur_features = features[b].float()

        if cur_features.shape[-1] == d_sh * 3:
            cur_shs = cur_features.reshape(-1, d_sh, 3).contiguous()
        else:
            cur_shs = None
            cur_colors = torch.sigmoid(cur_features[:, :3]).float() if cur_features.shape[-1] >= 3 else torch.sigmoid(cur_features).float()

        num_triangles = cur_opacity.shape[0]

        scaling = torch.zeros((num_triangles), device=device, dtype=torch.float32)
        density = torch.zeros((num_triangles), device=device, dtype=torch.float32)
        means2D = torch.zeros((num_triangles, 2), device=device, dtype=torch.float32)

        num_points_per_triangle = torch.full((num_triangles,), 3, dtype=torch.int32, device=device)
        cumsum_of_points_per_triangle = torch.arange(0, num_triangles * 3, 3, dtype=torch.int32, device=device)
        primitive_count = num_triangles

        for v in range(V):
            view_mat = w2c[b, v].transpose(0, 1).float()

            cur_fovx = fov_x[b, v].item()
            cur_fovy = fov_y[b, v].item()

            cur_K = intrinsics[b, v].clone()
            cur_K[0, :] *= W
            cur_K[1, :] *= H

            proj_mat = get_projection_matrix(
                znear=0.01, zfar=1000.0,
                fovX=cur_fovx, fovY=cur_fovy,
                K=cur_K, H=H, W=W, device=device
            )

            full_proj_mat = (proj_mat @ w2c[b, v]).transpose(0, 1).float()

            campos = extrinsics[b, v, :3, 3].float()
            bg_color_f32 = background_color.float()

            cur_tanfovx = math.tan(cur_fovx * 0.5)
            cur_tanfovy = math.tan(cur_fovy * 0.5)

            raster_settings = TriangleRasterizationSettings(
                image_height=H,
                image_width=W,
                tanfovx=cur_tanfovx,
                tanfovy=cur_tanfovy,
                bg=bg_color_f32,
                scale_modifier=1.0,
                viewmatrix=view_mat,
                projmatrix=full_proj_mat,
                sh_degree=sh_degree,
                campos=campos,
                prefiltered=False,
                debug=False
            )

            rasterizer = TriangleRasterizer(raster_settings)

            if cur_shs is not None:
                render_image, radii, scaling_map, density_map, allmap, max_blending = rasterizer(
                    triangles_points=cur_vertices,
                    sigma=cur_sigma,
                    num_points_per_triangle=num_points_per_triangle,
                    cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                    number_of_points=primitive_count,
                    opacities=cur_opacity,
                    means2D=means2D,
                    scaling=scaling,
                    density_factor=density,
                    shs=cur_shs,
                    colors_precomp=None
                )
            else:
                render_image, radii, scaling_map, density_map, allmap, max_blending = rasterizer(
                    triangles_points=cur_vertices,
                    sigma=cur_sigma,
                    num_points_per_triangle=num_points_per_triangle,
                    cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                    number_of_points=primitive_count,
                    opacities=cur_opacity,
                    means2D=means2D,
                    scaling=scaling,
                    density_factor=density,
                    shs=None,
                    colors_precomp=cur_colors
                )

            if log_render_stats or return_triangle_visibility_mask:
                visible_mask = radii > 0
                batch_visible_masks.append(visible_mask)

            if log_render_stats:
                rendered_triangles_per_view.append(int(visible_mask.sum().item()))

                cur_opacity_flat = cur_opacity.squeeze(-1)
                for threshold in VISIBLE_OPACITY_THRESHOLDS:
                    rendered_visible_opacity_per_view[threshold].append(
                        int((visible_mask & (cur_opacity_flat > threshold)).sum().item())
                    )

                culled_reasons = radii[radii <= 0]
                if culled_reasons.numel() > 0:
                    unique_reasons, counts = torch.unique(
                        culled_reasons,
                        return_counts=True,
                    )
                    for reason, count in zip(unique_reasons.tolist(), counts.tolist()):
                        culled_reason_counts[int(reason)] = (
                            culled_reason_counts.get(int(reason), 0) + int(count)
                        )

            render_alpha = allmap[1:2]

            render_normal = F.normalize(allmap[2:5], dim=0)
            render_normal = torch.nan_to_num(render_normal, 0.0, 0.0, 0.0)

            render_depth_expected = allmap[0:1]
            render_depth_expected = render_depth_expected / render_alpha
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0.0, 0.0, 0.0)

            surf_depth = render_depth_expected
            surf_normal = depth_to_normal(surf_depth.unsqueeze(0), cur_K.unsqueeze(0)).squeeze(0)
            surf_normal = torch.nan_to_num(surf_normal, 0.0, 0.0, 0.0)

            batch_images.append(render_image)
            batch_rend_normals.append(render_normal)
            batch_surf_normals.append(surf_normal)
            batch_depths.append(surf_depth.squeeze(0))
            batch_opacities.append(render_alpha.squeeze(0))
            batch_proj_matrices.append(proj_mat.T)

        if log_render_stats and batch_visible_masks:
            rendered_triangles_unique_per_batch.append(
                int(torch.stack(batch_visible_masks, dim=0).any(dim=0).sum().item())
            )

        all_images_list.append(torch.stack(batch_images))
        all_rend_normals_list.append(torch.stack(batch_rend_normals))
        all_surf_normals_list.append(torch.stack(batch_surf_normals))
        all_depths_list.append(torch.stack(batch_depths))
        all_opacities_list.append(torch.stack(batch_opacities))
        all_proj_matrices_list.append(torch.stack(batch_proj_matrices))
        if return_triangle_visibility_mask:
            all_visibility_masks_list.append(torch.stack(batch_visible_masks))

    if log_render_stats and rendered_triangles_per_view:
        per_view = torch.tensor(rendered_triangles_per_view, dtype=torch.float32)
        logger.info(
            "[decoder debug] step=%s triangle/rendered_triangles_per_view: num_views=%d min=%d max=%d mean=%.3f median=%.3f total=%d ratio_mean=%.6f",
            global_step,
            len(rendered_triangles_per_view),
            int(per_view.min().item()),
            int(per_view.max().item()),
            per_view.mean().item(),
            per_view.median().item(),
            int(per_view.sum().item()),
            per_view.mean().item() / max(num_triangles, 1),
        )
        for threshold, counts in rendered_visible_opacity_per_view.items():
            threshold_per_view = torch.tensor(counts, dtype=torch.float32)
            logger.info(
                "[decoder debug] step=%s triangle/rendered_triangles_per_view_visible_opacity_gt_%s: num_views=%d min=%d max=%d mean=%.3f median=%.3f total=%d ratio_vs_total_mean=%.6f ratio_vs_visible_mean=%.6f",
                global_step,
                f"{threshold:.0e}",
                len(counts),
                int(threshold_per_view.min().item()),
                int(threshold_per_view.max().item()),
                threshold_per_view.mean().item(),
                threshold_per_view.median().item(),
                int(threshold_per_view.sum().item()),
                threshold_per_view.mean().item() / max(num_triangles, 1),
                (threshold_per_view / per_view.clamp_min(1.0)).mean().item(),
            )
    if log_render_stats and rendered_triangles_unique_per_batch:
        per_batch_unique = torch.tensor(rendered_triangles_unique_per_batch, dtype=torch.float32)
        logger.info(
            "[decoder debug] step=%s triangle/rendered_triangles_unique_per_batch: num_batches=%d min=%d max=%d mean=%.3f median=%.3f ratio_mean=%.6f",
            global_step,
            len(rendered_triangles_unique_per_batch),
            int(per_batch_unique.min().item()),
            int(per_batch_unique.max().item()),
            per_batch_unique.mean().item(),
            per_batch_unique.median().item(),
            per_batch_unique.mean().item() / max(num_triangles, 1),
        )
    if log_render_stats and culled_reason_counts:
        total_culled = sum(culled_reason_counts.values())
        reason_summary = " ".join(
            f"{reason}:{count}({count / total_culled:.4f})"
            for reason, count in sorted(culled_reason_counts.items())
        )
        logger.info(
            "[decoder debug] step=%s triangle/culled_radii_reasons: total=%d %s",
            global_step,
            total_culled,
            reason_summary,
        )

    return DecoderOutput(
        color=torch.stack(all_images_list),
        depth=torch.stack(all_depths_list),
        opacity=torch.stack(all_opacities_list),
        rend_normal=torch.stack(all_rend_normals_list),
        surf_normal=torch.stack(all_surf_normals_list),
        projection_matrix=torch.stack(all_proj_matrices_list),
        triangle_visibility_mask=(
            torch.stack(all_visibility_masks_list)
            if return_triangle_visibility_mask
            else None
        ),
    )
