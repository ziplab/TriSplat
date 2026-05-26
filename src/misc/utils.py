import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from src.visualization.color_map import apply_color_map_to_image

logger = logging.getLogger(__name__)


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_depth_map(result):
    far = result.view(-1)[:16_000_000].quantile(0.99).log()
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
    except:
        logger.warning("No valid depth values found.")
        near = torch.zeros_like(far)
    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def confidence_map(result):
    result = result / result.view(-1).max()
    return apply_color_map_to_image(result, "magma")


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag


def subsample_point_cloud_views(point_cloud, max_total_points=300000):
    """
    Subsamples each view of a structured multi-view point cloud evenly
    using grid sampling to meet a total point limit.

    Args:
        point_cloud (np.ndarray): Input point cloud with shape [v, h, w, d].
                                  v: number of views
                                  h: height
                                  w: width
                                  d: point features dimension
        max_total_points (int): The maximum total number of points allowed
                                across all views after subsampling.

    Returns:
        np.ndarray: The subsampled point cloud. Shape will be [v, h', w', d] where
                    v * h' * w' <= max_total_points. Returns the original
                    array if no subsampling is needed or possible.
    """
    if not isinstance(point_cloud, np.ndarray):
        raise TypeError("Input point_cloud must be a NumPy array.")

    if point_cloud.ndim != 4:
        raise ValueError(f"Input point_cloud must have 4 dimensions [v, h, w, d], but got {point_cloud.ndim}")

    v, h, w, d = point_cloud.shape
    initial_total_points = v * h * w

    if initial_total_points <= max_total_points:
        return point_cloud

    # Calculate the minimum step size required to get under the max_total_points limit
    # We need v * (h / step) * (w / step) <= max_total_points
    # step^2 >= (v * h * w) / max_total_points
    # step >= sqrt((v * h * w) / max_total_points)
    # Use ceiling to ensure the total count is definitely <= max_total_points
    # Step must be at least 1
    step = max(1, int(math.ceil(math.sqrt(initial_total_points / max_total_points))))

    # Perform grid subsampling using slicing
    # Slicing applies to dimensions 1 (h) and 2 (w)
    subsampled_cloud = point_cloud[:, ::step, ::step, :]

    return subsampled_cloud


def align_pose_space(ctx_poses_pred, ctx_poses, target_poses):
    # Compute all transformation matrices at once
    ctx_poses_inv = torch.inverse(ctx_poses)  # b, n_ctx, 4, 4
    transformation_matrices = torch.bmm(
        ctx_poses_pred.view(-1, 4, 4),
        ctx_poses_inv.view(-1, 4, 4)
    ).view(ctx_poses.shape)  # b, n_ctx, 4, 4

    # Transform target poses for each context view
    b, n_ctx, _, _ = transformation_matrices.shape
    b, n_target, _, _ = target_poses.shape

    target_poses_per_ctx = torch.bmm(
        transformation_matrices.view(b * n_ctx, 4, 4).unsqueeze(1).expand(-1, n_target, -1, -1).reshape(-1, 4, 4),
        target_poses.unsqueeze(1).expand(-1, n_ctx, -1, -1, -1).reshape(-1, 4, 4)
    ).reshape(b, n_ctx, n_target, 4, 4)

    return target_poses_per_ctx

def clone_batch(batch):
    if torch.is_tensor(batch):
        return batch.clone()
    elif isinstance(batch, dict):
        return {k: clone_batch(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [clone_batch(v) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(clone_batch(v) for v in batch)
    else:
        return batch


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()