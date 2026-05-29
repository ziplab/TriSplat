from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _normalize(vector: Tensor, eps: float = 1e-8) -> Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _chaikin_open(points: Tensor, refinements: int) -> Tensor:
    if points.shape[0] < 3 or refinements <= 0:
        return points.clone()
    result = points
    for _ in range(refinements):
        left = result[:-1]
        right = result[1:]
        q = 0.75 * left + 0.25 * right
        r = 0.25 * left + 0.75 * right
        pieces = [result[:1]]
        for qi, ri in zip(q, r):
            pieces.extend([qi[None], ri[None]])
        pieces.append(result[-1:])
        result = torch.cat(pieces, dim=0)
    return result


def _arc_length(points: Tensor) -> Tensor:
    if points.shape[0] == 1:
        return torch.zeros(1, dtype=points.dtype, device=points.device)
    lengths = (points[1:] - points[:-1]).norm(dim=-1)
    return torch.cat([torch.zeros(1, dtype=points.dtype, device=points.device), lengths.cumsum(dim=0)])


def _dedupe_by_arc(points: Tensor, arc: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    if points.shape[0] <= 1:
        return points, arc
    keep = torch.cat([torch.ones(1, dtype=torch.bool, device=points.device), (arc[1:] - arc[:-1]) > eps])
    return points[keep], arc[keep]


def _interp_series(values: Tensor, times: Tensor, query: Tensor) -> Tensor:
    if values.shape[0] == 1:
        return values.expand(query.shape[0], *values.shape[1:]).clone()
    hi = torch.searchsorted(times, query, right=False).clamp(1, times.shape[0] - 1)
    lo = hi - 1
    denom = (times[hi] - times[lo]).clamp_min(1e-8)
    weight = ((query - times[lo]) / denom).reshape(-1, *([1] * (values.ndim - 1)))
    return values[lo] + (values[hi] - values[lo]) * weight


def _estimate_focus(context_extrinsics: Tensor) -> Tensor:
    origins = context_extrinsics[:, :3, 3]
    directions = _normalize(context_extrinsics[:, :3, 2])
    eye = torch.eye(3, dtype=context_extrinsics.dtype, device=context_extrinsics.device)
    projections = eye[None] - directions[:, :, None] * directions[:, None, :]
    lhs = projections.sum(dim=0)
    rhs = (projections @ origins[:, :, None]).sum(dim=0).squeeze(-1)
    try:
        focus = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        focus = origins.mean(dim=0)
    if not torch.isfinite(focus).all():
        focus = origins.mean(dim=0)
    return focus


def _build_look_at_rotations(
    origins: Tensor,
    focus: Tensor,
    context_extrinsics: Tensor,
) -> Tensor:
    forward = _normalize(focus[None] - origins)
    avg_up = _normalize(context_extrinsics[:, :3, 1].mean(dim=0))
    right = torch.linalg.cross(avg_up.expand_as(forward), forward, dim=-1)
    weak = right.norm(dim=-1) < 1e-5
    if weak.any():
        fallback = torch.tensor([0.0, 1.0, 0.0], dtype=origins.dtype, device=origins.device)
        right[weak] = torch.linalg.cross(fallback.expand_as(forward[weak]), forward[weak], dim=-1)
    weak = right.norm(dim=-1) < 1e-5
    if weak.any():
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=origins.dtype, device=origins.device)
        right[weak] = torch.linalg.cross(fallback.expand_as(forward[weak]), forward[weak], dim=-1)
    right = _normalize(right)
    up = _normalize(torch.linalg.cross(forward, right, dim=-1))
    return torch.stack([right, up, forward], dim=-1)


def build_smooth_context_trajectory(
    context_extrinsics: Tensor,
    context_intrinsics: Tensor,
    num_frames: int,
    chaikin_refinements: int = 5,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    if context_extrinsics.ndim != 3 or context_extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected context_extrinsics with shape [V,4,4], got {tuple(context_extrinsics.shape)}.")
    if context_intrinsics.ndim != 3 or context_intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected context_intrinsics with shape [V,3,3], got {tuple(context_intrinsics.shape)}.")
    if context_extrinsics.shape[0] != context_intrinsics.shape[0]:
        raise ValueError("Context extrinsics and intrinsics must have the same number of views.")
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2.")

    dtype = context_extrinsics.dtype
    device = context_extrinsics.device
    origins = context_extrinsics[:, :3, 3]
    smoothed = _chaikin_open(origins, chaikin_refinements)
    smoothed_arc = _arc_length(smoothed)
    smoothed, smoothed_arc = _dedupe_by_arc(smoothed, smoothed_arc)
    total = smoothed_arc[-1]
    if total <= 1e-8:
        query = torch.linspace(0, max(context_extrinsics.shape[0] - 1, 1), num_frames, dtype=dtype, device=device)
        source_times = torch.linspace(0, max(context_extrinsics.shape[0] - 1, 1), context_extrinsics.shape[0], dtype=dtype, device=device)
        sampled_origins = _interp_series(origins, source_times, query)
        sampled_intrinsics = _interp_series(context_intrinsics, source_times, query)
        smooth_path_length = torch.tensor(0.0, dtype=dtype, device=device)
    else:
        query = torch.linspace(0, total, num_frames, dtype=dtype, device=device)
        sampled_origins = _interp_series(smoothed, smoothed_arc, query)
        raw_arc_full = _arc_length(origins)
        if origins.shape[0] <= 1:
            keep = torch.ones(origins.shape[0], dtype=torch.bool, device=device)
        else:
            keep = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=device),
                    (raw_arc_full[1:] - raw_arc_full[:-1]) > 1e-8,
                ]
            )
        raw_arc = raw_arc_full[keep]
        raw_intrinsics = context_intrinsics[keep]
        sampled_intrinsics = _interp_series(raw_intrinsics, raw_arc, query.clamp(max=raw_arc[-1]))
        smooth_path_length = total

    focus = _estimate_focus(context_extrinsics)
    rotations = _build_look_at_rotations(sampled_origins, focus, context_extrinsics)
    target_extrinsics = torch.eye(4, dtype=dtype, device=device).expand(num_frames, 4, 4).clone()
    target_extrinsics[:, :3, :3] = rotations
    target_extrinsics[:, :3, 3] = sampled_origins

    raw_arc_full = _arc_length(origins)
    metadata: dict[str, Any] = {
        "type": "smooth_context",
        "target": [float(x) for x in focus.detach().cpu().tolist()],
        "chaikin_refinements": int(chaikin_refinements),
        "control_points": int(origins.shape[0]),
        "smoothed_control_points": int(smoothed.shape[0]),
        "raw_path_length": float(raw_arc_full[-1].detach().cpu().item()),
        "smooth_path_length": float(smooth_path_length.detach().cpu().item()),
        "context_origin_min": [float(x) for x in origins.amin(dim=0).detach().cpu().tolist()],
        "context_origin_max": [float(x) for x in origins.amax(dim=0).detach().cpu().tolist()],
        "trajectory_origin_min": [float(x) for x in sampled_origins.amin(dim=0).detach().cpu().tolist()],
        "trajectory_origin_max": [float(x) for x in sampled_origins.amax(dim=0).detach().cpu().tolist()],
        "context_origins": [[float(v) for v in row] for row in origins.detach().cpu().tolist()],
    }
    return target_extrinsics, sampled_intrinsics, metadata


def apply_smooth_context_target(
    batch: dict[str, Any],
    num_frames: int,
    chaikin_refinements: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context_extrinsics = batch["context"]["extrinsics"][0]
    context_intrinsics = batch["context"]["intrinsics"][0]
    target_extrinsics, target_intrinsics, metadata = build_smooth_context_trajectory(
        context_extrinsics,
        context_intrinsics,
        num_frames=num_frames,
        chaikin_refinements=chaikin_refinements,
    )
    batch["target"]["extrinsics"] = target_extrinsics[None]
    batch["target"]["intrinsics"] = target_intrinsics[None]
    batch["target"]["near"] = batch["target"]["near"][:, :1].expand(-1, num_frames).clone()
    batch["target"]["far"] = batch["target"]["far"][:, :1].expand(-1, num_frames).clone()
    batch["target"]["image"] = batch["target"]["image"][:, :1].expand(-1, num_frames, -1, -1, -1).clone()
    batch["target"]["index"] = torch.arange(num_frames, dtype=torch.long, device=context_extrinsics.device)[None]
    return batch, metadata
