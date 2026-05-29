import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


GT_FILTER_QUANTILE_LO = 0.01
GT_FILTER_QUANTILE_HI = 0.99
GT_FILTER_BBOX_MARGIN_RATIO = 0.02
GT_FILTER_DENSITY_RADIUS_RATIO = 0.01
GT_FILTER_DENSITY_MIN_NEIGHBORS = 3
PRED_FILTER_KEEP_RADIUS_RATIO = 0.015
PRED_FILTER_KEEP_RADIUS_MIN_FACTOR = 2.0


def _load_geometry_as_point_cloud(file_path: str | Path) -> o3d.geometry.PointCloud:
    path = Path(file_path)

    point_cloud = o3d.io.read_point_cloud(str(path))
    if len(point_cloud.points) > 0:
        return point_cloud

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load any geometry points from {path}")

    mesh_point_cloud = o3d.geometry.PointCloud()
    mesh_point_cloud.points = mesh.vertices
    return mesh_point_cloud


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    return point_cloud


def _load_points(
    file_path: str | Path,
    transform: np.ndarray | None = None,
    down_sample: float = 0.02,
) -> np.ndarray:
    point_cloud = _load_geometry_as_point_cloud(file_path)
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if transform is not None:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points = points @ rotation.T + translation[None]

    if down_sample:
        point_cloud = _make_point_cloud(points)
        point_cloud = point_cloud.voxel_down_sample(down_sample)
        points = np.asarray(point_cloud.points, dtype=np.float64)

    if len(points) == 0:
        raise ValueError(f"Point set is empty after loading/downsampling: {file_path}")
    return points


def _resolve_mesh_space_path(
    pred_mesh_path: str | Path,
    mesh_space_path: str | Path | None = None,
) -> Path | None:
    if mesh_space_path is not None:
        resolved_path = Path(mesh_space_path).expanduser().resolve()
        return resolved_path if resolved_path.exists() else None

    candidate = Path(pred_mesh_path).expanduser().resolve().parent / "mesh_space.json"
    return candidate if candidate.exists() else None


def _load_mesh_space_metadata(
    pred_mesh_path: str | Path,
    mesh_space_path: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    resolved_path = _resolve_mesh_space_path(pred_mesh_path, mesh_space_path)
    if resolved_path is None:
        return None, {}

    with resolved_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"mesh_space metadata must be a JSON object: {resolved_path}")
    return resolved_path, payload


def _extract_world_from_normalized_transform(
    mesh_space_metadata: dict[str, Any],
) -> np.ndarray | None:
    if mesh_space_metadata.get("mesh_space") != "normalized":
        return None

    matrix = mesh_space_metadata.get("world_from_normalized")
    if matrix is None:
        return None
    matrix_np = np.asarray(matrix, dtype=np.float64)
    if matrix_np.shape != (4, 4):
        raise ValueError(
            f"world_from_normalized must be a 4x4 matrix, got shape {matrix_np.shape}"
        )
    return matrix_np


def _normalize_for_alignment(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    extent = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-8:
        diag = 1.0
    return (points - center) / diag


def _umeyama_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = (target_centered.T @ source_centered) / len(source_points)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    variance = float((source_centered ** 2).sum() / len(source_points))
    if variance <= 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _estimate_similarity_alignment(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> dict[str, Any]:
    if len(pred_points) < 3 or len(gt_points) < 3:
        identity = np.eye(4, dtype=np.float64)
        return {
            "scale": 1.0,
            "rotation": identity[:3, :3].tolist(),
            "translation": identity[:3, 3].tolist(),
            "matrix": identity.tolist(),
            "num_correspondences": int(min(len(pred_points), len(gt_points))),
        }

    pred_norm = _normalize_for_alignment(pred_points)
    gt_norm = _normalize_for_alignment(gt_points)

    gt_norm_cloud = _make_point_cloud(gt_norm)
    kd_tree = o3d.geometry.KDTreeFlann(gt_norm_cloud)
    matched_gt_points = []
    for point in pred_norm:
        _, indices, _ = kd_tree.search_knn_vector_3d(point, 1)
        matched_gt_points.append(gt_points[indices[0]])
    matched_gt_points_np = np.asarray(matched_gt_points, dtype=np.float64)

    scale, rotation, translation = _umeyama_similarity(
        pred_points,
        matched_gt_points_np,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "matrix": matrix.tolist(),
        "num_correspondences": int(len(matched_gt_points_np)),
    }


def _apply_similarity_transform(
    points: np.ndarray,
    alignment: dict[str, Any],
) -> np.ndarray:
    matrix = np.asarray(alignment["matrix"], dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return points @ rotation.T + translation[None]


def _compute_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _compute_bbox_diag(points: np.ndarray) -> float:
    bbox_min, bbox_max = _compute_bbox(points)
    return float(np.linalg.norm(bbox_max - bbox_min))


def _filter_points_inside_bbox(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    mask = np.all((points >= bbox_min[None]) & (points <= bbox_max[None]), axis=1)
    return points[mask]


def _build_radius_neighbor_mask(
    points: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)

    point_cloud = _make_point_cloud(points)
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    keep_mask = np.zeros((len(points),), dtype=bool)
    for index, point in enumerate(points):
        _, indices, _ = kd_tree.search_radius_vector_3d(point, radius)
        keep_mask[index] = len(indices) >= min_neighbors
    return keep_mask


def _largest_voxel_component_mask(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return np.zeros((len(points),), dtype=bool)

    coords = np.floor((points - points.min(axis=0, keepdims=True)) / voxel_size).astype(np.int64)
    voxel_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for point_index, coord in enumerate(coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        voxel_to_indices.setdefault(key, []).append(point_index)

    if not voxel_to_indices:
        return np.zeros((len(points),), dtype=bool)

    visited: set[tuple[int, int, int]] = set()
    best_component_keys: set[tuple[int, int, int]] = set()
    best_component_count = -1
    neighbor_offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    for start_key in voxel_to_indices:
        if start_key in visited:
            continue
        queue = deque([start_key])
        visited.add(start_key)
        component_keys: set[tuple[int, int, int]] = set()
        component_count = 0

        while queue:
            key = queue.popleft()
            component_keys.add(key)
            component_count += len(voxel_to_indices[key])
            for dx, dy, dz in neighbor_offsets:
                neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                if neighbor not in voxel_to_indices or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        if component_count > best_component_count:
            best_component_count = component_count
            best_component_keys = component_keys

    keep_mask = np.zeros((len(points),), dtype=bool)
    for key in best_component_keys:
        keep_mask[voxel_to_indices[key]] = True
    return keep_mask


def _filter_gt_core(
    gt_points: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    gt_points_before = int(len(gt_points))
    if gt_points_before == 0:
        raise ValueError("GT filtering requires non-empty GT points.")

    gt_diag_before = _compute_bbox_diag(gt_points)
    bbox_margin = GT_FILTER_BBOX_MARGIN_RATIO * gt_diag_before
    bbox_lo = np.quantile(gt_points, GT_FILTER_QUANTILE_LO, axis=0)
    bbox_hi = np.quantile(gt_points, GT_FILTER_QUANTILE_HI, axis=0)
    bbox_lo = bbox_lo - bbox_margin
    bbox_hi = bbox_hi + bbox_margin

    gt_after_bbox = _filter_points_inside_bbox(gt_points, bbox_lo, bbox_hi)
    gt_filter_fallback_reason = ""
    if len(gt_after_bbox) == 0:
        gt_after_bbox = gt_points
        gt_filter_fallback_reason = "bbox_empty"

    gt_diag_bbox = _compute_bbox_diag(gt_after_bbox)
    gt_density_radius = max(float(threshold), GT_FILTER_DENSITY_RADIUS_RATIO * gt_diag_bbox)
    density_mask = _build_radius_neighbor_mask(
        gt_after_bbox,
        radius=gt_density_radius,
        min_neighbors=GT_FILTER_DENSITY_MIN_NEIGHBORS,
    )
    gt_after_density = gt_after_bbox[density_mask]
    if len(gt_after_density) == 0:
        gt_after_density = gt_after_bbox
        gt_filter_fallback_reason = (
            gt_filter_fallback_reason + "|density_empty"
            if gt_filter_fallback_reason
            else "density_empty"
        )

    component_mask = _largest_voxel_component_mask(
        gt_after_density,
        voxel_size=gt_density_radius,
    )
    gt_core = gt_after_density[component_mask]
    if len(gt_core) == 0:
        gt_core = gt_after_density
        gt_filter_fallback_reason = (
            gt_filter_fallback_reason + "|component_empty"
            if gt_filter_fallback_reason
            else "component_empty"
        )

    gt_diag_after = _compute_bbox_diag(gt_core)
    metadata = {
        "gt_filter_quantile_lo": GT_FILTER_QUANTILE_LO,
        "gt_filter_quantile_hi": GT_FILTER_QUANTILE_HI,
        "bbox_margin_ratio": GT_FILTER_BBOX_MARGIN_RATIO,
        "bbox_margin": float(bbox_margin),
        "gt_density_radius_ratio": GT_FILTER_DENSITY_RADIUS_RATIO,
        "gt_density_radius": float(gt_density_radius),
        "gt_density_min_neighbors": GT_FILTER_DENSITY_MIN_NEIGHBORS,
        "gt_diag_before": float(gt_diag_before),
        "gt_diag_after": float(gt_diag_after),
        "gt_points_before": gt_points_before,
        "gt_points_after_bbox": int(len(gt_after_bbox)),
        "gt_points_after_density": int(len(gt_after_density)),
        "gt_points_after_component": int(len(gt_core)),
        "gt_points_after": int(len(gt_core)),
        "gt_filter_fallback_reason": gt_filter_fallback_reason,
    }
    return gt_core, metadata


def _filter_pred_by_gt_support(
    pred_points: np.ndarray,
    gt_core: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_points_before = int(len(pred_points))
    if pred_points_before == 0:
        raise ValueError("Prediction filtering requires non-empty predicted points.")
    if len(gt_core) == 0:
        raise ValueError("Prediction filtering requires non-empty filtered GT points.")

    gt_diag = _compute_bbox_diag(gt_core)
    bbox_margin = GT_FILTER_BBOX_MARGIN_RATIO * gt_diag
    bbox_min, bbox_max = _compute_bbox(gt_core)
    pred_after_bbox = _filter_points_inside_bbox(
        pred_points,
        bbox_min - bbox_margin,
        bbox_max + bbox_margin,
    )
    pred_filter_fallback_reason = ""
    if len(pred_after_bbox) == 0:
        pred_after_bbox = pred_points
        pred_filter_fallback_reason = "bbox_empty"

    pred_keep_radius_min = PRED_FILTER_KEEP_RADIUS_MIN_FACTOR * float(threshold)
    pred_keep_radius = max(
        pred_keep_radius_min,
        PRED_FILTER_KEEP_RADIUS_RATIO * gt_diag,
    )
    _, pred_to_gt_after_bbox = nn_correspondance(gt_core, pred_after_bbox)
    pred_to_gt_after_bbox = np.asarray(pred_to_gt_after_bbox, dtype=np.float64)
    pred_keep_mask = pred_to_gt_after_bbox < pred_keep_radius
    pred_gt_support = pred_after_bbox[pred_keep_mask]
    if len(pred_gt_support) == 0:
        pred_gt_support = pred_after_bbox
        pred_filter_fallback_reason = (
            pred_filter_fallback_reason + "|neighbor_empty"
            if pred_filter_fallback_reason
            else "neighbor_empty"
        )

    metadata = {
        "pred_keep_radius_ratio": PRED_FILTER_KEEP_RADIUS_RATIO,
        "pred_keep_radius_min_factor": PRED_FILTER_KEEP_RADIUS_MIN_FACTOR,
        "pred_keep_radius_min": float(pred_keep_radius_min),
        "pred_keep_radius": float(pred_keep_radius),
        "pred_points_before": pred_points_before,
        "pred_points_after_bbox": int(len(pred_after_bbox)),
        "pred_points_after": int(len(pred_gt_support)),
        "pred_filter_fallback_reason": pred_filter_fallback_reason,
    }
    return pred_gt_support, metadata


def _compute_gt_trimmed_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    gt_core, gt_metadata = _filter_gt_core(gt_points, threshold=threshold)
    pred_gt_support, pred_metadata = _filter_pred_by_gt_support(
        pred_points,
        gt_core,
        threshold=threshold,
    )
    return {
        **gt_metadata,
        **pred_metadata,
        **compute_metrics_from_point_sets(
            pred_gt_support,
            gt_core,
            threshold=threshold,
        ),
    }


def compute_metrics_from_point_sets(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        raise ValueError(
            "Mesh evaluation requires non-empty predicted and target point sets."
        )

    _, gt_to_pred = nn_correspondance(pred_points, gt_points)
    _, pred_to_gt = nn_correspondance(gt_points, pred_points)
    gt_to_pred = np.asarray(gt_to_pred, dtype=np.float64)
    pred_to_gt = np.asarray(pred_to_gt, dtype=np.float64)

    precision = np.mean((pred_to_gt < threshold).astype(np.float64))
    recall = np.mean((gt_to_pred < threshold).astype(np.float64))
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    pred_to_gt_mean = float(np.mean(pred_to_gt))
    gt_to_pred_mean = float(np.mean(gt_to_pred))
    return {
        "dist1": pred_to_gt_mean,
        "dist2": gt_to_pred_mean,
        "pred_to_gt": pred_to_gt_mean,
        "gt_to_pred": gt_to_pred_mean,
        "cd": pred_to_gt_mean + gt_to_pred_mean,
        "prec": float(precision),
        "recal": float(recall),
        "fscore": float(fscore),
    }


def eval_mesh_core(
    file_pred: str | Path,
    file_trgt: str | Path,
    threshold: float = 0.05,
    down_sample: float = 0.02,
    pred_transform: np.ndarray | None = None,
) -> dict[str, float]:
    pred_points = _load_points(
        file_pred,
        transform=pred_transform,
        down_sample=down_sample,
    )
    gt_points = _load_points(file_trgt, down_sample=down_sample)
    return compute_metrics_from_point_sets(
        pred_points,
        gt_points,
        threshold=threshold,
    )


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def eval_mesh(
    pred_mesh_path: str | Path,
    gt_mesh_path: str | Path,
    scene: str,
    mesh_space_path: str | Path | None = None,
    gt_transform: np.ndarray | None = None,
    gt_transform_metadata: dict[str, Any] | None = None,
    threshold: float = 0.05,
    down_sample: float = 0.02,
) -> dict[str, Any]:
    gt_pointcloud_path = Path(gt_mesh_path) / f"{scene}.ply"
    resolved_mesh_space_path, mesh_space_metadata = _load_mesh_space_metadata(
        pred_mesh_path,
        mesh_space_path=mesh_space_path,
    )
    world_from_normalized = _extract_world_from_normalized_transform(mesh_space_metadata)
    pred_points_raw_world = _load_points(
        pred_mesh_path,
        transform=world_from_normalized,
        down_sample=down_sample,
    )
    gt_points = _load_points(
        gt_pointcloud_path,
        transform=gt_transform,
        down_sample=down_sample,
    )
    gt_transform_metadata = gt_transform_metadata or {}

    base_metadata = {
        "mesh_space_path": None
        if resolved_mesh_space_path is None
        else str(resolved_mesh_space_path),
        "mesh_space": mesh_space_metadata.get("mesh_space", "unknown"),
        "world_space": mesh_space_metadata.get("world_space", "raw_world"),
        "pose_norm_scale": mesh_space_metadata.get("pose_norm_scale"),
        "pose_norm_method": mesh_space_metadata.get("pose_norm_method"),
        "relative_pose": mesh_space_metadata.get("relative_pose"),
        "context_indices": mesh_space_metadata.get("context_indices"),
        "target_indices": mesh_space_metadata.get("target_indices"),
        "index_path": mesh_space_metadata.get("index_path"),
        "num_context_views": mesh_space_metadata.get("num_context_views"),
        "use_train_view": mesh_space_metadata.get("use_train_view"),
        "use_val_view": mesh_space_metadata.get("use_val_view"),
        "pred_transform_applied": world_from_normalized is not None,
        "gt_transform_applied": gt_transform is not None,
        **gt_transform_metadata,
    }

    raw_world = {
        **base_metadata,
        "evaluation_space": "raw_world",
        **compute_metrics_from_point_sets(
            pred_points_raw_world,
            gt_points,
            threshold=threshold,
        ),
    }

    alignment = _estimate_similarity_alignment(pred_points_raw_world, gt_points)
    pred_points_aligned = _apply_similarity_transform(pred_points_raw_world, alignment)
    aligned_similarity = {
        **base_metadata,
        "evaluation_space": "aligned_similarity",
        "alignment_scale": alignment["scale"],
        "alignment_rotation": alignment["rotation"],
        "alignment_translation": alignment["translation"],
        "alignment_matrix": alignment["matrix"],
        "alignment_num_correspondences": alignment["num_correspondences"],
        **compute_metrics_from_point_sets(
            pred_points_aligned,
            gt_points,
            threshold=threshold,
        ),
    }

    gt_trimmed = {
        **base_metadata,
        "evaluation_space": "gt_trimmed",
        **_compute_gt_trimmed_metrics(
            pred_points_raw_world,
            gt_points,
            threshold=threshold,
        ),
    }

    return {
        "mesh_space_path": base_metadata["mesh_space_path"],
        "mesh_space": base_metadata["mesh_space"],
        "raw_world": raw_world,
        "aligned_similarity": aligned_similarity,
        "gt_trimmed": gt_trimmed,
    }
