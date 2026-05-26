"""
Render exported meshes from target-view-centered overview viewpoints.

This script preserves the tracked `render_mesh_open3d.py` baseline and builds
an overview renderer on top of its data-loading and raycasting helpers.

The framing policy is intentionally conservative:

- estimate the subject from the mesh region visible in the target view
- trim far-depth outliers so sparse stray triangles do not dominate framing
- pull the camera back only slightly from the target view
- render `left`, `center`, and `right` viewpoints by default
- write both `textured` and `untextured` variants
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image

from render_mesh_open3d import (
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_ORIG_IMAGE_SHAPE,
    DEFAULT_SHADING_AMBIENT,
    DEFAULT_SHADING_DIFFUSE,
    POSE_NORM_METHODS,
    adjust_intrinsics_for_crop,
    apply_project_pose_normalization,
    build_ray_scene,
    convert_poses,
    discover_scene_dirs,
    infer_mesh_path,
    load_eval_indices,
    load_hydra_runtime_defaults,
    load_scene_example,
    norm_to_pixel_intrinsics,
    normalize_vectors_np,
    pick_eval_entry,
    resolve_eval_index_paths,
    resolve_scene_dir,
    resolve_shape,
    resolve_stage_root,
    select_context_target_indices,
    select_render_indices,
    write_json,
)


DEFAULT_DL3DV_DATA_ROOT = Path(os.environ.get("DL3DV_ROOT", "data/dl3dv"))
DEFAULT_DL3DV_ORIG_IMAGE_SHAPE = (270, 480)
DEFAULT_OVERVIEW_FX_SCALE = 1.0
DEFAULT_OVERVIEW_RETREAT_RATIO = 0.12
DEFAULT_OVERVIEW_SIDE_MARGIN_RATIO = 0.18
DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_UNTEXTURED_COLOR = np.array([0.75, 0.75, 0.75], dtype=np.float64)
VISIBLE_DEPTH_PERCENTILE = 70.0
VISIBLE_DEPTH_MULTIPLIER = 4.0
VISIBLE_DEPTH_FALLBACK_PERCENTILE = 85.0
VISIBLE_XY_PERCENTILE = 2.0
DL3DV_EVAL_INDEX_CANDIDATES = (
    "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v_first20.json",
    "assets/dl3dv_start_0_distance_100_ctx_12v_tgt_8v_first20.json",
    "assets/dl3dv_start_0_distance_150_ctx_24v_tgt_8v_first20.json",
    "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json",
    "assets/dl3dv_start_0_distance_100_ctx_12v_tgt_8v.json",
    "assets/dl3dv_start_0_distance_150_ctx_24v_tgt_8v.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render exported meshes from target-view-centered overview viewpoints."
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "re10k", "dl3dv"),
        default="auto",
        help=(
            "Dataset defaults to use. 'auto' reads the nearest Hydra config first, "
            "then infers from data_root/test_output paths."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Dataset root. When omitted, the script first tries the nearest .hydra/config.yaml, "
            f"then falls back to {DEFAULT_DATA_ROOT} for re10k or {DEFAULT_DL3DV_DATA_ROOT} for DL3DV."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split used to resolve index.json and chunks.",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="outputs/test",
        help="Directory containing per-scene test outputs. Can also point to a single scene directory.",
    )
    parser.add_argument(
        "--eval_index",
        type=str,
        default="auto",
        help=(
            "Evaluation index JSON. 'auto' tries the nearest hydra config first, then the "
            "repo assets fallbacks."
        ),
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        default="auto",
        help="Mesh file name under <scene>/mesh/. Use 'auto' to pick the first known mesh export.",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=None,
        help="Optional absolute or relative path to a specific mesh file. Requires --scene.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        action="append",
        default=None,
        help="Scene key to render. Can be passed multiple times. Defaults to all scenes found under test_output.",
    )
    parser.add_argument(
        "--frames",
        choices=("eval", "target", "saved_pred", "all"),
        default="target",
        help="Which frames to anchor the overview cameras on.",
    )
    parser.add_argument(
        "--context_indices",
        type=int,
        nargs="*",
        default=None,
        help="Manual context indices. Used when eval index is unavailable or you want to override it.",
    )
    parser.add_argument(
        "--target_indices",
        type=int,
        nargs="*",
        default=None,
        help="Manual target indices override.",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Target input image shape after the dataset crop shim. When omitted, load from .hydra if available.",
    )
    parser.add_argument(
        "--orig_image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Original packed image shape before crop/rescale. When omitted, load from .hydra if available.",
    )
    parser.add_argument(
        "--pose_norm_method",
        type=str,
        default=None,
        choices=POSE_NORM_METHODS,
        help="Pose normalization method used by the current project. When omitted, load from .hydra if available.",
    )
    parser.add_argument(
        "--relative_pose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to normalize poses relative to the first context camera. When omitted, load from .hydra if available.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional cap on the number of scenes to render.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on the number of anchor frames rendered per scene.",
    )
    parser.add_argument(
        "--overview_layout",
        choices=("sweep3", "single"),
        default="sweep3",
        help="Overview camera layout to render.",
    )
    parser.add_argument(
        "--appearance",
        choices=("textured", "untextured", "both"),
        default="both",
        help="Whether to render the vertex-colored mesh, the gray untextured mesh, or both.",
    )
    parser.add_argument(
        "--shading",
        choices=("none", "face", "vertex"),
        default="face",
        help="Preview shading mode applied to overview renders.",
    )
    parser.add_argument(
        "--shading_ambient",
        type=float,
        default=DEFAULT_SHADING_AMBIENT,
        help="Ambient term used when --shading is not 'none'.",
    )
    parser.add_argument(
        "--shading_diffuse",
        type=float,
        default=DEFAULT_SHADING_DIFFUSE,
        help="Diffuse term used when --shading is not 'none'.",
    )
    parser.add_argument(
        "--overview_fx_scale",
        type=float,
        default=DEFAULT_OVERVIEW_FX_SCALE,
        help="Horizontal focal scaling applied to the anchor intrinsics. 1.0 keeps the target-view focal length.",
    )
    parser.add_argument(
        "--overview_retreat_ratio",
        type=float,
        default=DEFAULT_OVERVIEW_RETREAT_RATIO,
        help="Slight pullback amount relative to the visible subject depth.",
    )
    parser.add_argument(
        "--overview_side_margin_ratio",
        type=float,
        default=DEFAULT_OVERVIEW_SIDE_MARGIN_RATIO,
        help="Extra left/right margin relative to the visible subject width.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default=DEFAULT_MANIFEST_NAME,
        help="Overview manifest file name written under each scene overview directory.",
    )
    return parser.parse_args()


def path_looks_like_dataset(path: Path | None, dataset_key: str) -> bool:
    if path is None:
        return False
    return dataset_key.lower() in str(path).lower()


def resolve_dataset_key(
    dataset_arg: str,
    hydra_defaults: dict[str, Any],
    data_root_arg: str | None,
    test_output: Path,
) -> str:
    if dataset_arg != "auto":
        return dataset_arg

    hydra_dataset_key = hydra_defaults.get("dataset_key")
    if hydra_dataset_key in {"re10k", "dl3dv"}:
        return str(hydra_dataset_key)

    data_root = Path(data_root_arg).expanduser() if data_root_arg is not None else hydra_defaults.get("data_root")
    if path_looks_like_dataset(data_root, "dl3dv") or path_looks_like_dataset(test_output, "dl3dv"):
        return "dl3dv"

    return "re10k"


def resolve_dl3dv_eval_index_paths(
    eval_index_arg: str,
    hydra_eval_index_path: Path | None = None,
) -> list[Path]:
    if eval_index_arg != "auto":
        path = Path(eval_index_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Evaluation index not found: {path}")
        return [path.resolve()]

    candidates: list[Path] = []
    if hydra_eval_index_path is not None:
        candidates.append(hydra_eval_index_path)
    candidates.extend(Path(path) for path in DL3DV_EVAL_INDEX_CANDIDATES)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.exists():
            continue
        resolved_path = candidate.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        resolved.append(resolved_path)
    return resolved


def resolve_dataset_eval_index_paths(
    dataset_key: str,
    eval_index_arg: str,
    hydra_eval_index_path: Path | None,
) -> list[Path]:
    if dataset_key == "dl3dv":
        return resolve_dl3dv_eval_index_paths(eval_index_arg, hydra_eval_index_path)
    return resolve_eval_index_paths(eval_index_arg, hydra_eval_index_path)


def resolve_runtime_settings(
    args: argparse.Namespace,
    test_output: Path,
) -> dict[str, Any]:
    hydra_defaults = load_hydra_runtime_defaults(test_output)

    dataset_key = resolve_dataset_key(args.dataset, hydra_defaults, args.data_root, test_output)
    default_data_root = DEFAULT_DL3DV_DATA_ROOT if dataset_key == "dl3dv" else DEFAULT_DATA_ROOT
    default_orig_image_shape = DEFAULT_DL3DV_ORIG_IMAGE_SHAPE if dataset_key == "dl3dv" else DEFAULT_ORIG_IMAGE_SHAPE

    data_root = Path(args.data_root).expanduser() if args.data_root is not None else hydra_defaults.get("data_root", default_data_root)
    image_shape = resolve_shape(args.image_shape, hydra_defaults.get("image_shape"), DEFAULT_IMAGE_SHAPE)
    orig_image_shape = resolve_shape(args.orig_image_shape, hydra_defaults.get("orig_image_shape"), default_orig_image_shape)
    pose_norm_method = args.pose_norm_method or hydra_defaults.get("pose_norm_method") or "max_pairwise_d"
    relative_pose = args.relative_pose if args.relative_pose is not None else hydra_defaults.get("relative_pose")
    if relative_pose is None:
        relative_pose = True

    eval_index_paths = resolve_dataset_eval_index_paths(
        dataset_key,
        args.eval_index,
        hydra_defaults.get("eval_index_path"),
    )

    return {
        "dataset_key": dataset_key,
        "hydra_defaults": hydra_defaults,
        "data_root": data_root,
        "image_shape": image_shape,
        "orig_image_shape": orig_image_shape,
        "pose_norm_method": pose_norm_method,
        "relative_pose": bool(relative_pose),
        "eval_index_paths": eval_index_paths,
    }


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def safe_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return vector / norm


def project_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    return vector - np.dot(vector, plane_normal) * plane_normal


def build_look_at_c2w(
    position: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
) -> np.ndarray:
    forward = normalize_vector(target - position)
    up = safe_normalize(project_onto_plane(up_hint, forward))
    if up is None:
        for candidate in (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ):
            up = safe_normalize(project_onto_plane(candidate, forward))
            if up is not None:
                break
    if up is None:
        raise ValueError("Failed to construct a valid up vector for look-at camera.")

    right = safe_normalize(np.cross(up, forward))
    if right is None:
        raise ValueError("Failed to construct a valid right vector for look-at camera.")
    up = normalize_vector(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def cast_rays_pinhole(
    ray_scene: o3d.t.geometry.RaycastingScene,
    k_pixel: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    intrinsic_t = o3d.core.Tensor(k_pixel.astype(np.float64))
    extrinsic_t = o3d.core.Tensor(w2c.astype(np.float64))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_t,
        extrinsic_t,
        width,
        height,
    )
    result = ray_scene.cast_rays(rays)
    return rays.numpy(), {key: value.numpy() for key, value in result.items()}


def transform_points(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (w2c @ points_h.T).T[:, :3]


def select_visible_subject_points(
    points_cam: np.ndarray,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if points_cam.shape[0] == 0:
        raise ValueError("No visible subject points were found.")

    z_values = points_cam[:, 2]
    median_depth = float(np.median(z_values))
    depth_cut = max(
        float(np.percentile(z_values, VISIBLE_DEPTH_PERCENTILE)),
        median_depth * VISIBLE_DEPTH_MULTIPLIER,
    )
    depth_mask = z_values <= depth_cut

    min_keep = max(256, int(0.20 * points_cam.shape[0]))
    if int(depth_mask.sum()) < min_keep:
        depth_cut = float(np.percentile(z_values, VISIBLE_DEPTH_FALLBACK_PERCENTILE))
        depth_mask = z_values <= depth_cut

    selected_cam = points_cam[depth_mask]
    selected_world = points_world[depth_mask]
    if selected_cam.shape[0] == 0:
        selected_cam = points_cam
        selected_world = points_world

    if selected_cam.shape[0] >= 512:
        x_low, x_high = np.percentile(selected_cam[:, 0], [VISIBLE_XY_PERCENTILE, 100.0 - VISIBLE_XY_PERCENTILE])
        y_low, y_high = np.percentile(selected_cam[:, 1], [VISIBLE_XY_PERCENTILE, 100.0 - VISIBLE_XY_PERCENTILE])
        xy_mask = (
            (selected_cam[:, 0] >= x_low)
            & (selected_cam[:, 0] <= x_high)
            & (selected_cam[:, 1] >= y_low)
            & (selected_cam[:, 1] <= y_high)
        )
        if int(xy_mask.sum()) >= max(256, int(0.50 * selected_cam.shape[0])):
            selected_cam = selected_cam[xy_mask]
            selected_world = selected_world[xy_mask]

    return selected_cam, selected_world, {
        "median_depth": median_depth,
        "depth_cut": float(depth_cut),
        "selected_fraction": float(selected_cam.shape[0] / points_cam.shape[0]),
    }


def estimate_visible_subject(
    ray_scene: o3d.t.geometry.RaycastingScene,
    k_pixel: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    w2c = np.linalg.inv(c2w)
    rays_np, result = cast_rays_pinhole(ray_scene, k_pixel, w2c, width, height)
    prim_ids = result["primitive_ids"]
    t_hit = result["t_hit"]
    hit_mask = prim_ids != ray_scene.INVALID_ID
    if not np.any(hit_mask):
        raise ValueError("Target view does not intersect the mesh.")

    ray_origins = rays_np[..., :3]
    ray_dirs = rays_np[..., 3:6]
    points_world = ray_origins[hit_mask] + ray_dirs[hit_mask] * t_hit[hit_mask, None]
    points_cam = transform_points(points_world, w2c)
    valid_mask = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 0.0)
    if not np.any(valid_mask):
        raise ValueError("Target view produced no valid positive-depth mesh hits.")

    visible_cam = points_cam[valid_mask]
    visible_world = points_world[valid_mask]
    subject_cam, subject_world, trim_stats = select_visible_subject_points(visible_cam, visible_world)

    low = np.percentile(subject_cam, 2.0, axis=0)
    high = np.percentile(subject_cam, 98.0, axis=0)
    center_cam = 0.5 * (low + high)
    extent_cam = np.maximum(high - low, 1e-6)
    center_world = (c2w[:3, :3] @ center_cam) + c2w[:3, 3]

    return {
        "center_cam": center_cam,
        "center_world": center_world,
        "extent_cam": extent_cam,
        "visible_points_cam": subject_cam,
        "visible_points_world": subject_world,
        "coverage": float(hit_mask.mean()),
        "num_visible_points": int(visible_cam.shape[0]),
        "num_subject_points": int(subject_cam.shape[0]),
        "median_depth": trim_stats["median_depth"],
        "depth_cut": trim_stats["depth_cut"],
        "selected_fraction": trim_stats["selected_fraction"],
    }


def build_overview_views(
    *,
    layout: str,
    target_c2w: np.ndarray,
    context_positions: np.ndarray,
    subject: dict[str, Any],
    retreat_ratio: float,
    side_margin_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    target_position = target_c2w[:3, 3].astype(np.float64)
    target_right = normalize_vector(target_c2w[:3, 0].astype(np.float64))
    target_up = normalize_vector(target_c2w[:3, 1].astype(np.float64))
    target_forward = normalize_vector(target_c2w[:3, 2].astype(np.float64))

    subject_center_cam = np.asarray(subject["center_cam"], dtype=np.float64)
    subject_center_world = np.asarray(subject["center_world"], dtype=np.float64)
    subject_extent_cam = np.asarray(subject["extent_cam"], dtype=np.float64)
    subject_width = float(max(subject_extent_cam[0], 1e-6))
    subject_height = float(max(subject_extent_cam[1], 1e-6))
    subject_depth_span = float(max(subject_extent_cam[2], 1e-6))
    subject_depth = float(max(subject_center_cam[2], 1e-6))

    context_offsets = (context_positions - target_position[None, :]) @ target_right
    context_span = float(context_offsets.max() - context_offsets.min()) if context_offsets.size else 0.0
    capped_context_span = min(context_span, 0.75 * subject_width)

    retreat = max(
        retreat_ratio * subject_depth,
        0.10 * max(subject_width, subject_height, subject_depth_span),
    )
    retreat = min(retreat, 0.50 * subject_depth)

    lateral_shift = (
        0.35 * subject_width
        + 0.25 * capped_context_span
        + side_margin_ratio * subject_width
    )
    lateral_shift = max(lateral_shift, 0.02 * subject_depth)

    center_position = target_position - retreat * target_forward
    view_specs: list[tuple[str, np.ndarray]] = [("center", center_position)]
    if layout == "sweep3":
        view_specs = [
            ("left", center_position - lateral_shift * target_right),
            ("center", center_position),
            ("right", center_position + lateral_shift * target_right),
        ]
    elif layout != "single":
        raise ValueError(f"Unknown overview layout: {layout}")

    views: list[dict[str, Any]] = []
    for name, position in view_specs:
        c2w = build_look_at_c2w(position=position, target=subject_center_world, up_hint=target_up)
        views.append(
            {
                "name": name,
                "position": position,
                "c2w": c2w,
            }
        )

    return views, {
        "retreat": float(retreat),
        "lateral_shift": float(lateral_shift),
        "subject_width": subject_width,
        "subject_height": subject_height,
        "subject_depth_span": subject_depth_span,
        "subject_depth": subject_depth,
        "context_span": context_span,
    }


def render_mesh_raycast_overview(
    ray_scene: o3d.t.geometry.RaycastingScene,
    triangles: np.ndarray,
    vertex_colors: np.ndarray | None,
    triangle_normals: np.ndarray,
    vertex_normals: np.ndarray | None,
    k_pixel: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
    *,
    appearance: str,
    shading: str,
    shading_ambient: float,
    shading_diffuse: float,
) -> tuple[np.ndarray, float]:
    rays_np, result = cast_rays_pinhole(ray_scene, k_pixel, w2c, width, height)
    prim_ids = result["primitive_ids"]
    prim_uvs = result["primitive_uvs"]
    ray_dirs = rays_np[..., 3:6]

    rendered = np.zeros((height, width, 3), dtype=np.float64)
    hit_mask = prim_ids != ray_scene.INVALID_ID

    if hit_mask.any():
        tri_ids = prim_ids[hit_mask]
        u = prim_uvs[hit_mask, 0]
        v = prim_uvs[hit_mask, 1]
        w0 = 1.0 - u - v
        vertex_indices = triangles[tri_ids]

        if appearance == "textured" and vertex_colors is not None:
            c0 = vertex_colors[vertex_indices[:, 0]]
            c1 = vertex_colors[vertex_indices[:, 1]]
            c2 = vertex_colors[vertex_indices[:, 2]]
            base_color = w0[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2
        elif appearance in {"textured", "untextured"}:
            base_color = np.broadcast_to(DEFAULT_UNTEXTURED_COLOR, (tri_ids.shape[0], 3))
        else:
            raise ValueError(f"Unknown appearance mode: {appearance}")

        if shading != "none":
            if shading == "face" or vertex_normals is None:
                shading_normals = triangle_normals[tri_ids]
            elif shading == "vertex":
                n0 = vertex_normals[vertex_indices[:, 0]]
                n1 = vertex_normals[vertex_indices[:, 1]]
                n2 = vertex_normals[vertex_indices[:, 2]]
                shading_normals = w0[:, None] * n0 + u[:, None] * n1 + v[:, None] * n2
            else:
                raise ValueError(f"Unknown shading mode: {shading}")

            shading_normals = normalize_vectors_np(shading_normals)
            view_dirs = normalize_vectors_np(-ray_dirs[hit_mask].astype(np.float64))
            diffuse = np.clip(np.sum(shading_normals * view_dirs, axis=-1), 0.0, 1.0)
            shade = np.clip(shading_ambient + shading_diffuse * diffuse, 0.0, 1.0)
            base_color = base_color * shade[:, None]

        rendered[hit_mask] = base_color

    coverage = float(hit_mask.mean())
    rendered_u8 = (rendered * 255.0).clip(0, 255).astype(np.uint8)
    return rendered_u8, coverage


def main() -> None:
    args = parse_args()

    if args.overview_fx_scale <= 0.0:
        raise ValueError("--overview_fx_scale must be > 0.")
    if args.overview_retreat_ratio < 0.0:
        raise ValueError("--overview_retreat_ratio must be >= 0.")
    if args.overview_side_margin_ratio < 0.0:
        raise ValueError("--overview_side_margin_ratio must be >= 0.")

    test_output = Path(args.test_output).expanduser()
    settings = resolve_runtime_settings(args, test_output)

    dataset_key: str = settings["dataset_key"]
    data_root: Path = settings["data_root"]
    stage_root = resolve_stage_root(data_root, args.split)
    eval_index_paths: list[Path] = settings["eval_index_paths"]
    eval_indices = load_eval_indices(eval_index_paths)
    target_h, target_w = settings["image_shape"]
    orig_h, orig_w = settings["orig_image_shape"]
    pose_norm_method: str = settings["pose_norm_method"]
    relative_pose: bool = settings["relative_pose"]

    if args.scene:
        scenes = []
        for scene in args.scene:
            scenes.append((scene, resolve_scene_dir(test_output, scene, args.mesh_path)))
    else:
        scenes = discover_scene_dirs(test_output, args.mesh_file, args.mesh_path)

    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise ValueError(f"No scenes found in {test_output}")

    appearance_modes = ["textured", "untextured"] if args.appearance == "both" else [args.appearance]

    print(f"Dataset:          {dataset_key}")
    print(f"Dataset root:     {stage_root}")
    print(f"Test output:      {test_output}")
    if eval_index_paths:
        print(f"Eval index:       {', '.join(str(path) for path in eval_index_paths)}")
    else:
        print("Eval index:       None")
    hydra_dir = settings["hydra_defaults"].get("hydra_dir")
    print(f"Hydra config:     {hydra_dir if hydra_dir is not None else 'None'}")
    print(f"Pose norm method: {pose_norm_method}")
    print(f"Relative pose:    {relative_pose}")
    print(f"Image shape:      {target_h}x{target_w}")
    print(f"Overview layout:  {args.overview_layout}")
    print(f"Appearance:       {', '.join(appearance_modes)}")
    print(
        f"Shading:          {args.shading}"
        if args.shading == "none"
        else f"Shading:          {args.shading} (ambient={args.shading_ambient:.2f}, diffuse={args.shading_diffuse:.2f})"
    )
    print(
        "Overview camera:  "
        f"fx_scale={args.overview_fx_scale:.2f} "
        f"retreat_ratio={args.overview_retreat_ratio:.2f} "
        f"side_margin_ratio={args.overview_side_margin_ratio:.2f}"
    )
    print(f"Scenes:           {len(scenes)}")
    print()

    for scene, scene_dir in scenes:
        print(f"=== {scene} ===")
        try:
            mesh_path = Path(args.mesh_path) if args.mesh_path is not None else infer_mesh_path(scene_dir, args.mesh_file)
            mesh_path = mesh_path.resolve()
            print(f"  Mesh: {mesh_path}")

            example = load_scene_example(stage_root, scene)
            c2w_all, k_all = convert_poses(example["cameras"])

            selected_eval_path, selected_eval_entry = pick_eval_entry(scene, scene_dir, eval_indices)
            context_indices, target_indices = select_context_target_indices(
                scene,
                {} if selected_eval_entry is None else {scene: selected_eval_entry},
                args.context_indices,
                args.target_indices,
            )

            if selected_eval_path is not None:
                print(f"  Eval entry: {selected_eval_path}")

            if not context_indices:
                if pose_norm_method != "none" or relative_pose:
                    raise ValueError(
                        f"Scene {scene} has no context indices. "
                        "Provide --eval_index or --context_indices to match the mesh coordinate system."
                    )
                context_indices = [0]

            c2w_all, scale = apply_project_pose_normalization(
                c2w_all,
                context_indices,
                pose_norm_method,
                relative_pose,
            )
            print(f"  Context idx: {context_indices}")
            print(f"  Target idx:  {target_indices}")
            print(f"  Pose scale:  {scale:.6f}")

            render_indices = select_render_indices(
                args.frames,
                scene_dir,
                example,
                context_indices,
                target_indices,
            )
            if args.max_frames is not None:
                render_indices = render_indices[: args.max_frames]
            print(f"  Anchor frames: {len(render_indices)} ({args.frames})")

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise ValueError(f"Predicted mesh has no geometry: {mesh_path}")
            print(f"  Mesh stats:    {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
            if not mesh.has_vertex_colors():
                print("  Note: mesh has no vertex colors; textured renders will fall back to gray shading.")

            print("  Building raycast scene ...", end="", flush=True)
            ray_scene, triangles, vertex_colors, triangle_normals, vertex_normals = build_ray_scene(mesh)
            print(" done")

            overview_root = (scene_dir / "mesh_render" / mesh_path.stem / "overview").resolve()
            overview_root.mkdir(parents=True, exist_ok=True)
            appearance_roots = {mode: overview_root / mode for mode in appearance_modes}
            for root in appearance_roots.values():
                root.mkdir(parents=True, exist_ok=True)

            context_positions = c2w_all[context_indices, :3, 3].numpy().astype(np.float64)
            frame_rows: list[dict[str, Any]] = []

            for frame_index in render_indices:
                target_c2w = c2w_all[frame_index].numpy().astype(np.float64)
                target_w2c = np.linalg.inv(target_c2w)
                k_norm = k_all[frame_index].numpy().astype(np.float64)
                k_adj = adjust_intrinsics_for_crop(k_norm, orig_h, orig_w, target_h, target_w)
                k_pixel = norm_to_pixel_intrinsics(k_adj, target_w, target_h)
                k_pixel_overview = k_pixel.copy()
                k_pixel_overview[0, 0] *= float(args.overview_fx_scale)

                subject = estimate_visible_subject(
                    ray_scene=ray_scene,
                    k_pixel=k_pixel,
                    c2w=target_c2w,
                    width=target_w,
                    height=target_h,
                )
                overview_views, view_stats = build_overview_views(
                    layout=args.overview_layout,
                    target_c2w=target_c2w,
                    context_positions=context_positions,
                    subject=subject,
                    retreat_ratio=float(args.overview_retreat_ratio),
                    side_margin_ratio=float(args.overview_side_margin_ratio),
                )

                frame_view_rows: list[dict[str, Any]] = []
                for view in overview_views:
                    view_name = str(view["name"])
                    overview_c2w = np.asarray(view["c2w"], dtype=np.float64)
                    overview_w2c = np.linalg.inv(overview_c2w)

                    appearance_rows: list[dict[str, Any]] = []
                    for appearance in appearance_modes:
                        rendered_u8, coverage = render_mesh_raycast_overview(
                            ray_scene,
                            triangles,
                            vertex_colors,
                            triangle_normals,
                            vertex_normals,
                            k_pixel_overview,
                            overview_w2c,
                            target_w,
                            target_h,
                            appearance=appearance,
                            shading=args.shading,
                            shading_ambient=float(args.shading_ambient),
                            shading_diffuse=float(args.shading_diffuse),
                        )
                        image_path = appearance_roots[appearance] / f"{frame_index:06d}_{view_name}.png"
                        Image.fromarray(rendered_u8).save(image_path)
                        appearance_rows.append(
                            {
                                "appearance": appearance,
                                "image_path": str(image_path.resolve()),
                                "coverage": float(coverage),
                            }
                        )
                        print(
                            f"    frame {frame_index:>4d} {view_name:>8s} {appearance:>10s}"
                            f" -> {image_path.name}  coverage={coverage:.3f}"
                        )

                    frame_view_rows.append(
                        {
                            "name": view_name,
                            "position": np.asarray(view["position"], dtype=np.float64).tolist(),
                            "look_at": np.asarray(subject["center_world"], dtype=np.float64).tolist(),
                            "renderings": appearance_rows,
                        }
                    )

                frame_rows.append(
                    {
                        "frame_index": int(frame_index),
                        "anchor_camera_position": target_c2w[:3, 3].tolist(),
                        "anchor_forward": target_c2w[:3, 2].tolist(),
                        "anchor_w2c": target_w2c.tolist(),
                        "anchor_target_coverage": float(subject["coverage"]),
                        "overview_fx_scale": float(args.overview_fx_scale),
                        "retreat": float(view_stats["retreat"]),
                        "lateral_shift": float(view_stats["lateral_shift"]),
                        "subject_width": float(view_stats["subject_width"]),
                        "subject_height": float(view_stats["subject_height"]),
                        "subject_depth_span": float(view_stats["subject_depth_span"]),
                        "subject_depth": float(view_stats["subject_depth"]),
                        "context_span": float(view_stats["context_span"]),
                        "visible_subject_center_cam": np.asarray(subject["center_cam"], dtype=np.float64).tolist(),
                        "visible_subject_center_world": np.asarray(subject["center_world"], dtype=np.float64).tolist(),
                        "visible_subject_extent_cam": np.asarray(subject["extent_cam"], dtype=np.float64).tolist(),
                        "visible_subject_points": int(subject["num_subject_points"]),
                        "visible_mesh_points": int(subject["num_visible_points"]),
                        "visible_subject_depth_cut": float(subject["depth_cut"]),
                        "visible_subject_selection_fraction": float(subject["selected_fraction"]),
                        "views": frame_view_rows,
                    }
                )

            manifest = {
                "scene": scene,
                "dataset": dataset_key,
                "scene_dir": str(scene_dir.resolve()),
                "mesh_path": str(mesh_path),
                "overview_root": str(overview_root),
                "overview_layout": args.overview_layout,
                "appearance": args.appearance,
                "appearance_modes": appearance_modes,
                "shading": args.shading,
                "shading_ambient": float(args.shading_ambient),
                "shading_diffuse": float(args.shading_diffuse),
                "image_shape": [int(target_h), int(target_w)],
                "orig_image_shape": [int(orig_h), int(orig_w)],
                "pose_norm_method": pose_norm_method,
                "relative_pose": bool(relative_pose),
                "pose_scale": float(scale),
                "eval_index_paths": [str(path.resolve()) for path in eval_index_paths],
                "selected_eval_index": None if selected_eval_path is None else str(selected_eval_path.resolve()),
                "context_indices": [int(index) for index in context_indices],
                "target_indices": [int(index) for index in target_indices],
                "render_indices": [int(index) for index in render_indices],
                "mesh_has_vertex_colors": bool(mesh.has_vertex_colors()),
                "num_vertices": int(len(mesh.vertices)),
                "num_triangles": int(len(mesh.triangles)),
                "frames": frame_rows,
            }
            manifest_path = overview_root / args.manifest_name
            write_json(manifest_path, manifest)
            print(f"  Manifest:      {manifest_path}")
        except Exception as exc:
            print(f"  Failed: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
