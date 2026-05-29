"""
Render a mesh from a user-controlled camera pose derived from a target view.

The script loads the target camera from the dataset scene, normalizes it into
the mesh coordinate system, then applies explicit local-camera translations and
rotations before rendering textured and untextured outputs.
"""

from __future__ import annotations

import argparse
import json
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
    load_eval_indices,
    load_hydra_runtime_defaults,
    load_scene_example,
    norm_to_pixel_intrinsics,
    normalize_vectors_np,
    resolve_eval_index_paths,
    resolve_shape,
    resolve_stage_root,
    write_json,
)


DEFAULT_FX_SCALE = 1.0
DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)
DEFAULT_UNTEXTURED_COLOR = np.array([0.75, 0.75, 0.75], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a mesh from a user-controlled camera pose derived from a target view."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Mesh file to render.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Dataset scene key used to load the target camera.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        required=True,
        help="Absolute frame index inside the scene used as the target camera anchor.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Dataset root. When omitted, the script first tries the nearest .hydra/config.yaml, "
            f"then falls back to {DEFAULT_DATA_ROOT}."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split used to resolve index.json and chunks.",
    )
    parser.add_argument(
        "--eval_index",
        type=str,
        default="auto",
        help="Evaluation index JSON. Used to recover context indices for pose normalization.",
    )
    parser.add_argument(
        "--context_indices",
        type=int,
        nargs="*",
        default=None,
        help="Manual context indices override used for pose normalization.",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Target input image shape after the dataset crop shim.",
    )
    parser.add_argument(
        "--orig_image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Original packed image shape before crop/rescale.",
    )
    parser.add_argument(
        "--pose_norm_method",
        type=str,
        default=None,
        choices=POSE_NORM_METHODS,
        help="Pose normalization method used by the current project.",
    )
    parser.add_argument(
        "--relative_pose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to normalize poses relative to the first context camera.",
    )
    parser.add_argument(
        "--appearance",
        choices=("textured", "untextured", "both"),
        default="both",
        help="Whether to render the vertex-colored mesh, the gray untextured mesh, or both.",
    )
    parser.add_argument(
        "--mesh_transform_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a 4x4 transform to apply to the mesh before rendering. "
            "Useful for aligning meshes exported in raw-world and normalized coordinate spaces."
        ),
    )
    parser.add_argument(
        "--mesh_transform_key",
        type=str,
        default="matrix",
        help=(
            "Key inside --mesh_transform_json containing the 4x4 matrix, e.g. "
            "'normalized_from_world' or 'world_from_normalized'."
        ),
    )
    parser.add_argument(
        "--shading",
        choices=("none", "face", "vertex"),
        default="face",
        help="Preview shading mode applied to the rendered images.",
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
        "--fx_scale",
        type=float,
        default=DEFAULT_FX_SCALE,
        help="Horizontal focal scaling applied to the target-view intrinsics.",
    )
    parser.add_argument(
        "--fy_scale",
        type=float,
        default=None,
        help="Vertical focal scaling applied to the target-view intrinsics. Defaults to --fx_scale.",
    )
    parser.add_argument(
        "--offset_back",
        type=float,
        default=0.0,
        help="Move backward along the target camera's local -forward direction.",
    )
    parser.add_argument(
        "--offset_forward",
        type=float,
        default=0.0,
        help="Move forward along the target camera's local +forward direction.",
    )
    parser.add_argument(
        "--offset_left",
        type=float,
        default=0.0,
        help="Move left along the target camera's local -right direction.",
    )
    parser.add_argument(
        "--offset_right",
        type=float,
        default=0.0,
        help="Move right along the target camera's local +right direction.",
    )
    parser.add_argument(
        "--offset_up",
        type=float,
        default=0.0,
        help="Move upward in semantic coordinates. Internally this is opposite to the repo's +Y-down axis.",
    )
    parser.add_argument(
        "--offset_down",
        type=float,
        default=0.0,
        help="Move downward in semantic coordinates.",
    )
    parser.add_argument(
        "--yaw_left_deg",
        type=float,
        default=0.0,
        help="Rotate the view left relative to the target camera orientation.",
    )
    parser.add_argument(
        "--yaw_right_deg",
        type=float,
        default=0.0,
        help="Rotate the view right relative to the target camera orientation.",
    )
    parser.add_argument(
        "--pitch_down_deg",
        type=float,
        default=0.0,
        help="Rotate the view downward relative to the target camera orientation.",
    )
    parser.add_argument(
        "--pitch_up_deg",
        type=float,
        default=0.0,
        help="Rotate the view upward relative to the target camera orientation.",
    )
    parser.add_argument(
        "--roll_ccw_deg",
        type=float,
        default=0.0,
        help="Rotate the image counter-clockwise around the forward axis.",
    )
    parser.add_argument(
        "--roll_cw_deg",
        type=float,
        default=0.0,
        help="Rotate the image clockwise around the forward axis.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional base output directory. The final output is written under <output_root>/<output_name>/.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output directory name. Defaults to a name derived from frame index and pose parameters.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default=DEFAULT_MANIFEST_NAME,
        help="Manifest file name written into the output directory.",
    )
    return parser.parse_args()


def resolve_runtime_settings(
    args: argparse.Namespace,
    mesh_path: Path,
) -> dict[str, Any]:
    hydra_defaults = load_hydra_runtime_defaults(mesh_path.parent)
    data_root = Path(args.data_root).expanduser() if args.data_root is not None else (hydra_defaults.get("data_root") or DEFAULT_DATA_ROOT)
    image_shape = resolve_shape(args.image_shape, hydra_defaults.get("image_shape"), DEFAULT_IMAGE_SHAPE)
    orig_image_shape = resolve_shape(args.orig_image_shape, hydra_defaults.get("orig_image_shape"), DEFAULT_ORIG_IMAGE_SHAPE)
    pose_norm_method = args.pose_norm_method or hydra_defaults.get("pose_norm_method") or "max_pairwise_d"
    relative_pose = args.relative_pose if args.relative_pose is not None else hydra_defaults.get("relative_pose")
    if relative_pose is None:
        relative_pose = True

    eval_index_paths = resolve_eval_index_paths(
        args.eval_index,
        hydra_defaults.get("eval_index_path"),
    )

    return {
        "hydra_defaults": hydra_defaults,
        "data_root": data_root,
        "image_shape": image_shape,
        "orig_image_shape": orig_image_shape,
        "pose_norm_method": pose_norm_method,
        "relative_pose": bool(relative_pose),
        "eval_index_paths": eval_index_paths,
    }


def pick_scene_eval_entry(
    scene: str,
    eval_indices: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path | None, dict[str, Any] | None]:
    for path, eval_index in eval_indices:
        entry = eval_index.get(scene)
        if entry is not None:
            return path, entry
    return None, None


def load_mesh_transform(
    transform_json: str | None,
    transform_key: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if transform_json is None:
        return None, None

    path = Path(transform_json).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh transform JSON not found: {path}")

    with path.open("r") as f:
        payload = json.load(f)
    if transform_key not in payload:
        raise KeyError(f"Transform key '{transform_key}' not found in {path}")

    matrix = np.asarray(payload[transform_key], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Transform '{transform_key}' in {path} must be a 4x4 matrix, got {matrix.shape}."
        )

    return matrix, {
        "json_path": str(path),
        "key": transform_key,
        "matrix": matrix.tolist(),
    }


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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


def render_mesh_raycast(
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

    rendered = np.full((height, width, 3), DEFAULT_BACKGROUND_COLOR, dtype=np.float64)
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


def resolve_pose_parameters(args: argparse.Namespace) -> dict[str, float]:
    semantic_forward = float(args.offset_forward) - float(args.offset_back)
    semantic_right = float(args.offset_right) - float(args.offset_left)
    semantic_up = float(args.offset_up) - float(args.offset_down)

    yaw_left_deg = float(args.yaw_left_deg) - float(args.yaw_right_deg)
    pitch_down_deg = float(args.pitch_down_deg) - float(args.pitch_up_deg)
    roll_ccw_deg = float(args.roll_ccw_deg) - float(args.roll_cw_deg)

    return {
        "semantic_forward": semantic_forward,
        "semantic_right": semantic_right,
        "semantic_up": semantic_up,
        "yaw_left_deg": yaw_left_deg,
        "pitch_down_deg": pitch_down_deg,
        "roll_ccw_deg": roll_ccw_deg,
    }


def build_target_pose(
    target_c2w: np.ndarray,
    pose_params: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    target_position = target_c2w[:3, 3].astype(np.float64)
    target_right = normalize_vector(target_c2w[:3, 0].astype(np.float64))
    target_down = normalize_vector(target_c2w[:3, 1].astype(np.float64))
    target_forward = normalize_vector(target_c2w[:3, 2].astype(np.float64))
    target_up = -target_down

    target_basis_semantic = np.column_stack([target_right, target_up, target_forward])

    yaw_rad = np.deg2rad(-pose_params["yaw_left_deg"])
    pitch_rad = np.deg2rad(pose_params["pitch_down_deg"])
    roll_rad = np.deg2rad(pose_params["roll_ccw_deg"])
    local_rotation = rotation_matrix_z(roll_rad) @ rotation_matrix_x(pitch_rad) @ rotation_matrix_y(yaw_rad)

    rotated_basis_semantic = target_basis_semantic @ local_rotation
    rendered_right = normalize_vector(rotated_basis_semantic[:, 0])
    rendered_up = normalize_vector(rotated_basis_semantic[:, 1])
    rendered_forward = normalize_vector(rotated_basis_semantic[:, 2])
    rendered_down = -rendered_up

    translation_world = (
        pose_params["semantic_right"] * target_right
        + pose_params["semantic_up"] * target_up
        + pose_params["semantic_forward"] * target_forward
    )
    rendered_position = target_position + translation_world

    rendered_c2w = np.eye(4, dtype=np.float64)
    rendered_c2w[:3, 0] = rendered_right
    rendered_c2w[:3, 1] = rendered_down
    rendered_c2w[:3, 2] = rendered_forward
    rendered_c2w[:3, 3] = rendered_position

    return rendered_c2w, {
        "target_position": target_position,
        "target_right": target_right,
        "target_down": target_down,
        "target_forward": target_forward,
        "target_up": target_up,
        "translation_world": translation_world,
        "local_rotation_matrix": local_rotation,
        "rendered_position": rendered_position,
        "rendered_right": rendered_right,
        "rendered_down": rendered_down,
        "rendered_forward": rendered_forward,
        "rendered_up": rendered_up,
    }


def derive_default_output_name(
    frame_index: int,
    pose_params: dict[str, float],
    fx_scale: float,
    fy_scale: float,
) -> str:
    return (
        f"frame_{frame_index:06d}"
        f"_back_{-pose_params['semantic_forward']:.3f}"
        f"_right_{pose_params['semantic_right']:.3f}"
        f"_up_{pose_params['semantic_up']:.3f}"
        f"_yaw_left_{pose_params['yaw_left_deg']:.1f}"
        f"_pitch_down_{pose_params['pitch_down_deg']:.1f}"
        f"_roll_ccw_{pose_params['roll_ccw_deg']:.1f}"
        f"_fx_{fx_scale:.3f}"
        f"_fy_{fy_scale:.3f}"
    )


def resolve_output_dir(
    mesh_path: Path,
    output_root: str | None,
    output_name: str | None,
    frame_index: int,
    pose_params: dict[str, float],
    fx_scale: float,
    fy_scale: float,
) -> Path:
    if mesh_path.parent.name == "mesh":
        default_base_root = mesh_path.parent.parent / "mesh_render" / mesh_path.stem / "target_pose"
    else:
        default_base_root = mesh_path.parent / "mesh_render" / mesh_path.stem / "target_pose"

    base_root = Path(output_root).expanduser() if output_root is not None else default_base_root
    name = output_name or derive_default_output_name(frame_index, pose_params, fx_scale, fy_scale)
    return (base_root / name).resolve()


def main() -> None:
    args = parse_args()

    mesh_path = Path(args.mesh_path).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if args.fx_scale <= 0.0:
        raise ValueError("--fx_scale must be > 0.")
    fy_scale = float(args.fx_scale if args.fy_scale is None else args.fy_scale)
    if fy_scale <= 0.0:
        raise ValueError("--fy_scale must be > 0.")

    settings = resolve_runtime_settings(args, mesh_path)
    data_root: Path = settings["data_root"]
    stage_root = resolve_stage_root(data_root, args.split)
    eval_index_paths: list[Path] = settings["eval_index_paths"]
    eval_indices = load_eval_indices(eval_index_paths)
    target_h, target_w = settings["image_shape"]
    orig_h, orig_w = settings["orig_image_shape"]
    pose_norm_method: str = settings["pose_norm_method"]
    relative_pose: bool = settings["relative_pose"]

    appearance_modes = ["textured", "untextured"] if args.appearance == "both" else [args.appearance]
    pose_params = resolve_pose_parameters(args)
    mesh_transform = None
    mesh_transform_info = None
    if args.mesh_transform_json is not None:
        mesh_transform, mesh_transform_info = load_mesh_transform(
            args.mesh_transform_json,
            args.mesh_transform_key,
        )
    output_dir = resolve_output_dir(
        mesh_path=mesh_path,
        output_root=args.output_root,
        output_name=args.output_name,
        frame_index=int(args.frame_index),
        pose_params=pose_params,
        fx_scale=float(args.fx_scale),
        fy_scale=fy_scale,
    )

    print(f"Dataset root:     {stage_root}")
    print(f"Mesh:             {mesh_path}")
    print(f"Scene:            {args.scene}")
    print(f"Frame index:      {args.frame_index}")
    if eval_index_paths:
        print(f"Eval index:       {', '.join(str(path) for path in eval_index_paths)}")
    else:
        print("Eval index:       None")
    hydra_dir = settings["hydra_defaults"].get("hydra_dir")
    print(f"Hydra config:     {hydra_dir if hydra_dir is not None else 'None'}")
    print(f"Pose norm method: {pose_norm_method}")
    print(f"Relative pose:    {relative_pose}")
    print(f"Image shape:      {target_h}x{target_w}")
    print(f"Appearance:       {', '.join(appearance_modes)}")
    if mesh_transform_info is not None:
        print(f"Mesh transform:   {mesh_transform_info['key']} from {mesh_transform_info['json_path']}")
    else:
        print("Mesh transform:   None")
    print(
        f"Shading:          {args.shading}"
        if args.shading == "none"
        else f"Shading:          {args.shading} (ambient={args.shading_ambient:.2f}, diffuse={args.shading_diffuse:.2f})"
    )
    print(
        "Pose params:      "
        f"back={args.offset_back:.3f} forward={args.offset_forward:.3f} "
        f"left={args.offset_left:.3f} right={args.offset_right:.3f} "
        f"up={args.offset_up:.3f} down={args.offset_down:.3f} "
        f"yaw_left={args.yaw_left_deg:.2f} yaw_right={args.yaw_right_deg:.2f} "
        f"pitch_down={args.pitch_down_deg:.2f} pitch_up={args.pitch_up_deg:.2f} "
        f"roll_ccw={args.roll_ccw_deg:.2f} roll_cw={args.roll_cw_deg:.2f}"
    )
    print(f"Output dir:       {output_dir}")
    print()

    example = load_scene_example(stage_root, args.scene)
    num_frames = len(example["images"])
    if args.frame_index < 0 or args.frame_index >= num_frames:
        raise IndexError(f"--frame_index {args.frame_index} is out of range for scene {args.scene} with {num_frames} frames.")

    c2w_all, k_all = convert_poses(example["cameras"])
    selected_eval_path, selected_eval_entry = pick_scene_eval_entry(args.scene, eval_indices)

    context_indices = list(args.context_indices) if args.context_indices is not None else []
    target_indices: list[int] = []
    if selected_eval_entry is not None:
        if not context_indices:
            context_indices = [int(index) for index in selected_eval_entry.get("context", [])]
        target_indices = [int(index) for index in selected_eval_entry.get("target", [])]

    if selected_eval_path is not None:
        print(f"Selected eval entry: {selected_eval_path}")
    if not context_indices:
        if pose_norm_method != "none" or relative_pose:
            raise ValueError(
                f"Scene {args.scene} has no context indices available. "
                "Pass --context_indices explicitly or provide an eval index containing the scene."
            )
        context_indices = [0]

    c2w_all, scale = apply_project_pose_normalization(
        c2w_all,
        context_indices,
        pose_norm_method,
        relative_pose,
    )
    print(f"Context idx:      {context_indices}")
    print(f"Target idx:       {target_indices}")
    print(f"Pose scale:       {scale:.6f}")

    target_c2w = c2w_all[args.frame_index].numpy().astype(np.float64)
    k_norm = k_all[args.frame_index].numpy().astype(np.float64)
    k_adj = adjust_intrinsics_for_crop(k_norm, orig_h, orig_w, target_h, target_w)
    k_pixel = norm_to_pixel_intrinsics(k_adj, target_w, target_h)
    k_pixel_render = k_pixel.copy()
    k_pixel_render[0, 0] *= float(args.fx_scale)
    k_pixel_render[1, 1] *= float(fy_scale)

    rendered_c2w, pose_debug = build_target_pose(target_c2w, pose_params)
    rendered_w2c = np.linalg.inv(rendered_c2w)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Predicted mesh has no geometry: {mesh_path}")
    if mesh_transform is not None:
        mesh.transform(mesh_transform)
    print(f"Mesh stats:       {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    if not mesh.has_vertex_colors():
        print("Note: mesh has no vertex colors; textured renders will fall back to gray shading.")

    print("Building raycast scene ...", end="", flush=True)
    ray_scene, triangles, vertex_colors, triangle_normals, vertex_normals = build_ray_scene(mesh)
    print(" done")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    scene_prefix = args.scene[:3]
    for appearance in appearance_modes:
        rendered_u8, coverage = render_mesh_raycast(
            ray_scene,
            triangles,
            vertex_colors,
            triangle_normals,
            vertex_normals,
            k_pixel_render,
            rendered_w2c,
            target_w,
            target_h,
            appearance=appearance,
            shading=args.shading,
            shading_ambient=float(args.shading_ambient),
            shading_diffuse=float(args.shading_diffuse),
        )
        image_path = output_dir / f"{scene_prefix}_{appearance}.png"
        Image.fromarray(rendered_u8).save(image_path)
        outputs[appearance] = {
            "image_path": str(image_path.resolve()),
            "coverage": float(coverage),
        }
        print(f"  {appearance:>10s} -> {image_path.name}  coverage={coverage:.3f}")

    manifest = {
        "scene": args.scene,
        "mesh_path": str(mesh_path),
        "frame_index": int(args.frame_index),
        "output_dir": str(output_dir),
        "eval_index_paths": [str(path.resolve()) for path in eval_index_paths],
        "selected_eval_index_path": None if selected_eval_path is None else str(selected_eval_path.resolve()),
        "pose_norm_method": pose_norm_method,
        "relative_pose": bool(relative_pose),
        "pose_scale": float(scale),
        "context_indices": [int(index) for index in context_indices],
        "target_indices": [int(index) for index in target_indices],
        "image_shape": [int(target_h), int(target_w)],
        "orig_image_shape": [int(orig_h), int(orig_w)],
        "appearance": args.appearance,
        "appearance_modes": appearance_modes,
        "mesh_transform": mesh_transform_info,
        "shading": args.shading,
        "shading_ambient": float(args.shading_ambient),
        "shading_diffuse": float(args.shading_diffuse),
        "background_color": DEFAULT_BACKGROUND_COLOR.tolist(),
        "intrinsics": {
            "anchor_k_pixel": k_pixel.tolist(),
            "render_k_pixel": k_pixel_render.tolist(),
            "fx_scale": float(args.fx_scale),
            "fy_scale": float(fy_scale),
        },
        "raw_pose_parameters": {
            "offset_back": float(args.offset_back),
            "offset_forward": float(args.offset_forward),
            "offset_left": float(args.offset_left),
            "offset_right": float(args.offset_right),
            "offset_up": float(args.offset_up),
            "offset_down": float(args.offset_down),
            "yaw_left_deg": float(args.yaw_left_deg),
            "yaw_right_deg": float(args.yaw_right_deg),
            "pitch_down_deg": float(args.pitch_down_deg),
            "pitch_up_deg": float(args.pitch_up_deg),
            "roll_ccw_deg": float(args.roll_ccw_deg),
            "roll_cw_deg": float(args.roll_cw_deg),
        },
        "resolved_pose_parameters": pose_params,
        "anchor_pose": {
            "c2w": target_c2w.tolist(),
            "position": pose_debug["target_position"].tolist(),
            "right": pose_debug["target_right"].tolist(),
            "down": pose_debug["target_down"].tolist(),
            "forward": pose_debug["target_forward"].tolist(),
            "up_semantic": pose_debug["target_up"].tolist(),
        },
        "render_pose": {
            "c2w": rendered_c2w.tolist(),
            "w2c": rendered_w2c.tolist(),
            "position": pose_debug["rendered_position"].tolist(),
            "right": pose_debug["rendered_right"].tolist(),
            "down": pose_debug["rendered_down"].tolist(),
            "forward": pose_debug["rendered_forward"].tolist(),
            "up_semantic": pose_debug["rendered_up"].tolist(),
            "translation_world": pose_debug["translation_world"].tolist(),
            "local_rotation_matrix": pose_debug["local_rotation_matrix"].tolist(),
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / args.manifest_name
    write_json(manifest_path, manifest)
    print(f"  manifest    -> {manifest_path.name}")


if __name__ == "__main__":
    main()
