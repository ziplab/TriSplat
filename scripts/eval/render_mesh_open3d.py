"""
Render exported meshes from dataset camera viewpoints using Open3D raycasting.

This script renders exported meshes from the current Trisplat test outputs:

- supports the current mesh export names via `--mesh_file auto`
- reproduces the current pose normalization pipeline
  (`pose_norm_method` + `relative_pose`)
- loads ground-truth RGB frames directly from the packed `.torch` dataset,
  so `color_gt/` does not need to exist in the test output

Typical usage:

    # Render all scenes found under a test output directory.
    python scripts/eval/render_mesh_open3d.py \
        --data_root data/re10k \
        --test_output outputs/trisplat_triangle_mesh_direct/trisplat_triangle_mesh_direct

    # Render a single scene and only the target frames from the eval index.
    python scripts/eval/render_mesh_open3d.py \
        --data_root data/re10k \
        --test_output outputs/trisplat_triangle_mesh_direct/trisplat_triangle_mesh_direct \
        --scene 5aca87f95a9412c6 \
        --frames target

    # Use a specific mesh file name.
    python scripts/eval/render_mesh_open3d.py \
        --data_root data/re10k \
        --test_output outputs/trisplat_triangle_mesh_direct/trisplat_triangle_mesh_direct \
        --mesh_file DIRECT_triangle_mesh.ply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import open3d as o3d
import torch
import yaml
from einops import rearrange, repeat
from PIL import Image
from PIL import ImageFilter

from src.evaluation.metrics import (
    compute_lpips,
    compute_psnr as compute_psnr_tensor,
    compute_ssim,
)


DEFAULT_MESH_CANDIDATES = (
    "TSDF_2dgs_post_mesh.ply",
    "DIRECT_triangle_mesh_post.ply",
    "DIRECT_triangle_mesh.ply",
    "TSDF_2dgs_mesh.ply",
    "TSDF_2dgs_post_mesh.off",
    "DIRECT_triangle_mesh_post.off",
    "DIRECT_triangle_mesh.off",
    "TSDF_2dgs_mesh.off",
)
DEFAULT_DATA_ROOT = Path(os.environ.get("RE10K_ROOT", "data/re10k"))
DEFAULT_IMAGE_SHAPE = (256, 256)
DEFAULT_ORIG_IMAGE_SHAPE = (360, 640)
DEFAULT_SUMMARY_NAME = "mesh_render_metrics_summary.json"
DEFAULT_SHADING_AMBIENT = 0.25
DEFAULT_SHADING_DIFFUSE = 0.75
DEFAULT_BLUR_SIGMA = 0.75
HIT_MODES = ("nearest", "front_or_nearest", "front_only")
POSE_NORM_METHODS = (
    "none",
    "start_end",
    "max_pairwise_d",
    "mean_pairwise_d",
    "max_trans",
    "max_view1_d",
)


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML file {path}, got {type(data)}")
    return data


def find_hydra_dir(start_path: Path) -> Path | None:
    candidate_roots = [start_path, *start_path.parents]
    for root in candidate_roots:
        hydra_dir = root / ".hydra"
        if (hydra_dir / "config.yaml").exists():
            return hydra_dir
    return None


def pick_dataset_cfg(config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    datasets = config.get("dataset", {})
    if not isinstance(datasets, dict):
        return None, {}
    if "re10k" in datasets and isinstance(datasets["re10k"], dict):
        return "re10k", datasets["re10k"]
    for key, value in datasets.items():
        if isinstance(value, dict):
            return str(key), value
    return None, {}


def load_hydra_runtime_defaults(test_output: Path) -> dict[str, Any]:
    hydra_dir = find_hydra_dir(test_output.resolve())
    if hydra_dir is None:
        return {}

    config_path = hydra_dir / "config.yaml"
    config = load_yaml_file(config_path)
    dataset_key, dataset_cfg = pick_dataset_cfg(config)
    view_sampler_cfg = dataset_cfg.get("view_sampler", {}) if isinstance(dataset_cfg, dict) else {}

    data_root = None
    roots = dataset_cfg.get("roots", []) if isinstance(dataset_cfg, dict) else []
    if isinstance(roots, list) and roots:
        data_root = Path(roots[0]).expanduser()

    eval_index_path = None
    if isinstance(view_sampler_cfg, dict):
        index_path = view_sampler_cfg.get("index_path")
        if index_path:
            eval_index_path = Path(index_path).expanduser()

    image_shape = dataset_cfg.get("input_image_shape") if isinstance(dataset_cfg, dict) else None
    orig_image_shape = dataset_cfg.get("original_image_shape") if isinstance(dataset_cfg, dict) else None

    return {
        "hydra_dir": hydra_dir,
        "config_path": config_path,
        "dataset_key": dataset_key,
        "data_root": data_root,
        "eval_index_path": eval_index_path,
        "num_context_views": view_sampler_cfg.get("num_context_views") if isinstance(view_sampler_cfg, dict) else None,
        "image_shape": tuple(int(v) for v in image_shape) if isinstance(image_shape, list) and len(image_shape) == 2 else None,
        "orig_image_shape": tuple(int(v) for v in orig_image_shape) if isinstance(orig_image_shape, list) and len(orig_image_shape) == 2 else None,
        "pose_norm_method": dataset_cfg.get("pose_norm_method") if isinstance(dataset_cfg, dict) else None,
        "relative_pose": dataset_cfg.get("relative_pose") if isinstance(dataset_cfg, dict) else None,
    }


def resolve_metric_device(metric_device: str) -> torch.device:
    if metric_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if metric_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--metric_device=cuda requested, but CUDA is not available.")
    return torch.device(metric_device)


def resolve_shape(
    cli_value: list[int] | None,
    hydra_value: tuple[int, int] | None,
    default_value: tuple[int, int],
) -> tuple[int, int]:
    if cli_value is not None:
        return int(cli_value[0]), int(cli_value[1])
    if hydra_value is not None:
        return int(hydra_value[0]), int(hydra_value[1])
    return default_value


def resolve_runtime_settings(
    args: argparse.Namespace,
    test_output: Path,
) -> dict[str, Any]:
    hydra_defaults = load_hydra_runtime_defaults(test_output)

    data_root = Path(args.data_root).expanduser() if args.data_root is not None else hydra_defaults.get("data_root", DEFAULT_DATA_ROOT)
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
    metric_device = resolve_metric_device(args.metric_device)

    return {
        "hydra_defaults": hydra_defaults,
        "data_root": data_root,
        "image_shape": image_shape,
        "orig_image_shape": orig_image_shape,
        "pose_norm_method": pose_norm_method,
        "relative_pose": bool(relative_pose),
        "eval_index_paths": eval_index_paths,
        "metric_device": metric_device,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def images_to_metric_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    if not images:
        raise ValueError("Expected at least one image for metric computation.")
    array = np.stack(images, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def compute_image_metrics(
    gt_images: list[np.ndarray],
    pred_images: list[np.ndarray],
    device: torch.device,
) -> dict[str, list[float]]:
    gt_tensor = images_to_metric_tensor(gt_images, device)
    pred_tensor = images_to_metric_tensor(pred_images, device)
    return {
        "psnr": compute_psnr_tensor(gt_tensor, pred_tensor).detach().cpu().tolist(),
        "ssim": compute_ssim(gt_tensor, pred_tensor).detach().cpu().tolist(),
        "lpips": compute_lpips(gt_tensor, pred_tensor).detach().cpu().tolist(),
    }


def mean_metric(metric_values: list[float]) -> float:
    return float(np.mean(metric_values)) if metric_values else float("nan")


def build_root_summary(
    *,
    scene_results: list[dict[str, Any]],
    failed_scenes: list[dict[str, Any]],
    metric_device: torch.device,
    stage_root: Path,
    test_output: Path,
    eval_index_paths: list[Path],
    pose_norm_method: str,
    relative_pose: bool,
    image_shape: tuple[int, int],
    orig_image_shape: tuple[int, int],
    frames_mode: str,
    shading: str,
    shading_ambient: float,
    shading_diffuse: float,
    hit_mode: str,
    supersample: int,
    render_tag: str | None,
    hydra_defaults: dict[str, Any],
) -> dict[str, Any]:
    frame_metrics = {
        "psnr": [float(frame["psnr"]) for scene in scene_results for frame in scene["frames"]],
        "ssim": [float(frame["ssim"]) for scene in scene_results for frame in scene["frames"]],
        "lpips": [float(frame["lpips"]) for scene in scene_results for frame in scene["frames"]],
    }
    scene_metrics = {
        "psnr": [float(scene["psnr_mean"]) for scene in scene_results],
        "ssim": [float(scene["ssim_mean"]) for scene in scene_results],
        "lpips": [float(scene["lpips_mean"]) for scene in scene_results],
    }
    overall_frame_mean = {
        metric: mean_metric(values)
        for metric, values in frame_metrics.items()
    } if scene_results else {}
    overall_scene_mean = {
        metric: mean_metric(values)
        for metric, values in scene_metrics.items()
    } if scene_results else {}

    scene_rows = []
    for scene in scene_results:
        scene_rows.append(
            {
                "scene": scene["scene"],
                "mesh_path": scene["mesh_path"],
                "metrics_path": scene["metrics_path"],
                "num_frames": scene["num_frames"],
                "psnr_mean": scene["psnr_mean"],
                "ssim_mean": scene["ssim_mean"],
                "lpips_mean": scene["lpips_mean"],
                "coverage_mean": scene["coverage_mean"],
                "context_indices": scene["context_indices"],
                "target_indices": scene["target_indices"],
            }
        )

    return {
        "test_output": str(test_output.resolve()),
        "stage_root": str(stage_root.resolve()),
        "eval_index_paths": [str(path.resolve()) for path in eval_index_paths],
        "hydra_dir": None if hydra_defaults.get("hydra_dir") is None else str(hydra_defaults["hydra_dir"].resolve()),
        "hydra_config_path": None if hydra_defaults.get("config_path") is None else str(hydra_defaults["config_path"].resolve()),
        "metric_device": str(metric_device),
        "frames_mode": frames_mode,
        "shading": shading,
        "shading_ambient": shading_ambient,
        "shading_diffuse": shading_diffuse,
        "hit_mode": hit_mode,
        "supersample": int(supersample),
        "render_tag": render_tag,
        "pose_norm_method": pose_norm_method,
        "relative_pose": relative_pose,
        "image_shape": list(image_shape),
        "orig_image_shape": list(orig_image_shape),
        "num_scenes_total": len(scene_results) + len(failed_scenes),
        "num_scenes_success": len(scene_results),
        "num_scenes_failed": len(failed_scenes),
        "num_frames_total": sum(int(scene["num_frames"]) for scene in scene_results),
        "overall_frame_mean": overall_frame_mean,
        "overall_scene_mean": overall_scene_mean,
        "scenes": scene_rows,
        "failed_scenes": failed_scenes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render exported meshes from dataset viewpoints using headless Open3D."
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
            "Evaluation index JSON. 'auto' tries "
            "assets/evaluation_index_re10k_mesh_6ctx.json, "
            "assets/evaluation_index_re10k_6tx.json."
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
        "--use_frame_meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, use <mesh_stem>_frame_XXXXXX meshes if present. "
            "Disable this for unified scene-level mesh evaluation."
        ),
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
        help=(
            "Which frames to render: union(context,target) from eval index, "
            "target only, PNGs present in <scene>/color/, or all dataset frames."
        ),
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
        help="Optional cap on the number of frames rendered per scene.",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        default=False,
        help="Also save cropped GT frames next to the rendered mesh images.",
    )
    parser.add_argument(
        "--apply_valid_mask_gt",
        action="store_true",
        default=False,
        help="Apply packed valid_masks to GT frames before computing render metrics.",
    )
    parser.add_argument(
        "--write_metrics_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-scene metrics.json under <scene>/mesh_render/<mesh_name>/.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default=DEFAULT_SUMMARY_NAME,
        help="Filename written under <test_output>/ for the root summary JSON.",
    )
    parser.add_argument(
        "--metric_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for PSNR/SSIM/LPIPS computation.",
    )
    parser.add_argument(
        "--shading",
        choices=("none", "face", "vertex"),
        default="none",
        help="Optional preview shading applied on top of mesh vertex colors. Use 'face' for MeshLab-like flat shading.",
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
        "--hit_mode",
        choices=HIT_MODES,
        default="nearest",
        help=(
            "Ray hit selection mode. 'nearest' uses Open3D's closest hit. "
            "'front_or_nearest' prefers the closest front-facing hit and falls back to nearest. "
            "'front_only' drops rays without a front-facing hit."
        ),
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="Render at Nx resolution and downsample for anti-aliased mesh edges.",
    )
    parser.add_argument(
        "--blur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=DEFAULT_BLUR_SIGMA,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--render_tag",
        type=str,
        default=None,
        help="Optional suffix appended to per-scene mesh_render directories to avoid overwriting metrics.",
    )
    parser.add_argument(
        "--save_summary",
        action=argparse.BooleanOptionalAction,
        dest="write_metrics_json",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_stage_root(data_root: Path, split: str) -> Path:
    split_root = data_root / split
    if (split_root / "index.json").exists():
        return split_root
    if (data_root / "index.json").exists():
        return data_root
    raise FileNotFoundError(
        f"Could not find index.json under {split_root} or {data_root}"
    )


def resolve_eval_index_paths(
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
    candidates.extend(
        [
            Path("assets/evaluation_index_re10k_mesh_6ctx.json"),
            Path("assets/evaluation_index_re10k_6tx.json"),
        ]
    )

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


def load_eval_indices(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    indices: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            indices.append((path, json.load(f)))
    return indices


def load_scene_example(stage_root: Path, scene_key: str) -> dict[str, Any]:
    index_path = stage_root / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    if scene_key not in index:
        raise KeyError(f"Scene {scene_key} not found in {index_path}")

    chunk_path = stage_root / index[scene_key]
    chunk = torch_load(chunk_path)
    for example in chunk:
        if example["key"] == scene_key:
            return example

    raise KeyError(f"Scene {scene_key} not found in chunk {chunk_path}")


def convert_poses(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset camera rows to c2w and normalized 3x3 intrinsics."""
    batch, _ = poses.shape

    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=batch).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=batch).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    c2w = torch.inverse(w2c)
    return c2w, intrinsics


def compute_pose_norm_scale(
    context_extrinsics: torch.Tensor,
    method: str,
) -> float:
    if method == "start_end":
        a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
        scale = (a - b).norm()
    elif method == "max_pairwise_d":
        scale = torch.tensor(0.0, dtype=context_extrinsics.dtype)
        for i in range(context_extrinsics.shape[0]):
            for j in range(i + 1, context_extrinsics.shape[0]):
                a, b = context_extrinsics[i, :3, 3], context_extrinsics[j, :3, 3]
                scale = torch.maximum(scale, (a - b).norm())
    elif method == "mean_pairwise_d":
        positions = context_extrinsics[:, :3, 3]
        positions_i = positions.unsqueeze(1)
        positions_j = positions.unsqueeze(0)
        distance_matrix = torch.norm(positions_i - positions_j, dim=2)
        mask = torch.triu(
            torch.ones(distance_matrix.shape[0], distance_matrix.shape[1], dtype=torch.bool),
            diagonal=1,
        )
        scale = distance_matrix[mask].mean()
    elif method == "max_trans":
        scale = torch.max(torch.abs(context_extrinsics[:, :3, 3]))
        scale = torch.norm(scale)
    elif method == "max_view1_d":
        view1 = context_extrinsics[0:1, :3, 3]
        remaining = context_extrinsics[1:, :3, 3]
        scale = (view1 - remaining).norm(dim=-1).max()
    elif method == "none":
        return 1.0
    else:
        raise ValueError(f"Unknown pose norm method: {method}")
    return float(scale.item())


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    canonical = torch.eye(4, dtype=poses.dtype, device=poses.device).unsqueeze(0)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm = torch.bmm(canonical, pivotal_pose_inv)
    return torch.bmm(camera_norm.repeat(poses.shape[0], 1, 1), poses)


def apply_project_pose_normalization(
    c2w_all: torch.Tensor,
    context_indices: list[int],
    pose_norm_method: str,
    relative_pose: bool,
) -> tuple[torch.Tensor, float]:
    if not context_indices:
        raise ValueError("At least one context index is required for pose normalization.")

    c2w_all = c2w_all.clone()
    context_extrinsics = c2w_all[context_indices]
    scale = compute_pose_norm_scale(context_extrinsics, pose_norm_method)
    if scale <= 1e-8:
        raise ValueError(
            f"Pose normalization scale is too small ({scale:.6e}) for context indices {context_indices}"
        )
    c2w_all[:, :3, 3] /= scale

    if relative_pose:
        c2w_all = camera_normalization(c2w_all[context_indices[:1]], c2w_all)

    return c2w_all, scale


def adjust_intrinsics_for_crop(
    k_norm: np.ndarray,
    orig_h: int,
    orig_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    scale_factor = max(target_h / orig_h, target_w / orig_w)
    h_scaled = round(orig_h * scale_factor)
    w_scaled = round(orig_w * scale_factor)

    k_adj = k_norm.copy()
    k_adj[0, 0] *= w_scaled / target_w
    k_adj[1, 1] *= h_scaled / target_h
    return k_adj


def norm_to_pixel_intrinsics(k_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    k_pixel = k_norm.copy()
    k_pixel[0, 0] *= width
    k_pixel[1, 1] *= height
    k_pixel[0, 2] *= width
    k_pixel[1, 2] *= height
    return k_pixel


def decode_image(raw_image: Any) -> Image.Image:
    if isinstance(raw_image, torch.Tensor):
        raw_bytes = raw_image.numpy().tobytes()
    elif isinstance(raw_image, np.ndarray):
        raw_bytes = raw_image.tobytes()
    elif isinstance(raw_image, (bytes, bytearray)):
        raw_bytes = bytes(raw_image)
    else:
        raise TypeError(f"Unsupported image container type: {type(raw_image)}")
    return Image.open(BytesIO(raw_bytes)).convert("RGB")


def crop_and_resize_image(
    image: Image.Image,
    target_h: int,
    target_w: int,
    resample: int = Image.LANCZOS,
) -> np.ndarray:
    scale_factor = max(target_h / image.height, target_w / image.width)
    h_scaled = round(image.height * scale_factor)
    w_scaled = round(image.width * scale_factor)

    image = image.resize((w_scaled, h_scaled), resample)
    row = (h_scaled - target_h) // 2
    col = (w_scaled - target_w) // 2
    image = image.crop((col, row, col + target_w, row + target_h))
    return np.array(image, dtype=np.uint8)


def decode_mask(raw_mask: Any) -> Image.Image:
    if isinstance(raw_mask, torch.Tensor):
        raw_bytes = raw_mask.numpy().tobytes()
    elif isinstance(raw_mask, np.ndarray):
        raw_bytes = raw_mask.tobytes()
    elif isinstance(raw_mask, (bytes, bytearray)):
        raw_bytes = bytes(raw_mask)
    else:
        raise TypeError(f"Unsupported mask container type: {type(raw_mask)}")
    return Image.open(BytesIO(raw_bytes)).convert("L")


def load_gt_frame(
    example: dict[str, Any],
    frame_index: int,
    target_h: int,
    target_w: int,
    apply_valid_mask: bool = False,
) -> np.ndarray:
    image = decode_image(example["images"][frame_index])
    gt = crop_and_resize_image(image, target_h, target_w)
    if not apply_valid_mask:
        return gt
    if "valid_masks" not in example:
        raise KeyError("Cannot apply valid mask to GT frame because example has no valid_masks field.")

    mask = decode_mask(example["valid_masks"][frame_index])
    mask = crop_and_resize_image(mask, target_h, target_w, resample=Image.NEAREST)
    return gt * (mask[..., None] > 0)


def list_saved_prediction_indices(scene_dir: Path) -> list[int]:
    color_dir = scene_dir / "color"
    if not color_dir.exists():
        return []
    indices = []
    for path in sorted(color_dir.glob("*.png")):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return indices


def pick_eval_entry(
    scene: str,
    scene_dir: Path,
    eval_indices: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path | None, dict[str, Any] | None]:
    saved_pred_indices = list_saved_prediction_indices(scene_dir)
    best_path = None
    best_entry = None
    best_score = -10**9

    for priority, (path, eval_index) in enumerate(eval_indices):
        entry = eval_index.get(scene)
        if entry is None:
            continue

        score = 100 - priority
        if saved_pred_indices:
            target = sorted(entry.get("target", []))
            union = sorted(set(entry.get("context", []) + entry.get("target", [])))
            if target == saved_pred_indices:
                score += 1000
            elif all(index in union for index in saved_pred_indices):
                score += 100
            score += 10 * len(set(saved_pred_indices) & set(target))

        if score > best_score:
            best_score = score
            best_path = path
            best_entry = entry

    return best_path, best_entry


def infer_mesh_path(scene_dir: Path, mesh_file: str) -> Path:
    mesh_dir = scene_dir / "mesh"

    if mesh_file != "auto":
        path = Path(mesh_file)
        if path.exists():
            return path
        candidate = mesh_dir / mesh_file
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Mesh not found: {candidate}")

    for name in DEFAULT_MESH_CANDIDATES:
        candidate = mesh_dir / name
        if candidate.exists():
            return candidate

    for suffix in ("*.ply", "*.off"):
        matches = sorted(mesh_dir.glob(suffix))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No supported mesh file found under {mesh_dir}")


def infer_frame_mesh_path(mesh_path: Path, frame_index: int) -> Path:
    candidate = mesh_path.with_name(f"{mesh_path.stem}_frame_{frame_index:06d}{mesh_path.suffix}")
    return candidate if candidate.exists() else mesh_path


def resolve_scene_dir(test_output: Path, scene: str | None, mesh_path: str | None) -> Path:
    if mesh_path is not None:
        if scene is None:
            raise ValueError("--mesh_path requires --scene so the script can load the right cameras.")
        return test_output / scene

    if scene is None:
        raise ValueError("Scene must be provided when resolving a single scene directory.")

    direct = test_output / scene
    if direct.is_dir():
        return direct

    if test_output.name == scene and test_output.is_dir():
        return test_output

    raise FileNotFoundError(f"Could not find scene directory for {scene} under {test_output}")


def discover_scene_dirs(test_output: Path, mesh_file: str, mesh_path: str | None) -> list[tuple[str, Path]]:
    if mesh_path is not None:
        raise ValueError("discover_scene_dirs should not be used with --mesh_path")

    if (test_output / "mesh").is_dir():
        inferred = infer_mesh_path(test_output, mesh_file)
        return [(test_output.name, test_output)] if inferred.exists() else []

    scenes: list[tuple[str, Path]] = []
    for child in sorted(test_output.iterdir()):
        if not child.is_dir():
            continue
        try:
            infer_mesh_path(child, mesh_file)
        except FileNotFoundError:
            continue
        scenes.append((child.name, child))
    return scenes


def select_context_target_indices(
    scene: str,
    eval_index: dict[str, Any],
    context_override: list[int] | None,
    target_override: list[int] | None,
) -> tuple[list[int], list[int]]:
    entry = eval_index.get(scene)
    context_indices = list(entry["context"]) if entry is not None else []
    target_indices = list(entry["target"]) if entry is not None else []

    if context_override is not None:
        context_indices = list(context_override)
    if target_override is not None:
        target_indices = list(target_override)

    return context_indices, target_indices


def select_render_indices(
    mode: str,
    scene_dir: Path,
    example: dict[str, Any],
    context_indices: list[int],
    target_indices: list[int],
) -> list[int]:
    if mode == "eval":
        indices = sorted(set(context_indices + target_indices))
    elif mode == "target":
        indices = sorted(set(target_indices))
    elif mode == "saved_pred":
        indices = list_saved_prediction_indices(scene_dir)
    elif mode == "all":
        indices = list(range(len(example["images"])))
    else:
        raise ValueError(f"Unknown frame selection mode: {mode}")

    if not indices:
        raise ValueError(
            f"No frame indices resolved for mode={mode}. "
            "Provide --eval_index or --context_indices/--target_indices."
        )
    return indices


def build_ray_scene(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[
    o3d.t.geometry.RaycastingScene,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
]:
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64) if mesh.has_vertex_normals() else None
    return ray_scene, triangles, vertex_colors, triangle_normals, vertex_normals


def normalize_vectors_np(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


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
    shading: str = "none",
    shading_ambient: float = DEFAULT_SHADING_AMBIENT,
    shading_diffuse: float = DEFAULT_SHADING_DIFFUSE,
    hit_mode: str = "nearest",
    supersample: int = 1,
) -> tuple[np.ndarray, float]:
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}.")
    render_width = int(width) * int(supersample)
    render_height = int(height) * int(supersample)
    k_render = k_pixel.astype(np.float64, copy=True)
    if supersample > 1:
        k_render[0, :] *= float(supersample)
        k_render[1, :] *= float(supersample)

    intrinsic_t = o3d.core.Tensor(k_render)
    extrinsic_t = o3d.core.Tensor(w2c.astype(np.float64))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_t,
        extrinsic_t,
        render_width,
        render_height,
    )
    ray_dirs = rays.numpy()[..., 3:6]
    ray_dirs_flat = ray_dirs.reshape(-1, 3)

    num_render_pixels = render_height * render_width
    rendered_flat = np.zeros((num_render_pixels, 3), dtype=np.float64)

    if hit_mode == "nearest":
        result = ray_scene.cast_rays(rays)
        prim_ids_2d = result["primitive_ids"].numpy()
        prim_uvs_2d = result["primitive_uvs"].numpy()
        hit_mask_2d = prim_ids_2d != ray_scene.INVALID_ID
        hit_mask_flat = hit_mask_2d.reshape(-1)
        tri_ids = prim_ids_2d.reshape(-1)[hit_mask_flat]
        prim_uvs = prim_uvs_2d.reshape(-1, 2)[hit_mask_flat]
        hit_flat_indices = np.flatnonzero(hit_mask_flat)
    elif hit_mode in ("front_or_nearest", "front_only"):
        result = ray_scene.list_intersections(rays)
        primitive_ids_all = result["primitive_ids"].numpy().astype(np.int64, copy=False)
        primitive_uvs_all = result["primitive_uvs"].numpy()
        ray_ids_all = result["ray_ids"].numpy().astype(np.int64, copy=False)
        ray_splits = result["ray_splits"].numpy().astype(np.int64, copy=False)
        selected = np.full((num_render_pixels,), -1, dtype=np.int64)

        if primitive_ids_all.size:
            starts = ray_splits[:-1]
            ends = ray_splits[1:]
            rays_with_hits = np.flatnonzero(ends > starts)
            if hit_mode == "front_or_nearest" and rays_with_hits.size:
                selected[rays_with_hits] = starts[rays_with_hits]

            normals = triangle_normals[primitive_ids_all]
            ray_dirs_for_hits = ray_dirs_flat[ray_ids_all]
            front_hit = np.einsum("ij,ij->i", normals, ray_dirs_for_hits) < 0.0
            front_indices = np.flatnonzero(front_hit)
            if front_indices.size:
                front_ray_ids = ray_ids_all[front_indices]
                unique_front_ray_ids, first_front = np.unique(front_ray_ids, return_index=True)
                selected[unique_front_ray_ids] = front_indices[first_front]

        hit_mask_flat = selected >= 0
        hit_flat_indices = np.flatnonzero(hit_mask_flat)
        selected_hits = selected[hit_flat_indices]
        tri_ids = primitive_ids_all[selected_hits] if selected_hits.size else np.empty((0,), dtype=np.int64)
        prim_uvs = primitive_uvs_all[selected_hits] if selected_hits.size else np.empty((0, 2), dtype=np.float32)
    else:
        raise ValueError(f"Unknown hit_mode={hit_mode!r}. Expected one of {HIT_MODES}.")

    if vertex_colors is not None and hit_flat_indices.size:
        u = prim_uvs[:, 0]
        v = prim_uvs[:, 1]
        w0 = 1.0 - u - v

        vertex_indices = triangles[tri_ids]
        c0 = vertex_colors[vertex_indices[:, 0]]
        c1 = vertex_colors[vertex_indices[:, 1]]
        c2 = vertex_colors[vertex_indices[:, 2]]
        rendered_flat[hit_flat_indices] = w0[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2

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
            view_dirs = normalize_vectors_np(-ray_dirs_flat[hit_flat_indices].astype(np.float64))
            diffuse = np.clip(np.sum(shading_normals * view_dirs, axis=-1), 0.0, 1.0)
            shade = np.clip(shading_ambient + shading_diffuse * diffuse, 0.0, 1.0)
            rendered_flat[hit_flat_indices] *= shade[:, None]

    coverage = float(hit_mask_flat.mean())
    rendered = rendered_flat.reshape(render_height, render_width, 3)
    rendered_u8 = (rendered * 255.0).clip(0, 255).astype(np.uint8)
    if supersample > 1:
        rendered_u8 = np.asarray(
            Image.fromarray(rendered_u8).resize(
                (width, height),
                resample=Image.Resampling.LANCZOS,
            )
        )
    return rendered_u8, coverage


def main() -> None:
    args = parse_args()

    test_output = Path(args.test_output).expanduser()
    settings = resolve_runtime_settings(args, test_output)

    data_root: Path = settings["data_root"]
    stage_root = resolve_stage_root(data_root, args.split)
    eval_index_paths: list[Path] = settings["eval_index_paths"]
    eval_indices = load_eval_indices(eval_index_paths)
    metric_device: torch.device = settings["metric_device"]

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
    print(f"Metric device:    {metric_device}")
    print(
        f"Shading:          {args.shading}"
        if args.shading == "none"
        else f"Shading:          {args.shading} (ambient={args.shading_ambient:.2f}, diffuse={args.shading_diffuse:.2f})"
    )
    print(f"Hit mode:         {args.hit_mode}")
    print(f"Supersample:      {args.supersample}x")
    if args.render_tag:
        print(f"Render tag:       {args.render_tag}")
    print(f"Scenes:           {len(scenes)}")
    print()

    successful_scene_results: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, Any]] = []

    for scene, scene_dir in scenes:
        print(f"=== {scene} ===")
        try:
            mesh_path = Path(args.mesh_path) if args.mesh_path is not None else infer_mesh_path(scene_dir, args.mesh_file)
            mesh_path = mesh_path.resolve()
            print(f"  Mesh: {mesh_path}")

            example = load_scene_example(stage_root, scene)
            c2w_all, k_all = convert_poses(example["cameras"])

            selected_eval_path, selected_eval_entry = pick_eval_entry(
                scene,
                scene_dir,
                eval_indices,
            )

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
            saved_pred_indices = list_saved_prediction_indices(scene_dir)
            if saved_pred_indices and target_indices and saved_pred_indices != sorted(target_indices):
                print(
                    "  Note: saved prediction frames differ from eval target frames. "
                    "Use --frames saved_pred or pass an explicit --eval_index if needed."
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
            print(f"  Frames:      {len(render_indices)} ({args.frames})")

            render_stem = mesh_path.stem if not args.render_tag else f"{mesh_path.stem}_{args.render_tag}"
            render_root = (scene_dir / "mesh_render" / render_stem).resolve()
            render_root.mkdir(parents=True, exist_ok=True)
            gt_root = render_root / "gt"
            if args.save_gt:
                gt_root.mkdir(parents=True, exist_ok=True)

            if args.use_frame_meshes:
                frame_mesh_paths = {
                    frame_index: infer_frame_mesh_path(mesh_path, frame_index)
                    for frame_index in render_indices
                }
            else:
                frame_mesh_paths = {frame_index: mesh_path for frame_index in render_indices}
            use_frame_meshes = any(path != mesh_path for path in frame_mesh_paths.values())
            mesh_cache: dict[Path, tuple[Any, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]] = {}
            if use_frame_meshes:
                print("  Frame meshes: enabled")
            else:
                if not args.use_frame_meshes:
                    print("  Frame meshes: disabled")
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                print(f"  Mesh stats:  {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
                if not mesh.has_vertex_colors():
                    print("  Warning: mesh has no vertex colors; rendered RGB will be black.")

                print("  Building raycast scene ...", end="", flush=True)
                mesh_cache[mesh_path] = build_ray_scene(mesh)
                print(" done")

            rendered_images: list[np.ndarray] = []
            gt_images: list[np.ndarray] = []
            render_paths: list[Path] = []
            scene_coverages: list[float] = []

            for frame_index in render_indices:
                c2w = c2w_all[frame_index].numpy().astype(np.float64)
                k_norm = k_all[frame_index].numpy().astype(np.float64)
                k_adj = adjust_intrinsics_for_crop(k_norm, orig_h, orig_w, target_h, target_w)
                k_pixel = norm_to_pixel_intrinsics(k_adj, target_w, target_h)
                w2c = np.linalg.inv(c2w)

                active_mesh_path = frame_mesh_paths[frame_index]
                if active_mesh_path not in mesh_cache:
                    mesh = o3d.io.read_triangle_mesh(str(active_mesh_path))
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()
                    if not mesh.has_vertex_colors():
                        print(f"  Warning: mesh has no vertex colors: {active_mesh_path}")
                    mesh_cache[active_mesh_path] = build_ray_scene(mesh)
                ray_scene, triangles, vertex_colors, triangle_normals, vertex_normals = mesh_cache[active_mesh_path]
                rendered_u8, coverage = render_mesh_raycast(
                    ray_scene,
                    triangles,
                    vertex_colors,
                    triangle_normals,
                    vertex_normals,
                    k_pixel,
                    w2c,
                    target_w,
                    target_h,
                    shading=args.shading,
                    shading_ambient=args.shading_ambient,
                    shading_diffuse=args.shading_diffuse,
                    hit_mode=args.hit_mode,
                    supersample=args.supersample,
                )
                scene_coverages.append(coverage)

                if args.blur:
                    rendered_u8 = np.asarray(
                        Image.fromarray(rendered_u8).filter(
                            ImageFilter.GaussianBlur(radius=float(args.blur_sigma))
                        )
                    )

                render_path = render_root / f"{frame_index:06d}.png"
                Image.fromarray(rendered_u8).save(render_path)
                render_paths.append(render_path.resolve())
                rendered_images.append(rendered_u8)

                gt_image = load_gt_frame(
                    example,
                    frame_index,
                    target_h,
                    target_w,
                    apply_valid_mask=args.apply_valid_mask_gt,
                )
                gt_images.append(gt_image)
                if args.save_gt:
                    Image.fromarray(gt_image).save(gt_root / f"{frame_index:06d}.png")

            metric_values = compute_image_metrics(gt_images, rendered_images, metric_device)

            frame_rows: list[dict[str, Any]] = []
            for idx, frame_index in enumerate(render_indices):
                frame_row = {
                    "frame_index": frame_index,
                    "mesh_render_path": str(render_paths[idx]),
                    "psnr": float(metric_values["psnr"][idx]),
                    "ssim": float(metric_values["ssim"][idx]),
                    "lpips": float(metric_values["lpips"][idx]),
                    "coverage": float(scene_coverages[idx]),
                    "mesh_path": str(frame_mesh_paths[frame_index].resolve()),
                }
                frame_rows.append(frame_row)
                print(
                    f"    frame {frame_index:>4d} -> {render_paths[idx].name}"
                    f"  PSNR={frame_row['psnr']:.2f}"
                    f"  SSIM={frame_row['ssim']:.4f}"
                    f"  LPIPS={frame_row['lpips']:.4f}"
                    f"  coverage={frame_row['coverage']:.3f}"
                )

            scene_result = {
                "scene": scene,
                "mesh_path": str(mesh_path),
                "metrics_path": str((render_root / "metrics.json").resolve()),
                "pose_norm_method": pose_norm_method,
                "relative_pose": relative_pose,
                "use_frame_meshes": bool(args.use_frame_meshes),
                "shading": args.shading,
                "shading_ambient": float(args.shading_ambient),
                "shading_diffuse": float(args.shading_diffuse),
                "hit_mode": args.hit_mode,
                "render_tag": args.render_tag,
                "pose_scale": float(scale),
                "context_indices": context_indices,
                "target_indices": target_indices,
                "frames_mode": args.frames,
                "num_frames": len(frame_rows),
                "psnr_mean": mean_metric([row["psnr"] for row in frame_rows]),
                "ssim_mean": mean_metric([row["ssim"] for row in frame_rows]),
                "lpips_mean": mean_metric([row["lpips"] for row in frame_rows]),
                "coverage_mean": mean_metric([row["coverage"] for row in frame_rows]),
                "frames": frame_rows,
            }

            print(f"  Scene avg PSNR:   {scene_result['psnr_mean']:.2f}")
            print(f"  Scene avg SSIM:   {scene_result['ssim_mean']:.4f}")
            print(f"  Scene avg LPIPS:  {scene_result['lpips_mean']:.4f}")
            print(f"  Scene avg cover.: {scene_result['coverage_mean']:.3f}")

            if args.write_metrics_json:
                write_json(render_root / "metrics.json", scene_result)
                write_json(render_root / "summary.json", scene_result)

            successful_scene_results.append(scene_result)
        except Exception as exc:
            error_row = {
                "scene": scene,
                "scene_dir": str(scene_dir.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed_scenes.append(error_row)
            print(f"  Failed: {error_row['error']}")

        print()

    root_summary = build_root_summary(
        scene_results=successful_scene_results,
        failed_scenes=failed_scenes,
        metric_device=metric_device,
        stage_root=stage_root,
        test_output=test_output,
        eval_index_paths=eval_index_paths,
        pose_norm_method=pose_norm_method,
        relative_pose=relative_pose,
        image_shape=(target_h, target_w),
        orig_image_shape=(orig_h, orig_w),
        frames_mode=args.frames,
        shading=args.shading,
        shading_ambient=float(args.shading_ambient),
        shading_diffuse=float(args.shading_diffuse),
        hit_mode=args.hit_mode,
        supersample=int(args.supersample),
        render_tag=args.render_tag,
        hydra_defaults=settings["hydra_defaults"],
    )
    root_summary_path = test_output / args.summary_name
    write_json(root_summary_path, root_summary)

    if root_summary["overall_frame_mean"]:
        frame_mean = root_summary["overall_frame_mean"]
        print(
            "Overall frame mean: "
            f"PSNR={frame_mean['psnr']:.2f} "
            f"SSIM={frame_mean['ssim']:.4f} "
            f"LPIPS={frame_mean['lpips']:.4f}"
        )
    if root_summary["overall_scene_mean"]:
        scene_mean = root_summary["overall_scene_mean"]
        print(
            "Overall scene mean: "
            f"PSNR={scene_mean['psnr']:.2f} "
            f"SSIM={scene_mean['ssim']:.4f} "
            f"LPIPS={scene_mean['lpips']:.4f}"
        )
    print(f"Summary JSON:      {root_summary_path.resolve()}")


if __name__ == "__main__":
    main()
