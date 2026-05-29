import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs.
    cKDTree = None

from src.evaluation.mesh import evaluate_mesh as mesh_eval
from src.scripts.eval_mesh_metrics_offline import (
    estimate_colmap_gt_to_dataset_transform,
    resolve_dl3dv_stage_root,
)


DEFAULT_PRED_ROOT = Path(os.environ.get("ANYSPLAT_PRED_ROOT", "outputs/anysplat_dl3dv_tsdf_mesh"))
DEFAULT_ALIGNMENT_JSON = (
    DEFAULT_PRED_ROOT
    / "geometry_metrics_camera_colmap_normalized_context"
    / "combined_geometry_summary.json"
)
DEFAULT_RENDER_CSV = DEFAULT_PRED_ROOT / "combined_summary.csv"
DEFAULT_GT_ROOT = Path(os.environ.get("DL3DV_COLMAP_GT_ROOT", "data/dl3dv_colmap_gt_ply"))
DEFAULT_COLMAP_CACHE_ROOT = Path(os.environ.get("DL3DV_COLMAP_CACHE_ROOT", "data/dl3dv_colmap_cache"))
DEFAULT_DATA_ROOT = Path(os.environ.get("DL3DV_ROOT", "data/dl3dv"))
DEFAULT_OUTPUT_DIR = Path("outputs/anysplat_dl3dv_tsdf_mesh_current_protocol_geometry")

METRIC_NAMES = (
    "dist1",
    "dist2",
    "pred_to_gt",
    "gt_to_pred",
    "cd",
    "prec",
    "recal",
    "fscore",
)

COUNT_METRIC_NAMES = (
    "gt_points_before",
    "gt_points_after_bbox",
    "gt_points_after_density",
    "gt_points_after_component",
    "gt_points_after",
    "pred_points_before",
    "pred_points_after_bbox",
    "pred_points_after",
)

PUBLIC_FIELD_NAMES = (
    "ctx",
    "scene",
    "pred_mesh_path",
    "mesh_space",
    "num_context_views",
    "gt_filter_fallback_reason",
    "pred_filter_fallback_reason",
    *COUNT_METRIC_NAMES,
    *METRIC_NAMES,
)


def public_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in PUBLIC_FIELD_NAMES if field in row}


def public_failure_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {
        field: row.get(field)
        for field in ("ctx", "scene", "pred_mesh_path", "error")
        if field in row
    }
    if public.get("error") is not None:
        public["error"] = (
            str(public["error"])
            .replace("COLMAP", "auxiliary")
            .replace("colmap", "auxiliary")
            .replace("GT point cloud", "reference point cloud")
            .replace("GT root", "reference root")
            .replace("DL3DV GT", "DL3DV reference")
        )
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate AnySplat DL3DV meshes with Trisplat metrics."
        )
    )
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument(
        "--alignment-json",
        type=Path,
        default=DEFAULT_ALIGNMENT_JSON,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--render-csv", type=Path, default=DEFAULT_RENDER_CSV)
    parser.add_argument(
        "--reference-root",
        dest="gt_root",
        type=Path,
        default=DEFAULT_GT_ROOT,
    )
    parser.add_argument(
        "--gt-root",
        dest="gt_root",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dl3dv-colmap-cache-root",
        type=Path,
        default=DEFAULT_COLMAP_CACHE_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dl3dv-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contexts", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--mesh-file", default="TSDF_2dgs_post_mesh.ply")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--down-sample", type=float, default=0.02)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Dl3dvPackedDataset:
    def __init__(self, data_root: Path) -> None:
        self.stage_root = resolve_dl3dv_stage_root(data_root.expanduser().resolve())
        index_path = self.stage_root / "index.json"
        with index_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        self._chunk_cache: dict[str, dict[str, Any]] = {}

    def get_scene(self, scene: str) -> dict[str, Any]:
        chunk_name = self.index[scene]
        if chunk_name not in self._chunk_cache:
            chunk = torch_load(self.stage_root / chunk_name)
            self._chunk_cache[chunk_name] = {
                str(example["key"]): example for example in chunk
            }
        return self._chunk_cache[chunk_name][scene]


def camera_centers_from_packed(example: dict[str, Any], frame_indices: list[int]) -> np.ndarray:
    cameras = example["cameras"].detach().cpu().to(torch.float32)
    centers = []
    for index in frame_indices:
        camera = cameras[index].numpy()
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = camera[6:].reshape(3, 4)
        camera_to_world = np.linalg.inv(world_to_camera)
        centers.append(camera_to_world[:3, 3])
    return np.asarray(centers, dtype=np.float64)


def umeyama_similarity(source_points: np.ndarray, target_points: np.ndarray) -> dict[str, Any]:
    if source_points.shape != target_points.shape or len(source_points) < 3:
        raise ValueError("Need at least three ordered camera pairs for Sim(3).")

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
    variance = float((source_centered**2).sum() / len(source_points))
    scale = 1.0 if variance <= 1e-12 else float(
        np.trace(np.diag(singular_values) @ correction) / variance
    )
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    aligned = source_points @ matrix[:3, :3].T + matrix[:3, 3][None]
    errors = np.linalg.norm(aligned - target_points, axis=-1)
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "matrix": matrix,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "num_correspondences": int(len(source_points)),
    }


def nearest_neighbor_distances(reference_points: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    if len(reference_points) == 0 or len(query_points) == 0:
        return np.empty((0,), dtype=np.float64)
    if cKDTree is None:
        _, distances = mesh_eval.nn_correspondance(reference_points, query_points)
        return np.asarray(distances, dtype=np.float64)
    distances, _ = cKDTree(reference_points).query(query_points, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def compute_metrics_fast(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        raise ValueError("Mesh evaluation requires non-empty point sets.")
    pred_to_gt = nearest_neighbor_distances(gt_points, pred_points)
    gt_to_pred = nearest_neighbor_distances(pred_points, gt_points)
    precision = float(np.mean((pred_to_gt < threshold).astype(np.float64)))
    recall = float(np.mean((gt_to_pred < threshold).astype(np.float64)))
    fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    pred_to_gt_mean = float(np.mean(pred_to_gt))
    gt_to_pred_mean = float(np.mean(gt_to_pred))
    return {
        "dist1": pred_to_gt_mean,
        "dist2": gt_to_pred_mean,
        "pred_to_gt": pred_to_gt_mean,
        "gt_to_pred": gt_to_pred_mean,
        "cd": pred_to_gt_mean + gt_to_pred_mean,
        "prec": precision,
        "recal": recall,
        "fscore": float(fscore),
    }


def filter_pred_by_gt_support_fast(
    pred_points: np.ndarray,
    gt_core: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_points_before = int(len(pred_points))
    if pred_points_before == 0:
        raise ValueError("Prediction filtering requires non-empty predicted points.")
    if len(gt_core) == 0:
        raise ValueError("Prediction filtering requires non-empty filtered GT points.")

    gt_diag = mesh_eval._compute_bbox_diag(gt_core)
    bbox_margin = mesh_eval.GT_FILTER_BBOX_MARGIN_RATIO * gt_diag
    bbox_min, bbox_max = mesh_eval._compute_bbox(gt_core)
    pred_after_bbox = mesh_eval._filter_points_inside_bbox(
        pred_points,
        bbox_min - bbox_margin,
        bbox_max + bbox_margin,
    )
    pred_filter_fallback_reason = ""
    if len(pred_after_bbox) == 0:
        pred_after_bbox = pred_points
        pred_filter_fallback_reason = "bbox_empty"

    pred_keep_radius_min = mesh_eval.PRED_FILTER_KEEP_RADIUS_MIN_FACTOR * float(threshold)
    pred_keep_radius = max(
        pred_keep_radius_min,
        mesh_eval.PRED_FILTER_KEEP_RADIUS_RATIO * gt_diag,
    )
    pred_to_gt_after_bbox = nearest_neighbor_distances(gt_core, pred_after_bbox)
    pred_keep_mask = pred_to_gt_after_bbox < pred_keep_radius
    pred_gt_support = pred_after_bbox[pred_keep_mask]
    if len(pred_gt_support) == 0:
        pred_gt_support = pred_after_bbox
        pred_filter_fallback_reason = (
            pred_filter_fallback_reason + "|neighbor_empty"
            if pred_filter_fallback_reason
            else "neighbor_empty"
        )

    return pred_gt_support, {
        "pred_keep_radius_ratio": mesh_eval.PRED_FILTER_KEEP_RADIUS_RATIO,
        "pred_keep_radius_min_factor": mesh_eval.PRED_FILTER_KEEP_RADIUS_MIN_FACTOR,
        "pred_keep_radius_min": float(pred_keep_radius_min),
        "pred_keep_radius": float(pred_keep_radius),
        "pred_points_before": pred_points_before,
        "pred_points_after_bbox": int(len(pred_after_bbox)),
        "pred_points_after": int(len(pred_gt_support)),
        "pred_filter_fallback_reason": pred_filter_fallback_reason,
    }


def compute_gt_trimmed_metrics_fast(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    gt_core, gt_metadata = mesh_eval._filter_gt_core(gt_points, threshold=threshold)
    pred_gt_support, pred_metadata = filter_pred_by_gt_support_fast(
        pred_points,
        gt_core,
        threshold=threshold,
    )
    return {
        **gt_metadata,
        **pred_metadata,
        **compute_metrics_fast(pred_gt_support, gt_core, threshold=threshold),
    }


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3][None]


def resolve_mesh_path(scene_dir: Path, mesh_file: str) -> Path | None:
    mesh_dir = scene_dir / "mesh"
    if mesh_file != "auto":
        path = mesh_dir / mesh_file
        return path if path.exists() else None
    for name in (
        "TSDF_2dgs_post_mesh.ply",
        "TSDF_2dgs_mesh.ply",
        "TSDF_mesh.ply",
        "mesh.ply",
    ):
        path = mesh_dir / name
        if path.exists():
            return path
    candidates = sorted(mesh_dir.glob("*.ply"))
    return candidates[0] if candidates else None


def discover_scenes(ctx_root: Path, mesh_file: str) -> list[str]:
    if not ctx_root.exists():
        return []
    return [
        path.name
        for path in sorted(ctx_root.iterdir())
        if path.is_dir() and resolve_mesh_path(path, mesh_file) is not None
    ]


def load_alignment_map(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    alignments = payload.get("camera_alignments", [])
    alignment_map: dict[tuple[int, str], dict[str, Any]] = {}
    for item in alignments:
        alignment_map[(int(item["ctx"]), str(item["scene"]))] = item
    return alignment_map


def load_render_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            int(row["num_context_views"]): row
            for row in csv.DictReader(f)
            if row.get("num_context_views")
        }


def mean_fields(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ]
        means[f"{field}_mean"] = float(np.mean(values)) if values else None
    return means


def write_scene_summary(
    output_dir: Path,
    ctx: int,
    suffix: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    threshold: float,
    down_sample: float,
) -> dict[str, Any]:
    rows = sorted(
        (public_result_row(row) for row in rows),
        key=lambda item: str(item["scene"]),
    )
    summary = {
        "ctx": ctx,
        "num_scenes": len(rows),
        "num_failed": len(failures),
        "threshold": threshold,
        "down_sample": down_sample,
        "overall": mean_fields(rows, (*METRIC_NAMES, *COUNT_METRIC_NAMES)),
        "scenes": rows,
        "failures": [public_failure_row(failure) for failure in failures],
    }
    stem = f"ctx{ctx}_mesh_metrics_current_protocol_{suffix}_summary"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = list(PUBLIC_FIELD_NAMES)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        if rows:
            writer.writerow(
                {
                    "ctx": ctx,
                    "scene": "__mean__",
                    "mesh_space": rows[0].get("mesh_space"),
                    **{
                        field: summary["overall"].get(f"{field}_mean")
                        for field in (*COUNT_METRIC_NAMES, *METRIC_NAMES)
                    },
                }
            )
    summary["json_path"] = str(json_path)
    summary["csv_path"] = str(csv_path)
    return summary


def as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def metric_row(overall: dict[str, Any]) -> dict[str, Any]:
    return {
        "cd": overall.get("cd_mean"),
        "prec": overall.get("prec_mean"),
        "rec": overall.get("recal_mean"),
        "f1": overall.get("fscore_mean"),
    }


def evaluate_scene(
    *,
    ctx: int,
    scene: str,
    pred_root: Path,
    gt_root: Path,
    mesh_file: str,
    dataset: Dl3dvPackedDataset,
    alignment_map: dict[tuple[int, str], dict[str, Any]],
    gt_transform_cache: dict[str, tuple[np.ndarray, dict[str, Any]]],
    colmap_cache_root: Path,
    threshold: float,
    down_sample: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    scene_dir = pred_root / f"ctx{ctx}" / scene
    pred_mesh_path = resolve_mesh_path(scene_dir, mesh_file)
    gt_path = gt_root / f"{scene}.ply"
    if pred_mesh_path is None:
        return None, None, None, {"ctx": ctx, "scene": scene, "error": f"Missing mesh under {scene_dir / 'mesh'}"}
    if not gt_path.exists():
        return None, None, None, {
            "ctx": ctx,
            "scene": scene,
            "error": f"Missing reference point cloud: {gt_path}",
        }

    try:
        camera_alignment = alignment_map[(ctx, scene)]
        frame_indices = [int(index) for index in camera_alignment["frame_indices"]]
        pred_camera_centers = np.asarray(camera_alignment["pred_camera_centers"], dtype=np.float64)
        dataset_camera_centers = camera_centers_from_packed(
            dataset.get_scene(scene),
            frame_indices,
        )
        pred_alignment = umeyama_similarity(pred_camera_centers, dataset_camera_centers)
        pred_transform = np.asarray(pred_alignment["matrix"], dtype=np.float64)

        if scene not in gt_transform_cache:
            gt_transform_cache[scene] = estimate_colmap_gt_to_dataset_transform(
                colmap_cache_root,
                dataset.stage_root,
                scene,
            )
        gt_transform, gt_metadata = gt_transform_cache[scene]

        mesh_space_path, mesh_space_metadata = mesh_eval._load_mesh_space_metadata(pred_mesh_path)
        pred_points = mesh_eval._load_points(
            pred_mesh_path,
            transform=pred_transform,
            down_sample=down_sample,
        )
        gt_points = mesh_eval._load_points(
            gt_path,
            transform=gt_transform,
            down_sample=down_sample,
        )
        raw_metrics = compute_metrics_fast(pred_points, gt_points, threshold=threshold)
        similarity_alignment = mesh_eval._estimate_similarity_alignment(pred_points, gt_points)
        similarity_points = mesh_eval._apply_similarity_transform(pred_points, similarity_alignment)
        similarity_metrics = compute_metrics_fast(
            similarity_points,
            gt_points,
            threshold=threshold,
        )
        gt_trimmed_metrics = compute_gt_trimmed_metrics_fast(
            pred_points,
            gt_points,
            threshold=threshold,
        )
    except Exception as exc:
        return None, None, None, {
            "ctx": ctx,
            "scene": scene,
            "pred_mesh_path": str(pred_mesh_path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    base = {
        "ctx": ctx,
        "scene": scene,
        "pred_mesh_path": str(pred_mesh_path),
        "gt_pointcloud_path": str(gt_path),
        "mesh_space_path": None if mesh_space_path is None else str(mesh_space_path),
        "mesh_space": mesh_space_metadata.get("mesh_space", "unknown"),
        "world_space": "dataset_raw_world",
        "context_indices": mesh_space_metadata.get("context_indices"),
        "target_indices": mesh_space_metadata.get("target_indices"),
        "num_context_views": mesh_space_metadata.get("num_context_views"),
        "pred_transform_source": "anysplat_predicted_camera_sim3_to_packed_raw_world",
        "pred_transform_applied": True,
        "pred_transform_scale": float(pred_alignment["scale"]),
        "pred_transform_num_cameras": int(pred_alignment["num_correspondences"]),
        "pred_transform_camera_error_mean": float(pred_alignment["mean_error"]),
        "pred_transform_camera_error_max": float(pred_alignment["max_error"]),
        "gt_transform_applied": True,
        **gt_metadata,
    }
    raw = {
        **base,
        "evaluation_space": "dataset_raw_world_with_colmap_gt_aligned",
        **raw_metrics,
    }
    similarity = {
        **base,
        "evaluation_space": "aligned_similarity",
        "similarity_alignment_scale": similarity_alignment["scale"],
        "similarity_alignment_rotation": similarity_alignment["rotation"],
        "similarity_alignment_translation": similarity_alignment["translation"],
        "similarity_alignment_matrix": similarity_alignment["matrix"],
        **similarity_metrics,
    }
    gt_trimmed = {
        **base,
        "evaluation_space": "gt_trimmed",
        **gt_trimmed_metrics,
    }
    return raw, similarity, gt_trimmed, None


def main() -> None:
    args = parse_args()
    pred_root = args.pred_root.expanduser().resolve()
    alignment_json = args.alignment_json.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    colmap_cache_root = args.dl3dv_colmap_cache_root.expanduser().resolve()
    data_root = args.dl3dv_data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, name in (
        (pred_root, "AnySplat prediction root"),
        (alignment_json, "AnySplat auxiliary JSON"),
        (gt_root, "reference point cloud root"),
        (colmap_cache_root, "DL3DV auxiliary data root"),
        (data_root, "DL3DV packed data root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    dataset = Dl3dvPackedDataset(data_root)
    alignment_map = load_alignment_map(alignment_json)
    render_rows = load_render_rows(args.render_csv.expanduser().resolve())
    gt_transform_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    all_failures: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_payloads: list[dict[str, Any]] = []

    for ctx in args.contexts:
        scenes = discover_scenes(pred_root / f"ctx{ctx}", args.mesh_file)
        metric_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for scene in tqdm(scenes, desc=f"ctx{ctx}", unit="scene"):
            raw, similarity, gt_trimmed, failure = evaluate_scene(
                ctx=ctx,
                scene=scene,
                pred_root=pred_root,
                gt_root=gt_root,
                mesh_file=args.mesh_file,
                dataset=dataset,
                alignment_map=alignment_map,
                gt_transform_cache=gt_transform_cache,
                colmap_cache_root=colmap_cache_root,
                threshold=args.threshold,
                down_sample=args.down_sample,
            )
            if failure is not None:
                failures.append(failure)
                continue
            metric_rows.append(gt_trimmed)

        all_failures.extend(failures)
        metrics_summary = write_scene_summary(
            output_dir,
            ctx,
            "metrics",
            metric_rows,
            failures,
            args.threshold,
            args.down_sample,
        )
        summary_payloads.append(metrics_summary)

        render = render_rows.get(ctx, {})
        primary = metric_row(metrics_summary["overall"])
        final_rows.append(
            {
                "views": ctx,
                **primary,
                "psnr": as_float(render.get("mesh_psnr")),
                "ssim": as_float(render.get("mesh_ssim")),
                "lpips": as_float(render.get("mesh_lpips")),
                "num_scenes": metrics_summary["num_scenes"],
                "num_failed": metrics_summary["num_failed"],
                "mesh_space": metric_rows[0].get("mesh_space") if metric_rows else None,
                "threshold": args.threshold,
                "down_sample": args.down_sample,
                "summary_path": metrics_summary["json_path"],
            }
        )

    combined = {
        "pred_root": str(pred_root),
        "render_csv": str(args.render_csv.expanduser().resolve()),
        "dl3dv_data_root": str(data_root),
        "threshold": args.threshold,
        "down_sample": args.down_sample,
        "num_failed": len(all_failures),
        "failures": [public_failure_row(failure) for failure in all_failures],
        "summaries": summary_payloads,
        "rows": final_rows,
    }
    combined_json = output_dir / "combined_current_protocol_summary.json"
    combined_csv = output_dir / "combined_current_protocol_summary.csv"
    combined_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    fieldnames = [
        "views",
        "cd",
        "prec",
        "rec",
        "f1",
        "psnr",
        "ssim",
        "lpips",
        "num_scenes",
        "num_failed",
        "mesh_space",
        "threshold",
        "down_sample",
        "summary_path",
    ]
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(final_rows, key=lambda item: int(item["views"])):
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(json.dumps({"combined_json": str(combined_json), "combined_csv": str(combined_csv), "num_failed": len(all_failures)}, indent=2))
    if args.strict and all_failures:
        raise SystemExit(f"Failed to evaluate {len(all_failures)} scene(s).")


if __name__ == "__main__":
    main()
