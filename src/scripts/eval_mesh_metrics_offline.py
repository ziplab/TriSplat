import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.mesh.evaluate_mesh import eval_mesh

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exported meshes against GT_ROOT/<scene>.ply without "
            "rerunning model inference."
        ),
    )
    parser.add_argument(
        "--test-output",
        required=True,
        help="Run output root containing <scene>/mesh/*.ply.",
    )
    parser.add_argument(
        "--reference-root",
        dest="gt_root",
        default=None,
        help="Directory containing reference point clouds named <scene>.ply.",
    )
    parser.add_argument("--gt-root", dest="gt_root", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--index-path",
        action="append",
        default=[],
        help=(
            "Evaluation index JSON. Can be repeated. "
            "If omitted, scene dirs are scanned."
        ),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Explicit scene key to evaluate. Can be repeated.",
    )
    parser.add_argument(
        "--mesh-file",
        default="auto",
        help=(
            "Mesh filename under <scene>/mesh. Use 'auto' to prefer "
            "DIRECT_triangle_mesh_post.ply, then DIRECT_triangle_mesh.ply, "
            "then TSDF_mesh.ply."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Distance threshold for precision/recall/fscore.",
    )
    parser.add_argument(
        "--down-sample",
        type=float,
        default=0.02,
        help="Voxel downsample size passed to mesh evaluation.",
    )
    parser.add_argument(
        "--summary-prefix",
        default="mesh_metrics",
        help="Output summary prefix under test-output.",
    )
    parser.add_argument(
        "--dl3dv-colmap-cache-root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dl3dv-data-root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested scene is missing or evaluation errors.",
    )
    args = parser.parse_args()
    if args.gt_root is None:
        parser.error("--reference-root is required")
    return args


def read_scene_keys(
    index_paths: list[str],
    explicit_scenes: list[str],
    test_output: Path,
) -> list[str]:
    scenes: set[str] = set(scene for scene in explicit_scenes if scene)
    for index_path in index_paths:
        path = Path(index_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            scenes.update(str(key) for key in payload.keys())
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    scenes.add(item)
                elif isinstance(item, dict) and item.get("scene") is not None:
                    scenes.add(str(item["scene"]))
                else:
                    raise TypeError(f"Unsupported scene entry in {path}: {item!r}")
        else:
            raise TypeError(f"Expected dict or list index: {path}")

    if not scenes:
        scenes.update(path.name for path in test_output.iterdir() if path.is_dir())
    if not scenes:
        raise ValueError("No scenes found to evaluate.")
    return sorted(scenes)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [
                1.0 - 2.0 * y * y - 2.0 * z * z,
                2.0 * x * y - 2.0 * w * z,
                2.0 * z * x + 2.0 * w * y,
            ],
            [
                2.0 * x * y + 2.0 * w * z,
                1.0 - 2.0 * x * x - 2.0 * z * z,
                2.0 * y * z - 2.0 * w * x,
            ],
            [
                2.0 * z * x - 2.0 * w * y,
                2.0 * y * z + 2.0 * w * x,
                1.0 - 2.0 * x * x - 2.0 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def read_colmap_images_bin(path: Path) -> dict[str, np.ndarray]:
    image_centers: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id_and_pose = f.read(64)
            if len(image_id_and_pose) != 64:
                raise ValueError(f"Unexpected EOF while reading {path}")
            image_id, qw, qx, qy, qz, tx, ty, tz, _camera_id = struct.unpack(
                "<i7di",
                image_id_and_pose,
            )
            name_bytes = []
            while True:
                byte = f.read(1)
                if byte == b"":
                    raise ValueError(f"Unexpected EOF while reading image name in {path}")
                if byte == b"\x00":
                    break
                name_bytes.append(byte)
            name = b"".join(name_bytes).decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2d * 24, 1)

            rotation = qvec2rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))
            translation = np.array([tx, ty, tz], dtype=np.float64)
            center = -rotation.T @ translation
            image_centers[name] = center
            image_centers[Path(name).name] = center
            image_centers[str(image_id)] = center
    return image_centers


def find_colmap_images_bin(colmap_cache_root: Path, scene: str) -> Path:
    direct = colmap_cache_root / scene
    if not direct.exists():
        matches = [path for path in colmap_cache_root.rglob(scene) if path.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"Could not find DL3DV scene directory for {scene}."
            )
        matches.sort(key=lambda path: (len(path.parts), str(path)))
        direct = matches[0]

    preferred = (
        direct / "colmap" / "sparse" / "0" / "images.bin",
        direct / scene / "colmap" / "sparse" / "0" / "images.bin",
        direct / "sparse" / "0" / "images.bin",
    )
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(direct.rglob("images.bin"), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise FileNotFoundError(f"Could not find images.bin under {direct}")
    return candidates[0]


def resolve_dl3dv_stage_root(dl3dv_data_root: Path) -> Path:
    test_root = dl3dv_data_root / "test"
    return test_root if test_root.exists() else dl3dv_data_root


def load_dl3dv_example(stage_root: Path, scene: str) -> dict[str, Any]:
    index_path = stage_root / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if scene not in index:
        raise KeyError(f"Scene {scene} is not present in {index_path}")

    chunk_path = stage_root / index[scene]
    chunk = torch.load(chunk_path, map_location="cpu")
    matches = [example for example in chunk if example.get("key") == scene]
    if len(matches) != 1:
        raise ValueError(f"Expected one example for {scene} in {chunk_path}, got {len(matches)}")
    return matches[0]


def dataset_cameras_to_centers(example: dict[str, Any]) -> list[tuple[list[str], np.ndarray]]:
    centers: list[tuple[list[str], np.ndarray]] = []
    cameras = example["cameras"]
    timestamps = example["timestamps"]
    for camera, timestamp in zip(cameras, timestamps):
        camera_np = camera.detach().cpu().numpy()
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = camera_np[6:].reshape(3, 4)
        camera_to_world = np.linalg.inv(world_to_camera)
        frame_id = int(timestamp.item())
        names = [
            f"images/frame_{frame_id:05d}.png",
            f"frame_{frame_id:05d}.png",
            str(frame_id),
        ]
        centers.append((names, camera_to_world[:3, 3]))
    return centers


def umeyama_similarity(
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
    variance = float((source_centered**2).sum() / len(source_points))
    scale = 1.0 if variance <= 1e-12 else float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def estimate_colmap_gt_to_dataset_transform(
    colmap_cache_root: Path,
    dl3dv_stage_root: Path,
    scene: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    images_bin = find_colmap_images_bin(colmap_cache_root, scene)
    colmap_centers_by_name = read_colmap_images_bin(images_bin)
    example = load_dl3dv_example(dl3dv_stage_root, scene)

    colmap_centers = []
    dataset_centers = []
    for names, dataset_center in dataset_cameras_to_centers(example):
        colmap_center = next(
            (colmap_centers_by_name[name] for name in names if name in colmap_centers_by_name),
            None,
        )
        if colmap_center is None:
            continue
        colmap_centers.append(colmap_center)
        dataset_centers.append(dataset_center)
    if len(colmap_centers) < 3:
        raise ValueError(
            f"Need at least 3 matched cameras for {scene}, got {len(colmap_centers)}."
        )

    colmap_centers_np = np.asarray(colmap_centers, dtype=np.float64)
    dataset_centers_np = np.asarray(dataset_centers, dtype=np.float64)
    scale, rotation, translation = umeyama_similarity(
        colmap_centers_np,
        dataset_centers_np,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation

    aligned_centers = colmap_centers_np @ matrix[:3, :3].T + matrix[:3, 3]
    errors = np.linalg.norm(aligned_centers - dataset_centers_np, axis=1)
    metadata = {
        "gt_transform_source": "dl3dv_colmap_cache_camera_similarity",
        "gt_transform_matrix": matrix.tolist(),
        "gt_transform_scale": float(scale),
        "gt_transform_rotation": rotation.tolist(),
        "gt_transform_translation": translation.tolist(),
        "gt_transform_num_cameras": int(len(colmap_centers_np)),
        "gt_transform_camera_error_mean": float(errors.mean()),
        "gt_transform_camera_error_max": float(errors.max()),
        "gt_transform_images_path": str(images_bin),
    }
    return matrix, metadata


def build_gt_transform_provider(
    colmap_cache_root: Path | None,
    dl3dv_data_root: Path | None,
) -> Any:
    if colmap_cache_root is None and dl3dv_data_root is None:
        return None
    if colmap_cache_root is None or dl3dv_data_root is None:
        raise ValueError(
            "--dl3dv-colmap-cache-root and --dl3dv-data-root must be provided together."
        )

    cache_root = colmap_cache_root.expanduser().resolve()
    stage_root = resolve_dl3dv_stage_root(dl3dv_data_root.expanduser().resolve())
    if not cache_root.exists():
        raise FileNotFoundError(f"DL3DV auxiliary data root not found: {cache_root}")
    if not (stage_root / "index.json").exists():
        raise FileNotFoundError(f"DL3DV index not found: {stage_root / 'index.json'}")

    transform_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    def provider(scene: str) -> tuple[np.ndarray, dict[str, Any]]:
        if scene not in transform_cache:
            transform_cache[scene] = estimate_colmap_gt_to_dataset_transform(
                cache_root,
                stage_root,
                scene,
            )
        return transform_cache[scene]

    return provider


def resolve_mesh_path(scene_dir: Path, mesh_file: str) -> Path | None:
    mesh_dir = scene_dir / "mesh"
    if mesh_file != "auto":
        path = mesh_dir / mesh_file
        return path if path.exists() else None

    candidates = (
        mesh_dir / "DIRECT_triangle_mesh_post.ply",
        mesh_dir / "DIRECT_triangle_mesh.ply",
        mesh_dir / "TSDF_mesh.ply",
        mesh_dir / "mesh.ply",
    )
    for path in candidates:
        if path.exists():
            return path
    ply_candidates = sorted(mesh_dir.glob("*.ply"))
    return ply_candidates[0] if ply_candidates else None


def attach_result_base(
    result: dict[str, Any],
    scene: str,
    pred_mesh_path: Path,
    gt_root: Path,
) -> dict[str, Any]:
    return {
        "scene": scene,
        "pred_mesh_path": str(pred_mesh_path),
        "gt_pointcloud_path": str(gt_root / f"{scene}.ply"),
        **result,
    }


PUBLIC_FIELD_NAMES = (
    "scene",
    "pred_mesh_path",
    "mesh_space",
    "gt_points_before",
    "gt_points_after_bbox",
    "gt_points_after_density",
    "gt_points_after_component",
    "gt_points_after",
    "pred_points_before",
    "pred_points_after_bbox",
    "pred_points_after",
    "gt_filter_fallback_reason",
    "pred_filter_fallback_reason",
    *METRIC_NAMES,
)


def public_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in PUBLIC_FIELD_NAMES if field in row}


def public_failure_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {
        field: row.get(field)
        for field in ("scene", "pred_mesh_path", "error")
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


def evaluate_scene(
    test_output: Path,
    gt_root: Path,
    scene: str,
    mesh_file: str,
    threshold: float,
    down_sample: float,
    gt_transform_provider: Any | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    scene_dir = test_output / scene
    pred_mesh_path = resolve_mesh_path(scene_dir, mesh_file)
    gt_path = gt_root / f"{scene}.ply"
    if pred_mesh_path is None:
        return None, None, None, {
            "scene": scene,
            "error": f"Missing mesh under {scene_dir / 'mesh'}",
        }
    if not gt_path.exists():
        return None, None, None, {
            "scene": scene,
            "pred_mesh_path": str(pred_mesh_path),
            "error": f"Missing reference point cloud: {gt_path}",
        }

    try:
        gt_transform = None
        gt_transform_metadata = None
        if gt_transform_provider is not None:
            gt_transform, gt_transform_metadata = gt_transform_provider(scene)

        metrics = eval_mesh(
            pred_mesh_path,
            gt_root,
            scene,
            gt_transform=gt_transform,
            gt_transform_metadata=gt_transform_metadata,
            threshold=threshold,
            down_sample=down_sample,
        )
    except Exception as exc:
        return None, None, None, {
            "scene": scene,
            "pred_mesh_path": str(pred_mesh_path),
            "error": str(exc),
        }

    return (
        attach_result_base(metrics["raw_world"], scene, pred_mesh_path, gt_root),
        attach_result_base(
            metrics["aligned_similarity"],
            scene,
            pred_mesh_path,
            gt_root,
        ),
        attach_result_base(metrics["gt_trimmed"], scene, pred_mesh_path, gt_root),
        None,
    )


def mean_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    overall = {
        f"{metric}_mean": float(np.mean([float(item[metric]) for item in results]))
        for metric in METRIC_NAMES
    }
    for metric in COUNT_METRIC_NAMES:
        values = [
            float(item[metric]) for item in results if item.get(metric) is not None
        ]
        if values:
            overall[f"{metric}_mean"] = float(np.mean(values))
    return overall


def write_summary(
    output_root: Path,
    summary_prefix: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    threshold: float,
    down_sample: float,
) -> None:
    sorted_results = sorted(
        (public_result_row(row) for row in results),
        key=lambda item: str(item["scene"]),
    )
    first_result = sorted_results[0] if sorted_results else {}
    summary = {
        "num_scenes": len(sorted_results),
        "num_failed": len(failures),
        "mesh_space": first_result.get("mesh_space"),
        "threshold": threshold,
        "down_sample": down_sample,
        "overall": mean_metrics(sorted_results) if sorted_results else {},
        "scenes": sorted_results,
        "failures": [public_failure_row(failure) for failure in failures],
    }

    stem = f"{summary_prefix}_summary"
    json_path = output_root / f"{stem}.json"
    csv_path = output_root / f"{stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    field_names = list(PUBLIC_FIELD_NAMES)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in sorted_results:
            writer.writerow({field: row.get(field) for field in field_names})
        if sorted_results:
            overall = summary["overall"]
            writer.writerow(
                {
                    "scene": "__mean__",
                    "mesh_space": first_result.get("mesh_space"),
                    **{
                        metric: overall.get(f"{metric}_mean")
                        for metric in (*COUNT_METRIC_NAMES, *METRIC_NAMES)
                    },
                }
            )


def main() -> None:
    args = parse_args()
    test_output = Path(args.test_output).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    if not test_output.exists():
        raise FileNotFoundError(f"Test output root not found: {test_output}")
    if not gt_root.exists():
        raise FileNotFoundError(f"Reference point cloud root not found: {gt_root}")
    gt_transform_provider = build_gt_transform_provider(
        Path(args.dl3dv_colmap_cache_root)
        if args.dl3dv_colmap_cache_root is not None
        else None,
        Path(args.dl3dv_data_root)
        if args.dl3dv_data_root is not None
        else None,
    )

    scenes = read_scene_keys(args.index_path, args.scene, test_output)
    metric_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for scene in tqdm(scenes, desc="Evaluating meshes"):
        raw, aligned, gt_trimmed, failure = evaluate_scene(
            test_output=test_output,
            gt_root=gt_root,
            scene=scene,
            mesh_file=args.mesh_file,
            threshold=args.threshold,
            down_sample=args.down_sample,
            gt_transform_provider=gt_transform_provider,
        )
        if failure is not None:
            failures.append(failure)
            continue
        metric_results.append(gt_trimmed)

    write_summary(
        test_output,
        args.summary_prefix,
        metric_results,
        failures,
        args.threshold,
        args.down_sample,
    )

    payload = {
        "test_output": str(test_output),
        "num_scenes_requested": len(scenes),
        "num_scenes_success": len(metric_results),
        "num_scenes_failed": len(failures),
        "summary": str(test_output / f"{args.summary_prefix}_summary.json"),
    }
    print(json.dumps(payload, indent=2))

    if args.strict and failures:
        raise SystemExit(f"Failed to evaluate {len(failures)} scene(s).")


if __name__ == "__main__":
    main()
