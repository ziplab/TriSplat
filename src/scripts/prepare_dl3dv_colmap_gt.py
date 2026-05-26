import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from tqdm import tqdm


@dataclass(frozen=True)
class ColmapPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    track_length: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DL3DV COLMAP cache points3D files into "
            "GT_ROOT/<scene>.ply point clouds for mesh metric evaluation."
        ),
    )
    parser.add_argument(
        "--colmap-cache-root",
        required=True,
        help="Root containing downloaded DL3DV COLMAP cache scene folders.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where <scene>.ply files will be written.",
    )
    parser.add_argument(
        "--index-path",
        action="append",
        default=[],
        help=(
            "Evaluation index JSON. Can be repeated. Scene keys from all indexes "
            "are converted."
        ),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Explicit scene hash to convert. Can be repeated.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=0,
        help="Drop COLMAP points observed in fewer than this many images.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Drop COLMAP points with reprojection error greater than this value.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output PLY files.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Name of the conversion manifest written under output-root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested scene cannot be converted.",
    )
    return parser.parse_args()


def read_scene_keys(index_paths: Iterable[str], scenes: Iterable[str]) -> list[str]:
    keys: set[str] = set(scene for scene in scenes if scene)
    for index_path in index_paths:
        path = Path(index_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            keys.update(str(key) for key in payload.keys())
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    keys.add(item)
                elif isinstance(item, dict) and item.get("scene") is not None:
                    keys.add(str(item["scene"]))
                else:
                    raise TypeError(f"Unsupported scene entry in {path}: {item!r}")
        else:
            raise TypeError(f"Expected dict or list index: {path}")
    if not keys:
        raise ValueError("No scenes requested. Pass --index-path and/or --scene.")
    return sorted(keys)


def find_scene_dir(cache_root: Path, scene: str) -> Path | None:
    direct = cache_root / scene
    if direct.exists():
        return direct

    matches = [path for path in cache_root.rglob(scene) if path.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def find_points3d_path(scene_dir: Path) -> Path | None:
    preferred = (
        scene_dir / "sparse" / "0" / "points3D.bin",
        scene_dir / "sparse" / "0" / "points3D.txt",
        scene_dir / "colmap" / "sparse" / "0" / "points3D.bin",
        scene_dir / "colmap" / "sparse" / "0" / "points3D.txt",
        scene_dir / "points3D.bin",
        scene_dir / "points3D.txt",
    )
    for path in preferred:
        if path.exists():
            return path

    candidates = list(scene_dir.rglob("points3D.bin"))
    if not candidates:
        candidates = list(scene_dir.rglob("points3D.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def read_points3d_bin(path: Path) -> ColmapPointCloud:
    xyz_rows: list[tuple[float, float, float]] = []
    rgb_rows: list[tuple[int, int, int]] = []
    errors: list[float] = []
    track_lengths: list[int] = []

    with path.open("rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_header = f.read(43)
            if len(point_header) != 43:
                raise ValueError(f"Unexpected EOF while reading {path}")
            unpacked = struct.unpack("<QdddBBBd", point_header)
            xyz_rows.append((unpacked[1], unpacked[2], unpacked[3]))
            rgb_rows.append((unpacked[4], unpacked[5], unpacked[6]))
            errors.append(unpacked[7])
            track_length = struct.unpack("<Q", f.read(8))[0]
            track_lengths.append(int(track_length))
            f.seek(track_length * 8, 1)

    return ColmapPointCloud(
        xyz=np.asarray(xyz_rows, dtype=np.float64),
        rgb=np.asarray(rgb_rows, dtype=np.uint8),
        error=np.asarray(errors, dtype=np.float64),
        track_length=np.asarray(track_lengths, dtype=np.int64),
    )


def read_points3d_txt(path: Path) -> ColmapPointCloud:
    xyz_rows: list[tuple[float, float, float]] = []
    rgb_rows: list[tuple[int, int, int]] = []
    errors: list[float] = []
    track_lengths: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            xyz_rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb_rows.append((int(parts[4]), int(parts[5]), int(parts[6])))
            errors.append(float(parts[7]))
            track_lengths.append(max(0, (len(parts) - 8) // 2))

    return ColmapPointCloud(
        xyz=np.asarray(xyz_rows, dtype=np.float64),
        rgb=np.asarray(rgb_rows, dtype=np.uint8),
        error=np.asarray(errors, dtype=np.float64),
        track_length=np.asarray(track_lengths, dtype=np.int64),
    )


def read_points3d(path: Path) -> ColmapPointCloud:
    if path.suffix == ".bin":
        return read_points3d_bin(path)
    if path.suffix == ".txt":
        return read_points3d_txt(path)
    raise ValueError(f"Unsupported COLMAP points3D file: {path}")


def filter_points(
    point_cloud: ColmapPointCloud,
    min_track_length: int,
    max_error: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    keep = np.ones((len(point_cloud.xyz),), dtype=bool)
    if min_track_length > 0:
        keep &= point_cloud.track_length >= min_track_length
    if max_error is not None:
        keep &= point_cloud.error <= max_error

    xyz = point_cloud.xyz[keep]
    rgb = point_cloud.rgb[keep]
    metadata = {
        "points_before": int(len(point_cloud.xyz)),
        "points_after": int(len(xyz)),
        "min_track_length": int(min_track_length),
        "max_error": max_error,
        "track_length_mean": None
        if len(point_cloud.track_length) == 0
        else float(np.mean(point_cloud.track_length)),
        "error_mean": None
        if len(point_cloud.error) == 0
        else float(np.mean(point_cloud.error)),
    }
    return xyz, rgb, metadata


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if len(rgb) == len(xyz):
        point_cloud.colors = o3d.utility.Vector3dVector(
            rgb.astype(np.float64) / 255.0
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def convert_scene(
    cache_root: Path,
    output_root: Path,
    scene: str,
    min_track_length: int,
    max_error: float | None,
    force: bool,
) -> dict[str, object]:
    output_path = output_root / f"{scene}.ply"
    if output_path.exists() and not force:
        return {
            "scene": scene,
            "status": "skipped_exists",
            "output_path": str(output_path),
        }

    scene_dir = find_scene_dir(cache_root, scene)
    if scene_dir is None:
        return {"scene": scene, "status": "missing_scene_dir"}
    points3d_path = find_points3d_path(scene_dir)
    if points3d_path is None:
        return {
            "scene": scene,
            "status": "missing_points3d",
            "scene_dir": str(scene_dir),
        }

    colmap_points = read_points3d(points3d_path)
    xyz, rgb, metadata = filter_points(
        colmap_points,
        min_track_length=min_track_length,
        max_error=max_error,
    )
    if len(xyz) == 0:
        return {
            "scene": scene,
            "status": "empty_after_filter",
            "scene_dir": str(scene_dir),
            "points3d_path": str(points3d_path),
            **metadata,
        }

    write_ply(output_path, xyz, rgb)
    return {
        "scene": scene,
        "status": "converted",
        "scene_dir": str(scene_dir),
        "points3d_path": str(points3d_path),
        "output_path": str(output_path),
        **metadata,
    }


def main() -> None:
    args = parse_args()
    cache_root = Path(args.colmap_cache_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not cache_root.exists():
        raise FileNotFoundError(f"COLMAP cache root not found: {cache_root}")

    scenes = read_scene_keys(args.index_path, args.scene)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in tqdm(scenes, desc="Converting DL3DV COLMAP point clouds"):
        results.append(
            convert_scene(
                cache_root=cache_root,
                output_root=output_root,
                scene=scene,
                min_track_length=args.min_track_length,
                max_error=args.max_error,
                force=args.force,
            )
        )

    manifest = {
        "colmap_cache_root": str(cache_root),
        "output_root": str(output_root),
        "num_scenes_requested": len(scenes),
        "num_converted": sum(item["status"] == "converted" for item in results),
        "num_skipped_exists": sum(
            item["status"] == "skipped_exists" for item in results
        ),
        "num_failed": sum(
            item["status"] not in {"converted", "skipped_exists"} for item in results
        ),
        "results": results,
    }
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))
    print(f"Wrote manifest: {manifest_path}")

    if args.strict and manifest["num_failed"]:
        raise SystemExit(f"Failed to convert {manifest['num_failed']} scene(s).")


if __name__ == "__main__":
    main()
