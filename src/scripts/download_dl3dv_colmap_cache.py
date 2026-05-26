import argparse
import json
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi

DL3DV_COLMAP_REPO = "DL3DV/DL3DV-ALL-ColmapCache"
DL3DV_SUBSETS = ("1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K")
CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DL3DV COLMAP cache zip files and extract them.",
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="Evaluation index JSON whose keys are DL3DV scene hashes.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("DL3DV_COLMAP_OUTPUT_ROOT", "data"),
        help="Root where dl3dv_colmap_cache_zips and dl3dv_colmap_cache are written.",
    )
    parser.add_argument(
        "--local-cache-root",
        default=os.environ.get("DL3DV_COLMAP_LOCAL_CACHE_ROOT", ".cache/dl3dv_colmap_cache_zips"),
        help="Local filesystem cache for zip downloads before copying to output-root.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help=(
            "Optional scene hash to download. Can be repeated; defaults to "
            "all index keys."
        ),
    )
    parser.add_argument("--repo-id", default=DL3DV_COLMAP_REPO)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep part files after merging, useful for debugging.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download/copy zip files; do not extract.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload zip even if the local cached zip has the expected size.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract even if an existing scene directory already has points3D files.",
    )
    parser.add_argument(
        "--manifest-name",
        default="dl3dv_colmap_first20_manifest.json",
        help="Manifest filename under output-root.",
    )
    return parser.parse_args()


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for the gated "
            "DL3DV COLMAP cache dataset."
        )
    return token


def read_scenes(index_path: Path, scene_filter: list[str]) -> list[str]:
    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        scenes = [str(key) for key in payload.keys()]
    elif isinstance(payload, list):
        scenes = [
            str(item if isinstance(item, str) else item["scene"])
            for item in payload
        ]
    else:
        raise TypeError(f"Unsupported index payload type: {type(payload)}")

    if scene_filter:
        requested = set(scene_filter)
        scenes = [scene for scene in scenes if scene in requested]
        missing = sorted(requested - set(scenes))
        if missing:
            raise ValueError(f"Requested scene(s) not present in index: {missing}")
    if not scenes:
        raise ValueError("No scenes requested.")
    return scenes


def build_remote_file_map(repo_id: str, token: str) -> dict[str, dict[str, object]]:
    api = HfApi(token=token)
    remote_files: dict[str, dict[str, object]] = {}
    for subset in DL3DV_SUBSETS:
        items = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subset,
            recursive=False,
        )
        for item in items:
            path = getattr(item, "path", "")
            name = path.split("/")[-1]
            if name.endswith(".zip"):
                remote_files[name[:-4]] = {
                    "path": path,
                    "size": getattr(item, "size", None),
                }
    return remote_files


def resolve_signed_url(
    repo_id: str,
    filename: str,
    token: str,
    timeout: tuple[float, float],
) -> tuple[str, int]:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(filename)}"
    response = requests.head(
        url,
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=True,
        timeout=timeout,
    )
    response.raise_for_status()
    size = int(response.headers.get("content-length", "0"))
    if size <= 0:
        raise RuntimeError(f"Could not resolve content length for {filename}")
    return response.url, size


def download_range(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: tuple[float, float],
) -> int:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return expected

    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}, expected 206")
                content_range = response.headers.get("content-range", "")
                match = CONTENT_RANGE_PATTERN.fullmatch(content_range)
                if match is None:
                    raise RuntimeError(f"Invalid Content-Range: {content_range!r}")
                actual_start = int(match.group(1))
                actual_end = int(match.group(2))
                if actual_start != start or actual_end != end:
                    raise RuntimeError(
                        "Content-Range mismatch: "
                        f"{actual_start}-{actual_end} != {start}-{end}"
                    )
                written = 0
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                if written != expected:
                    raise RuntimeError(f"part size mismatch {written} != {expected}")
                tmp_path.replace(part_path)
                return written
        except Exception as exc:
            last_error = exc
            time.sleep(min(60.0, 2.0 * attempt))

    raise RuntimeError(f"Failed part {start}-{end}: {last_error}")


def iter_ranges(size: int, chunk_size: int) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    part_index = 0
    while start < size:
        end = min(size - 1, start + chunk_size - 1)
        ranges.append((part_index, start, end))
        start = end + 1
        part_index += 1
    return ranges


def parallel_download(
    *,
    repo_id: str,
    filename: str,
    output_path: Path,
    token: str,
    expected_size: int | None,
    workers: int,
    chunk_size: int,
    retries: int,
    timeout: tuple[float, float],
    force: bool,
    keep_parts: bool,
) -> None:
    if (
        not force
        and expected_size is not None
        and output_path.exists()
        and output_path.stat().st_size == expected_size
    ):
        print(f"  local zip exists: {output_path}", flush=True)
        return

    signed_url, resolved_size = resolve_signed_url(repo_id, filename, token, timeout)
    if expected_size is not None and resolved_size != expected_size:
        print(
            "  warning: API size "
            f"{expected_size} differs from HEAD size {resolved_size}",
            flush=True,
        )
    size = resolved_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output_path.parent / f"{output_path.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = iter_ranges(size, chunk_size)
    print(
        f"  range download size={size / 1024 / 1024:.2f} MB "
        f"parts={len(ranges)} workers={workers}",
        flush=True,
    )

    started = time.time()
    completed = 0
    pending_ranges = []
    for part_index, start, end in ranges:
        part_path = parts_dir / f"part_{part_index:05d}"
        expected_part_size = end - start + 1
        if part_path.exists() and part_path.stat().st_size == expected_part_size:
            completed += expected_part_size
        else:
            pending_ranges.append((part_index, start, end, part_path))
    if completed:
        print(f"  resuming from {completed / 1024 / 1024:.2f} MB", flush=True)

    last_report = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_range,
                signed_url,
                part_path,
                start,
                end,
                retries,
                timeout,
            )
            for _, start, end, part_path in pending_ranges
        ]
        for future in as_completed(futures):
            completed += future.result()
            now = time.time()
            if now - last_report >= 15 or completed == size:
                rate = completed / max(now - started, 1e-6) / 1024 / 1024
                print(
                    f"  progress {completed / 1024 / 1024:.1f}/"
                    f"{size / 1024 / 1024:.1f} MB rate={rate:.2f} MB/s",
                    flush=True,
                )
                last_report = now
        if not futures:
            print("  all parts already present", flush=True)

    tmp_output = output_path.with_suffix(output_path.suffix + ".merge_tmp")
    with tmp_output.open("wb") as output:
        for part_index, _, _ in ranges:
            part_path = parts_dir / f"part_{part_index:05d}"
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, output, length=1024 * 1024)
    if tmp_output.stat().st_size != size:
        raise RuntimeError(
            f"merged size mismatch {tmp_output.stat().st_size} != {size}"
        )
    tmp_output.replace(output_path)
    if not keep_parts:
        shutil.rmtree(parts_dir)


def copy_to_output(local_zip: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists() and output_zip.stat().st_size == local_zip.stat().st_size:
        print(f"  output zip exists: {output_zip}", flush=True)
        return
    tmp_output_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
    if tmp_output_zip.exists():
        tmp_output_zip.unlink()
    shutil.copyfile(local_zip, tmp_output_zip)
    tmp_output_zip.replace(output_zip)


def extract_zip(local_zip: Path, scene_dir: Path, force_extract: bool) -> list[str]:
    existing_points = list(scene_dir.rglob("points3D.*")) if scene_dir.exists() else []
    if existing_points and not force_extract:
        return sorted(str(path) for path in existing_points)
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as zip_file:
        zip_file.extractall(scene_dir)
    return sorted(str(path) for path in scene_dir.rglob("points3D.*"))


def main() -> None:
    args = parse_args()
    token = get_token()
    index_path = Path(args.index_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    zip_root = output_root / "dl3dv_colmap_cache_zips"
    extract_root = output_root / "dl3dv_colmap_cache"
    manifest_path = output_root / args.manifest_name
    chunk_size = int(args.chunk_mb) * 1024 * 1024
    timeout = (float(args.connect_timeout), float(args.read_timeout))

    print(f"index={index_path}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"local_cache_root={local_cache_root}", flush=True)
    print(f"workers={args.workers} chunk_mb={args.chunk_mb}", flush=True)

    scenes = read_scenes(index_path, args.scene)
    for path in (zip_root, extract_root, local_cache_root):
        path.mkdir(parents=True, exist_ok=True)

    print("building remote file map...", flush=True)
    remote_files = build_remote_file_map(args.repo_id, token)
    print(f"remote_zip_count={len(remote_files)}", flush=True)

    results = []
    for index, scene in enumerate(scenes, start=1):
        remote = remote_files.get(scene)
        if remote is None:
            result = {"scene": scene, "status": "missing_remote_zip"}
            results.append(result)
            print(f"[{index:02d}/{len(scenes)}] missing remote zip {scene}", flush=True)
            continue

        filename = str(remote["path"])
        expected_size = remote.get("size")
        expected_size = int(expected_size) if expected_size is not None else None
        local_zip = local_cache_root / filename
        output_zip = zip_root / filename
        scene_dir = extract_root / scene

        print(f"[{index:02d}/{len(scenes)}] {filename}", flush=True)
        try:
            parallel_download(
                repo_id=args.repo_id,
                filename=filename,
                output_path=local_zip,
                token=token,
                expected_size=expected_size,
                workers=args.workers,
                chunk_size=chunk_size,
                retries=args.retries,
                timeout=timeout,
                force=args.force_download,
                keep_parts=args.keep_parts,
            )
            copy_to_output(local_zip, output_zip)
            points3d_paths = [] if args.no_extract else extract_zip(
                local_zip,
                scene_dir,
                force_extract=args.force_extract,
            )
            status = (
                "downloaded"
                if args.no_extract
                else ("downloaded_extracted" if points3d_paths else "missing_points3d")
            )
        except Exception as exc:
            points3d_paths = []
            status = "failed"
            error = str(exc)
            print(f"  failed: {error}", flush=True)
        else:
            error = None

        result = {
            "scene": scene,
            "status": status,
            "remote_path": filename,
            "expected_size": expected_size,
            "local_zip": str(local_zip),
            "zip_path": str(output_zip),
            "extract_dir": str(scene_dir),
            "points3d_paths": points3d_paths,
        }
        if error is not None:
            result["error"] = error
        results.append(result)
        manifest_path.write_text(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "index_path": str(index_path),
                    "zip_root": str(zip_root),
                    "extract_root": str(extract_root),
                    "local_cache_root": str(local_cache_root),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"  status={status} points3d={len(points3d_paths)} "
            f"manifest={manifest_path}",
            flush=True,
        )

    ok_statuses = {"downloaded", "downloaded_extracted"}
    summary = {
        "repo_id": args.repo_id,
        "index_path": str(index_path),
        "zip_root": str(zip_root),
        "extract_root": str(extract_root),
        "local_cache_root": str(local_cache_root),
        "num_requested": len(scenes),
        "num_ok": sum(result["status"] in ok_statuses for result in results),
        "num_failed": sum(result["status"] not in ok_statuses for result in results),
        "results": results,
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if summary["num_failed"]:
        raise SystemExit(f"Failed scenes: {summary['num_failed']}")


if __name__ == "__main__":
    main()
