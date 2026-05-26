import argparse
import bisect
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for lean environments.
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_SOURCE_ROOT = Path(os.environ.get("SCANNETPP_SOURCE_ROOT", "data/InsScene-15K/processed_scannetpp_v2"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SCANNETPP_ROOT", "data/scannetpp"))
DEFAULT_EVAL_INDEX_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "evaluation_index_scannetpp_iphone_local.json"
)

TARGET_BYTES_PER_CHUNK = int(1e8)

SplitName = Literal["train", "test"]
SensorName = Literal["iphone", "dslr"]

TRAIN_SPLIT = "nvs_sem_train.txt"
VAL_SPLIT = "nvs_sem_val.txt"
TEST_SPLIT = "nvs_test.txt"


class MultiPartReader(io.RawIOBase):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.files = [path.open("rb") for path in paths]
        self.sizes = [path.stat().st_size for path in paths]
        self.offsets: list[int] = []
        current = 0
        for size in self.sizes:
            self.offsets.append(current)
            current += size
        self.total_size = current
        self.position = 0

    def close(self) -> None:
        for file in self.files:
            try:
                file.close()
            except Exception:
                pass
        super().close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.position + offset
        elif whence == os.SEEK_END:
            new_position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence={whence}")

        if new_position < 0:
            raise ValueError("Negative seek position is invalid.")

        self.position = new_position
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.total_size:
            return 0

        memory = memoryview(buffer)
        remaining = min(len(memory), self.total_size - self.position)
        written = 0

        while remaining > 0:
            part_index = bisect.bisect_right(self.offsets, self.position) - 1
            part_offset = self.position - self.offsets[part_index]
            take = min(remaining, self.sizes[part_index] - part_offset)

            file = self.files[part_index]
            file.seek(part_offset)
            num_bytes = file.readinto(memory[written : written + take])
            if not num_bytes:
                break

            self.position += num_bytes
            written += num_bytes
            remaining -= num_bytes

        return written


class ScanNetPPArchiveSource:
    root: Path
    archive: zipfile.ZipFile | None
    archive_buffer: io.BufferedReader | None
    archive_reader: MultiPartReader | None
    metadata_presence: dict[str, set[SensorName]]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.archive = None
        self.archive_buffer = None
        self.archive_reader = None
        self.metadata_presence = {}

    def __enter__(self) -> "ScanNetPPArchiveSource":
        archive_parts = sorted(self.root.glob("processed_scannetpp_v2.zip.*"))
        if not archive_parts:
            raise FileNotFoundError(
                f"No multipart ScanNet++ archive parts found under {self.root}."
            )

        self.archive_reader = MultiPartReader(archive_parts)
        self.archive_buffer = io.BufferedReader(self.archive_reader, buffer_size=8 * 1024 * 1024)
        self.archive = zipfile.ZipFile(self.archive_buffer)
        self._index_metadata_presence()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.archive is not None:
            self.archive.close()
        if self.archive_buffer is not None:
            self.archive_buffer.close()
        if self.archive_reader is not None:
            self.archive_reader.close()

    def _index_metadata_presence(self) -> None:
        assert self.archive is not None
        metadata_presence: dict[str, set[SensorName]] = {}
        for info in self.archive.filelist:
            name = info.filename
            if name.endswith("scene_iphone_metadata.npz"):
                scene = name.split("/")[1]
                metadata_presence.setdefault(scene, set()).add("iphone")
            elif name.endswith("scene_dslr_metadata.npz"):
                scene = name.split("/")[1]
                metadata_presence.setdefault(scene, set()).add("dslr")
        self.metadata_presence = metadata_presence

    def has_sensor(self, scene: str, sensor: SensorName) -> bool:
        return sensor in self.metadata_presence.get(scene, set())

    def read_split(self, split_file: str) -> list[str]:
        assert self.archive is not None
        split_path = f"processed_scannetpp_v2/splits/{split_file}"
        with self.archive.open(split_path) as file:
            return [line.decode("utf-8").strip() for line in file if line.strip()]

    def load_metadata(self, scene: str, sensor: SensorName) -> dict[str, np.ndarray]:
        assert self.archive is not None
        metadata_path = f"processed_scannetpp_v2/{scene}/scene_{sensor}_metadata.npz"
        with self.archive.open(metadata_path) as file:
            return dict(np.load(io.BytesIO(file.read())))

    def load_image_bytes(self, scene: str, image_name: str) -> bytes:
        assert self.archive is not None
        image_path = f"processed_scannetpp_v2/{scene}/images/{image_name}"
        with self.archive.open(image_path) as file:
            return file.read()


class ScanNetPPDirectorySource:
    root: Path
    metadata_presence: dict[str, set[SensorName]]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata_presence = {}
        self._index_metadata_presence()

    def __enter__(self) -> "ScanNetPPDirectorySource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def _index_metadata_presence(self) -> None:
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir() or len(scene_dir.name) != 10:
                continue
            sensors: set[SensorName] = set()
            if (scene_dir / "scene_iphone_metadata.npz").exists():
                sensors.add("iphone")
            if (scene_dir / "scene_dslr_metadata.npz").exists():
                sensors.add("dslr")
            if sensors:
                self.metadata_presence[scene_dir.name] = sensors

    def has_sensor(self, scene: str, sensor: SensorName) -> bool:
        return sensor in self.metadata_presence.get(scene, set())

    def read_split(self, split_file: str) -> list[str]:
        split_path = self.root / "splits" / split_file
        return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]

    def load_metadata(self, scene: str, sensor: SensorName) -> dict[str, np.ndarray]:
        metadata_path = self.root / scene / f"scene_{sensor}_metadata.npz"
        return dict(np.load(metadata_path))

    def load_image_bytes(self, scene: str, image_name: str) -> bytes:
        return (self.root / scene / "images" / image_name).read_bytes()


def detect_source(root: Path):
    archive_parts = sorted(root.glob("processed_scannetpp_v2.zip.*"))
    if archive_parts:
        return ScanNetPPArchiveSource(root)

    extracted_root = root / "processed_scannetpp_v2"
    if (extracted_root / "splits").exists():
        return ScanNetPPDirectorySource(extracted_root)
    if (root / "splits").exists():
        return ScanNetPPDirectorySource(root)

    raise FileNotFoundError(
        f"Could not detect a supported ScanNet++ source layout under {root}."
    )


def load_raw_bytes(data: bytes) -> torch.Tensor:
    array = np.frombuffer(data, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def choose_eval_entry(num_views: int, num_target_views: int = 3) -> dict | None:
    if num_views < max(4, num_target_views + 2):
        return None

    left = max(0, int(round(num_views * 0.2)))
    right = min(num_views - 1, int(round(num_views * 0.8)))
    if right <= left:
        left = 0
        right = num_views - 1

    if right - left < num_target_views + 1:
        left = 0
        right = num_views - 1

    interior_positions = np.linspace(left, right, num_target_views + 2)[1:-1]
    targets: list[int] = []
    for position in interior_positions:
        index = int(round(float(position)))
        if index <= left:
            index = left + 1
        if index >= right:
            index = right - 1
        if index not in targets:
            targets.append(index)

    candidate = left + 1
    while len(targets) < num_target_views and candidate < right:
        if candidate not in targets:
            targets.append(candidate)
        candidate += 1

    targets = sorted(targets[:num_target_views])
    if len(targets) != num_target_views:
        return None

    return {
        "context": [left, right],
        "target": targets,
        "overlap": -1.0,
    }


def build_camera_rows(
    trajectories: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    trajectories = trajectories.astype(np.float32)
    intrinsics = intrinsics.astype(np.float32)

    w2c = np.linalg.inv(trajectories)[:, :3, :]
    camera_rows = np.zeros((len(trajectories), 18), dtype=np.float32)
    camera_rows[:, 0] = intrinsics[:, 0, 0] / image_width
    camera_rows[:, 1] = intrinsics[:, 1, 1] / image_height
    camera_rows[:, 2] = intrinsics[:, 0, 2] / image_width
    camera_rows[:, 3] = intrinsics[:, 1, 2] / image_height
    camera_rows[:, 6:] = w2c.reshape(len(trajectories), -1)
    return torch.from_numpy(camera_rows)


def save_index(stage_path: Path) -> None:
    index = {}
    for chunk_path in tqdm(
        sorted(stage_path.glob("*.torch")),
        desc=f"Indexing {stage_path}",
    ):
        chunk = torch.load(chunk_path)
        for example in chunk:
            index[example["key"]] = str(chunk_path.relative_to(stage_path))
    with (stage_path / "index.json").open("w") as file:
        json.dump(index, file)


def convert_sensor(
    source,
    sensor: SensorName,
    output_root: Path,
    include_val_in_train: bool,
    target_bytes_per_chunk: int,
    limit_train_scenes: int | None,
    limit_test_scenes: int | None,
    generate_eval_index: bool,
    eval_index_path: Path,
) -> None:
    def take_first_valid_scenes(scenes: list[str], limit: int | None) -> list[str]:
        if limit is None:
            return scenes

        selected = []
        for scene in scenes:
            if source.has_sensor(scene, sensor):
                selected.append(scene)
            if len(selected) >= limit:
                break
        return selected

    split_manifest: dict[SplitName, list[str]] = {
        "train": [],
        "test": [],
    }

    train_scenes = source.read_split(TRAIN_SPLIT)
    val_scenes = source.read_split(VAL_SPLIT)
    test_scenes = source.read_split(TEST_SPLIT)

    if any(source.has_sensor(scene, sensor) for scene in test_scenes):
        if include_val_in_train:
            train_scenes.extend(val_scenes)
    else:
        print(
            f"[convert_scannetpp] {sensor}: no usable scenes found in {TEST_SPLIT}; "
            f"falling back to {VAL_SPLIT} as the test split."
        )
        test_scenes = val_scenes

    train_scenes = take_first_valid_scenes(train_scenes, limit_train_scenes)
    test_scenes = take_first_valid_scenes(test_scenes, limit_test_scenes)

    split_manifest["train"] = train_scenes
    split_manifest["test"] = test_scenes

    sensor_root = output_root / sensor
    sensor_root.mkdir(parents=True, exist_ok=True)
    (sensor_root / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2)
    )

    evaluation_index = {}
    stage_shapes = {}

    for stage, scenes in split_manifest.items():
        stage_path = sensor_root / stage
        stage_path.mkdir(parents=True, exist_ok=True)

        chunk_size = 0
        chunk_index = 0
        chunk: list[dict] = []
        stage_summary = {
            "stage": stage,
            "sensor": sensor,
            "num_scenes_requested": len(scenes),
            "num_examples_written": 0,
            "num_scenes_skipped_missing_sensor": 0,
            "num_scenes_skipped_missing_images": 0,
            "num_frames_total": 0,
        }

        def flush_chunk() -> None:
            nonlocal chunk_size
            nonlocal chunk_index
            nonlocal chunk

            if not chunk:
                return

            chunk_key = f"{chunk_index:0>6}"
            torch.save(chunk, stage_path / f"{chunk_key}.torch")
            chunk = []
            chunk_size = 0
            chunk_index += 1

        for scene in tqdm(scenes, desc=f"Converting {sensor}/{stage}"):
            if not source.has_sensor(scene, sensor):
                stage_summary["num_scenes_skipped_missing_sensor"] += 1
                continue

            metadata = source.load_metadata(scene, sensor)
            image_names = metadata["images"].tolist()
            if not image_names:
                stage_summary["num_scenes_skipped_missing_images"] += 1
                continue

            raw_images = []
            scene_bytes = 0
            image_width = image_height = None
            missing_image = False
            for image_name in image_names:
                try:
                    image_bytes = source.load_image_bytes(scene, image_name)
                except KeyError:
                    missing_image = True
                    break
                raw_images.append(load_raw_bytes(image_bytes))
                scene_bytes += len(image_bytes)
                if image_width is None or image_height is None:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image_width, image_height = image.size

            if missing_image or image_width is None or image_height is None:
                stage_summary["num_scenes_skipped_missing_images"] += 1
                continue

            cameras = build_camera_rows(
                metadata["trajectories"],
                metadata["intrinsics"],
                image_width,
                image_height,
            )

            key = f"{scene}_{sensor}"
            example = {
                "key": key,
                "images": raw_images,
                "cameras": cameras,
            }
            chunk.append(example)
            chunk_size += scene_bytes
            stage_summary["num_examples_written"] += 1
            stage_summary["num_frames_total"] += len(image_names)
            stage_shapes[key] = [image_height, image_width]

            if stage == "test" and generate_eval_index:
                entry = choose_eval_entry(len(image_names))
                if entry is not None:
                    evaluation_index[key] = entry

            if chunk_size >= target_bytes_per_chunk:
                flush_chunk()

        flush_chunk()
        save_index(stage_path)
        (stage_path / "summary.json").write_text(json.dumps(stage_summary, indent=2))

    if generate_eval_index:
        eval_index_path.parent.mkdir(parents=True, exist_ok=True)
        eval_index_path.write_text(json.dumps(evaluation_index, indent=2))

    (sensor_root / "image_shapes.json").write_text(json.dumps(stage_shapes, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ScanNet++ metadata/images into repo-native .torch chunks."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing either the multipart archive or an extracted processed_scannetpp_v2 tree.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root. Sensor-specific outputs are written to <output-root>/<sensor>/train|test.",
    )
    parser.add_argument(
        "--sensor",
        choices=("iphone", "dslr", "both"),
        default="iphone",
        help="Which ScanNet++ sensor stream to convert.",
    )
    parser.add_argument(
        "--target-bytes-per-chunk",
        type=int,
        default=TARGET_BYTES_PER_CHUNK,
        help="Approximate uncompressed image bytes per saved .torch chunk.",
    )
    parser.add_argument(
        "--eval-index-path",
        type=Path,
        default=DEFAULT_EVAL_INDEX_PATH,
        help="Where to write the generated local evaluation index for the selected sensor.",
    )
    parser.add_argument(
        "--include-val-in-train",
        action="store_true",
        help="Also merge nvs_sem_val into train when a separate nvs_test split is available.",
    )
    parser.add_argument(
        "--no-generate-eval-index",
        action="store_true",
        help="Skip generating assets/evaluation_index_scannetpp_iphone_local.json.",
    )
    parser.add_argument(
        "--limit-train-scenes",
        type=int,
        default=None,
        help="Optional scene cap for train conversion, useful for smoke tests.",
    )
    parser.add_argument(
        "--limit-test-scenes",
        type=int,
        default=None,
        help="Optional scene cap for test conversion, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensors: list[SensorName]
    if args.sensor == "both":
        sensors = ["iphone", "dslr"]
    else:
        sensors = [args.sensor]

    with detect_source(args.source_root) as source:
        for sensor in sensors:
            eval_index_path = args.eval_index_path
            if len(sensors) > 1:
                eval_index_path = eval_index_path.with_name(
                    f"{eval_index_path.stem}_{sensor}{eval_index_path.suffix}"
                )
            convert_sensor(
                source=source,
                sensor=sensor,
                output_root=args.output_root,
                include_val_in_train=args.include_val_in_train,
                target_bytes_per_chunk=args.target_bytes_per_chunk,
                limit_train_scenes=args.limit_train_scenes,
                limit_test_scenes=args.limit_test_scenes,
                generate_eval_index=not args.no_generate_eval_index,
                eval_index_path=eval_index_path,
            )


if __name__ == "__main__":
    main()
