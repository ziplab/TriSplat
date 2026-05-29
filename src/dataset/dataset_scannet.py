import random
import struct
import zlib
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon
from .norm_scale import (
    build_normalized_from_world_transform,
    build_world_from_normalized_transform,
    compute_pose_norm_scale,
)
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim as apply_default_crop_shim
from .shims.crop_shim_gs import apply_crop_shim as apply_gs_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

SENS_VERSION = 4
DEPTH_COMPRESSION_ZLIB_USHORT = 1


@dataclass
class DatasetScanNetCfg(DatasetCfgCommon):
    name: Literal["scannet"]
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    test_len: int
    test_chunk_interval: int
    pose_norm_method: str
    near: float = 0.1
    far: float = 10.0
    skip_bad_pose: bool = True
    shuffle_val: bool = False
    train_times_per_scene: int = 1


@dataclass
class DatasetScanNetCfgWrapper:
    scannet: DatasetScanNetCfg


@dataclass(frozen=True)
class SensFrameRef:
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_offset: int
    color_size: int
    depth_offset: int
    depth_size: int


class ScanNetSensData:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.version = 0
        self.sensor_name = ""
        self.intrinsic_color = np.eye(4, dtype=np.float64)
        self.extrinsic_color = np.eye(4, dtype=np.float64)
        self.intrinsic_depth = np.eye(4, dtype=np.float64)
        self.extrinsic_depth = np.eye(4, dtype=np.float64)
        self.color_compression_type = -1
        self.depth_compression_type = -1
        self.color_width = 0
        self.color_height = 0
        self.depth_width = 0
        self.depth_height = 0
        self.depth_shift = 1000.0
        self.frames: list[SensFrameRef] = []
        self._load_index()

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def depth_intrinsics_3x3(self) -> np.ndarray:
        return self.intrinsic_depth[:3, :3].astype(np.float64).copy()

    @property
    def color_intrinsics_3x3(self) -> np.ndarray:
        return self.intrinsic_color[:3, :3].astype(np.float64).copy()

    @property
    def color_intrinsics_normalized_3x3(self) -> np.ndarray:
        intrinsics = self.color_intrinsics_3x3
        intrinsics[0, 0] /= float(self.color_width)
        intrinsics[1, 1] /= float(self.color_height)
        intrinsics[0, 2] /= float(self.color_width)
        intrinsics[1, 2] /= float(self.color_height)
        return intrinsics

    def color_intrinsics_normalized_stack(self) -> Tensor:
        intrinsics = self.color_intrinsics_normalized_3x3
        stack = np.broadcast_to(intrinsics[None], (self.num_frames, 3, 3)).copy()
        return torch.from_numpy(stack.astype(np.float32))

    def _load_index(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"ScanNet .sens file not found: {self.path}")

        with self.path.open("rb") as file_handle:
            self.version = _read_struct(file_handle, "<I")[0]
            if self.version != SENS_VERSION:
                raise ValueError(f"Unsupported .sens version {self.version} in {self.path}; expected {SENS_VERSION}.")

            sensor_name_len = _read_struct(file_handle, "<Q")[0]
            self.sensor_name = file_handle.read(sensor_name_len).decode("utf-8", errors="replace")
            self.intrinsic_color = _read_mat4(file_handle)
            self.extrinsic_color = _read_mat4(file_handle)
            self.intrinsic_depth = _read_mat4(file_handle)
            self.extrinsic_depth = _read_mat4(file_handle)
            (
                self.color_compression_type,
                self.depth_compression_type,
                self.color_width,
                self.color_height,
                self.depth_width,
                self.depth_height,
                self.depth_shift,
                num_frames,
            ) = _read_struct(file_handle, "<iiIIIIfQ")

            frame_header_size = struct.calcsize("<16fQQQQ")
            frames: list[SensFrameRef] = []
            for _ in range(int(num_frames)):
                frame_header = file_handle.read(frame_header_size)
                if len(frame_header) != frame_header_size:
                    raise EOFError(f"Unexpected EOF while reading frame index from {self.path}")
                unpacked = struct.unpack("<16fQQQQ", frame_header)
                camera_to_world = np.asarray(unpacked[:16], dtype=np.float64).reshape(4, 4)
                timestamp_color = int(unpacked[16])
                timestamp_depth = int(unpacked[17])
                color_size = int(unpacked[18])
                depth_size = int(unpacked[19])
                color_offset = int(file_handle.tell())
                file_handle.seek(color_size, 1)
                depth_offset = int(file_handle.tell())
                file_handle.seek(depth_size, 1)
                frames.append(
                    SensFrameRef(
                        camera_to_world=camera_to_world,
                        timestamp_color=timestamp_color,
                        timestamp_depth=timestamp_depth,
                        color_offset=color_offset,
                        color_size=color_size,
                        depth_offset=depth_offset,
                        depth_size=depth_size,
                    )
                )
            self.frames = frames

    def _check_frame_index(self, frame_index: int) -> None:
        if frame_index < 0 or frame_index >= self.num_frames:
            raise IndexError(
                f"Frame index {frame_index} is out of range for {self.path.name} with {self.num_frames} frames."
            )

    def camera_to_world_stack(self) -> Tensor:
        poses = np.stack([frame.camera_to_world for frame in self.frames], axis=0)
        return torch.from_numpy(poses.astype(np.float32))

    def load_color(self, frame_index: int) -> Image.Image:
        self._check_frame_index(frame_index)
        frame = self.frames[int(frame_index)]
        with self.path.open("rb") as file_handle:
            file_handle.seek(frame.color_offset)
            color_bytes = file_handle.read(frame.color_size)
        return Image.open(BytesIO(color_bytes)).convert("RGB")

    def load_depth_meters(self, frame_index: int) -> np.ndarray:
        self._check_frame_index(frame_index)
        if self.depth_compression_type != DEPTH_COMPRESSION_ZLIB_USHORT:
            raise ValueError(
                f"Unsupported ScanNet depth compression type {self.depth_compression_type}; "
                f"only zlib_ushort ({DEPTH_COMPRESSION_ZLIB_USHORT}) is supported."
            )

        frame = self.frames[frame_index]
        with self.path.open("rb") as file_handle:
            file_handle.seek(frame.depth_offset)
            compressed = file_handle.read(frame.depth_size)
        raw_depth = zlib.decompress(compressed)
        expected_bytes = int(self.depth_height) * int(self.depth_width) * np.dtype(np.uint16).itemsize
        if len(raw_depth) != expected_bytes:
            raise ValueError(
                f"Depth payload for frame {frame_index} has {len(raw_depth)} bytes; expected {expected_bytes}."
            )
        depth_u16 = np.frombuffer(raw_depth, dtype=np.uint16).reshape(int(self.depth_height), int(self.depth_width))
        depth_m = depth_u16.astype(np.float32) / float(self.depth_shift)
        depth_m[depth_u16 == 0] = 0.0
        return depth_m


def _read_struct(file_handle, fmt: str) -> tuple:
    size = struct.calcsize(fmt)
    data = file_handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected EOF while reading .sens file.")
    return struct.unpack(fmt, data)


def _read_mat4(file_handle) -> np.ndarray:
    return np.asarray(_read_struct(file_handle, "<16f"), dtype=np.float64).reshape(4, 4)


def is_valid_pose(pose: Tensor) -> bool:
    if not torch.isfinite(pose).all():
        return False
    det = torch.det(pose[:3, :3])
    return bool(torch.isfinite(det) and torch.abs(det - 1.0) < 1e-2)


def apply_configured_crop_shim(example, shape: tuple[int, int], crop_mode: str):
    if crop_mode == "default":
        return apply_default_crop_shim(example, shape)
    if crop_mode == "gs":
        return apply_gs_crop_shim(example, shape)
    raise ValueError(f"Unsupported crop_mode: {crop_mode}")


class DatasetScanNet(IterableDataset):
    cfg: DatasetScanNetCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetScanNetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.scene_dirs = self._collect_scene_dirs()
        if cfg.overfit_to_scene is not None:
            self.scene_dirs = [self.index[cfg.overfit_to_scene]]
        if self.stage == "test":
            self.scene_dirs = self.scene_dirs[:: max(int(self.cfg.test_chunk_interval), 1)]
            if self.cfg.test_len > 0:
                self.scene_dirs = self.scene_dirs[: self.cfg.test_len]

    def _stage_dir_candidates(self, root: Path) -> list[Path]:
        candidates = []
        if self.data_stage == "test":
            candidates.extend([root / "scans_test", root / "test"])
        candidates.extend([root / self.data_stage, root / "scans", root])
        seen = set()
        unique = []
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(resolved)
        return unique

    def _collect_scene_dirs(self) -> list[Path]:
        scene_dirs: list[Path] = []
        for root in self.cfg.roots:
            for candidate in self._stage_dir_candidates(root):
                if not candidate.is_dir():
                    continue
                if list(candidate.glob("*.sens")):
                    scene_dirs.append(candidate.resolve())
                    break
                root_scene_dirs = [
                    scene_dir
                    for scene_dir in sorted(candidate.iterdir())
                    if scene_dir.is_dir() and (scene_dir / f"{scene_dir.name}.sens").exists()
                ]
                if root_scene_dirs:
                    scene_dirs.extend(root_scene_dirs)
                    break
        return sorted(scene_dirs)

    def __iter__(self):
        scene_dirs = list(self.scene_dirs)
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train",)):
            random.shuffle(scene_dirs)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            scene_dirs = [
                scene_dir
                for scene_index, scene_dir in enumerate(scene_dirs)
                if scene_index % worker_info.num_workers == worker_info.id
            ]

        for scene_dir in scene_dirs:
            scene = scene_dir.name
            sens_path = scene_dir / f"{scene}.sens"
            if not sens_path.exists():
                matches = sorted(scene_dir.glob("*.sens"))
                if not matches:
                    continue
                sens_path = matches[0]
            try:
                sensor = ScanNetSensData(sens_path)
            except Exception:
                continue

            extrinsics = sensor.camera_to_world_stack()
            intrinsics = sensor.color_intrinsics_normalized_stack()
            try:
                context_indices, target_indices, _ = self.view_sampler.sample(scene, extrinsics, intrinsics)
            except ValueError:
                continue

            selected_indices = torch.cat([context_indices, target_indices], dim=0)
            if self.cfg.skip_bad_pose and any(
                not is_valid_pose(extrinsics[int(index.item())]) for index in selected_indices
            ):
                continue

            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                continue

            try:
                context_images = self.convert_images(sensor, context_indices)
                target_images = self.convert_images(sensor, target_indices)
            except Exception:
                continue

            context_extrinsics = extrinsics[context_indices]
            target_extrinsics = extrinsics[target_indices]
            num_context_images = len(context_extrinsics)
            all_used_extrinsics = torch.cat([context_extrinsics.clone(), target_extrinsics.clone()], dim=0)

            scale = compute_pose_norm_scale(context_extrinsics, self.cfg.pose_norm_method)
            scale_value = float(scale.detach().item()) if isinstance(scale, torch.Tensor) else float(scale)
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                continue
            if scale_value < self.cfg.baseline_min or scale_value > self.cfg.baseline_max:
                continue

            world_from_normalized = build_world_from_normalized_transform(
                context_extrinsics,
                scale,
                self.cfg.relative_pose,
            )
            normalized_from_world = build_normalized_from_world_transform(
                context_extrinsics,
                scale,
                self.cfg.relative_pose,
            )

            all_used_extrinsics[:, :3, 3] /= scale_value
            if self.cfg.relative_pose:
                all_used_extrinsics = camera_normalization(all_used_extrinsics[0:1], all_used_extrinsics)

            example = {
                "context": {
                    "extrinsics": all_used_extrinsics[:num_context_images],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / scale_value,
                    "far": self.get_bound("far", len(context_indices)) / scale_value,
                    "index": context_indices,
                },
                "target": {
                    "extrinsics": all_used_extrinsics[num_context_images:],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale_value,
                    "far": self.get_bound("far", len(target_indices)) / scale_value,
                    "index": target_indices,
                },
                "scene": scene,
                "scene_metadata": {
                    "mesh_space": "normalized",
                    "world_space": "raw_world",
                    "pose_norm_scale": torch.tensor(scale_value, dtype=torch.float32),
                    "pose_norm_method": self.cfg.pose_norm_method,
                    "relative_pose": torch.tensor(self.cfg.relative_pose, dtype=torch.bool),
                    "world_from_normalized": world_from_normalized.to(torch.float32),
                    "normalized_from_world": normalized_from_world.to(torch.float32),
                    "context_indices": context_indices.clone(),
                    "target_indices": target_indices.clone(),
                    "index_path": str(getattr(getattr(self.view_sampler, "cfg", None), "index_path", "")),
                    "num_context_views": torch.tensor(int(num_context_images), dtype=torch.int64),
                },
            }

            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_configured_crop_shim(example, tuple(self.cfg.input_image_shape), self.cfg.crop_mode)

    def convert_images(
        self,
        sensor: ScanNetSensData,
        indices: Tensor,
    ) -> Tensor:
        images = [self.to_tensor(sensor.load_color(int(index.item()))) for index in indices]
        return torch.stack(images)

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Tensor:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return value.repeat(num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        return {scene_dir.name: scene_dir for scene_dir in self.scene_dirs}

    def __len__(self) -> int:
        if self.stage == "test" and self.cfg.test_len > 0:
            return min(len(self.scene_dirs), self.cfg.test_len)
        return len(self.scene_dirs) * self.cfg.train_times_per_scene
