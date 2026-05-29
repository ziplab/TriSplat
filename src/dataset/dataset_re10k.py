import json
import logging
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim as apply_default_crop_shim
from .shims.crop_shim_gs import apply_crop_shim as apply_gs_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from .norm_scale import (
    build_normalized_from_world_transform,
    build_world_from_normalized_transform,
    compute_pose_norm_scale,
)
from ..misc.cam_utils import camera_normalization

logger = logging.getLogger(__name__)


def apply_configured_crop_shim(example, shape: tuple[int, int], crop_mode: str):
    if crop_mode == "default":
        return apply_default_crop_shim(example, shape)
    if crop_mode == "gs":
        return apply_gs_crop_shim(example, shape)
    raise ValueError(f"Unsupported crop_mode: {crop_mode}")


def indices_to_list(indices: Tensor) -> list[int]:
    return [int(index.item()) for index in indices]


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    test_roots: list[Path] | None = None
    pose_norm_method: str = "max_pairwise_d"  # "none", "start_end", "max_pairwise_d", "max_view1_d", "mean_pairwise_d", "max_trans"


@dataclass
class DatasetRE10kCfgWrapper:
    re10k: DatasetRE10kCfg


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetRE10kCfg


@dataclass
class DatasetScannetppCfgWrapper:
    scannetpp: DatasetRE10kCfg


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.name = cfg.name

        # Collect chunks.
        self.chunks = []
        roots = cfg.roots
        if self.data_stage == "test" and cfg.test_roots is not None:
            roots = cfg.test_roots

        for root in roots:
            stage_root = root / self.data_stage
            root = stage_root if stage_root.exists() else root
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.stage == "test":
            self.restrict_chunks_to_evaluation_index()
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def restrict_chunks_to_evaluation_index(self) -> None:
        """Avoid scanning every test chunk when an evaluation index is provided."""
        evaluation_index = getattr(self.view_sampler, "index", None)
        if evaluation_index is None:
            return

        scenes = {scene for scene, entry in evaluation_index.items() if entry is not None}
        chunk_paths = {self.index[scene] for scene in scenes if scene in self.index}
        if not chunk_paths:
            return

        original_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk in chunk_paths]
        if len(self.chunks) != original_count:
            logger.info(
                "Restricted test chunks from %d to %d using evaluation index.",
                original_count,
                len(self.chunks),
            )

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                raw_example = example
                extrinsics, intrinsics = self.convert_poses(raw_example["cameras"])
                scene = raw_example["key"]

                try:
                    context_indices, target_indices, overlap = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                try:
                    context_index_list = indices_to_list(context_indices)
                    target_index_list = indices_to_list(target_indices)
                    context_images = [raw_example["images"][index] for index in context_index_list]
                    context_images = self.convert_images(context_images)
                    target_images = [raw_example["images"][index] for index in target_index_list]
                    target_images = self.convert_images(target_images)
                except IndexError:
                    continue
                except OSError:
                    logger.warning(f"Skipped bad example {raw_example['key']}.")  # DL3DV-Full have some bad images
                    continue

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
                target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    logger.warning(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                target_extrinsics = extrinsics[target_indices]
                num_context_images = len(context_extrinsics)
                all_used_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)

                scale = compute_pose_norm_scale(context_extrinsics, self.cfg.pose_norm_method)
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

                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    logger.warning(
                        f"Skipped {scene} because of baseline out of range: "
                        f"{scale:.6f}"
                    )
                    continue
                all_used_extrinsics[:, :3, 3] /= scale

                if self.cfg.relative_pose:
                    all_used_extrinsics = camera_normalization(all_used_extrinsics[0:1], all_used_extrinsics)

                index_path = getattr(getattr(self.view_sampler, "cfg", None), "index_path", None)
                scale_value = (
                    float(scale.detach().item())
                    if isinstance(scale, torch.Tensor)
                    else float(scale)
                )

                example = {
                    "context": {
                        "extrinsics": all_used_extrinsics[:num_context_images],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / scale,
                        "far": self.get_bound("far", len(context_indices)) / scale,
                        "index": context_indices,
                        "overlap": overlap,
                    },
                    "target": {
                        "extrinsics": all_used_extrinsics[num_context_images:],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / scale,
                        "far": self.get_bound("far", len(target_indices)) / scale,
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
                        "index_path": "" if index_path is None else str(index_path),
                        "num_context_views": torch.tensor(int(num_context_images), dtype=torch.int64),
                    },
                }
                should_load_valid_mask = (
                    (self.cfg.load_valid_mask or self.cfg.apply_valid_mask)
                    and "valid_masks" in raw_example
                )
                if should_load_valid_mask:
                    example["context"]["valid_mask"] = self.convert_valid_masks(
                        [raw_example["valid_masks"][index] for index in context_index_list]
                    )
                    example["target"]["valid_mask"] = self.convert_valid_masks(
                        [raw_example["valid_masks"][index] for index in target_index_list]
                    )
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                example = apply_configured_crop_shim(
                    example,
                    tuple(self.cfg.input_image_shape),
                    self.cfg.crop_mode,
                )
                if "valid_mask" in example["context"]:
                    example["context"] = self.resize_valid_masks_to_views(example["context"])
                    example["target"] = self.resize_valid_masks_to_views(example["target"])
                    if self.cfg.apply_valid_mask:
                        example["context"] = self.apply_valid_masks_to_views(example["context"])
                        example["target"] = self.apply_valid_masks_to_views(example["target"])
                yield example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_valid_masks(
        self,
        masks: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 1 height width"]:
        torch_masks = []
        for mask in masks:
            mask_image = Image.open(BytesIO(mask.numpy().tobytes())).convert("L")
            torch_masks.append(self.to_tensor(mask_image))
        return torch.stack(torch_masks)

    def apply_valid_masks(
        self,
        images: Float[Tensor, "batch 3 height width"],
        masks: Float[Tensor, "batch 1 height width"],
    ) -> Float[Tensor, "batch 3 height width"]:
        background = torch.tensor(
            self.cfg.background_color,
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        masks = masks.to(dtype=images.dtype, device=images.device)
        return images * masks + background * (1 - masks)

    def apply_valid_masks_to_views(self, views):
        mask = views["valid_mask"]
        return {
            **views,
            "image": self.apply_valid_masks(views["image"], mask),
            "valid_mask": mask,
        }

    def resize_valid_masks_to_views(self, views):
        return {
            **views,
            "valid_mask": self.resize_valid_masks(
                views["valid_mask"],
                views["image"].shape[-2:],
            ),
        }

    def resize_valid_masks(
        self,
        masks: Float[Tensor, "batch 1 h_in w_in"],
        shape: tuple[int, int],
    ) -> Float[Tensor, "batch 1 h_out w_out"]:
        *_, h_in, w_in = masks.shape
        h_out, w_out = shape
        assert h_out <= h_in and w_out <= w_in

        scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_out or w_scaled == w_out

        *batch, c, h, w = masks.shape
        masks = masks.reshape(-1, c, h, w)
        masks = F.interpolate(masks, size=(h_scaled, w_scaled), mode="nearest")
        masks = masks.reshape(*batch, c, h_scaled, w_scaled)

        row = (h_scaled - h_out) // 2
        col = (w_scaled - w_out) // 2
        return masks[..., :, row : row + h_out, col : col + w_out]

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, "view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
    def test_len(self) -> int:
        if self.stage == "test":
            return len(self.view_sampler.index.keys())
        else:
            return 0
