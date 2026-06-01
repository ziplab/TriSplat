from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from dacite import Config, from_dict
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.checkpoint_utils import (
    extract_state_dict,
    get_checkpoint_schedule_step,
    load_checkpoint_file,
)

if TYPE_CHECKING:
    from src.dataset.types import BatchedExample
    from src.mesh.tsdf_gs2d import TsdfGs2dCfgWrapper
    from src.model.encoder.encoder_trisplat import EncoderTrisplatCfg


DEFAULT_EXPERIMENT = "trisplat_re10k_triangle_refiner_unet_10m_wide"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TriSplat pose-free mesh inference on a folder of raw images "
            "without Lightning Trainer or DataModules."
        ),
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--ckpt", required=True, help="TriSplat checkpoint path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Hydra experiment config name. Defaults to {DEFAULT_EXPERIMENT}.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="Number of images to select when --view-indices is not set.",
    )
    parser.add_argument(
        "--selection",
        choices=("uniform", "first"),
        default="uniform",
        help="Image selection strategy when --view-indices is not set.",
    )
    parser.add_argument(
        "--view-indices",
        default=None,
        help="Comma-separated zero-based image indices to use, e.g. 0,5,10.",
    )
    parser.add_argument(
        "--image-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Network input size. Defaults to the experiment dataset input_image_shape.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=60.0,
        help="Horizontal FOV used to create default normalized intrinsics.",
    )
    parser.add_argument(
        "--intrinsics-json",
        default=None,
        help=(
            "Optional JSON with a normalized 3x3 K matrix or a list/dict of per-view "
            "normalized 3x3 matrices."
        ),
    )
    parser.add_argument("--near", type=float, default=0.1, help="Near plane metadata.")
    parser.add_argument("--far", type=float, default=100.0, help="Far plane metadata.")
    parser.add_argument("--scene", default=None, help="Optional scene name for metadata.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Defaults to cuda; falls back to cpu if CUDA is unavailable.",
    )
    parser.add_argument(
        "--schedule-step",
        type=int,
        default=None,
        help="Opacity/scale schedule step. Defaults to the checkpoint global_step or filename step.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def resolve_path(path_str: str, root: Path, kind: str, must_exist: bool = True) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def collect_image_paths(image_dir: Path) -> list[Path]:
    paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    paths = sorted(paths, key=natural_key)
    if not paths:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise FileNotFoundError(f"No images with suffixes [{suffixes}] found in {image_dir}")
    return paths


def parse_view_indices(indices: str, num_images: int) -> list[int]:
    parsed = [int(part.strip()) for part in indices.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--view-indices did not contain any indices.")
    bad = [index for index in parsed if index < 0 or index >= num_images]
    if bad:
        raise ValueError(
            f"--view-indices contains out-of-range entries {bad}; "
            f"image folder has {num_images} images."
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError("--view-indices must not contain duplicates.")
    return parsed


def select_image_indices(
    num_images: int,
    *,
    view_indices: str | None,
    num_views: int,
    selection: str,
) -> list[int]:
    if view_indices is not None:
        return parse_view_indices(view_indices, num_images)
    if num_views <= 0:
        raise ValueError("--num-views must be positive.")
    if num_views > num_images:
        raise ValueError(
            f"Requested {num_views} views, but image folder only contains {num_images} images."
        )
    if selection == "first":
        return list(range(num_views))
    if selection == "uniform":
        if num_views == 1:
            return [0]
        return np.linspace(0, num_images - 1, num_views).round().astype(np.int64).tolist()
    raise ValueError(f"Unsupported selection strategy: {selection}")


def resize_center_crop(image: Image.Image, shape: tuple[int, int]) -> Image.Image:
    out_h, out_w = shape
    in_w, in_h = image.size
    scale = max(out_h / in_h, out_w / in_w)
    scaled_w = max(int(round(in_w * scale)), out_w)
    scaled_h = max(int(round(in_h * scale)), out_h)
    image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
    left = (scaled_w - out_w) // 2
    top = (scaled_h - out_h) // 2
    return image.crop((left, top, left + out_w, top + out_h))


def load_images(paths: Sequence[Path], shape: tuple[int, int]) -> torch.Tensor:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = resize_center_crop(image, shape)
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(images, dim=0)


def make_default_intrinsics(num_views: int, image_shape: tuple[int, int], fov_deg: float) -> torch.Tensor:
    if not (0.0 < fov_deg < 180.0):
        raise ValueError("--fov-deg must be in (0, 180).")
    height, width = image_shape
    fov_rad = math.radians(float(fov_deg))
    fx_px = 0.5 * width / math.tan(0.5 * fov_rad)
    fy_px = fx_px
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num_views, 1, 1)
    intrinsics[:, 0, 0] = fx_px / width
    intrinsics[:, 1, 1] = fy_px / height
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    return intrinsics


def _intrinsics_from_payload(payload: Any, selected_indices: Sequence[int]) -> torch.Tensor:
    if isinstance(payload, dict):
        if "intrinsics" in payload:
            payload = payload["intrinsics"]
        elif "K" in payload:
            payload = payload["K"]
        else:
            matrices = []
            for index in selected_indices:
                key = str(index)
                if key not in payload:
                    raise KeyError(f"Missing intrinsics entry for selected image index {index!r}.")
                matrices.append(payload[key])
            return torch.as_tensor(matrices, dtype=torch.float32)

    tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.shape == (3, 3):
        return tensor.unsqueeze(0).repeat(len(selected_indices), 1, 1)
    if tensor.dim() == 3 and tensor.shape[-2:] == (3, 3):
        if tensor.shape[0] == len(selected_indices):
            return tensor
        if tensor.shape[0] > max(selected_indices):
            return tensor[list(selected_indices)]
    raise ValueError(
        "Intrinsics JSON must be a 3x3 matrix, a selected-view list of 3x3 matrices, "
        "a full list indexed by input image index, or a dict keyed by image index."
    )


def load_intrinsics(
    intrinsics_json: Path | None,
    *,
    selected_indices: Sequence[int],
    image_shape: tuple[int, int],
    fov_deg: float,
) -> torch.Tensor:
    if intrinsics_json is None:
        return make_default_intrinsics(len(selected_indices), image_shape, fov_deg)
    with intrinsics_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    intrinsics = _intrinsics_from_payload(payload, selected_indices)
    if intrinsics.shape != (len(selected_indices), 3, 3):
        raise ValueError(
            "Loaded intrinsics have shape "
            f"{tuple(intrinsics.shape)}, expected {(len(selected_indices), 3, 3)}."
        )
    return intrinsics.float()


def resolve_dataset_cfg(cfg: DictConfig) -> DictConfig:
    dataset = cfg.get("dataset")
    if not isinstance(dataset, DictConfig) or len(dataset) == 0:
        raise ValueError("Experiment config does not contain a dataset section.")
    first_key = next(iter(dataset.keys()))
    return dataset[first_key]


def compose_cfg(args: argparse.Namespace, root: Path) -> DictConfig:
    overrides = [
        f"+experiment={args.experiment}",
        "mode=test",
        "mesh.tsdf_gs2d.export_mode=direct",
        "wandb.mode=disabled",
        *args.override,
    ]
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        return compose(config_name="main", overrides=overrides)


def typed_cfgs(cfg: DictConfig) -> tuple[EncoderTrisplatCfg, TsdfGs2dCfgWrapper]:
    from src.mesh.tsdf_gs2d import TsdfGs2dCfgWrapper
    from src.model.encoder.encoder_trisplat import EncoderTrisplatCfg

    type_cfg = Config(cast=[tuple])
    encoder_cfg = from_dict(
        EncoderTrisplatCfg,
        OmegaConf.to_container(cfg.model.encoder, resolve=True),
        config=type_cfg,
    )
    mesh_cfg = from_dict(
        TsdfGs2dCfgWrapper,
        OmegaConf.to_container(cfg.mesh, resolve=True),
        config=type_cfg,
    )
    return encoder_cfg, mesh_cfg


def load_encoder_weights(
    encoder: torch.nn.Module,
    ckpt_path: Path,
) -> tuple[list[str], list[str], int]:
    checkpoint = load_checkpoint_file(ckpt_path)
    schedule_step = get_checkpoint_schedule_step(checkpoint, ckpt_path)
    state_dict = extract_state_dict(checkpoint)
    if not state_dict:
        raise ValueError(f"Checkpoint contains an empty state_dict: {ckpt_path}")
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        encoder_state = state_dict
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    loaded_key_count = len(encoder_state) - len(incompatible.unexpected_keys)
    if loaded_key_count <= 0:
        raise ValueError(
            "Checkpoint did not contain any weights that match the encoder. "
            f"Check that --ckpt points to a TriSplat checkpoint: {ckpt_path}"
        )
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys), schedule_step


def build_batch(
    *,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    selected_indices: Sequence[int],
    scene: str,
    near: float,
    far: float,
) -> BatchedExample:
    num_views = images.shape[0]
    c2w_identity = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_views, 1, 1)
    near_tensor = torch.full((num_views,), float(near), dtype=torch.float32)
    far_tensor = torch.full((num_views,), float(far), dtype=torch.float32)
    index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)

    views = {
        "extrinsics": c2w_identity.unsqueeze(0),
        "intrinsics": intrinsics.unsqueeze(0),
        "image": images.unsqueeze(0),
        "near": near_tensor.unsqueeze(0),
        "far": far_tensor.unsqueeze(0),
        "index": index_tensor.unsqueeze(0),
    }
    return {
        "context": dict(views),
        "target": dict(views),
        "scene": [scene],
        "scene_metadata": {
            "mesh_space": "predicted_pose_free",
            "world_space": "predicted_first_view",
            "pose_norm_method": "none",
            "relative_pose": torch.tensor(True, dtype=torch.bool),
            "context_indices": index_tensor,
            "target_indices": index_tensor,
            "index_path": "",
            "num_context_views": torch.tensor(num_views, dtype=torch.int64),
        },
    }


def move_batch_to_device(batch: BatchedExample, device: torch.device) -> BatchedExample:
    moved: BatchedExample = {"scene": batch["scene"], "scene_metadata": {}}
    for split in ("context", "target"):
        moved[split] = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch[split].items()
        }
    for key, value in batch["scene_metadata"].items():
        moved["scene_metadata"][key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def tensor_to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): tensor_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(tensor_to_jsonable(payload), f, indent=2)


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --force to write into it.")
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    from src.mesh import get_mesh
    from src.model.encoder import get_encoder
    from src.model.types import Triangles

    args = parse_args()
    root = repo_root()
    image_dir = resolve_path(args.image_dir, root, "Image directory")
    ckpt_path = resolve_path(args.ckpt, root, "Checkpoint")
    out_dir = resolve_path(args.out_dir, root, "Output directory", must_exist=False)
    intrinsics_path = (
        None
        if args.intrinsics_json is None
        else resolve_path(args.intrinsics_json, root, "Intrinsics JSON")
    )

    prepare_output_dir(out_dir, args.force)

    cfg = compose_cfg(args, root)
    encoder_cfg, mesh_cfg = typed_cfgs(cfg)
    if not encoder_cfg.pose_free:
        raise ValueError("Custom image inference requires model.encoder.pose_free=true.")
    if not encoder_cfg.use_triangle:
        raise ValueError("Direct mesh export requires a triangle TriSplat checkpoint/config.")
    if mesh_cfg.tsdf_gs2d.export_mode != "direct":
        raise ValueError("Custom image inference currently supports direct mesh export only.")
    mesh_cfg.tsdf_gs2d.direct_surface_proxy = False
    mesh_cfg.tsdf_gs2d.direct_write_frame_meshes = False
    mesh_cfg.tsdf_gs2d.direct_surface_proxy_write_frame_meshes = False

    dataset_cfg = resolve_dataset_cfg(cfg)
    image_shape = tuple(args.image_shape or dataset_cfg.input_image_shape)
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    image_paths = collect_image_paths(image_dir)
    default_num_views = int(dataset_cfg.view_sampler.num_context_views)
    num_views = int(args.num_views or default_num_views)
    selected_indices = select_image_indices(
        len(image_paths),
        view_indices=args.view_indices,
        num_views=num_views,
        selection=args.selection,
    )
    selected_paths = [image_paths[index] for index in selected_indices]
    scene = args.scene or image_dir.name

    device = resolve_device(args.device)
    images = load_images(selected_paths, image_shape)
    intrinsics = load_intrinsics(
        intrinsics_path,
        selected_indices=selected_indices,
        image_shape=image_shape,
        fov_deg=args.fov_deg,
    )
    batch_cpu = build_batch(
        images=images,
        intrinsics=intrinsics,
        selected_indices=selected_indices,
        scene=scene,
        near=args.near,
        far=args.far,
    )

    encoder, _ = get_encoder(encoder_cfg)
    missing_keys, unexpected_keys, ckpt_schedule_step = load_encoder_weights(encoder, ckpt_path)
    schedule_step = int(args.schedule_step if args.schedule_step is not None else ckpt_schedule_step)
    encoder = encoder.to(device).eval()
    mesh = get_mesh(mesh_cfg)

    mesh_batch = move_batch_to_device(batch_cpu, device)
    encoder_batch = encoder.get_data_shim()(
        {
            "context": dict(mesh_batch["context"]),
            "target": dict(mesh_batch["target"]),
            "scene": list(mesh_batch["scene"]),
            "scene_metadata": dict(mesh_batch["scene_metadata"]),
        }
    )

    visualization_dump: dict[str, Any] = {"collect_geom_normal": True}
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        primitives = encoder(encoder_batch["context"], schedule_step, visualization_dump=visualization_dump)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    encoder_time = time.perf_counter() - t0
    if not isinstance(primitives, Triangles):
        raise TypeError("Expected triangle primitives from the encoder.")
    pred_c2w = visualization_dump.get("c2w")
    if pred_c2w is not None:
        pred_c2w = pred_c2w.detach().float()
        mesh_batch["context"]["extrinsics"] = pred_c2w
        mesh_batch["target"]["extrinsics"] = pred_c2w

    opacity_kwargs = {}
    decoder_cfg = cfg.model.decoder
    opacity_kwargs = {
        "global_step": schedule_step,
        "opacity_temp_initial": float(decoder_cfg.get("opacity_temp_initial", 1.0)),
        "opacity_temp_final": float(decoder_cfg.get("opacity_temp_final", 1.0)),
        "opacity_temp_warmup_steps": int(decoder_cfg.get("opacity_temp_warmup_steps", 5000)),
    }

    mesh_result = mesh.main(
        None,
        None,
        mesh_batch,
        primitives,
        str(out_dir / "mesh"),
        encoder_time=encoder_time,
        decoder_time=None,
        **opacity_kwargs,
    )

    pred_intrinsics = visualization_dump.get("intrinsic_pred")
    if pred_c2w is not None:
        pred_c2w_np = pred_c2w.cpu().numpy()[0]
        np.save(out_dir / "predicted_c2w.npy", pred_c2w_np)
        write_json(out_dir / "predicted_c2w.json", pred_c2w_np)
    np.save(out_dir / "intrinsics.npy", intrinsics.numpy())
    if pred_intrinsics is not None:
        pred_intrinsics_np = pred_intrinsics.detach().float().cpu().numpy()
        np.save(out_dir / "predicted_intrinsics_focal.npy", pred_intrinsics_np)

    write_json(
        out_dir / "selected_images.json",
        {
            "image_dir": str(image_dir),
            "selected_indices": selected_indices,
            "selected_paths": [str(path) for path in selected_paths],
            "image_shape": image_shape,
            "fov_deg": args.fov_deg,
            "intrinsics_source": str(intrinsics_path) if intrinsics_path is not None else "default_fov",
        },
    )
    OmegaConf.save(cfg, out_dir / "run_config.yaml")
    write_json(
        out_dir / "inference_summary.json",
        {
            "scene": scene,
            "checkpoint": str(ckpt_path),
            "experiment": args.experiment,
            "device": str(device),
            "schedule_step": schedule_step,
            "encoder_time_sec": encoder_time,
            "mesh_output_path": mesh_result.output_path,
            "mesh_space_metadata_path": mesh_result.space_metadata_path,
            "direct_timing": mesh_result.direct_timing,
            "missing_encoder_keys": missing_keys,
            "unexpected_encoder_keys": unexpected_keys,
        },
    )

    print(f"Saved mesh: {mesh_result.output_path}")
    if pred_c2w is not None:
        print(f"Saved predicted poses: {out_dir / 'predicted_c2w.npy'}")
    print(f"Saved summary: {out_dir / 'inference_summary.json'}")


if __name__ == "__main__":
    main()
