import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as tf
from PIL import Image

from src.dataset.shims.crop_shim_img import rescale_and_crop_img
from src.misc.image_io import prep_image


DEFAULT_ASSET_PATH = Path("assets/evaluation_index_re10k_mesh_6ctx_selected6.json")
DEFAULT_DATA_ROOT = Path(os.environ.get("RE10K_ROOT", "data/re10k"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save context images referenced by a Re10K evaluation asset.",
    )
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=DEFAULT_ASSET_PATH,
        help="Evaluation asset JSON containing scene entries with context indices.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Re10K dataset root, or the stage directory containing index.json.",
    )
    parser.add_argument(
        "--stage",
        default="test",
        help="Dataset stage to read when --data-root is the dataset root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/asset_context_images/<asset stem>.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="Image format for saved context frames.",
    )
    parser.add_argument(
        "--crop-shape",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optionally save images after the same resize-and-center-crop used for model inputs.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip scenes missing from the dataset index instead of failing.",
    )
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path, kind: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def resolve_stage_root(data_root: Path, stage: str) -> Path:
    data_root = data_root.expanduser().resolve()
    staged_root = data_root / stage
    if (staged_root / "index.json").is_file():
        return staged_root
    if (data_root / "index.json").is_file():
        return data_root
    raise FileNotFoundError(
        f"Could not find index.json in {staged_root} or {data_root}."
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk(path: Path) -> list[dict[str, Any]]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def decode_image(raw_image: torch.Tensor) -> Image.Image:
    raw_bytes = raw_image.cpu().numpy().tobytes()
    with Image.open(BytesIO(raw_bytes)) as image:
        return image.convert("RGB")


def crop_image_to_input(image: Image.Image, shape: tuple[int, int]) -> Image.Image:
    image_tensor = tf.ToTensor()(image)
    cropped = rescale_and_crop_img(image_tensor, shape)
    return Image.fromarray(prep_image(cropped))


def save_context_images(
    *,
    asset_path: Path,
    stage_root: Path,
    out_dir: Path,
    image_format: str,
    crop_shape: tuple[int, int] | None,
    skip_missing: bool,
) -> dict[str, Any]:
    asset = load_json(asset_path)
    if not isinstance(asset, dict):
        raise TypeError(f"Expected dict asset JSON, got {type(asset)} from {asset_path}")

    dataset_index = load_json(stage_root / "index.json")
    if not isinstance(dataset_index, dict):
        raise TypeError(f"Expected dict dataset index in {stage_root / 'index.json'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "asset_path": str(asset_path),
        "stage_root": str(stage_root),
        "out_dir": str(out_dir),
        "image_format": image_format,
        "crop_shape": list(crop_shape) if crop_shape is not None else None,
        "scenes": {},
        "skipped_scenes": [],
    }

    chunk_cache: dict[Path, list[dict[str, Any]]] = {}
    suffix = "jpg" if image_format == "jpg" else "png"

    for scene, entry in asset.items():
        if entry is None:
            manifest["skipped_scenes"].append(
                {"scene": scene, "reason": "null asset entry"}
            )
            continue
        if not isinstance(entry, dict) or "context" not in entry:
            raise ValueError(f"Scene {scene} does not contain a context entry.")

        context_indices = [int(index) for index in entry["context"]]
        chunk_rel = dataset_index.get(scene)
        if chunk_rel is None:
            if skip_missing:
                manifest["skipped_scenes"].append(
                    {"scene": scene, "reason": "missing from dataset index"}
                )
                continue
            raise KeyError(f"Scene {scene} is missing from {stage_root / 'index.json'}")

        chunk_path = Path(chunk_rel)
        if not chunk_path.is_absolute():
            chunk_path = stage_root / chunk_path
        chunk_path = chunk_path.resolve()
        if chunk_path not in chunk_cache:
            chunk_cache[chunk_path] = load_chunk(chunk_path)

        examples = [example for example in chunk_cache[chunk_path] if example["key"] == scene]
        if len(examples) != 1:
            raise ValueError(
                f"Expected exactly one example for scene {scene} in {chunk_path}, "
                f"found {len(examples)}."
            )
        example = examples[0]

        scene_out_dir = out_dir / scene
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for rank, frame_index in enumerate(context_indices):
            try:
                image = decode_image(example["images"][frame_index])
            except IndexError as exc:
                raise IndexError(
                    f"Context frame {frame_index} is out of range for scene {scene}."
                ) from exc
            if crop_shape is not None:
                image = crop_image_to_input(image, crop_shape)

            image_path = scene_out_dir / (
                f"context_{rank:02d}_frame_{frame_index:06d}.{suffix}"
            )
            image.save(image_path)
            saved_paths.append(str(image_path))

        manifest["scenes"][scene] = {
            "chunk_path": str(chunk_path),
            "context": context_indices,
            "saved_paths": saved_paths,
        }

    manifest["num_scenes"] = len(manifest["scenes"])
    manifest["num_images"] = sum(
        len(scene["saved_paths"]) for scene in manifest["scenes"].values()
    )
    manifest_path = out_dir / "context_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    asset_path = resolve_path(args.asset_path, repo_root, "Asset")
    stage_root = resolve_stage_root(args.data_root, args.stage)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = repo_root / "outputs" / "asset_context_images" / asset_path.stem
    elif not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir = out_dir.expanduser().resolve()

    manifest = save_context_images(
        asset_path=asset_path,
        stage_root=stage_root,
        out_dir=out_dir,
        image_format=args.image_format,
        crop_shape=tuple(args.crop_shape) if args.crop_shape is not None else None,
        skip_missing=args.skip_missing,
    )
    print(
        json.dumps(
            {
                "out_dir": manifest["out_dir"],
                "manifest_path": manifest["manifest_path"],
                "num_scenes": manifest["num_scenes"],
                "num_images": manifest["num_images"],
                "num_skipped_scenes": len(manifest["skipped_scenes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
