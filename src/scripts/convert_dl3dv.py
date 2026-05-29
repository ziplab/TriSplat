import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

INPUT_IMAGE_DIR = Path("/users/bye/scratch/dl3dv/DL3DV-ALL-480P")
OUTPUT_DIR = Path("/users/bye/scratch/dl3dv/DL3DV-ALL-480P/Chunks")


# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    subsets = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K']
    keys = []
    for subset in subsets:
        subdir = INPUT_IMAGE_DIR / subset
        # iterate through all the subdirectories
        for key in subdir.iterdir():
            if key.is_dir():
                item = key.name.split('/')[-1]
                item = '/'.join([subset, item])
                print(item)
                keys.append(item)

    keys.sort()
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[str, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {path.stem: load_raw(path) for path in example_path.iterdir()}


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c_opencv = np.linalg.inv(c2w)
    return w2c_opencv


def load_metadata(file_path: Path) -> Metadata:
    with open(file_path, 'r') as file:
        data = json.load(file)

    url = ""

    timestamps = []
    cameras = []

    # FIXME: igore k1, k2, p1, p2, is this proper?
    w = data['w']
    h = data['h']
    intrinsic = [data['fl_x'] / w, data['fl_y'] / h, data['cx'] / w, data['cy'] / h, 0.0, 0.0]
    intrinsic = np.array(intrinsic, dtype=np.float32)

    for frame in data['frames']:
        # extract number from string like "images/frame_00002.png"
        frame_id = int(frame['file_path'].split('_')[-1].split('.')[0])
        timestamps.append(frame_id)
        extrinsic = frame['transform_matrix']
        extrinsic = np.array(extrinsic, dtype=np.float32)
        w2c = opengl_c2w_to_opencv_w2c(extrinsic)
        w2c = w2c[:3, :]
        w2c = w2c.flatten()
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    # for stage in ("train", "test"):
    for stage in ["train"]:
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / key / 'images_8'
            metadata_file = INPUT_IMAGE_DIR / key / 'transforms.json'
            num_bytes = get_size(image_dir)

            if not image_dir.exists() or not metadata_file.exists():
                print(f"Skipping {key} because it is missing.")
                continue

            # Read images and metadata.
            images = load_images(image_dir)
            example = load_metadata(metadata_file)

            # Merge the images into the example.
            # from int to "frame_00001" format
            image_names = [f"frame_{timestamp.item():0>5}" for timestamp in example["timestamps"]]
            try:
                example["images"] = [
                    images[image_name] for image_name in image_names
                ]
            except KeyError:
                print(f"Skipping {key} because of missing images.")
                continue
            assert len(example["images"]) == len(example["timestamps"]), f"len(example['images'])={len(example['images'])}, len(example['timestamps'])={len(example['timestamps'])}"

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
