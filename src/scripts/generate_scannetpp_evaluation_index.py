import argparse
import json
import os
from pathlib import Path

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from src.geometry.epipolar_lines import project_rays
from src.geometry.projection import get_world_rays, sample_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local ScanNet++ evaluation index that matches the converted .torch dataset split.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("SCANNETPP_IPHONE_ROOT", "data/scannetpp/iphone"),
        help="Root containing train/test .torch chunks.",
    )
    parser.add_argument(
        "--stage",
        default="test",
        choices=("train", "test"),
        help="Dataset split used to build the evaluation index.",
    )
    parser.add_argument(
        "--output-path",
        default="assets/evaluation_index_scannetpp_iphone_local.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--num-target-views",
        type=int,
        default=3,
        help="Number of target views sampled between the two context endpoints.",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=5,
        help="Minimum frame gap between the two stored context endpoints.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=15,
        help="Maximum frame gap between the two stored context endpoints.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.6,
        help="Minimum bidirectional image overlap to accept a context pair.",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=1.0,
        help="Maximum bidirectional image overlap to accept a context pair.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Square ray grid resolution used for overlap estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used to pick context and target views.",
    )
    return parser.parse_args()


def convert_poses(
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _ = poses.shape

    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=batch_size).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=batch_size).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def compute_overlap(
    context_index: int,
    other_index: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    xy: torch.Tensor,
) -> tuple[float, float, float]:
    context_origins, context_directions = get_world_rays(
        xy,
        extrinsics[context_index],
        intrinsics[context_index],
    )
    other_origins, other_directions = get_world_rays(
        xy,
        extrinsics[other_index],
        intrinsics[other_index],
    )

    projection_onto_other = project_rays(
        context_origins,
        context_directions,
        extrinsics[other_index],
        intrinsics[other_index],
    )
    projection_onto_context = project_rays(
        other_origins,
        other_directions,
        extrinsics[context_index],
        intrinsics[context_index],
    )

    overlap_a = projection_onto_context["overlaps_image"].float().mean().item()
    overlap_b = projection_onto_other["overlaps_image"].float().mean().item()
    return min(overlap_a, overlap_b), overlap_a, overlap_b


def sample_scene_entry(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    generator: torch.Generator,
    num_target_views: int,
    min_distance: int,
    max_distance: int,
    min_overlap: float,
    max_overlap: float,
    xy: torch.Tensor,
) -> dict | None:
    num_views = extrinsics.shape[0]
    context_order = torch.randperm(num_views, generator=generator).tolist()

    for context_index in context_order:
        valid_indices: list[tuple[int, float]] = []
        for step in (1, -1):
            current_index = context_index + step * min_distance
            while 0 <= current_index < num_views:
                overlap, _, _ = compute_overlap(
                    context_index,
                    current_index,
                    extrinsics,
                    intrinsics,
                    xy,
                )
                distance = abs(current_index - context_index)
                if min_overlap <= overlap <= max_overlap:
                    valid_indices.append((current_index, overlap))
                if overlap < min_overlap or distance > max_distance:
                    break
                current_index += step

        if not valid_indices:
            continue

        chosen_index = torch.randint(
            len(valid_indices),
            size=(),
            generator=generator,
        ).item()
        partner_index, overlap = valid_indices[chosen_index]
        context_left = min(context_index, partner_index)
        context_right = max(context_index, partner_index)

        candidate_count = context_right - context_left + 1
        if candidate_count < num_target_views:
            continue

        while True:
            target_views = torch.randint(
                context_left,
                context_right + 1,
                (num_target_views,),
                generator=generator,
            )
            if len(target_views.unique()) == num_target_views:
                break

        return {
            "context": [context_left, context_right],
            "target": sorted(target_views.tolist()),
            "overlap": round(float(overlap), 6),
        }

    return None


def resolve_stage_root(dataset_root: Path, stage: str) -> Path:
    stage_root = dataset_root / stage
    return stage_root if stage_root.exists() else dataset_root


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    stage_root = resolve_stage_root(dataset_root, args.stage)
    output_path = Path(args.output_path).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()

    chunk_paths = sorted(stage_root.glob("*.torch"))
    if not chunk_paths:
        raise FileNotFoundError(f"No .torch chunks found under {stage_root}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    xy, _ = sample_image_grid((args.grid_size, args.grid_size))
    xy = rearrange(xy, "h w xy -> (h w) xy")

    index: dict[str, dict | None] = {}
    for chunk_path in tqdm(chunk_paths, desc="Chunks"):
        chunk = torch.load(chunk_path, map_location="cpu")
        for example in tqdm(chunk, desc=chunk_path.name, leave=False):
            extrinsics, intrinsics = convert_poses(example["cameras"].to(torch.float32))
            index[example["key"]] = sample_scene_entry(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                generator=generator,
                num_target_views=args.num_target_views,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                min_overlap=args.min_overlap,
                max_overlap=args.max_overlap,
                xy=xy,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(index, f, indent=2)

    valid_count = sum(entry is not None for entry in index.values())
    print(f"Wrote {output_path}")
    print(f"Scenes: {len(index)}, valid entries: {valid_count}")


if __name__ == "__main__":
    main()
