from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    overlap: Float[Tensor, "batch _"]  # batch view
    valid_mask: Float[Tensor, "batch _ 1 height width"]


class BatchedSceneMetadata(TypedDict, total=False):
    mesh_space: list[str] | str
    world_space: list[str] | str
    pose_norm_scale: Float[Tensor, "batch"]
    pose_norm_method: list[str] | str
    relative_pose: Tensor
    world_from_normalized: Float[Tensor, "batch 4 4"]
    normalized_from_world: Float[Tensor, "batch 4 4"]
    context_indices: Int64[Tensor, "batch _"]
    target_indices: Int64[Tensor, "batch _"]
    index_path: list[str] | str
    num_context_views: Int64[Tensor, "batch"]


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]
    scene_metadata: BatchedSceneMetadata


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]
    valid_mask: Float[Tensor, "_ 1 height width"]


class UnbatchedSceneMetadata(TypedDict, total=False):
    mesh_space: str
    world_space: str
    pose_norm_scale: Float[Tensor, ""]
    pose_norm_method: str
    relative_pose: Tensor
    world_from_normalized: Float[Tensor, "4 4"]
    normalized_from_world: Float[Tensor, "4 4"]
    context_indices: Int64[Tensor, "_"]
    target_indices: Int64[Tensor, "_"]
    index_path: str
    num_context_views: Int64[Tensor, ""]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str
    scene_metadata: UnbatchedSceneMetadata


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
