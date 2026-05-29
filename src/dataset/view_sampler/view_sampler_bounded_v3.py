'''
Modifiedy from latentSplat and pixelSplat to handle extrapolate and more context views
'''

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


@dataclass
class ViewSamplerBoundedV3Cfg:
    name: Literal["boundedv3"]
    num_context_views: int | list[int]
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    max_distance_to_context_views: int
    context_gap_warm_up_steps: int
    target_gap_warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    initial_max_distance_to_context_views: int
    hard_max_distance_between_context_views: int
    extra_views_sampling_strategy: Optional[Literal["random", "farthest_point", "equal"]] = "random"
    target_views_replace_sample: Optional[bool] = True
    permute_context_views: Optional[bool] = False
    max_img_per_gpu: int = 48


class ViewSamplerBoundedV3(ViewSampler[ViewSamplerBoundedV3Cfg]):
    def schedule(self, initial: int, final: int, steps: int, scale_factor: int, hard_max_distance_between_context_views: int) -> int:
        fraction = self.global_step / steps
        initial = initial * scale_factor
        final = final * scale_factor
        return min(initial + int((final - initial) * fraction), final, hard_max_distance_between_context_views)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        max_num_views: Optional[int] = None,
        num_context_views: Optional[int] = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape
        if max_num_views is not None:
            num_views = min(num_views, max_num_views)

        # determine number of context views if a range is given
        if num_context_views is None:
            if isinstance(self.cfg.num_context_views, list):
                assert len(self.cfg.num_context_views) == 2
                # To ensure all GPUs get the same number of views, we seed a generator
                # with the global step.
                generator = torch.Generator(device=device)
                generator.manual_seed(self.global_step)
                num_context_views = torch.randint(
                    self.cfg.num_context_views[0],
                    self.cfg.num_context_views[1] + 1,
                    size=(),
                    generator=generator,
                    device=device,
                ).item()
            else:
                num_context_views = self.cfg.num_context_views

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
                num_context_views - 1,
                self.cfg.hard_max_distance_between_context_views,
            )
            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
                num_context_views - 1,
                self.cfg.hard_max_distance_between_context_views,
            )
        else:
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_context_gap = min(num_views - 1, max_context_gap)
        min_context_gap = max(2 * self.cfg.min_distance_to_context_views, min_context_gap)
        if max_context_gap < min_context_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_context_gap,
            max_context_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Compute the margin from context window to target window based on the current global step
        # This is used for extrapolation during training
        if self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps,
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_context_gap

        index_target_left = index_context_left - max_target_gap
        index_target_right = index_context_right + max_target_gap

        if not self.cameras_are_circular:
            index_target_left = max(0, index_target_left)
            index_target_right = min(num_views - 1, index_target_right)

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_target_left,
                index_target_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            # NOTE: it is OK to occasionally pick the train views
            if self.cfg.target_views_replace_sample:
                index_target = torch.randint(
                    index_target_left,
                    index_target_right + 1,
                    size=(self.cfg.num_target_views,),
                    device=device,
                )
            else:  # sample without replacement
                # similarly, ok to index
                index_target_candidates = torch.arange(
                    index_target_left,
                    index_target_right + 1,
                    device=device,
                )
                indices = torch.randperm(index_target_right + 1 - index_target_left, device=device)[
                    : self.cfg.num_target_views
                ]
                index_target = index_target_candidates[indices]

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.
        if num_context_views > 2:
            num_extra_views = num_context_views - 2
            extra_views = []
            if self.cfg.extra_views_sampling_strategy == 'random':
                while len(set(extra_views)) != num_extra_views:
                    extra_views = torch.randint(
                        index_context_left + 1,
                        index_context_right,
                        (num_extra_views,),
                    ).tolist()
            elif self.cfg.extra_views_sampling_strategy == 'farthest_point':
                context_bounded_index = torch.arange(index_context_left, index_context_right + 1)
                candidate_views_position = extrinsics[context_bounded_index, :3, -1].unsqueeze(0)
                index_context_local = farthest_point_sample(
                    candidate_views_position, num_context_views
                ).squeeze(0)
                # remap context index back to global scene based index
                index_context = context_bounded_index[index_context_local]
                index_context_left = index_context[0].item()
                index_context_right = index_context[-1].item()
                extra_views = index_context[1:-1].tolist()
            elif self.cfg.extra_views_sampling_strategy == 'equal':
                pass
        else:
            extra_views = []

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        context_views = torch.tensor((index_context_left, *extra_views, index_context_right))
        if self.cfg.permute_context_views:
            context_views = context_views[torch.randperm(len(context_views))]

        return (
            context_views,
            index_target,
            overlap
        )

    @property
    def num_context_views(self) -> int | list[int]:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
