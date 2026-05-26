from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int | list[int]
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    permute_context_views: Optional[bool] = False
    max_img_per_gpu: int = 48


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        num_context_views: Optional[int] = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.
        if num_context_views is None:
            if isinstance(self.cfg.num_context_views, list):
                assert len(self.cfg.num_context_views) == 2

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

        if num_context_views > 2:
            num_extra_views = num_context_views - 2
            extra_views = []
            while len(set(extra_views)) != num_extra_views:
                extra_views = torch.randint(
                    index_context_left + 1,
                    index_context_right,
                    (num_extra_views,),
                ).tolist()
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
