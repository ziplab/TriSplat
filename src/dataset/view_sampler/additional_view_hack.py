import torch
from jaxtyping import Int
from torch import Tensor


def add_addtional_context_index(
    indices: Int[Tensor, "*batch 2"],
    number_of_context_views: int,
) -> Int[Tensor, "*batch view"]:
    left, right = indices[..., 0], indices[..., -1]
    # evenly distribute the additional context views between the left and right views
    ctx_indices = torch.stack(
        [
            torch.linspace(left, right, number_of_context_views).long()
        ],
        dim=-1,
    ).squeeze(-1)
    return ctx_indices
