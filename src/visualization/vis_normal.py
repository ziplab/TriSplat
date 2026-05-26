import torch
from jaxtyping import Float
from torch import Tensor


def normal_to_rgb(
    normal: Float[Tensor, "*batch 3 height width"],
    mask: Tensor | float | None = None,
    flip_y: bool = True,
    flip_z: bool = True,
) -> Float[Tensor, "*batch 3 height width"]:
    """Map channel-first normals in [-1, 1] to RGB in [0, 1].

    By default this matches Trisplat's visualization convention by flipping the
    Y and Z axes before mapping to RGB.
    """
    normal_draw = torch.cat(
        [
            normal[..., 0:1, :, :],
            -normal[..., 1:2, :, :] if flip_y else normal[..., 1:2, :, :],
            -normal[..., 2:3, :, :] if flip_z else normal[..., 2:3, :, :],
        ],
        dim=-3,
    )
    normal_draw = (normal_draw * 0.5 + 0.5).clamp(0, 1)
    if mask is not None:
        normal_draw = normal_draw * mask
    return normal_draw
