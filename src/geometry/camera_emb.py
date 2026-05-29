import torch
from einops import rearrange

from .projection import sample_image_grid, get_local_rays, get_world_rays
from ..misc.sht import rsh_cart_2, rsh_cart_4, rsh_cart_8


def get_intrinsic_embedding(context, degree=0, downsample=1, merge_hw=False):
    assert degree in [0, 2, 4, 8]

    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_local_rays(xy_ray, rearrange(context["intrinsics"], "b v i j -> b v () () i j"),)

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "b v h w d -> b v (h w) d")
    else:
        directions = rearrange(directions, "b v h w d -> b v d h w")

    return directions


def get_intrinsic_embedding_new(intrinsics, images, degree=0, downsample=1, merge_hw=False):
    assert degree in [0, 2, 4, 8]

    bv, _, h, w = images.shape
    device = images.device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, ...].expand(bv, -1, -1, -1)  # [bv, h, w, 2]
    directions = get_local_rays(xy_ray, rearrange(intrinsics, "b v i j -> (b v) () () i j"),)

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "bv h w d -> bv (h w) d")
    else:
        directions = rearrange(directions, "bv h w d -> bv d h w")

    return directions


def get_pluker_ray(img, intrinsics, extrinsics):
    bv, _, h, w = img.shape
    device = img.device
    xy_ray, _ = sample_image_grid((h, w), device)
    xy_ray = xy_ray[None, ...].expand(bv, -1, -1, -1)  # [bv, h, w, 2]
    origins, directions = get_world_rays(xy_ray,
                                         rearrange(extrinsics, "b v i j -> (b v) () () i j"),
                                         rearrange(intrinsics, "b v i j ->(b v) () () i j"),
                                         )

    pluker_ray = torch.cat([torch.cross(origins, directions, dim=-1), directions], dim=-1)  # [bv, h, w, 6]
    pluker_ray = rearrange(pluker_ray, "bv h w d -> bv d h w")
    return pluker_ray
