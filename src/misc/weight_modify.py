import logging
from typing import List, Dict

import math
import torch
from torch import nn as nn
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        import functorch
        vmap = functorch.vmap
    except ImportError:
        if hasattr(torch, 'vmap'):
            vmap = torch.vmap
        else:
            assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))

    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def adapt_head_conv(conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape

    conv_weight_new = torch.chunk(conv_weight, 6, dim=1)
    conv_weight_new = [conv_weight_new.mean(dim=1, keepdim=True) for conv_weight_new in conv_weight_new]
    conv_weight_new = torch.cat(conv_weight_new, dim=1) * 0.5
    conv_weight = torch.cat([conv_weight, conv_weight_new], dim=1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def adapt_linear(conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I = conv_weight.shape

    conv_weight_new = torch.tensor_split(conv_weight, 81, dim=1)
    conv_weight_new = [conv_weight_new.mean(dim=1, keepdim=True) for conv_weight_new in conv_weight_new]
    conv_weight_new = torch.cat(conv_weight_new, dim=1)
    # conv_weight = torch.cat([conv_weight, conv_weight_new], dim=1)
    conv_weight = torch.cat([conv_weight * 0.5, conv_weight_new * 0.5], dim=1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    prefix = ''

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.backbone.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.backbone.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
            if v.shape[1] != I:
                v = adapt_input_conv(I, v)
        elif 'decoder_embed.weight' in k:
            O, I = model.backbone.decoder_embed.weight.shape
            if v.shape[1] != I:
                v = adapt_linear(v)

        out_dict[k] = v

    # add prefix to make our model happy
    prefix = 'backbone.'
    out_dict = {prefix + k if 'downstream_head' not in k else k: v for k, v in out_dict.items()}

    out_dict['downstream_head1.dpt.head.4.weight'] = out_dict['downstream_head1.dpt.head.4.weight'][0:3]
    out_dict['downstream_head1.dpt.head.4.bias'] = out_dict['downstream_head1.dpt.head.4.bias'][0:3]
    out_dict['downstream_head2.dpt.head.4.weight'] = out_dict['downstream_head2.dpt.head.4.weight'][0:3]
    out_dict['downstream_head2.dpt.head.4.bias'] = out_dict['downstream_head2.dpt.head.4.bias'][0:3]

    return out_dict


def adapt_linear_weights(original_weight, original_bias, new_gaussians_per_axis, output_dim=3):
    """
    Adapt pretrained linear layer weights for downsampled output
    """
    original_patch_size = int((original_weight.shape[0] // output_dim) ** 0.5)

    # Reshape original weight: (output_dim * orig_patch², input_dim) -> (output_dim, orig_patch, orig_patch, input_dim)
    weight = original_weight.view(output_dim, original_patch_size, original_patch_size, -1)
    bias = original_bias.view(output_dim, original_patch_size, original_patch_size)

    # Downsample using interpolation
    # Convert to (output_dim, input_dim, orig_patch, orig_patch) for interpolation
    weight = weight.permute(0, 3, 1, 2)
    target_size = new_gaussians_per_axis

    # Interpolate weights and bias
    weight_downsampled = F.interpolate(weight, size=(target_size, target_size), mode='bilinear', align_corners=False)
    bias_downsampled = F.interpolate(bias.unsqueeze(1), size=(target_size, target_size), mode='bilinear',
                                     align_corners=False).squeeze(1)

    # Reshape back to linear layer format
    weight_downsampled = weight_downsampled.permute(0, 2, 3, 1).contiguous().view(-1, weight.shape[1])
    bias_downsampled = bias_downsampled.contiguous().view(-1)

    return weight_downsampled, bias_downsampled


def checkpoint_filter_fn_new(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        prefix_old: str = '',
        prefix_new: str = 'backbone.',
        interpolation: str = 'bicubic',
        antialias: bool = True,
        downsample_ratio: int = 1,
        gaussians_per_axis: int=14,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}

    skip_layers = ['point_decoder', 'point_head', 'camera_decoder', 'camera_head', 'conf_decoder', 'conf_head']

    for k, v in state_dict.items():
        if any([layer in k for layer in skip_layers]):
            new_k = k
        else:
            new_k = prefix_new + k

        out_dict[new_k] = v

    if 'point_head.proj.weight' in out_dict and 'point_head.proj.bias' in out_dict:
        # Get the original point head weights
        orig_weight = state_dict['point_head.proj.weight']
        orig_bias = state_dict['point_head.proj.bias']

        # Adapt weights for new downsample ratio
        new_weight, new_bias = adapt_linear_weights(
            orig_weight, orig_bias,
            new_gaussians_per_axis=gaussians_per_axis,
        )

        # Update state dict
        out_dict['point_head.proj.weight'] = new_weight
        out_dict['point_head.proj.bias'] = new_bias

    return out_dict
