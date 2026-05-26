# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import curope as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.rope_2d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        grad_tokens = grad_res.contiguous()
        _kernels.rope_2d(grad_tokens, positions, base, -F0)
        return grad_tokens, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions): 
        # The CUDA kernel expects contiguous tensors laid out as (B, N, H, D),
        # while callers pass tokens as (B, H, N, D).
        output_dtype = tokens.dtype
        kernel_dtype = torch.float32 if tokens.dtype == torch.bfloat16 else tokens.dtype
        tokens_bn_hd = tokens.transpose(1, 2).to(dtype=kernel_dtype).contiguous()
        positions_bn2 = positions.contiguous()
        tokens_bn_hd = cuRoPE2D_func.apply(tokens_bn_hd, positions_bn2, self.base, self.F0)
        return tokens_bn_hd.transpose(1, 2).to(dtype=output_dtype)
