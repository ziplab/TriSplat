# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os

from setuptools import setup
import torch
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _arch_flags() -> list[str]:
    if getattr(torch.version, "hip", None):
        # Let PyTorch choose ROCm arch flags and avoid NVIDIA-only -gencode flags.
        return []

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        flags: list[str] = []
        for arch in arch_list.replace(";", " ").split():
            use_ptx = arch.endswith("+PTX")
            arch = arch.removesuffix("+PTX")
            major, minor = arch.split(".")
            compute = f"compute_{major}{minor}"
            sm = f"sm_{major}{minor}"
            flags.extend(["-gencode", f"arch={compute},code={sm}"])
            if use_ptx:
                flags.extend(["-gencode", f"arch={compute},code={compute}"])
        return flags

    if cuda.is_available():
        flags = []
        caps = sorted({cuda.get_device_capability(i) for i in range(cuda.device_count())})
        for major, minor in caps:
            flags.extend(["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"])
        return flags

    # Last resort: fall back to PyTorch's bundled defaults when no GPU is visible.
    return cuda.get_gencode_flags().replace("compute=", "arch=").split()


all_cuda_archs = _arch_flags()

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3']+all_cuda_archs,
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
