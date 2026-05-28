#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
from pathlib import Path

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
GLM_INCLUDE = ROOT / "third_party" / "glm"
IS_ROCM = bool(getattr(torch.version, "hip", None))


def _split_flags(name: str) -> list[str]:
    value = os.environ.get(name, "").strip()
    return value.split() if value else []


def _nvcc_flags() -> list[str]:
    # Leave architecture selection to PyTorch's extension tooling. It will honor
    # TORCH_CUDA_ARCH_LIST or infer the visible GPU architectures at build time.
    flags = [
        f"-I{GLM_INCLUDE}",
        "-O3",
    ]
    if not IS_ROCM:
        flags.extend(["--use_fast_math", "--expt-relaxed-constexpr"])
    flags.extend(_split_flags("DIFF_TRIANGLE_NVCC_FLAGS"))
    return flags


def _cxx_flags() -> list[str]:
    flags = ["-O3"]
    flags.extend(_split_flags("DIFF_TRIANGLE_CXX_FLAGS"))
    return flags


def _sources() -> list[str]:
    if IS_ROCM:
        return [
            "hip_rasterizer/rasterizer_impl.hip",
            "hip_rasterizer/forward.hip",
            "hip_rasterizer/backward.hip",
            "hip_rasterizer/utils.hip",
            "rasterize_points.hip",
            "ext.cpp",
        ]

    return [
        "cuda_rasterizer/rasterizer_impl.cu",
        "cuda_rasterizer/forward.cu",
        "cuda_rasterizer/backward.cu",
        "cuda_rasterizer/utils.cu",
        "rasterize_points.cu",
        "ext.cpp",
    ]

setup(
    name="diff_triangle_rasterization",
    version="0.0.0",
    description="Differentiable triangle rasterization CUDA extension",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="diff_triangle_rasterization._C",
            sources=_sources(),
            extra_compile_args={
                "nvcc": _nvcc_flags(),
                "cxx": _cxx_flags(),
            },
            include_dirs=[str(GLM_INCLUDE)],
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
