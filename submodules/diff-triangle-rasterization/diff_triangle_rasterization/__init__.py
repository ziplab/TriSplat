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

from typing import NamedTuple
import torch.nn as nn
import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "diff_triangle_rasterization._C is unavailable. Rebuild the extension from "
        "`submodules/diff-triangle-rasterization` with a PyTorch/CUDA toolchain that "
        "supports your GPU, for example: `python -m pip install -e .`. If you changed "
        "PyTorch or CUDA, remove `~/.cache/torch_extensions` before rebuilding."
    ) from exc


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def _empty_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_empty((0,))


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(tensor).__name__}.")
    if not tensor.is_cuda:
        raise ValueError(f"`{name}` must be a CUDA tensor.")

def rasterize_triangles(
    triangles_points,
    sigma,
    num_points_per_triangle,
    cumsum_of_points_per_triangle,
    number_of_points,
    sh,
    colors_precomp,
    opacities,
    means2D,
    scaling,
    density_factor,
    raster_settings,
):
    return _RasterizeTriangles.apply(
        triangles_points,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        number_of_points,
        sh,
        colors_precomp,
        opacities,
        means2D,
        scaling,
        density_factor,
        raster_settings,
    )

class _RasterizeTriangles(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        triangles_points,
        sigma,
        num_points_per_triangle,
        cumsum_of_points_per_triangle,
        number_of_points,
        sh,
        colors_precomp,
        opacities,
        means2D,
        scaling,
        density_factor,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            colors_precomp,
            opacities,
            scaling,
            density_factor,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            number_of_points,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, scaling, density_factor, max_blending = _C.rasterize_triangles(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, scaling, density_factor, max_blending = _C.rasterize_triangles(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.number_of_points = number_of_points
        ctx.save_for_backward(triangles_points, sigma, num_points_per_triangle, cumsum_of_points_per_triangle, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, scaling, density_factor, depth, max_blending

    @staticmethod
    def backward(ctx, grad_out_color, _, __, ___, grad_depth, _____):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        number_of_points = ctx.number_of_points
        triangles_points, sigma, num_points_per_triangle, cumsum_of_points_per_triangle, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                triangles_points,
                sigma,
                num_points_per_triangle,
                cumsum_of_points_per_triangle,
                radii, 
                colors_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                number_of_points,
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_triangles, grad_sigma, grad_colors_precomp, grad_opacities, grad_sh, grad_means2D = _C.rasterize_triangles_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_triangles, grad_sigma, grad_colors_precomp, grad_opacities, grad_sh, grad_means2D = _C.rasterize_triangles_backward(*args)


        #print(torch.max(torch.abs(grad_triangles)), torch.min(torch.abs(grad_triangles)))

        #grad_triangles = grad_triangles.reshape(-1, 8, 3)
        grad_triangles = grad_triangles.flatten(0)

        grad_sigma = grad_sigma.view(-1, 1)

        grads = (
            grad_triangles, 
            grad_sigma,
            None,
            None,
            None,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_means2D,
            None,
            None,
            None
        )

        return grads

class TriangleRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, triangles_points, sigma, num_points_per_triangle, cumsum_of_points_per_triangle, number_of_points, opacities, means2D, scaling, density_factor,  shs = None, colors_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise ValueError("Please provide exactly one of `shs` or `colors_precomp`.")

        for name, tensor in (
            ("triangles_points", triangles_points),
            ("sigma", sigma),
            ("num_points_per_triangle", num_points_per_triangle),
            ("cumsum_of_points_per_triangle", cumsum_of_points_per_triangle),
            ("opacities", opacities),
            ("means2D", means2D),
            ("scaling", scaling),
            ("density_factor", density_factor),
        ):
            _require_cuda_tensor(name, tensor)
        
        if shs is None:
            shs = _empty_like_reference(triangles_points)
        else:
            _require_cuda_tensor("shs", shs)
        if colors_precomp is None:
            colors_precomp = _empty_like_reference(triangles_points)
        else:
            _require_cuda_tensor("colors_precomp", colors_precomp)

        # Invoke C++/CUDA rasterization routine
        return rasterize_triangles(
            triangles_points,
            sigma,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            number_of_points,
            shs,
            colors_precomp,
            opacities,
            means2D,
            scaling,
            density_factor,
            raster_settings, 
        )
