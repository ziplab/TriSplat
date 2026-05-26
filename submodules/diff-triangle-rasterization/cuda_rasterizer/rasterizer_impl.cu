/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

 #include "rasterizer_impl.h"
 #include <iostream>
 #include <fstream>
 #include <algorithm>
 #include <numeric>
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #include <cub/cub.cuh>
 #include <cub/device/device_radix_sort.cuh>
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;
 
 #include "auxiliary.h"
 #include "forward.h"
 #include "backward.h"
 
 // Helper function to find the next-highest bit of the MSB
 // on the CPU.
 uint32_t getHigherMsb(uint32_t n)
 {
	 uint32_t msb = sizeof(n) * 4;
	 uint32_t step = msb;
	 while (step > 1)
	 {
		 step /= 2;
		 if (n >> msb)
			 msb += step;
		 else
			 msb -= step;
	 }
	 if (n >> msb)
		 msb++;
	 return msb;
 }
 
 // Wrapper method to call auxiliary coarse frustum containment test.
 // Mark all triangles that pass it.
 __global__ void checkFrustum(int P,
	 const float* orig_points,
	 const float* viewmatrix,
	 const float* projmatrix,
	 bool* present)
 {
	 auto idx = cg::this_grid().thread_rank();
	 if (idx >= P)
		 return;
 
	 float3 p_view;
	 present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
 }
 
 // Generates one key/value pair for all Triangle / tile overlaps. 
 // Run once per Triangle (1:N mapping).
 __global__ void duplicateWithKeys(
	 int P,
	 const float2* points_xy,
	 const float* depths,
	 const uint32_t* offsets,
	 const float2* p_image,
	 const int* cumsum_of_points_per_triangle,
	 uint2* rect_min,
	 uint2* rect_max,
	 uint64_t* Triangle_keys_unsorted,
	 uint32_t* Triangle_values_unsorted,
	 int* radii,
	 dim3 grid)
 {
	 auto idx = cg::this_grid().thread_rank();
	 if (idx >= P)
		 return;
 
	 // Generate no key/value pair for invisible triangles
	 if (radii[idx] > 0)
	 {	 uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		 uint2 rect_min_conv = rect_min[idx];
	 	 uint2 rect_max_conv = rect_max[idx];
		 // For each tile that the bounding rect overlaps, emit a 
		 // key/value pair. The key is |  tile ID  |      depth      |,
		 // and the value is the ID of the Triangle. Sorting the values 
		 // with this key yields Triangle IDs in a list, such that they
		 // are first sorted by tile and then by depth. 
		 for (int y = rect_min_conv.y; y < rect_max_conv.y; y++)
		 {
			 for (int x = rect_min_conv.x; x < rect_max_conv.x; x++)
			 {
				 uint64_t key = y * grid.x + x;
				 key <<= 32;
				 key |= *((uint32_t*)&depths[idx]);
				 Triangle_keys_unsorted[off] = key;
				 Triangle_values_unsorted[off] = idx;
				 off++;
			 }
		 }
	 }
 }
 
 // Check keys to see if it is at the start/end of one tile's range in 
 // the full sorted list. If yes, write start/end of this tile. 
 // Run once per instanced (duplicated) Triangle ID.
 __global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
 {
	 auto idx = cg::this_grid().thread_rank();
	 if (idx >= L)
		 return;
 
	 // Read tile ID from key. Update start/end of tile range if at limit.
	 uint64_t key = point_list_keys[idx];
	 uint32_t currtile = key >> 32;
	 if (idx == 0)
		 ranges[currtile].x = 0;
	 else
	 {
		 uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		 if (currtile != prevtile)
		 {
			 ranges[prevtile].y = idx;
			 ranges[currtile].x = idx;
		 }
	 }
	 if (idx == L - 1)
		 ranges[currtile].y = L;
 }
 
 // Mark triangles as visible/invisible, based on view frustum testing
 void CudaRasterizer::Rasterizer::markVisible(
	 int P,
	 float* means3D,
	 float* viewmatrix,
	 float* projmatrix,
	 bool* present)
 {
	 checkFrustum << <(P + 255) / 256, 256 >> > (
		 P,
		 means3D,
		 viewmatrix, projmatrix,
		 present);
 }
 
 CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, size_t total_nb_points)
 {
	 GeometryState geom;
	 obtain(chunk, geom.depths, P, 128);
	 obtain(chunk, geom.clamped, P * 3, 128);
	 obtain(chunk, geom.internal_radii, P, 128);
	 obtain(chunk, geom.means2D, P, 128);
	 obtain(chunk, geom.conic_opacity, P, 128);
	 obtain(chunk, geom.phi_center, P, 128);
	 obtain(chunk, geom.rgb, P * 3, 128);
	 obtain(chunk, geom.cov3D, P * 6, 128);
	 obtain(chunk, geom.tiles_touched, P, 128);
	 cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	 obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	 obtain(chunk, geom.point_offsets, P, 128);
	 obtain(chunk, geom.p_image, total_nb_points, 128);
	 obtain(chunk, geom.indices, total_nb_points, 128);
	 obtain(chunk, geom.offsets, total_nb_points, 128);
	 obtain(chunk, geom.normals, total_nb_points, 128);
	 obtain(chunk, geom.p_w, total_nb_points, 128);
     obtain(chunk, geom.rect_min, P, 128);
     obtain(chunk, geom.rect_max, P, 128);
 
	 return geom;
 }
 
 CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
 {
	 ImageState img;
	 obtain(chunk, img.accum_alpha, N * 3, 128);
	 obtain(chunk, img.n_contrib, N * 2, 128);
	 obtain(chunk, img.ranges, N, 128);
	 return img;
 }
 
 CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
 {
	 BinningState binning;
	 obtain(chunk, binning.point_list, P, 128);
	 obtain(chunk, binning.point_list_unsorted, P, 128);
	 obtain(chunk, binning.point_list_keys, P, 128);
	 obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	 cub::DeviceRadixSort::SortPairs(
		 nullptr, binning.sorting_size,
		 binning.point_list_keys_unsorted, binning.point_list_keys,
		 binning.point_list_unsorted, binning.point_list, P);
	 obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	 return binning;
 }
 
 // Forward rendering procedure for differentiable rasterization
 // of triangles.
 int CudaRasterizer::Rasterizer::forward(
	 std::function<char* (size_t)> geometryBuffer,
	 std::function<char* (size_t)> binningBuffer,
	 std::function<char* (size_t)> imageBuffer,
	 const int P, int D, int M,
	 const float* background,
	 const int width, int height,
	 const float* triangles_points,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const int total_nb_points,
	 const float* shs,
	 const float* colors_precomp,
	 const float* opacities,
	 float* scaling,
	 float* density_factor,
	 const float* viewmatrix,
	 const float* projmatrix,
	 const float* cam_pos,
	 const float tan_fovx, float tan_fovy,
	 const bool prefiltered,
	 float* out_color,
	 float* out_others,
	 float* max_blending,
	 int* radii,
	 bool debug)
 {
	 const float focal_y = height / (2.0f * tan_fovy);
	 const float focal_x = width / (2.0f * tan_fovx);
 
	 size_t chunk_size = required<GeometryState>(P, total_nb_points);
	 char* chunkptr = geometryBuffer(chunk_size);
	 GeometryState geomState = GeometryState::fromChunk(chunkptr, P, total_nb_points);
 
	 
	 if (radii == nullptr)
	 {
		 radii = geomState.internal_radii;
	 }
 
	 dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	 dim3 block(BLOCK_X, BLOCK_Y, 1);
 
	 // Dynamically resize image-based auxiliary buffers during training
	 size_t img_chunk_size = required<ImageState>(width * height);
	 char* img_chunkptr = imageBuffer(img_chunk_size);
	 ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);
 
	 if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	 {
		 throw std::runtime_error("For non-RGB, provide precomputed Triangle colors!");
	 }
 
	 // Run preprocessing per-Triangle (transformation, bounding, conversion of SHs to RGB)
	 CHECK_CUDA(FORWARD::preprocess(
		 P, D, M,
		 triangles_points,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 opacities,
		 scaling,
		 density_factor,
		 shs,
		 geomState.clamped,
		 colors_precomp,
		 viewmatrix, projmatrix,
		 (glm::vec3*)cam_pos,
		 width, height,
		 focal_x, focal_y,
		 tan_fovx, tan_fovy,
		 radii,
		 geomState.normals,
		 geomState.offsets,
		 geomState.p_w,
		 geomState.p_image,
		 geomState.indices,
		 geomState.means2D,
		 geomState.depths,
		 geomState.rgb,
		 geomState.conic_opacity,
		 geomState.cov3D,
		 geomState.phi_center,
		 geomState.rect_min,
		 geomState.rect_max,
		 tile_grid,
		 geomState.tiles_touched,
		 prefiltered
	 ), debug)
 
	 // Compute prefix sum over full list of touched tile counts by triangles
	 // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	 CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
 
	 // Retrieve total number of Triangle instances to launch and resize aux buffers
	 int num_rendered;
	 CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
 
	 size_t binning_chunk_size = required<BinningState>(num_rendered);
	 char* binning_chunkptr = binningBuffer(binning_chunk_size);
	 BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
 
	 // For each instance to be rendered, produce adequate [ tile | depth ] key 
	 // and corresponding dublicated Triangle indices to be sorted
	 duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		 P,
		 geomState.means2D,
		 geomState.depths,
		 geomState.point_offsets,
		 geomState.p_image,
		 cumsum_of_points_per_triangle,
		 geomState.rect_min,
		 geomState.rect_max,
		 binningState.point_list_keys_unsorted,
		 binningState.point_list_unsorted,
		 radii,
		 tile_grid)
	 CHECK_CUDA(, debug)
 
	 int bit = getHigherMsb(tile_grid.x * tile_grid.y);
 
	 // Sort complete list of (duplicated) Triangle indices by keys
	 CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		 binningState.list_sorting_space,
		 binningState.sorting_size,
		 binningState.point_list_keys_unsorted, binningState.point_list_keys,
		 binningState.point_list_unsorted, binningState.point_list,
		 num_rendered, 0, 32 + bit), debug)
 
	 CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
 
	 // Identify start and end of per-tile workloads in sorted list
	 if (num_rendered > 0)
		 identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			 num_rendered,
			 binningState.point_list_keys,
			 imgState.ranges);
	 CHECK_CUDA(, debug)
 
	 // Let each tile blend its range of triangles independently in parallel
	 const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	 CHECK_CUDA(FORWARD::render(
		 tile_grid, block,
		 imgState.ranges,
		 binningState.point_list,
		 width, height,
		 geomState.normals,
		 geomState.offsets,
		 geomState.means2D,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 feature_ptr,
		 geomState.conic_opacity,
		 geomState.depths,
		 geomState.phi_center,
		 imgState.accum_alpha,
		 imgState.n_contrib,
		 background,
		 out_color,
		 out_others,
		 max_blending), debug)
 
	 return num_rendered;
 }
 
 // Produce necessary gradients for optimization, corresponding
 // to forward render pass
 void CudaRasterizer::Rasterizer::backward(
	 const int P, int D, int M, int R,
	 const float* background,
	 const int width, int height,
	 const float* triangles_points,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const int total_nb_points,
	 const float* shs,
	 const float* colors_precomp,
	 const float* viewmatrix,
	 const float* projmatrix,
	 const float* campos,
	 const float tan_fovx, float tan_fovy,
	 const int* radii,
	 char* geom_buffer,
	 char* binning_buffer,
	 char* img_buffer,
	 const float* dL_dpix,
	 const float* dL_depths,
	 float* dL_dmeans3D,
	 float* dL_dmeans2D,
	 float* dL_dcov3D,
	 float* dL_dnormal3D,
	 float* dL_dtriangle,
	 float* dL_dsigma,
	 float* dL_dnormals,
	 float* dL_doffsets,
	 float* dL_dconic,
	 float* dL_dopacity,
	 float* dL_dcolor,
	 float* dL_dsh,
	 float* dL_dsigma_factor,
	 bool debug)
 {
	 GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, total_nb_points);
	 BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	 ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
 
	 if (radii == nullptr)
	 {
		 radii = geomState.internal_radii;
	 }
 
	 const float focal_y = height / (2.0f * tan_fovy);
	 const float focal_x = width / (2.0f * tan_fovx);
 
	 const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	 const dim3 block(BLOCK_X, BLOCK_Y, 1);
 
	 // Compute loss gradients w.r.t. 2D mean position, conic matrix,
	 // opacity and RGB of triangles from per-pixel loss gradients.
	 // If we were given precomputed colors and not SHs, use them.
	 const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	 CHECK_CUDA(BACKWARD::render(
		 tile_grid,
		 block,
		 imgState.ranges,
		 binningState.point_list,
		 width, height,
		 background,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 geomState.normals,
		 geomState.offsets,
		 geomState.conic_opacity,
		 geomState.depths,
		 geomState.means2D,
		 geomState.phi_center,
		 color_ptr,
		 imgState.accum_alpha,
		 imgState.n_contrib,
		 dL_dpix,
		 dL_depths,
		 (float2*)dL_dnormals,
		 dL_doffsets,
		 dL_dsigma,
		 (float3*)dL_dmeans2D,
		 (float4*)dL_dconic,
		 dL_dopacity,
		 dL_dnormal3D,
		 dL_dcolor,
		 dL_dsigma_factor), debug)
 
	 // Take care of the rest of preprocessing. Was the precomputed covariance
	 // given to us or a scales/rot pair? If precomputed, pass that. If not,
	 // use the one we computed ourselves.
	 CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		 triangles_points,
		 width, height,
		 radii,
		 shs,
		 geomState.clamped,
		 viewmatrix,
		 projmatrix,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 geomState.means2D,
		 geomState.p_w,
		 geomState.p_image,
		 geomState.indices,
		 geomState.cov3D,
		 focal_x, focal_y,
		 tan_fovx, tan_fovy,
		 (glm::vec3*)campos,
		 (glm::vec3*)dL_dtriangle,
		 (float2*)dL_dnormals,
		 dL_doffsets,
		 (glm::vec3*)dL_dmeans3D,
		 (float3*)dL_dmeans2D,
		 dL_dconic,
		 dL_dcov3D,
		 dL_dnormal3D,
		 dL_dcolor,
		 dL_dsh,
		 dL_dsigma_factor), debug)
 }