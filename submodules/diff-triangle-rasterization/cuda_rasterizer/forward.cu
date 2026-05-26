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

 #include "forward.h"
 #include "auxiliary.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;
 
 
 
 // Forward method for converting the input spherical harmonics
 // coefficients of each Triangle to a simple RGB color.
 __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, bool* clamped)
 {
	 // The implementation is loosely based on code for 
	 // "Differentiable Point-Based Radiance Fields for 
	 // Efficient View Synthesis" by Zhang et al. (2022)
	 glm::vec3 pos = means;
	 glm::vec3 dir = pos - campos;
	 dir = dir / glm::length(dir);
 
	 glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	 glm::vec3 result = SH_C0 * sh[0];
 
	 if (deg > 0)
	 {
		 float x = dir.x;
		 float y = dir.y;
		 float z = dir.z;
		 result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
 
		 if (deg > 1)
		 {
			 float xx = x * x, yy = y * y, zz = z * z;
			 float xy = x * y, yz = y * z, xz = x * z;
			 result = result +
				 SH_C2[0] * xy * sh[4] +
				 SH_C2[1] * yz * sh[5] +
				 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				 SH_C2[3] * xz * sh[7] +
				 SH_C2[4] * (xx - yy) * sh[8];
 
			 if (deg > 2)
			 {
				 result = result +
					 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					 SH_C3[1] * xy * z * sh[10] +
					 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					 SH_C3[5] * z * (xx - yy) * sh[14] +
					 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			 }
		 }
	 }
	 result += 0.5f;
 
	 // RGB colors are clamped to positive values. If values are
	 // clamped, we need to keep track of this for the backward pass.
	 clamped[3 * idx + 0] = (result.x < 0);
	 clamped[3 * idx + 1] = (result.y < 0);
	 clamped[3 * idx + 2] = (result.z < 0);
	 return glm::max(result, 0.0f);
 }
 
 
 
 // Perform initial steps for each Triangle prior to rasterization.
 template<int C>
 __global__ void preprocessCUDA(int P, int D, int M,
	 const float* triangles_points,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const float* opacities,
	 float* scaling,
	 float* density_factor,
	 const float* shs,
	 bool* clamped,
	 const float* colors_precomp,
	 const float* viewmatrix,
	 const float* projmatrix,
	 const glm::vec3* cam_pos,
	 const int W, int H,
	 const float tan_fovx, float tan_fovy,
	 const float focal_x, float focal_y,
	 int* radii,
	 float2* normals,
	 float* offsets,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 float2* points_xy_image,
	 float* depths,
	 float* rgb,
	 float4* conic_opacity,
	 float* cov3Ds,
	 float2* phi_center,
	 uint2* rect_min,
	 uint2* rect_max,
	 const dim3 grid,
	 uint32_t* tiles_touched,
	 bool prefiltered)
 {
 
	 auto idx = cg::this_grid().thread_rank();
	 if (idx >= P)
		 return;
 
	 // Initialize radius and touched tiles to 0. If this isn't changed,
	 // this Triangle will not be processed further.
 
	 const int cumsum_for_triangle = cumsum_of_points_per_triangle[idx];
	 const int offset = 3 * cumsum_for_triangle;
 
	 radii[idx] = 0;
	 tiles_touched[idx] = 0;
	 scaling[idx] = 0.0f;
	 density_factor[idx] = 0.0f;

	 float stopping_influence = 0.01f;

 
	 // if the opacity is too low, we can skip the Triangle
	 if (opacities[idx] < stopping_influence)
		 return;
 
	 float3 center_triangle = {0.0f, 0.0f, 0.0f};
	 for (int i = 0; i < num_points_per_triangle[idx]; i++) {
		 indices[cumsum_for_triangle + i] = i;
		 center_triangle.x += triangles_points[offset + 3 * i];
		 center_triangle.y += triangles_points[offset + 3 * i + 1];
		 center_triangle.z += triangles_points[offset + 3 * i + 2];
	 }
 
	 center_triangle.x /= num_points_per_triangle[idx];
	 center_triangle.y /= num_points_per_triangle[idx];
	 center_triangle.z /= num_points_per_triangle[idx];
 
	 // Perform near culling, quit if outside.
	 float3 p_view_triangle;
	 if (!in_frustum_triangle(idx, center_triangle, viewmatrix, projmatrix, prefiltered, p_view_triangle)){
		 return;
	 }

	 // Calculate the normal of the Triangle
	 float3 normal_cvx = {0.0f, 0.0f, 0.0f};
	 float3 p0 = make_float3(
		triangles_points[offset + 0],
		triangles_points[offset + 1],
		triangles_points[offset + 2]
	 );
	 float3 p1 = make_float3(
		triangles_points[offset + 3],
		triangles_points[offset + 4],
		triangles_points[offset + 5]
	 );
	 float3 p2 = make_float3(
		triangles_points[offset + 6],
		triangles_points[offset + 7],
		triangles_points[offset + 8]
	 );

	 float3 v1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
	 float3 v2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

	 float3 cross_prod = make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	 );
	 cross_prod = transformVec4x3(cross_prod, viewmatrix);

	 float length_cross = __fsqrt_rn(cross_prod.x*cross_prod.x + cross_prod.y*cross_prod.y + cross_prod.z*cross_prod.z);
	 length_cross = max(length_cross, 1e-4f);
	 cross_prod.x /= length_cross;
	 cross_prod.y /= length_cross;
	 cross_prod.z /= length_cross;
	
	 normal_cvx = cross_prod;
	 // 2. Normalize the camera viewpoint direction
	float length_viewpoint = __fsqrt_rn(p_view_triangle.x * p_view_triangle.x + 
		p_view_triangle.y * p_view_triangle.y + 
		p_view_triangle.z * p_view_triangle.z);
	length_viewpoint = max(length_viewpoint, 1e-4f);

	float3 normalized_camera_center;
	normalized_camera_center.x = p_view_triangle.x / length_viewpoint;
	normalized_camera_center.y = p_view_triangle.y / length_viewpoint;
	normalized_camera_center.z = p_view_triangle.z / length_viewpoint;

	// 3. Compute cosine (before flipping the normal)
	float cos_theta = normal_cvx.x * normalized_camera_center.x +
	normal_cvx.y * normalized_camera_center.y +
	normal_cvx.z * normalized_camera_center.z;

	// 4. Flip the normal if needed (ensure it faces the camera)
	if (cos_theta > 0) {
		normal_cvx.x = -normal_cvx.x;
		normal_cvx.y = -normal_cvx.y;
		normal_cvx.z = -normal_cvx.z;
		cos_theta = -cos_theta; 
	}

	const float threshold = 0.001f;
	if (fabsf(cos_theta) < threshold) {
		return;
	}

	 float4 p_hom_center = transformPoint4x4(center_triangle, projmatrix);
	 float p_w_center = 1.0f / (p_hom_center.w + 0.0000001f);
	 float3 center_triangle_camera_view = { p_hom_center.x * p_w_center, p_hom_center.y * p_w_center, p_hom_center.z * p_w_center };
	 float2 center_triangle_2D = { ndc2Pix(center_triangle_camera_view.x, W), ndc2Pix(center_triangle_camera_view.y, H) };
 

	 float distance = 0.0f;
	 float distance_points = 0.0f;

	for (int i = 0; i < num_points_per_triangle[idx]; i++) {
		 float3 triangle_point = {triangles_points[offset + 3 * i], triangles_points[offset + 3 * i + 1], triangles_points[offset + 3 * i + 2]};
		 float4 p_hom = transformPoint4x4(triangle_point, projmatrix);
		 p_w[cumsum_for_triangle + i] = 1.0f / (p_hom.w + 0.0000001f);
		 float3 p_proj = { p_hom.x * p_w[cumsum_for_triangle + i], p_hom.y * p_w[cumsum_for_triangle + i], p_hom.z * p_w[cumsum_for_triangle + i] };
		 p_image[cumsum_for_triangle + i] = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

		 // calculate distance from p_image to center_triangle_2D
		 distance = __fsqrt_rn((p_image[cumsum_for_triangle + i].x - center_triangle_2D.x) * (p_image[cumsum_for_triangle + i].x - center_triangle_2D.x) + (p_image[cumsum_for_triangle + i].y - center_triangle_2D.y) * (p_image[cumsum_for_triangle + i].y - center_triangle_2D.y));

		 if (distance > distance_points) {
			 distance_points = distance;
		 }
	 }

	 // Get the three projected 2D points
	float2 A1 = p_image[cumsum_for_triangle + 0];
	float2 B1 = p_image[cumsum_for_triangle + 1];
	float2 C1 = p_image[cumsum_for_triangle + 2];

	// Compute side lengths (opposite each vertex)
	float a = __fsqrt_rn((B1.x - C1.x) * (B1.x - C1.x) + (B1.y - C1.y) * (B1.y - C1.y)); // Opposite A
	float b = __fsqrt_rn((A1.x - C1.x) * (A1.x - C1.x) + (A1.y - C1.y) * (A1.y - C1.y)); // Opposite B
	float c = __fsqrt_rn((A1.x - B1.x) * (A1.x - B1.x) + (A1.y - B1.y) * (A1.y - B1.y)); // Opposite C

	float sum = a + b + c;

	// Incenter weighted by opposite side lengths
	float2 incenter;
	incenter.x = (a * A1.x + b * B1.x + c * C1.x) / sum;
	incenter.y = (a * A1.y + b * B1.y + c * C1.y) / sum;
 
	 float max_distance_off = 0.0f;
	 int counter = 0;
	 float max_distance_x = 0.0f;
	 float dist = 0.0f;

	 float size = 0.0f;
 

	 float ratio = stopping_influence / opacities[idx];

	 float exponent = 1.0f / sigma[idx];

	 uint2 rect_min_triangle_test = { grid.x, grid.y };
	 uint2 rect_max_triangle_test = { 0,       0       };
	 
	 float previous_offsets[MAX_NB_POINTS];
 
	 for (int i = 0; i < 3; i++) {
		// Points forming the segment
		float2 p1_conv = p_image[cumsum_for_triangle + i];
		float2 p2_conv = p_image[cumsum_for_triangle + (i + 1) % 3];
 
		float nx = p2_conv.y - p1_conv.y;
		float ny = -(p2_conv.x - p1_conv.x);
		float norm = __fsqrt_rn(nx * nx + ny * ny);
		float inv_norm = 1.0f / norm;
	
		// Calculate normalized normal and offset
		float2 normal = {nx * inv_norm, ny * inv_norm};

		float offset = - (normal.x * p1_conv.x + normal.y * p1_conv.y);

		dist = normal.x * incenter.x + normal.y * incenter.y + offset;

		if (dist > 0) {
			normal.x = -normal.x;
			normal.y = -normal.y;
			offset = -offset;
			dist = -dist;
		}	

		if (size == 0){
			size = dist * powf(ratio, exponent);
		}

		normals[cumsum_for_triangle + i] = normal;
		offsets[cumsum_for_triangle + i] = offset; 

	
		offset = offset / __fsqrt_rn(normal.x * normal.x + normal.y * normal.y);
		offset -= size;
		previous_offsets[i] = offset;

		if (i != 0){
			float2 previous_normal;
			previous_normal.x = normals[cumsum_for_triangle + (i-1)].x;
			previous_normal.y = normals[cumsum_for_triangle + (i-1)].y;

			// Compute determinant
			float det = normal.x * previous_normal.y - normal.y * previous_normal.x;

			float intersect_x, intersect_y;
			if (fabsf(det) < 1e-3) {
				continue;
			} else {
				// Calculate intersection point
				intersect_x = -1*(offset * previous_normal.y - previous_offsets[i-1] * normal.y) / det;
				intersect_y = -1*(previous_offsets[i-1] * normal.x - offset * previous_normal.x) / det;

				uint bx0 = min(grid.x, max(0, (uint)(intersect_x / BLOCK_X)));
				uint by0 = min(grid.y, max(0, (uint)(intersect_y / BLOCK_Y)));
				uint bx1 = min(grid.x, max(0, (uint)((intersect_x + BLOCK_X - 1) / BLOCK_X)));
				uint by1 = min(grid.y, max(0, (uint)((intersect_y + BLOCK_Y - 1) / BLOCK_Y)));

				rect_min_triangle_test.x = min(rect_min_triangle_test.x, bx0);
				rect_min_triangle_test.y = min(rect_min_triangle_test.y, by0);
				rect_max_triangle_test.x = max(rect_max_triangle_test.x, bx1);
				rect_max_triangle_test.y = max(rect_max_triangle_test.y, by1);
			}
		}

	 }
 
	   
	 if (distance_points > 1600 or distance_points < 1 or dist > -1) {
		 radii[idx] = 0;
		 tiles_touched[idx] = 0;
		 scaling[idx] = 0.0f;
		 return;
	 }

	/*####################################################################################################
	#### Calculations of the final distance 														     #
	#####################################################################################################*/
	float2 normal = normals[cumsum_for_triangle];
	float offset_ = previous_offsets[0];
  
	float2 previous_normal = normals[cumsum_for_triangle+2];
	float previous_offset = previous_offsets[2];
 	float det = normal.x * previous_normal.y - normal.y * previous_normal.x;
  
	float intersect_x, intersect_y;
	if (fabsf(det) > 1e-3) {
		// Calculate intersection point
		intersect_x = -1*(offset_ * previous_normal.y - previous_offset * normal.y) / det;
		intersect_y = -1*(previous_offset * normal.x - offset_ * previous_normal.x) / det;
		uint bx0 = min(grid.x, max(0, (uint)(intersect_x / BLOCK_X)));
		uint by0 = min(grid.y, max(0, (uint)(intersect_y / BLOCK_Y)));
		uint bx1 = min(grid.x, max(0, (uint)((intersect_x + BLOCK_X - 1) / BLOCK_X)));
		uint by1 = min(grid.y, max(0, (uint)((intersect_y + BLOCK_Y - 1) / BLOCK_Y)));

		rect_min_triangle_test.x = min(rect_min_triangle_test.x, bx0);
		rect_min_triangle_test.y = min(rect_min_triangle_test.y, by0);
		rect_max_triangle_test.x = max(rect_max_triangle_test.x, bx1);
		rect_max_triangle_test.y = max(rect_max_triangle_test.y, by1);
	}

	rect_max[idx] = rect_max_triangle_test;
	rect_min[idx] = rect_min_triangle_test;


	if ((rect_max_triangle_test.x - rect_min_triangle_test.x) * (rect_max_triangle_test.y - rect_min_triangle_test.y) == 0){
		radii[idx] = 0;
		tiles_touched[idx] = 0;
		scaling[idx] = 0.0f;
		return;
	}

	float phi_center_min = dist;
	float max_distance = ceil(distance_points);

	 // We save the 2D Size in Image Space
	 scaling[idx] = max_distance;
	 density_factor[idx] = -dist;
	  
	 // If colors have been precomputed, use them, otherwise convert
	 // spherical harmonics coefficients to RGB color.
	 if (colors_precomp == nullptr)
	 {
		 glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z), *cam_pos, shs, clamped);
		 rgb[idx * C + 0] = result.x;
		 rgb[idx * C + 1] = result.y;
		 rgb[idx * C + 2] = result.z;
	 }


	 phi_center[idx] = {1.0f / phi_center_min, size};
	 depths[idx] = p_view_triangle.z; 
	 radii[idx] = max_distance;
	 points_xy_image[idx] = center_triangle_2D;
	 conic_opacity[idx] = {normal_cvx.x, normal_cvx.y, normal_cvx.z, opacities[idx]};
	 tiles_touched[idx] = (rect_max_triangle_test.y - rect_min_triangle_test.y) * (rect_max_triangle_test.x - rect_min_triangle_test.x);
 }
 
 // Main rasterization method. Collaboratively works on one tile per
 // block, each thread treats one pixel. Alternates between fetching 
 // and rasterizing data.
 template <uint32_t CHANNELS>
 __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
 renderCUDA(
	 const uint2* __restrict__ ranges,
	 const uint32_t* __restrict__ point_list,
	 int W, int H,
	 const float2* __restrict__ normals,
	 const float* __restrict__ offsets,
	 const float2* __restrict__ points_xy_image,
	 const float* __restrict__ sigma,
	 const int* __restrict__ num_points_per_triangle,
	 const int* __restrict__ cumsum_of_points_per_triangle,
	 const float* __restrict__ features,
	 const float4* __restrict__ conic_opacity,
	 const float* __restrict__ depths,
	 const float2* __restrict__ phi_center,
	 float* __restrict__ final_T,
	 uint32_t* __restrict__ n_contrib,
	 const float* __restrict__ bg_color,
	 float* __restrict__ out_color,
	 float* __restrict__ out_others,
	 float* __restrict__ max_blending)
 {
	 // Identify current tile and associated min/max pixel range.
	 auto block = cg::this_thread_block();
	 uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	 uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	 uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	 uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	 uint32_t pix_id = W * pix.y + pix.x;
	 float2 pixf = { (float)pix.x, (float)pix.y };
 
	 // Check if this thread is associated with a valid pixel or outside.
	 bool inside = pix.x < W&& pix.y < H;
	 // Done threads can help with fetching, but don't rasterize
	 bool done = !inside;
 
	 // Load start/end range of IDs to process in bit sorted list.
	 uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	 const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	 int toDo = range.y - range.x;
 
	 // Allocate storage for batches of collectively fetched data.
	 __shared__ int collected_id[BLOCK_SIZE];
	 __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
 
	 /*
	 ADDED FOR TRIANGLE PURPOSES ==========================================================================
	 */
	 __shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_sigma[BLOCK_SIZE];
	 __shared__ float collected_depths[BLOCK_SIZE];
	 __shared__ float2 collected_xy[BLOCK_SIZE];
	 __shared__ float2 collected_phi_center[BLOCK_SIZE];
	 /*
	 ===================================================================================================
	 */
 
	 // Initialize helper variables
	 float T = 1.0f;
	 uint32_t contributor = 0;
	 uint32_t last_contributor = 0;
	 float C[CHANNELS] = { 0 };

	 // Added from 2DGS
	 float N[3] = {0};
	 float D = { 0 };
	 float M1 = {0};
	 float M2 = {0};
	 float distortion = {0};
	 float median_depth = {0};
	 float median_contributor = {-1};
 
	 // Iterate over batches until all done or range is complete
	 for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	 {
		 // End if entire block votes that it is done rasterizing
		 int num_done = __syncthreads_count(done);
		 if (num_done == BLOCK_SIZE)
			 break;
 
		 // Collectively fetch per-Triangle data from global to shared
		 int progress = i * BLOCK_SIZE + block.thread_rank();
		 if (range.x + progress < range.y)
		 {
			 int coll_id = point_list[range.x + progress];
			 collected_id[block.thread_rank()] = coll_id;
			 collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			 collected_sigma[block.thread_rank()] = sigma[coll_id];
			 collected_depths[block.thread_rank()] = depths[coll_id];
			 collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			 for (int k = 0; k < 3; k++) {
				collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_triangle[coll_id] + k];
				collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_triangle[coll_id] + k];
			}
			collected_phi_center[block.thread_rank()] = phi_center[coll_id];
		 }
		 block.sync();
 
		 // Iterate over current batch
		 for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		 {
			 // Keep track of current position in range
			 contributor++;
 
			 int j_id = collected_id[j];
			 float4 con_o = collected_conic_opacity[j];
			 float normal[3] = {con_o.x, con_o.y, con_o.z};
			 float2 phi_center_min = collected_phi_center[j];
			 float sigma_pre = collected_sigma[j];
			 float max_val = -INFINITY;
			 int base = j * MAX_NB_POINTS;
			 bool outside = false;

			 for (int k = 0; k < 3; k++) {
				 // Compute the current distance
				 float dist = (collected_normals[base + k].x * pixf.x
						  + collected_normals[base + k].y * pixf.y
						  + collected_offsets[base + k]);

				 if (dist > 0) {
					outside = true;
					break;
				 }
 
				 max_val = fmaxf(max_val, dist);
			 }

			 if (outside)
				continue;
 
			 float phi_x = max_val;
			 float phi_final = phi_x * phi_center_min.x;
			 float Cx = fmaxf(0.0f,  __powf(phi_final, sigma_pre));

 
			 float alpha = min(0.99f, con_o.w * Cx); 
			 if (alpha < 1.0f / 255.0f)
				 continue;
			 float test_T = T * (1 - alpha);
			 if (test_T < 0.0001f)
			 {
				 done = true;
				 continue;
			 }


			 
			 float blending_weight = alpha * T;
			 // Update the maximum blending weight in a thread-safe way
			 atomicMax(((int*)max_blending) + j_id, *((int*)(&blending_weight)));

			 float A = 1-T;
			 float m = far_n / (far_n - near_n) * (1 - near_n / collected_depths[j]);
			 distortion += (m * m * A + M2 - 2 * m * M1) * blending_weight;
			 D  += collected_depths[j] * blending_weight;
			 M1 += m * blending_weight;
			 M2 += m * m * blending_weight;
 
			 if (T > 0.5) {
				 median_depth = collected_depths[j];
				 median_contributor = contributor;
			 }
			 // Render normal map
			 for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * blending_weight;

			 for (int ch = 0; ch < CHANNELS; ch++)
				 C[ch] += features[j_id * CHANNELS + ch] * alpha * T;
 
			 T = test_T;
 
			 // Keep track of last range entry to update this
			 // pixel.
			 last_contributor = contributor;
		 }
	 }
 
	 // All threads that treat valid pixel write out their final
	 // rendering data to the frame and auxiliary buffers.
	 if (inside)
	 {	
		out_others[pix_id + 0 * H * W] = last_contributor;
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
	 }
 }
 
 void FORWARD::render(
	 const dim3 grid, dim3 block,
	 const uint2* ranges,
	 const uint32_t* point_list,
	 int W, int H,
	 const float2* normals,
	 const float* offsets,
	 const float2* points_xy_image,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const float* colors,
	 const float4* conic_opacity,
	 const float* depths,
	 const float2* phi_center,
	 float* final_T,
	 uint32_t* n_contrib,
	 const float* bg_color,
	 float* out_color,
	 float* out_others,
	float* max_blending)
 {
	 renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		 ranges,
		 point_list,
		 W, H,
		 normals,
		 offsets,
		 points_xy_image,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 colors,
		 conic_opacity,
		 depths,
		 phi_center,
		 final_T,
		 n_contrib,
		 bg_color,
		 out_color,
		 out_others,
		 max_blending
		 );
 }
 
 void FORWARD::preprocess(int P, int D, int M,
	 const float* triangles_points,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const float* opacities,
	 float* scaling,
	 float* density_factor,
	 const float* shs,
	 bool* clamped,
	 const float* colors_precomp,
	 const float* viewmatrix,
	 const float* projmatrix,
	 const glm::vec3* cam_pos,
	 const int W, int H,
	 const float focal_x, float focal_y,
	 const float tan_fovx, float tan_fovy,
	 int* radii,
	 float2* normals,
	 float* offsets,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 float2* means2D,
	 float* depths,
	 float* rgb,
	 float4* conic_opacity,
	 float* cov3Ds,
	 float2* phi_center,
	 uint2* rect_min,
	 uint2* rect_max,
	 const dim3 grid,
	 uint32_t* tiles_touched,
	 bool prefiltered)
 {
	 preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		 P, D, M,
		 triangles_points,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 opacities,
		 scaling,
		 density_factor,
		 shs,
		 clamped,
		 colors_precomp,
		 viewmatrix, 
		 projmatrix,
		 cam_pos,
		 W, H,
		 tan_fovx, tan_fovy,
		 focal_x, focal_y,
		 radii,
		 normals,
		 offsets,
		 p_w,
		 p_image,
		 indices,
		 means2D,
		 depths,
		 rgb,
		 conic_opacity,
		 cov3Ds,
		 phi_center,
		 rect_min,
		 rect_max,
		 grid,
		 tiles_touched,
		 prefiltered
		 );
 }