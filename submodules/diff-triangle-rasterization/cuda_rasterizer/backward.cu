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

 #include "backward.h"
 #include "auxiliary.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;
 
 
 
 // Backward pass for conversion of spherical harmonics to RGB for
 // each Triangle.
 __device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dtriangle, glm::vec3* dL_dshs, const int cumsum_for_triangle, const int num_points_per_triangle)
 {
	 // Compute intermediate values, as it is done during forward
	 glm::vec3 pos = means;
	 glm::vec3 dir_orig = pos - campos;
	 glm::vec3 dir = dir_orig / glm::length(dir_orig);
 
	 glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
 
	 // Use PyTorch rule for clamping: if clamping was applied,
	 // gradient becomes 0.
	 glm::vec3 dL_dRGB = dL_dcolor[idx];
	 dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	 dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	 dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
 
	 glm::vec3 dRGBdx(0, 0, 0);
	 glm::vec3 dRGBdy(0, 0, 0);
	 glm::vec3 dRGBdz(0, 0, 0);
	 float x = dir.x;
	 float y = dir.y;
	 float z = dir.z;
 
	 // Target location for this Triangle to write SH gradients to
	 glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;
 
	 // No tricks here, just high school-level calculus.
	 float dRGBdsh0 = SH_C0;
	 dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	 if (deg > 0)
	 {
		 float dRGBdsh1 = -SH_C1 * y;
		 float dRGBdsh2 = SH_C1 * z;
		 float dRGBdsh3 = -SH_C1 * x;
		 dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		 dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		 dL_dsh[3] = dRGBdsh3 * dL_dRGB;
 
		 dRGBdx = -SH_C1 * sh[3];
		 dRGBdy = -SH_C1 * sh[1];
		 dRGBdz = SH_C1 * sh[2];
 
		 if (deg > 1)
		 {
			 float xx = x * x, yy = y * y, zz = z * z;
			 float xy = x * y, yz = y * z, xz = x * z;
 
			 float dRGBdsh4 = SH_C2[0] * xy;
			 float dRGBdsh5 = SH_C2[1] * yz;
			 float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			 float dRGBdsh7 = SH_C2[3] * xz;
			 float dRGBdsh8 = SH_C2[4] * (xx - yy);
			 dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			 dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			 dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			 dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			 dL_dsh[8] = dRGBdsh8 * dL_dRGB;
 
			 dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			 dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			 dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];
 
			 if (deg > 2)
			 {
				 float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				 float dRGBdsh10 = SH_C3[1] * xy * z;
				 float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				 float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				 float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				 float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				 float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				 dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				 dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				 dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				 dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				 dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				 dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				 dL_dsh[15] = dRGBdsh15 * dL_dRGB;
 
				 dRGBdx += (
					 SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					 SH_C3[1] * sh[10] * yz +
					 SH_C3[2] * sh[11] * -2.f * xy +
					 SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					 SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					 SH_C3[5] * sh[14] * 2.f * xz +
					 SH_C3[6] * sh[15] * 3.f * (xx - yy));
 
				 dRGBdy += (
					 SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					 SH_C3[1] * sh[10] * xz +
					 SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					 SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					 SH_C3[4] * sh[13] * -2.f * xy +
					 SH_C3[5] * sh[14] * -2.f * yz +
					 SH_C3[6] * sh[15] * -3.f * 2.f * xy);
 
				 dRGBdz += (
					 SH_C3[1] * sh[10] * xy +
					 SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					 SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					 SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					 SH_C3[5] * sh[14] * (xx - yy));
			 }
		 }
	 }
 
	 // The view direction is an input to the computation. View direction
	 // is influenced by the Triangle's mean, so SHs gradients
	 // must propagate back into 3D position.
	 glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
 
	 // Account for normalization of direction
	 float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });
 
	 glm::vec3 scaled_dL_dmean = glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z) / static_cast<float>(num_points_per_triangle);
 
	 // Gradients of loss w.r.t. Triangle means, but only the portion 
	 // that is caused because the mean affects the view-dependent color.
	 // Additional mean gradient is accumulated in below methods.
 
	 for (int i = 0; i < num_points_per_triangle; i++)
	 {
		 dL_dtriangle[cumsum_for_triangle + i] += scaled_dL_dmean;
	 }
 
 }
 
 
 
 
// Backward pass of the preprocessing steps, except
 // for the covariance computation and inversion
 // (those are handled by a previous kernel call)
 template<int C>
 __global__ void preprocessCUDA(
	 int P, int D, int M,
	 const float* triangles_points,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* proj,
	 const float* viewmatrix,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const glm::vec3* campos,
	 glm::vec3* dL_dtriangle,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 glm::vec3* dL_dmeans,
	 float3* dL_dmean2D,
	 float* dL_dcov3D,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh,
	float* dL_dsigma_factor)
 {
	 auto idx = cg::this_grid().thread_rank();
	 if (idx >= P || !(radii[idx] > 0))
		 return;
 
	 const int cumsum_for_triangle = cumsum_of_points_per_triangle[idx];
	 const int offset = 3 * cumsum_for_triangle;
	 float3 center_triangle = {0.0f, 0.0f, 0.0f};
	 float sum_x[MAX_NB_POINTS] = {0.0f};
	 float sum_y[MAX_NB_POINTS] = {0.0f};
	 float sum_z[MAX_NB_POINTS] = {0.0f};
	 for (int i = 0; i < 3; i++) {
		 center_triangle.x += triangles_points[offset + 3 * i];
		 center_triangle.y += triangles_points[offset + 3 * i + 1];
		 center_triangle.z += triangles_points[offset + 3 * i + 2];
	 }
 
	 float3 total_sum = {center_triangle.x, center_triangle.y, center_triangle.z};
 
	 center_triangle.x /= 3;
	 center_triangle.y /= 3;
	 center_triangle.z /= 3;

	 float3 p_view_triangle;
	 if (!in_frustum_triangle(idx, center_triangle, viewmatrix, proj, false, p_view_triangle)){
		 return;
	 }

	 // Initialize loss accumulators for normals and offsets
	 float loss_points_x[MAX_NB_POINTS] = {0.0f};
	 float loss_points_y[MAX_NB_POINTS] = {0.0f};
 
	 
	 for (int i = 0; i < 3; i++) {
		float dL_dnormal_x = dL_dnormals[cumsum_for_triangle + i].x;
		float dL_dnormal_y = dL_dnormals[cumsum_for_triangle + i].y;
		float dL_doffset = dL_doffsets[cumsum_for_triangle + i];

		float2 p1_conv = p_image[cumsum_for_triangle + i];
		float2 p2_conv = p_image[cumsum_for_triangle + (i + 1) % 3];

		// Calculate unnormalized normal components
		float nx = p2_conv.y - p1_conv.y;
		float ny = -(p2_conv.x - p1_conv.x);
		float norm = __fsqrt_rn(nx * nx + ny * ny);
		float inv_norm = 1.0f / norm;
	
		// Calculate normalized normal and offset
		float2 normal = {nx * inv_norm, ny * inv_norm};
		float offset = -(normal.x * p1_conv.x + normal.y * p1_conv.y);

		if (normal.x * points_xy_image[idx].x + normal.y * points_xy_image[idx].y + offset > 0) {
			dL_dnormal_x = -dL_dnormal_x;
			dL_dnormal_y = -dL_dnormal_y;
			dL_doffset = -dL_doffset;
		}

		// Add gradients from offset to normal gradients
		dL_dnormal_x += (-p1_conv.x) * dL_doffset;
		dL_dnormal_y += (-p1_conv.y) * dL_doffset;
	
		// Backprop through normalization
		float norm_sq = norm * norm;
		float inv_norm_cubed = 1.0f / (norm * norm_sq);
		float dL_dnx = (dL_dnormal_x * ny * ny - dL_dnormal_y * nx * ny) * inv_norm_cubed;
		float dL_dny = (-dL_dnormal_x * nx * ny + dL_dnormal_y * nx * nx) * inv_norm_cubed;
	
		// Compute gradients from unnormalized normal to points
		float2 dL_dp1_conv = {dL_dny, -dL_dnx}; // From ny: p1_conv.x, from nx: p1_conv.y
		float2 dL_dp2_conv = {-dL_dny, dL_dnx}; // From ny: p2_conv.x, from nx: p2_conv.y
	
		// Add gradients from offset to p1_conv
		dL_dp1_conv.x += (-normal.x) * dL_doffset;
		dL_dp1_conv.y += (-normal.y) * dL_doffset;

		loss_points_x[indices[cumsum_for_triangle + i]] += dL_dp1_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + i]] += dL_dp1_conv.y;

		loss_points_x[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.y;
	}
 

	 float3 dL_ddepht = {0.0f, 0.0f, dL_dmean2D[idx].x};
	 float3 transposed_dL_ddepth = transformPoint4x3Transpose(dL_ddepht, viewmatrix);

 
	 for (int i = 0; i < 3; i++) {
 
		 float mul1 = (proj[0] * triangles_points[offset + 3 * i] + proj[4] * triangles_points[offset + 3 * i + 1] + proj[8] * triangles_points[offset + 3 * i + 2] + proj[12]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
		 float mul2 = (proj[1] * triangles_points[offset + 3 * i] + proj[5] * triangles_points[offset + 3 * i + 1] + proj[9] * triangles_points[offset + 3 * i + 2] + proj[13]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
		 dL_dtriangle[cumsum_for_triangle + i].x = (proj[0] * p_w[cumsum_for_triangle + i] - proj[3] * mul1) * loss_points_x[i]  + (proj[1] * p_w[cumsum_for_triangle + i] - proj[3] * mul2) * loss_points_y[i] + transposed_dL_ddepth.x / 3;
		 dL_dtriangle[cumsum_for_triangle + i].y = (proj[4] * p_w[cumsum_for_triangle + i] - proj[7] * mul1) * loss_points_x[i] + (proj[5] * p_w[cumsum_for_triangle + i] - proj[7] * mul2) * loss_points_y[i] + transposed_dL_ddepth.y / 3;
		 dL_dtriangle[cumsum_for_triangle + i].z = (proj[8] * p_w[cumsum_for_triangle + i] - proj[11] * mul1) * loss_points_x[i] + (proj[9] * p_w[cumsum_for_triangle + i] - proj[11] * mul2) * loss_points_y[i] + transposed_dL_ddepth.z / 3;
	 
	 }
 
	 // Compute gradient updates due to computing colors from SHs
	 if (shs)
		 computeColorFromSH(idx, D, M, (glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z), *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dtriangle, (glm::vec3*)dL_dsh, cumsum_for_triangle, 3);

	 
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
	 float3 v3 = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
 
	 float3 unnorm_ross_prod = make_float3(
		 v1.y * v2.z - v1.z * v2.y,
		 v1.z * v2.x - v1.x * v2.z,
		 v1.x * v2.y - v1.y * v2.x
	 );
	 float3 cross_prod = transformVec4x3(unnorm_ross_prod, viewmatrix);
 
	 float length = sqrtf(cross_prod.x*cross_prod.x + cross_prod.y*cross_prod.y + cross_prod.z*cross_prod.z);
	 if (length > 1e-8f) {
		 cross_prod.x /= length;
		 cross_prod.y /= length;
		 cross_prod.z /= length;
	 }
	 normal_cvx = cross_prod;

	 float3 p_view_triangle_ = transformPoint4x3(center_triangle, viewmatrix);
	 // we normalize such that we have a unit vector and cos is between -1 and 1
	 float length_viewpoint = sqrtf(p_view_triangle_.x*p_view_triangle_.x + p_view_triangle_.y*p_view_triangle_.y + p_view_triangle_.z*p_view_triangle_.z);
	 length_viewpoint = max(length_viewpoint, 1e-4f);
	 float3 normalized_camera_center;
	 normalized_camera_center.x = p_view_triangle_.x / length_viewpoint;
	 normalized_camera_center.y = p_view_triangle_.y / length_viewpoint;
	 normalized_camera_center.z = p_view_triangle_.z / length_viewpoint;
 
	 float3 dir = make_float3(
		 normalized_camera_center.x * normal_cvx.x,
		 normalized_camera_center.y * normal_cvx.y,
		 normalized_camera_center.z * normal_cvx.z
	 );
	 
	 float cos = -sumf3(dir);
	 
	 const float threshold = 0.001f;
	 if (fabsf(cos) < threshold) {
		return;
	 }
		
	 float multiplier = cos > 0 ? 1 : -1;
	 normal_cvx = {cross_prod.x * multiplier, cross_prod.y * multiplier, cross_prod.z * multiplier};
 
	 // ## BACKWARD
	 float3 dL_dtn = {dL_dnormal3D[idx * 3 + 0]*multiplier, dL_dnormal3D[idx * 3 + 1]*multiplier, dL_dnormal3D[idx * 3 + 2]*multiplier};
 
	 float matrix_w0[9], matrix_w1[9], matrix_w2[9];
 
	 matrix_w0[0] = 0.0f;   matrix_w0[1] = -v3.z; matrix_w0[2] = v3.y;
	 matrix_w0[3] = v3.z;   matrix_w0[4] = 0.0f;  matrix_w0[5] = -v3.x;
	 matrix_w0[6] = -v3.y;  matrix_w0[7] = v3.x;  matrix_w0[8] = 0.0f;
 
	 matrix_w1[0] = 0.0f;   matrix_w1[1] = v2.z;  matrix_w1[2] = -v2.y;
	 matrix_w1[3] = -v2.z;  matrix_w1[4] = 0.0f;  matrix_w1[5] = v2.x;
	 matrix_w1[6] = v2.y;   matrix_w1[7] = -v2.x; matrix_w1[8] = 0.0f;
 
	 matrix_w2[0] = 0.0f;   matrix_w2[1] = -v1.z; matrix_w2[2] = v1.y;
	 matrix_w2[3] = v1.z;   matrix_w2[4] = 0.0f;  matrix_w2[5] = -v1.x;
	 matrix_w2[6] = -v1.y;  matrix_w2[7] = v1.x;  matrix_w2[8] = 0.0f;
 
	 float normal_transpose[9];
	 normal_transpose[0] = normal_cvx.x*normal_cvx.x; 
	 normal_transpose[1] = normal_transpose[3] = normal_cvx.x*normal_cvx.y;
	 normal_transpose[2] = normal_transpose[6] = normal_cvx.x*normal_cvx.z;
	 normal_transpose[4] = normal_cvx.y*normal_cvx.y;
	 normal_transpose[5] = normal_transpose[7] = normal_cvx.y*normal_cvx.z;
	 normal_transpose[8] = normal_cvx.z*normal_cvx.z;
 
	 float length_inv = 1 / length;
 
	 float projection_matrix[9];
	 projection_matrix[0] = viewmatrix[0];
	 projection_matrix[1] = viewmatrix[4];
	 projection_matrix[2] = viewmatrix[8];
	 projection_matrix[3] = viewmatrix[1];
	 projection_matrix[4] = viewmatrix[5];
	 projection_matrix[5] = viewmatrix[9];
	 projection_matrix[6] = viewmatrix[2];
	 projection_matrix[7] = viewmatrix[6];
	 projection_matrix[8] = viewmatrix[10];
 
	 float matrix_w0_transformed[9], matrix_w1_transformed[9], matrix_w2_transformed[9];
	 transformMat3x3(projection_matrix, matrix_w0, matrix_w0_transformed);
	 transformMat3x3(projection_matrix, matrix_w1, matrix_w1_transformed);
	 transformMat3x3(projection_matrix, matrix_w2, matrix_w2_transformed);
 
	 float norm_times_matrix0[9], norm_times_matrix1[9], norm_times_matrix2[9];
	 transformMat3x3(normal_transpose, matrix_w0_transformed, norm_times_matrix0);
	 transformMat3x3(normal_transpose, matrix_w1_transformed, norm_times_matrix1);
	 transformMat3x3(normal_transpose, matrix_w2_transformed, norm_times_matrix2);
 
	 float matrix_substraction0[9], matrix_substraction1[9], matrix_substraction2[9];
	 substractionMat3x3(matrix_w0_transformed, norm_times_matrix0, matrix_substraction0);
	 substractionMat3x3(matrix_w1_transformed, norm_times_matrix1, matrix_substraction1);
	 substractionMat3x3(matrix_w2_transformed, norm_times_matrix2, matrix_substraction2);
 
	 float dL_dp0x = length_inv * matrix_substraction0[0] * dL_dtn.x + length_inv * matrix_substraction0[3] * dL_dtn.y + length_inv * matrix_substraction0[6] * dL_dtn.z;
	 float dL_dp0y = length_inv * matrix_substraction0[1] * dL_dtn.x + length_inv * matrix_substraction0[4] * dL_dtn.y + length_inv * matrix_substraction0[7] * dL_dtn.z;
	 float dL_dp0z = length_inv * matrix_substraction0[2] * dL_dtn.x + length_inv * matrix_substraction0[5] * dL_dtn.y + length_inv * matrix_substraction0[8] * dL_dtn.z;
 
	 float dL_dp1x = length_inv * matrix_substraction1[0] * dL_dtn.x + length_inv * matrix_substraction1[3] * dL_dtn.y + length_inv * matrix_substraction1[6] * dL_dtn.z;
	 float dL_dp1y = length_inv * matrix_substraction1[1] * dL_dtn.x + length_inv * matrix_substraction1[4] * dL_dtn.y + length_inv * matrix_substraction1[7] * dL_dtn.z;
	 float dL_dp1z = length_inv * matrix_substraction1[2] * dL_dtn.x + length_inv * matrix_substraction1[5] * dL_dtn.y + length_inv * matrix_substraction1[8] * dL_dtn.z;
 
	 float dL_dp2x = length_inv * matrix_substraction2[0] * dL_dtn.x + length_inv * matrix_substraction2[3] * dL_dtn.y + length_inv * matrix_substraction2[6] * dL_dtn.z;
	 float dL_dp2y = length_inv * matrix_substraction2[1] * dL_dtn.x + length_inv * matrix_substraction2[4] * dL_dtn.y + length_inv * matrix_substraction2[7] * dL_dtn.z;
	 float dL_dp2z = length_inv * matrix_substraction2[2] * dL_dtn.x + length_inv * matrix_substraction2[5] * dL_dtn.y + length_inv * matrix_substraction2[8] * dL_dtn.z;


	 dL_dtriangle[cumsum_for_triangle + 0].x += dL_dp0x;
	 dL_dtriangle[cumsum_for_triangle + 0].y += dL_dp0y;
	 dL_dtriangle[cumsum_for_triangle + 0].z += dL_dp0z;
 
	 dL_dtriangle[cumsum_for_triangle + 1].x += dL_dp1x;
	 dL_dtriangle[cumsum_for_triangle + 1].y += dL_dp1y;
	 dL_dtriangle[cumsum_for_triangle + 1].z += dL_dp1z;
 
	 dL_dtriangle[cumsum_for_triangle + 2].x += dL_dp2x;
	 dL_dtriangle[cumsum_for_triangle + 2].y += dL_dp2y;
	 dL_dtriangle[cumsum_for_triangle + 2].z += dL_dp2z; 




 }
 
 // Backward version of the rendering procedure.
 template <uint32_t C>
 __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
 renderCUDA(
	 const uint2* __restrict__ ranges,
	 const uint32_t* __restrict__ point_list,
	 int W, int H,
	 const float* __restrict__ bg_color,
	 const float* __restrict__ sigma,
	 const int* __restrict__ num_points_per_triangle,
	 const int* __restrict__ cumsum_of_points_per_triangle,
	 const float2* __restrict__ normals,
	 const float* __restrict__ offsets,
	 const float4* __restrict__ conic_opacity,
	 const float* __restrict__ depths,
	 const float2* __restrict__ means2D,
	 const float2* __restrict__ phi_center,
	 const float* __restrict__ colors,
	 const float* __restrict__ final_Ts,
	 const uint32_t* __restrict__ n_contrib,
	 const float* __restrict__ dL_dpixels,
	 const float* __restrict__ dL_depths,
	 float2* __restrict__ dL_dnormals,
	 float* __restrict__ dL_doffsets,
	 float* __restrict__ dL_dsigma,
	 float3* __restrict__ dL_dmean2D,
	 float4* __restrict__ dL_dconic2D,
	 float* __restrict__ dL_dopacity,
	 float* __restrict__ dL_dnormal3D,
	 float* __restrict__ dL_dcolors,
	 float* __restrict__ dL_dsigma_factor)
 {
	 // We rasterize again. Compute necessary block info.
	 auto block = cg::this_thread_block();
	 const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	 const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	 const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	 const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	 const uint32_t pix_id = W * pix.y + pix.x;
	 const float2 pixf = { (float)pix.x, (float)pix.y };
 
	 const bool inside = pix.x < W&& pix.y < H;
	 const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
 
	 const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
	 bool done = !inside;
	 int toDo = range.y - range.x;
 
	 __shared__ int collected_id[BLOCK_SIZE];
	 __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	 __shared__ float collected_colors[C * BLOCK_SIZE];
 
	 /*
	 ADDED FOR Triangle PURPOSES ==========================================================================
	 */
	 __shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ int collected_cumsum_of_points_per_triangle[BLOCK_SIZE];
	 __shared__ float collected_sigma[BLOCK_SIZE];
	 __shared__ float collected_depths[BLOCK_SIZE];
	 __shared__ float2 collected_xy[BLOCK_SIZE];
	 __shared__ float2 collected_phi_center[BLOCK_SIZE];
	 /*
	 ===================================================================================================
	 */
 
	 // In the forward, we stored the final value for T, the
	 // product of all (1 - alpha) factors. 
	 const float T_final = inside ? final_Ts[pix_id] : 0;
	 float T = T_final;
 
	 // We start from the back. The ID of the last contributing
	 // Triangle is known from each pixel from the forward.
	 uint32_t contributor = toDo;
	 const int last_contributor = inside ? n_contrib[pix_id] : 0;
 
	 float accum_rec[C] = { 0 };
	 float dL_dpixel[C];
	 if (inside)
		 for (int i = 0; i < C; i++)
			 dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];


	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
 
	 float last_alpha = 0;
	 float last_color[C] = { 0 };

 
	 // Traverse all triangles
	 for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	 {
		 // Load auxiliary data into shared memory, start in the BACK
		 // and load them in revers order.
		 block.sync();
		 const int progress = i * BLOCK_SIZE + block.thread_rank();
		 if (range.x + progress < range.y)
		 {
			 const int coll_id = point_list[range.y - progress - 1];
			 collected_id[block.thread_rank()] = coll_id;
			 collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			 for (int i = 0; i < C; i++)
				 collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			 collected_cumsum_of_points_per_triangle[block.thread_rank()] = cumsum_of_points_per_triangle[coll_id];
			 collected_sigma[block.thread_rank()] = sigma[coll_id];
			 collected_depths[block.thread_rank()] = depths[coll_id];
			 collected_xy[block.thread_rank()] = means2D[coll_id];
			 for (int k = 0; k < 3; k++) {
				collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[cumsum_of_points_per_triangle[coll_id] + k];
				collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[cumsum_of_points_per_triangle[coll_id] + k];
			}
			collected_phi_center[block.thread_rank()] = phi_center[coll_id];
		 }
		 block.sync();
 
		 // Iterate over triangles
		 for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		 {
 
			 contributor--;
			 if (contributor >= last_contributor)
				 continue;
 
			 float4 con_o = collected_conic_opacity[j];
			 float normal[3] = {con_o.x, con_o.y, con_o.z};
			 float2 phi_center_min = collected_phi_center[j];
			 float distances[MAX_NB_POINTS];
			 float sigma_pre = collected_sigma[j];
			 float depth = collected_depths[j];
			 float sum_exp = 0.0f;
			 float max_val = -INFINITY;
			 int base = j * MAX_NB_POINTS;
			 bool outside = false;
			 float c_d = collected_depths[j];
 
			 for (int k = 0; k < 3; k++) {
				 // Compute the current distance
				 float dist = (collected_normals[base + k].x * pixf.x
						  + collected_normals[base + k].y * pixf.y
						  + collected_offsets[base + k]);
				
				 if (dist > 0) {
					outside = true;
					break;
				 }
 
				 distances[k] = dist;
				 max_val = fmaxf(max_val, dist);
			 }

			 if (outside)
				 continue;
 
			 float phi_x = max_val;
			 float phi_final = phi_x * phi_center_min.x;
			 float Cx = fmaxf(0.0f,  __powf(phi_final, sigma_pre));
 
			 const float alpha = min(0.99f, con_o.w * Cx);
 
			 if (alpha < 1.0f / 255.0f)
				 continue;
 
			 T = T / (1.f - alpha);
			 const float dchannel_dcolor = alpha * T;
 
			 // Propagate gradients to per-Triangle colors and keep
			 // gradients w.r.t. alpha (blending factor for a Triangle/pixel
			 // pair).
			 float dL_dalpha = 0.0f;
			 const int global_id = collected_id[j];
			 for (int ch = 0; ch < C; ch++)
			 {
				 const float c = collected_colors[ch * BLOCK_SIZE + j];
				 // Update last color (to be used in the next iteration)
				 accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				 last_color[ch] = c;
 
				 const float dL_dchannel = dL_dpixel[ch];
				 dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				 // Update the gradients w.r.t. color of the Triangle. 
				 // Atomic, since this pixel is just one of potentially
				 // many that were affected by this Triangle.
				 atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			 }

			 float dL_dz = 0.0f;
			 float dL_dweight = 0;
 
			 const float m_d = far_n / (far_n - near_n) * (1 - near_n / collected_depths[j]);
			  const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * collected_depths[j] * collected_depths[j]);
			  if (contributor == median_contributor-1) {
				  dL_dz += dL_dmedian_depth;
			  }
 
			 dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
			 dL_dalpha += dL_dweight - last_dL_dT;
			 // propagate the current weight W_{i} to next weight W_{i-1}
			 last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			 const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			 dL_dz += dL_dmd * dmd_dd;
 
			 // Propagate gradients w.r.t ray-splat depths
			 accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			 last_depth = collected_depths[j];
			 dL_dalpha += (collected_depths[j] - accum_depth_rec) * dL_ddepth;
			 // Propagate gradients w.r.t. color ray-splat alphas
			 accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			 dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;
  
			 for (int ch = 0; ch < 3; ch++) {
				 accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				 last_normal[ch] = normal[ch];
				 dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				 atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			 }
 
			 dL_dalpha *= T;
			 // Update last alpha (to be used in the next iteration)
			 last_alpha = alpha;
 
			 // Account for fact that alpha also influences how much of
			 // the background color is added if nothing left to blend
			 float bg_dot_dpixel = 0;
			 for (int i = 0; i < C; i++)
				 bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			 dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			 dL_dz += alpha * T * dL_ddepth; 
			 atomicAdd(&(dL_dmean2D[global_id].x), dL_dz);
 
			 // Helpful reusable temporary variables
			 const float dL_dC = con_o.w * dL_dalpha;

			 if (phi_final > 0.0f) {
				// derivative with respect to sigma
				float dL_dsigma_value = dL_dC * Cx * __logf(phi_final);
				atomicAdd(&dL_dsigma[global_id], dL_dsigma_value);
			}
			
			// Calculate gradient w.r.t phi_x 
			float dL_dphi_x = dL_dC * (sigma_pre / phi_x) * Cx;
 
			 #pragma unroll
			 for (int k = 0; k < 3; k++) {
				if (fabsf(distances[k] - max_val) < 1e-6f) {
					float dL_dnormal_x = dL_dphi_x * pixf.x;
					float dL_dnormal_y = dL_dphi_x * pixf.y;
					atomicAdd(&(dL_dnormals[collected_cumsum_of_points_per_triangle[j] + k].x), dL_dnormal_x);
					atomicAdd(&(dL_dnormals[collected_cumsum_of_points_per_triangle[j] + k].y), dL_dnormal_y);
					atomicAdd(&(dL_doffsets[collected_cumsum_of_points_per_triangle[j] + k]), dL_dphi_x);
				}
			 }
 
			 // Update gradients w.r.t. opacity of the Triangle
			 atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * Cx);
 
		 }
	 }
 }
 
 void BACKWARD::preprocess(
	 int P, int D, int M,
	 const float* triangles_points,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* viewmatrix,
	 const float* projmatrix,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const float* cov3Ds,
	 const float focal_x, float focal_y,
	 const float tan_fovx, float tan_fovy,
	 const glm::vec3* campos,
	 glm::vec3* dL_dtriangle,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 glm::vec3* dL_dmean3D,
	 float3* dL_dmean2D,
	 const float* dL_dconic,
	 float* dL_dcov3D,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh,
	 float* dL_dsigma_factor
	 )
 {
	 
	 // Propagate gradients for remaining steps: finish 3D mean gradients,
	 // propagate color gradients to SH (if desireD), propagate 3D covariance
	 // matrix gradients to scale and rotation.
	 preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		 P, D, M,
		 triangles_points,
		 W, H,
		 radii,
		 shs,
		 clamped,
		 projmatrix,
		 viewmatrix,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 points_xy_image,
		 p_w,
		 p_image,
		 indices,
		 campos,
		 (glm::vec3*)dL_dtriangle,
		 (float2*) dL_dnormals,
		 dL_doffsets,
		 (glm::vec3*)dL_dmean3D,
		 (float3*)dL_dmean2D,
		 dL_dcov3D,
		 dL_dnormal3D,
		 dL_dcolor,
		 dL_dsh,
		 dL_dsigma_factor);
 }
 
 void BACKWARD::render(
	 const dim3 grid, const dim3 block,
	 const uint2* ranges,
	 const uint32_t* point_list,
	 int W, int H,
	 const float* bg_color,
	 const float* sigma,
	 const int* num_points_per_triangle,
	 const int* cumsum_of_points_per_triangle,
	 const float2* normals,
	 const float* offsets,
	 const float4* conic_opacity,
	 const float* depths,
	 const float2* means2D,
	 const float2* phi_center,
	 const float* colors,
	 const float* final_Ts,
	 const uint32_t* n_contrib,
	 const float* dL_dpixels,
	 const float* dL_depths,
	 float2* dL_dnormals,
	 float* dL_doffsets,
	 float* dL_dsigma,
	 float3* dL_dmean2D,
	 float4* dL_dconic2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolors,
	 float* dL_dsigma_factor)
 {
	 renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		 ranges,
		 point_list,
		 W, H,
		 bg_color,
		 sigma,
		 num_points_per_triangle,
		 cumsum_of_points_per_triangle,
		 normals,
		 offsets,
		 conic_opacity,
		 depths,
		 means2D,
		 phi_center,
		 colors,
		 final_Ts,
		 n_contrib,
		 dL_dpixels,
		 dL_depths,
		 dL_dnormals,
		 dL_doffsets,
		 dL_dsigma,
		 dL_dmean2D,
		 dL_dconic2D,
		 dL_dopacity,
		 dL_dnormal3D,
		 dL_dcolors,
		 dL_dsigma_factor
		 );
 }