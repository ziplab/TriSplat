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
 
 #pragma once

 #include <iostream>
 #include <vector>
 #include "rasterizer.h"
 #include <cuda_runtime_api.h>
 
 namespace CudaRasterizer
 {
	 template <typename T>
	 static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	 {
		 std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		 ptr = reinterpret_cast<T*>(offset);
		 chunk = reinterpret_cast<char*>(ptr + count);
	 }
 
	 struct GeometryState
	 {
		 size_t scan_size;
		 float* depths;
		 char* scanning_space;
		 bool* clamped;
		 int* internal_radii;
		 float2* means2D;
		 float4* conic_opacity;
		 float* rgb;
		 uint32_t* point_offsets;
		 uint32_t* tiles_touched;
 
		 float* cov3D;
  
		 float* p_w;
		 float2* p_image;
		 int* indices;
 
		 float* offsets;
		 float2* normals;
 
		 float2* phi_center;

		 uint2* rect_min;
		 uint2* rect_max;

		 static GeometryState fromChunk(char*& chunk, size_t P, size_t total_nb_points);
	 };
 
	 struct ImageState
	 {
		 uint2* ranges;
		 uint32_t* n_contrib;
		 float* accum_alpha;
 
		 static ImageState fromChunk(char*& chunk, size_t N);
	 };
 
	 struct BinningState
	 {
		 size_t sorting_size;
		 uint64_t* point_list_keys_unsorted;
		 uint64_t* point_list_keys;
		 uint32_t* point_list_unsorted;
		 uint32_t* point_list;
		 char* list_sorting_space;
 
		 static BinningState fromChunk(char*& chunk, size_t P);
	 };
 
	 template <typename T>
	 size_t required(size_t P, size_t total_nb_points = 0);
 
	 // General case for states that don't need `total_nb_points`
	 template<typename T> 
	 size_t required(size_t P, size_t total_nb_points)
	 {
		 char* size = nullptr;
		 T::fromChunk(size, P);
		 return ((size_t)size) + 128;
	 }
 
	 // Specialization for GeometryState
	 template<> 
	 size_t required<GeometryState>(size_t P, size_t total_nb_points)
	 {
		 char* size = nullptr;
		 GeometryState::fromChunk(size, P, total_nb_points);
		 return ((size_t)size) + 128;
	 }
 };