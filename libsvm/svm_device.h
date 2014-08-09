/*
** Copyright 2014 Edward Walker
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
** http ://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
** 
** Description: Function prototypes for svm_device.cu
** @author: Ed Walker
*/
#ifndef _CUDA_DEVICE_FUNCTION_H_
#define _CUDA_DEVICE_FUNCTION_H_
#include "svm_defs.h"

/*********** Device function kernels ************/

void launch_cuda_setup_x_square(size_t num_blocks, size_t block_size, int N);

void launch_cuda_setup_QD(size_t num_blocks, size_t block_size, int N);

void launch_cuda_compute_obj_diff(size_t num_blocks, size_t block_size, GradValue_t Gmax, CValue_t *dh_obj_diff_array, int *result_indx, int N);

void launch_cuda_update_gradient(size_t num_blocks, size_t block_size, int N);

void launch_cuda_init_gradient(size_t num_blocks, size_t block_size, int start, int step, int N);

void launch_cuda_prep_gmax(size_t num_blocks, size_t block_size, GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, int N);

void launch_cuda_compute_alpha(size_t num_blocks, size_t block_size);

void launch_cuda_update_alpha_status(size_t num_blocks, size_t block_size);

struct find_gmax_param
{
	GradValue_t *dh_gmax;
	GradValue_t *dh_gmax2;
	int *dh_gmax_idx;
	GradValue_t *result_gmax;
	GradValue_t *result_gmax2;
	int *result_gmax_idx;
};
/**
cuda_find_gmax:
shared memory requirement: block_size * (2 * sizeof(GradValue_t) + sizeof(int))
*/
void launch_cuda_find_gmax(size_t num_blocks, size_t block_size, size_t share_mem_size, find_gmax_param param, int N, bool debug);
/**
cuda_find_min_idx:
shared memory requirement: block_size * (sizeof(CValue_t) + sizeof(int))
*/
void launch_cuda_find_min_idx(size_t num_blocks, size_t block_size, size_t share_mem_size, CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N);



/****** NU Solver device function kernels *********/
struct find_nu_gmax_param
{
	GradValue_t *dh_gmaxp;
	GradValue_t *dh_gmaxp2;
	GradValue_t *dh_gmaxn;
	GradValue_t *dh_gmaxn2;
	int *dh_gmaxp_idx;
	int *dh_gmaxn_idx;
	GradValue_t *result_gmaxp;
	GradValue_t *result_gmaxp2;
	GradValue_t *result_gmaxn;
	GradValue_t *result_gmaxn2;
	int *result_gmaxp_idx;
	int *result_gmaxn_idx;
};

/**
cuda_find_nu_gmax:
shared memory requirement: block_size * (4 * sizeof(GradValue_t) + 2 * sizeof(int) )
*/
void launch_cuda_find_nu_gmax(size_t num_blocks, size_t block_size, size_t share_mem_size, find_nu_gmax_param param, int N);
/**
cuda_find_nu_min_idx:
shared memory requirement: block_size * (sizeof(CValue_t) + sizeof(int))
*/
void launch_cuda_find_nu_min_idx(size_t num_blocks, size_t block_size, size_t share_mem_size, CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N);
void launch_cuda_compute_nu_obj_diff(size_t num_blocks, size_t block_size, GradValue_t Gmaxp, GradValue_t Gmaxn, CValue_t *dh_obj_diff_array, int *result_indx, int N);
void launch_cuda_prep_nu_gmax(size_t num_blocks, size_t block_size, GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2,
	int *dh_gmaxp_idx, int *dh_gmaxn_idx, int N);

/********** Host functions ***********/
cudaError_t update_solver_variables(SChar_t *dh_y, CValue_t *dh_QD, GradValue_t *dh_G, GradValue_t *dh_alpha, char *dh_alpha_status, double Cp, double Cn);
cudaError_t update_rbf_variables(CValue_t *dh_x_square);
cudaError_t update_param_constants(const svm_parameter &param, int *dh_x, cuda_svm_node *dh_space, size_t dh_space_size, int l);
void unbind_texture();
void init_device_gradient(int block_size, int startj, int stepj, int N);

#endif