/**
Function prototypes for cuda_device_functions.cu
@author: Ed Walker
*/
#ifndef _CUDA_DEVICE_FUNCTION_H_
#define _CUDA_DEVICE_FUNCTION_H_
#include "CudaSolver.h"

/*********** Device function kernels ************/

/**
cuda_find_min_idx:
shared memory requirement: block_size * (sizeof(CValue_t) + sizeof(int))
*/
__global__ void cuda_find_min_idx(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N);

__global__ void cuda_compute_obj_diff(GradValue_t Gmax, CValue_t *dh_obj_diff_array, int *result_indx, int N);

__global__ void cuda_update_gradient(int N);

__global__ void cuda_init_gradient(int start, int step, int N);

__global__ void cuda_prep_gmax(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, int N);

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
__global__ void cuda_find_gmax(find_gmax_param param, int N);

__global__ void cuda_compute_alpha();

__global__ void cuda_update_alpha_status();


/****** NU Solver device function kernels *********/

/**
cuda_find_nu_min_idx:
shared memory requirement: block_size * (sizeof(CValue_t) + sizeof(int))
*/
__global__ void cuda_find_nu_min_idx(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N);

__global__ void cuda_compute_nu_obj_diff(GradValue_t Gmaxp, GradValue_t Gmaxn, CValue_t *dh_obj_diff_array, int *result_indx, int N);

__global__ void cuda_prep_nu_gmax(GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2,
	int *dh_gmaxp_idx, int *dh_gmaxn_idx, int N);

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
__global__ void cuda_find_nu_gmax(find_nu_gmax_param param, int N);


/********** Host functions ***********/
cudaError_t update_solver_variables(SChar_t *dh_y, CValue_t *dh_QD, GradValue_t *dh_G, GradValue_t *dh_alpha, char *dh_alpha_status, double Cp, double Cn);
cudaError_t update_rbf_variables(CValue_t *dh_x_square);
cudaError_t update_param_constants(const svm_parameter &param, int *dh_x, cuda_svm_node *dh_space, size_t dh_space_size);
void unbind_texture();

#endif