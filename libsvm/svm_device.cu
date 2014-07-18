#include "CudaSolver.h"
#include "math.h"

#include "svm.h"
#include <stdexcept>
#include <iostream>
using namespace std;
#include "svm_device.h"
#include "CudaReducer.h"

enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

#ifdef USE_CONSTANT_SVM_NODE
__constant__ cuda_svm_node      *d_space;
#else
texture<float2, 1, cudaReadModeElementType> d_tex_space;
#endif
__constant__	int				*d_x;
__device__		int				d_kernel_type;	// enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
__device__		int				d_svm_type;		// enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
__constant__	double			d_gamma;
__constant__	double			d_coef0;
__constant__	int				d_degree;

__constant__	CValue_t		*d_x_square;
__constant__	CValue_t		*d_QD;
__constant__	SChar_t			*d_y;
__constant__	double			d_Cp;
__constant__	double			d_Cn;

__device__		GradValue_t		*d_G;
__device__		GradValue_t		*d_alpha;
__device__		char			*d_alpha_status;

__device__		GradValue_t		d_delta_alpha_i;
__device__		GradValue_t		d_delta_alpha_j;

__device__		int2			d_solver; // member x and y hold the selected i and j working set indices respectively
__device__		int2			d_nu_solver; // member x and y hold the Gmaxp_idx and Gmaxn_idx indices respectively.  

cudaError_t update_param_constants(const svm_parameter &param, int *dh_x, cuda_svm_node *dh_space, size_t dh_space_size)
{
	cudaError_t err;
	err = cudaMemcpyToSymbol(d_kernel_type, &param.kernel_type, sizeof(param.kernel_type));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_kernel_type\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_svm_type, &param.svm_type, sizeof(param.svm_type));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_svm_type\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_gamma, &param.gamma, sizeof(param.gamma));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_gamma\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_coef0, &param.coef0, sizeof(param.coef0));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_coef0\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_degree, &param.degree, sizeof(param.degree));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_degree\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_x, &dh_x, sizeof(dh_x));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_x\n");
		return err;
	}
#ifdef USE_CONSTANT_SVM_NODE
	err = cudaMemcpyToSymbol(d_space, &dh_space, sizeof(dh_space));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_space\n");
		return err;
	}
#else
	err = cudaBindTexture(0, d_tex_space, dh_space, dh_space_size);
#endif
	return err;
}

cudaError_t update_solver_variables(SChar_t *dh_y, CValue_t *dh_QD, GradValue_t *dh_G, GradValue_t *dh_alpha, char *dh_alpha_status, double Cp, double Cn)
{
	cudaError_t err;

	err = cudaMemcpyToSymbol(d_y, &dh_y, sizeof(dh_y));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_y\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_QD, &dh_QD, sizeof(dh_QD));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_QD\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_G, &dh_G, sizeof(dh_G));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_G\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_alpha, &dh_alpha, sizeof(dh_alpha));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_alpha\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_alpha_status, &dh_alpha_status, sizeof(dh_alpha_status));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error copying to symbol d_alpha_status\n");
		return err;
	}

	err = cudaMemcpyToSymbol(d_Cp, &Cp, sizeof(Cp));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_Cp\n");
		return err;
	}
	err = cudaMemcpyToSymbol(d_Cn, &Cn, sizeof(Cn));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_Cn\n");
		return err;
	}
	return err;
}

cudaError_t update_rbf_variables(CValue_t *dh_x_square)
{
	cudaError_t err;
	if (dh_x_square != nullptr) {
		err = cudaMemcpyToSymbol(d_x_square, &dh_x_square, sizeof(dh_x_square));
		if (err != cudaSuccess) {
			fprintf(stderr, "Error copying to symbol d_x_square\n");
			return err;
		}
	}
	return err;
}

void unbind_texture()
{
#ifndef USE_CONSTANT_SVM_NODE
	cudaUnbindTexture(d_tex_space);
#endif
}


__device__ __forceinline__ cuda_svm_node get_col_value(int i)
{
#ifdef USE_CONSTANT_SVM_NODE
	return d_space[i];
#else
	return tex1Dfetch(d_tex_space, i);
#endif
}

/**
Compute dot product of 2 vectors
*/
__device__ CValue_t dot(int i, int j)
{
	int i_col = d_x[i];
	int j_col = d_x[j];
	/**
	remember: cuda_svm_node.x == svm_node.index
	cuda_svm_node.y == svm_node.value
	*/
#define index x
#define value y

	cuda_svm_node x = get_col_value(i_col);
	cuda_svm_node y = get_col_value(j_col);

	double sum = 0;
	while(x.x != -1 && y.x != -1)
	{
		if(x.index == y.index)
		{
			sum += x.value * y.value;
			x = get_col_value(++i_col);
			y = get_col_value(++j_col);
		}
		else
		{

			if(x.index > y.index) {
				y = get_col_value(++j_col);
			} else {
				x = get_col_value(++i_col);
			}
		}			
	}
	return sum;
}

__device__ CValue_t device_kernel_rbf(int i, int j)
{
	double q = d_x_square[i] + d_x_square[j] - 2 * dot(i,j);
	dbgprintf(false, "cuda_eval_kernel: x_square[%d]=%.10g x_square[%d]=%.10g q=%.10g\n",
		i, d_x_square[i], j, d_x_square[j], q);
	return exp(-d_gamma * q);
}

__device__ CValue_t device_kernel_poly(int i, int j)
{
	return pow(d_gamma * dot(i,j) + d_coef0, d_degree);
}

__device__ CValue_t device_kernel_sigmoid(int i, int j)
{
	return tanh(d_gamma * dot(i,j) + d_coef0);
}

__device__ CValue_t device_kernel_linear(int i, int j)
{
	return dot(i, j);
}

__device__ CValue_t device_kernel_precomputed(int i, int j)
{
	int i_col = d_x[i];
	int j_col = d_x[j];
	int offset = static_cast<int>(get_col_value(j_col).y);
	return get_col_value(i_col+offset).y;
	// return x[i][(int)(x[j][0].value)].value;
}

__device__ CValue_t cuda_evalQ(int i, int j)
{
	CValue_t q = 0; 
	switch(d_kernel_type)
	{
	case RBF:
		q = device_kernel_rbf(i, j);
		break;
	case POLY:
		q = device_kernel_poly(i, j);
		break;
	case LINEAR:
		q = device_kernel_linear(i, j);
		break;
	case SIGMOID:
		q = device_kernel_sigmoid(i, j);
		break;
	case PRECOMPUTED:
		q = device_kernel_precomputed(i, j);
		break;
	}

	switch (d_svm_type) 
	{
	case C_SVC:
	case NU_SVC:
		// SVC_Q
		return (CValue_t)(d_y[i] * d_y[j] * q);
	case ONE_CLASS:
		// ONE_CLASS_Q
		return q;
	case EPSILON_SVR:
	case NU_SVR:
		// SVR_Q
		return 0;
	default:
		return q;
	}
}

__global__ void cuda_find_min_idx(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N) 
{
	D_MinIdxFunctor func(obj_diff_array, obj_diff_indx, result_obj_min, result_indx); // Class defined in CudaReducer.h
	device_block_reducer(func, N); // Template function defined in CudaReducer.h
	if (blockIdx.x == 0)
		d_solver.y = func.return_idx();
}

__global__ void cuda_compute_obj_diff(GradValue_t Gmax, CValue_t *dh_obj_diff_array, int *result_indx, int N)
{
	int i = d_solver.x;

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= N)
		return ;

	dh_obj_diff_array[j] = CVALUE_MAX;
	result_indx[j] = -1;
	if(d_y[j] == 1)
	{
		if (!(d_alpha_status[j] == LOWER_BOUND)/*is_lower_bound(j)*/)
		{
			double grad_diff = Gmax + d_G[j];
			if (grad_diff > 1e-4) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[i] + d_QD[j] - 2.0 * d_y[i] * cuda_evalQ(i, j);
				CValue_t obj_diff = CVALUE_MAX;

				if (quad_coef > 0) {
					obj_diff = -(grad_diff*grad_diff)/quad_coef;
				} else {
					obj_diff = -(grad_diff*grad_diff)/TAU;
				}
				CHECK_FLT_RANGE(obj_diff);
				CHECK_FLT_INF(obj_diff);
				dh_obj_diff_array[j] = obj_diff;
				result_indx[j] = j;
			}

		}
	}
	else
	{
		if (!(d_alpha_status[j] == UPPER_BOUND) /*is_upper_bound(j)*/ )
		{
			double grad_diff = Gmax - d_G[j];
			if (grad_diff > 1e-4) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[i] + d_QD[j] + 2.0 * d_y[i] * cuda_evalQ(i, j);
				CValue_t obj_diff = CVALUE_MAX;

				if (quad_coef > 0) {
					obj_diff = -(grad_diff*grad_diff)/quad_coef;
				} else {
					obj_diff = -(grad_diff*grad_diff)/TAU;
				}
				CHECK_FLT_RANGE(obj_diff);
				CHECK_FLT_INF(obj_diff);
				dh_obj_diff_array[j] = obj_diff;
				result_indx[j] = j;
			}
		}
	}

}

__global__ void cuda_update_gradient(int N)
{
	int i = d_solver.x; // d_selected_i;
	int j = d_solver.y; // d_selected_j;

	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (k >= N)
		return;

	d_G[k] += (cuda_evalQ(i, k) * d_delta_alpha_i + cuda_evalQ(j, k) * d_delta_alpha_j);
	// d_G[k] += t; // Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
}

__global__ void cuda_find_gmax(find_gmax_param param, int N)
{
	D_GmaxFunctor func(param.dh_gmax, param.dh_gmax2, param.dh_gmax_idx, param.result_gmax, param.result_gmax2, param.result_gmax_idx); // class defined in CudaReducer.h
	
	device_block_reducer(func, N); // Template function defined in CudaReducer.h
	
	if (blockIdx.x == 0)
		d_solver.x = func.return_idx();
}


__global__ void cuda_prep_gmax(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, int N)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if (t >= N)
		return ;

	dh_gmax[t] = -GRADVALUE_MAX;
	dh_gmax2[t] = -GRADVALUE_MAX;
	dh_gmax_idx[t] = -1;

	if(d_y[t] == +1)	
	{
		if(!(d_alpha_status[t] == UPPER_BOUND) /*is_upper_bound(t)*/) {
			dh_gmax[t] = -d_G[t];
			dh_gmax_idx[t] = t;
		}
		if (!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
			dh_gmax2[t] = d_G[t];
		}
	}
	else
	{
		if(!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
			dh_gmax[t] = d_G[t];
			dh_gmax_idx[t] = t;
		}
		if (!(d_alpha_status[t] == UPPER_BOUND) /*is_upper_bound(t)*/) {
			dh_gmax2[t] = -d_G[t];
		}
	}
}


__device__	__forceinline__ double device_get_C(int i)
{
	return (d_y[i] > 0)? d_Cp : d_Cn;
}

__global__ void cuda_compute_alpha()
{
	int i = d_solver.x; // d_selected_i;
	int j = d_solver.y; // d_selected_j;

	double C_i = device_get_C(i);
	double C_j = device_get_C(j);

	GradValue_t old_alpha_i = d_alpha[i];
	GradValue_t old_alpha_j = d_alpha[j];

	if(d_y[i] != d_y[j])
	{
		GradValue_t quad_coef = d_QD[i] + d_QD[j] + 2 * cuda_evalQ(i, j); //  Q_i[j];
		if (quad_coef <= 0)
			quad_coef = TAU;
		GradValue_t delta = (-d_G[i] - d_G[j])/quad_coef;
		GradValue_t diff = d_alpha[i] - d_alpha[j];
		d_alpha[i] += delta;
		d_alpha[j] += delta;

		if(diff > 0)
		{
			if(d_alpha[j] < 0)
			{
				d_alpha[j] = 0;
				d_alpha[i] = diff;
			}
		}
		else
		{
			if(d_alpha[i] < 0)
			{
				d_alpha[i] = 0;
				d_alpha[j] = -diff;
			}
		}
		if(diff > C_i - C_j)
		{
			if(d_alpha[i] > C_i)
			{
				d_alpha[i] = C_i;
				d_alpha[j] = C_i - diff;
			}
		}
		else
		{
			if(d_alpha[j] > C_j)
			{
				d_alpha[j] = C_j;
				d_alpha[i] = C_j + diff;
			}
		}
	}
	else
	{
		GradValue_t quad_coef = d_QD[i] + d_QD[j] - 2 * cuda_evalQ(i, j); // Q_i[j];
		if (quad_coef <= 0)
			quad_coef = TAU;
		GradValue_t delta = (d_G[i]-d_G[j])/quad_coef;
		GradValue_t sum = d_alpha[i] + d_alpha[j];
		d_alpha[i] -= delta;
		d_alpha[j] += delta;

		if(sum > C_i)
		{
			if(d_alpha[i] > C_i)
			{
				d_alpha[i] = C_i;
				d_alpha[j] = sum - C_i;
			}
		}
		else
		{
			if(d_alpha[j] < 0)
			{
				d_alpha[j] = 0;
				d_alpha[i] = sum;
			}
		}
		if(sum > C_j)
		{
			if(d_alpha[j] > C_j)
			{
				d_alpha[j] = C_j;
				d_alpha[i] = sum - C_j;
			}
		}
		else
		{
			if(d_alpha[i] < 0)
			{
				d_alpha[i] = 0;
				d_alpha[j] = sum;
			}
		}
	}
	d_delta_alpha_i = d_alpha[i] - old_alpha_i;
	d_delta_alpha_j = d_alpha[j] - old_alpha_j;
}

__device__ void device_update_alpha_status(int i)
{
	if(d_alpha[i] >= device_get_C(i))
		d_alpha_status[i] = UPPER_BOUND;
	else if(d_alpha[i] <= 0)
		d_alpha_status[i] = LOWER_BOUND;
	else 
		d_alpha_status[i] = FREE;
}

__global__ void cuda_update_alpha_status()
{
	int i = d_solver.x;
	int j = d_solver.y;

	device_update_alpha_status(i);
	device_update_alpha_status(j);
}

/*********** NU Solver ************/
__global__ void cuda_prep_nu_gmax(GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2,
	int *dh_gmaxp_idx, int *dh_gmaxn_idx, int N)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if (t >= N)
		return;

	dh_gmaxp[t] = -GRADVALUE_MAX;
	dh_gmaxp2[t] = -GRADVALUE_MAX;
	dh_gmaxn[t] = -GRADVALUE_MAX;
	dh_gmaxn2[t] = -GRADVALUE_MAX;
	dh_gmaxp_idx[t] = -1;
	dh_gmaxn_idx[t] = -1;

	if (d_y[t] == +1)
	{
		if (!(d_alpha_status[t] == UPPER_BOUND) /*is_upper_bound(t)*/) {
			dh_gmaxp[t] = -d_G[t];
			dh_gmaxp_idx[t] = t;
		}
		if (!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
			dh_gmaxp2[t] = d_G[t];
		}
	}
	else
	{
		if (!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
			dh_gmaxn[t] = d_G[t];
			dh_gmaxn_idx[t] = t;
		}
		if (!(d_alpha_status[t] == UPPER_BOUND) /*is_upper_bound(t)*/) {
			dh_gmaxn2[t] = -d_G[t];
		}
	}
}

__global__ void cuda_find_nu_gmax(find_nu_gmax_param param, int N)
{
	D_NuGmaxFunctor func(param.dh_gmaxp, param.dh_gmaxn, param.dh_gmaxp2, param.dh_gmaxn2, param.dh_gmaxp_idx, param.dh_gmaxn_idx,
		param.result_gmaxp, param.result_gmaxn, param.result_gmaxp2, param.result_gmaxn2, param.result_gmaxp_idx, param.result_gmaxn_idx);

	device_block_reducer(func, N);

	if (blockIdx.x == 0) {
		int ip, in;
		func.return_idx(ip, in);
		d_nu_solver.x = ip;
		d_nu_solver.y = in;
	}
}

__global__ void cuda_compute_nu_obj_diff(GradValue_t Gmaxp, GradValue_t Gmaxn, CValue_t *dh_obj_diff_array, int *result_indx, int N)
{
	int ip = d_nu_solver.x;
	int in = d_nu_solver.y;

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= N)
		return;

	dh_obj_diff_array[j] = CVALUE_MAX;
	result_indx[j] = -1;
	if (d_y[j] == 1)
	{
		if (!(d_alpha_status[j] == LOWER_BOUND)/*is_lower_bound(j)*/)
		{
			double grad_diff = Gmaxp + d_G[j];
			if (grad_diff > 1e-4) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[ip] + d_QD[j] - 2.0 * cuda_evalQ(ip, j); 
				CValue_t obj_diff = CVALUE_MAX;

				if (quad_coef > 0) {
					obj_diff = -(grad_diff*grad_diff) / quad_coef;
				}
				else {
					obj_diff = -(grad_diff*grad_diff) / TAU;
				}
				CHECK_FLT_RANGE(obj_diff);
				CHECK_FLT_INF(obj_diff);
				dh_obj_diff_array[j] = obj_diff;
				result_indx[j] = j;
			}

		}
	}
	else
	{
		if (!(d_alpha_status[j] == UPPER_BOUND) /*is_upper_bound(j)*/)
		{
			double grad_diff = Gmaxn - d_G[j];
			if (grad_diff > 1e-4) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[in] + d_QD[j] - 2.0 * cuda_evalQ(in, j); 
				CValue_t obj_diff = CVALUE_MAX;

				if (quad_coef > 0) {
					obj_diff = -(grad_diff*grad_diff) / quad_coef;
				}
				else {
					obj_diff = -(grad_diff*grad_diff) / TAU;
				}
				CHECK_FLT_RANGE(obj_diff);
				CHECK_FLT_INF(obj_diff);
				dh_obj_diff_array[j] = obj_diff;
				result_indx[j] = j;
			}
		}
	}

}

__global__ void cuda_find_nu_min_idx(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N)
{
	D_MinIdxFunctor func(obj_diff_array, obj_diff_indx, result_obj_min, result_indx); // Class defined in CudaReducer.h
	device_block_reducer(func, N); // Template function defined in CudaReducer.h
	if (blockIdx.x == 0) {
		int j = func.return_idx();
		d_solver.y = j; /* Gmin_idx */
		if (d_y[j] == +1)
			d_solver.x = d_nu_solver.x; /* Gmaxp_idx */
		else
			d_solver.x = d_nu_solver.y; /* Gmaxn_idx */	
	}
}

