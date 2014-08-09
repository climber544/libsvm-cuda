#include "svm.h"
#include <stdexcept>
#include <iostream>
using namespace std;
#include <stdio.h>
#include "math.h"
#include "svm_device.h"
#include "cuda_reducer.h"

#define DEVICE_EPS	0

enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

#ifdef USE_CONSTANT_SVM_NODE
__constant__ cuda_svm_node      *d_space;
#else
texture<float2, 1, cudaReadModeElementType> d_tex_space;
#endif
__constant__	int				*d_x;
__device__		int				d_kernel_type;	// enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
__device__		int				d_svm_type;		// enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
__constant__	double			d_gamma;		// rbf, poly, and sigmoid kernel
__constant__	double			d_coef0;		// poly and sigmoid kernel
__constant__	int				d_degree;		// poly kernel
__constant__	int				d_l;			// original # SV

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

cudaError_t update_param_constants(const svm_parameter &param, int *dh_x, cuda_svm_node *dh_space, size_t dh_space_size, int l)
{
	cudaError_t err;
	err = cudaMemcpyToSymbol(d_l, &l, sizeof(l));
	if (err != cudaSuccess) {
		fprintf(stderr, "Error with copying to symbol d_l\n");
		return err;
	}
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
	if (dh_x_square != NULL) {
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
	remember: 
	cuda_svm_node.x == svm_node.index
	cuda_svm_node.y == svm_node.value
	*/
#define index x
#define value y

	cuda_svm_node x = get_col_value(i_col);
	cuda_svm_node y = get_col_value(j_col);

	double sum = 0;
	while (x.index != -1 && y.index != -1)
	{
		if (x.index == y.index)
		{
			sum += x.value * y.value;
			x = get_col_value(++i_col);
			y = get_col_value(++j_col);
		}
		else
		{

			if (x.index > y.index) {
				y = get_col_value(++j_col);
			}
			else {
				x = get_col_value(++i_col);
			}
		}
	}
	return sum;
}

__device__ CValue_t device_kernel_rbf(const int &i, const int &j)
{
	CValue_t q = d_x_square[i] + d_x_square[j] - 2 * dot(i, j);
	return exp(-(CValue_t)d_gamma * q);
}

__device__ CValue_t device_kernel_poly(const int &i, const int &j)
{
	return pow((CValue_t)d_gamma * dot(i, j) + (CValue_t)d_coef0, d_degree);
}

__device__ CValue_t device_kernel_sigmoid(const int &i, const int &j)
{
	return tanh((CValue_t)d_gamma * dot(i, j) + (CValue_t)d_coef0);
}

__device__ CValue_t device_kernel_linear(const int &i, const int &j)
{
	return dot(i, j);
}

__device__ CValue_t device_kernel_precomputed(const int &i, const int &j)
{
	int i_col = d_x[i];
	int j_col = d_x[j];
	int offset = static_cast<int>(get_col_value(j_col).y);
	return get_col_value(i_col + offset).y;
	// return x[i][(int)(x[j][0].value)].value;
}

/**
Returns the product of the kernel function multiplied with rc
@param i	index i
@param j	index j
@param rc	multiplier for the kernel function
*/
__device__ __forceinline__ CValue_t kernel(const int &i, const int &j, const CValue_t &rc)
{
	switch (d_kernel_type)
	{
	case RBF:
		return rc * device_kernel_rbf(i, j);
	case POLY:
		return rc * device_kernel_poly(i, j);
	case LINEAR:
		return rc * device_kernel_linear(i, j);
	case SIGMOID:
		return rc * device_kernel_sigmoid(i, j);
	case PRECOMPUTED:
		return rc * device_kernel_precomputed(i, j);
	}

	return 0;
}

/**
	Implements schar *SVR_Q::sign
	[0..l-1] --> 1
	[l..2*l) --> -1
*/
__device__ __forceinline__ SChar_t device_SVR_sign(int i)
{
	return (i < d_l ? 1 : -1);
}

/**
	Implements int *SVR_Q::index
	[0..l-1] --> [0..l-1]
	[l..2*l) --> [0..1-1]
*/
__device__ __forceinline__ int device_SVR_real_index(int i)
{
	return (i < d_l ? i : (i - d_l));
}

__device__ CValue_t cuda_evalQ(int i, int j)
{
	CValue_t rc = 1;

	switch (d_svm_type)
	{
	case C_SVC:
	case NU_SVC:
		// SVC_Q
		rc = (CValue_t)(d_y[i] * d_y[j]);
		break;
	case ONE_CLASS:
		// ONE_CLASS_Q - nothing to do
		break;
	case EPSILON_SVR:
	case NU_SVR:
		// SVR_Q
		rc = (CValue_t)(device_SVR_sign(i) * device_SVR_sign(j));
		i = device_SVR_real_index(i); // use the kernel calculation
		j = device_SVR_real_index(j); // use for kernel calculation
		break;
	}

	return kernel(i, j, rc);
}

__global__ void cuda_find_min_idx(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx, int N)
{
	D_MinIdxReducer func(obj_diff_array, obj_diff_indx, result_obj_min, result_indx); // Class defined in CudaReducer.h
	device_block_reducer(func, N); // Template function defined in CudaReducer.h
	if (blockIdx.x == 0)
		d_solver.y = func.return_idx();
}


__global__ void cuda_compute_obj_diff(GradValue_t Gmax, CValue_t *dh_obj_diff_array, int *result_indx, int N)
{
	int i = d_solver.x;

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= N)
		return;

	dh_obj_diff_array[j] = CVALUE_MAX;
	result_indx[j] = -1;
	if (d_y[j] == 1)
	{
		if (!(d_alpha_status[j] == LOWER_BOUND)/*is_lower_bound(j)*/)
		{
			GradValue_t grad_diff = Gmax + d_G[j];
			if (grad_diff > DEVICE_EPS) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[i] + d_QD[j] - 2.0 * d_y[i] * cuda_evalQ(i, j);
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
			GradValue_t grad_diff = Gmax - d_G[j];
			if (grad_diff > DEVICE_EPS) // original: grad_diff > 0
			{
				CValue_t quad_coef = d_QD[i] + d_QD[j] + 2.0 * d_y[i] * cuda_evalQ(i, j);
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

__global__ void cuda_init_gradient(int start, int step, int N)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= N)
		return;

	GradValue_t acc = 0;
	for (int i = start; i < N && i < start + step; ++i)
	{
		if (!(d_alpha_status[i] == LOWER_BOUND) /*is_lower_bound(i)*/)
		{
			acc += d_alpha[i] * cuda_evalQ(i, j);
		}
	}

	d_G[j] += acc;
}

/**
double version of atomicAdd
*/
__device__ double atomicAdd(double * address, double val)
{
	unsigned long long int *address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ GradValue_t device_computer_gradient(int i, int j)
{
	if (!(d_alpha_status[i] == LOWER_BOUND) /*is_lower_bound(i)*/)
	{
		return d_alpha[i] * cuda_evalQ(i, j);
	}
	else
		return 0;
}

__global__ void cuda_init_gradient_block(int startj, int N)
{
	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y + startj;

	if (j >= N || i >= N)
		return;

	D_GradientAdder func(j, N);
	device_block_reducer(func, N);

	if (threadIdx.x == 0) { // every block in the x-axis (ie. i)
		GradValue_t s = func.return_sum();
		atomicAdd(&d_G[j], s);
	}

	return;
}

/**
	Initializes the gradient vector on the device
	@param block_size	number of threads per block
	@param startj		starting index j for G_j
	@param stepj		number of steps from startj to update
	@param N			size of gradient vector
*/
void init_device_gradient(int block_size, int startj, int stepj, int N)
{
	int reduce_block_size = 2 * block_size;
	dim3 grid;
	grid.x = N / reduce_block_size;
	if (N%reduce_block_size != 0) ++grid.x; // the number of blocks in the ith dimension
	grid.y = stepj; // the number of blocks in the jth dimension == G_j that will be updated

	dim3 block;
	block.x = block_size; // number of threads in the ith dimension
	block.y = 1; // number of threads per block in the jth dimension (one thread per block)
	
	size_t shared_mem = block.x * sizeof(GradValue_t);
	cuda_init_gradient_block << <grid, block, shared_mem >> > (startj, N);
	check_cuda_kernel_launch("fail in cuda_init_gradient_block");
}

__global__ void cuda_find_gmax(find_gmax_param param, int N, bool debug)
{
	D_GmaxReducer func(param.dh_gmax, param.dh_gmax2, param.dh_gmax_idx, param.result_gmax, 
		param.result_gmax2, param.result_gmax_idx, debug); // class defined in CudaReducer.h

	device_block_reducer(func, N); // Template function defined in CudaReducer.h

	if (blockIdx.x == 0)
		d_solver.x = func.return_idx();
}

__global__ void cuda_setup_x_square(int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;
	d_x_square[i] = dot(i, i);
}

__global__ void cuda_setup_QD(int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;

	d_QD[i] = kernel(i, i, 1);

	if (d_svm_type == NU_SVR || d_svm_type == EPSILON_SVR)
		d_QD[i + d_l] = d_QD[i];
}

__global__ void cuda_prep_gmax(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, int N)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	if (t >= N)
		return;

	dh_gmax[t] = -GRADVALUE_MAX;
	dh_gmax2[t] = -GRADVALUE_MAX;
	dh_gmax_idx[t] = -1;
	if (d_y[t] == +1)
	{
		if (!(d_alpha_status[t] == UPPER_BOUND) /*is_upper_bound(t)*/) {
			dh_gmax[t] = -d_G[t];
			dh_gmax_idx[t] = t;
		}
		if (!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
			dh_gmax2[t] = d_G[t];
		}
	}
	else
	{
		if (!(d_alpha_status[t] == LOWER_BOUND) /*is_lower_bound(t)*/) {
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
	return (d_y[i] > 0) ? d_Cp : d_Cn;
}

__global__ void cuda_compute_alpha()
{
	int i = d_solver.x; // d_selected_i;
	int j = d_solver.y; // d_selected_j;

	GradValue_t C_i = device_get_C(i);
	GradValue_t C_j = device_get_C(j);

	GradValue_t old_alpha_i = d_alpha[i];
	GradValue_t old_alpha_j = d_alpha[j];

	if (d_y[i] != d_y[j])
	{
		GradValue_t quad_coef = d_QD[i] + d_QD[j] + 2 * cuda_evalQ(i, j); //  Q_i[j];
		if (quad_coef <= 0)
			quad_coef = TAU;
		GradValue_t delta = (-d_G[i] - d_G[j]) / quad_coef;
		GradValue_t diff = d_alpha[i] - d_alpha[j];
		d_alpha[i] += delta;
		d_alpha[j] += delta;

		if (diff > 0)
		{
			if (d_alpha[j] < 0)
			{
				d_alpha[j] = 0;
				d_alpha[i] = diff;
			}
		}
		else
		{
			if (d_alpha[i] < 0)
			{
				d_alpha[i] = 0;
				d_alpha[j] = -diff;
			}
		}
		if (diff > C_i - C_j)
		{
			if (d_alpha[i] > C_i)
			{
				d_alpha[i] = C_i;
				d_alpha[j] = C_i - diff;
			}
		}
		else
		{
			if (d_alpha[j] > C_j)
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
		GradValue_t delta = (d_G[i] - d_G[j]) / quad_coef;
		GradValue_t sum = d_alpha[i] + d_alpha[j];
		d_alpha[i] -= delta;
		d_alpha[j] += delta;

		if (sum > C_i)
		{
			if (d_alpha[i] > C_i)
			{
				d_alpha[i] = C_i;
				d_alpha[j] = sum - C_i;
			}
		}
		else
		{
			if (d_alpha[j] < 0)
			{
				d_alpha[j] = 0;
				d_alpha[i] = sum;
			}
		}
		if (sum > C_j)
		{
			if (d_alpha[j] > C_j)
			{
				d_alpha[j] = C_j;
				d_alpha[i] = sum - C_j;
			}
		}
		else
		{
			if (d_alpha[i] < 0)
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
	if (d_alpha[i] >= device_get_C(i))
		d_alpha_status[i] = UPPER_BOUND;
	else if (d_alpha[i] <= 0)
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

__global__ void cuda_compute_nu_obj_diff(GradValue_t Gmaxp, GradValue_t Gmaxn, CValue_t *dh_obj_diff_array, int *result_idx, int N)
{
	int ip = d_nu_solver.x;
	int in = d_nu_solver.y;

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j >= N)
		return;

	dh_obj_diff_array[j] = CVALUE_MAX;
	result_idx[j] = -1;
	if (d_y[j] == 1)
	{
		if (!(d_alpha_status[j] == LOWER_BOUND)/*is_lower_bound(j)*/)
		{
			GradValue_t grad_diff = Gmaxp + d_G[j];
			if (grad_diff > DEVICE_EPS) // original: grad_diff > 0
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
				result_idx[j] = j;
			}

		}
	}
	else
	{
		if (!(d_alpha_status[j] == UPPER_BOUND) /*is_upper_bound(j)*/)
		{
			GradValue_t grad_diff = Gmaxn - d_G[j];
			if (grad_diff > DEVICE_EPS) // original: grad_diff > 0
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
				result_idx[j] = j;
			}
		}
	}

}



__global__ void cuda_find_nu_gmax(find_nu_gmax_param param, int N)
{
	D_NuGmaxReducer func(param.dh_gmaxp, param.dh_gmaxn, param.dh_gmaxp2, param.dh_gmaxn2, param.dh_gmaxp_idx, param.dh_gmaxn_idx,
		param.result_gmaxp, param.result_gmaxn, param.result_gmaxp2, param.result_gmaxn2, param.result_gmaxp_idx, param.result_gmaxn_idx);

	device_block_reducer(func, N);

	if (blockIdx.x == 0) {
		int ip, in;
		func.return_idx(ip, in);
		d_nu_solver.x = ip;
		d_nu_solver.y = in;
	}
}



__global__ void cuda_find_nu_min_idx(CValue_t *obj_diff_array, int *obj_diff_idx, CValue_t *result_obj_min, int *result_idx, int N)
{
	D_MinIdxReducer func(obj_diff_array, obj_diff_idx, result_obj_min, result_idx); // Class defined in CudaReducer.h
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

/************DEVICE KERNEL LAUNCHERS***************/
void launch_cuda_setup_x_square(size_t num_blocks, size_t block_size, int N)
{
	cuda_setup_x_square << <num_blocks, block_size >> >(N);
}

void launch_cuda_setup_QD(size_t num_blocks, size_t block_size, int N)
{
	cuda_setup_QD << <num_blocks, block_size >> >(N);
}


void launch_cuda_compute_obj_diff(size_t num_blocks, size_t block_size, GradValue_t Gmax, CValue_t *dh_obj_diff_array, int *result_idx, int N)
{
	cuda_compute_obj_diff << <num_blocks, block_size >> > (Gmax, dh_obj_diff_array, result_idx, N);
}

void launch_cuda_update_gradient(size_t num_blocks, size_t block_size, int N)
{
	cuda_update_gradient << <num_blocks, block_size >> > (N);
}

void launch_cuda_init_gradient(size_t num_blocks, size_t block_size, int start, int step, int N)
{
	cuda_init_gradient << < num_blocks, block_size>> > (start, step, N);
}

void launch_cuda_prep_gmax(size_t num_blocks, size_t block_size, GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, int N)
{
	cuda_prep_gmax << < num_blocks, block_size>> > (dh_gmax, dh_gmax2, dh_gmax_idx, N);
}

void launch_cuda_compute_alpha(size_t num_blocks, size_t block_size)
{
	cuda_compute_alpha << <num_blocks, block_size >> >();
}

void launch_cuda_update_alpha_status(size_t num_blocks, size_t block_size)
{
	cuda_update_alpha_status << <num_blocks, block_size >> >();
}

void launch_cuda_find_min_idx(size_t num_blocks, size_t block_size, size_t share_mem_size, CValue_t *obj_diff_array, int *obj_diff_idx, CValue_t *result_obj_min, int *result_idx, int N)
{
	cuda_find_min_idx << <num_blocks, block_size, share_mem_size >> >(obj_diff_array, obj_diff_idx, result_obj_min, result_idx, N);
}

void launch_cuda_find_gmax(size_t num_blocks, size_t block_size, size_t share_mem_size, find_gmax_param param, int N, bool debug)
{
	cuda_find_gmax << <num_blocks, block_size, share_mem_size >> >(param, N, debug);
}

void launch_cuda_find_nu_min_idx(size_t num_blocks, size_t block_size, size_t share_mem_size, CValue_t *obj_diff_array, int *obj_diff_idx, CValue_t *result_obj_min, int *result_idx, int N)
{
	cuda_find_nu_min_idx << <num_blocks, block_size, share_mem_size >> >(obj_diff_array, obj_diff_idx, result_obj_min, result_idx, N);
}

void launch_cuda_find_nu_gmax(size_t num_blocks, size_t block_size, size_t share_mem_size, find_nu_gmax_param param, int N)
{
	cuda_find_nu_gmax << <num_blocks, block_size, share_mem_size >> >(param, N);
}

void launch_cuda_compute_nu_obj_diff(size_t num_blocks, size_t block_size, GradValue_t Gmaxp, GradValue_t Gmaxn, CValue_t *dh_obj_diff_array, int *result_idx, int N)
{
	cuda_compute_nu_obj_diff << <num_blocks, block_size >> > (Gmaxp, Gmaxn, dh_obj_diff_array, result_idx, N);
}

void launch_cuda_prep_nu_gmax(size_t num_blocks, size_t block_size, GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2,
	int *dh_gmaxp_idx, int *dh_gmaxn_idx, int N)
{
	cuda_prep_nu_gmax << <num_blocks, block_size >> > (dh_gmaxp, dh_gmaxn, dh_gmaxp2, dh_gmaxn2, dh_gmaxp_idx, dh_gmaxn_idx, N);
}



/**************** DEBUGGING ********************/
/**
useful for peeking at various misc values when debugging
*/
__global__ void cuda_peek(int i, int j)
{
	printf("Q(%d,%d)=%g\n", i, j, cuda_evalQ(i, j));
}


