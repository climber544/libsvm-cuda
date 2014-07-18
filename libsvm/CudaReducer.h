#ifndef _CUDA_TEMPLATES_H_
#define _CUDA_TEMPLATES_H_
#include "CudaSolver.h"

extern __shared__ char ss[];

class D_MinIdxFunctor
{
private:
	CValue_t *obj_diff_array;
	int *obj_diff_indx;
	CValue_t *result_obj_min;
	int *result_indx;

	/** shared memory arrays */
	volatile CValue_t *s_obj_diff;
	volatile int *s_indx;

public:
	__device__ D_MinIdxFunctor(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx)
		: obj_diff_array(obj_diff_array), obj_diff_indx(obj_diff_indx), result_obj_min(result_obj_min), result_indx(result_indx)
	{
		s_obj_diff = (CValue_t *)&ss[0];
		s_indx = (int *)&ss[sizeof(CValue_t)*blockDim.x];
	}

	__device__ void block_out_of_range(const int &bid)
	{
		result_obj_min[bid] = CVALUE_MAX;
		result_indx[bid] = -1;
	}

	__device__ void load_shared_memory(const int &tid, const int &g_indx, const int &p_indx, const int &N)
	{
		s_obj_diff[tid] = obj_diff_array[g_indx];
		s_indx[tid] = obj_diff_indx[g_indx];
		if (p_indx < N) {
			if (obj_diff_array[p_indx] <= obj_diff_array[g_indx]) {
				s_obj_diff[tid] = obj_diff_array[p_indx];
				s_indx[tid] = obj_diff_indx[p_indx];
			}
		}
	}

	__device__ void reduce(const int &tid1, const int &tid2)
	{
		if (s_obj_diff[tid2] <  s_obj_diff[tid1] || 
			(s_obj_diff[tid2] == s_obj_diff[tid1] && s_indx[tid2] > s_indx[tid1])) {
				s_obj_diff[tid1] = s_obj_diff[tid2];
				s_indx[tid1] = s_indx[tid2];
		}
	}

	__device__ void store_result(const int &bid) {
		result_indx[bid] = s_indx[0];
		result_obj_min[bid] = s_obj_diff[0];
	}

	__device__ int return_idx() {
		return s_indx[0];
	}
};

class D_GmaxFunctor
{
private:
	GradValue_t *dh_gmax, *result_gmax; /* Gmax */
	GradValue_t *dh_gmax2, *result_gmax2; /* Gmax2 */
	int *dh_gmax_idx, *result_gmax_idx; /* Gmax_idx */

	/** shared memory arrays */
	volatile GradValue_t *s_gmax;
	volatile GradValue_t *s_gmax2;
	volatile int *s_gmax_idx;

public:
	__device__ D_GmaxFunctor(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, GradValue_t *result_gmax, GradValue_t *result_gmax2, int *result_gmax_idx)	
		: dh_gmax(dh_gmax), dh_gmax2(dh_gmax2), dh_gmax_idx(dh_gmax_idx), result_gmax(result_gmax), result_gmax2(result_gmax2), result_gmax_idx(result_gmax_idx)
	{
		s_gmax = (GradValue_t *)&ss[0];
		s_gmax2 = (GradValue_t *)&ss[sizeof(GradValue_t)*blockDim.x];
		s_gmax_idx = (int *)&ss[2*sizeof(GradValue_t)*blockDim.x];
	}

	__device__ void block_out_of_range(const int &bid)
	{}

	__device__ void load_shared_memory(const int &tid, const int &g_indx, const int &p_indx, const int &N)
	{
		s_gmax[tid] = dh_gmax[g_indx];
		s_gmax_idx[tid] = dh_gmax_idx[g_indx];
		s_gmax2[tid] = dh_gmax2[g_indx];
		if (p_indx < N) {
			if (s_gmax[tid] < dh_gmax[p_indx] ||
				(s_gmax[tid] == dh_gmax[p_indx] && s_gmax_idx[tid] < dh_gmax_idx[p_indx])) {
					s_gmax[tid] = dh_gmax[p_indx];
					s_gmax_idx[tid] = dh_gmax_idx[p_indx];
			}
			if (s_gmax2[tid] < dh_gmax2[p_indx])
				s_gmax2[tid] = dh_gmax2[p_indx];
		}
	}

	__device__ void reduce(const int &tid1, const int &tid2)
	{
		if (s_gmax[tid2] >  s_gmax[tid1] || 
			(s_gmax[tid2] == s_gmax[tid1] && s_gmax_idx[tid2] > s_gmax_idx[tid1])) {
				s_gmax[tid1] = s_gmax[tid2];
				s_gmax_idx[tid1] = s_gmax_idx[tid2];
		}
		if (s_gmax2[tid2] >  s_gmax2[tid1]) {
			s_gmax2[tid1] = s_gmax2[tid2];
		}
	}

	__device__ void store_result(const int &bid) {
		result_gmax_idx[bid] = s_gmax_idx[0];
		result_gmax[bid] = s_gmax[0];
		result_gmax2[bid] = s_gmax2[0];
	}

	__device__ int return_idx() {
		return s_gmax_idx[0];
	}
};

class D_NuGmaxFunctor
{
private:
	GradValue_t *dh_gmaxp, *result_gmaxp; /* Gmaxp */
	GradValue_t *dh_gmaxn, *result_gmaxn; /* Gmaxn */
	GradValue_t *dh_gmaxp2, *result_gmaxp2; /* Gmaxp2 */
	GradValue_t *dh_gmaxn2, *result_gmaxn2; /* Gmaxn2 */
	int *dh_gmaxp_idx, *result_gmaxp_idx; /* Gmaxp_idx */
	int *dh_gmaxn_idx, *result_gmaxn_idx; /* Gmaxn_idx */

	/** shared memory arrays */
	volatile GradValue_t *s_gmaxp;
	volatile GradValue_t *s_gmaxn;
	volatile GradValue_t *s_gmaxp2;
	volatile GradValue_t *s_gmaxn2;
	volatile int *s_gmaxp_idx;
	volatile int *s_gmaxn_idx;

public:
	__device__ D_NuGmaxFunctor(GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2, 
		int *dh_gmaxp_idx, int *dh_gmaxn_idx,
		GradValue_t *result_gmaxp, GradValue_t *result_gmaxn, GradValue_t *result_gmaxp2, GradValue_t *result_gmaxn2,
		int *result_gmaxp_idx, int *result_gmaxn_idx)
		: dh_gmaxp(dh_gmaxp), dh_gmaxn(dh_gmaxn), dh_gmaxp_idx(dh_gmaxp_idx), dh_gmaxn_idx(dh_gmaxn_idx),
		result_gmaxp(result_gmaxp), result_gmaxn(result_gmaxn), 
		result_gmaxp_idx(result_gmaxp_idx), result_gmaxn_idx(result_gmaxn_idx)
	{
		s_gmaxp = (GradValue_t *)&ss[0];
		s_gmaxn = (GradValue_t *)&ss[sizeof(GradValue_t)*blockDim.x];
		s_gmaxp2 = (GradValue_t *)&ss[2 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxn2 = (GradValue_t *)&ss[3 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxp_idx = (int *)&ss[4 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxn_idx = (int *)&ss[4 * sizeof(GradValue_t)*blockDim.x + sizeof(int)*blockDim.x];
	}

	__device__ void block_out_of_range(const int &bid)
	{}

	__device__ void load_shared_memory(const int &tid, const int &g_idx, const int &p_idx, const int &N)
	{
		s_gmaxp[tid] = dh_gmaxp[g_idx];
		s_gmaxp_idx[tid] = dh_gmaxp_idx[g_idx];
		s_gmaxn[tid] = dh_gmaxn[g_idx];
		s_gmaxn_idx[tid] = dh_gmaxn_idx[g_idx];
		s_gmaxp2[tid] = dh_gmaxp2[g_idx];
		s_gmaxn2[tid] = dh_gmaxn2[g_idx];

		if (p_idx < N) {
			if (s_gmaxp[tid] < dh_gmaxp[p_idx] ||
				(s_gmaxp[tid] == dh_gmaxp[p_idx] && s_gmaxp_idx[tid] < dh_gmaxp_idx[p_idx])) {
				s_gmaxp[tid] = dh_gmaxp[p_idx];
				s_gmaxp_idx[tid] = dh_gmaxp_idx[p_idx];
			}
			if (s_gmaxn[tid] < dh_gmaxn[p_idx] ||
				(s_gmaxn[tid] == dh_gmaxn[p_idx] && s_gmaxn_idx[tid] < dh_gmaxn_idx[p_idx])) {
				s_gmaxn[tid] = dh_gmaxn[p_idx];
				s_gmaxn_idx[tid] = dh_gmaxn_idx[p_idx];
			}
			if (s_gmaxp2[tid] < dh_gmaxp2[p_idx])
				s_gmaxp2[tid] = dh_gmaxp2[p_idx];
			if (s_gmaxn2[tid] < dh_gmaxn2[p_idx])
				s_gmaxn2[tid] = dh_gmaxn2[p_idx];
		}
	}

	__device__ void reduce(const int &tid1, const int &tid2)
	{
		if (s_gmaxp[tid2] >  s_gmaxp[tid1] ||
			(s_gmaxp[tid2] == s_gmaxp[tid1] && s_gmaxp_idx[tid2] > s_gmaxp_idx[tid1])) {
			s_gmaxp[tid1] = s_gmaxp[tid2];
			s_gmaxp_idx[tid1] = s_gmaxp_idx[tid2];
		}
		if (s_gmaxn[tid2] >  s_gmaxn[tid1] ||
			(s_gmaxn[tid2] == s_gmaxn[tid1] && s_gmaxn_idx[tid2] > s_gmaxn_idx[tid1])) {
			s_gmaxn[tid1] = s_gmaxn[tid2];
			s_gmaxn_idx[tid1] = s_gmaxn_idx[tid2];
		}
		if (s_gmaxp2[tid2] > s_gmaxp2[tid1])
			s_gmaxp2[tid1] = s_gmaxp2[tid2];
		if (s_gmaxn2[tid2] > s_gmaxn2[tid1])
			s_gmaxn2[tid1] = s_gmaxn2[tid2];
	}

	__device__ void store_result(const int &bid) {
		result_gmaxp_idx[bid] = s_gmaxp_idx[0];
		result_gmaxn_idx[bid] = s_gmaxn_idx[0];
		result_gmaxp[bid] = s_gmaxp[0];
		result_gmaxn[bid] = s_gmaxn[0];
		result_gmaxp2[bid] = s_gmaxp2[0];
		result_gmaxn2[bid] = s_gmaxn2[0];
	}

	__device__ void return_idx(int &gmaxp_idx, int &gmaxn_idx) {
		gmaxp_idx = s_gmaxp_idx[0];
		gmaxn_idx = s_gmaxn_idx[0];
	}
};

template <class T>
__device__ void device_block_reducer(T &f, int N) 
{
	int g_indx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;

	if (g_indx >= N) {
		if (threadIdx.x == 0) {
			f.block_out_of_range(blockIdx.x);
		}
		return ;
	}

	int p_indx;
	p_indx = g_indx + blockDim.x; // calculate pair-wise index into the next block

	f.load_shared_memory(threadIdx.x, g_indx, p_indx, N);

	__syncthreads();

	for (int halfPoint = (blockDim.x >> 1); halfPoint > 32; halfPoint >>= 1)
	{
		if (threadIdx.x < halfPoint) {
			int thread2 = threadIdx.x + halfPoint;

			p_indx = (blockDim.x * 2) * blockIdx.x + thread2;

			if (p_indx < N) {// check if pair-wise index is within valid range 
				f.reduce(threadIdx.x, thread2);
			}
		}

		__syncthreads();
	}

	if (threadIdx.x < 32) {
		p_indx = (blockDim.x * 2 ) * blockIdx.x + (threadIdx.x + 32);

		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+32);
		p_indx -= 16;
		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+16);
		p_indx -= 8;
		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+8);
		p_indx -= 4;
		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+4);
		p_indx -= 2;
		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+2);
		p_indx -= 1;
		if (p_indx < N) f.reduce(threadIdx.x, threadIdx.x+1);
	}

	if (threadIdx.x == 0) {
		f.store_result(blockIdx.x);
	}
}
#endif