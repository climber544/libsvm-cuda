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

** Description: Cuda device objects and functions for misc parallel block reduction routines
** @author: Ed Walker
*/
#ifndef _CUDA_TEMPLATES_H_
#define _CUDA_TEMPLATES_H_
#include "svm_defs.h"

extern __shared__ char ss[];

class D_MinIdxReducer
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
	__device__ D_MinIdxReducer(CValue_t *obj_diff_array, int *obj_diff_indx, CValue_t *result_obj_min, int *result_indx)
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

	__device__ void load_shared_memory(const int &tid, const int &g_idx, const int &p_idx, const int &N)
	{
		s_obj_diff[tid] = obj_diff_array[g_idx];
		s_indx[tid] = obj_diff_indx[g_idx];
		if (p_idx < N) {
			if (obj_diff_array[p_idx] <= obj_diff_array[g_idx]) {
				s_obj_diff[tid] = obj_diff_array[p_idx];
				s_indx[tid] = obj_diff_indx[p_idx];
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

class D_GmaxReducer
{
private:
	GradValue_t *dh_gmax, *result_gmax; /* Gmax */
	GradValue_t *dh_gmax2, *result_gmax2; /* Gmax2 */
	int *dh_gmax_idx, *result_gmax_idx; /* Gmax_idx */
	bool debug;

	/** shared memory arrays */
	volatile GradValue_t *s_gmax;
	volatile GradValue_t *s_gmax2;
	volatile int *s_gmax_idx;

public:
	__device__ D_GmaxReducer(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, GradValue_t *result_gmax, GradValue_t *result_gmax2, int *result_gmax_idx, bool debug=false)	
		: dh_gmax(dh_gmax), dh_gmax2(dh_gmax2), dh_gmax_idx(dh_gmax_idx), result_gmax(result_gmax), result_gmax2(result_gmax2), result_gmax_idx(result_gmax_idx), debug(debug)
	{
		s_gmax = (GradValue_t *)&ss[0];
		s_gmax2 = (GradValue_t *)&ss[sizeof(GradValue_t)*blockDim.x];
		s_gmax_idx = (int *)&ss[2*sizeof(GradValue_t)*blockDim.x];
	}

	__device__ void block_out_of_range(const int &bid)
	{
		result_gmax_idx[bid] = -1;
		result_gmax[bid] = result_gmax2[bid] = -GRADVALUE_MAX;
	}

	__device__ void load_shared_memory(const int &tid, const int &g_idx, const int &p_idx, const int &N)
	{
		s_gmax[tid] = dh_gmax[g_idx];
		s_gmax_idx[tid] = dh_gmax_idx[g_idx];
		s_gmax2[tid] = dh_gmax2[g_idx];

		if (p_idx < N) {
			if (s_gmax[tid] < dh_gmax[p_idx] ||
				(s_gmax[tid] == dh_gmax[p_idx] && s_gmax_idx[tid] < dh_gmax_idx[p_idx])) {
					s_gmax[tid] = dh_gmax[p_idx];
					s_gmax_idx[tid] = dh_gmax_idx[p_idx];
			}

			if (s_gmax2[tid] < dh_gmax2[p_idx])
				s_gmax2[tid] = dh_gmax2[p_idx];
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
		//if (debug) printf("DEBUG: D_GMaxReducer::store_result: bid=%d gmax2 %g %g gmax_idx %d\n", bid, s_gmax2[0], result_gmax2[bid], s_gmax_idx[0]); // DEBUG
	}

	__device__ int return_idx() {
		return s_gmax_idx[0];
	}
};

class D_NuGmaxReducer
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
	__device__ D_NuGmaxReducer(GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2, 
		int *dh_gmaxp_idx, int *dh_gmaxn_idx,
		GradValue_t *result_gmaxp, GradValue_t *result_gmaxn, GradValue_t *result_gmaxp2, GradValue_t *result_gmaxn2,
		int *result_gmaxp_idx, int *result_gmaxn_idx)
		: dh_gmaxp(dh_gmaxp), result_gmaxp(result_gmaxp), /* Gmaxp */
		dh_gmaxn(dh_gmaxn), result_gmaxn(result_gmaxn), /* Gmaxn */
		dh_gmaxp2(dh_gmaxp2), result_gmaxp2(result_gmaxp2), /* Gmaxp2 */
		dh_gmaxn2(dh_gmaxn2), result_gmaxn2(result_gmaxn2), /* Gmaxn2 */
		dh_gmaxp_idx(dh_gmaxp_idx), result_gmaxp_idx(result_gmaxp_idx), /* Gmaxp_idx */
		dh_gmaxn_idx(dh_gmaxn_idx), result_gmaxn_idx(result_gmaxn_idx) /* Gmaxn_idx */
	{
		s_gmaxp = (GradValue_t *)&ss[0];
		s_gmaxn = (GradValue_t *)&ss[sizeof(GradValue_t)*blockDim.x];
		s_gmaxp2 = (GradValue_t *)&ss[2 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxn2 = (GradValue_t *)&ss[3 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxp_idx = (int *)&ss[4 * sizeof(GradValue_t)*blockDim.x];
		s_gmaxn_idx = (int *)&ss[4 * sizeof(GradValue_t)*blockDim.x + sizeof(int)*blockDim.x];
	}

	__device__ void block_out_of_range(const int &bid)
	{
		result_gmaxp[0] = result_gmaxn[0] = result_gmaxp2[0] = result_gmaxn2[0] = -GRADVALUE_MAX;
		result_gmaxp_idx[0] = result_gmaxn_idx[0] = -1;
	}

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

__device__ GradValue_t device_computer_gradient(int i, int j);

class D_GradientAdder
{
private:
	int j;
	int N;

	/** shared memory arrays */
	volatile GradValue_t *s_acc;

public:
	__device__ D_GradientAdder(int j, int N)
		: j(j), N(N)
	{
		s_acc = (GradValue_t *)&ss[0];
	}

	__device__ void block_out_of_range(const int &bid)
	{
		s_acc[0] = 0;
	}

	__device__ void load_shared_memory(const int &tid, const int &g_idx, const int &p_idx, const int &N)
	{
		s_acc[tid] = device_computer_gradient(g_idx, j);

		if (p_idx < N) {
			s_acc[tid] += device_computer_gradient(p_idx, j);
		}
	}

	__device__ void reduce(const int &tid1, const int &tid2) {
		s_acc[tid1] += s_acc[tid2];
	}

	__device__ void store_result(const int &bid) {
	}

	__device__ GradValue_t return_sum() {
		return s_acc[0];
	}
};

template <class T>
__device__ void device_block_reducer(T &f, int N) 
{
	int g_idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;

	if (g_idx >= N) {
		if (threadIdx.x == 0) {
			f.block_out_of_range(blockIdx.x);
		}
		return ;
	}

	int p_idx;
	p_idx = g_idx + blockDim.x; // calculate pair-wise index into the next block

	f.load_shared_memory(threadIdx.x, g_idx, p_idx, N);

	__syncthreads();

	int upperPoint = blockDim.x;
	int halfPoint = (blockDim.x >> 1);
	if (blockDim.x & 1) ++halfPoint;

	while (halfPoint > 0)
	{
		if (threadIdx.x < halfPoint) {
			int thread2 = threadIdx.x + halfPoint;

			p_idx = (blockDim.x * 2) * blockIdx.x + thread2;

			if (p_idx < N && thread2 < upperPoint) {// check if pair-wise index is within valid range 
				f.reduce(threadIdx.x, thread2);
			}
		}

		if (halfPoint > warpSize) __syncthreads();

		upperPoint = halfPoint;
		if (halfPoint > 1 && (halfPoint & 1)) ++halfPoint;
		halfPoint >>= 1;
	}

	if (threadIdx.x == 0) {
		f.store_result(blockIdx.x);
	}
}
#endif