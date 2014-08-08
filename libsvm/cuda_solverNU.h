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
** Description: Cuda implementation of NU Sequential Minimal Optimization (SMO) solver
** @author: Ed Walker
*/
#ifndef _CUDA_NU_SOLVER_H_
#define _CUDA_NU_SOLVER_H_

#include "cuda_solver.h"

class CudaSolverNU : public CudaSolver
{
private:
	/**
	CUDA device memory arrays
	*/
	CudaArray_t<GradValue_t> dh_gmaxp; 
	CudaArray_t<GradValue_t> dh_gmaxn; 
	CudaArray_t<GradValue_t> dh_gmaxp2;
	CudaArray_t<GradValue_t> dh_gmaxn2;
	CudaArray_t<int> dh_gmaxp_idx; 
	CudaArray_t<int> dh_gmaxn_idx;

	CudaArray_t<GradValue_t> dh_result_gmaxp; 
	CudaArray_t<GradValue_t> dh_result_gmaxn;
	CudaArray_t<GradValue_t> dh_result_gmaxp2;
	CudaArray_t<GradValue_t> dh_result_gmaxn2;
	CudaArray_t<int> dh_result_gmaxp_idx; 
	CudaArray_t<int> dh_result_gmaxn_idx;

	void init_gmax_space(int l);

	class NuMinIdxReducer; // class object used for cross_block_reducer() template function

	class NuGmaxReducer; // class object used for cross_block_reducer template function

	void select_working_set_j(GradValue_t Gmaxp, GradValue_t Gmaxn, int l); 

public:
	CudaSolverNU(const svm_problem &prob, const svm_parameter &param, bool quiet_mode=true) :
		CudaSolver(prob, param, quiet_mode) {}

	virtual int select_working_set(int &out_i, int &out_j, int l); // overrides the version in CudaSolver
};

#endif