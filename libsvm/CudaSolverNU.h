/**
Cuda implementation of NU SMO solver
@author: Ed Walker
*/
#ifndef _CUDA_NU_SOLVER_H_
#define _CUDA_NU_SOLVER_H_

#include "CudaSolver.h"

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
	CudaSolverNU(const svm_problem &prob, const svm_parameter &param) :
		CudaSolver(prob, param) {}

	virtual int select_working_set(int &out_i, int &out_j, int l); // overrides the version in CudaSolver
};

#endif