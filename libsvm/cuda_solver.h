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
** Description: Cuda implementation of Sequential Minimal Optimization (SMO) solver
** @author: Ed Walker
*/
#ifndef _CUDA_SOLVER_H_
#define _CUDA_SOLVER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_constants.h"
#include "svm.h"
#include <iostream>
#include <memory>
#include <ctime>

#include "svm_defs.h"

class CudaSolver
{
protected:

	/**
	Cross cuda block reducer -- Because the device function performs a block reduction only,
	this function can be used to orchestrate the device functions to reduce across blocks.
	*/
	template <class T>
	int cross_block_reducer(int def_block_size, T &f, int N)
	{
		int reduce_block_size = std::min(N, def_block_size);  // default block size or N, whichever is smaller
		int elements_per_block = 2 * reduce_block_size; // because we are reducing with block stride 2!!
		int reduce_blocks = N / elements_per_block;
		if (N % elements_per_block != 0) ++reduce_blocks;

		while (reduce_blocks > 0) {
			f.compute(reduce_blocks, reduce_block_size, N);

			if (reduce_blocks == 1)
				break;

			N = reduce_blocks; // new number of elements to reduce

			reduce_block_size = std::min(N, def_block_size);  // default block size or N, whichever is smaller
			elements_per_block = 2 * reduce_block_size; // because we are reducing with block stride 2!!
			reduce_blocks = N / elements_per_block;
			if (N % elements_per_block != 0) ++reduce_blocks;

			f.swap();
		}

		return f.process_output();
	}

	class MinIdxReducer; // class object used for cross_block_reducer() template function

	class GmaxReducer; // class object used for cross_block_reducer() template function

	/**
	Smart pointers for CUDA arrays.  Their semantics are similar to C++11 std::unique_ptr.
	*/
	struct CudaDeleter
	{
		void operator()(void *p)
		{
			if (p != nullptr) {
				cudaFree(p);
			} 
		}
	};

	template <typename T>
	class CudaArray_t
	{
	public:
		std::unique_ptr<T[], CudaDeleter> dh_ptr;

	public:
		T &operator[](size_t idx) { // subscript operator
			return dh_ptr[idx];
		}

		CudaArray_t(T *ptr) {
			dh_ptr.reset(ptr);
		}

		CudaArray_t() {} // No-arg constructor

		// Move operator to maintian unique_ptr semantics
		CudaArray_t &operator=(CudaArray_t &&other) { // Note: parameter is not const
			if (this != &other) {
				this->dh_ptr = std::move(other.dh_ptr);
			}
			return *this;
		}

		// Copy constructor to enable move semantics for unique_ptr member
		CudaArray_t(CudaArray_t &&other) : dh_ptr(std::move(other.dh_ptr))
		{} // this object no longer holds the cuda array	
	};

	template <typename T>
	CudaArray_t<T> make_unique_cuda_array(size_t size)
	{
		void *ptr;
		cudaError_t err = cudaMalloc(&ptr, size*sizeof(T));
		check_cuda_return("cudaMalloc error", err);
		mem_size += size*sizeof(T);
		return CudaArray_t<T>(static_cast<T *>(ptr));
	}

	/**
	Properties of this CUDA solver
	*/
	int num_blocks;
	int block_size;

	double eps;
	int kernel_type;
	int svm_type;
	int l; // #SVs
	int mem_size; // amount of cuda memory allocated
	int startup_time;

	bool quiet_mode;

	/**
	CUDA device memory arrays
	*/
	CudaArray_t<GradValue_t> dh_gmax; 
	CudaArray_t<GradValue_t> dh_gmax2; 
	CudaArray_t<int> dh_gmax_idx; 
	CudaArray_t<GradValue_t> dh_result_gmax; 
	CudaArray_t<GradValue_t> dh_result_gmax2; 
	CudaArray_t<int> dh_result_gmax_idx; 

	CudaArray_t<CValue_t> dh_obj_diff_array; 
	CudaArray_t<int> dh_obj_diff_idx;
	CudaArray_t<CValue_t> dh_result_obj_diff; 
	CudaArray_t<int> dh_result_idx; 

	CudaArray_t<SChar_t> dh_y; 
	CudaArray_t<GradValue_t> dh_G;	
	CudaArray_t<CValue_t> dh_QD; 
	CudaArray_t<int> dh_x; 
	CudaArray_t<cuda_svm_node> dh_space; 
	CudaArray_t<CValue_t> dh_x_square; 
	CudaArray_t<GradValue_t> dh_alpha; 
	CudaArray_t<char> dh_alpha_status; 	

	/**
	The following arrays are required by the reducers
	*/
	std::unique_ptr<int[]> result_idx;
	std::unique_ptr<CValue_t[]> result_obj_diff;
	std::unique_ptr<GradValue_t[]> result_gmax;
	std::unique_ptr<GradValue_t[]> result_gmax2;

	enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

private:
	/**
	Initializes the cuda device memory array
	*/
	void init_obj_diff_space(int l);
	virtual void init_gmax_space(int l); // CudaSolverNU will override this

	/**
	init_memory_arrays:
	The main method for initializing all the unique arrays.  
	The unique arrays will be automatically deallocated when they go out-of-scope.
	*/
	void init_memory_arrays(int l);

	/**
	Shows amount of memory allocated on cuda device
	*/
	void show_memory_usage(const int &total_space);

	/**
	Loads the SVM problem parameters onto cuda device
	*/
	void load_problem_parameters(const svm_problem &prob, const svm_parameter &param);

	/**
	Used by select_working_set() to find the j index
	*/
	void select_working_set_j(GradValue_t Gmax, int l);

	/**
	Utility function for finding the launch parameters for N instances
	*/
	void find_launch_parameters(int &num_blocks, int &block_size, int N);

public:

	CudaSolver(const svm_problem &prob, const svm_parameter &param, bool quiet_mode=true);
	~CudaSolver();

	void setup_solver(const SChar_t *y, double *G, double *alpha, 
		char *alpha_status, double Cp, double Cn, int l) ;

	void setup_rbf_variables(int l); // for RBF kernel only

	// return 1 if already optimal, return 0 otherwise
	virtual int select_working_set(int &out_i, int &out_j, int l);

	void update_gradient(int l);

	void compute_alpha();

	void update_alpha_status();

	void fetch_vectors(double *G, double *alpha, char *alpha_status, int l);
};

extern CudaSolver *cudaSolver;

#endif
