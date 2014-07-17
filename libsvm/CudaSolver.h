#ifndef _CUDA_GLOBALS_H_
#define _CUDA_GLOBALS_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_constants.h"
#include "svm.h"
#include <iostream>
#include <stdexcept>
#include <memory>

#ifdef __CUDACC__
#define ALIGN(x)  __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
// Visual C++ .NET and later
#define ALIGN(x) __declspec(align(x))
#else
#if defined(__GNUC__)
// GCC
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
// all other compilers
#define ALIGN(x)
#endif
#endif
#endif

//#define USE_CONSTANT_SVM_NODE
//#define DEBUG_VERIFY // for verifying ... more critical than debugging
//#define DEBUG_CHECK // for debugging

#ifdef DEBUG_VERIFY
#define CHECK_FLT_RANGE(x)	\
	if (x < -FLT_MAX || x > FLT_MAX) \
	printf("DEBUG_VERIFY WARNING: CHECK_FLT_RANGE fail in %s:%d\n", __FILE__, __LINE__);
#define CHECK_FLT_INF(x)	\
	if (x == CUDART_INF_F || x == -CUDART_INF_F)	\
	printf("DEBUG_VERIFY WARNING: CHECK_FLT_INF fail in %s:%d\n", __FILE__, __LINE__);
#else
#define CHECK_FLT_RANGE(x)
#define CHECK_FLT_INF(x)

#endif
#ifdef DEBUG_CHECK
#define dbgprintf(debug, ...) if (debug) printf (__VA_ARGS__)
#else
#define dbgprintf(debug, ...)
#endif

typedef signed char SChar_t;

typedef float CValue_t;
#define CVALUE_MAX  FLT_MAX

#define CUDA_BLOCK_SIZE	256

typedef double GradValue_t;
#define GRADVALUE_MAX	DBL_MAX
//typedef float GradValue_t;
//#define GRADVALUE_MAX FLT_MAX

/**
cuda_svm_node.x == svm_node.index
cuda_svm_node.y == svm_node.value
*/
typedef float2 cuda_svm_node;

#define TAU 1e-12

class CudaSolver
{
protected:
	static inline void check_cuda_return(const char *msg, cudaError_t err)
	{
		if (err != cudaSuccess) {
			std::cerr << "CUDA Error (" << __FILE__ << ":" << __LINE__ << "): ";
			std::cerr << msg << " " << cudaGetErrorString(err) << std::endl;
			cudaDeviceReset();
			throw std::runtime_error(msg);
		}
	}

protected:

	/**
	Cross cuda block reducers -- every device function only reducers per block
	We need to reduce across blocks as well
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

	class MinIdxFunctor; // class object used for cross_block_reducer() template function
	class GmaxFunctor;  // class object used for cross_block_reducer() template function

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
		T &operator[](size_t idx) {
			return dh_ptr[idx];
		}

		CudaArray_t(T *ptr) {
			dh_ptr.reset(ptr);
		}

		CudaArray_t() {}

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
	double gamma;

	int kernel_type;
	int svm_type;

	int mem_size; // amount of cuda memory allocated

	/**
	CUDA device memory arrays
	*/
	CudaArray_t<GradValue_t> dh_gmax; // GradValue_t *dh_gmax;
	CudaArray_t<GradValue_t> dh_gmax2; // GradValue_t *dh_gmax2;
	CudaArray_t<int> dh_gmax_idx; // int *dh_gmax_idx;
	CudaArray_t<GradValue_t> dh_result_gmax; // GradValue_t *dh_result_gmax;
	CudaArray_t<GradValue_t> dh_result_gmax2; // GradValue_t *dh_result_gmax2;
	CudaArray_t<int> dh_result_gmax_idx; // int *dh_result_gmax_idx;

	CudaArray_t<CValue_t> dh_obj_diff_array; // CValue_t *dh_obj_diff_array;
	CudaArray_t<int> dh_obj_diff_idx;
	CudaArray_t<CValue_t> dh_result_obj_diff; // CValue_t *dh_result_obj_diff;
	CudaArray_t<int> dh_result_idx; // int *dh_result_idx;

	CudaArray_t<SChar_t> dh_y; // SChar_t *dh_y;
	CudaArray_t<GradValue_t> dh_G; // GradValue_t *dh_G;	
	CudaArray_t<CValue_t> dh_QD; // CValue_t *dh_QD;
	CudaArray_t<int> dh_x; // int *dh_x;
	CudaArray_t<cuda_svm_node> dh_space; // cuda_svm_node *dh_space;
	CudaArray_t<CValue_t> dh_x_square; // CValue_t *dh_x_square;
	CudaArray_t<GradValue_t> dh_alpha; // GradValue_t *dh_alpha;
	CudaArray_t<char> dh_alpha_status; // char *dh_alpha_status;	

	/**
	The following arrays are required by the reducers
	*/
	std::unique_ptr<int[]> result_idx;
	std::unique_ptr<CValue_t[]> result_obj_diff;
	std::unique_ptr<GradValue_t[]> result_gmax;
	std::unique_ptr<GradValue_t[]> result_gmax2;

	enum { LOWER_BOUND = 0, UPPER_BOUND = 1, FREE = 2 };

	int select_working_set_j(double Gmax, int &Gmin_idx, int l);
	void init_obj_diff_space(int l);
	void init_gmax_space(int l);
	void show_memory_usage(const int &total_space);

	/**
	Initializes all the unique arrays.  These unique arrays will be automatically deallocated when they go out-of-scope.
	*/
	void init_memory_arrays(int l);

	void load_problem_parameters(const svm_problem &prob, const svm_parameter &param);

public:

	CudaSolver(const svm_problem &prob, const svm_parameter &param);
	~CudaSolver();

	void setup_solver(const SChar_t *y, const double *QD, double *G, double *alpha, 
		char *alpha_status, double Cp, double Cn, int l) ;

	void setup_rbf_variables(double *x_square, int l); // for RBF kernel only

	// return 1 if already optimal, return 0 otherwise
	int select_working_set(int &out_i, int &out_j, int l);

	void update_gradient(int l);

	void compute_alpha();

	void update_alpha_status();

	void fetch_vectors(double *G, double *alpha, char *alpha_status, int l);

};

extern CudaSolver *cudaSolver;

#endif