#ifndef _SVM_DEFS_H_
#define _SVM_DEFS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_constants.h"

#include <iostream>
#include <stdexcept>
#include <cfloat>
#include <stdint.h>
const double libsvm_cuda_version = 0.318;

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

#define USE_LIBSVM_SPARSE_FORMAT 0
//#define DEBUG_VERIFY // for verifying ... more critical than debugging
//#define DEBUG_CHECK // for debugging
//#define DEBUG_TRACE // for tracing calls

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

#ifdef DEBUG_TRACE
#define logtrace(...)	printf(__VA_ARGS__)
#else
#define logtrace(...)
#endif

#ifdef DEBUG_CHECK
#define dbgprintf(debug, ...) if (debug) printf (__VA_ARGS__)
#define check_cuda_kernel_launch(msg)	check_cuda_return(msg, cudaDeviceSynchronize());
#else
#define dbgprintf(debug, ...)
#define check_cuda_kernel_launch(msg)
#endif

typedef signed char SChar_t;

typedef float CValue_t; // used for computing kernel values
#define CVALUE_MAX  FLT_MAX

#define THREADS_PER_BLOCK	256
#define WARP_SIZE			32

#define USE_DOUBLE_GRADIENT 0
#if USE_DOUBLE_GRADIENT // used for storing gradient values
typedef double GradValue_t;
#define GRADVALUE_MAX	DBL_MAX
#else
typedef float GradValue_t;
#define GRADVALUE_MAX FLT_MAX
#endif

#if USE_LIBSVM_SPARSE_FORMAT
/**
cuda_svm_node.x == svm_node.value
cuda_svm_node.y == svm_node.index
*/
typedef float2 cuda_svm_node;
#else
typedef float1 cuda_svm_node;
#endif

struct ALIGN(16) CacheNode {
	struct CacheNode *next; // next node in LRU list
	struct CacheNode *prev; // previous node in LRU list
	int col_idx;   // column that this buffer currently represents
	int stage_idx; // column that this buffer is being modifed for
	bool used; // cache node is currently being read
	CValue_t *column; // buffer for column "col_idx", unless it is being staged
};

struct CacheNodeBucket {
	int last_seen;
	CacheNode *column;
};

#define WORD_SIZE	32

#define TAU 1e-12

static inline void _check_cuda_return(const char *msg, cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error (" << file << ":" << line << "): ";
		std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
		cudaDeviceReset();
		throw std::runtime_error(msg);
	}
}
#define check_cuda_return(msg, err)	{	_check_cuda_return(msg, err, __FILE__, __LINE__); }

#endif