#ifndef _CUDA_NU_SOLVER_H_
#define _CUDA_NU_SOLVER_H_

#include "CudaSolver.h"

class CudaNuSolver : public CudaSolver
{
public:
	virtual int select_working_set(int &out_i, int &out_j, int l);
};

#endif