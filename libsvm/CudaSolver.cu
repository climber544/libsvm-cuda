#include "CudaSolver.h"
#include "cuda_device_functions.h"
#include <memory>

CudaSolver *cudaSolver;

class CudaSolver::MinIdxFunctor // class object used for cross_block_reducer() template function
{
private:
	CValue_t *input_array, *output_array;
	int *input_idx, *output_idx;

public:
	MinIdxFunctor(CValue_t *obj_diff_array, int *obj_diff_idx, CValue_t *result_obj_min, int *result_idx) 
		: input_array(obj_diff_array), output_array(result_obj_min), input_idx(obj_diff_idx), output_idx(result_idx)
	{}

	void compute(size_t reduce_blocks, size_t reduce_block_size, int N) {
		size_t share_mem_size = reduce_block_size*(sizeof(CValue_t)+sizeof(int));
		cuda_find_min_idx<<<reduce_blocks, reduce_block_size, share_mem_size>>>(input_array, input_idx, output_array, output_idx, N);
	}

	void swap() {
		std::swap(input_array, output_array);
		std::swap(input_idx, output_idx);
	}

	int process_output() {
		// int Gmin_idx = -1; 
		// cudaMemcpy(&Gmin_idx, &output_idx[0], sizeof(int), cudaMemcpyDeviceToHost); // Gmin_idx should be in the first position now
		return -1;
	}
};

class CudaSolver::GmaxFunctor // class object used for cross_block_reducer() template function
{
private:
	GradValue_t *input_array1, *input_array2, *output_array1, *output_array2;
	int *input_idx, *output_idx;
	GradValue_t Gmax, Gmax2;
public:
	GmaxFunctor(GradValue_t *dh_gmax, GradValue_t *dh_gmax2, int *dh_gmax_idx, GradValue_t *result_gmax, GradValue_t *result_gmax2, int *result_gmax_idx)
		: input_array1(dh_gmax), input_array2(dh_gmax2), input_idx(dh_gmax_idx), output_array1(result_gmax), output_array2(result_gmax2), output_idx(result_gmax_idx)
	{}

	void compute(size_t reduce_blocks, size_t reduce_block_size, int N) {
		size_t share_mem_size = reduce_block_size*(2*sizeof(GradValue_t)+sizeof(int));
		cuda_find_gmax<<<reduce_blocks, reduce_block_size, share_mem_size>>>(input_array1, input_array2, input_idx, output_array1, output_array2, output_idx, N);
	}

	void swap() {
		std::swap(input_array1, output_array1);
		std::swap(input_array2, output_array2);
		std::swap(input_idx, output_idx);
	}

	int process_output() {
		int Gmax_idx = -1; 
		/* check_cuda_return("fail to copy output_idx from device",
		cudaMemcpy(&Gmax_idx, &output_idx[0], sizeof(int), cudaMemcpyDeviceToHost)); */ // Gmax_idx should be in the first position now
		CudaSolver::check_cuda_return("fail to copy output_array1 from device",
			cudaMemcpy(&Gmax, &output_array1[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		CudaSolver::check_cuda_return("fail to copy output_array2 from device",
			cudaMemcpy(&Gmax2, &output_array2[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		return Gmax_idx;
	}

	void get_gmax_values(GradValue_t &ret_Gmax, GradValue_t &ret_Gmax2)
	{
		ret_Gmax = Gmax;
		ret_Gmax2 = Gmax2;
	}
};

void CudaSolver::init_obj_diff_space(int l)
{
	dh_obj_diff_array = make_unique_cuda_array<CValue_t>(l);
	dh_obj_diff_idx = make_unique_cuda_array<int>(l);
	dh_result_obj_diff = make_unique_cuda_array<CValue_t>(num_blocks);
	dh_result_idx = make_unique_cuda_array<int>(num_blocks);
	return;
}

void CudaSolver::init_gmax_space(int l)
{
	dh_gmax = make_unique_cuda_array<GradValue_t>(l);
	dh_gmax2 = make_unique_cuda_array<GradValue_t>(l);
	dh_gmax_idx = make_unique_cuda_array<int>(l);
	dh_result_gmax = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmax2 = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmax_idx = make_unique_cuda_array<int>(num_blocks);
	return;
}

void CudaSolver::init_memory_arrays(int l) {
	int bsize = CUDA_BLOCK_SIZE; // TODO: query device for max thread block size
	while (l / bsize < 10 && bsize > 32) {
		bsize >>= 1; // halve it
	}

	block_size = bsize;
	num_blocks = l / block_size;
	if (l % block_size != 0) ++num_blocks;

	std::cout << "CUDA Integration\n";
	std::cout << "----------------\n";
	std::cout << "Selected thread block size:         " << bsize << std::endl;
	std::cout << "Selected number of blocks:          " << num_blocks << std::endl;
	std::cout << "Problem size:                       " << l << std::endl;
	std::cout << "Gradient vector stored as:          " << typeid(GradValue_t).name() << std::endl;

	result_idx.reset(new int[num_blocks]);
	result_obj_diff.reset(new CValue_t[num_blocks]);
	result_gmax.reset(new GradValue_t[num_blocks]);
	result_gmax2.reset(new GradValue_t[num_blocks]);

	init_obj_diff_space(l);
	init_gmax_space(l);
}

CudaSolver::CudaSolver(const svm_problem &prob, const svm_parameter &param)
	: eps(param.eps), kernel_type(param.kernel_type), svm_type(param.svm_type), mem_size(0)
{
	load_problem_parameters(prob, param);
}

CudaSolver::~CudaSolver()
{
	unbind_texture();
	cudaDeviceReset();
}

void CudaSolver::compute_alpha()
{
	cuda_compute_alpha<<<1, 1>>>();
}

void CudaSolver::update_alpha_status()
{
	cuda_update_alpha_status<<<1,1>>>();
}

int CudaSolver::select_working_set_j(double Gmax, int &Gmin_idx, int l)
{

	cuda_compute_obj_diff<<<num_blocks, block_size>>>(Gmax, &dh_obj_diff_array[0], &dh_obj_diff_idx[0], l);

	MinIdxFunctor func(&dh_obj_diff_array[0], &dh_obj_diff_idx[0], &dh_result_obj_diff[0], &dh_result_idx[0]);
	Gmin_idx = cross_block_reducer(block_size, func, l);
	return Gmin_idx;
}


int CudaSolver::select_working_set(int &out_i, int &out_j, int l)
{
	GradValue_t Gmax = -GRADVALUE_MAX; // -INF;
	GradValue_t Gmax2 = -GRADVALUE_MAX; // -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;

	cuda_prep_gmax<<<num_blocks, block_size, block_size*(2*sizeof(GradValue_t)+sizeof(int))>>>(&dh_gmax[0], &dh_gmax2[0], &dh_gmax_idx[0],  
		&dh_result_gmax[0], &dh_result_gmax2[0], &dh_result_gmax_idx[0], l);

	GmaxFunctor func(&dh_gmax[0], &dh_gmax2[0], &dh_gmax_idx[0], &dh_result_gmax[0], &dh_result_gmax2[0], &dh_result_gmax_idx[0]);
	Gmax_idx = cross_block_reducer(block_size, func, l);
	func.get_gmax_values(Gmax, Gmax2);

	dbgprintf(true, "Device: Gmax_idx %d Gmax %g Gmax2 %g\n", Gmax_idx, Gmax, Gmax2);

	if(Gmax+Gmax2 < eps)
		return 1;

	select_working_set_j(Gmax, Gmin_idx, l);

	out_i = Gmax_idx;
	out_j = Gmin_idx;

	return 0;
}

void CudaSolver::update_gradient(int l)
{
	cuda_update_gradient<<<num_blocks, block_size>>>(l);
}

void CudaSolver::fetch_vectors(double *G, double *alpha, char *alpha_status, int l)
{
	cudaError_t err;
	{
		std::unique_ptr<GradValue_t[]> h_G(new GradValue_t[l]);
		err = cudaMemcpy(&h_G[0], &dh_G[0], sizeof(GradValue_t) * l, cudaMemcpyDeviceToHost);
		check_cuda_return("fail to copy from device dh_G", err);
		for (int i = 0; i < l; ++i)
			G[i] = h_G[i];
	}
	{
		std::unique_ptr<GradValue_t[]> h_alpha(new GradValue_t[l]);
		err = cudaMemcpy(&h_alpha[0], &dh_alpha[0], sizeof(GradValue_t) * l, cudaMemcpyDeviceToHost);
		check_cuda_return("fail to copy from device dh_alpha", err);
		for (int i = 0; i < l; ++i)
			alpha[i] = h_alpha[i];
	}

	err = cudaMemcpy(alpha_status, &dh_alpha_status[0], sizeof(char) * l, cudaMemcpyDeviceToHost);
	check_cuda_return("fail to copy from device dh_alpha_status", err);
}


void CudaSolver::setup_solver(const SChar_t *y, const double *QD, double *G, double *alpha, char *alpha_status, double Cp, double Cn, int l) 
{
	init_memory_arrays(l);
	
	cudaError_t err;

	// allocate space for labels
	dh_y = make_unique_cuda_array<SChar_t>(l);

	err = cudaMemcpy(&dh_y[0], y, sizeof(SChar_t) * l, cudaMemcpyHostToDevice);
	check_cuda_return("fail to copy to device for dh_y", err);

	dh_QD = make_unique_cuda_array<CValue_t>(l);
	{
		std::unique_ptr<CValue_t[]> h_QD(new CValue_t[l]);
		for (int i = 0; i < l; ++i) {
			CHECK_FLT_RANGE(QD[i]);
			h_QD[i] = static_cast<CValue_t>(QD[i]);
		}

		err = cudaMemcpy(&dh_QD[0], &h_QD[0], sizeof(CValue_t) * l, cudaMemcpyHostToDevice);
		check_cuda_return("fail to copy to device for dh_QD", err);
	}

	/** allocate space for gradient vector */
	dh_G = make_unique_cuda_array<GradValue_t>(l);

	{
		std::unique_ptr<GradValue_t[]> h_G(new GradValue_t[l]);
		for (int i = 0; i < l; ++i)
			h_G[i] = static_cast<GradValue_t>(G[i]);

		err = cudaMemcpy(&dh_G[0], &h_G[0], sizeof(GradValue_t) * l, cudaMemcpyHostToDevice);
		check_cuda_return("fail to copy to device for dh_G", err);
	}

	dh_alpha = make_unique_cuda_array<GradValue_t>(l);

	{
		std::unique_ptr<GradValue_t[]> h_alpha(new GradValue_t[l]);
		for (int i = 0; i < l; ++i)
			h_alpha[i] = static_cast<GradValue_t>(alpha[i]);

		err = cudaMemcpy(&dh_alpha[0], &h_alpha[0], sizeof(GradValue_t) * l, cudaMemcpyHostToDevice);
		check_cuda_return("fail to copy to device for dh_alpha", err);
	}

	dh_alpha_status = make_unique_cuda_array<char>(l);

	cudaMemcpy(&dh_alpha_status[0], alpha_status, sizeof(char) * l, cudaMemcpyHostToDevice);
	check_cuda_return("fail to copy to device for dh_alpha_status", err);

	/** setup constants */
	err = update_solver_variables(&dh_y[0], &dh_QD[0],  &dh_G[0], &dh_alpha[0], &dh_alpha_status[0], Cp, Cn);

	check_cuda_return("fail to setup constants/textures", err);

	return ;
}

/**
Loads: kernel_type, svm_type, gamma, coef0, degree, x
*/
void CudaSolver::load_problem_parameters(const svm_problem &prob, const svm_parameter &param)
{
	cudaError_t err;
	svm_node **x = prob.x;
	int l = prob.l;

	/** allocate space for support vectors */
	int elements = 0;
	for (int i = 0; i < l; ++i) 
	{
		const svm_node *tmp = x[i];
		while (tmp->index != -1) { // row terminator
			++elements; // count each row svm_node element
			++tmp;
		}
		++elements; // count the row terminating svm_node
	}

	/**
	NOTE: cuda_svm_node is typedef to float2
	float2.x == svm_node.index
	float2.y == svm_node.value
	*/
	std::unique_ptr<cuda_svm_node[]> x_space(new cuda_svm_node[elements]);
	for (int i = 0, j = 0; i < l; ++i) {
		const svm_node *tmp = x[i];
		while (tmp->index != -1) {
			x_space[j].x = static_cast<float>(tmp->index);
			x_space[j].y = static_cast<CValue_t>(tmp->value);
#ifdef DEBUG_VERIFY
			if (abs(tmp->value - x_space[j].y) > 1e-4) {
				std::cerr << "WARNING!: sample space value truncated by " 
					<< abs(tmp->value-x_space[j].y) << std::endl;
			}
#endif
			++j;
			++tmp;
		}
		x_space[j++].x = -1;
	}

	dh_space = make_unique_cuda_array<cuda_svm_node>(elements);

	err = cudaMemcpy(&dh_space[0], &x_space[0], sizeof(cuda_svm_node) * elements, cudaMemcpyHostToDevice);
	check_cuda_return("fail to copy to device for dh_space", err); 

	dh_x = make_unique_cuda_array<int>(l);

	{
		std::unique_ptr<int[]> h_x(new int[l]);

		int i = 0;
		bool assign_flag = false;
		for (int j = 0; j < elements; ++j)
		{
			if (!assign_flag) {
				if (i >= l) {
					throw std::runtime_error("error in updating h_x");
				}
				h_x[i] = j;
				assign_flag = true;
			}
			if (x_space[j].x == -1) {
				++i;
				assign_flag = false;
			}
		}

		err = cudaMemcpy(&dh_x[0], &h_x[0], sizeof(int) * l, cudaMemcpyHostToDevice);
		check_cuda_return("fail to copy to device for dh_x", err);
	}

	err = update_param_constants (param, &dh_x[0], &dh_space[0], sizeof(cuda_svm_node)*elements);
	check_cuda_return("fail to setup parameter constants", err);
}

void CudaSolver::setup_rbf_variables(double *x_square, int l)
{
	if (kernel_type != RBF)
		return ;

	/* x_square is only needed in computing the RBF kernel */
	std::unique_ptr<CValue_t[]> h_x_square(new CValue_t[l]);
	for (int i = 0; i < l; ++i)
		h_x_square[i] = static_cast<CValue_t>(x_square[i]);

	dh_x_square = make_unique_cuda_array<CValue_t>(l);

	cudaError_t err = cudaMemcpy(&dh_x_square[0], &h_x_square[0], sizeof(CValue_t) * l, cudaMemcpyHostToDevice);
	check_cuda_return("fail to copy to device for dh_x_square", err);

	err = update_rbf_variables(&dh_x_square[0]);
	check_cuda_return("fail to update rbf variables", err);
}

void CudaSolver::show_memory_usage(const int &total_space)
{
	printf("Total space allocated on device:	%d\n", total_space);
	int devNum;
	cudaGetDevice(&devNum);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, devNum);
	printf("Total global memory:                %lu\n",  devProp.totalGlobalMem);
	printf("Percentage allocated:               %f%%\n", (double)total_space/devProp.totalGlobalMem * 100);
}

