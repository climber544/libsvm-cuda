#include "cuda_solverNU.h"
#include "svm_device.h"

/*********** NuGmaxReducer ****************/
class CudaSolverNU::NuGmaxReducer // class object used for cross_block_reducer() template function
{
private:
	GradValue_t *input_array1, *output_array1; // Gmaxp
	GradValue_t *input_array2, *output_array2; // Gmaxn
	GradValue_t *input_array3, *output_array3; // Gmaxp2
	GradValue_t *input_array4, *output_array4; // Gmaxn2
	int *input_idx1, *output_idx1; // Gmaxp_idx
	int *input_idx2, *output_idx2; // Gmaxn_idx
	GradValue_t Gmaxp, Gmaxn;
	GradValue_t Gmaxp2, Gmaxn2;

public:
	NuGmaxReducer(GradValue_t *dh_gmaxp, GradValue_t *dh_gmaxn, GradValue_t *dh_gmaxp2, GradValue_t *dh_gmaxn2, 
		int *dh_gmaxp_idx, int *dh_gmaxn_idx,
		GradValue_t *result_gmaxp, GradValue_t *result_gmaxn, GradValue_t *result_gmaxp2, GradValue_t *result_gmaxn2,
		int *result_gmaxp_idx, int *result_gmaxn_idx)
		: input_array1(dh_gmaxp), output_array1(result_gmaxp), /* Gmaxp */
		input_array2(dh_gmaxn), output_array2(result_gmaxn), /* Gmaxn */
		input_array3(dh_gmaxp2), output_array3(result_gmaxp2), /* Gmaxp2 */
		input_array4(dh_gmaxn2), output_array4(result_gmaxn2), /* Gmaxn2 */
		input_idx1(dh_gmaxp_idx), output_idx1(result_gmaxp_idx), /* Gmaxp_idx */
		input_idx2(dh_gmaxn_idx), output_idx2(result_gmaxn_idx) /* Gmaxn_idx */
	{}

	void compute(size_t reduce_blocks, size_t reduce_block_size, int N) {

		size_t share_mem_size = reduce_block_size * (4 * sizeof(GradValue_t) + 2 * sizeof(int));

		find_nu_gmax_param param;
		param.dh_gmaxp = input_array1;
		param.dh_gmaxn = input_array2;
		param.dh_gmaxp2 = input_array3;
		param.dh_gmaxn2 = input_array4;
		param.dh_gmaxp_idx = input_idx1;
		param.dh_gmaxn_idx = input_idx2;

		param.result_gmaxp = output_array1;
		param.result_gmaxn = output_array2;
		param.result_gmaxp2 = output_array3;
		param.result_gmaxn2 = output_array4;
		param.result_gmaxp_idx = output_idx1;
		param.result_gmaxn_idx = output_idx2;

		launch_cuda_find_nu_gmax(reduce_blocks, reduce_block_size, share_mem_size, param, N);
	}

	void swap() {
		std::swap(input_array1, output_array1);
		std::swap(input_array2, output_array2);
		std::swap(input_array3, output_array3);
		std::swap(input_array4, output_array4);
		std::swap(input_idx1, output_idx1);
		std::swap(input_idx2, output_idx2);
	}

	int process_output() {
		/* check_cuda_return("fail to copy output_idx from device",
		cudaMemcpy(&Gmax_idx, &output_idx[0], sizeof(int), cudaMemcpyDeviceToHost)); */ // Gmax_idx should be in the first position now
		check_cuda_return("fail to copy output_array1 from device",
			cudaMemcpy(&Gmaxp, &output_array1[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		check_cuda_return("fail to copy output_array2 from device",
			cudaMemcpy(&Gmaxn, &output_array2[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		check_cuda_return("fail to copy output_array1 from device",
			cudaMemcpy(&Gmaxp2, &output_array3[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		check_cuda_return("fail to copy output_array2 from device",
			cudaMemcpy(&Gmaxn2, &output_array4[0], sizeof(GradValue_t), cudaMemcpyDeviceToHost));
		return 0;
	}

	void get_gmax_values(GradValue_t &ret_Gmaxp, GradValue_t &ret_Gmaxn, GradValue_t &ret_Gmaxp2, GradValue_t &ret_Gmaxn2)
	{
		ret_Gmaxp = Gmaxp;
		ret_Gmaxn = Gmaxn;
		ret_Gmaxp2 = Gmaxp2;
		ret_Gmaxn2 = Gmaxn2;
	}
};

/********* NuMinIdxReducer **************/
class CudaSolverNU::NuMinIdxReducer
{
private:
	CValue_t *input_array, *output_array;
	int *input_idx, *output_idx;

public:
	NuMinIdxReducer(CValue_t *obj_diff_array, int *obj_diff_idx, CValue_t *result_obj_min, int *result_idx)
		: input_array(obj_diff_array), output_array(result_obj_min), input_idx(obj_diff_idx), output_idx(result_idx)
	{}

	void compute(size_t reduce_blocks, size_t reduce_block_size, int N)
	{
		size_t share_mem_size = reduce_block_size*(sizeof(CValue_t) + sizeof(int));
		launch_cuda_find_nu_min_idx(reduce_blocks, reduce_block_size, share_mem_size, input_array, input_idx, output_array, output_idx, N);
	}

	void swap()
	{
		std::swap(input_array, output_array);
		std::swap(input_idx, output_idx);
	}

	int process_output()
	{
		// int Gmin_idx = -1; 
		// cudaMemcpy(&Gmin_idx, &output_idx[0], sizeof(int), cudaMemcpyDeviceToHost); // Gmin_idx should be in the first position now
		return -1;
	}
};

void CudaSolverNU::init_gmax_space(int l)
{
	dh_gmaxp = make_unique_cuda_array<GradValue_t>(l);
	dh_gmaxn = make_unique_cuda_array<GradValue_t>(l);
	dh_gmaxp2 = make_unique_cuda_array<GradValue_t>(l);
	dh_gmaxn2 = make_unique_cuda_array<GradValue_t>(l);
	dh_gmaxp_idx = make_unique_cuda_array<int>(l);
	dh_gmaxn_idx = make_unique_cuda_array<int>(l);

	dh_result_gmaxp = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmaxn = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmaxp2 = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmaxn2 = make_unique_cuda_array<GradValue_t>(num_blocks);
	dh_result_gmaxp_idx = make_unique_cuda_array<int>(num_blocks);
	dh_result_gmaxn_idx = make_unique_cuda_array<int>(num_blocks);

	return;
}

void CudaSolverNU::select_working_set_j(GradValue_t Gmaxp, GradValue_t Gmaxn, int l)
{
	launch_cuda_compute_nu_obj_diff(num_blocks, block_size, Gmaxp, Gmaxn, &dh_obj_diff_array[0], &dh_obj_diff_idx[0], l);

	NuMinIdxReducer func(&dh_obj_diff_array[0], &dh_obj_diff_idx[0], &dh_result_obj_diff[0], &dh_result_idx[0]);

	cross_block_reducer(block_size, func, l);

	return ;
}

int CudaSolverNU::select_working_set(int &out_i, int &out_j, int l)
{
	GradValue_t Gmaxp = -GRADVALUE_MAX;
	GradValue_t Gmaxp2 = -GRADVALUE_MAX;

	GradValue_t Gmaxn = -GRADVALUE_MAX;
	GradValue_t Gmaxn2 = -GRADVALUE_MAX;

	launch_cuda_prep_nu_gmax(num_blocks, block_size, &dh_gmaxp[0], &dh_gmaxn[0], &dh_gmaxp2[0], &dh_gmaxn2[0],
		&dh_gmaxp_idx[0], &dh_gmaxn_idx[0], l);

	NuGmaxReducer func(&dh_gmaxp[0], &dh_gmaxn[0], &dh_gmaxp2[0], &dh_gmaxn2[0],
		&dh_gmaxp_idx[0], &dh_gmaxn_idx[0],
		&dh_result_gmaxp[0], &dh_result_gmaxn[0], &dh_result_gmaxp2[0], &dh_result_gmaxn2[0], 
		&dh_result_gmaxp_idx[0], &dh_result_gmaxn_idx[0]);
	
	cross_block_reducer(block_size, func, l);

	func.get_gmax_values(Gmaxp, Gmaxn, Gmaxp2, Gmaxn2);

	if (std::max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps)
		return 1;

	select_working_set_j(Gmaxp, Gmaxn, l);

	return 0;
}
