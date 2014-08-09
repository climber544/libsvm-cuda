NVCC = /opt/apps/cuda/6.0/bin/nvcc
CXX = icpc

COMPAT_FLAG=-Xcompiler "-std=c++11"
INCLUDE_FLAG=-I${TACC_CUDA_INC}
CCBIN_FLAG = -ccbin=icpc
CCFLAGS := $(CCBIN_FLAG) -m64 -O3 -cudart static
LDFLAGS := $(CCBIN_FLAG) -m64 -O3 
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_35
LIBRARIES := -lcudart

all: libsvm.a

cuda_solver.o: cuda_solver.cpp
	$(NVCC) $(INCLUDE_FLAG) $(CCFLAGS) $(COMPAT_FLAG) $(GENCODE_FLAGS) -o $@ -c $<

cuda_solverNU.o: cuda_solverNU.cpp
	$(NVCC) $(INCLUDE_FLAG) $(CCFLAGS) $(COMPAT_FLAG) $(GENCODE_FLAGS) -o $@ -c $<

svm.o: svm.cpp
	$(NVCC) $(INCLUDE_FLAG) $(CCFLAGS) $(COMPAT_FLAG) $(GENCODE_FLAGS) -o $@ -c $<

svm_device.o: svm_device.cu
	$(NVCC) $(INCLUDE_FLAG) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

libsvm.a: cuda_solver.o cuda_solverNU.o svm.o svm_device.o
	ar cr $@ $+ 
	ranlib $@

clean:
	rm -f libsvm.a cuda_solver.o cuda_solverNU.o smv.o svm_device.o
