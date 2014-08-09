NVCC = /opt/apps/cuda/6.0/bin/nvcc
CXX = icpc

COMPAT_FLAG=-Xcompiler "-std=c++11"
INCLUDE_FLAG=-I${TACC_CUDA_INC} -I../libsvm
CCBIN_FLAG = -ccbin=icpc
CCFLAGS := $(CCBIN_FLAG) -m64 -O3 -cudart static
LDFLAGS := $(CCBIN_FLAG) -m64 -O3 
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_35
LIBRARIES := -L../libsvm -lsvm -lcudart

all: svm-train

svm-train.o: svm-train.c
	$(NVCC) $(INCLUDE_FLAG) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

svm-train: svm-train.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f svm-train.o
