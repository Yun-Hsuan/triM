GPU_LIB		:= libuni10_gpu_magma

GPU_MAGMA_OBJ := $(GPU_LIB)/qnum.o $(GPU_LIB)/bond.o \
  $(GPU_LIB)/block.o $(GPU_LIB)/blockreal.o $(GPU_LIB)/blockcomplex.o $(GPU_LIB)/blocktools.o \
  $(GPU_LIB)/uni10_tools.o $(GPU_LIB)/uni10_tools_gpu_kernel.o $(GPU_LIB)/uni10_tools_gpu_magma.o \
  $(GPU_LIB)/uni10_dgemm.o $(GPU_LIB)/uni10_lapack_gpu_magma.o \
  $(GPU_LIB)/hdf5io.o \
  $(GPU_LIB)/matrix.o $(GPU_LIB)/matrixreal.o $(GPU_LIB)/matrixcomplex.o $(GPU_LIB)/matrixtools.o \
  $(GPU_LIB)/UniTensor.o $(GPU_LIB)/UniTensorreal.o $(GPU_LIB)/UniTensorcomplex.o $(GPU_LIB)/UniTensortools.o $(GPU_LIB)/network.o \

MAGMA_SRC_ROOT := /home/Yun-Hsuan/myLib/magma

CC		:= icpc
CFLAGS 		:= -O3 -Wall -openmp -std=c++11
MAGMA_CFLAGS	:= -O3 -fPIC -DADD_ -Wall -Wshadow -openmp -DMAGMA_WITH_MKL -DMKL_ILP64 -pedantic -Wno-long-long -std=c++11 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200 
MAGMA_INC	:= -I$(CUDADIR)/include \
  -I$(MKLROOT)/include \
  -I$(MAGMA_SRC_ROOT)/include \

LDFLAGS		:= -fPIC -openmp -Wl,-rpath,$(MAGMA_SRC_ROOT)/lib 
MAGMA_LIB	:= -L$(MAGMA_SRC_ROOT)/lib -lmagma\
  -L$(CUDADIR)/lib64 \
  -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm -lcublas -lcusparse -lcudart -lcusolver

NVCCOPT 	:= -std=c++11 -O3 -m64 -arch=sm_30
NVCC 		:= nvcc

SRC 		:= /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10

UNI10_SRC_ROOT 	:= /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10

all: libuni10_gpu.a

cpu: cpu_main.o cpu_main.e

gpu: gpu_main.o gpu_main.e

OBJ		:= gpu_main.o

$(GPU_LIB)/qnum.o: $(UNI10_SRC_ROOT)/datatype/lib/Qnum.cpp $(UNI10_SRC_ROOT)/datatype/Qnum.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/bond.o: $(UNI10_SRC_ROOT)/data-structure/lib/Bond.cpp $(UNI10_SRC_ROOT)/data-structure/Bond.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/block.o: $(UNI10_SRC_ROOT)/data-structure/lib/Block.cpp $(UNI10_SRC_ROOT)/datatype.hpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/blockreal.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockReal.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/blockcomplex.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockComplex.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/blocktools.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockTools.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/uni10_tools.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools.cpp $(UNI10_SRC_ROOT)/tools/uni10_tools.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/uni10_tools_cpu.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools_cpu.cpp $(UNI10_SRC_ROOT)/tools/uni10_tools.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/uni10_tools_gpu_kernel.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools_gpu_kernel.cu $(UNI10_SRC_ROOT)/tools/uni10_tools_gpu_kernel.h $(UNI10_SRC_ROOT)/tools/uni10_tools.h $(UNI10_SRC_ROOT)/tools/helper_uni10.h
	$(NVCC) $(NVCCOPT) -DCUDA_SUPPORT -I $(SRC) -c -o $@ $<

$(GPU_LIB)/uni10_tools_gpu_magma.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools_gpu_magma.cpp $(UNI10_SRC_ROOT)/tools/uni10_tools.h $(UNI10_SRC_ROOT)/tools/helper_uni10.h
	$(CC) $(MAGMA_CFLAGS) $(MAGMA_INC) -DCUDA_SUPPORT -I $(SRC) -c -o $@ $< 


$(GPU_LIB)/uni10_dgemm.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_dgemm.cu $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h $(UNI10_SRC_ROOT)/tools/helper_uni10.h
	$(NVCC) -c $(NVCCOPT) -DCUDA_SUPPORT -I $(SRC) $< -o $@

$(GPU_LIB)/uni10_lapack_cpu.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_lapack_cpu.cpp $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack_wrapper.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/uni10_lapack_gpu_magma.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_lapack_gpu_magma.cpp $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h $(UNI10_SRC_ROOT)/tools/uni10_tools.h $(UNI10_SRC_ROOT)/tools/helper_uni10.h
	$(CC) $(MAGMA_CFLAGS) $(MAGMA_INC) -DCUDA_SUPPORT -I $(SRC) -c -o $@ $< 

$(GPU_LIB)/hdf5io.o: $(UNI10_SRC_ROOT)/hdf5io/lib/uni10_hdf5io.cpp $(UNI10_SRC_ROOT)/hdf5io/uni10_hdf5io.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/matrix.o: $(UNI10_SRC_ROOT)/tensor-network/lib/Matrix.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/matrixreal.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixReal.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
	
$(GPU_LIB)/matrixcomplex.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixComplex.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/matrixtools.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixTools.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/UniTensor.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensor.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/UniTensorreal.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorReal.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/UniTensorcomplex.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorComplex.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/UniTensortools.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorTools.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

$(GPU_LIB)/network.o: $(UNI10_SRC_ROOT)/tensor-network/lib/Network.cpp $(UNI10_SRC_ROOT)/tensor-network/Network.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

libuni10_gpu.a: $(GPU_MAGMA_OBJ)
	ar rcs $(GPU_LIB)/$@ $^


%.o: %.cpp
	$(CC) $(CFLAGS) $(MAGMA_INC) -c -o $@ $<

%.e: %.cpp
	$(CC) $(LDFLAGS) -o $@ $(OBJ) $(GPU_LIB)/libuni10_gpu.a $(MAGMA_LIB)

clean:
	rm *.e
