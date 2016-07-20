#MAGMA_SRC_ROOT := /home/Yun-Hsuan/Installization/MAGMA/magma-2.0.2
MAGMA_SRC_ROOT := /home/Yun-Hsuan/myLib/magma

CC		:= icpc
CFLAGS		:= -O3 -fPIC -DADD_ -Wall -Wshadow -openmp -DMAGMA_WITH_MKL -DMKL_ILP64 -pedantic -Wno-long-long -std=c++11 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200 
INC		:= -I$(CUDADIR)/include \
  -I$(MKLROOT)/include \
  -I$(MAGMA_SRC_ROOT)/include \

LDFLAGS		:= -fPIC -openmp -Wl,-rpath,$(MAGMA_SRC_ROOT)/lib 
LIB		:= -L$(MAGMA_SRC_ROOT)/lib -lmagma\
  -L$(CUDADIR)/lib64 \
  -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -lm -lcublas -lcusparse -lcudart
  
#$(info INC is $(INC))

OBJ	:= magma_svd.o

#all: gpu_magma_test.o gpu_magma_test.e
all: magma_svd.o magma_svd.e

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

%.e: %.cpp
	$(CC) $(LDFLAGS) -o $@ $(OBJ) $(LIB)

clean:
	rm *.e *.o
