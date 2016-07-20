#ifndef UNI10_TOOL_GPU_KERNEL_H
#define UNI10_TOOL_GPU_KERNEL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <uni10/tools/uni10_tools.h>

namespace uni10{

const size_t UNI10_GPU_GLOBAL_MEM = ((size_t)5) * 1<<30;
const int UNI10_THREADMAX = 1024;
const int UNI10_BLOCKMAX = 65535;

__global__ void gpu_rand(double* elem, size_t N);

//void elemRand(double* elem, size_t N, bool ongpu);

__global__ void _setDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N);

//void setDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

__global__ void _getDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N);

//void getDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

__global__ void _reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem);

//void reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem);

}; 	/* namespace uni10 */

#endif

