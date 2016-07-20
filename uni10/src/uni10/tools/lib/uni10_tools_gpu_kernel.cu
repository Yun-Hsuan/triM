#include <uni10/tools/uni10_tools.h>
#include <uni10/tools/uni10_tools_gpu_kernel.h>
#include <uni10/tools/helper_uni10.h>

namespace uni10{

__global__ void gpu_rand(double* elem, size_t N){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int r = (1664525 * ((1664525 * idx + 1013904223) % UINT_MAX) + 1013904223) % UINT_MAX;
  if(idx < N)
    elem[idx] = double(r) / UINT_MAX;
}

void elemRand(double* elem, size_t N, bool ongpu){
  if(ongpu){
    size_t blockNum = (N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    gpu_rand<<<gridSize, UNI10_THREADMAX>>>(elem, N);
  }
  else{
    for(size_t i = 0; i < N; i++)
      elem[i] = ((double)rand()) / RAND_MAX; //lapack_uni01_sampler();
  }
}

__global__ void _setDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < diag_N && idx < M && idx < N)
    elem[idx * N + idx] = diag_elem[idx];
}

__global__ void _getDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < diag_N && idx < M && idx < N)
    diag_elem[idx] = elem[idx * N + idx];
}

void setDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu){
  if((ongpu)){
    size_t blockNum = (N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    if(diag_ongpu){
      _setDiag<<<gridSize, UNI10_THREADMAX>>>(elem, diag_elem, M, N, diag_N);
    }
    else{
      size_t memsize = diag_N * sizeof(double);
      double* src_elem;
      checkCudaError(cudaMalloc(&src_elem, memsize));
      checkCudaError(cudaMemcpy(src_elem, diag_elem, memsize, cudaMemcpyHostToDevice));
      //printf("mvGPU");
      _setDiag<<<gridSize, UNI10_THREADMAX>>>(elem, src_elem, M, N, diag_N);
      checkCudaError(cudaFree(src_elem));
    }
  }else{
    double* src_elem;
    if(diag_ongpu){
      size_t memsize = diag_N * sizeof(double);
      src_elem = (double*) malloc(memsize);
      checkCudaError(cudaMemcpy(src_elem, diag_elem, memsize, cudaMemcpyDeviceToHost));
      //printf("mvCPU");
    }
    else
      src_elem = diag_elem;
    int min = M < N ? M : N;
    min = min < diag_N ? min : diag_N;
    for(size_t i = 0; i < min; i++)
      elem[i * N + i] = src_elem[i];
  }
}

void getDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu){
  if((ongpu)){
    size_t blockNum = (N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    if(diag_ongpu){
      _getDiag<<<gridSize, UNI10_THREADMAX>>>(elem, diag_elem, M, N, diag_N);
    }
    else{
      size_t memsize = diag_N * sizeof(double);
      double* tmp_elem;
      checkCudaError(cudaMalloc(&tmp_elem, memsize));
      _getDiag<<<gridSize, UNI10_THREADMAX>>>(elem, tmp_elem, M, N, diag_N);
      checkCudaError(cudaMemcpy(diag_elem, tmp_elem, memsize, cudaMemcpyHostToDevice));
      //printf("mvGPU");
      checkCudaError(cudaFree(tmp_elem));
    }
  }else{
    double* tmp_elem;
    size_t memsize = diag_N * sizeof(double);
    if(diag_ongpu)
      tmp_elem = (double*)malloc(memsize);
    else
      tmp_elem = diag_elem;
    int min = M < N ? M : N;
    min = min < diag_N ? min : diag_N;
    for(size_t i = 0; i < min; i++)
      tmp_elem[i] = elem[i * N + i];
    if(diag_ongpu){
      checkCudaError(cudaMemcpy(diag_elem, tmp_elem, memsize, cudaMemcpyDeviceToHost));
      //printf("mvCPU");
    }
  }
}

__global__ void _reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem){
  size_t oldIdx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  size_t idx = oldIdx;
  size_t newIdx = 0;
  if(idx < elemNum){
    for(int i = 0; i < bondNum; i++){
      newIdx += (idx/offset[i]) * offset[bondNum + i];
      idx = idx % offset[i];
    }
    newElem[newIdx] = oldElem[oldIdx];
  }
}

void reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem){
  size_t* D_offset;
  assert(cudaMalloc((void**)&D_offset, 2 * sizeof(size_t) * bondNum) == cudaSuccess);
  assert(cudaMemcpy(D_offset, offset, 2 * sizeof(size_t) * bondNum, cudaMemcpyHostToDevice) == cudaSuccess);
  size_t blockNum = (elemNum + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
  dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
  _reshapeElem<<<gridSize, UNI10_THREADMAX>>>(oldElem, bondNum, elemNum, D_offset, newElem);
}

};
