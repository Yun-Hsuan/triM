/****************************************************************************
*  @file uni10_lapack_gpu.cu
*  @license
*    Universal Tensor Network Library
*    Copyright (c) 2013-2016
*    National Taiwan University
*    National Tsing-Hua University
*
*    This file is part of Uni10, the Universal Tensor Network Library.
*
*    Uni10 is free software: you can redistribute it and/or modify
*    it under the terms of the GNU Lesser General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Uni10 is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public License
*    along with Uni10.  If not, see <http://www.gnu.org/licenses/>.
*  @endlicense
*  @brief Implementation file for the BLAS and LAPACK wrappers
*  @author Yun-Da Hsieh, Yun-Hsuan Chou
*  @date 2014-05-06
*  @since 0.1.0
*
*****************************************************************************/
#ifdef MKL
  #include "mkl.h"
#else
  #include <uni10/numeric/lapack/uni10_lapack_wrapper.h>
#endif

#include <uni10/numeric/lapack/uni10_lapack.h>

namespace uni10{

const size_t GPU_OPERATE_MEM = UNI10_GPU_GLOBAL_MEM / 3;

void matrixMul(double* A, double* B, int M, int N, int K, double* C, bool ongpuA, bool ongpuB, bool ongpuC){
  //std::cout << "ongpuA: " << ongpuA << "  ongpuB: " << ongpuB << "  ongpuC: " << ongpuC << std::endl;
  double alpha = 1, beta = 0;
  if(GMODE == 2){
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle));
    assert( ongpuA && ongpuB && ongpuC );
    checkCublasError(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
  }
  else if(GMODE == 1){
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle));
    mmtype types[] = {MM_DDD, MM_DDH, MM_DHD, MM_DHH, MM_HDD, MM_HDH, MM_HHD, MM_HHH};
    int mm_idx = 0;
    int p, q;
    if(!ongpuA)
      mm_idx |= 4;
    if(!ongpuB)
      mm_idx |= 2;
    if(!ongpuC)
      mm_idx |= 1;
    //printf("mm_idx = %d\n", mm_idx);
    //printf("M = %u, K = %u, N = %u\n", M, K, N);
    mmtype mm_t = types[mm_idx];
    size_t elemPool = GPU_OPERATE_MEM / sizeof(double);
    size_t min_chunk_size = 8;
    int KM_min_ratio = 4;
    if(mm_t == MM_DDH){
      p = ((M * N) + elemPool - 1) / elemPool;
      q = 1;
    }
    else if(mm_t == MM_DHD){
      if(K * N < elemPool){	//allocate K * N
	p = 1;
	q = 1;
      }
      else{	// allocate K * qN + M * qN;
	if(K / M < KM_min_ratio)
	  p = (KM_min_ratio * M + K - 1) / K;
	if(M / p < min_chunk_size)
	  p = M / min_chunk_size;
	int pM = M / p;
	q = ((K + pM) * N + elemPool - 1) / elemPool;
      }
    }
    else if(mm_t == MM_HDD){
      p = (M * K + elemPool - 1) / elemPool;
      q = 1;
    }
    else if(mm_t == MM_DHH){
      if(K * N + M * N < elemPool){
	p = 1;
	q = 1;
      }
      else{	// The same as MM_DHD
	if(K / M < KM_min_ratio)
	  p = (KM_min_ratio * M + K - 1) / K;
	if(M / p < min_chunk_size)
	  p = M / min_chunk_size;
	int pM = M / p;
	q = ((K + pM) * N + elemPool - 1) / elemPool;
      }
    }
    else if(mm_t == MM_HDH){
      q = 1;
      p = (M * (K + N) + elemPool - 1) / elemPool;
    }
    else if(mm_t == MM_HHD){
      if((K * N + min_chunk_size * K) < elemPool){
	q = 1;
	size_t elem_left = elemPool - K * N;
	p = (M * K + elem_left - 1) / elem_left;
      }
      else{
	size_t elem_left = elemPool - min_chunk_size * K;
	if(K / M < KM_min_ratio)
	  p = (KM_min_ratio * M + K - 1) / K;
	if(M / p < min_chunk_size)
	  p = M / min_chunk_size;
	int pM = M / p;
	q = ((K + pM) * N + elem_left - 1) / elem_left;
	int qN = N / q;
	elem_left = elemPool - (K * qN + M * qN);
	p = (M * K + elem_left - 1) / elem_left;
      }
    }
    else{	// MM_HHH
      if((K * N + M * N + min_chunk_size * K) < elemPool){
	q = 1;
	size_t elem_left = elemPool - (K * N + M * N);
	p = (M * K + elem_left - 1) / elem_left;
      }
      else{	// The same as MM_HHD
	size_t elem_left = elemPool - min_chunk_size * K;
	if(K / M < KM_min_ratio)
	  p = (KM_min_ratio * M + K - 1) / K;
	if(M / p < min_chunk_size)
	  p = M / min_chunk_size;
	int pM = M / p;
	q = ((K + pM) * N + elem_left - 1) / elem_left;
	int qN = N / q;
	elem_left = elemPool - (K * qN + M * qN);
	p = (M * K + elem_left - 1) / elem_left;
      }
    }
    //printf("p = %d, q = %d, mm_t = %d\n", p, q, mm_idx);
    uni10Dgemm(p, q, M, N, K, A, B, C, mm_t);
  }else{ //All matrix on host and without gpu accelerate
    dgemm((char*)"N", (char*)"N", &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
  }
}

void vectorAdd(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  double a = 1.0;
  int inc = 1;
  if(y_ongpu){ //GPU version
    cublasHandle_t cublasHandle = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    if(x_ongpu){
      checkCublasError(cublasDaxpy(cublasHandle, N, &a, X, inc, Y, inc));
    }
    else{
      // CMODE=1 is not ready
      size_t memsize = N * sizeof(double);
      double* elem = (double*)elemAllocForce(memsize, true);
      elemCopy(elem, X, memsize, true, false);
      checkCublasError(cublasDaxpy(cublasHandle, N, &a, X, inc, Y, inc));
      elemFree(elem, memsize, true);
    }
  }
  else{  //CPU version
    double *elem;
    size_t memsize = N * sizeof(double);
    if(x_ongpu){
      double* elem = (double*)elemAllocForce(memsize, false);
      elemCopy(elem, X, memsize, false, true);
    }
    else
      elem = X;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      daxpy(&chunk, &a, elem + offset, &inc, Y + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }
    if(x_ongpu)
      elemFree(elem, memsize, false);
  }

}

void vectorScal(double a, double* X, size_t N, bool ongpu){

  int inc = 1;
  if(ongpu){
    cublasHandle_t cublasHandle = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    checkCublasError(cublasDscal(cublasHandle, N, &a, X, inc));
  }
  else{
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      dscal(&chunk, &a, X + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }
  }

}// Y = a * X

void vectorMul(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// Y = Y * X, element-wise multiplication;

double vectorSum(double* X, size_t N, int inc, bool ongpu){

  if(ongpu){
    cublasHandle_t cublasHandle = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    double result;
    checkCublasError(cublasDasum(cublasHandle, N, X, inc, &result));
    return result;
  }
  else{
    double sum = 0;
    size_t idx = 0;
    for(size_t i = 0; i < N; i++){
      sum += X[idx];
      idx += inc;
    }
    return sum;
  }

}

double vectorNorm(double* X, size_t N, int inc, bool ongpu){

  if(ongpu){
    cublasHandle_t cublasHandle = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    double result;
    checkCublasError(cublasDnrm2(cublasHandle, N, X, inc, &result));
    return result;
  }
  else{
    double norm2 = 0;
    double tmp = 0;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      tmp = dnrm2(&chunk, X + offset, &inc);
      norm2 += tmp * tmp;
      offset += chunk;
      left -= INT_MAX;
    }
    return sqrt(norm2);
  }

}

__global__ void _vectorExp(double a, double* X, size_t N){

  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
    X[idx] = std::exp(a * X[idx]);

}

void vectorExp(double a, double* X, size_t N, bool ongpu){

  if(ongpu){
    size_t blockNum = (N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    _vectorExp<<<gridSize, UNI10_THREADMAX>>>(a, X, N);
  }
  else
    for(size_t i = 0; i < N; i++)
      X[i] = std::exp(a * X[i]);

}

__global__ void _diagRowMul(double* mat, double* diag, size_t M, size_t N){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  double scalar = diag[idx / N];
  if(idx < M * N)
    mat[idx] *= scalar;
}

void diagRowMul(double* mat, double* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  double* d_elem = diag;
  size_t d_memsize = M * sizeof(double);
  if(mat_ongpu){
    if(!diag_ongpu){
      d_elem = (double*)elemAllocForce(d_memsize, true);
      elemCopy(d_elem, diag, d_memsize, true, diag_ongpu);
    }
    size_t blockNum = (M * N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    _diagRowMul<<<gridSize, UNI10_THREADMAX>>>(mat, d_elem, M, N);
    if(!diag_ongpu)
      elemFree(d_elem, d_memsize, true);
  }
  else{
    if(diag_ongpu){
      d_elem = (double*)malloc(d_memsize);
      elemCopy(d_elem, diag, d_memsize, false, diag_ongpu);
    }
    for(size_t i = 0; i < M; i++)
      vectorScal(d_elem[i], &(mat[i * N]), N, mat_ongpu);
    if(diag_ongpu)
      free(d_elem);
  }

}

void diagColMul(double* mat, double* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
/*Generate a set of row vectors which form a othonormal basis
 *For the incoming matrix "elem", the number of row <= the number of column, M <= N
 */
void orthoRandomize(double* elem, int M, int N, bool ongpu){

  int eleNum = M*N;
  double *random = (double*)elemAllocForce(eleNum * sizeof(double), ongpu);
  elemRand(random, M * N, ongpu);
  int min = M < N ? M : N;
  double *S = (double*)elemAllocForce(min*sizeof(double), ongpu);
  if(M <= N){
    double *U = (double*)elemAllocForce(M * min * sizeof(double), ongpu);
    matrixSVD(random, M, N, U, S, elem, ongpu);
    elemFree(U, M * min * sizeof(double), ongpu);
  }
  else{
    double *VT = (double*)elemAllocForce(min * N * sizeof(double), ongpu);
    matrixSVD(random, M, N, elem, S, VT, ongpu);
    elemFree(VT, min * N * sizeof(double), ongpu);
  }
  elemFree(random, eleNum * sizeof(double), ongpu);
  elemFree(S, min * sizeof(double), ongpu);

}

void eigDecompose(double* Kij, int N, std::complex<double>* Eig, std::complex<double> *EigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void eigSyDecompose(double* Kij, int N, double* Eig, double* EigVec, bool ongpu){

  elemCopy(EigVec, Kij, N * N * sizeof(double), ongpu, ongpu);
  int ldA = N;
  if(ongpu){  //It's not ready. Now, it is just moved to host and decompose by CPU.
    double* EigVecH = (double*)malloc(sizeof(double)*N*N);
    double* EigH = (double*)malloc(sizeof(double)*N);
    elemCopy(EigVecH, EigVec, N * N * sizeof(double), false, ongpu );
    elemCopy(EigH, Eig, N * sizeof(double), false, ongpu );
    int lwork = -1;
    double worktest;
    int info;
    dsyev((char*)"V", (char*)"U", &N, EigVecH, &ldA, EigH, &worktest, &lwork, &info);
    assert(info == 0);
    lwork = (int)worktest;
    double* work= (double*)malloc(sizeof(double)*lwork);
    dsyev((char*)"V", (char*)"U", &N, EigVecH, &ldA, EigH, work, &lwork, &info);
    assert(info == 0);
    elemCopy(EigVec, EigVecH, N * N * sizeof(double), ongpu, false);
    elemCopy(Eig, EigH, N * sizeof(double), ongpu, false);
    free(work);
    free(EigVecH);
  }
  else{
    int lwork = -1;
    double worktest;
    int info;
    dsyev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, &worktest, &lwork, &info);
    assert(info == 0);
    lwork = (int)worktest;
    double* work= (double*)malloc(sizeof(double)*lwork);
    dsyev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, work, &lwork, &info);
    assert(info == 0);
    free(work);
  }

}

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT, bool ongpu){
  
  bool flag = M > N;

  if(ongpu){
    cusolverDnHandle_t cusolverHandle = NULL;
    checkCusolverError(cusolverDnCreate(&cusolverHandle));
    // elem copy
    size_t memsize = M * N * sizeof(double);
    double* Mij = NULL;
    checkCudaError(cudaMalloc(&Mij, memsize));
    if(flag){
      setTranspose(Mij_ori, M, N, Mij, ongpu, true); 
      int tmp = M;
      M = N;
      N = tmp;
    }else{
      checkCudaError(cudaMemcpy(Mij, Mij_ori, memsize, cudaMemcpyDeviceToDevice));
    }
    double* bufM = NULL;
    checkCudaError( cudaMalloc(&bufM, N*N*sizeof(double)));
    // cuda info
    int* info = NULL;
    checkCudaError(cudaMalloc(&info, sizeof(int)));
    checkCudaError(cudaMemset(info, 0, sizeof(int)));
    // cuda workdge
    int min = std::min(M, N);
    int ldA = N, ldu = N, ldvT = min; 
    int lwork = 0;
    double* rwork = NULL;
    double* work = NULL;
    //int K = M;
    checkCusolverError(cusolverDnDgesvd_bufferSize(cusolverHandle, N, M, &lwork));
    checkCudaError(cudaMalloc(&rwork, sizeof(double)*lwork));

    checkCudaError(cudaMalloc(&work, sizeof(double)*lwork));

    cusolverStatus_t cusolverflag = !flag ? cusolverDnDgesvd(cusolverHandle, 'A', 'A', N, M, Mij, ldA, S, bufM, ldu, U, ldvT, work, lwork, rwork, info) : cusolverDnDgesvd(cusolverHandle, 'A', 'A', N, M, Mij, ldA, S, bufM, ldu, vT, ldvT, work, lwork, rwork, info);
    checkCusolverError(cusolverflag);

    if(!flag){
      checkCudaError(cudaMemcpy(vT, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice));
    }else{
      checkCudaError(cudaMemcpy(U, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice));
      setTranspose(U, M, N, ongpu);
      setTranspose(vT, M, M, ongpu);
    }
    int h_info = 0;
    checkCudaError(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(Mij);
    cudaFree(bufM);
    cudaFree(work);
    cudaFree(info);

  }else{

    double* Mij = (double*)malloc(M * N * sizeof(double));
    memcpy(Mij, Mij_ori, M * N * sizeof(double));
    int min = std::min(M, N);
    int ldA = N, ldu = N, ldvT = min;
    int lwork = -1;
    double worktest;
    int info;
    dgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)worktest;
    double *work = (double*)malloc(lwork*sizeof(double));
    dgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(work);
    free(Mij);

  }
}

__global__ void _diagMatInv(double* diag, size_t N){

  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N)
    diag[idx] = (diag[idx] < 1E-14) ? 0 : 1. / diag[idx];

}

void matrixInv(double* A, int N, bool diag, bool ongpu){

  if(ongpu){
    if(diag){
      size_t blockNum = (N + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
      dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
      _diagMatInv<<<gridSize, UNI10_THREADMAX>>>(A, N);
      return ;
    }
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }
  else{
    if(diag){
      for(int i = 0; i < N; i++)
	A[i] = A[i] == 1E-14 ? 0 : 1/A[i];
      return;
    }
    int *ipiv = (int*)malloc(N+1 * sizeof(int));
    int info;
    dgetrf(&N, &N, A, &N, ipiv, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetrf': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    int lwork = -1;
    double worktest;
    dgetri(&N, A, &N, ipiv, &worktest, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)worktest;
    double *work = (double*)malloc(lwork * sizeof(double));
    dgetri(&N, A, &N, ipiv, work, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(ipiv);
    free(work);
  }

}

void _transposeCPU(double* A, size_t M, size_t N, double* AT){

  for(size_t i = 0; i < M; i++)
    for(size_t j = 0; j < N; j++)
      AT[j * M + i] = A[i * N + j];

}

__global__ void _transposeGPU(double* A, size_t M, size_t N, double* AT){

  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y < M && x < N)
    AT[x * M + y] = A[y * N + x];

}

void setTranspose(double* A, size_t M, size_t N, double* AT, bool ongpu, bool ongpuT){

  if(ongpu && ongpuT){
    int thread = 32;
    size_t blockXNum = (N + thread - 1) / thread;
    size_t blockYNum = (M + thread - 1) / thread;
    dim3 blockSize(thread, thread);
    dim3 gridSize(blockXNum, blockYNum);
    _transposeGPU<<<gridSize, blockSize>>>(A, M, N, AT);
  }
  else if((!ongpu) && (!ongpuT)){
    _transposeCPU(A, M, N, AT);
  }
  else{
    size_t memsize = M*N*sizeof(double);
    double* h_A = (double*)malloc(memsize);
    cudaMemcpy(h_A, A, memsize, cudaMemcpyDeviceToHost);
    _transposeCPU(h_A, M, N, AT);
    free(h_A);
  }

}

void setTranspose(double* A, size_t M, size_t N, bool ongpu){
  //need check memory and enumerate more restricty.
  size_t memsize = M * N * sizeof(double);
  double *AT;
  if(ongpu){
    cudaError_t cuflag = cudaMalloc(&AT, memsize);
    assert(cuflag == cudaSuccess);
    setTranspose(A, M, N, AT, ongpu, true);
    elemCopy(A, AT, memsize, ongpu, true);
    cudaFree(AT);
  }else{
    AT = (double*)malloc(memsize);
    setTranspose(A, M, N, AT, ongpu, false);
    elemCopy(A, AT, memsize, ongpu, false);
    free(AT);
  }

}

void setCTranspose(double* A, size_t M, size_t N, double *AT, bool ongpu, bool ongpuT){
  // conj = trans in real 
  setTranspose(A, M, N, AT, ongpu, ongpuT);
}

void setCTranspose(double* A, size_t M, size_t N, bool ongpu){
  // conj = trans in real 
  setTranspose(A, M, N, ongpu);
}

__global__ void _identity(double* mat, size_t elemNum, size_t col){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < elemNum)
    mat[idx * col + idx] = 1;
}

void setIdentity(double* elem, size_t M, size_t N, bool ongpu){

  size_t min = std::min(M, N);
  if(ongpu){
    cudaMemset(elem, 0, M * N * sizeof(double));
    size_t blockNum = (min + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    _identity<<<gridSize, UNI10_THREADMAX>>>(elem, min, N);
  }else{
    memset(elem, 0, M * N * sizeof(double));
    for(size_t i = 0; i < min; i++)
      elem[i * N + i] = 1;
  }

}
// [1, 1 ,.. 1]
__global__ void _Diagidentity(double* mat, size_t elemNum){
  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < elemNum)
    mat[idx] = 1;
}

void setDiagIdentity(double* elem, size_t M, size_t N, bool ongpu){

  size_t min = std::min(M, N);
  if(ongpu){
    cudaMemset(elem, 0, M * N * sizeof(double));
    size_t blockNum = (min + UNI10_THREADMAX - 1) / UNI10_THREADMAX;
    dim3 gridSize(blockNum % UNI10_BLOCKMAX, (blockNum + UNI10_BLOCKMAX - 1) / UNI10_BLOCKMAX);
    _Diagidentity<<<gridSize, UNI10_THREADMAX>>>(elem, min);
  }else{
    memset(elem, 0, M * sizeof(double));
    for(size_t i = 0; i < min; i++)
      elem[i] = 1;
  }

}

void reseapeElem(double* elem, size_t* transOffset){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

bool lanczosEV(double* A, double* psi, size_t dim, size_t& max_iter, double err_tol, double& eigVal, double* eigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixQR(double* Mij_ori, int M, int N, double* Q, double* R, bool ongpu){

  if(ongpu){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }
  else{
    assert(M >= N);
    double* Mij = (double*)malloc(N*M*sizeof(double));
    memcpy(Mij, Mij_ori, N*M*sizeof(double));
    double* tau = (double*)malloc(M*sizeof(double));
    int lda = N;
    int lwork = -1;
    double worktestdge;
    double worktestdor;
    int info;
    int K = N;
    dgelqf(&N, &M, Mij, &lda, tau, &worktestdge, &lwork, &info);
    dorglq(&N, &M, &K, Mij, &lda, tau, &worktestdor, &lwork, &info);
    lwork = (int)worktestdge;
    double* workdge = (double*)malloc(lwork*sizeof(double));
    dgelqf(&N, &M, Mij, &lda, tau, workdge, &lwork, &info);
    //getQ
    lwork = (int)worktestdor;
    double* workdor = (double*)malloc(lwork*sizeof(double));
    dorglq(&N, &M, &K, Mij, &lda, tau, workdor, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(double));
    //getR
    double alpha = 1, beta = 0;
    dgemm((char*)"N", (char*)"T", &N, &N, &M, &alpha, Mij_ori, &N, Mij, &N, &beta, R, &N);

    free(Mij);
    free(tau);
    free(workdge);
    free(workdor);
  }

}

void matrixRQ(double* Mij_ori, int M, int N, double* Q, double* R, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixQL(double* Mij_ori, int M, int N, double* Q, double* L, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixLQ(double* Mij_ori, int M, int N, double* Q, double* L, bool ongpu){

  assert(N >= M);
  if(ongpu){

    cusolverDnHandle_t cusolverHandle = NULL;
    checkCusolverError(cusolverDnCreate(&cusolverHandle));
    cublasHandle_t cublasHandle = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    // elem copy
    size_t memsize = M * N * sizeof(double);
    double* Mij = NULL;
    checkCudaError(cudaMalloc(&Mij, memsize));
    checkCudaError(cudaMemcpy(Mij, Mij_ori, memsize, cudaMemcpyDeviceToDevice));
    // cuda info
    int* info = NULL;
    checkCudaError(cudaMalloc(&info, sizeof(int)));
    checkCudaError(cudaMemset(info, 0, sizeof(int)));
    // cuda workdge
    int lwork = 0;
    double* workdge = NULL;
    double* tau = NULL;
    int lda = N;
    int ldc = N;
    //int K = M;
    checkCusolverError(cusolverDnDgeqrf_bufferSize(cusolverHandle, N, M, (double*)Mij, lda, &lwork));

    checkCudaError(cudaMalloc(&workdge, sizeof(double)*lwork));
    checkCudaError(cudaMalloc((void**)&tau, sizeof(double)*M));

    checkCusolverError(cusolverDnDgeqrf(cusolverHandle, N, M, Mij, lda, tau, workdge, lwork, info));
    // error info to host	
    int h_info = 0;
    checkCudaError(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    double* elemI;
    checkCudaError(cudaMalloc(&elemI, N*M*sizeof(double)));
    setIdentity(elemI, M, N, true);

    checkCusolverError(cusolverDnDormqr(cusolverHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, N, M, M, Mij, lda, tau, elemI, ldc, workdge, lwork, info));

    checkCudaError(cudaMemcpy(Q, elemI, memsize, cudaMemcpyDeviceToDevice));

    checkCudaError(cudaDeviceSynchronize());

    double alpha = 1, beta = 0;
    checkCublasError(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, M, M, N, &alpha, elemI, N, Mij_ori, N, &beta, L, M));
    
    cudaFree(elemI);
    cudaFree(Mij);
    cudaFree(tau);
    cudaFree(workdge);
    cudaFree(info);

  }
  else{

    double* Mij = (double*)malloc(M*N*sizeof(double));
    memcpy(Mij, Mij_ori, M*N*sizeof(double));
    double* tau = (double*)malloc(M*sizeof(double));
    int lda = N;
    int lwork = -1;
    double worktestdge;
    double worktestdor;
    int info;
    int K = M;
    dgeqrf(&N, &M, Mij, &lda, tau, &worktestdge, &lwork, &info);
    dorgqr(&N, &M, &K, Mij, &lda, tau, &worktestdor, &lwork, &info);
    lwork = (int)worktestdge;
    double* workdge = (double*)malloc(lwork*sizeof(double));
    dgeqrf(&N, &M, Mij, &lda, tau, workdge, &lwork, &info);
    //getQ
    lwork = (int)worktestdor;
    double* workdor = (double*)malloc(lwork*sizeof(double));
    dorgqr(&N, &M, &K, Mij, &lda, tau, workdor, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(double));
    //getR
    double alpha = 1, beta = 0;
    dgemm((char*)"T", (char*)"N", &M, &M, &N, &alpha, Mij, &N, Mij_ori, &N, &beta, L, &M);

    free(Mij);
    free(tau);
    free(workdge);
    free(workdor);

  }
}

/***** Complex version *****/

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double *S, std::complex<double>* vT, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    std::complex<double>* Mij = (std::complex<double>*)malloc(M * N * sizeof(std::complex<double>));
    memcpy(Mij, Mij_ori, M * N * sizeof(std::complex<double>));
    int min = std::min(M, N);
    int ldA = N, ldu = N, ldvT = min;
    int lwork = -1;
    std::complex<double> worktest;
    int info;
    double *rwork = (double*) malloc(std::max(1, 5 * min) * sizeof(double));
    zgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)(worktest.real());
    std::complex<double> *work = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'zgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(rwork);
    free(work);
    free(Mij);


  }

}

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, std::complex<double>* S_ori, std::complex<double>* vT, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    
    int min = std::min(M, N);
    double* S = (double*)malloc(min * sizeof(double));
    matrixSVD(Mij_ori, M, N, U, S, vT, ongpu);
    elemCast(S_ori, S, min, false, false);
    free(S);

  }

}

void matrixInv(std::complex<double>* A, int N, bool diag, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    if(diag){
      for(int i = 0; i < N; i++)
	A[i] = std::abs(A[i]) == 0 ? 0.0 : 1.0/A[i];
      return;
    }
    int *ipiv = (int*)malloc((N+1) * sizeof(int));
    int info;
    zgetrf(&N, &N, A, &N, ipiv, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetrf': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    int lwork = -1;
    std::complex<double> worktest;
    zgetri(&N, A, &N, ipiv, &worktest, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)(worktest.real());
    std::complex<double> *work = (std::complex<double>*)malloc(lwork * sizeof(std::complex<double>));
    zgetri(&N, A, &N, ipiv, work, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(ipiv);
    free(work);

  }

}

std::complex<double> vectorSum(std::complex<double>* X, size_t N, int inc, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    std::complex<double> sum = 0.0;
    size_t idx = 0;
    for(size_t i = 0; i < N; i++){
      sum += X[idx];
      idx += inc;
    }
    return sum;
  }

}
double vectorNorm(std::complex<double>* X, size_t N, int inc, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    double norm2 = 0;
    double tmp = 0;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      tmp = dznrm2(&chunk, X + offset, &inc);
      norm2 += tmp * tmp;
      offset += chunk;
      left -= INT_MAX;
    }
    return sqrt(norm2);
  }

}
void matrixMul(std::complex<double>* A, std::complex<double>* B, int M, int N, int K, std::complex<double>* C, bool ongpuA, bool ongpuB, bool ongpuC){

  std::complex<double> alpha = 1.0, beta = 0.0;
  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
      zgemm((char*)"N", (char*)"N", &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
  }

}

void vectorAdd(std::complex<double>* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < N; i++)
      Y[i] += X[i];
  
  }

}// Y = Y + X

void vectorAdd(std::complex<double>* Y, std::complex<double>* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::complex<double> a = 1.0;
  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    int inc = 1;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      zaxpy(&chunk, &a, X + offset, &inc, Y + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }

  }

}// Y = Y + X
void vectorScal(double a, std::complex<double>* X, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    int inc = 1;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      zdscal(&chunk, &a, X + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }

  }
	
}// X = a * X
void vectorScal(const std::complex<double>& a, std::complex<double>* X, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    int inc = 1;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
	chunk = INT_MAX;
      else
	chunk = left;
      zscal(&chunk, &a, X + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }
  }

}// X = a * X

void vectorMul(std::complex<double>* Y, std::complex<double>* X, size_t N, bool y_ongpu, bool x_ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < N; i++)
      Y[i] *= X[i];

  }

} // Y = Y * X, element-wise multiplication;
void diagRowMul(std::complex<double>* mat, std::complex<double>* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < M; i++)
      vectorScal(diag[i], &(mat[i * N]), N, false);

  }

}

void diagColMul(std::complex<double>* mat, std::complex<double>* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < M; i++){
      size_t ridx = i * N;
      for(size_t j = 0; j < N; j++)
	mat[ridx + j] *= diag[j];
    }

  }

}

void vectorExp(double a, std::complex<double>* X, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < N; i++)
      X[i] = std::exp(a * X[i]);

  }

}

void vectorExp(const std::complex<double>& a, std::complex<double>* X, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < N; i++)
      X[i] = std::exp(a * X[i]);

  }

}

void orthoRandomize(std::complex<double>* elem, int M, int N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    int eleNum = M*N;
    std::complex<double> *random = (std::complex<double>*)malloc(eleNum * sizeof(std::complex<double>));
    elemRand(random, M * N, false);
    int min = M < N ? M : N;
    double *S = (double*)malloc(min*sizeof(double));
    if(M <= N){
      std::complex<double> *U = (std::complex<double>*)malloc(M * min * sizeof(std::complex<double>));
      matrixSVD(random, M, N, U, S, elem, false);
      free(U);
    }
    else{
      std::complex<double> *VT = (std::complex<double>*)malloc(min * N * sizeof(std::complex<double>));
      matrixSVD(random, M, N, elem, S, VT, false);
      free(VT);
    }
    free(random);
    free(S);

  }

}

void setTranspose(std::complex<double>* A, size_t M, size_t N, std::complex<double>* AT, bool ongpu, bool ongpuT){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < M; i++)
      for(size_t j = 0; j < N; j++)
	AT[j * M + i] = A[i * N + j];

  }

}

void setTranspose(std::complex<double>* A, size_t M, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    size_t memsize = M * N * sizeof(std::complex<double>);
    std::complex<double> *AT = (std::complex<double>*)malloc(memsize);
    setTranspose(A, M, N, AT, ongpu, ongpu);
    memcpy(A, AT, memsize);
    free(AT);

  }

}

void setCTranspose(std::complex<double>* A, size_t M, size_t N, std::complex<double>* AT, bool ongpu, bool ongpuT){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < M; i++)
      for(size_t j = 0; j < N; j++)
	AT[j * M + i] = std::conj(A[i * N + j]);
  }

}

void setCTranspose(std::complex<double>* A, size_t M, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    size_t memsize = M * N * sizeof(std::complex<double>);
    std::complex<double> *AT = (std::complex<double>*)malloc(memsize);
    setCTranspose(A, M, N, AT, ongpu, ongpu);
    memcpy(A, AT, memsize);
    free(AT);

  }

}
void eigDecompose(std::complex<double>* Kij, int N, std::complex<double>* Eig, std::complex<double> *EigVec, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    size_t memsize = N * N * sizeof(std::complex<double>);
    std::complex<double> *A = (std::complex<double>*) malloc(memsize);
    memcpy(A, Kij, memsize);
    int ldA = N;
    int ldvl = 1;
    int ldvr = N;
    int lwork = -1;
    double *rwork = (double*) malloc(2 * N * sizeof(double));
    std::complex<double> worktest;
    int info;
    zgeev((char*)"N", (char*)"V", &N, A, &ldA, Eig, NULL, &ldvl, EigVec, &ldvr, &worktest, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'zgeev': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)worktest.real();
    std::complex<double>* work = (std::complex<double>*)malloc(sizeof(std::complex<double>)*lwork);
    zgeev((char*)"N", (char*)"V", &N, A, &ldA, Eig, NULL, &ldvl, EigVec, &ldvr, work, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'zgeev': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(work);
    free(rwork);
    free(A);

  }

}

void eigSyDecompose(std::complex<double>* Kij, int N, double* Eig, std::complex<double>* EigVec, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    //eigDecompose(Kij, N, Eig, EigVec, ongpu);
    memcpy(EigVec, Kij, N * N * sizeof(std::complex<double>));
    int ldA = N;
    int lwork = -1;
    std::complex<double> worktest;
    double* rwork = (double*) malloc((3*N+1) * sizeof(double));
    int info;
    zheev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, &worktest, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'zheev': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)worktest.real();
    std::complex<double>* work= (std::complex<double>*)malloc(sizeof(std::complex<double>)*lwork);
    zheev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, work, &lwork, rwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'zheev': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(work);
    free(rwork);

  }

}
void setConjugate(std::complex<double> *A, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    for(size_t i = 0; i < N; i++)
      A[i] = std::conj(A[i]);

  }

}

void setIdentity(std::complex<double>* elem, size_t M, size_t N, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    size_t min;
    if(M < N)	min = M;
    else		min = N;
    memset(elem, 0, M * N * sizeof(std::complex<double>));
    for(size_t i = 0; i < min; i++)
      elem[i * N + i] = 1.0;
  }

}

bool lanczosEV(std::complex<double>* A, std::complex<double>* psi, size_t dim, size_t& max_iter, double err_tol, double& eigVal, std::complex<double>* eigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

bool lanczosEVL(std::complex<double>* A, std::complex<double>* psi, size_t dim, size_t& max_iter, double err_tol, double& eigVal, std::complex<double>* eigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixQR(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* R, bool ongpu){
  
  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    std::complex<double>* Mij = (std::complex<double>*)malloc(N*M*sizeof(std::complex<double>));
    memcpy(Mij, Mij_ori, N*M*sizeof(std::complex<double>));
    std::complex<double>* tau = (std::complex<double>*)malloc(M*sizeof(std::complex<double>));
    int lda = N;
    int lwork = -1;
    std::complex<double> worktestzge;
    std::complex<double> worktestzun;
    int info;
    int K = N;
    zgelqf(&N, &M, Mij, &lda, tau, &worktestzge, &lwork, &info);
    zunglq(&N, &M, &K, Mij, &lda, tau, &worktestzun, &lwork, &info);
    lwork = (int)worktestzge.real();
    std::complex<double>* workzge = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zgelqf(&N, &M, Mij, &lda, tau, workzge, &lwork, &info);
    //getQ
    lwork = (int)worktestzun.real();
    std::complex<double>* workzun = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zunglq(&N, &M, &K, Mij, &lda, tau, workzun, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(std::complex<double>));
    //getR
    std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
    zgemm((char*)"N", (char*)"C", &N, &N, &M, &alpha, Mij_ori, &N, Mij, &N, &beta, R, &N);

    free(Mij);
    free(tau);
    free(workzge);
    free(workzun);

  }

}
void matrixRQ(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* R, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    std::complex<double>* Mij = (std::complex<double>*)malloc(M*N*sizeof(std::complex<double>));
    memcpy(Mij, Mij_ori, M*N*sizeof(std::complex<double>));
    std::complex<double>* tau = (std::complex<double>*)malloc(M*sizeof(std::complex<double>));
    int lda = N;
    int lwork = -1;
    std::complex<double> worktestzge;
    std::complex<double> worktestzun;
    int info;
    int K = M;
    zgeqlf(&N, &M, Mij, &lda, tau, &worktestzge, &lwork, &info);
    zungql(&N, &M, &K, Mij, &lda, tau, &worktestzun, &lwork, &info);
    lwork = (int)worktestzge.real();
    std::complex<double>* workzge = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zgeqlf(&N, &M, Mij, &lda, tau, workzge, &lwork, &info);
    //getQ
    lwork = (int)worktestzun.real();
    std::complex<double>* workzun = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zungql(&N, &M, &K, Mij, &lda, tau, workzun, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(std::complex<double>));
    //getR
    std::complex<double> alpha (1.0, 0.0), beta (0.0, 0.0);
    zgemm((char*)"C", (char*)"N", &M, &M, &N, &alpha, Mij, &N, Mij_ori, &N, &beta, R, &M);

    free(Mij);
    free(tau);
    free(workzge);
    free(workzun);

  }

}

void matrixQL(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* L, bool ongpu){

  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{

    assert(M >= N);
    std::complex<double>* Mij = (std::complex<double>*)malloc(N*M*sizeof(std::complex<double>));
    memcpy(Mij, Mij_ori, N*M*sizeof(std::complex<double>));
    std::complex<double>* tau = (std::complex<double>*)malloc(M*sizeof(std::complex<double>));
    int lda = N;
    int lwork = -1;
    std::complex<double> worktestzge;
    std::complex<double> worktestzun;
    int info;
    int K = N;
    zgerqf(&N, &M, Mij, &lda, tau, &worktestzge, &lwork, &info);
    zungrq(&N, &M, &K, Mij, &lda, tau, &worktestzun, &lwork, &info);
    lwork = (int)worktestzge.real();
    std::complex<double>* workzge = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zgerqf(&N, &M, Mij, &lda, tau, workzge, &lwork, &info);
    //getQ
    lwork = (int)worktestzun.real();
    std::complex<double>* workzun = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zungrq(&N, &M, &K, Mij, &lda, tau, workzun, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(std::complex<double>));
    //getR
    std::complex<double> alpha (1.0, 0.0), beta (1.0, 1.0);
    zgemm((char*)"N", (char*)"C", &N, &N, &M, &alpha, Mij_ori, &N, Mij, &N, &beta, L, &N);

    free(Mij);
    free(tau);
    free(workzge);
    free(workzun);

  }

}

void matrixLQ(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* L, bool ongpu){
	
  if(GMODE == 2){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else if(GMODE == 1){
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }else{
    std::complex<double>* Mij = (std::complex<double>*)malloc(M*N*sizeof(std::complex<double>));
    memcpy(Mij, Mij_ori, M*N*sizeof(std::complex<double>));
    std::complex<double>* tau = (std::complex<double>*)malloc(M*sizeof(std::complex<double>));
    int lda = N;
    int lwork = -1;
    std::complex<double> worktestzge;
    std::complex<double> worktestzun;
    int info;
    int K = M;
    zgeqrf(&N, &M, Mij, &lda, tau, &worktestzge, &lwork, &info);
    zungqr(&N, &M, &K, Mij, &lda, tau, &worktestzun, &lwork, &info);
    lwork = (int)worktestzge.real();
    std::complex<double>* workzge = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zgeqrf(&N, &M, Mij, &lda, tau, workzge, &lwork, &info);
    //getQ
    lwork = (int)worktestzun.real();
    std::complex<double>* workzun = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
    zungqr(&N, &M, &K, Mij, &lda, tau, workzun, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(std::complex<double>));
    //getR
    std::complex<double> alpha (1.0, 0.0), beta (0.0, 0.0);
    zgemm((char*)"C", (char*)"N", &M, &M, &N, &alpha, Mij, &N, Mij_ori, &N, &beta, L, &M);

    free(Mij);
    free(tau);
    free(workzge);
    free(workzun);

  }

}

};	/* namespace uni10 */

// debug
//double* h_Mij_ori = (double*)malloc(M*N*sizeof(double));
//double* h_Mij = (double*)malloc(M*N*sizeof(double));
//cudaMemcpy(h_Mij_ori, Mij_ori, M*N*sizeof(double), cudaMemcpyDeviceToHost);
//cudaMemcpy(h_Mij, Mij, M*N*sizeof(double), cudaMemcpyDeviceToHost);
//for(size_t i = 0; i < M; i++){
//  for(size_t j = 0; j < N; j++){
//    std::cout << h_Mij_ori[i*N + j] << " ";
//  }
//  std::cout << std::endl << std::endl;
//}
//for(size_t i = 0; i < N; i++){
//  for(size_t j = 0; j < M; j++){
//    std::cout << h_Mij[i*M + j] << " ";
//  }
//  std::cout << std::endl;
//}
//free(h_Mij_ori);
//free(h_Mij);
//------------------------------- 
