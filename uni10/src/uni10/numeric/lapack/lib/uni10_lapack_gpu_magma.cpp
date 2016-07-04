/****************************************************************************
*  @file uni10_lapack_gpu.cpp
*  @license
*    Universal Tensor Network Library
*    Copyright (c) 2013-2014
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
*  @author Yun-Da Hsieh
*  @date 2014-05-06
*  @since 0.1.0
*
*****************************************************************************/
#ifdef MKL
  #include "mkl.h"
#else
  //#include <uni10/numeric/lapack/uni10_lapack_wrapper.h>
#endif

#include <string.h>
#include <uni10/numeric/lapack/uni10_lapack.h>
#include <uni10/tools/uni10_tools.h>

//#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <cublas_v2.h>
#include <cusolverDn.h>
#include <magma_v2.h>
#include <magma_lapack.h>

namespace uni10{

const size_t GPU_OPERATE_MEM = UNI10_GPU_GLOBAL_MEM / 3;

bool IN_MEM(size_t memsize);

bool IN_MEM(size_t memsize){
  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));


}

void matrixMul(double* A, double* B, int M, int N, int K, double* C, bool ongpuA, bool ongpuB, bool ongpuC){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void vectorAdd(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// Y = Y + X

void vectorScal(double a, double* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// X = a * X

void vectorMul(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// Y = Y * X, element-wise multiplication;
double vectorSum(double* X, size_t N, int inc, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

double vectorNorm(double* X, size_t N, int inc, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void vectorExp(double a, double* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void diagRowMul(double* mat, double* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){
	
  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

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

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void eigDecompose(double* Kij, int N, std::complex<double>* Eig, std::complex<double> *EigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void eigSyDecompose(double* Kij, int N, double* Eig, double* EigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT, bool ongpu){

  //Mij = U * S * VT
  magma_int_t min = M < N ? M : N;	//min = min(M,N)
  magma_int_t ldA = N, ldu = N, ldvT = min;
  magma_vec_t jobu = MagmaSomeVec, jobvt = MagmaSomeVec;
  // Magma haven't supply on gpu version
  if(ongpu){
    bool flag = M > N;
    cusolverDnHandle_t cusolverHandle = NULL;
    cusolverDnCreate(&cusolverHandle);
    // elem copy
    size_t memsize = M * N * sizeof(double);
    double* Mij = NULL;
    cudaError_t cuflag = cudaMalloc(&Mij, memsize);
    if(flag){
      setTranspose(Mij_ori, M, N, Mij, ongpu, true); 
      int tmp = M;
      M = N;
      N = tmp;
    }else{
      cuflag = cudaMemcpy(Mij, Mij_ori, memsize, cudaMemcpyDeviceToDevice);
      assert(cuflag == cudaSuccess);
    }
    double* bufM = NULL;
    cuflag = cudaMalloc(&bufM, N*N*sizeof(double));
    assert(cuflag == cudaSuccess);
    // cuda info
    int* info = NULL;
    cuflag = cudaMalloc(&info, sizeof(int));
    assert(cuflag == cudaSuccess);
    cuflag = cudaMemset(info, 0, sizeof(int));
    assert(cuflag == cudaSuccess);
    // cuda workdge
    int min = std::min(M, N);
    int ldA = N, ldu = N, ldvT = min; 
    int lwork = 0;
    double* rwork = NULL;
    double* work = NULL;
    //int K = M;
    cusolverStatus_t cusolverflag = cusolverDnDgesvd_bufferSize(cusolverHandle, N, M, &lwork);
    assert(cusolverflag == CUSOLVER_STATUS_SUCCESS);
    cuflag = cudaMalloc(&rwork, sizeof(double)*lwork);
    assert(cuflag == cudaSuccess);

    cuflag = cudaMalloc(&work, sizeof(double)*lwork);
    assert(cuflag == cudaSuccess);

    cusolverflag = !flag ? cusolverDnDgesvd(cusolverHandle, 'A', 'A', N, M, Mij, ldA, S, bufM, ldu, U, ldvT, work, lwork, rwork, info) : cusolverDnDgesvd(cusolverHandle, 'A', 'A', N, M, Mij, ldA, S, bufM, ldu, vT, ldvT, work, lwork, rwork, info);

    if(!flag){
    cuflag = cudaMemcpy(vT, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice);
    }else{
    cuflag = cudaMemcpy(U, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice);
    setTranspose(U, M, N, ongpu);
    setTranspose(vT, M, M, ongpu);
    }
    int h_info = 0;
    cuflag = cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cusolverflag == CUSOLVER_STATUS_SUCCESS);
    
    cudaFree(Mij);
    cudaFree(bufM);
    cudaFree(work);
    cudaFree(info);
  }
  else{
    double* Mij = (double*)malloc(M * N * sizeof(double));
    memcpy(Mij, Mij_ori, M * N * sizeof(double));
    int lwork = -1;
    double worktest;
    int info;
    lapackf77_dgesvd(lapack_vec_const(jobu), lapack_vec_const(jobvt), &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (int)worktest;
    double *work = (double*)malloc(lwork*sizeof(double));
    lapackf77_dgesvd(lapack_vec_const(jobu), lapack_vec_const(jobvt), &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    free(work);
    free(Mij);
  }

}

void matrixInv(double* A, int N, bool diag, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void _transposeCPU(double* A, size_t M, size_t N, double* AT){

  for(size_t i = 0; i < M; i++)
    for(size_t j = 0; j < N; j++)
      AT[j * M + i] = A[i * N + j];

}

__global__ void _transposeGPU(double* A, size_t M, size_t N, double* AT){

}

void setTranspose(double* A, size_t M, size_t N, double* AT, bool ongpu, bool ongpuT){

}

void setTranspose(double* A, size_t M, size_t N, bool ongpu){

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

}

void setIdentity(double* elem, size_t M, size_t N, bool ongpu){


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

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

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


}

/***** Complex version *****/

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double *S, std::complex<double>* vT, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, std::complex<double>* S, std::complex<double>* vT, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void matrixInv(std::complex<double>* A, int N, bool diag, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

std::complex<double> vectorSum(std::complex<double>* X, size_t N, int inc, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
double vectorNorm(std::complex<double>* X, size_t N, int inc, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void matrixMul(std::complex<double>* A, std::complex<double>* B, int M, int N, int K, std::complex<double>* C, bool ongpuA, bool ongpuB, bool ongpuC){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void vectorAdd(std::complex<double>* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// Y = Y + X
void vectorAdd(std::complex<double>* Y, std::complex<double>* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// Y = Y + X
void vectorScal(double a, std::complex<double>* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));
	
}// X = a * X
void vectorScal(const std::complex<double>& a, std::complex<double>* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}// X = a * X

void vectorMul(std::complex<double>* Y, std::complex<double>* X, size_t N, bool y_ongpu, bool x_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

} // Y = Y * X, element-wise multiplication;
void diagRowMul(std::complex<double>* mat, std::complex<double>* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void diagColMul(std::complex<double>* mat, std::complex<double>* diag, size_t M, size_t N, bool mat_ongpu, bool diag_ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void vectorExp(double a, std::complex<double>* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));
	
}

void vectorExp(const std::complex<double>& a, std::complex<double>* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void orthoRandomize(std::complex<double>* elem, int M, int N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void setTranspose(std::complex<double>* A, size_t M, size_t N, std::complex<double>* AT, bool ongpu, bool ongpuT){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void setTranspose(std::complex<double>* A, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void setCTranspose(std::complex<double>* A, size_t M, size_t N, std::complex<double>* AT, bool ongpu, bool ongpuT){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void setCTranspose(std::complex<double>* A, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void eigDecompose(std::complex<double>* Kij, int N, std::complex<double>* Eig, std::complex<double> *EigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void eigSyDecompose(std::complex<double>* Kij, int N, double* Eig, std::complex<double>* EigVec, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void setConjugate(std::complex<double> *A, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void setIdentity(std::complex<double>* elem, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

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

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
void matrixRQ(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* R, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixQL(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* L, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void matrixLQ(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* Q, std::complex<double>* L, bool ongpu){
	
  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

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
