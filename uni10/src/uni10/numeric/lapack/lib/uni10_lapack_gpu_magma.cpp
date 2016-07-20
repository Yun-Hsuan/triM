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

#include <uni10/numeric/lapack/uni10_lapack.h>
#include <uni10/tools/helper_uni10.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <magma.h>
#include <magmablas_v1.h>
#include <magma_lapack.h>
//#include <magma_operators.h>
//#include <device_launch_parameters.h>
//#include <cublas_v2.h>
//#include <magma_v2.h>
//#include <magma_v2.h>
//#include <cuda.h>

namespace uni10{

const size_t GPU_OPERATE_MEM = UNI10_GPU_GLOBAL_MEM / 3;

void matrixMul(double* A, double* B, int M, int N, int K, double* C, bool ongpuA, bool ongpuB, bool ongpuC){

  double alpha = 1, beta = 0;
  if(GMODE == 2){       // Fully GPU.

    magma_int_t ldda, lddb, lddc;
    ldda = magma_roundup(K, blocksize);
    lddb = magma_roundup(N, blocksize);
    lddc = magma_roundup(N, blocksize);
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, N, M, K, alpha, B, lddb, A, ldda, beta, C, lddc);

  }
  else if(GMODE == 0){  // Fully CPU.

    magma_int_t magmaN = N;
    magma_int_t magmaM = M;
    magma_int_t magmaK = K;
    blasf77_dgemm((char*)"N", (char*)"N", &magmaN, &magmaM, &magmaK, &alpha, B, &magmaN, A, &magmaK, &beta, C, &magmaN);

  }
  else if(GMODE == 1){
    ongpuA = true;
    ongpuB = true;
    ongpuC = true;
    std::ostringstream err;
    err<<"GMODE == 1 is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }

}

void vectorAdd(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  double a = 1.0;
  magma_int_t inc = 1;
  if(GMODE == 2){  // Fully GPU.
    magma_init();
    magma_daxpy(N, a, X, inc, Y, inc);
  }
  else if(GMODE == 0){  // Fully CPU.
    int64_t left = N;
    size_t offset = 0;
    magma_int_t chunk;
    while(left > 0){
      if(left > INT_MAX)
        chunk = INT_MAX;
      else
        chunk = left;
      blasf77_daxpy(&chunk, &a, X + offset, &inc, Y + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }
  }
  else if(GMODE == 1){  //CPU version
    y_ongpu = true;
    x_ongpu = true;
    std::ostringstream err;
    err<<"GMODE == 1 is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }

}

void vectorScal(double a, double* X, size_t N, bool ongpu){

  magma_int_t inc = 1;
  if(GMODE == 2){  // Fully GPU.

    magma_dscal(N, a, X, inc);

  }
  else if(GMODE == 0){  // Fully CPU.

    int64_t left = N;
    size_t offset = 0;
    magma_int_t chunk;

    while(left > 0){
      if(left > INT_MAX)
        chunk = INT_MAX;
      else
        chunk = left;
      blasf77_dscal(&chunk, &a, X + offset, &inc);
      offset += chunk;
      left -= INT_MAX;
    }

  }
  else if(GMODE == 1){  //CPU version

    ongpu = true;
    //x_ongpu = true;
    std::ostringstream err;
    err<<"GMODE == 1 is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

}// Y = a * X

void vectorMul(double* Y, double* X, size_t N, bool y_ongpu, bool x_ongpu){

  if(GMODE == 2){

    std::ostringstream err;
    err<<"GPU version (GMODE2) is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
    
  }else if(GMODE == 0){

    for(size_t i = 0; i < N; i++)
      Y[i] *= X[i];

  }else if(GMODE == 1){

    std::ostringstream err;
    err<<"GPU version (GMODE1) is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

}// Y = Y * X, element-wise multiplication;

double vectorSum(double* X, size_t N, int inc, bool ongpu){

  double sum = 0;
  if(GMODE == 2){  // Fully GPU.
    magma_init();
    sum = magma_dasum(N, X, inc);
  }
  else if(GMODE == 0){  // Fully CPU.
    size_t idx = 0;
    for(size_t i = 0; i < N; i++){
      sum += X[idx];
      idx += inc;
    }
  }
  else if(GMODE == 1){  //CPU version
    ongpu = true;
    std::ostringstream err;
    err<<"GMODE == 1 is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }
  return sum;

}

double vectorNorm(double* X, size_t N, int inc, bool ongpu){

  double norm2 = 0;
  double tmp = 0;
  if(GMODE == 2){       // Fully GPU.
    magma_init();
    tmp = magma_dnrm2(N, X, inc);
    norm2 += tmp * tmp;
  }
  else if(GMODE == 0){  // Fully CPU.
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
      if(left > INT_MAX)
        chunk = INT_MAX;
      else
        chunk = left;
      tmp = magma_cblas_dnrm2(chunk, X + offset, inc);
      norm2 += tmp * tmp;
      offset += chunk;
      left -= INT_MAX;
    }
  }
  else if(GMODE == 1){
    ongpu = true;
    std::ostringstream err;
    err<<"GMODE == 1 is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));
  }
  return sqrt(norm2);

}

//__global__ void _vectorExp(double a, double* X, size_t N){
//
//  std::ostringstream err;
//  err<<"GPU version is not ready !!!!";
//  throw std::runtime_error(exception_msg(err.str()));
//
//}

void vectorExp(double a, double* X, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

//__global__ void _diagRowMul(double* mat, double* diag, size_t M, size_t N){
//  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
//  double scalar = diag[idx / N];
//  if(idx < M * N)
//    mat[idx] *= scalar;
//}

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

    magma_int_t info;
    elemCopy(EigVec, Kij, N, N, ongpu, ongpu, false);

    if(ongpu){

      double aux_work[1];
      magma_int_t aux_iwork[1];
      magma_int_t lda = N;
      magma_int_t ldda = magma_roundup(N, blocksize);

      magma_dsyevd_gpu( MagmaVec, MagmaUpper, 
                        N, NULL, ldda, NULL, // A, w
                        NULL, lda,            // host A
                        aux_work, -1,  
                        aux_iwork, -1,
                        &info
                        );         


      double* pin_Sy = NULL, *h_work = NULL;

      double* h_eig = (double*)malloc(N*sizeof(double));
      magma_dmalloc_pinned(&pin_Sy, N * ldda);

      magma_int_t liwork = aux_iwork[0];
      magma_int_t* iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

      magma_int_t lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
      magma_dmalloc_pinned(&h_work, lwork);

      magma_dsyevd_gpu( MagmaVec, MagmaUpper,
                        N, EigVec, ldda, h_eig,
                        pin_Sy, lda,
                        h_work, lwork,
                        iwork, liwork,
                        &info 
                        );

      elemCopy(Eig, h_eig, N, N, ongpu, false, true);


      free(h_eig);
      free(iwork);
      magma_free_pinned(pin_Sy);
      magma_free_pinned(h_work);

      //printf("============askjdhfla\n");
      //magma_dprint_gpu(N, N, EigVec, magma_roundup(N, blocksize));
      //magma_dprint(N, 1, h_eig, N);
      //magma_dprint_gpu(N, 1, Eig, ldda);
      ////magma_dprint_gpu(N, N, Kij, magma_roundup(N, blocksize));
      //printf("============1askjdhfla\n");

    }
    else if(GMODE == 0){

      double worktest;
      double* work = NULL;

      magma_int_t lwork = -1; 
      
      magma_int_t magmaN = N;
      magma_int_t lda = N;

      lapackf77_dsyev((char*)"V", (char*)"U", &magmaN, EigVec, &lda, Eig, &worktest, &lwork, &info);

      lwork = (magma_int_t) MAGMA_D_REAL(worktest);

      magma_dmalloc_cpu(&work, lwork);

      lapackf77_dsyev((char*)"V", (char*)"U", &magmaN, EigVec, &lda, Eig, work, &lwork, &info);

      free(work);

    }
    else if(GMODE == 1){

      std::ostringstream err;
      err<<"GPU version is not ready !!!!";
      throw std::runtime_error(exception_msg(err.str()));

    }

}

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT, bool ongpu){
  
  bool flag = M > N;
  
  if(GMODE == 2){
      
    size_t min = std::min(M, N);
    double* Mij_cu = NULL, *U_cu = NULL, *vT_cu = NULL; 

    checkCudaError( cudaMalloc(&Mij_cu, M   * N   * sizeof(double)));
    checkCudaError( cudaMalloc(&U_cu  , M   * min * sizeof(double)));
    checkCudaError( cudaMalloc(&vT_cu , min * N   * sizeof(double)));

    size_t ldda    = magma_roundup(N, blocksize);
    size_t lddat   = magma_roundup(M, blocksize);
    size_t ldda_cu = flag ? lddat : ldda;

    cusolverDnHandle_t cusolverHandle = NULL;
    checkCusolverError(cusolverDnCreate(&cusolverHandle));
    // elem copy
    size_t memsize = 0;
    double* Mij_buf = NULL;

    if(flag){
      memsize = N * lddat * sizeof(double);
      magma_malloc((void**)&Mij_buf, memsize);
      setTranspose(Mij_ori, M, N, Mij_buf, true, true); 
      int tmp = M;
      M = N;
      N = tmp;
    }else{
      memsize = M * ldda * sizeof(double);
      magma_malloc((void**)&Mij_buf, memsize);
      checkCudaError(cudaMemcpy(Mij_buf, Mij_ori, memsize, cudaMemcpyDeviceToDevice));
    }
    
    magma_to_cuda(M, N, Mij_buf, ldda_cu, Mij_cu);

    double* bufM = NULL;
    checkCudaError( cudaMalloc(&bufM, N*N*sizeof(double)));
    // cuda info
    int* info = NULL;
    checkCudaError(cudaMalloc(&info, sizeof(int)));
    checkCudaError(cudaMemset(info, 0, sizeof(int)));
    // cuda workdge
    int ldA = N, ldu = N, ldvT = min; 
    int lwork = 0;
    double* rwork = NULL;
    double* work = NULL;
    //int K = M;
    checkCusolverError(cusolverDnDgesvd_bufferSize(cusolverHandle, N, M, &lwork));
    checkCudaError(cudaMalloc(&rwork, sizeof(double)*lwork));

    checkCudaError(cudaMalloc(&work, sizeof(double)*lwork));

    cusolverStatus_t cusolverflag = (!flag) ? 
      cusolverDnDgesvd(cusolverHandle, 'A', 'A', 
                        N   , M    , Mij_cu , ldA , 
                        S   , bufM , ldu    , U_cu, ldvT, 
                        work, lwork, rwork  , info
                        ) : 
      cusolverDnDgesvd(cusolverHandle, 'A', 'A', 
                        N   , M    , Mij_cu  , ldA, 
                        S   , bufM , ldu     , vT_cu , ldvT, 
                        work, lwork, rwork   , info
                        );
    checkCusolverError(cusolverflag);


    if(flag){

      //M > N
      checkCudaError(cudaMemcpy(U_cu, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice));

      cuda_to_magma(M, N, U_cu , U , magma_roundup(N, blocksize));
      cuda_to_magma(M, M, vT_cu, vT, magma_roundup(M, blocksize));

      setTranspose(U, M, N, ongpu);
      setTranspose(vT, M, M, ongpu);

    }
    else{

      checkCudaError(cudaMemcpy(vT_cu, bufM, M*N*sizeof(double), cudaMemcpyDeviceToDevice));

      cuda_to_magma(M, M, U_cu , U , magma_roundup(M, blocksize));
      cuda_to_magma(M, N, vT_cu, vT, magma_roundup(N, blocksize));

    }
    
    int h_info = 0;
    checkCudaError(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(Mij_buf);
    cudaFree(U_cu);
    cudaFree(vT_cu);
    cudaFree(work);
    cudaFree(info);
    cudaFree(bufM);

  }else if(GMODE == 0){

    magma_int_t magmaN = N;
    magma_int_t magmaM = M;

    double* Mij = (double*)malloc(M * N * sizeof(double));
    memcpy(Mij, Mij_ori, M * N * sizeof(double));
    magma_int_t min = std::min(M, N);
    magma_int_t ldA = N, ldu = N, ldvT = min;
    magma_int_t lwork = -1;
    double worktest;
    magma_int_t info;
    lapackf77_dgesvd((char*)"S", (char*)"S", &magmaN, &magmaM, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }
    lwork = (magma_int_t)worktest;
    double *work = (double*)malloc(lwork*sizeof(double));
    lapackf77_dgesvd((char*)"S", (char*)"S", &magmaN, &magmaM, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgesvd': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    free(work);
    free(Mij);

  }else if(GMODE == 1){
    
      std::ostringstream err;
      err<<"GPU MODE1 version is not ready !!!!";
      throw std::runtime_error(exception_msg(err.str()));
      return; 

  }

}

//__global__ void _diagMatInv(double* diag, size_t N){
//
//  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
//  if(idx < N)
//    diag[idx] = (diag[idx] < 1E-14) ? 0 : 1. / diag[idx];
//
//}

void matrixInv(double* A, int N, bool diag, bool ongpu){

  if(GMODE == 2){       // Fully GPU.

    if(diag){
      std::ostringstream err;
      err<<"GPU version is not ready !!!!";
      throw std::runtime_error(exception_msg(err.str()));
      return; 
    }

    magma_int_t *ipiv = (magma_int_t*)malloc(N*sizeof(magma_int_t));
    magma_int_t info;
    magma_int_t ldd = magma_roundup(N, blocksize);

    magma_dgetrf_gpu(N, N, A, ldd, ipiv, &info);

    if(info != 0){
      std::ostringstream err;
      err<<"Error in Magma function 'magma_dgetrf': Magma_lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    double *dwork;
    magma_int_t ldwork = N * magma_get_dgetri_nb(N);
    magma_malloc((void**)&dwork, ldwork*sizeof(double));

    magma_dgetri_gpu(N, A, ldd, ipiv, dwork, ldwork, &info);
    if(info != 0){
      std::ostringstream err;
      err<<"Error in Magma function 'magma_dgetri': Magma_lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    free(ipiv);
    magma_free(dwork);

  }
  else if(GMODE == 0){

    magma_int_t magmaN = N;

    if(diag){
      for(int i = 0; i < N; i++)
        A[i] = A[i] == 0 ? 0 : 1.0/A[i];
      return;
    }

    magma_int_t *ipiv = (magma_int_t*)malloc((N+1)*sizeof(magma_int_t));
    magma_int_t info;
    lapackf77_dgetrf(&magmaN, &magmaN, A, &magmaN, ipiv, &info);

    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetrf': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    magma_int_t lwork = -1;
    double worktest;
    lapackf77_dgetri(&magmaN, A, &magmaN, ipiv, &worktest, &lwork, &info);

    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    lwork = (int)worktest;
    double *work = (double*)malloc(lwork * sizeof(double));
    lapackf77_dgetri(&magmaN, A, &magmaN, ipiv, work, &lwork, &info);

    if(info != 0){
      std::ostringstream err;
      err<<"Error in Lapack function 'dgetri': Lapack INFO = "<<info;
      throw std::runtime_error(exception_msg(err.str()));
    }

    free(ipiv);
    free(work);
  } 
  else if(GMODE == 1){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

}

//void _transposeCPU(double* A, size_t M, size_t N, double* AT){
//
//  for(size_t i = 0; i < M; i++)
//    for(size_t j = 0; j < N; j++)
//      AT[j * M + i] = A[i * N + j];
//
//}

//__global__ void _transposeGPU(double* A, size_t M, size_t N, double* AT){
//
//  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
//  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
//  if(y < M && x < N)
//    AT[x * M + y] = A[y * N + x];
//
//}

void setTranspose(double* A, size_t M, size_t N, double* AT, bool ongpu, bool ongpuT){

  if(ongpu){
    
    magma_int_t ldda    =   magma_roundup(N, blocksize);
    magma_int_t lddat   =   magma_roundup(M, blocksize);
    magmablas_dtranspose(N, M, A, ldda, AT, lddat);

  }else{

    for(size_t i = 0; i < M; i++)
      for(size_t j = 0; j < N; j++)
        AT[j * M + i] = A[i * N + j];

  }

}

void setTranspose(double* A, size_t M, size_t N, bool ongpu){

  size_t memsize_AT = ongpu ? magma_roundup(M, blocksize) * N * sizeof(double): N * M * sizeof(double);

  double* AT = NULL;

  if(ongpu){

    magma_malloc((void**)&AT, memsize_AT);

  }else{
    
    AT = (double*)malloc(memsize_AT);

  }

  setTranspose(A, M, N, AT, ongpu, ongpu);

  if(ongpu){
    
    magma_free(A);
    magma_malloc((void**)&A, memsize_AT);

  }

  elemCopy(A, AT, N, M, ongpu, ongpu, false);

  elemFree(AT, memsize_AT, ongpu);
  

}

void setCTranspose(double* A, size_t M, size_t N, double *AT, bool ongpu, bool ongpuT){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

void setCTranspose(double* A, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}

//__global__ void _identity(double* mat, size_t elemNum, size_t col){
//  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
//  if(idx < elemNum)
//    mat[idx * col + idx] = 1;
//}

void setIdentity(double* elem, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

}
// [1, 1 ,.. 1]
//__global__ void _Diagidentity(double* mat, size_t elemNum){
//  size_t idx = blockIdx.y * UNI10_BLOCKMAX * UNI10_THREADMAX +  blockIdx.x * blockDim.x + threadIdx.x;
//  if(idx < elemNum)
//    mat[idx] = 1;
//}

void setDiagIdentity(double* elem, size_t M, size_t N, bool ongpu){

  std::ostringstream err;
  err<<"GPU version is not ready !!!!";
  throw std::runtime_error(exception_msg(err.str()));

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

  assert(M >= N);

  if(GMODE == 2){
    

    double *Mij_t = NULL, *Qt = NULL, *Rt = NULL;

    magma_int_t ldda    =   magma_roundup(N, blocksize);
    magma_int_t lddat   =   magma_roundup(M, blocksize);

    magma_dmalloc(&Mij_t, lddat*N);
    magma_malloc((void**)&Qt, lddat * N * sizeof(double));
    magma_malloc((void**)&Rt, ldda  * N * sizeof(double));
  
    magmablas_dtranspose(N, M, Mij_ori, ldda, Mij_t, lddat);

    //magma_dprint_gpu(N, M, Mij_ori, ldda);
    //magma_dprint_gpu(M, N, Mij_t, lddat);

    //magma_dprint_gpu(M, N, Qt, lddat);
    //magma_dprint_gpu(N, N, Rt, ldda);

    matrixLQ(Mij_t, N, M, Qt, Rt, true);

    //magma_dprint_gpu(M, N, Qt, lddat);
    //magma_dprint_gpu(N, N, Rt, ldda);

    magma_free(Mij_t); 

    magmablas_dtranspose(M, N, Qt, lddat, Q, ldda);
    magmablas_dtranspose(N, N, Rt, lddat, R, lddat);

    magma_free(Qt); 
    magma_free(Rt); 
    //magma_dprint_gpu(N, M, Q, ldda);
    //magma_dprint_gpu(N, N, R, ldda);

  }
  else if(GMODE == 0){

    magma_int_t magmaN = N;
    magma_int_t magmaM = M;
    magma_int_t magmaK = N;
    magma_int_t lda = magmaN;
    magma_int_t info;

    double* Mij = (double*)malloc(M*N*sizeof(double));
    memcpy(Mij, Mij_ori, M*N*sizeof(double));
    double* tau = (double*)malloc(M*sizeof(double));
    magma_int_t lwork = -1;
    double worktestdge;
    double worktestdor;

    lapackf77_dgelqf(&magmaN, &magmaM, Mij, &lda, tau, &worktestdge, &lwork, &info);
    lapackf77_dorglq(&magmaN, &magmaM, &magmaK, Mij, &lda, tau, &worktestdor, &lwork, &info);

    lwork = (int)worktestdge;
    double* workdge = (double*)malloc(lwork*sizeof(double));
    lapackf77_dgelqf(&magmaN, &magmaM, Mij, &lda, tau, workdge, &lwork, &info);

    //getQ
    lwork = (int)worktestdor;
    double* workdor = (double*)malloc(lwork*sizeof(double));
    lapackf77_dorglq(&magmaN, &magmaM, &magmaK, Mij, &lda, tau, workdor, &lwork, &info);
    memcpy(Q, Mij, N*M*sizeof(double));

    //getR
    double alpha = 1, beta = 0;
    blasf77_dgemm( (char*)"N", (char*)"T", &magmaN, &magmaN, &magmaM, 
                   &alpha, Mij_ori, &magmaN, 
                           Mij    , &magmaN, 
                   &beta,  R      , &magmaN
                 );

    free(Mij);
    free(tau);
    free(workdge);
    free(workdor);

  }
  else if(GMODE == 1){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

  if(GMODE == 2){
    
    magma_int_t ldda = magma_roundup(N, blocksize); 
    magma_int_t lddc = magma_roundup(M, blocksize); 
    magma_int_t nb = magma_get_dgeqrf_nb(M, N); 
    magma_int_t info;
    magma_int_t K = M;

    elemCopy(Q, Mij_ori, M, N, ongpu, ongpu, false);

    double *tau, *dT;

    magma_dmalloc_cpu( &tau,  M );

    magma_dmalloc( &dT, (2 * M + lddc) * nb );

    magma_dgeqrf3_gpu(N, M, Q, ldda, tau, dT, &info);

    magma_dorgqr_gpu(N, M, K, Q, ldda, tau, dT, nb, &info);

    //getR
    double alpha = 1, beta = 0;
    magmablas_dgemm( MagmaTrans, MagmaNoTrans, M, M, N, 
                     alpha, Q      , ldda, 
                            Mij_ori, ldda, 
                     beta,  L      , lddc
                    );

    //magma_dprint_gpu(M, M, L, magma_roundup(M, blocksize));

    free(tau);
    magma_free(dT);

  }
  else if(GMODE == 0){

    magma_int_t magmaN = N;
    magma_int_t magmaM = M;
    magma_int_t magmaK = M;
    magma_int_t lda = magmaN;

    double* Mij = (double*)malloc(M*N*sizeof(double));
    memcpy(Mij, Mij_ori, M*N*sizeof(double));
    double* tau = (double*)malloc(M*sizeof(double));
    magma_int_t lwork = -1;
    double worktestdge;
    double worktestdor;
    magma_int_t info;

    lapackf77_dgeqrf(&magmaN, &magmaM, Mij, &lda, tau, &worktestdge, &lwork, &info);
    lapackf77_dorgqr(&magmaN, &magmaM, &magmaK, Mij, &lda, tau, &worktestdor, &lwork, &info);

    lwork = (int)worktestdge;
    double* workdge = (double*)malloc(lwork*sizeof(double));
    lapackf77_dgeqrf(&magmaN, &magmaM, Mij, &lda, tau, workdge, &lwork, &info);
    //getQ
    lwork = (int)worktestdor;
    double* workdor = (double*)malloc(lwork*sizeof(double));
    lapackf77_dorgqr(&magmaN, &magmaM, &magmaK, Mij, &lda, tau, workdor, &lwork, &info);

    memcpy(Q, Mij, N*M*sizeof(double));

    //getR
    double alpha = 1, beta = 0;
    blasf77_dgemm((char*)"T", (char*)"N", &magmaM, &magmaM, &magmaN, 
                   &alpha, Mij      , &magmaN, 
                           Mij_ori  , &magmaN, 
                   &beta,  L        , &magmaM);

    //free(Mij);
    free(tau);
    free(workdge);
    free(workdor);

  }
  else if(GMODE == 1){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

}

/***** Complex version *****/

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double *S, std::complex<double>* vT, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

}

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, std::complex<double>* S_ori, std::complex<double>* vT, bool ongpu){

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

  if(GMODE == 2){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }else if(GMODE == 1){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }else{

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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
    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

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

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

}

};	/* namespace uni10 */
