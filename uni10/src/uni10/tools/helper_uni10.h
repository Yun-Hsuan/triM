#ifndef HELPER_UNI10
#define HELPER_UNI10

#include <uni10/data-structure/uni10_struct.h>

#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#define checkUni10TypeError(ans){uni10TypeAssert((ans),__FILE__,__LINE__,__PRETTY_FUNCTION__);}
inline void uni10TypeAssert(uni10::rflag tp, const char *file, int line, const char* func, bool abort=true){
	if(tp != uni10::RTYPE){
		char err[] = "Set a wrong uni10::rflag. Please use RTYPE instead of RNULL.";
		fprintf(stderr, "error: %s\n%s(%d)\nuni10::rflag error: %s\n", func, file, line, err) ;
		if(abort) exit(tp);
	}
}

inline void uni10TypeAssert(uni10::cflag tp, const char *file, int line, const char* func, bool abort=true){
	if(tp != uni10::CTYPE){
		char err[] = "Set a wrong uni10::cflag. Please use CTYPE instead of CNULL.";
		fprintf(stderr, "error: %s\n%s(%d)\nuni10::cflag error: %s\n", func, file, line, err) ;
		if(abort) exit(tp);
	}
}

#ifdef CUDA_SUPPORT

#define checkCudaError(ans){cudaAssert((ans),__FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define checkCublasError(ans) {cublasAssert((ans),__FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define checkCusolverError(ans) {cusolverAssert((ans),__FILE__,__LINE__,__PRETTY_FUNCTION__);}

static const char *cublasGetErrorEnum(cublasStatus_t error){
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS :
      return "Success.";
    case CUBLAS_STATUS_NOT_INITIALIZED :
      return "The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call.";
    case CUBLAS_STATUS_ALLOC_FAILED :
      return "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.";
    case CUBLAS_STATUS_INVALID_VALUE :
      return "An unsupported value or parameter was passed to the function.";
    case CUBLAS_STATUS_ARCH_MISMATCH :
      return "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "An access to GPU memory space failed.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "The functionnality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return " The functionnality requested requires some license and an error was detected when trying to check the current licensing.";
  }

  return "<unknown>";
}

static const char *cusolverGetErrorEnum(cusolverStatus_t error){
  switch (error)
  { 
    case CUSOLVER_STATUS_SUCCESS :
      return "Success.";
    case CUSOLVER_STATUS_NOT_INITIALIZED :
      return "The cuSolver library was not initialized.";
    case CUSOLVER_STATUS_ALLOC_FAILED :
      return "Resource allocation failed inside the cuSolver library.";
    case CUSOLVER_STATUS_INVALID_VALUE :
      return "An unsupported value or parameter was passed to the function";
    case CUSOLVER_STATUS_ARCH_MISMATCH :
      return "The function requires a feature absent from the device architecture";
    case CUSOLVER_STATUS_EXECUTION_FAILED :
      return "The GPU program failed to execute.";
    case CUSOLVER_STATUS_INTERNAL_ERROR :
      return "An internal cuSolver operation failed.";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED :
      return "The matrix type is not supported by this function.";
  }
  return "<unknown>";
}

inline void cudaAssert(cudaError_t flag, const char *file, int line, const char* func, bool abort=true){
	if(flag != cudaSuccess){
		fprintf(stderr, "error: %s\n%s(%d)\ncuda error: %s\n", func, file, line, cudaGetErrorString(flag)) ;
		if(abort) exit(flag);
	}
}

inline void cublasAssert(cublasStatus_t flag, const char *file, int line, const char* func, bool abort=true){
  if(flag != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "error: %s\n%s(%d)\ncudlas error: %s\n", func, file, line, cublasGetErrorEnum(flag)) ;
    if(abort) exit(flag);
  }
}

inline void cusolverAssert(cusolverStatus_t flag, const char *file, int line, const char* func, bool abort=true){
  if(flag != CUSOLVER_STATUS_SUCCESS){
    fprintf(stderr, "error: %s\n%s(%d)\ncusolver error: %s\n", func, file, line, cusolverGetErrorEnum(flag)) ;
    if(abort) exit(flag);
  }
}

#endif

#endif
