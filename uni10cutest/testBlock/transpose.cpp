#include <stdio.h>
#include "uni10.hpp"

using namespace std;
using namespace uni10;

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

int main(){
	
  //double elem[] = {0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, -1.0, -1.0, 1.5, -1.0 , -1.0};
 // cusolverDnHandle_t handle = NULL;	
 // cublasHandle_t cublasHandle = NULL;
 // cudaStream_t stream = NULL;

 // checkCudaErrors(cusolverDnCreate(&handle));

  double elem[] = {11, 4, -32, 12, 6, -4, -51, 167, 24};//, 4, -68, -41};
  Matrix A(3, 3, false);
  A.setElem(elem);
  //vector<Matrix> LQA = A.lq();
  //cout << LQA[0] << endl;
  //cout << LQA[1] << endl;
  //cout << A << endl;
  //cout << LQA[0]*LQA[1] << endl;
  cout << A << endl;

  vector<Matrix> SVDA = A.svd();

  cout << SVDA[0] << endl;
  cout << SVDA[1] << endl;
  cout << SVDA[2] << endl;
  cout << A << endl;
  //cout << SVDA[0]*LQA[1] << endl;

  cout << "======= GPU ======" << endl;

  Matrix B(3, 3, false, true);	
  B.setElem(elem);
  vector<Matrix> SVDB = B.svd();
  //cout << LQB[0] << endl;
  //cout << LQB[1] << endl;
  //cout << LQB[0] * LQB[1] << endl;
  cout << SVDB[0] << endl;
  cout << SVDB[1] << endl;
  cout << SVDB[2] << endl;
  cout << B << endl;
  exit(0);
  //A.max(true);

  return 0;

}

//  size_t free_byte;
//  size_t total_byte;
//
//  cudaError_t cuda_status;
//  cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
//
//  if(cudaSuccess != cuda_status){
//    printf("Error:cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status)); 
//    exit(0);
//  }
//  
//  size_t used_byte = total_byte - free_byte;
//
//  printf("GPU memory usage: used = %ld bytes, free = %ld, bytes, total = %ld bytes\n", used_byte, free_byte, total_byte);
//
