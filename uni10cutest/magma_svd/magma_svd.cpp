#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <magma.h>
#include <magma_v2.h>
#include <magma_lapack.h>
#include <magma_types.h>

//#include "dgesvd_gpu.cpp"

void printMatrix(const double* v, int m, int n, int ldda, magma_queue_t queues);
void cudaPrintMatrix(const double* v, int m, int n, int ldda, magma_queue_t queues);

void magma_to_cuda(int m, int n, double* mag, int ldda, double* cu);
void cuda_to_magma(int m, int n, double* cu, double* mag, int ldda);

void magma_to_cuda(int m, int n, double* mag, int ldda, double* cu){

    for(int i = 0; i < m; i++)
        cudaMemcpy(cu+(i*n), mag+(i*ldda),n*sizeof(double), cudaMemcpyDeviceToDevice);
        
}

void cuda_to_magma(int m, int n, double* cu, double* mag, int ldda){

    for(int i = 0; i < m; i++)
        cudaMemcpy(mag+(i*ldda), cu+(i*n), n*sizeof(double), cudaMemcpyDeviceToDevice);

}

void printMatrix(const double* v, int m, int n, int ldda, magma_queue_t queues){

    printf("[\n");
    double* bufv = (double*)malloc(m*n*sizeof(double));
    if(magma_is_devptr(v) == 1){
        //cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
        printf("-- On GPU --\n");
        magma_dgetmatrix(n, m, v, ldda, bufv, n, queues);
    }else if(magma_is_devptr(v) == 0){
        printf("-- On CPU --\n");
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf(" %8.4f  ", bufv[i*n+j]);
        printf("\n");
    }
    printf("];\n");
}


void cudaPrintMatrix(const double* v, int m, int n){

    printf("[\n");
    double* bufv = (double*)malloc(m*n*sizeof(double));
    if(magma_is_devptr(v) == 1){
        printf("-- On GPU --\n");
        cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    }else if(magma_is_devptr(v) == 0){
        printf("-- On CPU --\n");
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf(" %8.4f  ", bufv[i*n+j]);
        printf("\n");
    }
    printf("];\n");
}

int main( int argc, char** argv )
{
    magma_init();
    magma_print_environment();
    
    magma_int_t err;
    magma_int_t num = 0;

    magma_device_t dev;

    magma_queue_t queues;
    magma_queue_create( 0, &queues );

    const double c_zero     = MAGMA_D_ZERO;
    const double c_one      = MAGMA_D_ONE;
    const double c_neg_one  = MAGMA_D_NEG_ONE;
    
    double dummy[1];
    magma_int_t M, N, MN,lda, ldb, ldc, ldda, info;
    double *h_A, *h_S, *h_U, *h_VT;
    double *d_Acu, *d_test;
    magmaDouble_ptr d_A, d_U, d_S, d_VT;
    magma_int_t ione  = 1;
    magma_int_t ISEED[4] = {0, 0, 0, 1};
    double tmp;
    double error, rwork[1];
    magma_int_t status = 0;

    M = 3;
    N = 4;
    MN = M*N;
    
    ldda = magma_roundup(N, 32);
    lda = N;

    h_A = (double*)malloc(M*N*sizeof(double));
    cudaMalloc((void**)& d_Acu, M*N*sizeof(double));
    cudaMemset(d_Acu, 0, M*N*sizeof(double));

    cudaPrintMatrix(d_Acu, M, N);

    magma_malloc((void**)&d_A, M*ldda*sizeof(double));
    magma_malloc((void**)&d_test, M*ldda*sizeof(double));
    cudaMemset(d_test, 0, M*ldda*sizeof(double));


    //magma_malloc((void**)&d_A, M*ldda*sizeof(double));
    //magma_malloc((void**)&d_S, N*ldda*sizeof(double));
    //magma_malloc((void**)&d_U, lddbm*lddbn*sizeof(double));
    //magma_malloc((void**)&d_VT, lddcm*lddcn*sizeof(double));

    //printMatrix(d_A, M, K, lddan);
    //exit(0);

    //printf("\n\n ldda: %d, M: %d, N: %d \n\n", (int)lddan, (int)M, (int)N);

    // Initialize the matrix
    lapackf77_dlarnv(&ione, ISEED, &MN, h_A);

    //cudaMemcpy(d_Acu, h_A, M*N*sizeof(double), cudaMemcpyHostToDevice);
    magma_dsetmatrix(N, M, h_A, lda, d_A, ldda, queues);
    
    printMatrix(h_A, M, N, lda, queues);

    printf("========MTOC============\n");
    magma_to_cuda(M, N, d_A, ldda, d_Acu);
    //printMatrix(d_test, M, N, lda, queues);
    printMatrix(d_A, M, N, ldda, queues);
    cudaPrintMatrix(d_Acu, M, N);
    printf("========CTOM============\n");

    cuda_to_magma(M, N, d_Acu, d_test, ldda);

    cudaPrintMatrix(d_Acu, M, N);
    printMatrix(d_test, M, N, ldda, queues);

    //exit(0);
    printf("====================\n");

    //cudaMemcpy(d_A, h_A, n2*sizeof(double), cudaMemcpyHostToDevice);

    //if(M >= N){

    //    printMatrix(h_U, M, N, ldda, queues);
    //    printMatrix(h_S, 1, N, ldda, queues);
    //    printMatrix(h_VT, N, N, ldda, queues);

    //}else{

    //    printMatrix(h_U, M, M, ldda, queues);
    //    printMatrix(h_S, 1, M, ldda, queues);
    //    printMatrix(h_VT, M, N, ldda, queues);

    //}
    

    magma_finalize();

    return 0; 
}

