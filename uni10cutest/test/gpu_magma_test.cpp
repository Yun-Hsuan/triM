#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <magma.h>
#include <magma_v2.h>
#include <magma_lapack.h>
#include <magma_types.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void printMatrix(double* v, int m, int n, int ldda, magma_queue_t queues);

void printMatrix(double* v, int m, int n, int ldda, magma_queue_t queues){

    double* bufv = (double*)malloc(m*n*sizeof(double));
    if(magma_is_devptr(v) == 1){
        //cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
        magma_dgetmatrix(n, m, v, ldda, bufv, n, queues);
        printf("help QQ Lu\n");
    }else if(magma_is_devptr(v) == 0){
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf("%.5f  ", bufv[i*n+j]);
        printf("\n");
    }
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
    
    //if ( err != MAGMA_SUCCESS ) {
    //    fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
    //    exit(-1);
    //}

    const double c_zero     = MAGMA_D_ZERO;
    const double c_one      = MAGMA_D_ONE;
    const double c_neg_one  = MAGMA_D_NEG_ONE;
    
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    double *h_A, *h_Ainv, *h_R, *work;
    magmaDouble_ptr d_A, dwork;
    magma_int_t ione  = 1;
    magma_int_t ISEED[4] = {0, 0, 0, 1};
    double tmp;
    double error, rwork[1];
    magma_int_t *ipiv;
    magma_int_t status = 0;
    
    N = 4;
    lda = N;
    n2 = lda*N; 
    ldda = magma_roundup(N, 32);
    ldwork = N * magma_get_dgetri_nb(N);

    ipiv = (magma_int_t*)malloc(N*sizeof(magma_int_t));
    h_A = (double*)malloc(n2*sizeof(double));
    h_Ainv = (double*)malloc(n2*sizeof(double));
    h_R = (double*)malloc(n2*sizeof(double));

    magma_malloc((void**)&d_A, ldda*N*sizeof(double));
    magma_malloc((void**)&dwork, ldwork*sizeof(double));

    printf("ldda: %.d\n\n", (int)ldda);

    // Initialize the matrix
    lapackf77_dlarnv(&ione, ISEED, &n2, h_A);
    printMatrix(h_A, N, N, ldda, queues);
    //cudaMemcpy(d_A, h_A, n2*sizeof(double), cudaMemcpyHostToDevice);
    //magma_dsetmatrix(N, N, h_A, lda, d_A, ldda, queues);
    
    printMatrix(d_A, N, N, ldda, queues);
    magma_dgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
    printMatrix(d_A, N, N, ldda, queues);
    magma_dgetmatrix( N, N, d_A, ldda, h_Ainv, lda, queues);
    if (info != 0) {
        printf("magma_dgetrf_gpu returned error %d: %s.\n",
                (int) info, magma_strerror( info ));
    }

    magma_dgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
    printMatrix(d_A, N, N, ldda, queues);
    printf("=================\n");
    magma_finalize();
    return 0; 
}

