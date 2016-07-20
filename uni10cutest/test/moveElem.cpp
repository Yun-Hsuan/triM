#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <magma.h>
#include <magma_lapack.h>
#include <magma_types.h>

void printMatrix(const double* v, int m, int n, int ldda);

void printMatrix(const double* v, int m, int n, int ldda){

    printf("[\n");
    double* bufv = (double*)malloc(m*n*sizeof(double));
    if(magma_is_devptr(v) == 1){
        //cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
        printf("-- On GPU --\n");
        magma_dgetmatrix(n, m, v, ldda, bufv, n);
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

    //magma_queue_t queues;
    //magma_queue_create( 0, &queues );
    
    //if ( err != MAGMA_SUCCESS ) {
    //    fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
    //    exit(-1);
    //}

    const double c_zero     = MAGMA_D_ZERO;
    const double c_one      = MAGMA_D_ONE;
    const double c_neg_one  = MAGMA_D_NEG_ONE;
    
    magma_int_t M, K, N, MK, KN, MN, lda, ldb, ldc, lddn,info;
    magma_int_t lddam, lddbm, lddcm, lddan, lddbn, lddcn;
    double *h_A, *h_B, *h_C;
    magmaDouble_ptr d_A, d_B, d_C, d_Ainv;
    magma_int_t ione  = 1;
    magma_int_t ISEED[4] = {0, 0, 0, 1};
    double tmp;
    double error, rwork[1];
    magma_int_t status = 0;
    
    M = 4;
    K = 3;
    N = 4;

    //lda = M;
    //ldb = K;
    //ldc = M;
    
    lda = K;
    ldb = N;
    ldc = N;

    MK = M*K; 
    KN = K*N; 
    MN = M*N; 

    //lddam = magma_roundup(M, 32);
    //lddbm = magma_roundup(K, 32);
    //lddcm = magma_roundup(M, 32);
    //lddn = magma_roundup(N, 32);
    lddan = magma_roundup(K, 32);
    lddbn = magma_roundup(N, 32);
    lddcn = magma_roundup(N, 32);
    //lddan = K;
    //lddbn = N;
    //lddcn = N;
    lddam = M;
    lddbm = K;
    lddcm = M;

    h_A = (double*)malloc(MK*sizeof(double));
    h_B = (double*)malloc(KN*sizeof(double));
    h_C = (double*)malloc(MN*sizeof(double));

    magma_malloc((void**)&d_A, lddam*lddan*sizeof(double));
    magma_malloc((void**)&d_Ainv, lddam*lddan*sizeof(double));
    magma_malloc((void**)&d_B, lddbm*lddbn*sizeof(double));
    magma_malloc((void**)&d_C, lddcm*lddcn*sizeof(double));

    //printMatrix(d_A, M, K, lddan);
    //exit(0);

    printf("\n\n ldda: %d, M: %d, N: %d \n\n", (int)lddan, (int)M, (int)N);

    // Initialize the matrix
    lapackf77_dlarnv(&ione, ISEED, &MK, h_A);
    lapackf77_dlarnv(&ione, ISEED, &KN, h_B);
    memset(h_C, 0, MN*sizeof(double));

    printf("KKKK: %d\n\n\n", (int)K);

    printMatrix(h_A, M, K, lddan);
    printf("====================\n");
    printMatrix(h_B, K, N, lddbn);
    printf("====================\n");
    printMatrix(h_C, M, N, lddcn);
    printf("====================\n");
    //cudaMemcpy(d_A, h_A, n2*sizeof(double), cudaMemcpyHostToDevice);
    magma_dsetmatrix(K, M, h_A, K, d_A, magma_roundup(K, 32));
    magma_dsetmatrix(N, K, h_B, ldb, d_B, lddbn);
    magma_dsetmatrix(N, M, h_C, ldc, d_C, lddcn);
    magma_int_t idc = 1;
    magma_dcopy(M*lddan, d_A, idc, d_Ainv, idc);


    printf("=========A===========\n");
    printMatrix(d_A, M, K, lddan);
    printf("=========A_inv===========\n");
    printMatrix(d_Ainv, M, K, lddan);
    printf("=========B===========\n");
    printMatrix(d_B, K, N, lddbn);
    printf("=========C===========\n");
    printMatrix(d_C, M, N, lddbn);
    printf("====================\n");

    double alpha = 1.;
    double beta = 0;
    blasf77_dgemm((char*)"N", (char*)"N", &N, &M, &K, &alpha, h_B, &N, h_A, &K, &beta, h_C, &N);

    printf("---- CPU dgemm ----\n");
    printMatrix(h_C, M, N, lddcm);
    printf("---- CPU  end  ----\n\n\n");

    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, N, M, K, alpha, d_B, lddbn, d_A, lddan, beta, d_C, lddcn);

    printf("---- GPU dgemm ----\n");
    printMatrix(d_C, M, N, lddcn);
    printf("---- GPU  end  ----\n\n\n");

    magma_int_t *ipiv = (magma_int_t*)malloc(M*sizeof(magma_int_t));
    magma_dgetrf_gpu( N, N, d_Ainv, lddan, ipiv, &info );
    //printMatrix(d_A, N, N, lddan, queues);
    //magma_dgetmatrix( N, N, d_Ainv, ldda, h_Ainv, lda, queues);
    if (info != 0) {
        printf("magma_dgetrf_gpu returned error %d: %s.\n",
                (int) info, magma_strerror( info ));
    }

    double* dwork;
    magma_int_t ldwork = N * magma_get_dgetri_nb(N);
    magma_malloc((void**)&dwork, ldwork*sizeof(double));

    magma_dgetri_gpu( N, d_Ainv, lddan, ipiv, dwork, ldwork, &info );
    printf("=======d_Ainv==========\n");
    printMatrix(d_Ainv, N, N, lddan);
    printf("=================\n");

    double *Idn;
    magma_malloc((void**)&Idn, M*lddcn*sizeof(double));
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, N, N, N, alpha, d_Ainv, lddan, d_A, lddan, beta, Idn, lddcn);
    printf("=======d_Idn==========\n");
    printMatrix(Idn, N, N, lddan);
    printf("=================\n");

    magmaDouble_ptr d_AT;
    magma_dmalloc(&d_AT, magma_roundup(M, 32)*K);
    magmablas_dtranspose(K, M, d_A, magma_roundup(K, 32), d_AT, magma_roundup(M, 32));
    magma_dprint_gpu(K, M, d_A, lddan);
    magma_dprint_gpu(M, K, d_AT, magma_roundup(M, 32));

    // dsyevd
    //
    //
    N = 3;
    double Syelem[] = {0.4, 2., 0.,
                      2, 0.3, 9.,
                      0., 9., 0.5 };
    double* h_sy = (double*)malloc(N*N*sizeof(double));
    memcpy(h_sy, Syelem, N*N*sizeof(double));
    double* d_sy = NULL;
    magma_int_t ldda = magma_roundup(N, 32);
    magma_dmalloc(&d_sy, N*ldda);
    magma_dsetmatrix(N, N, h_sy, N, d_sy, ldda);

    magma_dprint_gpu(N, N, d_sy, ldda);

    double aux_work[1];
    magma_int_t aux_iwork[1];

    magma_dsyevd_gpu( MagmaVec, MagmaUpper, 
                      N, NULL, ldda, NULL, // A, w
                      NULL, lda,            // host A
                      aux_work, -1,  
                      aux_iwork, -1,
                      &info
                      );         

    double* w1 = (double*)malloc(N*sizeof(double));
    double* w2 = (double*)malloc(N*sizeof(double));
    double* pin_Sy = NULL, *h_work = NULL;

    magma_int_t lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
    magma_dmalloc_pinned(&pin_Sy, N * ldda);
    magma_dmalloc_pinned(&h_work, lwork);
    magma_int_t liwork = aux_iwork[0];
    magma_int_t* iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));

    magma_dsyevd_gpu( MagmaVec, MagmaUpper,
            N, d_sy, ldda, w1,
            pin_Sy, lda,
            h_work, lwork,
            iwork, liwork,
            &info );

    //magma_dprint_gpu(N, N, d_sy, ldda);

    printMatrix(d_sy, N, N, ldda);
    printMatrix(w1, 1, N, ldda);
    // CPU
    double worktest;
    magma_int_t lhwork = -1; 
    magma_int_t ldA = N;
    lapackf77_dsyev((char*)"V", (char*)"U", &N, h_sy, &ldA, w2, &worktest, &lhwork, &info);
    lhwork = (magma_int_t) MAGMA_D_REAL(worktest);
    double* workspace = (double*)malloc(sizeof(double)*lhwork);
    lapackf77_dsyev((char*)"V", (char*)"U", &N, h_sy, &ldA, w2, workspace, &lhwork, &info);

    printMatrix(h_sy, N, N, ldda);
    printMatrix(w2, 1, N, ldda);

    magma_finalize();

    return 0; 
}

