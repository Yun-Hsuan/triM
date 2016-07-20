#include <stdio.h>
#include "uni10.hpp"

using namespace std;
using namespace uni10;

//#include <magma.h>
//#include <magma_v2.h>
//#include <magma_lapack.h>

//void printMatrix(double* v, int m, int n, int ldda = 0);
/*
void printMatrix(double* v, int m, int n, int ldda){

    double* bufv = (double*)malloc(m*n*sizeof(double));
    if(magma_is_devptr(v) == 1){
        //cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
        magma_dgetmatrix(n, m, v, ldda, bufv, n);
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
*/

int main()
{
    //magma_print_environment();
    guni10Create(0);
    Matrix M(2, 2);
    M.randomize();
    cout << M;
    Matrix invM = M.inverse();
    cout << M*invM;
    //int rsp_label[] = {1, 0};
    //vector<Bond> bonds(2, Bond(BD_IN, 2));
    //bonds[1] = Bond(BD_OUT, 2);
    //UniTensor A(bonds);
    //Matrix M_cpu(RTYPE, 12, 12);
    //M_cpu.randomize();
    //M_cpu *= -1;
    //cout << M_cpu << endl;
    //Matrix invM = M_cpu.inverse();
    //Matrix invM(M_cpu);
    //cout << invM << endl;
    //cout << M_cpu << endl;
    //invM.sum();
    //cout << M_cpu << endl;
    //cout << invM << endl;
    //cout << invM*M_cpu << endl;
    //A.putBlock(M_cpu);
    //cout << A << endl;
    //cout << A+A << endl;
    //cout << 2.*A << endl;
    //cout << M_cpu.norm() << endl;
    //A.permute(rsp_label, 2);

    return 0; 
}

