#include <stdio.h>
#include "uni10.hpp"

using namespace std;
using namespace uni10;


#include <magma.h>
//#include <magma_v2.h>
#include <magma_types.h>

void printMatrix2(double* v, int m, int n, int ldda = 0);

void printMatrix2(double* v, int m, int n, int ldda){
    
    
    double* bufv = NULL;
    if(magma_is_devptr(v) == 1){
        bufv = (double*)malloc(ldda*m*sizeof(double));
        magma_dgetmatrix(n, m, v, ldda, bufv, n);
        printf("help QQ Lu\n");
    }else if(magma_is_devptr(v) == 0){
        bufv = (double*)malloc(m*n*sizeof(double));
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf("%.5f  ", bufv[i*n+j]);
        printf("\n");
    }
}

int main( )
{
    //magma_print_environment();
    guni10_create(2);

    double Syelem[] = {0.4, 2., 0.,
                       2, 0.3, 9.,
                       0., 9., 0.5 };
    double diaelem[] = {2.1, 3.313, 4.139};

    Matrix Msy(3, 3, Syelem);
    Matrix Mdia(3, 3, diaelem, true);
    Matrix Mrand(RTYPE, 3, 8);

    Mrand.randomize();
    cout << Mrand;
    //cout << Mrand;
    vector<Matrix> svd = Mrand.svd();
    cout << svd[0];
    cout << svd[1];
    cout << svd[2];
    //cout << svd[0]*svd[1]*svd[2];
    exit(0);

    cout << Msy;
    cout << Mdia;
    cout << Mdia + Mdia;
    Matrix invM = Msy.inverse();
    cout << invM;
    cout << invM * Msy;

    vector<Matrix> EIG = Msy.eigh();
    cout << EIG[0];
    cout << EIG[1];
    //M.randomize();
    //M.randomize();
    //cout << M;
    //M.randomize();
    //cout << M;
    //Matrix invM(2, 2);
    //invM.setElem(M.getElem(), true);
    //cout << invM.transpose();
    //exit(0);
    //printMatrix(invM.getElem(), 3, 3, uni10::ldda(3, 32));
    //cout << M*invM;
    
    //int rsp_label[] = {1, 0};
    //vector<Bond> bonds(2, Bond(BD_IN, 2));
    //bonds[1] = Bond(BD_OUT, 2);
    //UniTensor A(bonds);
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

