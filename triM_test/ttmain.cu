//#include <lib/uni10_triM.cu>
#include <uni10_triM.hpp>
#include <iostream>
#include <cuda.h>
using namespace std;

///test the function.
__global__ void dev_test(){

	printf("thr:%d-%s\n",threadIdx.x,"Hello");


}


///override function for chking GPU avalible space
bool Check_GPU_space(){
    // in here try to calcuate the GPU space usage
    return 1;
}




int main(int argc,char* argv[]){


	///Get all the devices info.
	GPU_enviroment GPUenv;

    ///print all the info of current GPUs status.
	cout << GPUenv << endl;


    ///define triM containter:
    triM MMM;



    ///simple programming model:
    /*
       1 Do whatever you did as in CPU code.
        e.g. malloc... cudaMalloc...
        example code:

    */
     //4x32 array on GPU
     size_t A_sz =  sizeof(double)*4*32;
     double* A = (double*)malloc(A_sz);


    /*
       2 If wish to put on GPU,
        remember additional step is to put the allocated pointer into triM
    */
    MMM.put_hostptr(A,A_sz);
    //exchage the pointer
    A = MMM.switchH2D(A);  // now "A" is on device!




    /* 3 Do calculation*/



    /* 4 switch to CPU*/
    //exchage the pointer
    A = MMM.switchD2H(A); // now "A" is on host!


    /* 5 free the memory: */
    bool status;
    status= MMM.Memfree(A);

    if(status==0) cout << "OK"<<endl;
    else cout << "ERROR! pointer is not handle by triM."<<endl;








	//cout << "Hello" << endl;
	//dev_test<<<1,32>>>();

    cudaDeviceSynchronize();

	return 0;

}
