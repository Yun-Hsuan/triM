//#include <lib/uni10_triM.cu>
#include <uni10_triM.hpp>
#include <iostream>
#include <cuda.h>
///test the function.
__global__ void dev_test(){

	printf("thr:%d-%s\n",threadIdx.x,"Hello");


}


///override function for chking GPU avalible space
bool Check_GPU_space(){

    return 1;
}


void example_driver_fx(){
    triM MMM;
    //4x4 array
    size_t A_sz =  sizeof(double)*4*4;
    double* A = (double*)malloc(A_sz);



    if( Check_GPU_space() ){
        ///on GPU
        double *DA;
        cudaMalloc((void**)&DA,A_sz);
        MMM.put_devptr(DA,A_sz);

    }else{
        ///on CPU
        double *HA = (double*)malloc(A_sz);
        MMM.put_hostptr(HA,A_sz);

    }


}

int main(int argc,char* argv[]){

	cout << "OK" << endl;
	//cout << "Hello" << endl;
	dev_test<<<1,32>>>();

    cudaDeviceSynchronize();

	return 0;

}
