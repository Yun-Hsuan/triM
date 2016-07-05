#include <uni10/tools/uni10_triM.h>
#include <cassert>
#include <vector>
#include <map>
#include <cuda.h>
#include <complex>



double* triM::switchD2H(double* Diptr){
    dit=dmap.find(Diptr);

    assert(dit!=dmap.end());
    int triMidx = dit.second;
    //malloc on host as UAV , cpy to Dev ,and free device side
    double* Hptr;
    cudaHostAlloc( (void**)&Hptr, dptrlist[triMidx].memsz,cudaHostAllocPortable)
    cudaMemcpy(Hptr,dptrlist[triMidx].devptr,cudaMemcpyDefault);

    //relese Dev space
    cudaFree(dptrlist[triMidx].devptr);
    //triM info update
    dptrlist[triMidx].devptr = NULL;
    dptrlist[triMidx].hostptr = Hptr;

    //maintain map
    dmap.erase(dit);
    dmap[Hptr] = triMidx;

    //return ptr for usr
    return Hptr;
}

double* triM::switchH2D(double* Hiptr){
    /* require Hiptr to be in UVA*/

    dit=dmap.find(Hiptr);

    assert(dit!=dmap.end());
    int triMidx = dit.second;

    //malloc on GPU ,cpy to Dev, free host side
    double* Dptr;
    cudaMalloc((void**)&Dptr, dptrlist[triMidx].memsz );
    cudaMemcpy(Dptr,dptrlist[triMidx].hostptr,cudaMemcpyDefault);

    //relese Host ptr in UVA
    cudaFreeHost(dptrlist[triMidx].hostptr);


    //triM info update
    dptrlist[triMidx].devptr = Dptr;
    dptrlist[triMidx].hostptr = NULL;

    //maintain map
    dmap.erase(dit);
    dmap[Dptr] = triMidx;

    //return ptr for usr
    return Dptr;

}

complex* triM::switchD2H(complex* Diptr){
    cit=cmap.find(Diptr);

    assert(cit!=cmap.end());
    int triMidx = cit.second;

    //malloc on host as UAV , cpy to Dev ,and free device side
    complex* Hptr;
    cudaHostAlloc( (void**)&Hptr, cptrlist[triMidx].memsz,cudaHostAllocPortable);
    cudaMemcpy(Hptr,cptrlist[triMidx].devptr,cudaMemcpyDefault);

    //relese Dev space
    cudaFree(cptrlist[triMidx].devptr);
    //triM info update
    cptrlist[triMidx].devptr  = NULL;
    cptrlist[triMidx].hostptr = Hptr;

    //maintain map
    cmap.erase(cit);
    cmap[Hptr] = triMidx;

    //return ptr for usr
    return Hptr;



}

complex* triM::switchH2D(complex* Hiptr){
    /* require Hiptr to be in UVA*/
    cit=cmap.find(Hiptr);

    assert(cit!=cmap.end());
    int triMidx = cit.second;

    //malloc on GPU & cpy to Dev
    complex* Dptr ;
    cudaMalloc((void**)&Dptr, cptrlist[triMidx].memsz );
    cudaMemcpy(Dptr,cptrlist[triMidx].hostptr,cudaMemcpyDefault);

    //relese Host space
    cudaFree(cptrlist[triMidx].hostptr);


    //triM info update
    cptrlist[triMidx].devptr = Dptr;
    cptrlist[triMidx].hostptr = NULL;

    //maintain map
    cmap.erase(cit);
    cmap[Dptr] = triMidx;

    //return ptr for usr
    return Dptr;
}


void triM::put_devptr(double* Diptr,size_t &memsz){

    dptrelem newdelem;
    newdelem.devptr = Diptr;
    newdelem.hostptr = NULL;
    newdelem.memsz = memsz;
    dptrlist.push_back(newdelem);
    dmap[Diptr] = dptrlist.size()-1;
}

void triM::put_hostptr(double* Hiptr,size_t &memsz){
     /* require Hiptr to be in UVA */
    dptrelem newdelem;
    newdelem.devptr = NULL;
    newdelem.hostptr = Hiptr;
    newdelem.memsz = memsz;
    dptrlist.push_back(newdelem);
    dmap[Hiptr] = dptrlist.size()-1;


}
  void triM::put_devptr(complex* Diptr,size_t &memsz){

    cptrelem newcelem;
    newcelem.devptr = Diptr;
    newcelem.hostptr = NULL;
    newcelem.memsz = memsz;
    cptrlist.push_back(newcelem);
    cmap[Diptr] = cptrlist.size()-1;

  }
  void triM::put_hostptr(complex* Hiptr, size_t &memsz){
    /* require Hiptr to be in UVA */
    cptrelem newcelem;
    newcelem.devptr = NULL;
    newcelem.hostptr = Hiptr;
    newcelem.memsz = memsz
    cptrlist.push_back(newcelem);
    cmap[Hiptr] = cptrlist.size()-1;

  }





/*
//glb on launch
triM MMM;
void example_driver_fx(){

    //4x4 array
    size_t A_sz =  sizeof(double)*4*4;
    double* A = malloc(A_sz);

    int retv=Check_GPU_space();

    if(retv== is_OK){
        ///on GPU
        double *DA;
        cudaMalloc((void**)&DA,A_sz);
        MMM.put_devptr(DA);

    }else{
        ///on CPU
        double *HA = malloc(A_sz);
        MMMM.put_hostptr(HA);

    }


}

*/
