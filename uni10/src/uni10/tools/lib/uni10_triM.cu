#include <uni10_triM.hpp>
#include <cassert>
#include <vector>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <complex>

using namespace std;


//#####################################################################################
//#####################################################################################
//                                  triM class:
//
//#####################################################################################


double* triM::switchD2H(double* Diptr){
    dit=dmap.find(Diptr);

    assert(dit!=dmap.end());
    int triMidx = dit->second;
    //malloc on host as UAV , cpy to Dev ,and free device side
    double* Hptr;
    cudaHostAlloc( (void**)&Hptr, dptrlist[triMidx].memsz,cudaHostAllocPortable);
    cudaMemcpy(Hptr,dptrlist[triMidx].devptr,dptrlist[triMidx].memsz,cudaMemcpyDefault);

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
    int triMidx = dit->second;

    //malloc on GPU ,cpy to Dev, free host side
    double* Dptr;
    cudaMalloc((void**)&Dptr, dptrlist[triMidx].memsz );
    cudaMemcpy(Dptr,dptrlist[triMidx].hostptr,dptrlist[triMidx].memsz,cudaMemcpyDefault);

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
/*
complex* triM::switchD2H(complex* Diptr){
    cit=cmap.find(Diptr);

    assert(cit!=cmap.end());
    int triMidx = cit->second;

    //malloc on host as UAV , cpy to Dev ,and free device side
    complex* Hptr;
    cudaHostAlloc( (void**)&Hptr, cptrlist[triMidx].memsz,cudaHostAllocPortable);
    cudaMemcpy(Hptr,cptrlist[triMidx].devptr,cptrlist[triMidx].memsz,cudaMemcpyDefault);

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
    /// require Hiptr to be in UVA
    cit=cmap.find(Hiptr);

    assert(cit!=cmap.end());
    int triMidx = cit->second;

    //malloc on GPU & cpy to Dev
    complex* Dptr ;
    cudaMalloc((void**)&Dptr, cptrlist[triMidx].memsz );
    cudaMemcpy(Dptr,cptrlist[triMidx].hostptr,cptrlist[triMidx].memsz,cudaMemcpyDefault);

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

*/
void triM::put_devptr(double* Diptr,size_t &memsz){

    dptrelem newdelem;
    newdelem.devptr = Diptr;
    newdelem.hostptr = NULL;
    newdelem.memsz = memsz;
    dptrlist.push_back(newdelem);
    dmap[Diptr] = dptrlist.size()-1;
}

void triM::put_hostptr(double* Hiptr,size_t &memsz){

    dptrelem newdelem;
    newdelem.devptr = NULL;
    //change the memory to UAV:
    cudaHostRegister ( Hiptr, memsz, cudaHostRegisterPortable );
    newdelem.hostptr = Hiptr;
    newdelem.memsz = memsz;
    dptrlist.push_back(newdelem);
    dmap[Hiptr] = dptrlist.size()-1;


}

void triM::put_hostUAVptr(double* Hiptr,size_t &memsz){
     /* require Hiptr to be in UVA */
    dptrelem newdelem;
    newdelem.devptr = NULL;

    newdelem.hostptr = Hiptr;
    newdelem.memsz = memsz;
    dptrlist.push_back(newdelem);
    dmap[Hiptr] = dptrlist.size()-1;


}



/*
  void triM::put_devptr(complex* Diptr,size_t &memsz){

    cptrelem newcelem;
    newcelem.devptr = Diptr;
    newcelem.hostptr = NULL;
    newcelem.memsz = memsz;
    cptrlist.push_back(newcelem);
    cmap[Diptr] = cptrlist.size()-1;

  }
  void triM::put_hostptr(complex* Hiptr, size_t &memsz){

    cptrelem newcelem;
    newcelem.devptr = NULL;
    newcelem.hostptr = Hiptr;
    newcelem.memsz = memsz
    cptrlist.push_back(newcelem);
    cmap[Hiptr] = cptrlist.size()-1;

  }

  void triM::put_hostUAVptr(complex* Hiptr, size_t &memsz){
    /// require Hiptr to be in UVA /
    cptrelem newcelem;
    newcelem.devptr = NULL;
    //change the memory to UAV:
    cudaHostRegister ( Hiptr, mem_sz, cudaHostRegisterPortable );
    newcelem.hostptr = Hiptr;
    newcelem.memsz = memsz
    cptrlist.push_back(newcelem);
    cmap[Hiptr] = cptrlist.size()-1;

  }
*/




  bool triM:: Memfree(double* Hiptr){

    dit=dmap.find(Hiptr);
    if(dit==dmap.end()) return 1;
    else{
        int triMidx= dit->second;
        ///free and set the current index as NULL (no remove!)
        if(dptrlist[triMidx].hostptr==NULL){
            cudaFree(Hiptr);
            dptrlist[triMidx].devptr==NULL;
        }else{
            cudaFreeHost(Hiptr);
            dptrlist[triMidx].hostptr==NULL;
        }
        dmap.erase(dit); //de-associate the map key
    }
    return 0;


  }
/*
  bool triM:: Memfree(complex* Hiptr){

    cit=cmap.find(Hiptr);
    if(cit==cmap.end()) return 1;
    else{
        triMidx= cit->second;
        ///free and set the current index as NULL (no remove!)
        if(cptrlist[triMidx].hostptr==NULL){
            cudaFree(Hiptr);
            cptrlist[triMidx].devptr==NULL
        }else{
            cudaFreeHost(Hiptr);
            cptrlist[triMidx].hostptr==NULL
        }
        cmap.erase(cit); //de-associate the map key
    }
    return 0;



  }
*/


//#####################################################################################
//#####################################################################################
//#####################################################################################



GPU_enviroment::GPU_enviroment(){
    Available_cnt=0;

    ///get detected devices:
    cudaGetDeviceCount(&cnt);
        if(cnt==0) cout << "[ERROR] no device detected";
        assert(cnt!=0);


    ///explicitly query busy signal:
    bool *Dtest;
    for(int i = 0; i < cnt; i++){
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        if(cudaMalloc((void**)&Dtest, 100*sizeof(bool)) == cudaSuccess){
            ///GPU OK

            DevInfos.push_back(pair<bool,cudaDeviceProp>(1,prop));
            Available_cnt++;
        }else{
            ///GPU unavailable
            DevInfos.push_back(pair<bool,cudaDeviceProp>(0,prop));

        }
    }



}

void GPU_enviroment::print_info(){



    int kb = 1024;
    int mb = kb * kb;

    cout << "CUDA version:   v" << CUDART_VERSION << endl;

    cout << "CUDA Devices:   total: " << cnt << "  available: "<< Available_cnt <<  endl;
    if(DevInfos.size()!=0){
        cout << "Avalible Devices Index:   [";
        for(int i=0;i< cnt;i++){
            if(DevInfos[i].first) cout << " " << i << " ";
        }
        cout << "]"<<endl;
    }
    cout << endl;

    for(int i = 0; i < cnt; ++i)
    {
        cudaDeviceProp props = DevInfos[i].second;
        cout << "Device   :Idx: " << i <<endl;
        cout << "____________________________________________"<<endl;
        cout << ": " << props.name << ": " << props.major << "." << props.minor << endl;

        if(DevInfos[i].first){

            cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
            cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
            cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
            cout << "  Block registers: " << props.regsPerBlock << endl << endl;

            cout << "  Warp size:         " << props.warpSize << endl;
            cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
            cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
            cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
            cout << "  Unified Virtual Address: " << props.unifiedAddressing <<endl;
            cout << endl;


        }else{

            cout << "  Unavailble." << endl;


        }
        cout << "=============================================" << endl;


    }






}



ostream& operator<<(ostream& os, GPU_enviroment& gpue){

    gpue.print_info();

    return os;
}


