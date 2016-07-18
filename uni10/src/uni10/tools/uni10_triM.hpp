#ifndef UNI10_TRIM_H
#define UNI10_TRIM_H


#include <iostream>
#include <cstdio>

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <map>
#include <cuda.h>
//#include <complex>


class triM{
private:
    ///struct for each <double> variable pointer info
    struct dptrelem{
        ///pointer on device part of UVA
        double* devptr;
        ///pointer on host part of UVA
        double* hostptr;
        ///memory size of variable
        size_t memsz;
    };

    /*
    ///struct for each <complex> variable pointer info
    struct cptrelem{
        ///pointer on device part of UVA
        std::complex* devptr;
        ///pointer on host part of UVA
        std::complex* hostptr;
        ///memory size of variable
        size_t memsz;
    };
    */

    ///@brief: the container of the now-a-live variables infos.
    ///@content: for maintain the pointer in UVA
    std::vector<dptrelem> dptrlist; // double type
    //std::vector<cptrelem> cptrlist; // complex type

    ///@brief: look up table for quick get the currespont pointer info.
    ///@content: [NOTE] all the pointer must alloc as UVA
    std::map<double*, int> dmap; //double type
    std::map<double*, int>::iterator dit; //double type iterator
    //std::map<std::complex*, int> cmap;  //complex type
    //std::map<std::complex*, int>::iterator cit; //complex type iterator

public:


  ///switch from Device to Host
  double* switchD2H(double* Diptr); //double
  //std::complex* switchD2H(std::complex* Diptr);   //complex

  ///switch from Host to Device
  double* switchH2D(double* Hiptr); //double
 // std::csomplex* switchH2D(std::complex* Hiptr); //complex

  /**@brief: func. for putting the "device" pointers into container
   * @param: [1] *Diptr <double> / <complex> : device pointer in UVA (<double> type or <complex> type)
   * @param: [2] &memsz <size_t> : the memory size of the pointer.
  */
  void put_devptr(double* Diptr, size_t &memsz);
  //void put_devptr(std::complex* Diptr, size_t &memsz);

  /**@brief: func. for putting the "host" pointers into container
   * @param: [1] *Hiptr <double> / <complex> : device pointer allocated using malloc/calloc (<double> type or <complex> type)
   * @param: [2] &memsz <size_t> : the memory size of the pointer.
  */
  void put_hostptr(double* Hiptr, size_t &memsz);
  //void put_hostptr(std::complex* Hiptr, size_t &memsz);

  /**@brief: func. for putting the "host UAV" pointers into container
   * @param: [1] *Hiptr <double> / <complex> : device pointer in UVA (<double> type or <complex> type)
   * @param: [2] &memsz <size_t> : the memory size of the pointer.
  */
  void put_hostUAVptr(double* Hiptr, size_t &memsz);
  //void put_hostUAVptr(std::complex* Hiptr, size_t &memsz);

  /**@brief: func. for free the memory.
   * @param: [1] *Hiptr <double> / <complex> : existance pointers.
   * @return: 0 - OK
   *          1 - the pointer <Hiptr> is not in triM
  */
  bool Memfree(double* Hiptr);
  //bool Memfree(complex* Hiptr);




};




/**@brief: Class contain all the GPUs info. of current execution
 * @content: query all the GPUs info. for
 *            1. occupency and warp spanning
 *            2. dynamic pipe line(D2H/H2D).
*/
//Note , this should be call in the init stage
class GPU_enviroment{
public:

    ///@brief: number of avalible GPUs
    int cnt;
    int Available_cnt;
    std::vector< std::pair<bool,cudaDeviceProp> > DevInfos;

    ///@brief: constructor
    GPU_enviroment();

    //~GPU_enviroment();
    ///@brief: print the info of GPUs.
    void print_info();




};

///@brief: overload << for stdout GPU_enviroment class.
std::ostream& operator<<(std::ostream& os, GPU_enviroment& gpue);

#endif

