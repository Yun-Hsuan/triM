/****************************************************************************
*  @file uni10_tools_cpu.cpp
*  @license
*    Universal Tensor Network Library
*    Copyright (c) 2013-2016
*    National Taiwan University
*    National Tsing-Hua University
*
*    This file is part of Uni10, the Universal Tensor Network Library.
*
*    Uni10 is free software: you can redistribute it and/or modify
*    it under the terms of the GNU Lesser General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Uni10 is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public License
*    along with Uni10.  If not, see <http://www.gnu.org/licenses/>.
*  @endlicense
*  @brief Implementation file for helper functions on CPU
*  @author Yun-Da Hsieh,Yun-Hsuan Chou
*  @date 2014-05-06
*  @since 1.0.0
*
*****************************************************************************/
#include <uni10/tools/uni10_tools.h>



namespace uni10{

  void guni10_create(int GMODE_, bool printUsage){
    if(GMODE_ != 0){
      fprintf(stderr, "assert to GMODE = 0 in cpu version\n");
      exit(0);
    }
    if(printUsage){
      fprintf(stderr, "Does not support in cpu version\n");
    }
  }

  void guni10_destroy(){}

  void* elemAlloc(size_t memsize, bool& ongpu){
    void* ptr = NULL;
    ptr = malloc(memsize);
    if(ptr == NULL){
      std::ostringstream err;
      err<<"Fails in allocating memory.";
      throw std::runtime_error(exception_msg(err.str()));
    }
    MEM_USAGE += memsize;
    ongpu = false;
    return ptr;
  }

  /**********  Modified for magma  **********/

  void* elemAlloc(size_t m, size_t n, size_t typesize, bool& ongpu, bool diag){

    void* ptr = NULL;

    size_t memsize = diag ? std::min(m, n) * typesize : m * n * typesize;
    ptr = malloc(memsize);

    if(ptr == NULL){
      std::ostringstream err;
      err<<"Fails in allocating memory.";
      throw std::runtime_error(exception_msg(err.str()));
    }

    MEM_USAGE += memsize;
    ongpu = false;

    return ptr;
  }

  void* elemAllocForce(size_t m, size_t n, size_t typesize, bool ongpu, bool diag){

    void* ptr = NULL;

    size_t memsize = diag ? std::min(m, n) * typesize : m * n * typesize;
    ptr = malloc(memsize);

    if(ptr == NULL){
      std::ostringstream err;
      err<<"Fails in allocating memory.";
      throw std::runtime_error(exception_msg(err.str()));
    }

    MEM_USAGE += memsize;
    return ptr;

  }

  void* elemCopy(double* des, const double* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag){

    size_t memsize = diag ? std::min(m, n) * sizeof(double) : m * n * sizeof(double);
    return memcpy(des, src, memsize);

  }

  void* elemCopy(std::complex<double>* des, const std::complex<double>* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag){

    size_t memsize = diag ? std::min(m, n) * sizeof(std::complex<double>) : m * n * sizeof(std::complex<double>);
    return memcpy(des, src, memsize);

  }

  void elemBzero(double* ptr, size_t m, size_t n, bool ongpu, bool diag){

    size_t memsize = diag ? std::min(m, n) * sizeof(double) : m * n * sizeof(double);
    memset(ptr, 0, memsize);

  }

  void elemBzero(std::complex<double>* ptr, size_t m, size_t n, bool ongpu, bool diag){

    size_t memsize = diag ? std::min(m, n) * sizeof(std::complex<double>) : m * n * sizeof(std::complex<double>);
    memset(ptr, 0, memsize);

  }

  void elemRand(double* elem, size_t m, size_t n, bool ongpu, bool diag){
      
    size_t N = diag ? std::min(m, n) : m * n;
    elemRand(elem, N, ongpu);

  }

  /******************************************/

  void* elemAllocForce(size_t memsize, bool ongpu){
    void* ptr = NULL;
    ptr = malloc(memsize);
    if(ptr == NULL){
      std::ostringstream err;
      err<<"Fails in allocating memory.";
      throw std::runtime_error(exception_msg(err.str()));
    }
    MEM_USAGE += memsize;
    return ptr;
  }

  void* elemCopy(void* des, const void* src, size_t memsize, bool des_ongpu, bool src_ongpu){
    return memcpy(des, src, memsize);
  }

  void elemFree(void* ptr, size_t memsize, bool ongpu){
    free(ptr);
    MEM_USAGE -= memsize;
    ptr = NULL;
  }

  void elemBzero(void* ptr, size_t memsize, bool ongpu){
    memset(ptr, 0, memsize);
  }

  void elemRand(double* elem, size_t N, bool ongpu){

    for(size_t i = 0; i < N; i++)
      elem[i] = ((double)rand()) / RAND_MAX; //lapack_uni01_sampler();

  }

  void setDiag(double* elem, double* diag_elem, size_t m, size_t n, size_t diag_n, bool ongpu, bool diag_ongpu){
    size_t min = m < n ? m : n;
    min = min < diag_n ? min : diag_n;
    for(size_t i = 0; i < min; i++)
      elem[i * n + i] = diag_elem[i];
  }
  void getDiag(double* elem, double* diag_elem, size_t m, size_t n, size_t diag_n, bool ongpu, bool diag_ongpu){
    size_t min = m < n ? m : n;
    min = min < diag_n ? min : diag_n;
    for(size_t i = 0; i < min; i++)
      diag_elem[i] = elem[i * n + i];
  }
  void* mvGPU(void* elem, size_t memsize, bool& ongpu){
    ongpu = false;
    return elem;
  }
  void* mvCPU(void* elem, size_t memsize, bool& ongpu){
    ongpu = false;
    return elem;
  }
  void syncMem(void** elemA, void** elemB, size_t memsizeA, size_t memsizeB, bool& ongpuA, bool& ongpuB){
    ongpuA = false;
    ongpuB = false;
  }

  void shrinkWithoutFree(size_t memsize, bool ongpu){
    MEM_USAGE -= memsize;
  }

  void reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem){
    std::ostringstream err;
    err<<"Fatal error(code = T1). Please contact the developer of the uni10 library.";
    throw std::runtime_error(exception_msg(err.str()));
  }

  double getElemAt(size_t idx, double* elem, bool ongpu){
    return elem[idx];
  }

  void setElemAt(size_t idx, double val, double* elem, bool ongpu){
    elem[idx] = val;
  }


  double  elemMax(double* elem, size_t elemNum, bool ongpu){

    if (ongpu) {
      // GPU not implemented
      std::ostringstream err;
      err<<"Fatal error(code = T1). GPU version is not implemented.";
      throw std::runtime_error(exception_msg(err.str()));
    } else {
      double max;
      max=elem[0];

      for (size_t i=1; i<elemNum; i++)
	if (max < elem[i]) max=elem[i];
      return max;
    }
  }

  double  elemAbsMax(double* elem, size_t elemNum, bool ongpu){

    if (ongpu) {
      // GPU not implemented
      std::ostringstream err;
      err<<"Fatal error(code = T1). GPU version is not implemented.";
      throw std::runtime_error(exception_msg(err.str()));
    } else {

      size_t idx = 0;
      double max = fabs(elem[0]);

      for (size_t i=1; i<elemNum; i++)
	if (max < fabs(elem[i])){
	  max=fabs(elem[i]);
	  idx = i;
	}
      return elem[idx];
    }
  }

  /***** Complex version *****/
  std::complex<double> getElemAt(size_t idx, std::complex<double>* elem, bool ongpu){
    return elem[idx];
  }

  void setElemAt(size_t idx, std::complex<double> val, std::complex<double> *elem, bool ongpu){
    elem[idx] = val;
  }

  void elemRand(std::complex<double>* elem, size_t N, bool ongpu){
    for(size_t i = 0; i < N; i++)
      elem[i] = std::complex<double>(((double)rand()) / RAND_MAX, ((double)rand()) / RAND_MAX); //lapack_uni01_sampler();
  }

  void elemCast(std::complex<double>* des, double* src, size_t N, bool des_ongpu, bool src_ongpu){
    for(size_t i = 0; i < N; i++)
      des[i] = src[i];
  }
  void elemCast(double* des, std::complex<double>* src, size_t N, bool des_ongpu, bool src_ongpu){
    for(size_t i = 0; i < N; i++)
      des[i] = src[i].real();
  }
  void setDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t m, size_t n, size_t diag_n, bool ongpu, bool diag_ongpu){
    size_t min = m < n ? m : n;
    min = min < diag_n ? min : diag_n;
    for(size_t i = 0; i < min; i++)
      elem[i * n + i] = diag_elem[i];
  }
  void getDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t m, size_t n, size_t diag_n, bool ongpu, bool diag_ongpu){
    size_t min = m < n ? m : n;
    min = min < diag_n ? min : diag_n;
    for(size_t i = 0; i < min; i++)
      diag_elem[i] = elem[i * n + i];
  }

  void reshapeElem(std::complex<double>* oldElem, int bondNum, size_t elemNum, size_t* offset, std::complex<double>* newElem){
    std::ostringstream err;
    err<<"Fatal error(code = T1). Please contact the developer of the uni10 library.";
    throw std::runtime_error(exception_msg(err.str()));
  }


};	/* namespace uni10 */
