/****************************************************************************
*  @file uni10_tools.h
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
*  @brief Header file for helper string functions and wrapper functions
*  @author Yun-Da Hsieh
*  @date 2014-05-06
*  @since 0.1.0
*
*****************************************************************************/
#ifndef UNI10_TOOLS_H
#define UNI10_TOOLS_H

#include <uni10/data-structure/uni10_struct.h>
#include <uni10/tools/helper_uni10.h>

namespace uni10{

/* Global variable */
extern bool ONGPU;
extern int GMODE;
extern size_t MEM_USAGE;
extern size_t GPU_FREE_MEM;
extern int blocksize;

void magma_print_matrix(double* v, int m, int n, int ldda);

void cuda_print_matrix(double* v, int m, int n);

void guni10_create(int GMODE_ = 0, bool printUsage = false);

void guni10_destroy();

/* ========= Magma ========== */

void* elemAlloc(size_t m, size_t n, size_t typesize, bool& ongpu, bool diag);

void* elemAllocForce(size_t m, size_t n, size_t typesize, bool ongpu, bool diag);

void* elemCopy(double* des, const double* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag);

void* elemCopy(std::complex<double>* des, const std::complex<double>* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag);

void elemBzero(double* ptr, size_t m, size_t n, bool ongpu, bool diag);

void elemBzero(std::complex<double>* ptr, size_t m, size_t n, bool ongpu, bool diag);

void elemRand(double* elem, size_t m, size_t n, bool ongpu, bool diag);
//#ifdef CUDA_SUPPORT
void magma_to_cuda(size_t m, size_t n, double* mag_ptr, size_t ldda, double* cu_ptr);

void cuda_to_magma(size_t m, size_t n, double* cu_ptr, double* mag_ptr, size_t ldda);
//#endif

/********************************/

//For CPU version
void* elemAlloc(size_t memsize, bool& ongpu);  

void* elemAllocForce(size_t memsize, bool ongpu);
//For CPU version
void* elemCopy(void* des, const void* src, size_t memsize, bool des_ongpu, bool src_ongpu);

void elemFree(void* ptr, size_t memsize, bool ongpu);
//For CPU version
void elemBzero(void* ptr, size_t memsize, bool ongpu);
//For CPU version
void elemRand(double* elem, size_t N, bool ongpu);

std::vector<_Swap> recSwap(std::vector<int>& ord, std::vector<int>& ordF);

std::vector<_Swap> recSwap(std::vector<int>& ord);	//Given the reshape order out to in.

void setDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

void getDiag(double* elem, double* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

void* mvGPU(void* elem, size_t memsize, bool& ongpu);

void* mvCPU(void* elem, size_t memsize, bool& ongpu);

void syncMem(void** elemA, void** elemB, size_t memsizeA, size_t memsizeB, bool& ongpuA, bool& ongpuB);

void shrinkWithoutFree(size_t memsize, bool ongpu);

void reshapeElem(double* oldElem, int bondNum, size_t elemNum, size_t* offset, double* newElem);

double getElemAt(size_t idx, double* elem, bool ongpu);

void setElemAt(size_t idx, double val, double* elem, bool ongpu);

void propogate_exception(const std::exception& e, const std::string& func_msg);

std::string exception_msg(const std::string& msg);

double elemMax(double *elem, size_t ElemNum, bool ongpu);

double elemAbsMax(double *elem, size_t ElemNum, bool ongpu);

/***** Complex version *****/
std::complex<double> getElemAt(size_t idx, std::complex<double>* elem, bool ongpu);

void setElemAt(size_t idx, std::complex<double> val, std::complex<double>* elem, bool ongpu);

void elemRand(std::complex<double>* elem, size_t N, bool ongpu);

void elemCast(std::complex<double>* des, double* src, size_t N, bool des_ongpu, bool src_ongpu);

void elemCast(double *des, std::complex<double> *src, size_t N, bool des_ongpu, bool src_ongpu);

void setDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

void getDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu);

void reshapeElem(std::complex<double>* oldElem, int bondNum, size_t elemNum, size_t* offset, std::complex<double>* newElem);

// trim from start
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}
// trim from end
static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}
// trim from both ends
static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}

// compute ldda of matrix. 
static inline int ldda(int x, int y){
    return ((x + y -1) / y) * y;
}

static inline int uni10_memsize(int m, int n, int size, bool ongpu, int blocksize){
   return ongpu ? ldda(m, blocksize) * n * size : m*n*size;
}

};	/* namespace uni10 */
#endif /* UNI10_TOOLS_H */
