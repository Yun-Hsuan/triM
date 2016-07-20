/****************************************************************************
*  @file datatype.hpp
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
*  @brief Uni10 general header file for datatype
*  @author Yun-Da Hsieh, Tama Ma
*  @date 2014-05-06
*  @since 0.1.0
*
*****************************************************************************/

#ifndef UNI10_DATATYPE_HPP
#define UNI10_DATATYPE_HPP

//C++ STL Library
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <map>
#include <algorithm>
#include <functional>
#include <locale>
// C Library
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <cctype>
// C++ STL buf
#include <deque>
#include <set>


namespace uni10 {
/// @typedef Real 
/// @brief Short for double
typedef double Real;

/// @typedef Complex 
/// @brief Short for std::complex<double>
typedef std::complex<double> Complex;

//! Parity/Z2 types
enum parityType {
    PRT_EVEN = 0, ///< Parity/Z2 even
    PRT_ODD = 1   ///< Parity/Z2 odd
};
//! Fermion parity types
enum parityFType {
    PRTF_EVEN = 0, ///< Fermion parity even
    PRTF_ODD = 1   ///< Fermion parity odd
};

//!  Bond types
enum bondType {
    BD_IN = 1, ///<Defines an incoming Bond
    BD_OUT = -1  ///<Defines an outgoing Bond
};

//! Real datatype flag
enum rflag{
	RNULL = 0, ///< Real datatype not defined
	RTYPE = 1 ///< Real datatype defined
};

//! Complex datatype flag
enum cflag{
	CNULL = 0,///< Complex datatype not defined
	CTYPE = 2 ///< Complex datatype defined
};

}
#endif
