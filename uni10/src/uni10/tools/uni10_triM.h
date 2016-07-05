#ifndef UNI10_TRIM_H
#define UNI10_TRIM_H


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <map>
#include <cuda.h>
#include <complex>

class triM{
private:
   struct dptrelem{
    double* devptr;
    double* hostptr;
    size_t memsz;
  };
  struct cptrelem{
    complex* devptr;
    complex* hostptr;
    size_t memsz;
  };

  vector<dptrelem> dptrlist;
  vector<cptrelem> cptrlist;

  map<double*, int> dmap;   map<double*, int>::iterator dit;
  map<complex*, int> cmap;  map<complex*, int>::iterator cit;

public:



  double* switchD2H(double* Diptr);
  double* switchH2D(double* Hiptr);
  complex* switchD2H(complex* Diptr);
  complex* switchH2D(complex* Hiptr);

  void put_devptr(double* Diptr, size_t &memsz);

  void put_hostptr(double* Hiptr, size_t &memsz);
  void put_devptr(complex* Diptr, size_t &memsz);
  void put_hostptr(complex* Hiptr, size_t &memsz);

};


#endif

