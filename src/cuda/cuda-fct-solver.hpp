#pragma once

#include <math.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cufft.h>
#include <iostream>
#include <math_constants.h>

#define DIM 3

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const *const func, char const *const file, int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *const file, int const line);

template <typename T>
class cufctSolver {
  int                    dims[DIM];
  T                     *realBuffer;
  cuda::std::complex<T> *compBuffer;
  cufftHandle            r2cPlan;
  cufftHandle            c2rPlan;

public:
  cufctSolver(const int _M, const int _N, const int _P);

  ~cufctSolver();

  void fctForward(const T *in, T *out_hat);

  void fctBackward(const T *in_hat, T *out);
};

template class cufctSolver<float>;

template class cufctSolver<double>;