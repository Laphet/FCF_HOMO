#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>
#include <iostream>

#define DIM 3

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const *const func, char const *const file, int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *const file, int const line);

template <typename T>
struct cuTraits;

template <>
struct cuTraits<float> {
  static const cufftType_t r2cType{CUFFT_R2C};
  static const cufftType_t c2rType{CUFFT_C2R};
  static cuComplex         compVar;
};

template <>
struct cuTraits<double> {
  static const cufftType_t r2cType{CUFFT_D2Z};
  static const cufftType_t c2rType{CUFFT_Z2D};
  static cuDoubleComplex   compVar;
};

template <typename T>
class cufctSolver {
  using cuCompType = decltype(cuTraits<T>::compVar);
  int         dims[DIM];
  T          *realBuffer;
  cuCompType *compBuffer;
  cufftHandle r2cPlan;
  cufftHandle c2rPlan;

public:
  cufctSolver(const int _M, const int _N, const int _P);

  void fctForward(const T *in, T *out_hat);

  void fctBackward(const T *in_hat, T *out);

  ~cufctSolver();
};

template class cufctSolver<float>;

template class cufctSolver<double>;
