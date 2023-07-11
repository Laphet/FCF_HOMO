#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <iomanip>

// static constexpr int DIM{3};

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t status, char const *const func, char const *const file, int const line);
void check(cufftResult status, char const *const func, char const *const file, int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *const file, int const line);

template <typename T>
struct cuTraits;

template <>
struct cuTraits<float> {
  static const cufftType_t r2cType{CUFFT_R2C};
  static const cufftType_t c2rType{CUFFT_C2R};
  static const cufftType_t c2cType{CUFFT_C2C};
  static cuComplex         compVar;
};

template <>
struct cuTraits<double> {
  static const cufftType_t r2cType{CUFFT_D2Z};
  static const cufftType_t c2rType{CUFFT_Z2D};
  static const cufftType_t c2cType{CUFFT_Z2Z};
  static cuDoubleComplex   compVar;
};

template <typename T>
class cufctSolver {
  using cuCompType = decltype(cuTraits<T>::compVar);
  int         dims[3];
  T          *realBuffer;
  cuCompType *compBuffer;
  cufftHandle r2cPlan;
  cufftHandle c2rPlan;
  // cufftHandle c2cPlan;

public:
  cufctSolver(const int _M, const int _N, const int _P);

  void fctForward(T *v);

  void fctBackward(T *v);

  ~cufctSolver();
};

template class cufctSolver<float>;

template class cufctSolver<double>;
