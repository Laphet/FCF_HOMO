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
struct cuTraits;

template <>
struct cuTraits<float> {
  static const cufftType_t r2cType{CUFFT_R2C};
  static const cufftType_t c2rType{CUFFT_C2R};
  static cufftResult       cufftReal2Comp(cufftHandle plan, float *idata, cuda::std::complex<float> *odata) { return cufftExecR2C(plan, reinterpret_cast<cufftReal *>(idata), reinterpret_cast<cufftComplex *>(odata)); }
  static cufftResult       cufftComp2Real(cufftHandle plan, cuda::std::complex<float> *idata, float *odata) { return cufftExecC2R(plan, reinterpret_cast<cufftComplex *>(idata), reinterpret_cast<cufftReal *>(odata)); }
};

template <>
struct cuTraits<double> {
  static const cufftType_t r2cType{CUFFT_D2Z};
  static const cufftType_t c2rType{CUFFT_Z2D};
  static cufftResult       cufftReal2Comp(cufftHandle plan, double *idata, cuda::std::complex<double> *odata) { return cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal *>(idata), reinterpret_cast<cufftDoubleComplex *>(odata)); }
  static cufftResult       cufftComp2Real(cufftHandle plan, cuda::std::complex<double> *idata, double *odata) { return cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex *>(idata), reinterpret_cast<cufftDoubleReal *>(odata)); }
};

template <typename T>
class cufctSolver {
  int                    dims[DIM];
  T                     *realBuffer;
  cuda::std::complex<T> *compBuffer;
  cufftHandle            r2cPlan;
  cufftHandle            c2rPlan;

public:
  cufctSolver(const int _M, const int _N, const int _P);

  void fctForward(const T *in, T *out_hat);

  void fctBackward(const T *in_hat, T *out);

  ~cufctSolver();
};

template class cufctSolver<float>;

template class cufctSolver<double>;
