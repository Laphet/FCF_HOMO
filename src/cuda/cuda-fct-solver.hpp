#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusparse.h>
#include <iostream>
#include <vector>

// static constexpr int DIM{3};

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t status, char const *const func, char const *const file, int const line);
void check(cufftResult status, char const *const func, char const *const file, int const line);
void check(cusparseStatus_t status, char const *const func, char const *const file, int const line);
void check(cublasStatus_t status, char const *const func, char const *const file, int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *const file, int const line);

class gpuTimer {
  cudaEvent_t start_;
  cudaEvent_t stop_;

public:
  gpuTimer();

  ~gpuTimer();

  void start();

  void stop();

  float elapsed();
};

template <typename T>
struct cuTraits;

template <>
struct cuTraits<float> {
  static const cufftType_t  r2cType{CUFFT_R2C};
  static const cufftType_t  c2rType{CUFFT_C2R};
  static const cufftType_t  c2cType{CUFFT_C2C};
  static const cudaDataType valueType{CUDA_R_32F};
  static cuComplex          compVar;
};

template <>
struct cuTraits<double> {
  static const cufftType_t  r2cType{CUFFT_D2Z};
  static const cufftType_t  c2rType{CUFFT_Z2D};
  static const cufftType_t  c2cType{CUFFT_Z2Z};
  static const cudaDataType valueType{CUDA_R_64F};
  static cuDoubleComplex    compVar;
};

template <typename T>
struct dnVec {
  cusparseDnVecDescr_t descr;
  T                   *ptr;
};

template <typename T>
struct spMat {
  cusparseSpMatDescr_t descr;
  int                 *rowOffsetsPtr;
  int                 *colIndPtr;
  T                   *valuesPtr;
};

template <typename T>
class cufctSolver {
  using cuCompType = decltype(cuTraits<T>::compVar);
  int              dims[3];
  T               *realBuffer;
  cuCompType      *compBuffer;
  cufftHandle      r2cPlan;
  cufftHandle      c2rPlan;
  cusparseHandle_t sprHandle;
  T               *dlPtr;
  T               *dPtr;
  T               *duPtr;
  void            *tridSolverBuffer;
  spMat<T>         csrMat;
  size_t           nnz;
  cublasHandle_t   blasHandle;

public:
  cufctSolver(const int _M, const int _N, const int _P);

  void fctForward(T *v); // v is in the device memory.

  void fctBackward(T *v); // v is in the device memory.

  void setTridSolverData(T *dl, T *d, T *du); // dl, d and du are in the host memory.

  void precondSolver(T *rhs); // rhs is in the device memory.

  void setSprMatData(int *csrRowOffsets, int *csrColInd, T *csrValues); // All vectors are in the host memory.

  void solve(T *u, const T *b, const int maxIter = 1024, const T rtol = 1.0e-5, const T atol = 1.0e-8);

  void solveWithoutPrecond(T *u, const T *b, const int maxIter = 1024, const T rtol = 1.0e-5, const T atol = 1.0e-8);

  void solveWithSsor(T *u, const T *b, T *ssorValues, const int maxIter = 1024, const T rtol = 1.0e-5, const T atol = 1.0e-8);

  void solveWithICC(T *u, const T *b, const int maxIter = 1024, const T rtol = 1.0e-5, const T atol = 1.0e-8);

  ~cufctSolver();
};
