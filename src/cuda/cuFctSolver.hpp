#pragma once

#include <math.h>
#include <iostream>
#include <cuda/std/complex>
#include <cufft.h>
#include <math_constants.h>

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE             32
#define DIM                   3

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const *const func, char const *const file, int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *const file, int const line);

template <typename T>
class cuFctSolver {
  int                    dims[DIM];
  T                     *realBuffer;
  cuda::std::complex<T> *compBuffer;
  cufftHandle            fft_r2c_plan;
  cufftHandle            fft_c2r_plan;

public:
  cuFctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, realBuffer(NULL), compBuffer(NULL), fft_r2c_plan(0), fft_c2r_plan(0)
  {
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&realBuffer), sizeof(T) * _M * _N * _P));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&compBuffer), sizeof(cuda::std::complex<T>) * _M * _N * _P));
    // Works on the cufft context.
    CHECK_CUDA_ERROR(cufftCreate(&fft_r2c_plan));
    CHECK_CUDA_ERROR(cufftPlanMany(&fft_r2c_plan, 2, &dims[0], NULL, dims[2], 1, NULL, dims[2], 1, CUFFT_R2C, dims[2]));
    CHECK_CUDA_ERROR(cufftCreate(&fft_c2r_plan));
    CHECK_CUDA_ERROR(cufftPlanMany(&fft_c2r_plan, 2, &dims[0], NULL, dims[2], 1, NULL, dims[2], 1, CUFFT_C2R, dims[2]));
  }

  ~cuFctSolver()
  {
    CHECK_CUDA_ERROR(cufftDestroy(fft_c2r_plan));
    CHECK_CUDA_ERROR(cufftDestroy(fft_r2c_plan));
    CHECK_CUDA_ERROR(cudaFree(compBuffer));
    compBuffer = NULL;
    CHECK_CUDA_ERROR(cudaFree(realBuffer));
    realBuffer = NULL;
  }

  void fctForward(const T *in, T *out_hat);

  void fctBackward(const T *in_hat, T *out_hat);
};
