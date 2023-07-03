#pragma once

#include <math.h>
#include <cuda/std/complex>
#include <cufft.h>
#include <thrust/device_vector.h>
#include "math_constants.h"
#include "cuda_utils.hpp"

constexpr int MAX_THREADS_PER_BLOCK{1024};
constexpr int WARP_SIZE{32};

template <typename T>
class cuFctSolver {
  using complex_t = cuda::std::complex<T>;
  const std::size_t                M, N, P;
  thrust::device_vector<T>         realBuffer;
  thrust::device_vector<complex_t> compBuffer;
  cufftHandle                      fft_plan;

public:
  __global__ cuFctSolver(const std::size_t _M, const std::size_t _N, const std::size_t _P) : M(_M), N(_N), P(_P), realBuffer(_M * _N * _P), compBuffer(_M * _N * _P), fft_plan(0)
  {
    // Works on the cufft context.
    CHECK_CUDA_ERROR(cufftCreate(&fft_plan));
    CHECK_CUDA_ERROR();
  }

  __global__ void fctForward(const thrust::device_vector<T> &in, thrust::device_vector<T> &out_hat);

  __global__ void fctBackward(const thrust::device_vector<T> &in_hat, thrust::device_vector<T> &out_hat);
};

#include "cuFctSolver.tpp"
