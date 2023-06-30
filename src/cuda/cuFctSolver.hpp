#pragma once

#include <math.h>
#include <cuda/std/complex>
#include <cufft.h>
#include "math_constants.h"
#include "cuda_utils.hpp"

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE             32

template <typename T>
__global__ void fctForward(T *out_hat, T const *in, T *realBuffer, cuda::std::complex<T> *compBuffer, cufftHandle plan, const int M, const int N, const int P);

template <typename T>
__global__ void fctBackward(T *out, T const *in_hat, T *realBuffer, cuda::std::complex<T> *compBuffer, cufftHandle plan, const int M, const int N, const int P);

#include "cuFctSolver.tpp"
