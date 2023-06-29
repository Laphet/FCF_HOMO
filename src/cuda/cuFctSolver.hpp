#pragma once

#include <math.h>

#include <cuda/std/complex>
#include "cuda_runtime.h"
#include "math_constants.h"

#define MAX_THREADS_PER_BLOCK   1024
#define FCT_POST_STENCIL_WIDTH  2
#define IFCT_PRE_STENCIL_WIDTH  4
#define IFCT_POST_STENCIL_WIDTH 4
#define WARP_SIZE               32

__device__ int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P)
{
  return i * N * P + j * P + k;
}

/*
    Note that P may not be a 32x integer, which my cause warp divergences.
    Hence this routine is designed to make every 32 threads operate the
    same i and j.
*/
__device__ void get3dIdxFromThreadIdx(int &i, int &j, int &k, const int glbThreadIdx, const int N, const int P, const int P_mod)
{
  i = glbThreadIdx / (P_mod * N);
  j = (glbThreadIdx / P_mod) % N;
  k = (glbThreadIdx % P_mod) % P;
}

__device__ int getIdxFrom3dIdxHalf(const int i_p, const int j_p, const int k, const int N, const int P)
{
  return i_p * (N / 2 + 1) * P + j_p * P + k;
}

template <typename T>
__device__ T getPi()
{
  if (sizeof(T) == sizeof(float)) return CUDART_PI_F;
  if (sizeof(T) == sizeof(double)) return CUDART_PI;
}

template float getPi<float>();

template double getPi<double>();

template <typename T>
__device__ cuda::std::complex<T> getExpItheta(const T theta)
{
  cuda::std::complex<T> r(static_cast<T>(0.0), static_cast<T>(0.0));
  if (sizeof(T) == sizeof(float)) {
    r.real(cosf(theta));
    r.imag(sinf(theta));
  }
  if (sizeof(T) == sizeof(double)) {
    r.real(cos(theta));
    r.imag(sin(theta));
  }
  return r;
}

template cuda::std::complex<float> getExpItheta<float>(const float theta);

template cuda::std::complex<double> getExpItheta<double>(const double theta);

template <typename T>
__global__ void fctPre(T *out, T const *in, const int M, const int N, const int P);

template <typename T>
__global__ void fctPost(T *out_hat, cuda::std::complex<T> const *in_hat, const int M, const int N, const int P);

template <typename T>
__global__ void ifctPre(cuda::std::complex<T> *out_hat, T const *in_hat, const int M, const int N, const int P);

template <typename T>
__global__ void ifctPost(T *out, T const *in, const int M, const int N, const int P);
