#pragma once

#include <math.h>

#include <cuda/std/complex>
#include "cuda_runtime.h"
#include "math_constants.h"

#define MAX_THREADS_PER_BLOCK 1024
#define FCT_PRE_STENCIL_WIDTH 4;
#define FCT_POST_STENCIL_WIDTH 2;
#define IFCT_PRE_STENCIL_WIDTH 4;
#define IFCT_POST_STENCIL_WIDTH 4;
#define PADDING_WIDTH_AVOID_BANK_CONFLICTS 3;

__device__ int getIdxFrom3dIdx(const int i, const int j, const int k,
                               const int N, const int P)
{
  return i * N * P + j * P + k;
}

__device__ void get3dIdxFromIdx(int &i, int &j, int &k,
                                const int idx, const int N, const int P)
{
  i = idx / (P * N);
  j = (idx / P) % N;
  k = idx % P;
}

__device__ int getIdxFrom3dIdxHalf(const int i_p, const int j_p,
                                   const int k, const int N, const int P)
{
  return i_p * (N / 2 + 1) * P + j_p * P + k;
}

// template <typename T>
// __device__ void complexMult(T &out_re, T &out_im,
//                             const T a_re, const T a_im,
//                             const T b_re, const T b_im)
// {
//   out_re = a_re * b_re - a_im * b_im;
//   out_im = a_re * b_im + a_im * b_re;
// }

// template void complexMult<float>(float &out_re, float &out_im,
//                                  const float a_re, const float a_im,
//                                  const float b_re, const float b_im);

// template void complexMult<double>(double &out_re, double &out_im,
//                                   const double a_re, const double a_im,
//                                   const double b_re, const double b_im);

template <typename T>
__device__ T getPi()
{
  if (sizeof(T) == sizeof(float))
    return CUDART_PI_F;
  if (sizeof(T) == sizeof(double))
    return CUDART_PI;
}

template float getPi<float>();

template double getPi<double>();

template <typename T>
__device__ cuda::std::complex<T> getExpItheta(const T theta)
{
  cuda::std::complex<T> r(static_cast<T>(0.0), static_cast<T>(0.0));
  if (sizeof(T) == sizeof(float))
  {
    r.real(cosf(theta));
    r.imag(sinf(theta));
  }
  if (sizeof(T) == sizeof(double))
  {
    r.real(cos(theta));
    r.imag(sin(theta));
  }
  return r;
}

template cuda::std::complex<float> getExpItheta<float>(const float theta);

template cuda::std::complex<double> getExpItheta<double>(const double theta);

/*
  This routine is for in the R2C and C2R transformations,
  the dimensions will reduce by 2.
  If real input is M*N, the complex output is M*(N/2+1), and vice versa.
  Given a 3d Idx (i, j, k) that 0<=i<M, 0<=j<N, and 0<=K<P,
  We need to map the Idx by different j and utilize the conjugate symmetry.
*/
template <typename T>
__device__ void getValFrom3dIdx(T *val,
                                T *data_ptr,
                                const int i_p, const int j_p, const int k,
                                const int M, const int N, const int P)
{
  int idx{0};
  if (j_p < (N / 2) + 1)
  {
    idx = i_p * (N / 2 + 1) * P + j_p * P + k;
    val[0] = data_ptr[2 * idx];
    val[1] = data_ptr[2 * idx + 1];
    return;
  }
  if (i_p >= 1 && j_p >= (N / 2) + 1)
  {
    idx = (M - i_p) * (N / 2 + 1) * P + (N - j_p) * P + k;
    val[0] = data_ptr[2 * idx];
    val[1] = -data_ptr[2 * idx + 1];
    return;
  }
  if (i_p == 0 && j_p >= (N / 2) + 1)
  {
    idx = (N - j_p) * P + k;
    val[0] = data_ptr[2 * idx];
    val[1] = -data_ptr[2 * idx + 1];
    return;
  }
}

template <typename T>
__global__ void fctPre(T *out, T const *in,
                       const int M, const int N, const int P);

template <typename T>
__global__ void fctPost(T *out_hat, cuda::std::complex<T> const *in_hat,
                        const int M, const int N, const int P);

template <typename T>
__global__ void ifctPre(cuda::std::complex<T> *out_hat, T const *in_hat,
                        const int M, const int N, const int P);

template <typename T>
__global__ void ifctPost(T *out, T const *in,
                         const int M, const int N, const int P);
