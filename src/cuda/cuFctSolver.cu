#include "cuFctSolver.hpp"

template <typename T>
__global__ void fctPre(T *out, T const *in,
                       const int M, const int N, const int P)
{
  int idx{blockIdx.x * blockDim.x + threadIdx.x};
  int i{0}, j{0}, k{0}, buffer_head_idx{0}, idx_req{0};
  __shared__ T in_buffer[FCT_PRE_STENCIL_WIDTH]
                        [MAX_THREADS_PER_BLOCK +
                         PADDING_WIDTH_AVOID_BANK_CONFLICTS];

  if (idx < M * N * P)
  {
    get3dIdxFromIdx(i, j, k, idx, N, P);
    if (i < (M + 1) / 2 && j < (N + 1) / 2)
    {
      idx_req = getIdxFrom3dIdx(2 * i, 2 * j, k, N, P);
      buffer_head_idx = 0;
    }
    else if ((M + 1) / 2 <= i && j < (N + 1) / 2)
    {
      idx_req = getIdxFrom3dIdx(2 * M - 2 * i - 1, 2 * j, k, N, P);
      buffer_head_idx = 1;
    }
    else if (i < (M + 1) / 2 && (N + 1) / 2 <= j)
    {
      idx_req = getIdxFrom3dIdx(2 * i, 2 * N - 2 * j - 1, k, N, P);
      buffer_head_idx = 2;
    }
    else
    {
      idx_req = getIdxFrom3dIdx(2 * M - 2 * i - 1, 2 * N - 2 * j - 1,
                                k, N, P);
      buffer_head_idx = 3;
    }
    in_buffer[buffer_head_idx][threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (idx < M * N * P)
  {
    out[idx] = in_buffer[buffer_head_idx][threadIdx.x];
  }
}

template <typename T>
__global__ void fctPost(T *out_hat, cuda::std::complex<T> const *in_hat,
                        const int M, const int N, const int P)
{
  using complex_t = cuda::std::complex<T>;
  int idx{blockIdx.x * blockDim.x + threadIdx.x};
  int i_p{0}, j_p{0}, k{0}, idx_req{0};
  __shared__ complex_t in_hat_buffer[FCT_POST_STENCIL_WIDTH]
                                    [MAX_THREADS_PER_BLOCK +
                                     PADDING_WIDTH_AVOID_BANK_CONFLICTS];

  if (idx < M * N * P)
  {
    get3dIdxFromIdx(i_p, j_p, k, idx, N, P);
    if (j_p <= N / 2)
    {
      idx_req = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];

      idx_req = getIdxFrom3dIdxHalf(M - i_p, j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];
    }
    if (N / 2 + 1 <= j_p)
    {
      idx_req = getIdxFrom3dIdxHalf(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req];

      idx_req = getIdxFrom3dIdxHalf(i_p, N - j_p, k, N, P);
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req];
    }
  }
  __syncthreads();

  T i_theta{static_cast<T>(0.0)}, j_theta{static_cast<T>(0.0)},
      cuPi{getPi<T>()};
  complex_t i_exp, j_exp, temp;

  if (idx < M * N * P)
  {
    i_theta = static_cast<T>(i_p) / static_cast(2 * M) * cuPi;
    j_theta = static_cast<T>(j_p) / static_cast(2 * N) * cuPi;
    i_exp = getExpItheta<T>(i_theta);
    j_exp = getExpItheta<T>(j_theta);
    if (1 <= i_p && 1 <= j_p && j_p <= N / 2)
    {
      temp = cuda::std::conj(j_exp) * in_hat_buffer[0][threadIdx.x];
      temp += j_exp * cuda::std::conj(in_hat_buffer[1][threadIdx.x]);
      temp *= cuda::std::conj(i_exp);
      out_hat[threadIdx.x] = temp.real() * static_cast<T>(0.5);
      return;
    }
    if (0 == i_p && 1 <= j_p && j_p <= N / 2)
    {
      temp = cuda::std::conj(j_exp) * in_hat_buffer[0][threadIdx.x];
      temp += j_exp * cuda::std::conj(in_hat_buffer[3][threadIdx.x]);
      out_hat[threadIdx.x] = temp.real() * static_cast<T>(0.5);
      return;
    }
    if (0 <= i_p && N / 2 + 1 <= j_p)
    {
      temp = cuda::std::conj(j_exp) *
             cuda::std::conj(in_hat_buffer[2][threadIdx.x]);
      temp += i_exp * in_hat_buffer[3][threadIdx.x];
      out_hat[threadIdx.x] = temp.real() * static_cast<T>(0.5);
      return;
    }
    if (0 == j_p)
    {
      temp = cuda::std::conj(i_exp) * in_hat_buffer[0][threadIdx.x];
      out_hat[threadIdx.x] = temp.real();
      return;
    }
  }
  else
    return;
}