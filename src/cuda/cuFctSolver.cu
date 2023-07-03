#include "cuFctSolver.hpp"

#define FCT_POST_STENCIL_WIDTH 2
#define IFCT_PRE_STENCIL_WIDTH 4

__device__ int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P)
{
  return i * N * P + j * P + k;
}

__device__ int getIdxFrom3dIdxHalf(const int i, const int j, const int k, const int N, const int P)
{
  return i * (N / 2 + 1) * P + j * P + k;
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

__device__ float getPi(float t)
{
  return CUDART_PI_F;
}

__device__ double getPi(double t)
{
  return CUDART_PI;
}

__device__ cuda::std::complex<float> getExpItheta(const float theta)
{
  cuda::std::complex<float> r(cosf(theta), sinf(theta));
  return r;
}

__device__ cuda::std::complex<double> getExpItheta(const double theta)
{
  cuda::std::complex<double> r(cos(theta), sin(theta));
  return r;
}

inline cufftResult cufftReal2Comp(cufftHandle plan, float *idata, cuda::std::complex<float> *odata)
{
  return cufftExecR2C(plan, reinterpret_cast<cufftReal *>(idata), reinterpret_cast<cufftComplex *>(odata));
}

inline cufftResult cufftReal2Comp(cufftHandle plan, double *idata, cuda::std::complex<double> *odata)
{
  return cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal *>(idata), reinterpret_cast<cufftDoubleComplex *>(odata));
}

inline cufftResult cufftComp2Real(cufftHandle plan, cuda::std::complex<float> *idata, float *odata)
{
  return cufftExecC2R(plan, reinterpret_cast<cufftComplex *>(idata), reinterpret_cast<cufftReal *>(odata));
}

inline cufftResult cufftComp2Real(cufftHandle plan, cuda::std::complex<double> *idata, double *odata)
{
  return cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex *>(idata), reinterpret_cast<cufftDoubleReal *>(odata));
}

template <typename T>
__global__ void fctPre(T *out, T const *in, const int M, const int N, const int P);

template <typename T>
__global__ void fctPost(T *out_hat, cuda::std::complex<T> const *in_hat, const int M, const int N, const int P);

template <typename T>
__global__ void ifctPre(cuda::std::complex<T> *out_hat, T const *in_hat, const int M, const int N, const int P);

template <typename T>
__global__ void ifctPost(T *out, T const *in, const int M, const int N, const int P);

template <typename T>
__global__ void fctPre(T *out, T const *in, const int M, const int N, const int P)
{
  int          glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, P_mod);

    if (i < (M + 1) / 2 && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx(2 * i, 2 * j, k, N, P);
    if ((M + 1) / 2 <= i && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx(2 * M - 2 * i - 1, 2 * j, k, N, P);
    if (i < (M + 1) / 2 && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx(2 * i, 2 * N - 2 * j - 1, k, N, P);
    if ((M + 1) / 2 <= i && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx(2 * M - 2 * i - 1, 2 * N - 2 * j - 1, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * P_mod) {
    idx_tar      = getIdxFrom3dIdx(i, j, k, N, P);
    out[idx_tar] = in_buffer[threadIdx.x];
  }
}

template <typename T>
__global__ void fctPost(T *out_hat, cuda::std::complex<T> const *in_hat, const int M, const int N, const int P)
{
  using complex_t = cuda::std::complex<T>;
  int                  glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int                  i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int                  P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ complex_t in_hat_buffer[FCT_POST_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  T                    myZERO{static_cast<T>(0.0)}, myHALF{static_cast<T>(0.5)}; // Avoid bank conflicts, we add a padding to every row here.

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, idx, N, P, P_mod);
    if (1 <= i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];
    }
    if (0 == i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];
    }
    if (1 <= i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdxHalf(i_p, N - j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];
    }
    if (0 == i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];
    }
  }
  __syncthreads();

  T         i_theta{myZERO}, j_theta{myZERO}, cuPi{getPi<T>(myZERO)};
  complex_t ninj_exp, nipj_exp, temp;

  if (glbThreadIdx < M * N * P_mod) {
    i_theta  = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta  = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    ninj_exp = getExpItheta<T>(-i_theta - j_theta);
    nipj_exp = getExpItheta<T>(-i_theta + j_theta);
    idx_tar  = getIdxFrom3dIdx(i_p, j_p, k, N, P);

    if (1 <= j_p && j_p <= N / 2) {
      temp = ninj_exp * in_hat_buffer[0][threadIdx.x];
      temp += nipj_exp * cuda::std::conj(in_hat_buffer[1][threadIdx.x]);
      out_hat[idx_tar] = temp.real() * myHALF;
      return;
    }
    if (N / 2 + 1 <= j_p) {
      temp = ninj_exp * cuda::std::conj(in_hat_buffer[0][threadIdx.x]);
      temp += nipj_exp * in_hat_buffer[1][threadIdx.x];
      out_hat[idx_tar] = temp.real() * myHALF;
      return;
    }
    if (0 == j_p) {
      temp             = ninj_exp * in_hat_buffer[0][threadIdx.x];
      out_hat[idx_tar] = temp.real();
      return;
    }
  } else return;
}

template <typename T>
__global__ void ifctPre(cuda::std::complex<T> *out_hat, T const *in_hat, const int M, const int N, const int P)
{
  using complex_t = cuda::std::complex<T>;
  int          glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  T            myZERO{static_cast<T>(0.0)};
  __shared__ T in_hat_buffer[IFCT_PRE_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1]; // Avoid bank conflicts, we add a pad to every row here.

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, idx, N, P, P_mod);
    idx_req                       = getIdxFrom3dIdx(i_p, j_p, k, N, P);
    in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];
    if (0 < i_p && 0 < j_p) {
      idx_req                       = getIdxFrom3dIdx(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdx(M - i_p, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdx(i_p, N - j_p, k, N, P);
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req];
    }
    if (0 == i_p && 0 < j_p) {
      in_hat_buffer[1][threadIdx.x] = myZERO;

      in_hat_buffer[2][threadIdx.x] = myZERO;

      idx_req                       = getIdxFrom3dIdx(0, N - j_p, k, N, P);
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req];
    }
    if (0 < i_p && 0 == j_p) {
      in_hat_buffer[1][threadIdx.x] = myZERO;

      idx_req                       = getIdxFrom3dIdx(M - i_p, 0, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req];

      in_hat_buffer[3][threadIdx.x] = myZERO;
    }
    if (0 == i_p && 0 == j_p) {
      in_hat_buffer[1][threadIdx.x] = myZERO;

      in_hat_buffer[2][threadIdx.x] = myZERO;

      in_hat_buffer[3][threadIdx.x] = myZERO;
    }
  }
  __syncthreads();

  T         i_theta{myZERO}, j_theta{myZERO}, cuPi{getPi<T>(myZERO)};
  complex_t temp;

  if (glbThreadIdx < M * N * P_mod && j_p <= N / 2) {
    i_theta = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;

    temp.real(in_hat_buffer[0][threadIdx.x] - in_hat_buffer[1][threadIdx.x]);
    temp.imag(-(in_hat_buffer[2][threadIdx.x] - in_hat_buffer[3][threadIdx.x]));
    temp *= cuda::std::conj(getExpItheta(i_theta)) * cuda::std::conj(getExpItheta(j_theta));

    idx_tar          = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
    out_hat[idx_tar] = temp;
    return;
  } else return;
}

template <typename T>
__global__ void ifctPost(T *out, T const *in, const int M, const int N, const int P)
{
  int          glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, P_mod);

    if (0 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx(i / 2, j / 2, k, N, P);
    if (1 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx(i / 2, N - (j + 1) / 2, k, N, P);
    if (1 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx(M - (i + 1) / 2, j / 2, k, N, P);
    if (1 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx(M - (i + 1) / 2, N - (j + 1) / 2, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * P_mod) {
    idx_tar      = getIdxFrom3dIdx(i, j, k, N, P);
    out[idx_tar] = in_buffer[threadIdx.x];
  }
}

template <typename T>
__global__ void cuFctSolver<T>::fctForward(const thrust::device_vector<T> &in, thrust::device_vector<T> &out_hat)
{
  int blockSize{0};   // The launch configurator returned block size
  int minGridSize{0}; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSize{0};    // The actual grid size needed, based on input size
  int P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, fctPre, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  fctPre<T><<<gridSize, blockSize>>>(&realBuffer[0], &in[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftReal2Comp(fft_plan, &realBuffer[0], &compBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, fctPost, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  fctPost<T><<<gridSize, blockSize>>>(&out_hat[0], &compBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();
}

template <typename T>
__global__ void cuFctSolver<T>::fctBackward(const thrust::device_vector<T> &in_hat, thrust::device_vector<T> &out_hat)
{
  int blockSize{0};   // The launch configurator returned block size
  int minGridSize{0}; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSize{0};    // The actual grid size needed, based on input size
  int P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ifctPre, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  ifctPre<T><<<gridSize, blockSize>>>(&compBuffer[0], &in_hat[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftComp2Real(fft_plan, &compBuffer[0], &realBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ifctPost, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  ifctPost<T><<<gridSize, blockSize>>>(&out[0], &realBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();
}
