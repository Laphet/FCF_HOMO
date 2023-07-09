#include "cuda-fct-solver.hpp"

#define MAX_THREADS_PER_BLOCK  1024
#define WARP_SIZE              32
#define FCT_POST_STENCIL_WIDTH 2
#define IFCT_PRE_STENCIL_WIDTH 4

template <typename T>
void check(T err, char const *const func, char const *const file, int const line)
{
  auto status = static_cast<cudaError_t>(err);
  if (status != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(status) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void checkLast(char const *const file, int const line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__device__ int getIdxFrom3dIdx_d(const int i, const int j, const int k, const int N, const int P)
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

__device__ cuComplex getExpItheta(const float theta)
{
  return make_cuComplex(cosf(theta), sin(theta));
}

__device__ cuDoubleComplex getExpItheta(const double theta)
{
  return make_cuDoubleComplex(cos(theta), sin(theta));
}

__device__ cuComplex cuConjWraper(cuComplex cVar)
{
  return cuConjf(cVar);
}

__device__ cuDoubleComplex cuConjWraper(cuDoubleComplex cVar)
{
  return cuConj(cVar);
}

__device__ cuComplex cuMult(cuComplex cVar1, cuComplex cVar2)
{
  return cuCmulf(cVar1, cVar2);
}

__device__ cuDoubleComplex cuMult(cuDoubleComplex cVar1, cuDoubleComplex cVar2)
{
  return cuCmul(cVar1, cVar2);
}

template <typename T>
__global__ void fctPre(T *out, T const *in, const int M, const int N, const int P)
{
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, P_mod);

    if (i < (M + 1) / 2 && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx_d(2 * i, 2 * j, k, N, P);
    if ((M + 1) / 2 <= i && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx_d(2 * M - 2 * i - 1, 2 * j, k, N, P);
    if (i < (M + 1) / 2 && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx_d(2 * i, 2 * N - 2 * j - 1, k, N, P);
    if ((M + 1) / 2 <= i && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx_d(2 * M - 2 * i - 1, 2 * N - 2 * j - 1, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * P_mod) {
    idx_tar      = getIdxFrom3dIdx_d(i, j, k, N, P);
    out[idx_tar] = in_buffer[threadIdx.x];
  }
}

template <typename T>
__global__ void fctPost(T *out_hat, decltype(cuTraits<T>::compVar) const *in_hat, const int M, const int N, const int P)
{
  using complex_T = decltype(cuTraits<T>::compVar);
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_hat_buffer[2 * FCT_POST_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  // Cannot use cuda::std::complex<T> here.
  // Avoid bank conflicts, we add a padding to every row here.
  T myHALF{static_cast<T>(0.5)};

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, P_mod);
    if (1 <= i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].y;

      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].y;
    }
    if (0 == i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].y;

      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].y;
    }
    if (1 <= i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].y;

      idx_req                       = getIdxFrom3dIdxHalf(i_p, N - j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].y;
    }
    if (0 == i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].y;

      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].y;
    }
  }
  __syncthreads();

  T         i_theta, j_theta, cuPi{static_cast<T>(M_PI)}, temp0, temp1;
  complex_T ninj_exp, nipj_exp, tempBuff0, tempBuff1;

  if (glbThreadIdx < M * N * P_mod) {
    i_theta  = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta  = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    ninj_exp = getExpItheta(-i_theta - j_theta);
    nipj_exp = getExpItheta(-i_theta + j_theta);
    idx_tar  = getIdxFrom3dIdx_d(i_p, j_p, k, N, P);

    if (1 <= j_p && j_p <= N / 2) {
      tempBuff0.x      = in_hat_buffer[0][threadIdx.x];
      tempBuff0.y      = in_hat_buffer[1][threadIdx.x];
      temp0            = ninj_exp.x * tempBuff0.x - ninj_exp.y * tempBuff0.y;
      tempBuff1.x      = in_hat_buffer[2][threadIdx.x];
      tempBuff1.y      = -in_hat_buffer[3][threadIdx.x];
      temp1            = nipj_exp.x * tempBuff1.x - nipj_exp.y * tempBuff1.y;
      out_hat[idx_tar] = (temp0 + temp1) * myHALF;
      return;
    }
    if (N / 2 + 1 <= j_p) {
      tempBuff0.x      = in_hat_buffer[0][threadIdx.x];
      tempBuff0.y      = -in_hat_buffer[1][threadIdx.x];
      temp0            = ninj_exp.x * tempBuff0.x - ninj_exp.y * tempBuff0.y;
      tempBuff1.x      = in_hat_buffer[2][threadIdx.x];
      tempBuff1.y      = in_hat_buffer[3][threadIdx.x];
      temp1            = nipj_exp.x * tempBuff1.x - nipj_exp.y * tempBuff1.y;
      out_hat[idx_tar] = (temp0 + temp1) * myHALF;
      return;
    }
    if (0 == j_p) {
      tempBuff0.x      = in_hat_buffer[0][threadIdx.x];
      tempBuff0.y      = in_hat_buffer[1][threadIdx.x];
      out_hat[idx_tar] = ninj_exp.x * tempBuff0.x - ninj_exp.y * tempBuff0.y;
      return;
    }
  } else return;
}

template <typename T>
__global__ void ifctPre(decltype(cuTraits<T>::compVar) *out_hat, T const *in_hat, const int M, const int N, const int P)
{
  using complex_T = decltype(cuTraits<T>::compVar);
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_hat_buffer[IFCT_PRE_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  // Avoid bank conflicts, we add a pad to every row here.

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, P_mod);
    idx_req                       = getIdxFrom3dIdx_d(i_p, j_p, k, N, P);
    in_hat_buffer[0][threadIdx.x] = in_hat[idx_req];
    if (0 < i_p && 0 < j_p) {
      idx_req                       = getIdxFrom3dIdx_d(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdx_d(M - i_p, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req];

      idx_req                       = getIdxFrom3dIdx_d(i_p, N - j_p, k, N, P);
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req];
    }
    if (0 == i_p && 0 < j_p) {
      in_hat_buffer[1][threadIdx.x] = 0;

      in_hat_buffer[2][threadIdx.x] = 0;

      idx_req                       = getIdxFrom3dIdx_d(0, N - j_p, k, N, P);
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req];
    }
    if (0 < i_p && 0 == j_p) {
      in_hat_buffer[1][threadIdx.x] = 0;

      idx_req                       = getIdxFrom3dIdx_d(M - i_p, 0, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req];

      in_hat_buffer[3][threadIdx.x] = 0;
    }
    if (0 == i_p && 0 == j_p) {
      in_hat_buffer[1][threadIdx.x] = 0;

      in_hat_buffer[2][threadIdx.x] = 0;

      in_hat_buffer[3][threadIdx.x] = 0;
    }
  }
  __syncthreads();

  T         i_theta, j_theta, cuPi{static_cast<T>(M_PI)};
  complex_T temp, pipj_exp;

  if (glbThreadIdx < M * N * P_mod && j_p <= N / 2) {
    i_theta          = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta          = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    pipj_exp         = getExpItheta(i_theta + j_theta);
    temp.x           = in_hat_buffer[0][threadIdx.x] - in_hat_buffer[1][threadIdx.x];
    temp.y           = -(in_hat_buffer[2][threadIdx.x] - in_hat_buffer[3][threadIdx.x]);
    idx_tar          = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
    out_hat[idx_tar] = cuMult(pipj_exp, temp);
    return;
  } else return;
}

template <typename T>
__global__ void ifctPost(T *out, T const *in, const int M, const int N, const int P)
{
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, P_mod);

    if (0 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx_d(i / 2, j / 2, k, N, P);
    if (1 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx_d(i / 2, N - (j + 1) / 2, k, N, P);
    if (1 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx_d(M - (i + 1) / 2, j / 2, k, N, P);
    if (1 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx_d(M - (i + 1) / 2, N - (j + 1) / 2, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * P_mod) {
    idx_tar      = getIdxFrom3dIdx_d(i, j, k, N, P);
    out[idx_tar] = in_buffer[threadIdx.x];
  }
}

template <typename T>
cufctSolver<T>::cufctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, realBuffer(nullptr), compBuffer(nullptr), r2cPlan(0), c2rPlan(0)
{
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&realBuffer), sizeof(T) * _M * _N * _P));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&compBuffer), sizeof(cuCompType) * _M * _N * _P));
  // Works on the cufft context.
  CHECK_CUDA_ERROR(cufftCreate(&r2cPlan));
  CHECK_CUDA_ERROR(cufftPlanMany(&r2cPlan, 2, &dims[0], nullptr, dims[2], 1, nullptr, dims[2], 1, cuTraits<T>::r2cType, dims[2]));
  CHECK_CUDA_ERROR(cufftCreate(&c2rPlan));
  CHECK_CUDA_ERROR(cufftPlanMany(&c2rPlan, 2, &dims[0], nullptr, dims[2], 1, nullptr, dims[2], 1, cuTraits<T>::c2rType, dims[2]));
}

cufftResult cufftReal2Comp(cufftHandle plan, float *idata, cuComplex *odata)
{
  return cufftExecR2C(plan, reinterpret_cast<cufftReal *>(idata), reinterpret_cast<cufftComplex *>(odata));
}

cufftResult cufftReal2Comp(cufftHandle plan, double *idata, cuDoubleComplex *odata)
{
  return cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal *>(idata), reinterpret_cast<cufftDoubleComplex *>(odata));
}

template <typename T>
void cufctSolver<T>::fctForward(T *v)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int blockSize{0};   // The launch configurator returned block size
  int minGridSize{0}; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSize{0};    // The actual grid size needed, based on input size
  int P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &fctPre<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  fctPre<T><<<gridSize, blockSize>>>(&realBuffer[0], &v[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftReal2Comp(r2cPlan, &realBuffer[0], &compBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &fctPost<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  fctPost<T><<<gridSize, blockSize>>>(&v[0], &compBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();
}

cufftResult cufftComp2Real(cufftHandle plan, cuComplex *idata, float *odata)
{
  return cufftExecC2R(plan, reinterpret_cast<cufftComplex *>(idata), reinterpret_cast<cufftReal *>(odata));
}

cufftResult cufftComp2Real(cufftHandle plan, cuDoubleComplex *idata, double *odata)
{
  return cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex *>(idata), reinterpret_cast<cufftDoubleReal *>(odata));
}

template <typename T>
void cufctSolver<T>::fctBackward(T *v)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int blockSize{0};   // The launch configurator returned block size
  int minGridSize{0}; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSize{0};    // The actual grid size needed, based on input size
  int P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &ifctPre<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  ifctPre<T><<<gridSize, blockSize>>>(&compBuffer[0], &v[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftComp2Real(c2rPlan, &compBuffer[0], &realBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &ifctPost<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  ifctPost<T><<<gridSize, blockSize>>>(&v[0], &realBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();
}

template <typename T>
cufctSolver<T>::~cufctSolver()
{
  CHECK_CUDA_ERROR(cufftDestroy(c2rPlan));
  CHECK_CUDA_ERROR(cufftDestroy(r2cPlan));
  CHECK_CUDA_ERROR(cudaFree(compBuffer));
  compBuffer = nullptr;
  CHECK_CUDA_ERROR(cudaFree(realBuffer));
  realBuffer = nullptr;
}
