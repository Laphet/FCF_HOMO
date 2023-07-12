#include "cuda-fct-solver.hpp"

#define MAX_THREADS_PER_BLOCK  1024
#define WARP_SIZE              32
#define FCT_POST_STENCIL_WIDTH 2
#define IFCT_PRE_STENCIL_WIDTH 4

void check(cudaError_t status, char const *const func, char const *const file, int const line)
{
  if (status != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(status) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void check(cufftResult status, char const *const func, char const *const file, int const line)
{
  if (status != CUFFT_SUCCESS) {
    std::cerr << "cufft Error at: " << file << ":" << line << std::endl;
    std::cerr << "with code (" << status << ") " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void check(cusparseStatus_t status, char const *const func, char const *const file, int const line)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "cusparse Error at: " << file << ":" << line << std::endl;
    std::cerr << cusparseGetErrorString(status) << " " << func << std::endl;
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
  return i * N * P + (j * P) + k;
}

__device__ int getIdxFrom3dIdxHalf(const int i, const int j, const int k, const int N, const int P)
{
  return i * (N / 2 + 1) * P + (j * P) + k;
}

/*
    Note that P may not be a 32x integer, which my cause warp divergences.
    Hence this routine is designed to make every 32 threads operate the
    same i and j.
*/

__host__ __device__ int getPmod(const int P)
{
  return (P / WARP_SIZE + 1) * WARP_SIZE;
}

__device__ void get3dIdxFromThreadIdx(int &i, int &j, int &k, const int glbThreadIdx, const int N, const int P, const int Pmod)
{
  i = glbThreadIdx / (N * Pmod);
  j = (glbThreadIdx / Pmod) % N;
  k = (glbThreadIdx % Pmod) % P;
}

__device__ cuComplex getExpItheta(const float theta)
{
  return make_cuComplex(cosf(theta), sinf(theta));
}

__device__ cuDoubleComplex getExpItheta(const double theta)
{
  return make_cuDoubleComplex(cos(theta), sin(theta));
}

__device__ cuComplex cuCmul(cuComplex cVar1, cuComplex cVar2)
{
  return cuCmulf(cVar1, cVar2);
}

template <typename T>
__global__ void fctPre(T *out, const T *in, const int M, const int N, const int P)
{
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0};
  int          Pmod{getPmod(P)};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * Pmod) {
    int idx_req{0};
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, Pmod);

    if (i < (M + 1) / 2 && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx_d(2 * i, 2 * j, k, N, P);
    if ((M + 1) / 2 <= i && j < (N + 1) / 2) idx_req = getIdxFrom3dIdx_d(2 * M - 2 * i - 1, 2 * j, k, N, P);
    if (i < (M + 1) / 2 && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx_d(2 * i, 2 * N - 2 * j - 1, k, N, P);
    if ((M + 1) / 2 <= i && (N + 1) / 2 <= j) idx_req = getIdxFrom3dIdx_d(2 * M - 2 * i - 1, 2 * N - 2 * j - 1, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * Pmod) {
    // get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, Pmod);
    int idx_tar{getIdxFrom3dIdx_d(i, j, k, N, P)};
    out[idx_tar] = in_buffer[threadIdx.x];
  }
}

template <typename T>
__global__ void fctPost(T *out_hat, const decltype(cuTraits<T>::compVar) *in_hat, const int M, const int N, const int P)
{
  using complex_T = decltype(cuTraits<T>::compVar);
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0};
  int          Pmod{getPmod(P)};
  __shared__ T in_hat_buffer[2 * FCT_POST_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  // Cannot use cuda::std::complex<T> here.
  // Avoid bank conflicts, we add a padding to every row here.

  if (glbThreadIdx < M * N * Pmod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, Pmod);
    int idx_req{0};
    if (1 <= i_p && j_p < N / 2 + 1) {
      idx_req                       = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].y;

      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].x;
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].y;
    }
    if (0 == i_p && j_p < N / 2 + 1) {
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

  if (glbThreadIdx < M * N * Pmod) {
    // get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, Pmod);
    int       idx_tar{getIdxFrom3dIdx_d(i_p, j_p, k, N, P)};
    T         i_theta, j_theta, cuPi{static_cast<T>(M_PI)}, temp0, temp1;
    complex_T ninj_exp, nipj_exp, tempBuff0, tempBuff1;
    i_theta  = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta  = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    ninj_exp = getExpItheta(-i_theta - j_theta);
    nipj_exp = getExpItheta(-i_theta + j_theta);

    if (1 <= j_p && j_p < N / 2 + 1) {
      tempBuff0.x      = in_hat_buffer[0][threadIdx.x];
      tempBuff0.y      = in_hat_buffer[1][threadIdx.x];
      temp0            = ninj_exp.x * tempBuff0.x - ninj_exp.y * tempBuff0.y;
      tempBuff1.x      = in_hat_buffer[2][threadIdx.x];
      tempBuff1.y      = -in_hat_buffer[3][threadIdx.x];
      temp1            = nipj_exp.x * tempBuff1.x - nipj_exp.y * tempBuff1.y;
      out_hat[idx_tar] = (temp0 + temp1) / 2;
      return;
    }
    if (N / 2 + 1 <= j_p) {
      tempBuff0.x      = in_hat_buffer[0][threadIdx.x];
      tempBuff0.y      = -in_hat_buffer[1][threadIdx.x];
      temp0            = ninj_exp.x * tempBuff0.x - ninj_exp.y * tempBuff0.y;
      tempBuff1.x      = in_hat_buffer[2][threadIdx.x];
      tempBuff1.y      = in_hat_buffer[3][threadIdx.x];
      temp1            = nipj_exp.x * tempBuff1.x - nipj_exp.y * tempBuff1.y;
      out_hat[idx_tar] = (temp0 + temp1) / 2;
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
__global__ void ifctPre(decltype(cuTraits<T>::compVar) *out_hat, const T *in_hat, const int M, const int N, const int P)
{
  using complex_T = decltype(cuTraits<T>::compVar);
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0};
  int          Pmod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_hat_buffer[IFCT_PRE_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  /* Avoid bank conflicts, we add a pad to every row here. */

  if (glbThreadIdx < M * N * Pmod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, Pmod);
    int idx_req{getIdxFrom3dIdx_d(i_p, j_p, k, N, P)};
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

  if (glbThreadIdx < M * N * Pmod && j_p < N / 2 + 1) {
    T         i_theta, j_theta, cuPi{static_cast<T>(M_PI)};
    complex_T temp, pipj_exp;
    i_theta  = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta  = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    pipj_exp = getExpItheta(i_theta + j_theta);
    temp.x   = in_hat_buffer[0][threadIdx.x] - in_hat_buffer[1][threadIdx.x];
    temp.y   = -(in_hat_buffer[2][threadIdx.x] + in_hat_buffer[3][threadIdx.x]);
    int idx_tar{getIdxFrom3dIdxHalf(i_p, j_p, k, N, P)};
    out_hat[idx_tar] = cuCmul(pipj_exp, temp);
    return;
  } else return;
}

template <typename T>
__global__ void ifctPost(T *out, const T *in, const int M, const int N, const int P)
{
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i{0}, j{0}, k{0};
  int          Pmod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_buffer[MAX_THREADS_PER_BLOCK];

  if (glbThreadIdx < M * N * Pmod) {
    get3dIdxFromThreadIdx(i, j, k, glbThreadIdx, N, P, Pmod);
    int idx_req{0};
    if (0 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx_d(i / 2, j / 2, k, N, P);
    if (0 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx_d(i / 2, N - (j + 1) / 2, k, N, P);
    if (1 == i % 2 && 0 == j % 2) idx_req = getIdxFrom3dIdx_d(M - (i + 1) / 2, j / 2, k, N, P);
    if (1 == i % 2 && 1 == j % 2) idx_req = getIdxFrom3dIdx_d(M - (i + 1) / 2, N - (j + 1) / 2, k, N, P);

    in_buffer[threadIdx.x] = in[idx_req];
  }
  __syncthreads();

  if (glbThreadIdx < M * N * Pmod) {
    int     idx_tar{getIdxFrom3dIdx_d(i, j, k, N, P)};
    const T scalFactor{static_cast<T>(1) / (M * N)};
    /* cuFFT performs un-normalized FFTs! */
    out[idx_tar] = in_buffer[threadIdx.x] * scalFactor;
  }
}

template <typename T>
cufctSolver<T>::cufctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, realBuffer{nullptr}, compBuffer{nullptr}, dlPtr{nullptr}, dPtr{nullptr}, duPtr{nullptr}, tridSolverBuffer{nullptr}
{
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&realBuffer), sizeof(T) * _M * _N * _P));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&compBuffer), sizeof(cuCompType) * _M * (_N / 2 + 1) * _P));
  // CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&compBuffer), sizeof(cuCompType) * _M * _N * _P));
  /* Due to the C2R transformation, we could use less memory for complex vectors. */
  /* Works on the cufft context. */
  int dimsHalf[3]{_M, _N / 2 + 1, _P};

  CHECK_CUDA_ERROR(cufftCreate(&r2cPlan));
  CHECK_CUDA_ERROR(cufftPlanMany(&r2cPlan, 2, &dims[0], &dims[0], _P, 1, &dimsHalf[0], _P, 1, cuTraits<T>::r2cType, _P));
  /* It is strange that cufft does not explain why need dimsHalf here. */
  /* There are some mismatch between fftw and cufft. */
  CHECK_CUDA_ERROR(cufftCreate(&c2rPlan));
  CHECK_CUDA_ERROR(cufftPlanMany(&c2rPlan, 2, &dims[0], &dimsHalf[0], _P, 1, &dims[0], _P, 1, cuTraits<T>::c2rType, _P));
  // CHECK_CUDA_ERROR(cufftCreate(&c2cPlan));
  // CHECK_CUDA_ERROR(cufftPlanMany(&c2rPlan, 2, &dims[0], &dims[0], dims[2], 1, &dims[0], dims[2], 1, cuTraits<T>::c2cType, dims[2]));

  /* Works on the cusparse context. */
  CHECK_CUDA_ERROR(cusparseCreate(&cusprHandle));
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
void viewRealVec(std::vector<T> &vec)
{
  for (int i{0}; i < vec.size(); ++i) std::cout << '[' << i << "]=" << vec[i] << ' ';
  std::cout << '\n';
}

template <typename T>
void viewCompVec(std::vector<decltype(cuTraits<T>::compVar)> &vec)
{
  for (int i{0}; i < vec.size(); ++i) {
    std::cout << '[' << i << "]=" << vec[i].x;
    if (vec[i].y >= 0) std::cout << '+';
    std::cout << vec[i].y << "I\n";
  }
  std::cout << '\n';
}

template <typename T>
void cufctSolver<T>::fctForward(T *v)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int blockSize{0};   // The launch configurator returned block size
  int minGridSize{0}; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSize{0};    // The actual grid size needed, based on input size
  int Pmod{getPmod(P)};

  // std::vector<T>                              reViewer(M * N * P);
  // std::vector<decltype(cuTraits<T>::compVar)> cmViewer(M * (N / 2 + 1) * P);
  // std::cout << std::fixed << std::showpoint << std::setprecision(3);

  // CHECK_CUDA_ERROR(cudaMemcpy(&reViewer[0], &v[0], M * N * P * sizeof(T), cudaMemcpyDeviceToHost));
  // viewRealVec(reViewer);

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &fctPre<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * Pmod + blockSize - 1) / blockSize;
  fctPre<T><<<gridSize, blockSize>>>(&realBuffer[0], &v[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  // CHECK_CUDA_ERROR(cudaMemcpy(&reViewer[0], &realBuffer[0], M * N * P * sizeof(T), cudaMemcpyDeviceToHost));
  // viewRealVec(reViewer);

  CHECK_CUDA_ERROR(cufftReal2Comp(r2cPlan, &realBuffer[0], &compBuffer[0]));

  // CHECK_CUDA_ERROR(cudaMemcpy(&cmViewer[0], &compBuffer[0], M * (N / 2 + 1) * P * sizeof(cuCompType), cudaMemcpyDeviceToHost));
  // viewCompVec<T>(cmViewer);

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &fctPost<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * Pmod + blockSize - 1) / blockSize;
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
  int Pmod{(P / WARP_SIZE + 1) * WARP_SIZE};

  // std::vector<T>                              reViewer(M * N * P);
  // std::vector<decltype(cuTraits<T>::compVar)> cmViewer(M * (N / 2 + 1) * P);
  // std::cout << std::fixed << std::showpoint << std::setprecision(3);

  // CHECK_CUDA_ERROR(cudaMemcpy(&reViewer[0], &v[0], M * N * P * sizeof(T), cudaMemcpyDeviceToHost));
  // viewRealVec(reViewer);

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &ifctPre<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * Pmod + blockSize - 1) / blockSize;
  ifctPre<T><<<gridSize, blockSize>>>(&compBuffer[0], &v[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  // CHECK_CUDA_ERROR(cudaMemcpy(&cmViewer[0], &compBuffer[0], M * (N / 2 + 1) * P * sizeof(cuCompType), cudaMemcpyDeviceToHost));
  // viewCompVec<T>(cmViewer);

  CHECK_CUDA_ERROR(cufftComp2Real(c2rPlan, &compBuffer[0], &realBuffer[0]));

  // CHECK_CUDA_ERROR(cudaMemcpy(&reViewer[0], &realBuffer[0], M * N * P * sizeof(T), cudaMemcpyDeviceToHost));
  // viewRealVec(reViewer);

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &ifctPost<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * Pmod + blockSize - 1) / blockSize;
  ifctPost<T><<<gridSize, blockSize>>>(&v[0], &realBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  // CHECK_CUDA_ERROR(cudaMemcpy(&reViewer[0], &v[0], M * N * P * sizeof(T), cudaMemcpyDeviceToHost));
  // viewRealVec(reViewer);
}

void gtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float *dl, const float *d, const float *du, const float *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
  CHECK_CUDA_ERROR(cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
}

void gtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double *dl, const double *d, const double *du, const double *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
  CHECK_CUDA_ERROR(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
}

template <typename T>
void cufctSolver<T>::setTridSolverData(T *dl, T *d, T *du)
{
  if (dlPtr != nullptr || dPtr != nullptr || duPtr != nullptr) std::cerr << "The internal data have been initialized, be careful!\n";
  size_t size = dims[0] * dims[1] * dims[2];

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&dlPtr), size * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(dlPtr), reinterpret_cast<void *>(dl), size * sizeof(T), cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&dPtr), size * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(dPtr), reinterpret_cast<void *>(d), size * sizeof(T), cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&duPtr), size * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(duPtr), reinterpret_cast<void *>(du), size * sizeof(T), cudaMemcpyHostToDevice));

  size_t bufferSizeInBytes{0};
  int    M{dims[0]}, N{dims[1]}, P{dims[2]};
  gtsv2StridedBatch_bufferSizeExt(cusprHandle, P, dlPtr, dPtr, duPtr, realBuffer, M * N, P, &bufferSizeInBytes);
  if (tridSolverBuffer != nullptr) std::cerr << "The internal data have been initialized, be careful!\n";
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&tridSolverBuffer), bufferSizeInBytes));
}

void gtsv2StridedBatch(cusparseHandle_t handle, int m, const float *dl, const float *d, const float *du, float *x, int batchCount, int batchStride, void *pBuffer)
{
  CHECK_CUDA_ERROR(cusparseSgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

void gtsv2StridedBatch(cusparseHandle_t handle, int m, const double *dl, const double *d, const double *du, double *x, int batchCount, int batchStride, void *pBuffer)
{
  CHECK_CUDA_ERROR(cusparseDgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

template <typename T>
void cufctSolver<T>::precondSolver(T *rhs)
{
  if (dlPtr == nullptr || dPtr == nullptr || duPtr == nullptr || tridSolverBuffer == nullptr) {
    std::cerr << "The internal data have not been initialized!\n";
    std::cerr << "There will be nothing to do in this routine.\n";
    return;
  }

  fctForward(rhs);

  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  gtsv2StridedBatch(cusprHandle, P, dlPtr, dPtr, duPtr, rhs, M * N, P, tridSolverBuffer);

  fctBackward(rhs);
}

template <typename T>
void cuFreeMod(T *&ptr)
{
  if (ptr != nullptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
    ptr = nullptr;
  }
}

template <typename T>
cufctSolver<T>::~cufctSolver()
{
  cuFreeMod(tridSolverBuffer);
  cuFreeMod(duPtr);
  cuFreeMod(dPtr);
  cuFreeMod(dlPtr);

  CHECK_CUDA_ERROR(cufftDestroy(c2rPlan));
  CHECK_CUDA_ERROR(cufftDestroy(r2cPlan));

  cuFreeMod(compBuffer);
  cuFreeMod(realBuffer);
}
