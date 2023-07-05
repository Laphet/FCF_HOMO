#include "cuda_fct_solver.hpp"

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

template <typename T>
__device__ T getPi();

template <>
__device__ float getPi<float>()
{
  return CUDART_PI_F;
}

template <>
__device__ double getPi<double>()
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
__global__ void fctPre(T *out, T const *in, const int M, const int N, const int P)
{
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
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
  using complex_T = cuda::std::complex<T>;
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  __shared__ T in_hat_buffer[2 * FCT_POST_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  // Cannot use cuda::std::complex<T> here.
  // Avoid bank conflicts, we add a padding to every row here.
  T myZERO{static_cast<T>(0.0)}, myHALF{static_cast<T>(0.5)};

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, P_mod);
    if (1 <= i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(i_p, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].imag();

      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].imag();
    }
    if (0 == i_p && j_p <= N / 2) {
      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].imag();

      idx_req                       = getIdxFrom3dIdxHalf(0, j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].imag();
    }
    if (1 <= i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(M - i_p, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].imag();

      idx_req                       = getIdxFrom3dIdxHalf(i_p, N - j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].imag();
    }
    if (0 == i_p && N / 2 + 1 <= j_p) {
      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[0][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[1][threadIdx.x] = in_hat[idx_req].imag();

      idx_req                       = getIdxFrom3dIdxHalf(0, N - j_p, k, N, P);
      in_hat_buffer[2][threadIdx.x] = in_hat[idx_req].real();
      in_hat_buffer[3][threadIdx.x] = in_hat[idx_req].imag();
    }
  }
  __syncthreads();

  T         i_theta{myZERO}, j_theta{myZERO}, cuPi{getPi<T>()};
  complex_T ninj_exp, nipj_exp, temp, tempBuff0, tempBuff1;

  if (glbThreadIdx < M * N * P_mod) {
    i_theta  = static_cast<T>(i_p) / static_cast<T>(2 * M) * cuPi;
    j_theta  = static_cast<T>(j_p) / static_cast<T>(2 * N) * cuPi;
    ninj_exp = getExpItheta(-i_theta - j_theta);
    nipj_exp = getExpItheta(-i_theta + j_theta);
    idx_tar  = getIdxFrom3dIdx(i_p, j_p, k, N, P);

    if (1 <= j_p && j_p <= N / 2) {
      tempBuff0.real(in_hat_buffer[0][threadIdx.x]);
      tempBuff0.imag(in_hat_buffer[1][threadIdx.x]);
      temp = ninj_exp * tempBuff0;
      tempBuff1.real(in_hat_buffer[2][threadIdx.x]);
      tempBuff1.imag(in_hat_buffer[3][threadIdx.x]);
      temp += nipj_exp * cuda::std::conj(tempBuff1);
      out_hat[idx_tar] = temp.real() * myHALF;
      return;
    }
    if (N / 2 + 1 <= j_p) {
      tempBuff0.real(in_hat_buffer[0][threadIdx.x]);
      tempBuff0.imag(in_hat_buffer[1][threadIdx.x]);
      temp = ninj_exp * cuda::std::conj(tempBuff0);
      tempBuff1.real(in_hat_buffer[2][threadIdx.x]);
      tempBuff1.imag(in_hat_buffer[3][threadIdx.x]);
      temp += nipj_exp * tempBuff1;
      out_hat[idx_tar] = temp.real() * myHALF;
      return;
    }
    if (0 == j_p) {
      tempBuff0.real(in_hat_buffer[0][threadIdx.x]);
      tempBuff0.imag(in_hat_buffer[1][threadIdx.x]);
      temp             = ninj_exp * tempBuff0;
      out_hat[idx_tar] = temp.real();
      return;
    }
  } else return;
}

template <typename T>
__global__ void ifctPre(cuda::std::complex<T> *out_hat, T const *in_hat, const int M, const int N, const int P)
{
  using complex_T = cuda::std::complex<T>;
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
  int          i_p{0}, j_p{0}, k{0}, idx_req{0}, idx_tar{0};
  int          P_mod{(P / WARP_SIZE + 1) * WARP_SIZE};
  T            myZERO{static_cast<T>(0.0)};
  __shared__ T in_hat_buffer[IFCT_PRE_STENCIL_WIDTH][MAX_THREADS_PER_BLOCK + 1];
  // Avoid bank conflicts, we add a pad to every row here.

  if (glbThreadIdx < M * N * P_mod) {
    get3dIdxFromThreadIdx(i_p, j_p, k, glbThreadIdx, N, P, P_mod);
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

  T         i_theta{myZERO}, j_theta{myZERO}, cuPi{getPi<T>()};
  complex_T temp;

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
  size_t       glbThreadIdx{blockIdx.x * blockDim.x + threadIdx.x};
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
void cufctSolver<T>::fctForward(const T *in, T *out_hat)
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
  fctPre<T><<<gridSize, blockSize>>>(&realBuffer[0], &in[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftReal2Comp(fft_r2c_plan, &realBuffer[0], &compBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &fctPost<T>, 0, 0));
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
void cufctSolver<T>::fctBackward(const T *in_hat, T *out_hat)
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
  ifctPre<T><<<gridSize, blockSize>>>(&compBuffer[0], &in_hat[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cufftComp2Real(fft_c2r_plan, &compBuffer[0], &realBuffer[0]));

  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &ifctPost<T>, 0, 0));
  blockSize = (blockSize / WARP_SIZE) * WARP_SIZE; // This should be useless.
  if (blockSize < P) {
    std::cout << "Recommended blocksize=" << blockSize << " < P=" << P << ", reset blocksize=" << MAX_THREADS_PER_BLOCK << '\n';
    blockSize = MAX_THREADS_PER_BLOCK;
  }
  gridSize = (M * N * P_mod + blockSize - 1) / blockSize;
  ifctPost<T><<<gridSize, blockSize>>>(&out_hat[0], &realBuffer[0], M, N, P);
  CHECK_LAST_CUDA_ERROR();
}

template <typename T>
cufftType_t getR2C_t();

template <>
cufftType_t getR2C_t<float>()
{
  return CUFFT_R2C;
}

template <>
cufftType_t getR2C_t<double>()
{
  return CUFFT_D2Z;
}

template <typename T>
cufftType_t getC2R_t();

template <>
cufftType_t getC2R_t<float>()
{
  return CUFFT_C2R;
}

template <>
cufftType_t getC2R_t<double>()
{
  return CUFFT_Z2D;
}

template <typename T>
cufctSolver<T>::cufctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, realBuffer(nullptr), compBuffer(nullptr), fft_r2c_plan(0), fft_c2r_plan(0)
{
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&realBuffer), sizeof(T) * _M * _N * _P));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&compBuffer), sizeof(cuda::std::complex<T>) * _M * _N * _P));
  // Works on the cufft context.
  CHECK_CUDA_ERROR(cufftCreate(&fft_r2c_plan));
  CHECK_CUDA_ERROR(cufftPlanMany(&fft_r2c_plan, 2, &dims[0], nullptr, dims[2], 1, nullptr, dims[2], 1, getR2C_t<T>(), dims[2]));
  CHECK_CUDA_ERROR(cufftCreate(&fft_c2r_plan));
  CHECK_CUDA_ERROR(cufftPlanMany(&fft_c2r_plan, 2, &dims[0], nullptr, dims[2], 1, nullptr, dims[2], 1, getC2R_t<T>(), dims[2]));
}

template <typename T>
cufctSolver<T>::~cufctSolver()
{
  CHECK_CUDA_ERROR(cufftDestroy(fft_c2r_plan));
  CHECK_CUDA_ERROR(cufftDestroy(fft_r2c_plan));
  CHECK_CUDA_ERROR(cudaFree(compBuffer));
  compBuffer = nullptr;
  CHECK_CUDA_ERROR(cudaFree(realBuffer));
  realBuffer = nullptr;
}

#include "cuda_fct_solver.tpp"
