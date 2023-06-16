#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;
// Identical to typedef.

int main(int argc, char *argv[])
{
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  const int m = 2;
  const int n = 2;
  const int k = 2;
  const int lda = 2;
  const int ldb = 2;
  const int ldc = 2;
  /*
   *   A = | 1.0 | 2.0 |
   *       | 3.0 | 4.0 |
   *
   *   B = | 5.0 | 6.0 |
   *       | 7.0 | 8.0 |
   */
  const std::vector<double> A = {1.0, 3.0, 2.0, 4.0};
  const std::vector<data_type> B = {5.0, 7.0, 6.0, 8.0};
  std::vector<data_type> C(m * n);
  const data_type alpha = 1.0;

  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  printf("A\n");
  print_matrix(m, k, A.data(), lda);
  printf("=====\n");

  printf("B\n");
  print_matrix(k, n, B.data(), ldb);
  printf("=====\n");

  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C),
                        sizeof(data_type) * C.size()));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  CUBLAS_CHECK(cublasDtrmm(cublasH, side, uplo, transa, diag, m, n, &alpha, d_A,
                           lda, d_B, ldb, d_C, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   *   C = | 19.0 | 22.0 |
   *       | 28.0 | 32.0 |
   */

  printf("C\n");
  print_matrix(m, n, C.data(), ldc);
  printf("=====\n");

  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}