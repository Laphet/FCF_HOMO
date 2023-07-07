#include "common.hpp"
// #include "cuda-fct-solver.hpp"
#include "cpu-fct-solver.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
  using T = double;
  int                 M{3}, N{4}, P{5};
  int                 size{M * N * P};
  double              delta_p{1.0}, lenZ{1.0};
  T                   homoCoeffZ = static_cast<T>(0.0);
  std::vector<int>    dims{M, N, P}, csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
  std::vector<double> kappa(size, 1.0);
  std::vector<T>      csrValues(size * STENCIL_WIDTH, static_cast<T>(0.0)), rhs(size, static_cast<T>(0.0));

  getCsrMatData<T>(csrRowOffsets, csrColInd, csrValues, dims, kappa, kappa, kappa);

  getStdRhsVec<T>(rhs, dims, kappa, delta_p);

  getHomoCoeffZ<T>(homoCoeffZ, rhs, dims, kappa, delta_p, lenZ);

  // T *v{nullptr}, *v_hat{nullptr};
  // CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v), sizeof(T) * M * N * P));
  // CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v_hat), sizeof(T) * M * N * P));

  // cufctSolver<T> solver(M, N, P);
  // solver.fctBackward(&v[0], &v_hat[0]);
  // solver.fctBackward(&v_hat[0], &v[0]);

  // CHECK_CUDA_ERROR(cudaFree(v_hat));
  // CHECK_CUDA_ERROR(cudaFree(v));

  std::vector<T> v(size), v_hat(size), r(size);
  setTestVecs<T>(v, v_hat, dims);

  using fftwVec = std::vector<T, fftwAllocator<T>>;
  fftwVec v_fftw(size);
  mklTraits<T>::mklCopy(size, &v[0], &v_fftw[0]);
  fctSolver<T> cpuSolver(M, N, P);
  cpuSolver.fctForward(v_fftw);

  mklTraits<T>::mklResi(size, &v_hat[0], &v_fftw[0], &r[0]);
  T err{static_cast<T>(0)};
  err = mklTraits<T>::mklNorm(size, &r[0]);

  std::cout << "Error=" << err << '\n';
}
