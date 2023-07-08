#include "common.hpp"
#include "cuda-fct-solver.hpp"
#include "cpu-fct-solver.hpp"
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
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

  T *v_d{nullptr}, *v_hat_d{nullptr}, *v_h{nullptr}, *v_hat_h{nullptr};
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v_d), sizeof(T) * M * N * P));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v_hat_d), sizeof(T) * M * N * P));
  //   thrust::device_vector<T> v_d(size), v_hat_d(size);
  //   T                       *v_d_ptr{thrust::raw_pointer_cast(v_d.data())};
  //   T                       *v_hat_d_ptr{thrust::raw_pointer_cast(v_hat_d.data())};

  cufctSolver<T> solver(M, N, P);
  //   solver.fctBackward(v_d_ptr, v_hat_d_ptr);
  //   solver.fctBackward(v_hat_d_ptr, v_d_ptr);

  CHECK_CUDA_ERROR(cudaFree(v_hat_d));
  CHECK_CUDA_ERROR(cudaFree(v_d));

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

  cpuSolver.fctBackward(v_fftw);
  mklTraits<T>::mklResi(size, &v[0], &v_fftw[0], &r[0]);
  err = mklTraits<T>::mklNorm(size, &r[0]);
  std::cout << "Error=" << err << '\n';
}
