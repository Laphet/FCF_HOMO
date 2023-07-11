#include "common.hpp"
#include "cuda-fct-solver.hpp"
#include "cpu-fct-solver.hpp"
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <iostream>

int main(int argc, char *argv[])
{
  using T = float;
  int                 M{7}, N{11}, P{5};
  int                 size{M * N * P};
  double              delta_p{1.0}, lenZ{1.0};
  T                   homoCoeffZ{static_cast<T>(0)};
  std::vector<int>    dims{M, N, P}, csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
  std::vector<double> kappa(size, 1.0);
  std::vector<T>      csrValues(size * STENCIL_WIDTH, {static_cast<T>(0)}), rhs(size, {static_cast<T>(0)});

  getCsrMatData<T>(csrRowOffsets, csrColInd, csrValues, dims, kappa, kappa, kappa);
  getStdRhsVec<T>(rhs, dims, kappa, delta_p);
  getHomoCoeffZ<T>(homoCoeffZ, rhs, dims, kappa, delta_p, lenZ);

  std::vector<T> v(size), w(size);
  setTestVecs<T>(v, w, dims);

  using fftwVec = std::vector<T, fftwAllocator<T>>;
  fftwVec v_fftw(size);
  mklTraits<T>::mklCopy(size, &v[0], &v_fftw[0]);
  fctSolver<T> cpuSolver(M, N, P);
  cpuSolver.fctForward(v_fftw);

  T             *v_d{nullptr}; // A vector in the device.
  std::vector<T> w_h(size);    // Save the result of the cuda function.
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v_d), sizeof(T) * M * N * P));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(v_d), reinterpret_cast<void *>(&v[0]), sizeof(T) * M * N * P, cudaMemcpyHostToDevice));

  cufctSolver<T> cudaSolver(M, N, P);
  cudaSolver.fctForward(v_d);

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&w_h[0]), reinterpret_cast<void *>(v_d), sizeof(T) * M * N * P, cudaMemcpyDeviceToHost));

  std::vector<T> r(size);
  T              err{static_cast<T>(0)};

  mklTraits<T>::mklResi(size, &w[0], &v_fftw[0], &r[0]);
  err = mklTraits<T>::mklNorm(size, &r[0]);
  std::cout << "fftw Error=" << err << '\n';

  mklTraits<T>::mklResi(size, &w[0], &w_h[0], &r[0]);
  err = mklTraits<T>::mklNorm(size, &r[0]);
  std::cout << "cuda Error=" << err << '\n';

  cpuSolver.fctBackward(v_fftw);

  cudaSolver.fctBackward(v_d);
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&w_h[0]), reinterpret_cast<void *>(v_d), sizeof(T) * M * N * P, cudaMemcpyDeviceToHost));

  mklTraits<T>::mklResi(size, &v[0], &v_fftw[0], &r[0]);
  err = mklTraits<T>::mklNorm(size, &r[0]);
  std::cout << "fftw Error=" << err << '\n';

  mklTraits<T>::mklResi(size, &v[0], &w_h[0], &r[0]);
  err = mklTraits<T>::mklNorm(size, &r[0]);
  std::cout << "cuda Error=" << err << '\n';

  CHECK_CUDA_ERROR(cudaFree(v_d));

  // cpuSolver.fctBackward(v_fftw);
  // mklTraits<T>::mklResi(size, &v[0], &v_fftw[0], &r[0]);
  // err = mklTraits<T>::mklNorm(size, &r[0]);
  // std::cout << "Error=" << err << '\n';
  // CHECK_CUDA_ERROR(cudaDeviceReset());

  return EXIT_SUCCESS;
}
