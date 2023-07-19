#include "common.hpp"
#include "cpu-fct-solver.hpp"
#include "cuda-fct-solver.hpp"
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <algorithm>
#include <iostream>

class InputParser {
public:
  InputParser(int &argc, char **argv)
  {
    for (int i = 1; i < argc; ++i) this->tokens.push_back(std::string(argv[i]));
  }
  /// @author iain
  const std::string &getCmdOption(const std::string &option) const
  {
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) { return *itr; }
    static const std::string empty_string("");
    return empty_string;
  }
  /// @author iain
  bool cmdOptionExists(const std::string &option) const { return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end(); }

private:
  std::vector<std::string> tokens;
};

// int main(int argc, char *argv[])
// {
//   using T = float;
//   int                 M{7}, N{11}, P{5};
//   int                 size{M * N * P};
//   double              delta_p{1.0}, lenZ{1.0};
//   T                   homoCoeffZ{static_cast<T>(0)};
//   std::vector<int>    dims{M, N, P}, csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
//   std::vector<double> kappa(size, 1.0);
//   std::vector<T>      csrValues(size * STENCIL_WIDTH, {static_cast<T>(0)}), rhs(size, {static_cast<T>(0)});

//   getSprMatData<T>(csrRowOffsets, csrColInd, csrValues, dims, kappa, kappa, kappa);
//   getStdRhsVec<T>(rhs, dims, kappa, delta_p);
//   getHomoCoeffZ<T>(homoCoeffZ, rhs, dims, kappa, delta_p, lenZ);

//   std::vector<T> v(size), w(size);
//   setTestVecs<T>(v, w, dims);

//   using fftwVec = std::vector<T, fftwAllocator<T>>;
//   fftwVec v_fftw(size);
//   mklTraits<T>::mklCopy(size, &v[0], &v_fftw[0]);
//   fctSolver<T> cpuSolver(M, N, P);
//   cpuSolver.fctForward(&v_fftw[0]);

//   T             *v_d{nullptr}; // A vector in the device.
//   std::vector<T> w_h(size);    // Save the result of the cuda function.
//   CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&v_d), sizeof(T) * M * N * P));
//   CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(v_d), reinterpret_cast<void *>(&v[0]), sizeof(T) * M * N * P, cudaMemcpyHostToDevice));

//   cufctSolver<T> cudaSolver(M, N, P);
//   cudaSolver.fctForward(v_d);

//   CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&w_h[0]), reinterpret_cast<void *>(v_d), sizeof(T) * M * N * P, cudaMemcpyDeviceToHost));

//   std::vector<T> r(size);
//   T              err{static_cast<T>(0)};

//   mklTraits<T>::mklResi(size, &w[0], &v_fftw[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   std::cout << "fftw Error=" << err << '\n';

//   mklTraits<T>::mklResi(size, &w[0], &w_h[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   std::cout << "cuda Error=" << err << '\n';

//   cpuSolver.fctBackward(&v_fftw[0]);

//   cudaSolver.fctBackward(v_d);
//   CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&w_h[0]), reinterpret_cast<void *>(v_d), sizeof(T) * M * N * P, cudaMemcpyDeviceToHost));

//   mklTraits<T>::mklResi(size, &v[0], &v_fftw[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   std::cout << "fftw Error=" << err << '\n';

//   mklTraits<T>::mklResi(size, &v[0], &w_h[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   std::cout << "cuda Error=" << err << '\n';

//   CHECK_CUDA_ERROR(cudaFree(v_d));

//   return EXIT_SUCCESS;
// }

// int main(int argc, char *argv[])
// {
//   InputParser cmdInputs(argc, argv);

//   int         M{3}, N{4}, P{5};
//   std::string input;
//   input = cmdInputs.getCmdOption("-M");
//   if (!input.empty()) M = std::stoi(input);
//   input = cmdInputs.getCmdOption("-N");
//   if (!input.empty()) N = std::stoi(input);
//   input = cmdInputs.getCmdOption("-P");
//   if (!input.empty()) P = std::stoi(input);

//   if (M <= 0 || N <= 0 || P <= 0) {
//     std::cerr << "Input wrong arguments, M=" << M << ", N=" << N << ", P=" << P << ".\n";
//     return EXIT_FAILURE;
//   }

//   using T = double;
//   common<T>      cmmn(M, N, P);
//   T              k_x{static_cast<T>(1)}, k_y{static_cast<T>(2)}, k_z{static_cast<T>(3)};
//   int            size{M * N * P};
//   std::vector<T> u(size), rhs(size);
//   cmmn.setTestForPrecondSolver(u, rhs, k_x, k_y, k_z);

//   std::vector<T> homoParas(5);
//   homoParas[0] = k_x * M * M;
//   homoParas[1] = k_y * N * N;
//   homoParas[2] = k_z * P * P;
//   homoParas[3] = k_z * P * P;
//   homoParas[4] = k_z * P * P;
//   std::vector<T> dl(size), d(size), du(size);
//   cmmn.getTridSolverData(dl, d, du, homoParas);

//   cufctSolver<T> cudaSolver(M, N, P);
//   T             *rhs_d;
//   CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&rhs_d), size * sizeof(T)));
//   CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(rhs_d), reinterpret_cast<void *>(&rhs[0]), size * sizeof(T), cudaMemcpyHostToDevice));
//   cudaSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
//   cudaSolver.precondSolver(rhs_d);
//   std::vector<T> u_h(size);
//   CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&u_h[0]), reinterpret_cast<void *>(rhs_d), size * sizeof(T), cudaMemcpyDeviceToHost));

//   fctSolver<T> cpuSolver(M, N, P);
//   cpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
//   using fftwVec = std::vector<T, fftwAllocator<T>>;
//   fftwVec rhs_fftw(size);
//   mklTraits<T>::mklCopy(size, &rhs[0], &rhs_fftw[0]);
//   cpuSolver.precondSolver(&rhs_fftw[0]);

//   std::vector<T> r(size);
//   T              err{static_cast<T>(0)};
//   T              scalFactor{1 / std::sqrt(static_cast<T>(size))};

//   mklTraits<T>::mklResi(size, &u[0], &u_h[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   err *= scalFactor;
//   std::cout << "cuda L^2 Error=" << err << '\n';

//   mklTraits<T>::mklResi(size, &u[0], &rhs_fftw[0], &r[0]);
//   err = mklTraits<T>::mklNorm(size, &r[0]);
//   err *= scalFactor;
//   std::cout << "fftw L^2 Error=" << err << '\n';

//   CHECK_CUDA_ERROR(cudaFree(rhs_d));

//   return EXIT_SUCCESS;
// }

int main(int argc, char *argv[])
{
  InputParser cmdInputs(argc, argv);

  int         M{5}, N{7}, P{11};
  std::string input;
  input = cmdInputs.getCmdOption("-M");
  if (!input.empty()) M = std::stoi(input);
  input = cmdInputs.getCmdOption("-N");
  if (!input.empty()) N = std::stoi(input);
  input = cmdInputs.getCmdOption("-P");
  if (!input.empty()) P = std::stoi(input);

  if (M <= 0 || N <= 0 || P <= 0) {
    std::cerr << "Input wrong arguments, M=" << M << ", N=" << N << ", P=" << P << ".\n";
    return EXIT_FAILURE;
  }

  int                 size{M * N * P};
  std::vector<double> k_x(size), k_y(size), k_z(size);

  using T = double;
  std::vector<T> u(size), rhs(size);

  common<T> cmmn(M, N, P);
  cmmn.setTestForSolver(k_x, k_y, k_z, u, rhs);
  cmmn.analysisCoeff(k_x, k_y, k_z);

  // std::vector<T> homoParas(5);
  // homoParas[0] = 1.0;
  // homoParas[1] = 1.0;
  // homoParas[2] = 1.0;
  // homoParas[3] = 1.0;
  // homoParas[4] = 1.0;
  // std::vector<T> dl(size), d(size), du(size);
  // cmmn.getTridSolverData(dl, d, du, homoParas);

  // cufctSolver<T> cudaSolver(M, N, P);
  // T             *rhs_d;
  // CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&rhs_d), size * sizeof(T)));
  // CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(rhs_d), reinterpret_cast<void *>(&rhs[0]), size * sizeof(T), cudaMemcpyHostToDevice));
  // cudaSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  // cudaSolver.precondSolver(rhs_d);
  // std::vector<T> u_h(size);
  // CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&u_h[0]), reinterpret_cast<void *>(rhs_d), size * sizeof(T), cudaMemcpyDeviceToHost));

  // fctSolver<T> cpuSolver(M, N, P);
  // cpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  // using fftwVec = std::vector<T, fftwAllocator<T>>;
  // fftwVec rhs_fftw(size);
  // mklTraits<T>::mklCopy(size, &rhs[0], &rhs_fftw[0]);
  // cpuSolver.precondSolver(&rhs_fftw[0]);

  // std::vector<T> r(size);
  // T              err{static_cast<T>(0)};
  // T              scalFactor{1 / std::sqrt(static_cast<T>(size))};

  // mklTraits<T>::mklResi(size, &u[0], &u_h[0], &r[0]);
  // err = mklTraits<T>::mklNorm(size, &r[0]);
  // err *= scalFactor;
  // std::cout << "cuda L^2 Error=" << err << '\n';

  // mklTraits<T>::mklResi(size, &u[0], &rhs_fftw[0], &r[0]);
  // err = mklTraits<T>::mklNorm(size, &r[0]);
  // err *= scalFactor;
  // std::cout << "fftw L^2 Error=" << err << '\n';

  // CHECK_CUDA_ERROR(cudaFree(rhs_d));

  return EXIT_SUCCESS;
}
