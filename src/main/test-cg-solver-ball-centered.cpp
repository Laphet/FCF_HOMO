#include "common.hpp"
#include "cpu-fct-solver.hpp"
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif
#include <chrono>
#include <iostream>
#include <string>

constexpr double RADIUS{0.25};

/* The output k_vec has been scaled. */
void getCentBallConfig(std::vector<double> &k_vec, const int n, const double contrast)
{
  int    M{n}, N{n}, P{n};
  size_t size = M * N * P;
#pragma omp parallel for
  for (int idx{0}; idx < size; ++idx) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, idx, N, P);
    double x{(i + 0.5) / M}, y{(j + 0.5) / N}, z{(k + 0.5) / P};
    if (std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) <= RADIUS) k_vec[idx] = contrast * n * n;
    else k_vec[idx] = 1.0 * n * n;
  }
}

#ifdef ENABLE_CUDA
template <typename T>
T gpuTestCase(int n, double contrast, T rtol = static_cast<T>(1.0e-5))
{
  // Prepare data.
  int    M{n}, N{n}, P{n};
  size_t size = M * N * P;
  // The coefficient field, a ball centered.
  auto                start = std::chrono::steady_clock::now();
  std::vector<double> k(size);
  getCentBallConfig(k, n, contrast);
  std::vector<double> &k_x{k}, &k_y{k}, &k_z{k};
  // The right hand vector by the homogenization theory.
  std::vector<T> rhs(size);
  double         deltaP{1.0};
  common<T>      cmmn(M, N, P);
  cmmn.getStdRhsVec(rhs, k_z, deltaP);
  // Analyse, get the optimal reference parameters.
  constexpr int       VALS_LENGHT{5};
  std::vector<double> kvals(3 * VALS_LENGHT);
  cmmn.analysisCoeff(k_x, k_y, k_z, kvals);
  std::vector<T> homoParas(5);
  std::copy(kvals.begin() + 10, kvals.end(), homoParas.begin());
  // Tridiagonal solver data.
  std::vector<T> dl(size), d(size), du(size);
  cmmn.getTridSolverData(dl, d, du, homoParas);
  // Sparse matrix data.
  std::vector<int> csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
  std::vector<T>   csrValues(size * STENCIL_WIDTH);
  cmmn.getSprMatData(csrRowOffsets, csrColInd, csrValues, k_x, k_y, k_z);
  auto end         = std::chrono::steady_clock::now();
  auto timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  common=" << timeMillSec << "ms" << std::endl;

  // GPU solver.
  start = std::chrono::steady_clock::now();
  cufctSolver<T> gpuSolver(M, N, P);
  std::vector<T> u_gpu(size);
  std::cout << "GPU solver with FCT preconditioner.\n";
  gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
  gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Warm-up=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  gpuSolver.solve(&u_gpu[0], &rhs[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Solver=" << timeMillSec << "ms" << std::endl;

  start             = std::chrono::steady_clock::now();
  T      homoCoeffZ = 0;
  double lenZ{1.0};
  cmmn.getHomoCoeffZ(homoCoeffZ, u_gpu, k_z, deltaP, lenZ);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Post=" << timeMillSec << "ms" << std::endl;

  return homoCoeffZ;
}
#endif

template <typename T>
T cpuTestCase(int n, double contrast, T rtol = static_cast<T>(1.0e-5))
{
  // Prepare data.
  int    M{n}, N{n}, P{n};
  size_t size = M * N * P;
  // The coefficient field, a ball centered.
  auto                start = std::chrono::steady_clock::now();
  std::vector<double> k(size);
  getCentBallConfig(k, n, contrast);
  std::vector<double> &k_x{k}, &k_y{k}, &k_z{k};
  // The right hand vector by the homogenization theory.
  std::vector<T> rhs(size);
  double         deltaP{1.0};
  common<T>      cmmn(M, N, P);
  cmmn.getStdRhsVec(rhs, k_z, deltaP);
  // Analyse, get the optimal reference parameters.
  constexpr int       VALS_LENGHT{5};
  std::vector<double> kvals(3 * VALS_LENGHT);
  cmmn.analysisCoeff(k_x, k_y, k_z, kvals);
  std::vector<T> homoParas(5);
  std::copy(kvals.begin() + 10, kvals.end(), homoParas.begin());
  // Tridiagonal solver data.
  std::vector<T> dl(size), d(size), du(size);
  cmmn.getTridSolverData(dl, d, du, homoParas);
  // Sparse matrix data.
  std::vector<int> csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
  std::vector<T>   csrValues(size * STENCIL_WIDTH);
  cmmn.getSprMatData(csrRowOffsets, csrColInd, csrValues, k_x, k_y, k_z);
  auto end         = std::chrono::steady_clock::now();
  auto timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  common=" << timeMillSec << "ms" << std::endl;

  // GPU solver.
  start = std::chrono::steady_clock::now();
  fctSolver<T>   cpuSolver(M, N, P);
  std::vector<T> u_cpu(size);
  std::cout << "CPU solver with FCT preconditioner.\n";
  cpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
  cpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Warm-up=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  cpuSolver.solve(&u_cpu[0], &rhs[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Solver=" << timeMillSec << "ms" << std::endl;

  start             = std::chrono::steady_clock::now();
  T      homoCoeffZ = 0;
  double lenZ{1.0};
  cmmn.getHomoCoeffZ(homoCoeffZ, u_cpu, k_z, deltaP, lenZ);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Post=" << timeMillSec << "ms" << std::endl;

  return homoCoeffZ;
}

int main(int argc, char *argv[])
{
  using T = double;
  std::vector<int>    nList{64, 128, 256, 512};
  std::vector<double> contrastList{0.001, 0.01, 100.0, 1000.0};

  for (auto contrast : contrastList)
    for (auto n : nList) {
      std::cout << "========================================================\n";
      std::cout << "Contrast=" << contrast << std::endl;
      // T homoCoeffZ = gpuTestCase<T>(n, contrast);
      T homoCoeffZ = cpuTestCase<T>(n, contrast);
      std::cout << "  n=" << n << ", homoCoeffZ=" << homoCoeffZ << std::endl;
    }

  return EXIT_SUCCESS;
}
