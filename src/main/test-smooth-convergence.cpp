#include "common.hpp"
#include "cpu-fct-solver.hpp"
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif
#include <iostream>

template <typename T>
T gpuTestCase(int n)
{
  // Prepare data.
  int    M{n}, N{n}, P{n};
  size_t size = M * N * P;
  // The smooth coefficient field.
  std::vector<double> k_x(size), k_y(size), k_z(size);
  std::vector<T>      u_ref(size), rhs(size);
  common<T>           cmmn(M, N, P);
  cmmn.setTestForSolver(k_x, k_y, k_z, u_ref, rhs);
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

  // GPU solver.
  cufctSolver<T> gpuSolver(M, N, P);
  gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
  gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  std::vector<T> u_gpu(size);
  gpuSolver.solve(&u_gpu[0], &rhs[0], 2048, static_cast<T>(1.0e-5));

  // Get the L2 error.
  std::vector<T> r(size);
  T              scalFactor{1 / std::sqrt(static_cast<T>(size))};
  mkl::cblas::getResidual(size, &u_ref[0], &u_gpu[0], &r[0]);
  return scalFactor * mkl::cblas::nrm2(size, &r[0], 1);
}

int main(int argc, char *argv[])
{
  std::vector<int> nList{32, 64, 128, 256, 512};
  using T = double;
  std::cout << "========================================================\n";
  for (auto n : nList) {
    T l2Err{gpuTestCase<T>(n)};
    std::cout << "n=" << n << ", L2 error=" << l2Err << std::endl;
  }

  return EXIT_SUCCESS;
}
