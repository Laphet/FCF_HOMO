#include "common.hpp"
#include "cpu-fct-solver.hpp"
#include "cuda-fct-solver.hpp"
#include <algorithm>
#include <iostream>

struct op {
  bool withoutPrecond;
  bool withSsor;
  bool withICC;
};

op glbOps{false, false, false};

int main(int argc, char *argv[])
{
  /* Analysis cmd options. */
  inputParser cmdInputs(argc, argv);
  int         M{4}, N{4}, P{4};

  std::string input;
  input = cmdInputs.getCmdOption("-M");
  if (!input.empty()) M = std::stoi(input);
  input = cmdInputs.getCmdOption("-N");
  if (!input.empty()) N = std::stoi(input);
  input = cmdInputs.getCmdOption("-P");
  if (!input.empty()) P = std::stoi(input);

  glbOps.withoutPrecond = cmdInputs.cmdOptionExists("-no-pc");
  glbOps.withSsor       = cmdInputs.cmdOptionExists("-ssor");
  glbOps.withICC        = cmdInputs.cmdOptionExists("-icc");

  if (M <= 0 || N <= 0 || P <= 0) {
    std::cerr << "Input wrong arguments, M=" << M << ", N=" << N << ", P=" << P << ".\n";
    return EXIT_FAILURE;
  }

  /* Prepare data for solvers. */
  int                 size{M * N * P};
  std::vector<double> k_x(size), k_y(size), k_z(size);

  using T = double;
  std::vector<T> u_ref(size), rhs(size);

  common<T> cmmn(M, N, P);
  cmmn.setTestForSolver(k_x, k_y, k_z, u_ref, rhs);

  constexpr int       VALS_LENGHT{5};
  std::vector<double> kvals(3 * VALS_LENGHT);
  cmmn.analysisCoeff(k_x, k_y, k_z, kvals);
  std::vector<T> homoParas(5);
  std::copy(kvals.begin() + 10, kvals.end(), homoParas.begin());

  std::vector<T> dl(size), d(size), du(size);
  cmmn.getTridSolverData(dl, d, du, homoParas);

  std::vector<int> csrRowOffsets(size + 1, -1), csrColInd(size * STENCIL_WIDTH, -1);
  std::vector<T>   csrValues(size * STENCIL_WIDTH);
  cmmn.getSprMatData(csrRowOffsets, csrColInd, csrValues, k_x, k_y, k_z);

  size_t         nnz = csrRowOffsets[size];
  std::vector<T> ssorValues(nnz);
  T              omega = 1.0;

  /* The CPU solver. */
  fctSolver<T>   cpuSolver(M, N, P);
  std::vector<T> u_cpu(size);
  if (glbOps.withoutPrecond) {
    std::cout << "CPU solver without preconditioner.\n";
    cpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    cpuSolver.solveWithoutPrecond(&u_cpu[0], &rhs[0]);
  } else if (glbOps.withSsor) {
    std::cout << "CPU solver with SSOR preconditioner.\n";
    cmmn.getSsorData(csrRowOffsets, csrColInd, csrValues, omega, ssorValues);
    cpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    cpuSolver.solveWithSsor(&u_cpu[0], &rhs[0], &ssorValues[0], 20);
  } else {
    std::cout << "CPU solver with FCT preconditioner.\n";
    cpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    cpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
    cpuSolver.solve(&u_cpu[0], &rhs[0]);
  }

  /* The GPU solver. */
  cufctSolver<T> gpuSolver(M, N, P);
  std::vector<T> u_gpu(size);
  if (glbOps.withoutPrecond) {
    std::cout << "GPU solver without preconditioner.\n";
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.solveWithoutPrecond(&u_gpu[0], &rhs[0]);
  } else if (glbOps.withICC) {
    std::cout << "GPU solver with ICC preconditioner.\n";
    // To use the icc preconditioner by cuSPARSE, we need to sort all column indices.
    cmmn.sortSprMatData(csrRowOffsets, csrColInd, csrValues);
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.solveWithICC(&u_gpu[0], &rhs[0], 20);
  } else if (glbOps.withSsor) {
    std::cout << "GPU solver with SSOR preconditioner.\n";
    cmmn.getSsorData(csrRowOffsets, csrColInd, csrValues, omega, ssorValues);
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.solveWithSsor(&u_gpu[0], &rhs[0], &ssorValues[0]);
  } else {
    std::cout << "GPU solver with FCT preconditioner.\n";
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
    gpuSolver.solve(&u_gpu[0], &rhs[0]);
  }

  /* Get errors comparing with the reference solution. */
  std::vector<T> r(size);
  T              scalFactor{1 / std::sqrt(static_cast<T>(size))};
  mkl::cblas::getResidual(size, &u_ref[0], &u_gpu[0], &r[0]);
  std::printf("gpu L^2 Error=%.6e.\n", scalFactor * mkl::cblas::nrm2(size, &r[0], 1));
  mkl::cblas::getResidual(size, &u_ref[0], &u_cpu[0], &r[0]);
  std::printf("cpu L^2 Error=%.6e.\n", scalFactor * mkl::cblas::nrm2(size, &r[0], 1));

  return EXIT_SUCCESS;
}
