#include "common.hpp"
#include "cpu-fct-solver.hpp"
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif

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

struct op {
  bool   withoutPrecond;
  bool   withSsor;
  bool   withICC;
  double omega;
};

op glbOps{false, false, false, 1.0};

template <typename T>
T gpuTestCase(int n, double contrast, T rtol = static_cast<T>(1.0e-9))
{
  // Prepare data.
  int    M{n}, N{n}, P{n};
  size_t size = M * N * P;
  // The coefficient field, a ball centered.
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

  // GPU solver.
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
    gpuSolver.solveWithICC(&u_gpu[0], &rhs[0]);
  } else if (glbOps.withSsor) {
    std::cout << "GPU solver with SSOR preconditioner with omega=" << glbOps.omega << ".\n";
    std::vector<T> ssorValues(csrRowOffsets[size]);
    cmmn.getSsorData(csrRowOffsets, csrColInd, csrValues, glbOps.omega, ssorValues);
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.solveWithSsor(&u_gpu[0], &rhs[0], &ssorValues[0]);
  } else {
    std::cout << "GPU solver with FCT preconditioner.\n";
    gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
    gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
    gpuSolver.solve(&u_gpu[0], &rhs[0]);
  }

  T      homoCoeffZ = 0;
  double lenZ{1.0};
  cmmn.getHomoCoeffZ(homoCoeffZ, u_gpu, k_z, deltaP, lenZ);
  return homoCoeffZ;
}

int main(int argc, char *argv[])
{
  using T = double;
  /* Analysis cmd options. */
  inputParser cmdInputs(argc, argv);

  std::string input;
  input = cmdInputs.getCmdOption("-omega");
  if (!input.empty()) glbOps.omega = std::stod(input);

  glbOps.withoutPrecond = cmdInputs.cmdOptionExists("-no-pc");
  glbOps.withSsor       = cmdInputs.cmdOptionExists("-ssor");
  glbOps.withICC        = cmdInputs.cmdOptionExists("-icc");

  std::vector<int>    nList{400, 200, 100, 50};
  std::vector<double> contrastList{0.01, 0.1, 10.0, 100.0};

  for (auto n : nList)
    for (auto contrast : contrastList) {
      std::cout << "========================================================\n";
      std::cout << "Contrast=" << contrast << std::endl;
      auto homoCoeffZ = gpuTestCase<T>(n, contrast);
      std::cout << "  n=" << n << ", homoCoeffZ=" << homoCoeffZ << std::endl;
    }

  return EXIT_SUCCESS;
}
