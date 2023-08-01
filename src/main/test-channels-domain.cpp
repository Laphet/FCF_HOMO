#include "common.hpp"
#include "cpu-fct-solver.hpp"
#include <algorithm> // std::min_element
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator> // std::begin, std::end
#include <vector>
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif

constexpr int RESOLUTION{512};
constexpr int CELL_LENGTH{8};

void get_k_vecs(std::vector<double> &k_x, std::vector<double> &k_y, std::vector<double> &k_z, const double contrast_x, const double contrast_y, const double contrast_z)
{
  size_t size = RESOLUTION * RESOLUTION * RESOLUTION;
#pragma omp parallel for
  for (int idx{0}; idx < size; ++idx) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, idx, RESOLUTION, RESOLUTION);
    int i_mod{i % CELL_LENGTH}, j_mod{j % CELL_LENGTH}, k_mod{k % CELL_LENGTH};
    if ((3 <= i_mod && i_mod < 5 && 3 <= j_mod && j_mod < 5) || (3 <= j_mod && j_mod < 5 && 3 <= k_mod && k_mod < 5) || (3 <= k_mod && k_mod < 5 && 3 <= i_mod && i_mod < 5)) {
      k_x[idx] = contrast_x * RESOLUTION * RESOLUTION;
      k_y[idx] = contrast_y * RESOLUTION * RESOLUTION;
      k_z[idx] = contrast_z * RESOLUTION * RESOLUTION;
    } else {
      k_x[idx] = 0.01 * RESOLUTION * RESOLUTION;
      k_y[idx] = 0.1 * RESOLUTION * RESOLUTION;
      k_z[idx] = 1.0 * RESOLUTION * RESOLUTION;
    }
  }
}

struct op {
  int refParasType;
};

op glbOps{0};

template <typename T>
T gpuTestCase(std::vector<double> &k_x, std::vector<double> &k_y, std::vector<double> &k_z, T rtol = static_cast<T>(1.0e-9))
{
  // Prepare data.
  int    M{RESOLUTION}, N{RESOLUTION}, P{RESOLUTION};
  size_t size = M * N * P;
  // The right hand vector by the homogenization theory.
  std::vector<T> rhs(size);
  double         deltaP{1.0};
  common<T>      cmmn(M, N, P);
  cmmn.getStdRhsVec(rhs, k_z, deltaP);
  // Analyse, get the optimal reference parameters.
  constexpr int       VALS_LENGHT{5};
  std::vector<double> kvals(3 * VALS_LENGHT);
  cmmn.analysisCoeff(k_x, k_y, k_z, kvals);
  std::cout << "Get kvals from python.\n";
  for (auto val : kvals) std::cout << val << "  ";
  std::cout << std::endl;
  std::vector<T> homoParas(5);
  switch (glbOps.refParasType) {
  case 1:
    std::cout << "  Use all 1.0.\n";
    std::fill(homoParas.begin(), homoParas.end(), static_cast<T>(1.0) * RESOLUTION * RESOLUTION);
    break;
  case 2: {
    std::cout << "  Use the average of min and max for all.\n";
    T k_allMax = *std::max_element(kvals.begin(), kvals.begin() + 5);
    T k_allMin = *std::min_element(kvals.begin() + 5, kvals.begin() + 10);
    std::fill(homoParas.begin(), homoParas.end(), k_allMax + k_allMin / 2);
    break;
  }
  case 3:
    std::cout << "  Use the average of min and max for seperate directions.\n";
    for (int i{0}; i < 5; ++i) homoParas[i] = (kvals[i] + kvals[i + 5]) / 2;
  default:
    std::copy(kvals.begin() + 10, kvals.end(), homoParas.begin());
    break;
  }
  std::cout << "Use the following reference parameters:\n";
  for (auto val : homoParas) std::cout << val << "  ";
  std::cout << std::endl;

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

  std::cout << "GPU solver with FCT preconditioner.\n";
  gpuSolver.setSprMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0]);
  gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  gpuSolver.solve(&u_gpu[0], &rhs[0]);

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
  input = cmdInputs.getCmdOption("-type");
  if (!input.empty()) glbOps.refParasType = std::stoi(input);

  std::vector<double> contrast_x_List{2.0, 4.0, 8.0, 16.0};
  std::vector<double> contrast_y_List{5.0, 25.0, 125.0, 625.0};
  std::vector<double> contrast_z_List{10.0, 100.0, 1000.0, 10000.0};

  for (int i{0}; i < 4; ++i) {
    std::cout << "========================================================\n";
    size_t              size = RESOLUTION * RESOLUTION * RESOLUTION;
    std::vector<double> k_x(size), k_y(size), k_z(size);
    get_k_vecs(k_x, k_y, k_z, contrast_x_List[i], contrast_y_List[i], contrast_z_List[i]);
    std::cout << "Contrast_x,y,z=" << contrast_x_List[i] << "  " << contrast_y_List[i] << "  " << contrast_z_List[i] << std::endl;
    auto homoCoeffZ = gpuTestCase<T>(k_x, k_y, k_z, static_cast<T>(1.0e-5));
    std::cout << "  homoCoeffZ=" << homoCoeffZ << std::endl;
  }

  return EXIT_SUCCESS;
}
