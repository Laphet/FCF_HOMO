#include "common.hpp"
#include "cpu-fct-solver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif

constexpr int    RESOLUTION{400};
constexpr double h{1.0 / RESOLUTION};

void get_k_vec(std::vector<double> &k_vec, const double contrast, const std::string &filename, const int num)
{
  std::vector<double> ballData(4 * num);
  std::ifstream       binFileReader(filename, std::ios::in | std::ios::binary);
  binFileReader.read(reinterpret_cast<char *>(&ballData[0]), num * sizeof(double));
  binFileReader.close();
#pragma omp parallel for
  for (int idx{0}; idx < num; ++idx) {
    double ball_x{ballData[4 * idx]}, ball_y{ballData[4 * idx + 1]}, ball_z{ballData[4 * idx + 2]};
    double ball_r{ballData[4 * idx + 3]};
    int    lattice_x = static_cast<int>(ball_x / h);
    int    lattice_y = static_cast<int>(ball_y / h);
    int    lattice_z = static_cast<int>(ball_z / h);
    int    lattice_r = static_cast<int>(ball_r / h) + 1;
    double x{0.0}, y{0.0}, z{0.0}, distance{0.0};
    for (int i{std::max(0, lattice_x - lattice_r)}; i < std::min(RESOLUTION, lattice_x + lattice_r + 1); ++i) {
      x = (i + 0.5) / RESOLUTION;
      for (int j{std::max(0, lattice_y - lattice_r)}; j < std::min(RESOLUTION, lattice_y + lattice_r + 1); ++j) {
        y = (j + 0.5) / RESOLUTION;
        for (int k{std::max(0, lattice_z - lattice_r)}; k < std::min(RESOLUTION, lattice_z + lattice_r + 1); ++k) {
          z        = (k + 0.5) / RESOLUTION;
          distance = (x - ball_x) * (x - ball_x);
          distance += (y - ball_y) * (y - ball_y);
          distance += (z - ball_z) * (z - ball_z);
          distance = std::sqrt(distance);
          if (distance <= ball_r) k_vec[i * RESOLUTION * RESOLUTION + j * RESOLUTION + k] = contrast * RESOLUTION * RESOLUTION;
        }
      }
    }
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
T gpuTestCase(std::vector<double> &k_vec, T rtol = static_cast<T>(1.0e-9))
{
  // Prepare data.
  int    M{RESOLUTION}, N{RESOLUTION}, P{RESOLUTION};
  size_t size = M * N * P;
  // Balls parked.
  std::vector<double> &k_x{k_vec}, &k_y{k_vec}, &k_z{k_vec};
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

  std::vector<double>      contrastList{0.01, 0.1, 10.0, 100.0};
  std::vector<std::string> filenameList{std::string("bin/ball-list-1716-r1.bin"), std::string("bin/ball-list-258-r2.bin"), std::string("bin/ball-list-55-r3.bin")};
  std::vector<int>         ballNumList{1716, 258, 55};

  for (auto contrast : contrastList) {
    for (int i{0}; i < filenameList.size(); ++i) {
      std::cout << "========================================================\n";
      size_t              size = RESOLUTION * RESOLUTION * RESOLUTION;
      std::vector<double> k_vec(size);
      std::fill(k_vec.begin(), k_vec.end(), RESOLUTION * RESOLUTION);
      get_k_vec(k_vec, contrast, filenameList[i], ballNumList[i]);
      std::cout << "Contrast=" << contrast << ", config=" << filenameList[i] << std::endl;
      auto homoCoeffZ = gpuTestCase<T>(k_vec, static_cast<T>(1.0e-5));
      std::cout << "  homoCoeffZ=" << homoCoeffZ << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
