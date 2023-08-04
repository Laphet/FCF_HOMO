#include "common.hpp"
#include "cpu-fct-solver.hpp"
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif
#include <chrono>
#include <iostream>
#include <string>

constexpr int    RESOLUTION{512};
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

#ifdef ENABLE_CUDA
template <typename T>
T gpuTestCase(double contrast, T rtol = static_cast<T>(1.0e-5))
{
  // Prepare data.
  int    M{RESOLUTION}, N{RESOLUTION}, P{RESOLUTION};
  size_t size = M * N * P;
  // The coefficient field, a ball centered.
  auto                start = std::chrono::steady_clock::now();
  std::vector<double> k(size);
  std::fill(k.begin(), k.end(), RESOLUTION * RESOLUTION);
  get_k_vec(k, contrast, std::string("bin/ball-list-1716-r1.bin"), 1716);
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
  gpuSolver.solve(&u_gpu[0], &rhs[0], 1024, rtol);
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

int main(int argc, char *argv[])
{
  std::vector<double> contrastList{0.01, 0.1, 10.0, 100.0};
  bool                useFloat{false};
  /* Analysis cmd options. */
  inputParser cmdInputs(argc, argv);

  useFloat = cmdInputs.cmdOptionExists(std::string("-float"));

  if (useFloat) {
    std::vector<float> rtolList{1.0e-5f, 1.0e-6f, 1.0e-7f, 1.0e-8f, 1.0e-9f};
    for (auto contrast : contrastList)
      for (auto rtol : rtolList) {
        std::cout << "========================================================\n";
        std::cout << "Contrast=" << contrast << ", rtol=" << rtol << std::endl;
        float homoCoeffZ = gpuTestCase<float>(contrast, rtol);
        std::cout << "  homoCoeffZ=" << homoCoeffZ << std::endl;
      }
  } else {
    double rtol = 1.0e-5;
    for (auto contrast : contrastList) {
      for (int i{0}; i < 10; ++i) {
        std::cout << "========================================================\n";
        std::cout << "Contrast=" << contrast << ", rtol=" << rtol << std::endl;
        double homoCoeffZ = gpuTestCase<double>(contrast, rtol);
        std::cout << "  homoCoeffZ=" << homoCoeffZ << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
