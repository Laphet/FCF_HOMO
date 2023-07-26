#include "common.hpp"
#include "cpu-fct-solver.hpp"
#ifdef ENABLE_CUDA
  #include "cuda-fct-solver.hpp"
#endif
#include <chrono>
#include <iostream>
#include <string>

struct op {
  bool withoutParallelFor;
  bool withSingle;
};

op glbOp{false, false};

template <typename T>
void cpuTestCase(int M, int N, int P)
{
  std::printf("Test case (M, N, P)=(%d, %d, %d), length of float=%d\n", M, N, P, static_cast<int>(sizeof(T)));
  if (M <= 0 || N <= 0 || P <= 0) {
    std::cerr << "Input wrong arguments, M=" << M << ", N=" << N << ", P=" << P << ".\n";
    return;
  }
  std::cout << "CPU solver\n";

  // Prepare data for solvers.
  auto           start = std::chrono::steady_clock::now();
  common<T>      cmmn(M, N, P);
  size_t         size = M * N * P;
  std::vector<T> dl(size), d(size), du(size), u(size), rhs(size);
  T              k_x = 1.0, k_y = 1.0, k_z = 1.0;
  cmmn.setTestForPrecondSolver(u, rhs, k_x, k_y, k_z);
  std::vector<T> homoParas{k_x * M * M, k_y * N * N, k_z * P * P, k_z * P * P, k_z * P * P};
  cmmn.getTridSolverData(dl, d, du, homoParas);
  auto end         = std::chrono::steady_clock::now();
  auto timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  common=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  fctSolver<T> cpuSolver(M, N, P);
  cpuSolver.setTridSolverData(&dl[0], &d[0], &du[0], !glbOp.withoutParallelFor);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Warm-up=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  cpuSolver.precondSolver(&rhs[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  precondSolver=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  mkl::cblas::axpy(size, static_cast<T>(-1), &u[0], 1, &rhs[0], 1);
  T error     = mkl::cblas::nrm2(size, &rhs[0], 1) / std::sqrt(static_cast<T>(size));
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  check=" << timeMillSec << "ms, error=" << error << std::endl;
}

#ifdef ENABLE_CUDA
template <typename T>
void gpuTestCase(int M, int N, int P)
{
  std::printf("Test case (M, N, P)=(%d, %d, %d), length of float=%d\n", M, N, P, static_cast<int>(sizeof(T)));
  if (M <= 0 || N <= 0 || P <= 0) {
    std::cerr << "Input wrong arguments, M=" << M << ", N=" << N << ", P=" << P << ".\n";
    return;
  }
  std::cout << "GPU solver\n";

  // Prepare data for solvers.
  auto           start = std::chrono::steady_clock::now();
  common<T>      cmmn(M, N, P);
  size_t         size = M * N * P;
  std::vector<T> dl(size), d(size), du(size), u(size), rhs(size);
  T              k_x = 1.0, k_y = 1.0, k_z = 1.0;
  cmmn.setTestForPrecondSolver(u, rhs, k_x, k_y, k_z);
  std::vector<T> homoParas{k_x * M * M, k_y * N * N, k_z * P * P, k_z * P * P, k_z * P * P};
  cmmn.getTridSolverData(dl, d, du, homoParas);
  auto end         = std::chrono::steady_clock::now();
  auto timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  common=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  cufctSolver<T> gpuSolver(M, N, P);
  gpuSolver.setTridSolverData(&dl[0], &d[0], &du[0]);
  T *rhs_d{nullptr};
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&rhs_d), size * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(rhs_d), reinterpret_cast<void *>(&rhs[0]), size * sizeof(T), cudaMemcpyHostToDevice));
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  Warm-up=" << timeMillSec << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  gpuSolver.precondSolver(&rhs_d[0]);
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "  precondSolver=" << timeMillSec << "us" << std::endl;

  start = std::chrono::steady_clock::now();
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void *>(&rhs[0]), reinterpret_cast<void *>(&rhs_d[0]), size * sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(rhs_d));
  mkl::cblas::axpy(size, static_cast<T>(-1), &u[0], 1, &rhs[0], 1);
  T error     = mkl::cblas::nrm2(size, &rhs[0], 1) / std::sqrt(static_cast<T>(size));
  end         = std::chrono::steady_clock::now();
  timeMillSec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "  check=" << timeMillSec << "ms, error=" << error << std::endl;
}
#endif

int main(int argc, char *argv[])
{
  // Get options.

  inputParser cmdInputs(argc, argv);
  glbOp.withoutParallelFor = cmdInputs.cmdOptionExists(std::string("-no-pfor"));
  glbOp.withSingle         = cmdInputs.cmdOptionExists(std::string("-single"));

  std::cout << "========================================================\n";
  std::cout << "withoutParallelFor=" << glbOp.withoutParallelFor << ", withSingle=" << glbOp.withSingle << ".\n";

  std::vector<int> casesList{64, 128, 256, 512};
  for (auto n : casesList) {
    if (glbOp.withSingle) {
      cpuTestCase<float>(n, n, n);
#ifdef ENABLE_CUDA
      gpuTestCase<float>(n, n, n);
#endif
    } else {
      cpuTestCase<double>(n, n, n);
#ifdef ENABLE_CUDA
      gpuTestCase<double>(n, n, n);
#endif
    }
  }

  return EXIT_SUCCESS;
}
