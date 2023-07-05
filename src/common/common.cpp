#include "common.hpp"

template <typename T>
int getCsrMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

  csrRowOffsets[0] = 0;
#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int    i{row / (P * N)}, j{(row / P) % N}, k{row % P}, col{0};
    double mean_k = 0.0;

    csrRowOffsets[row + 1] = 0;
    // cols order, 0, z-, z+, y-, y+, x-, x+
    csrColInd[row * STENCIL_WIDTH] = row;
    csrValues[row * STENCIL_WIDTH] = static_cast<T>(0.0);
    if (k - 1 >= 0) {
      col    = i * N * P + j * P + k - 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (k + 1 < P) {
      col    = i * N * P + j * P + k + 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j - 1 >= 0) {
      col    = i * N * P + (j - 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j + 1 < N) {
      col    = i * N * P + (j + 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i - 1 >= 0) {
      col    = (i - 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i + 1 < M) {
      col    = (i + 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (k == 0 || k == P - 1) csrValues[row * STENCIL_WIDTH] += static_cast<T>(2.0 * k_z[row]);
    csrRowOffsets[row + 1]++;
  }

  // Clean the unused memory.
  for (row = 0; row < size; ++row) {
    std::memmove(&csrColInd[csrRowOffsets[row]], &csrColInd[row * STENCIL_WIDTH], sizeof(int) * csrRowOffsets[row + 1]);
    std::memmove(&csrValues[csrRowOffsets[row]], &csrValues[row * STENCIL_WIDTH], sizeof(T) * csrRowOffsets[row + 1]);
    csrRowOffsets[row + 1] += csrRowOffsets[row];
  }

  return 0;
}

template <typename T>
int getStdRhsVec(std::vector<T> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int i{row / (P * N)}, j{(row / P) % N}, k{row % P}, col{0};
    rhs[row] = static_cast<T>(0.0);
    if (k == 0) rhs[row] += static_cast<T>(2.0 * k_z[row] * delta_p);
  }

  return 0;
}

template <typename T>
int getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ)
{
  int    M{dims[0]}, N{dims[1]}, P{dims[2]}, i{0}, j{0}, row{0};
  double temp{0.0};

  homoCoeffZ = static_cast<T>(0.0);
#pragma omp parallel for reduction(+ : homoCoeffZ)
  for (i = 0; i < M; ++i)
    for (j = 0; j < N; ++j) {
      row  = i * N * P + j * P;
      temp = 2.0 * lenZ * lenZ / (M * N * P) * k_z[row] * (static_cast<double>(p[row]) - delta_p) / delta_p;
      homoCoeffZ += static_cast<T>(temp);
    }

  return 0;
}

#include "common.tpp"