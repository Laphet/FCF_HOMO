#include "common.hpp"

int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P)
{
  return i * N * P + j * P + k;
}

void get3dIdxFromIdx(int &i, int &j, int &k, const int idx, const int N, const int P)
{
  i = idx / (N * P);
  j = (idx / P) % N;
  k = idx % P;
}

template <typename T>
void getCsrMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

  csrRowOffsets[0] = 0;
#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int    i{0}, j{0}, k{0}, col{0};
    double mean_k = 0.0;

    get3dIdxFromIdx(i, j, k, row, N, P);
    csrRowOffsets[row + 1] = 0;
    // cols order, 0, z-, z+, y-, y+, x-, x+
    csrColInd[row * STENCIL_WIDTH] = row;
    csrValues[row * STENCIL_WIDTH] = static_cast<T>(0.0);
    if (k - 1 >= 0) {
      col = getIdxFrom3dIdx(i, j, k - 1, N, P);
      // col    = i * N * P + j * P + k - 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (k + 1 < P) {
      col = getIdxFrom3dIdx(i, j, k + 1, N, P);
      // col    = i * N * P + j * P + k + 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j - 1 >= 0) {
      col = getIdxFrom3dIdx(i, j - 1, k, N, P);
      // col    = i * N * P + (j - 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j + 1 < N) {
      col = getIdxFrom3dIdx(i, j + 1, k, N, P);
      // col    = i * N * P + (j + 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i - 1 >= 0) {
      col = getIdxFrom3dIdx(i - 1, j, k, N, P);
      // col    = (i - 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i + 1 < M) {
      col = getIdxFrom3dIdx(i + 1, j, k, N, P);
      // col    = (i + 1) * N * P + j * P + k;
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
}

template <typename T>
void getStdRhsVec(std::vector<T> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, row, N, P);
    rhs[row] = static_cast<T>(0.0);
    if (k == 0) rhs[row] += static_cast<T>(2.0 * k_z[row] * delta_p);
  }
}

template <typename T>
void getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ)
{
  int    M{dims[0]}, N{dims[1]}, P{dims[2]}, i{0}, j{0}, row{0};
  double temp{0.0};

  homoCoeffZ = static_cast<T>(0.0);
#pragma omp parallel for reduction(+ : homoCoeffZ)
  for (i = 0; i < M; ++i)
    for (j = 0; j < N; ++j) {
      row = getIdxFrom3dIdx(i, j, 0, N, P);
      // row  = i * N * P + j * P;
      temp = 2.0 * lenZ * lenZ / (M * N * P) * k_z[row] * (static_cast<double>(p[row]) - delta_p) / delta_p;
      homoCoeffZ += static_cast<T>(temp);
    }
}

template <typename T>
void setTestVecs(std::vector<T> &v, std::vector<T> &v_hat, const std::vector<int> &dims)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};
  int i{0}, j{0}, k{0}, i_t{1}, j_t{2};
  T   myPi{mathTraits<T>::mathPi}, myHalf{static_cast<T>(0.5)};
  for (row = 0; row < size; ++row) {
    get3dIdxFromIdx(i, j, k, row, N, P);
    if (1 == i_t && 2 == j_t) v_hat[row] = static_cast<T>(1.0);
    else v_hat[row] = static_cast<T>(0.0);
    v[row] = static_cast<T>(4) / static_cast<T>(M * N);
    v[row] *= mathTraits<T>::mathCos(myPi * (static_cast<T>(i) + myHalf) * static_cast<T>(i_t) / static_cast<T>(M));
    v[row] *= mathTraits<T>::mathCos(myPi * (static_cast<T>(j) + myHalf) * static_cast<T>(j_t) / static_cast<T>(N));
  }
}

template void getCsrMatData<float>(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<float> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template void getCsrMatData<double>(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<double> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template void getStdRhsVec<float>(std::vector<float> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template void getStdRhsVec<double>(std::vector<double> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template void getHomoCoeffZ<float>(float &homoCoeffZ, const std::vector<float> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);

template void getHomoCoeffZ<double>(double &homoCoeffZ, const std::vector<double> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);

template void setTestVecs<float>(std::vector<float> &v, std::vector<float> &v_hat, const std::vector<int> &dims);

template void setTestVecs<double>(std::vector<double> &v, std::vector<double> &v_hat, const std::vector<int> &dims);