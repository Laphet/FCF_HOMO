#pragma once

#include <cstring>
#include <omp.h>
#include <vector>

// #define getIdxFrom3dIdx(i, j, k, N, P) ((i) * (N) * (P) + (j) * (P) + (k))
// The device code also need this routine, it seems that writing a macro is acceptable.

constexpr int DIM           = 3;
constexpr int STENCIL_WIDTH = 7;

template <typename T>
int getCsrMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template <typename T>
int getStdRhsVec(std::vector<T> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template <typename T>
int getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);
