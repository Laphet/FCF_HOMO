#pragma once

#include <vector>
#include <cstring>
#include <omp.h>

const int DIM           = 3;
const int STENCIL_WIDTH = 7;

template <typename T>
int getCsrMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template <typename T>
int getStdRhsVec(std::vector<T> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template <typename T>
int getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);

#include "common_module.tpp"
