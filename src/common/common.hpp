#pragma once

#include <cstring>
#include <omp.h>
#include <vector>
#include <cmath>

template <typename T>
struct mathTraits;

template <>
struct mathTraits<float> {
  static constexpr float mathPi{static_cast<float>(M_PI)};
  static float           mathCos(float x) { return cosf(x); }
};

template <>
struct mathTraits<double> {
  static constexpr double mathPi{M_PI};
  static double           mathCos(double x) { return cos(x); }
};

constexpr int STENCIL_WIDTH = 7;

int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P);

void get3dIdxFromIdx(int &i, int &j, int &k, const int idx, const int N, const int P);

template <typename T>
void getCsrMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template <typename T>
void getStdRhsVec(std::vector<T> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template <typename T>
void getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);

float mathCos(float x);

double mathCos(double x);

template <typename T>
void setTestVecs(std::vector<T> &v, std::vector<T> &v_hat, const std::vector<int> &dims);
