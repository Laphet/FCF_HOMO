#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

constexpr int STENCIL_WIDTH = 7;

int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P);

void get3dIdxFromIdx(int &i, int &j, int &k, const int idx, const int N, const int P);

template <typename T>
struct common {
  int dims[3];

  common(int _M, int _N, int _P) : dims{_M, _N, _P} {};

  void analysisCoeff(const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z, std::vector<double> &k_vals);

  void getSprMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

  void getStdRhsVec(std::vector<T> &rhs, const std::vector<double> &k_z, const double delta_p);

  void getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<double> &k_z, const double delta_p, const double lenZ);

  void getTridSolverData(std::vector<T> &dl, std::vector<T> &d, std::vector<T> &du, const std::vector<T> &homoParas);

  void setTestVecs(std::vector<T> &v, std::vector<T> &v_hat);

  void setTestForPrecondSolver(std::vector<T> &u, std::vector<T> &rhs, const T k_x, const T k_y, const T k_z);

  void setTestForSolver(std::vector<double> &k_x, std::vector<double> &k_y, std::vector<double> &k_z, std::vector<T> &u, std::vector<T> &rhs);
};
