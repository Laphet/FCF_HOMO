#pragma once

#include <vector>
#include <cstring>

const int DIM = 3;
const int STENCIL_WIDTH = 7;

template <typename T = double>
int getCsrMatData(
    std::vector<int> &csrRowOffsets,
    std::vector<int> &csrColInd,
    std::vector<T> &csrValues,
    const std::vector<int> &dims,
    const std::vector<double> &k_x,
    const std::vector<double> &k_y,
    const std::vector<double> &k_z);
