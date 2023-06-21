#include "common_module.h"

void _addByType(
    double *d_ptr,
    float *f_ptr,
    const int ind,
    const double val,
    const data_type dt)
{
  if (dt == _p_float)
    f_ptr[ind] += (float)val;
  else
    d_ptr[ind] += val;
}

void _zeroByType(
    double *d_ptr,
    float *f_ptr,
    const int ind,
    const data_type dt)
{
  if (dt == _p_float)
    f_ptr[ind] = 0.0f;
  else
    d_ptr[ind] = 0.0;
}

int getCsrMatData(
    int *csrRowOffsets,
    int *csrColInd,
    void *csrValues,
    const int *dims,
    const double *k_x,
    const double *k_y,
    const double *k_z,
    const data_type dt)
{
  int M = dims[0], N = dims[1], P = dims[2];
  int size = M * N * P, row = 0;
  double *d_csrValues = (double *)csrValues;
  float *f_csrValues = (float *)csrValues;

  csrRowOffsets[0] = 0;

#pragma omp parallel for
  for (row = 0; row < size; ++row)
  {
    int i = row % P, j = (row / P) % N, k = row / (P * N), col = 0;
    double mean_k = 0.0;
    csrRowOffsets[row + 1] = 0;
    // cols order, 0, z-, z+, y-, y+, x-, x+
    csrColInd[row * STENCIL_WIDTH] = row;
    _zeroByType(d_csrValues, f_csrValues, row * STENCIL_WIDTH, dt);
    if (k - 1 >= 0)
    {
      col = i * N * P + j * P + k - 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      _addByType(d_csrValues, f_csrValues, row * STENCIL_WIDTH,
                 mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (k + 1 < P)
    {
      col = i * N * P + j * P + k + 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      _addByType(d_csrValues, f_csrValues, row * STENCIL_WIDTH,
                 mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (j - 1 >= 0)
    {
      col = i * N * P + (j - 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      _addByType(d_csrValues, f_csrValues, row * STENCIL_WIDTH,
                 mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (j + 1 < N)
    {
      col = i * N * P + (j + 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH, mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (i - 1 >= 0)
    {
      col = (i - 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      _addByType(d_csrValues, f_csrValues, row * STENCIL_WIDTH,
                 mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (i + 1 < M)
    {
      col = (i + 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH, mean_k, dt);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      _zeroByType(d_csrValues, f_csrValues,
                  row * STENCIL_WIDTH + csrRowOffsets[row + 1], dt);
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH + csrRowOffsets[row + 1],
                 -mean_k, dt);
    }
    if (k == 0 || k == P - 1)
      _addByType(d_csrValues, f_csrValues,
                 row * STENCIL_WIDTH, 2.0 * k_z[row], dt);
    csrRowOffsets[row + 1]++;
  }

  // Clean unused memory.
  for (row = 0; row < size; ++row)
  {
    memmove(&csrRowOffsets[row], &csrColInd[row * STENCIL_WIDTH],
            sizeof(int) * csrRowOffsets[row + 1]);
    if (dt == _p_float)
      memmove(&f_csrValues[row], &f_csrValues[row * STENCIL_WIDTH],
              sizeof(float) * csrRowOffsets[row + 1]);
    else
      memmove(&d_csrValues[row], &d_csrValues[row * STENCIL_WIDTH],
              sizeof(double) * csrRowOffsets[row + 1]);
    csrRowOffsets[row + 1] += csrRowOffsets[row];
  }

  return 0;
}

int getStdRhsVec(
    void *rhs,
    const double *dims,
    const double *k_z,
    double delta_p,
    const data_type dt)
{
  int M = dims[0], N = dims[1], P = dims[2];
  int size = M * N * P, row = 0;
  double *d_rhs = (double *)rhs;
  float *f_rhs = (float *)rhs;

#pragma omp parallel for
  for (row = 0; row < size; ++row)
  {
    int i = row % P, j = (row / P) % N, k = row / (P * N), col = 0;
    _zeroByType(d_rhs, f_rhs, row, dt);
    if (k == 0)
      _addByType(d_rhs, f_rhs, row, 2.0 * k_z[row] * delta_p, dt);
  }

  return 0;
}

int getHomoCondZ(
    void *homoCondZ,
    const void *p,
    const double *dims,
    const double *k_z,
    const double delta_p,
    const double lenZ,
    const data_type dt)
{
  int M = dims[0], N = dims[1], P = dims[2], i = 0, j = 0, row = 0;
  double temp_p = 0.0, temp = 0.0;
  double *d_p = (double *)p, *d_homoCondZ = (double *)homoCondZ;
  float *f_p = (float *)p, *f_homoCondZ = (float *)homoCondZ;

  _zeroByType(d_homoCondZ, f_homoCondZ, 0, dt);
  for (i = 0; i < M; ++i)
    for (j = 0; j < N; ++j)
    {
      row = i * N * P + j * P;
      if (dt == _p_float)
        temp_p = (double)f_p[row];
      else
        temp_p = d_p[row];
      temp = k_z[row] * (temp_p - delta_p) / delta_p;
      temp *= 2.0 * lenZ * lenZ / (M * N * P);
      _addByType(d_homoCondZ, f_homoCondZ, 0, temp, dt);
    }

  return 0;
}

int getOnesVec(
    void *vec,
    const int size,
    const double alpha,
    const data_type dt)
{
  int i = 0;
  double *d_vec = (double *)(vec);
  float *f_vec = (float *)(vec);

  if (dt == _p_float)
  {
#pragma omp parallel for
    for (i = 0; i < size; ++i)
      f_vec[i] = (float)alpha;
  }
  else
  {
#pragma omp parallel for
    for (i = 0; i < size; ++i)
      d_vec[i] = alpha;
  }

  return 0;
}