#ifndef _FCT_HOMO_COMMON_MODULE_H
#define _FCT_HOMO_COMMON_MODULE_H

#include <stdio.h>
#include <string.h>
#include <omp.h>

#define DIM 3
#define STENCIL_WIDTH 7

typedef enum data_type
{
  _p_double,
  _p_float
} data_type;

int getCsrMatData(
    int *csrRowOffsets,
    int *csrColInd,
    void *csrValues,
    const int *dims,
    const double *k_x,
    const double *k_y,
    const double *k_z,
    const data_type dt);

#endif