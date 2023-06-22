#include <stdio.h>
#include "common_module.h"

int main(int argc, char *argv[])
{
  data_type dt = _p_float;
  int M = 3, N = 4, P = 5;
  int size = M * N * P, dims[DIM] = {M, N, P};
  double kappa[size];

  getOnesVec(&kappa[0], size, 1.0, _p_double);

  int csrRowOffsets[size + 1], csrColInd[size * STENCIL_WIDTH];
  float csrValues[size * STENCIL_WIDTH];

  getCsrMatData(&csrRowOffsets[0], &csrColInd[0], &csrValues[0],
                &dims[0], &kappa[0], &kappa[0], &kappa[0], dt);
  printf("Hello world.\n");
}