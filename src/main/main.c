#include <stdio.h>
#include "common_module.h"

int main(int argc, char *argv[])
{
  data_type dt = _p_float;
  int M = 3, N = 4, P = 5;
  int size = M * N * P, dims[DIM] = {M, N, P};
  double kappa[size];

  getOnesVec(&kappa[0], size, 1.0, _p_double);
  printf("Hello world.\n");
}