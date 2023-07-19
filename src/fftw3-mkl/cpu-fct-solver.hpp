#pragma once

#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <vector>

// constexpr int DIM{3};
const int          EXPECTED_CALLS{1024};
const matrix_descr DESCR{SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};

namespace fftw
{
template <typename T>
struct traits;

template <>
struct traits<float> {
  static fftwf_plan     planType;
  static fftwf_r2r_kind r2rKind;
  static void          *malloc(size_t n);
  static void           free(void *p);
  static int            initThreads(void);
  static void           planWithNthreads(int nthreads);
  static void           cleanupThreads(void);
};

template <>
struct traits<double> {
  static fftw_plan     planType;
  static fftw_r2r_kind r2rKind;
  static void         *malloc(size_t n);
  static void          free(void *p);
  static int           initThreads(void);
  static void          planWithNthreads(int nthreads);
  static void          cleanupThreads(void);
};

template <typename T>
class vector {
  size_t size;
  T     *data;

public:
  vector(size_t size);
  T &operator[](size_t idx);
  ~vector();
};

auto planManyR2R(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags);
auto planManyR2R(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags);
void execute(fftwf_plan _plan);
void execute(fftw_plan _plan);
void executeR2R(fftwf_plan _plan, float *in, float *out);
void executeR2R(fftw_plan _plan, double *in, double *out);
void destroyPlan(fftwf_plan _plan);
void destroyPlan(fftw_plan _plan);
} // namespace fftw

namespace mkl
{
namespace cblas
{
void   scal(const MKL_INT n, const float a, float *x, const MKL_INT incx);
void   scal(const MKL_INT n, const double a, double *x, const MKL_INT incx);
void   copy(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
void   copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy);
void   axpy(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
void   axpy(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy);
float  dot(const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy);
double dot(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy);
float  nrm2(const MKL_INT n, const float *x, const MKL_INT incx);
double nrm2(const MKL_INT n, const double *x, const MKL_INT incx);
void   getResidual(const MKL_INT n, const float *x, const float *y, float *r);
void   getResidual(const MKL_INT n, const double *x, const double *y, double *r);
} // namespace cblas

namespace LAPACKE
{
void pttrf(lapack_int n, float *d, float *e);
void pttrf(lapack_int n, double *d, double *e);
void pttrs(int matrix_layout, lapack_int n, lapack_int nrhs, const float *d, const float *e, float *b, lapack_int ldb);
void pttrs(int matrix_layout, lapack_int n, lapack_int nrhs, const double *d, const double *e, double *b, lapack_int ldb);
} // namespace LAPACKE

namespace sparse
{
void createCsr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values);
void createCsr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values);
void setMvHint(const sparse_matrix_t A, const sparse_operation_t operation, const struct matrix_descr descr, const MKL_INT expected_calls);
void optimize(sparse_matrix_t A);
void mv(const sparse_operation_t operation, const float alpha, const sparse_matrix_t A, const struct matrix_descr descr, const float *x, const float beta, float *y);
void mv(const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const double *x, const double beta, double *y);
} // namespace sparse
} // namespace mkl

template <typename T>
class fctSolver {
  using fftw_plan_T = decltype(fftw::traits<T>::planType);
  using fftwVec     = fftw::vector<T>;
  int             dims[3];
  fftwVec         resiBuffer;
  fftw_plan_T     forwardPlan;  // in-place data manipulation.
  fftw_plan_T     backwardPlan; // in-place data manipulation.
  T              *dlPtr;        // Use the data in the wider scope, and also modify it.
  T              *dPtr;
  T              *duPtr;
  bool            useTridSolverParallelLoop;
  sparse_matrix_t csrMat;

public:
  fctSolver(const int _M, const int _N, const int _P);

  void fctForward(T *v);

  void fctBackward(T *v);

  void setTridSolverData(T *dl, T *d, T *du, bool _parallelFor = true);

  void precondSolver(T *rhs); // rhs should be a pointer from a fftw vector.

  void setSprMatData(MKL_INT *csrRowOffsets, MKL_INT *csrColInd, T *csrValues);

  void solve(T *u, const T *b, int maxIter = 1024, T rtol = 1.0e-5, T atol = 1.0e-8);

  void solveWithoutPrecond(T *u, const T *b, int maxIter = 1024, T rtol = 1.0e-5, T atol = 1.0e-8);

  ~fctSolver();
};
