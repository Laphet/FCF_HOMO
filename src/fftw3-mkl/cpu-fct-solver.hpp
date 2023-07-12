#pragma once

#include <fftw3.h>
#include <mkl.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <iostream>

// constexpr int DIM{3};
const int          EXPECTED_CALLS{1024};
const matrix_descr DESCR{SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};

template <typename T>
struct fftwTraits;

template <>
struct fftwTraits<float> {
  static fftwf_plan     planType;
  static fftwf_r2r_kind r2rKind;
  static void          *fftwMalloc(size_t n) { return fftwf_malloc(n); }
  static void           fftwFree(void *p) { fftwf_free(p); }
  static int            fftwInitThreads() { return fftwf_init_threads(); }
  static void           fftwPlanWithNthreads(int nthreads) { fftwf_plan_with_nthreads(nthreads); }
  static auto fftwCreatePlan(const int *n, int howmany, float *in, int istride, float *out, int ostride, const fftw_r2r_kind *kind) { return fftwf_plan_many_r2r(2, n, howmany, in, nullptr, istride, 1, out, nullptr, ostride, 1, kind, FFTW_PATIENT); }
  static void fftwExec(fftwf_plan _plan) { fftwf_execute(_plan); }
  static void fftwExecNewArray(fftwf_plan _plan, float *in, float *out) { fftwf_execute_r2r(_plan, in, out); }
  static void fftwDestroyPlan(fftwf_plan _plan) { fftwf_destroy_plan(_plan); }
  static void fftwCleanupThreads(void) { fftwf_cleanup_threads(); }
};

template <>
struct fftwTraits<double> {
  static fftw_plan     planType;
  static fftw_r2r_kind r2rKind;
  static void         *fftwMalloc(size_t n) { return fftw_malloc(n); }
  static void          fftwFree(void *p) { fftw_free(p); }
  static int           fftwInitThreads() { return fftw_init_threads(); }
  static void          fftwPlanWithNthreads(int nthreads) { fftw_plan_with_nthreads(nthreads); }
  static auto fftwCreatePlan(const int *n, int howmany, double *in, int istride, double *out, int ostride, const fftw_r2r_kind *kind) { return fftw_plan_many_r2r(2, n, howmany, in, nullptr, istride, 1, out, nullptr, ostride, 1, kind, FFTW_PATIENT); }
  static void fftwExec(fftw_plan _plan) { fftw_execute(_plan); }
  static void fftwExecNewArray(fftw_plan _plan, double *in, double *out) { fftw_execute_r2r(_plan, in, out); }
  static void fftwDestroyPlan(fftw_plan plan) { fftw_destroy_plan(plan); }
  static void fftwCleanupThreads(void) { fftw_cleanup_threads(); }
};

template <typename T>
struct mklTraits;

template <>
struct mklTraits<float> {
  static void mklScal(const MKL_INT n, const float a, float *x) { cblas_sscal(n, a, x, 1); }
  static void mklCopy(const MKL_INT n, const float *a, float *b) { cblas_scopy(n, a, 1, b, 1); }
  static void mklResi(const MKL_INT n, const float *x, const float *y, float *r)
  {
    cblas_scopy(n, x, 1, r, 1);
    cblas_saxpy(n, -1.0f, y, 1, r, 1);
  }
  static float mklNorm(const MKL_INT n, const float *r) { return sqrtf(cblas_sdot(n, r, 1, r, 1)); }
  static void  mklTridMatFact(lapack_int n, float *d, float *e)
  {
    lapack_int info{LAPACKE_spttrf(n, d, e)};
    if (info != 0) std::cerr << "mkl ?pttrf fails, info=" << info << "!\n";
  }
  static void mklTridMatSolve(lapack_int n, const float *d, const float *e, float *b)
  {
    lapack_int info{LAPACKE_spttrs(LAPACK_ROW_MAJOR, n, 1, d, e, b, 1)};
    if (info != 0) std::cerr << "mkl ?pttrs fails, info=" << info << "!\n";
  }
  static void mklCreateSprMat(sparse_matrix_t *A, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values)
  {
    sparse_status_t info{mkl_sparse_s_create_csr(A, SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, values)};
    if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mkl_sparse_?_create_csr fails, info=" << info << "!\n";
  }
  static void mklSprMatMulVec(const float alpha, const sparse_matrix_t A, const float *x, const float beta, float *y)
  {
    sparse_status_t info{mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, DESCR, x, beta, y)};
    if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mmkl_sparse_?_mv fails, info=" << info << "!\n";
  }
};

template <>
struct mklTraits<double> {
  static void mklScal(const MKL_INT n, const double a, double *x) { cblas_dscal(n, a, x, 1); }
  static void mklCopy(const MKL_INT n, const double *a, double *b) { cblas_dcopy(n, a, 1, b, 1); }
  static void mklResi(const MKL_INT n, const double *x, const double *y, double *r)
  {
    cblas_dcopy(n, x, 1, r, 1);
    cblas_daxpy(n, -1.0, y, 1, r, 1);
  }
  static double mklNorm(const MKL_INT n, const double *r) { return sqrt(cblas_ddot(n, r, 1, r, 1)); }
  static void   mklTridMatFact(lapack_int n, double *d, double *e)
  {
    lapack_int info{LAPACKE_dpttrf(n, d, e)};
    if (info != 0) std::cerr << "mkl ?pttrf fails, info=" << info << "!\n";
  }
  static void mklTridMatSolve(lapack_int n, const double *d, const double *e, double *b)
  {
    lapack_int info{LAPACKE_dpttrs(LAPACK_ROW_MAJOR, n, 1, d, e, b, 1)};
    if (info != 0) std::cerr << "mkl ?pttrs fails, info=" << info << "!\n";
  }
  static void mklCreateSprMat(sparse_matrix_t *A, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values)
  {
    sparse_status_t info{mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, values)};
    if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mkl_sparse_?_create_csr fails, info=" << info << "!\n";
  }
  static void mklSprMatMulVec(const double alpha, const sparse_matrix_t A, const double *x, const double beta, double *y)
  {
    sparse_status_t info{mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, DESCR, x, beta, y)};
    if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mmkl_sparse_?_mv fails, info=" << info << "!\n";
  }
};

/* Create a fftw allocator (begin). */
template <typename T>
struct fftwAllocator {
  typedef T value_type;

  fftwAllocator() = default;

  template <class U>
  constexpr fftwAllocator(const fftwAllocator<U> &) noexcept;

  [[nodiscard]] T *allocate(std::size_t n);

  void deallocate(T *p, std::size_t n) noexcept;
};

template <class T, class U>
bool operator==(const fftwAllocator<T> &, const fftwAllocator<U> &);

template <class T, class U>
bool operator!=(const fftwAllocator<T> &, const fftwAllocator<U> &);
/* Create a fftw allocator (end). */

template <typename T>
class fctSolver {
  using fftw_plan_T = decltype(fftwTraits<T>::planType);
  using fftwVec     = std::vector<T, fftwAllocator<T>>;
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

  void precondSolver(T *rhs);
  /* rhs should be a pointer from a fftw vector. */

  void setSprMatData(MKL_INT *csrRowOffsets, MKL_INT *csrColInd, T *csrValues);

  void setTridSolverData(T *dl, T *d, T *du, bool _parallelFor = true);

  void solve(T *u, T *rhs, int maxIter = 1024, T *rtol = 1.0e-5, T *atol = 1.0e-8);

  ~fctSolver();
};
