#include "cpu-fct-solver.hpp"

namespace fftw
{
void *traits<float>::malloc(size_t n)
{
  return fftwf_malloc(n);
}

void *traits<double>::malloc(size_t n)
{
  return fftw_malloc(n);
}

void traits<float>::free(void *p)
{
  fftwf_free(p);
}

void traits<double>::free(void *p)
{
  fftw_free(p);
}

int traits<float>::initThreads()
{
  return fftwf_init_threads();
}

int traits<double>::initThreads()
{
  return fftw_init_threads();
}

void traits<float>::planWithNthreads(int nthreads)
{
  fftwf_plan_with_nthreads(nthreads);
}

void traits<double>::planWithNthreads(int nthreads)
{
  fftw_plan_with_nthreads(nthreads);
}

void traits<float>::cleanupThreads(void)
{
  fftwf_cleanup_threads();
}

void traits<double>::cleanupThreads(void)
{
  fftw_cleanup_threads();
}

auto planManyR2R(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags)
{
  return fftwf_plan_many_r2r(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, kind, flags);
}

auto planManyR2R(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags)
{
  return fftw_plan_many_r2r(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, kind, flags);
}

void execute(fftwf_plan _plan)
{
  fftwf_execute(_plan);
}

void execute(fftw_plan _plan)
{
  fftw_execute(_plan);
}

void executeR2R(fftwf_plan _plan, float *in, float *out)
{
  fftwf_execute_r2r(_plan, in, out);
}

void executeR2R(fftw_plan _plan, double *in, double *out)
{
  fftw_execute_r2r(_plan, in, out);
}

void destroyPlan(fftwf_plan _plan)
{
  fftwf_destroy_plan(_plan);
}

void destroyPlan(fftw_plan _plan)
{
  fftw_destroy_plan(_plan);
}
} // namespace fftw

namespace mkl
{
namespace cblas
{
void scal(const MKL_INT n, const float a, float *x, const MKL_INT incx)
{
  cblas_sscal(n, a, x, incx);
}

void scal(const MKL_INT n, const double a, double *x, const MKL_INT incx)
{
  cblas_dscal(n, a, x, incx);
}

void copy(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy)
{
  cblas_scopy(n, x, incx, y, incy);
}

void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
{
  cblas_dcopy(n, x, incx, y, incy);
}

void axpy(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy)
{
  cblas_saxpy(n, a, x, incx, y, incy);
}

void axpy(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
{
  cblas_daxpy(n, a, x, incx, y, incy);
}

float dot(const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy)
{
  return cblas_sdot(n, x, incx, y, incy);
}

double dot(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy)
{
  return cblas_ddot(n, x, incx, y, incy);
}

float nrm2(const MKL_INT n, const float *x, const MKL_INT incx)
{
  return cblas_snrm2(n, x, incx);
}

double nrm2(const MKL_INT n, const double *x, const MKL_INT incx)
{
  return cblas_dnrm2(n, x, incx);
}

void getResidual(const MKL_INT n, const float *x, const float *y, float *r)
{
  cblas_scopy(n, x, 1, r, 1);
  cblas_saxpy(n, -1.0f, y, 1, r, 1);
}

void getResidual(const MKL_INT n, const double *x, const double *y, double *r)
{
  cblas_dcopy(n, x, 1, r, 1);
  cblas_daxpy(n, -1.0, y, 1, r, 1);
}
} // namespace cblas

namespace LAPACKE
{
void pttrf(lapack_int n, float *d, float *e)
{
  lapack_int info{LAPACKE_spttrf(n, d, e)};
  if (info != 0) std::cerr << "mkl ?pttrf fails, info=" << info << "!\n";
}

void pttrf(lapack_int n, double *d, double *e)
{
  lapack_int info{LAPACKE_dpttrf(n, d, e)};
  if (info != 0) std::cerr << "mkl ?pttrf fails, info=" << info << "!\n";
}

void pttrs(int matrix_layout, lapack_int n, lapack_int nrhs, const float *d, const float *e, float *b, lapack_int ldb)
{
  lapack_int info{LAPACKE_spttrs(matrix_layout, n, nrhs, d, e, b, ldb)};
  if (info != 0) std::cerr << "mkl ?pttrs fails, info=" << info << "!\n";
}

void pttrs(int matrix_layout, lapack_int n, lapack_int nrhs, const double *d, const double *e, double *b, lapack_int ldb)
{
  lapack_int info{LAPACKE_dpttrs(matrix_layout, n, nrhs, d, e, b, ldb)};
  if (info != 0) std::cerr << "mkl ?pttrs fails, info=" << info << "!\n";
}
} // namespace LAPACKE

namespace sparse
{
void createCsr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values)
{
  sparse_status_t info{mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mkl_sparse_?_create_csr fails, info=" << info << "!\n";
}

void createCsr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values)
{
  sparse_status_t info{mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mkl_sparse_?_create_csr fails, info=" << info << "!\n";
}

void setMvHint(const sparse_matrix_t A, const sparse_operation_t operation, const struct matrix_descr descr, const MKL_INT expected_calls)
{
  sparse_status_t info{mkl_sparse_set_mv_hint(A, operation, descr, expected_calls)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mkl_sparse_set_mv_hint fails, info=" << info << "!\n";
}

void optimize(sparse_matrix_t A)
{
  sparse_status_t info{mkl_sparse_optimize(A)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl_sparse_optimize fails, info=" << info << "!\n";
}

void mv(const sparse_operation_t operation, const float alpha, const sparse_matrix_t A, const struct matrix_descr descr, const float *x, const float beta, float *y)
{
  sparse_status_t info{mkl_sparse_s_mv(operation, alpha, A, descr, x, beta, y)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mmkl_sparse_?_mv fails, info=" << info << "!\n";
}

void mv(const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const double *x, const double beta, double *y)
{
  sparse_status_t info{mkl_sparse_d_mv(operation, alpha, A, descr, x, beta, y)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl mmkl_sparse_?_mv fails, info=" << info << "!\n";
}
} // namespace sparse
} // namespace mkl

template <typename T>
fctSolver<T>::fctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, resiBuffer{nullptr}, dlPtr{nullptr}, dPtr{nullptr}, duPtr{nullptr}, useTridSolverParallelLoop{true}, csrMat{nullptr}
{
  const decltype(fftw::traits<T>::r2rKind) r2rKinds[4]{FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT01, FFTW_REDFT01};
  fftw::traits<T>::initThreads();
  resiBuffer = reinterpret_cast<T *>(fftw::traits<T>::malloc(_M * _N * _P * sizeof(T)));
  fftw::traits<T>::planWithNthreads(omp_get_max_threads());
  forwardPlan  = fftw::planManyR2R(2, &dims[0], dims[2], &resiBuffer[0], nullptr, dims[2], 1, &resiBuffer[0], nullptr, dims[2], 1, &r2rKinds[0], FFTW_PATIENT);
  backwardPlan = fftw::planManyR2R(2, &dims[0], dims[2], &resiBuffer[0], nullptr, dims[2], 1, &resiBuffer[0], nullptr, dims[2], 1, &r2rKinds[2], FFTW_PATIENT);
}

template <typename T>
void fctSolver<T>::fctForward(T *v)
{
  // std::cout << "fftw3 uses " << fftw_planner_nthreads() << " threads.\n";
  T *resiBuffer_ptr{&resiBuffer[0]};
  if (v == resiBuffer_ptr) {
    // std::cout << "Use fftwExec!\n";
    // It is wired that "if (&v[0] == &rhsBuffer[0])" enters this branch.
    fftw::execute(forwardPlan);
  } else {
    // std::cout << "Use fftwExecNewArray!\n";
    fftw::executeR2R(forwardPlan, &v[0], &v[0]);
  }
  mkl::cblas::scal(dims[0] * dims[1] * dims[2], static_cast<T>(0.25), &v[0], 1);
}

template <typename T>
void fctSolver<T>::fctBackward(T *v)
{
  T *resiBuffer_ptr{&resiBuffer[0]};
  if (v == resiBuffer_ptr) fftw::execute(backwardPlan);
  else fftw::executeR2R(backwardPlan, &v[0], &v[0]);
  const T scalFactor{static_cast<T>(1) / (dims[0] * dims[1])};
  mkl::cblas::scal(dims[0] * dims[1] * dims[2], scalFactor, &v[0], 1);
}

template <typename T>
void fctSolver<T>::setTridSolverData(T *dl, T *d, T *du, bool _parallelFor)
{
  if (dlPtr != nullptr || dPtr != nullptr || duPtr != nullptr) std::cerr << "The internal data have been initialized, be careful!\n";
  dlPtr = &dl[0];
  dPtr  = &d[0];
  duPtr = &du[0];

  if (_parallelFor) {
#pragma omp parallel for
    for (int idx{0}; idx < dims[0] * dims[1]; ++idx) mkl::LAPACKE::pttrf(dims[2], &dPtr[idx * dims[2]], &duPtr[idx * dims[2]]);
  } else {
    useTridSolverParallelLoop = false;
    mkl::LAPACKE::pttrf(dims[0] * dims[1] * dims[2], dPtr, duPtr);
  }
}

template <typename T>
void fctSolver<T>::precondSolver(T *rhs)
{
  if (dlPtr == nullptr || dPtr == nullptr || duPtr == nullptr) {
    std::cerr << "The internal data have not been initialized!\n";
    std::cerr << "There will be nothing to do in this routine.\n";
    return;
  }

  fctForward(rhs);

  if (useTridSolverParallelLoop) {
#pragma omp parallel for
    for (int idx{0}; idx < dims[0] * dims[1]; ++idx) mkl::LAPACKE::pttrs(LAPACK_ROW_MAJOR, dims[2], 1, &dPtr[idx * dims[2]], &duPtr[idx * dims[2]], &rhs[idx * dims[2]], 1);
  } else {
    mkl::LAPACKE::pttrs(LAPACK_ROW_MAJOR, dims[0] * dims[1] * dims[2], 1, dPtr, duPtr, &rhs[0], 1);
  }

  fctBackward(rhs);
}

template <typename T>
void fctSolver<T>::setSprMatData(MKL_INT *csrRowOffsets, MKL_INT *csrColInd, T *csrValues)
{
  if (csrMat != nullptr) std::cerr << "The internal data have been initialized, be careful!\n";

  MKL_INT size = dims[0] * dims[1] * dims[2];
  mkl::sparse::createCsr(&csrMat, SPARSE_INDEX_BASE_ZERO, size, size, &csrRowOffsets[0], &csrRowOffsets[1], &csrColInd[0], &csrValues[0]);
  mkl::sparse::setMvHint(csrMat, SPARSE_OPERATION_NON_TRANSPOSE, DESCR, EXPECTED_CALLS);
  mkl::sparse::optimize(csrMat);
}

template <typename T>
void fctSolver<T>::solve(T *u, const T *b, int maxIter, T rtol, T atol)
{
  if (dlPtr == nullptr || dPtr == nullptr || duPtr == nullptr || csrMat == nullptr) {
    std::cerr << "The internal data have not been initialized!\n";
    std::cerr << "There will be nothing to do in this routine.\n";
    return;
  }

  int size{dims[0] * dims[1] * dims[2]};
  T   myOne{static_cast<T>(1)}, myZero{static_cast<T>(0)};
  /* r <= b, r <- r - A*u */
  std::vector<T> r(size);
  mkl::cblas::copy(size, &b[0], 1, &r[0], 1);
  mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, -myOne, csrMat, DESCR, &u[0], myOne, &r[0]);

  /* resi <= r, resi <- inv(A)*resi */
  mkl::cblas::copy(size, &r[0], 1, &resiBuffer[0], 1);
  precondSolver(&resiBuffer[0]);

  /* p <= resi */
  std::vector<T> p(size);
  mkl::cblas::copy(size, &resiBuffer[0], 1, &p[0], 1);

  /* Some variables will be used in iterations. */
  T              alpha{myZero};
  T              bNorm{mkl::cblas::nrm2(size, &b[0], 1)}, rNorm{myZero};
  T              rDresi{mkl::cblas::dot(size, &r[0], 1, &resiBuffer[0], 1)}, rDresiNew{myZero};
  std::vector<T> aux(size);

  for (int itrIdx{0}; itrIdx < maxIter; ++itrIdx) {
    /* aux <- A*p + 0*aux, alpha <- rDresi / p (dot) aux */
    mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, myOne, csrMat, DESCR, &p[0], myZero, &aux[0]);
    alpha = rDresi / mkl::cblas::dot(size, &p[0], 1, &aux[0], 1);

    /* u <- u + alpha*p */
    mkl::cblas::axpy(size, alpha, &p[0], 1, &u[0], 1);

    /* r <- r - alpha*aux */
    mkl::cblas::axpy(size, -alpha, &aux[0], 1, &r[0], 1);

    /* Check convergence reasons. */
    rNorm = mkl::cblas::nrm2(size, &r[0], 1);
    if (rNorm <= bNorm * rtol) {
      std::printf("Reach rtol=%.6e, the solver exits with residual=%.6e and iterations=%d.\n", rtol, rNorm, itrIdx + 1);
      break;
    }
    if (rNorm <= atol) {
      std::printf("Reach atol=%.6e, the solver exits with residual=%.6e and iterations=%d.\n", atol, rNorm, itrIdx + 1);
      break;
    }
    if (maxIter - 1 == itrIdx) {
      std::printf("Reach maxIter=%d, the solver exits with residual=%.6e and iterations=%d.\n", maxIter, rNorm, itrIdx + 1);
      break;
    }

#ifdef DEBUG
    std::printf("itrIdx=%d,\tresidual=%.6e.\n", itrIdx + 1, rNorm);
#endif

    /* resi <= r, resi <- inv(A)*resi */
    mkl::cblas::copy(size, &r[0], 1, &resiBuffer[0], 1);
    precondSolver(&resiBuffer[0]);

    /* rDresiNew <- r (dot) resi, beta <- rDresiNew / rDresi */
    rDresiNew = mkl::cblas::dot(size, &r[0], 1, &resiBuffer[0], 1);

    /* p <- beta*p, p <- resi + p */
    mkl::cblas::scal(size, rDresiNew / rDresi, &p[0], 1);
    mkl::cblas::axpy(size, myOne, &resiBuffer[0], 1, &p[0], 1);

    /* rDresi <- rDresiNew */
    rDresi = rDresiNew;
  }

/* Check residual again, this is the true residual of the solution. */
#ifdef DEBUG
  mkl::cblas::copy(size, &b[0], 1, &r[0], 1);
  mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, -myOne, csrMat, DESCR, &u[0], myOne, &r[0]);
  rNorm = mkl::cblas::nrm2(size, &r[0], 1);
  std::printf("The true residual norm=%.6e.\n", rNorm);
#endif
}

template <typename T>
void fctSolver<T>::solveWithoutPrecond(T *u, const T *b, int maxIter, T rtol, T atol)
{
  if (csrMat == nullptr) {
    std::cerr << "The internal data have not been initialized!\n";
    std::cerr << "There will be nothing to do in this routine.\n";
    return;
  }

  int size{dims[0] * dims[1] * dims[2]};
  T   myOne{static_cast<T>(1)}, myZero{static_cast<T>(0)};
  /* r <= b, r <- r - A*u */
  std::vector<T> r(size);
  mkl::cblas::copy(size, &b[0], 1, &r[0], 1);
  mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, -myOne, csrMat, DESCR, &u[0], myOne, &r[0]);

  /* p <= r */
  std::vector<T> p(size);
  mkl::cblas::copy(size, &r[0], 1, &p[0], 1);

  T              alpha{myZero};
  T              bNorm{mkl::cblas::nrm2(size, &b[0], 1)}, rNorm{myZero};
  T              rDr{mkl::cblas::dot(size, &r[0], 1, &r[0], 1)}, rDrNew{myZero};
  std::vector<T> aux(size);

  for (int itrIdx{0}; itrIdx < maxIter; ++itrIdx) {
    /* aux <- A*p - 0*aux, alpha <- rDr / p (dot) aux */
    mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, myOne, csrMat, DESCR, &p[0], myZero, &aux[0]);
    alpha = rDr / mkl::cblas::dot(size, &p[0], 1, &aux[0], 1);

    /* u <- u + alpha*p */
    mkl::cblas::axpy(size, alpha, &p[0], 1, &u[0], 1);

    /* r <- r - alpha*aux */
    mkl::cblas::axpy(size, -alpha, &aux[0], 1, &r[0], 1);

    /* Check convergence reasons. */
    /* rDrNew <- r (dot) r */
    rDrNew = mkl::cblas::dot(size, &r[0], 1, &r[0], 1);
    rNorm  = std::sqrt(rDrNew);
    if (rNorm <= bNorm * rtol) {
      std::printf("Reach rtol=%.6e, the solver exits with residual=%.6e and iterations=%d.\n", rtol, rNorm, itrIdx + 1);
      break;
    }
    if (rNorm <= atol) {
      std::printf("Reach atol=%.6e, the solver exits with residual=%.6e and iterations=%d.\n", atol, rNorm, itrIdx + 1);
      break;
    }
    if (maxIter - 1 == itrIdx) {
      std::printf("Reach maxIter=%d, the solver exits with residual=%.6e and iterations=%d.\n", maxIter, rNorm, itrIdx + 1);
      break;
    }
#ifdef DEBUG
    std::printf("itrIdx=%d,\tresidual=%.6e.\n", itrIdx + 1, rNorm);
#endif

    /* p <- beta*p, p <- resi + p */
    mkl::cblas::scal(size, rDrNew / rDr, &p[0], 1);
    mkl::cblas::axpy(size, myOne, &r[0], 1, &p[0], 1);
    /* rDresi <- rDresiNew */
    rDr = rDrNew;
  }

  /* Check residual again, this is the true residual of the solution. */
#ifdef DEBUG
  mkl::cblas::copy(size, &b[0], 1, &r[0], 1);
  mkl::sparse::mv(SPARSE_OPERATION_NON_TRANSPOSE, -myOne, csrMat, DESCR, &u[0], myOne, &r[0]);
  rNorm = mkl::cblas::nrm2(size, &r[0], 1);
  std::printf("The true residual norm=%.6e.\n", rNorm);
#endif
}

template <typename T>
fctSolver<T>::~fctSolver()
{
  if (csrMat != nullptr) {
    mkl_sparse_destroy(csrMat);
    csrMat = nullptr;
  }
  duPtr = nullptr;
  dPtr  = nullptr;
  dlPtr = nullptr;
  fftw::destroyPlan(backwardPlan);
  backwardPlan = nullptr;
  fftw::destroyPlan(forwardPlan);
  forwardPlan = nullptr;
  fftw::traits<T>::free(resiBuffer);
  resiBuffer = nullptr;
  fftw::traits<T>::cleanupThreads();
}

template class fctSolver<float>;

template class fctSolver<double>;
