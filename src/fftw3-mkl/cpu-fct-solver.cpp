#include "cpu-fct-solver.hpp"

/* Create a fftw allocator (begin). */
template <typename T>
template <class U>
constexpr fftwAllocator<T>::fftwAllocator(const fftwAllocator<U> &) noexcept
{
  std::cout << "This is a copy constructor of the fftw allocator!\n";
}

template <typename T>
T *fftwAllocator<T>::allocate(std::size_t n)
{
  return reinterpret_cast<T *>(fftwTraits<T>::fftwMalloc(n * sizeof(T)));
}

template <typename T>
void fftwAllocator<T>::deallocate(T *p, std::size_t n) noexcept
{
  fftwTraits<T>::fftwFree(p);
  p = nullptr;
}

template <class T, class U>
bool operator==(const fftwAllocator<T> &, const fftwAllocator<U> &)
{
  return true;
}

template <class T, class U>
bool operator!=(const fftwAllocator<T> &, const fftwAllocator<U> &)
{
  return false;
}
/* Create a fftw allocator (end). */

template <typename T>
fctSolver<T>::fctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, resiBuffer(_M * _N * _P), dlPtr{nullptr}, dPtr{nullptr}, duPtr{nullptr}, useTridSolverParallelLoop{true}, csrMat{nullptr}
{
  const decltype(fftwTraits<T>::r2rKind) r2rKinds[4]{FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT01, FFTW_REDFT01};
  fftwTraits<T>::fftwInitThreads();
  fftwTraits<T>::fftwPlanWithNthreads(omp_get_max_threads());
  forwardPlan  = fftwTraits<T>::fftwCreatePlan(&dims[0], dims[2], &resiBuffer[0], dims[2], &resiBuffer[0], dims[2], &r2rKinds[0]);
  backwardPlan = fftwTraits<T>::fftwCreatePlan(&dims[0], dims[2], &resiBuffer[0], dims[2], &resiBuffer[0], dims[2], &r2rKinds[2]);
}

template <typename T>
void fctSolver<T>::fctForward(T *v)
{
  // std::cout << "fftw3 uses " << fftw_planner_nthreads() << " threads.\n";
  T *resiBuffer_ptr{&resiBuffer[0]};
  if (v == resiBuffer_ptr) {
    //   if (&v[0] == &rhsBuffer[0] && v.size() == rhsBuffer.size()) {
    std::cout << "Use fftwExec!\n";
    // It is wired that "if (&v[0] == &rhsBuffer[0])" enters this branch.
    fftwTraits<T>::fftwExec(forwardPlan);
  } else {
    // std::cout << "Use fftwExecNewArray!\n";
    fftwTraits<T>::fftwExecNewArray(forwardPlan, &v[0], &v[0]);
  }
  mklTraits<T>::mklScal(dims[0] * dims[1] * dims[2], 0.25, &v[0]);
}

template <typename T>
void fctSolver<T>::fctBackward(T *v)
{
  T *resiBuffer_ptr{&resiBuffer[0]};
  if (v == resiBuffer_ptr) fftwTraits<T>::fftwExec(backwardPlan);
  else fftwTraits<T>::fftwExecNewArray(backwardPlan, &v[0], &v[0]);
  const T scalFactor{static_cast<T>(1) / (dims[0] * dims[1])};
  mklTraits<T>::mklScal(dims[0] * dims[1] * dims[2], scalFactor, &v[0]);
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
    for (int idx{0}; idx < dims[0] * dims[1]; ++idx) mklTraits<T>::mklTridMatFact(dims[2], &dPtr[idx * dims[2]], &duPtr[idx * dims[2]]);
  } else {
    useTridSolverParallelLoop = false;
    mklTraits<T>::mklTridMatFact(dims[0] * dims[1] * dims[2], dPtr, duPtr);
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
    for (int idx{0}; idx < dims[0] * dims[1]; ++idx) mklTraits<T>::mklTridMatSolve(dims[2], &dPtr[idx * dims[2]], &duPtr[idx * dims[2]], &rhs[idx * dims[2]]);
  } else {
    mklTraits<T>::mklTridMatSolve(dims[0] * dims[1] * dims[2], dPtr, duPtr, &rhs[0]);
  }

  fctBackward(rhs);
}

template <typename T>
void fctSolver<T>::setSprMatData(MKL_INT *csrRowOffsets, MKL_INT *csrColInd, T *csrValues)
{
  if (csrMat != nullptr) std::cerr << "The internal data have been initialized, be careful!\n";

  MKL_INT size = dims[0] * dims[1] * dims[2];
  mklTraits<T>::mklCreateSprMat(&csrMat, size, size, &csrRowOffsets[0], &csrRowOffsets[1], &csrColInd[0], &csrValues[0]);
  // matrix_descr    descr{SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
  sparse_status_t info{mkl_sparse_set_mv_hint(csrMat, SPARSE_OPERATION_NON_TRANSPOSE, DESCR, EXPECTED_CALLS)};
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl_sparse_set_mv_hint fails, info=" << info << "!\n";
  // info = mkl_sparse_set_dotmv_hint(csrMat, SPARSE_OPERATION_NON_TRANSPOSE, DESCR, EXPECTED_CALLS);
  // if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl_sparse_set_dotmv_hint fails, info=" << info << "!\n";
  info = mkl_sparse_optimize(csrMat);
  if (info != SPARSE_STATUS_SUCCESS) std::cerr << "mkl_sparse_optimize fails, info=" << info << "!\n";
}

template <typename T>
void fctSolver<T>::solve(T *u, T *rhs, int maxIter, T *rtol, T *atol)
{
  if (dlPtr == nullptr || dPtr == nullptr || duPtr == nullptr || csrMat == nullptr) {
    std::cerr << "The internal data have not been initialized!\n";
    std::cerr << "There will be nothing to do in this routine.\n";
    return;
  }

  int size{dims[0] * dims[1] * dims[2]};
  /* Use the zero vector as the initial guess. */
  /* rhs <- rhs - A*u */
  T myMinusOne{static_cast<T>(-1.0)}, myOne{static_cast<T>(1.0)};
  mklTraits<T>::mklSprMatMulVec(myMinusOne, csrMat, u, myOne, rhs);
  /* resi <= rhs, resi <- inv(M)*resi */
  mklTraits<T>::mklCopy(size, &rhs[0], &resiBuffer[0]);
  precondSolver(&resiBuffer[0]);
  /* p <= resi */
  std::vector<T> p(size);
  mklTraits<T>::mklCopy(size, &resiBuffer[0], &p[0]);
  for (int itrIdx{0}; itrIdx < maxIter; ++itrIdx) { }
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
  fftwTraits<T>::fftwDestroyPlan(backwardPlan);
  fftwTraits<T>::fftwDestroyPlan(forwardPlan);
  fftwTraits<T>::fftwCleanupThreads();
}

template class fctSolver<float>;

template class fctSolver<double>;
