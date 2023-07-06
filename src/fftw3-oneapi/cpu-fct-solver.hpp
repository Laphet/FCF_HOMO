#pragma once

#include <fftw3.h>
#include <omp.h>
#include <vector>
#include <cstddef>

constexpr int DIM{3};

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
struct fftwAllocator {
  typedef T value_type;

  fftwAllocator() = default;

  template <class U>
  constexpr fftwAllocator(const fftwAllocator<U> &) noexcept;

  [[nodiscard]] T *allocate(std::size_t n);

  void deallocate(T *p, std::size_t n) noexcept;
};

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

template <typename T>
class fctSolver {
  using fftw_plan_T = decltype(fftwTraits<T>::planType);
  using fftwVec     = std::vector<T, fftwAllocator<T>>;
  int         dims[DIM];
  fftwVec     rhsBuffer;
  fftwVec     pBuffer;
  fftw_plan_T forwardPlan;  // in-place data manipulation.
  fftw_plan_T backwardPlan; // in-place data manipulation.
public:
  fctSolver(const int _M, const int _N, const int _P);

  void fctForward(fftwVec &v);

  void fctBackward(fftwVec &v);

  ~fctSolver();
};

template class fctSolver<float>;

template class fctSolver<double>;
