#include "cpu-fct-solver.hpp"

template <typename T>
fctSolver<T>::fctSolver(const int _M, const int _N, const int _P) : dims{_M, _N, _P}, rhsBuffer{nullptr}, pBuffer{nullptr}
{
  fftwTraits<T>::fftwInitThreads();
  rhsBuffer = reinterpret_cast<T *>(fftwTraits<T>::fftwMalloc(_M * _N * _P * sizeof(T)));
  pBuffer   = reinterpret_cast<T *>(fftwTraits<T>::fftwMalloc(_M * _N * _P * sizeof(T)));
  const decltype(fftwTraits<T>::r2rKind) r2rKinds[4]{FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT01, FFTW_REDFT01};
  fftwTraits<T>::fftwPlanWithNthreads(omp_get_max_threads());
  forwardPlan  = fftwTraits<T>::fftwCreatePlan(&dims[0], dims[2], &rhsBuffer[0], dims[2], &rhsBuffer[0], dims[2], &r2rKinds[0]);
  backwardPlan = fftwTraits<T>::fftwCreatePlan(&dims[0], dims[2], &rhsBuffer[0], dims[2], &rhsBuffer[0], dims[2], &r2rKinds[2]);
}

template <typename T>
fctSolver<T>::~fctSolver()
{
  fftwTraits<T>::fftwDestroyPlan(backwardPlan);
  fftwTraits<T>::fftwDestroyPlan(forwardPlan);
  fftwTraits<T>::fftwCleanupThreads();
  fftwTraits<T>::fftwFree(pBuffer);
  pBuffer = nullptr;
  fftwTraits<T>::fftwFree(rhsBuffer);
  rhsBuffer = nullptr;
}

template <typename T>
void fctSolver<T>::fctForward(T *v)
{
  if (v == rhsBuffer) fftwTraits<T>::fftwExec(forwardPlan);
  else fftwTraits<T>::fftwExecNewArray(forwardPlan, v, v);
}

template <typename T>
void fctSolver<T>::fctBackward(T *v)
{
  if (v == rhsBuffer) fftwTraits<T>::fftwExec(backwardPlan);
  else fftwTraits<T>::fftwExecNewArray(backwardPlan, v, v);
}
