# set(TBBROOT "$ENV{HOME}/intel/oneapi/tbb/latest")
# list(APPEND CMAKE_PREFIX_PATH "${TBBROOT}/lib/cmake")
# message("${CMAKE_PREFIX_PATH}")
# find_package(TBB 2021.9.0 CONFIG REQUIRED)
# message("${TBB_IMPORTED_TARGETS}")

add_library(cpu-fct-solver OBJECT cpu-fct-solver.cpp)
target_include_directories(cpu-fct-solver 
  PUBLIC ${FFTW3_OMP_INCLUDE_DIRS}
  PUBLIC ${MKL_INCLUDE_DIRS}
  # PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
  INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cpu-fct-solver
  PRIVATE PkgConfig::OPENMP
  PUBLIC PkgConfig::FFTW3_OMP
  # PRIVATE OpenMP::OpenMP_CXX

  # PUBLIC $<LINK_ONLY:MKL::MKL>
  # PUBLIC PkgConfig::TBB
  PUBLIC PkgConfig::MKL)
  # PUBLIC PkgConfig::TBB)
  # PUBLIC "${TBBROOT}/lib/intel64/gcc4.8/libtbb.so")