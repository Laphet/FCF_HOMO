cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(fct-homo LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  # message(CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG}")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${CMAKE_CURRENT_LIST_DIR}/externals")

pkg_check_modules(OPENMP REQUIRED IMPORTED_TARGET openmp)
# find_package(OpenMP REQUIRED)
add_subdirectory(src/common)

option(USE_CUDA "Use cuda libraries." ON)
option(USE_FFTW3_MKL "Use the fftw3 and oneapi libraries." ON)

if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/cuda)
endif()

if(USE_FFTW3_MKL)
  # The path should be where contains {NAME}.pc files.
  # This fftw3_omp.pc file is a modification of the original .pc file
  # to force linking to the OpenMP version of fftw3.
  
  # message("$ENV{PKG_CONFIG_PATH}")
  # The import modules should be {NAME} which are taken from {NAME}.pc filenames.
  pkg_check_modules(FFTW3_OMP REQUIRED IMPORTED_TARGET fftw3_omp)
  # message("${FFTW3_OMP_LIBRARIES}")
  # message("${FFTW3_OMP_LINK_LIBRARIES}")
  # message("${FFTW3_OMP_LIBRARY_DIRS}")
  # message("${FFTW3_OMP_LDFLAGS}")
  # message("${FFTW3_OMP_LDFLAGS_OTHER}")
  # message("${FFTW3_OMP_INCLUDE_DIRS}")
  # message("${FFTW3_OMP_CFLAGS}")
  # message("${FFTW3_OMP_CFLAGS_OTHER}")

  # pkg_check_modules(TBB REQUIRED IMPORTED_TARGET tbb)
  # pkg_check_modules(MKL REQUIRED IMPORTED_TARGET mkl-dynamic-ilp64-iomp)
  # pkg_check_modules(TBB REQUIRED IMPORTED_TARGET tbb)
  pkg_check_modules(MKL REQUIRED IMPORTED_TARGET mkl-dynamic-ilp64-iomp)
  # set(MKL_ROOT "$ENV{HOME}/intel/oneapi/mkl/latest")
  # list(APPEND CMAKE_MODULE_PATH "${MKL_ROOT}/lib/cmake")
  # set(MKL_LINK "static")
  # find_package(MKL CONFIG REQUIRED)
  # message(CMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}")
  # message(STATUS "${MKL_IMPORTED_TARGETS}")

  # set(TBBROOT "$ENV{HOME}/intel/oneapi/tbb/latest")
  # list(APPEND CMAKE_MODULE_PATH "${TBBROOT}/lib/cmake")
  # message(CMAKE_MODULE_PATH)
  # message("${CMAKE_PREFIX_PATH}")
  # find_package(TBB 2021.9.0 CONFIG REQUIRED)
  # message("${TBB_IMPORTED_TARGETS}")

  add_subdirectory(src/fftw3-mkl)
endif()

add_subdirectory(src/main)


# message(CMAKE_RUNTIME_OUTPUT_DIRECTORY="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")




