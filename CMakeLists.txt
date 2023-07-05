cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(fct-homo LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

find_package(OpenMP REQUIRED)
add_subdirectory(src/common)

# if(DEFINED ENV{MKLROOT})
#   set(MKL_ROOT "$ENV{MKLROOT}")
# else()
#   set(MKL_ROOT "$ENV{HOME}/intel/oneapi/mkl/latest")
# endif()
# list(APPEND CMAKE_MODULE_PATH "${MKL_ROOT}/lib/cmake")
# message(CMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}")
# find_package(MKL CONFIG REQUIRED)

option(USE_CUDA "Use cuda libraries." ON)
option(USE_FFTW3_ONEAPI "Use the fftw3 and oneapi libraries." ON)

if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/cuda)
endif()

if(USE_FFTW3_ONEAPI)
  
  find_package(PkgConfig REQUIRED)
  # The path should be where contains {NAME}.pc files.
  # This fftw3_omp.pc file is a modification of the original .pc file
  # to force linking to the OpenMP version of fftw3.
  set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR}:$ENV{PKG_CONFIG_PATH}")
  message("$ENV{PKG_CONFIG_PATH}")
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
  add_subdirectory(src/fftw3-oneapi)
endif()


add_subdirectory(src/main)


# message(CMAKE_RUNTIME_OUTPUT_DIRECTORY="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")




