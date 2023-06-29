cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(fct_homo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

if(DEFINED ENV{MKLROOT})
  set(MKL_ROOT "$ENV{MKLROOT}")
else()
  set(MKL_ROOT "$ENV{HOME}/intel/oneapi/mkl/latest")
endif()
list(APPEND CMAKE_MODULE_PATH "${MKL_ROOT}/lib/cmake")
# message(CMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}")
find_package(MKL CONFIG REQUIRED)

option(USE_CUDA "Use cuda libraries." ON)
option(USE_FFTW "Use the fftw library." OFF)

if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  find_package(CUDAToolkit REQUIRED)
endif()

if(USE_FFTW)
  # set(CMAKE_C_COMPILER "icx")
  # set(CMAKE_CXX_COMPILER "icpx")
  # find_package(IntelSYCL REQUIRED)
endif()

find_package(OpenMP REQUIRED)


add_subdirectory(src/common_module)
add_subdirectory(src/cuda)
add_subdirectory(src/main)


# message(CMAKE_RUNTIME_OUTPUT_DIRECTORY="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")




