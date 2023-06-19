cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)
project(fct_homo LANGUAGES C)
add_subdirectory(common_module)

option(USE_CUDA "Use cuda libraries." ON)
option(USE_FFTW_ONEAPI "Use fftw and oneAPI libraries." OFF)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()



