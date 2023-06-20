cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fct_homo LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

option(USE_CUDA "Use cuda libraries." ON)
option(USE_FFTW_ONEAPI "Use fftw and oneAPI libraries." OFF)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(src/common_module)
add_subdirectory(src/main)


message(CMAKE_RUNTIME_OUTPUT_DIRECTORY="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")




