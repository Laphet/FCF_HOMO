cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)


project(fft_homo LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(HELLO_WORLD helloworld)
add_executable(${HELLO_WORLD} axpy.cu)
target_link_libraries(${HELLO_WORLD} ${CUDA_LIBRARIES} cublas cufft cusparse)
# set_property(TARGET ${HELLO_WORLD} PROPERTY CUDA_ARCHITECTURES OFF)


# project(fft_homo LANGUAGES CXX)

# set(HELLO_WORLD helloworld)
# add_executable(${HELLO_WORLD} test.cxx)