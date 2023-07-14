add_library(cuda-fct-solver OBJECT cuda-fct-solver.cu)
target_include_directories(cuda-fct-solver 
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cuda-fct-solver
  PUBLIC cudart cufft cusparse cublas)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -allow-unsupported-compiler")
endif()


# project(fft_homo LANGUAGES CXX)

# set(HELLO_WORLD helloworld)
# add_executable(${HELLO_WORLD} test.cxx)