add_library(cuFctSolver OBJECT cuFctSolver.cu)
target_include_directories(cuFctSolver 
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cuFctSolver
  PUBLIC cudart cufft)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuFctSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


# project(fft_homo LANGUAGES CXX)

# set(HELLO_WORLD helloworld)
# add_executable(${HELLO_WORLD} test.cxx)