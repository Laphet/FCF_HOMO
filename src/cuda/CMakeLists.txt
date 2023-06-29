add_library(cuFctSolver OBJECT cuFctSolver.cu)
target_include_directories(cuFctSolver 
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuFctSolver
  PUBLIC cudart)


# project(fft_homo LANGUAGES CXX)

# set(HELLO_WORLD helloworld)
# add_executable(${HELLO_WORLD} test.cxx)