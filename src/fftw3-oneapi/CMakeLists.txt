add_library(cpu-fct-solver OBJECT cpu-fct-solver.cpp)
target_include_directories(cpu-fct-solver 
  PUBLIC ${FFTW3_OMP_INCLUDE_DIRS}
  INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cpu-fct-solver
  PUBLIC PkgConfig::FFTW3_OMP
  PRIVATE OpenMP::OpenMP_CXX)