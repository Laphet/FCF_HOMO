add_library(cpu-fct-solver OBJECT cpu-fct-solver.cpp)

target_include_directories(cpu-fct-solver 
  PUBLIC ${FFTW3_OMP_INCLUDE_DIRS}
  PUBLIC ${MKL_INCLUDE_DIRS}
  INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(cpu-fct-solver
  PRIVATE PkgConfig::OPENMP
  PUBLIC PkgConfig::FFTW3_OMP
  PUBLIC PkgConfig::MKL)

target_compile_definitions(cpu-fct-solver
  PRIVATE $<$<CONFIG:Debug>:DEBUG>)

target_compile_options(cpu-fct-solver
  PUBLIC -qmkl)
