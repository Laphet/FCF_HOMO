add_executable(main test-stability-parked-balls.cpp)
# target_link_libraries(common
#   # PRIVATE OpenMP::OpenMP_CXX
#   PRIVATE PkgConfig::OPENMP
# )
if(USE_CUDA)
  target_link_libraries(main common cuda-fct-solver cpu-fct-solver)
else()
  target_link_libraries(main common cpu-fct-solver)
endif()