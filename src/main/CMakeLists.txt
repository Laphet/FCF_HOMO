add_executable(main test-precond-solver.cpp)
target_link_libraries(main common cuda-fct-solver cpu-fct-solver)