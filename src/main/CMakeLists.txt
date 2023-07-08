add_executable(main main.cpp)
target_link_libraries(main common cuda-fct-solver cpu-fct-solver)