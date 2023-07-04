add_library(common_module OBJECT common_module.cpp)
target_include_directories(common_module
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(common_module
  PRIVATE OpenMP::OpenMP_CXX
)