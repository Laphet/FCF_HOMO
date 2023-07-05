add_library(common OBJECT common.cpp)
target_include_directories(common
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(common
  PRIVATE OpenMP::OpenMP_CXX
)