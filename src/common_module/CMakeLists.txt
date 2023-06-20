add_library(common_module OBJECT common_module.c)
target_include_directories(common_module PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)