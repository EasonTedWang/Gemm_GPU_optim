# CMake build rules for the code directory

add_library(gemm_common STATIC common/matrix_utils.cpp)
target_include_directories(gemm_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_compile_features(gemm_common PUBLIC cxx_std_17)

add_executable(origin_gemm origin_gemm.c)
set_source_files_properties(origin_gemm.c PROPERTIES LANGUAGE CXX)
target_compile_features(origin_gemm PRIVATE cxx_std_17)
target_include_directories(origin_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(origin_gemm PRIVATE gemm_common)

if(MSVC)
  target_compile_options(origin_gemm PRIVATE /W4 /permissive-)
else()
  target_compile_options(origin_gemm PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CUDAToolkit_FOUND)
  add_subdirectory(01_naive)
  add_subdirectory(02_tiled)
  add_subdirectory(03_warp)
  add_subdirectory(04_tensor_core)
endif()
