add_executable(gemm_tensor_core gemm_tensor_core.cu)

target_compile_features(gemm_tensor_core PRIVATE cxx_std_17)

target_include_directories(gemm_tensor_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(gemm_tensor_core PRIVATE gemm_common)

if(NOT CMAKE_CUDA_STANDARD)
  set_target_properties(gemm_tensor_core PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
endif()

if(MSVC)
  target_compile_options(gemm_tensor_core PRIVATE /W4)
else()
  target_compile_options(gemm_tensor_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
