add_executable(gemm_tiled_cuda gemm_tiled.cu)

target_compile_features(gemm_tiled_cuda PRIVATE cxx_std_17)

target_include_directories(gemm_tiled_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(gemm_tiled_cuda PRIVATE gemm_common)

if(NOT CMAKE_CUDA_STANDARD)
  set_target_properties(gemm_tiled_cuda PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
endif()

if(MSVC)
  target_compile_options(gemm_tiled_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
  )
else()
  target_compile_options(gemm_tiled_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  )
endif()