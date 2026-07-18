function(add_tiled_target target source)
  add_executable(${target} ${source})

  target_compile_features(${target} PRIVATE cxx_std_17)

  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

  target_link_libraries(${target} PRIVATE gemm_common)

  if(NOT CMAKE_CUDA_STANDARD)
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
  endif()
endfunction()

add_tiled_target(gemm_tiled_cuda gemm_tiled.cu)
add_tiled_target(gemm_tiled32_cuda gemm_tiled32.cu)
add_tiled_target(gemm_tiled2x2_cuda gemm_tiled2x2.cu)
add_tiled_target(gemm_tiled4x4_cuda gemm_tiled4x4.cu)
