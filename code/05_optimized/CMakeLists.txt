add_executable(gemm_cublas gemm_cublas.cu)
add_executable(gemm_optimized_cuda gemm_optimized.cu)
add_executable(gemm_optimized_k8_cuda gemm_optimized.cu)
add_executable(gemm_optimized_async_cuda gemm_optimized.cu)
add_executable(gemm_tensor_tiled_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tiled128_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tiled128_k32_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tiled128_s3_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128_k32_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128_s3_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128x64_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128x64_k32_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128x64_w4x1_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128x64_w2x2_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tma128x64_w2x2_k32_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tiled128x64_s3_cuda gemm_tensor_tiled.cu)
add_executable(gemm_tensor_tiled64x128_s3_cuda gemm_tensor_tiled.cu)
add_executable(gemm_cublas_tensor gemm_cublas_tensor.cu)
add_executable(gemm_cublaslt_tensor gemm_cublaslt_tensor.cu)
target_compile_definitions(gemm_tensor_tiled128_cuda PRIVATE GEMM_TENSOR_CTA=128)
target_compile_definitions(gemm_tensor_tiled128_k32_cuda PRIVATE
  GEMM_TENSOR_CTA=128 GEMM_TENSOR_CTA_K=32)
target_compile_definitions(gemm_tensor_tiled128_s3_cuda PRIVATE
  GEMM_TENSOR_CTA=128 GEMM_TENSOR_CTA_K=16 GEMM_TENSOR_STAGES=3)
target_compile_definitions(gemm_tensor_tma128_cuda PRIVATE
  GEMM_TENSOR_CTA=128 GEMM_TENSOR_CTA_K=16 GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1)
target_compile_definitions(gemm_tensor_tma128_k32_cuda PRIVATE
  GEMM_TENSOR_CTA=128 GEMM_TENSOR_CTA_K=32 GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1)
target_compile_definitions(gemm_tensor_tma128_s3_cuda PRIVATE
  GEMM_TENSOR_CTA=128 GEMM_TENSOR_CTA_K=16 GEMM_TENSOR_STAGES=3 GEMM_TENSOR_USE_TMA=1)
target_compile_definitions(gemm_tensor_tma128x64_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=16
  GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1)
target_compile_definitions(gemm_tensor_tma128x64_k32_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=32
  GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1)
target_compile_definitions(gemm_tensor_tma128x64_w4x1_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=16
  GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1
  GEMM_TENSOR_WARP_GRID_M=4 GEMM_TENSOR_WARP_GRID_N=1)
target_compile_definitions(gemm_tensor_tma128x64_w2x2_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=16
  GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1
  GEMM_TENSOR_WARP_GRID_M=2 GEMM_TENSOR_WARP_GRID_N=2)
target_compile_definitions(gemm_tensor_tma128x64_w2x2_k32_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=32
  GEMM_TENSOR_STAGES=2 GEMM_TENSOR_USE_TMA=1
  GEMM_TENSOR_WARP_GRID_M=2 GEMM_TENSOR_WARP_GRID_N=2)
target_compile_definitions(gemm_tensor_tiled128x64_s3_cuda PRIVATE
  GEMM_TENSOR_CTA_M=128 GEMM_TENSOR_CTA_N=64 GEMM_TENSOR_CTA_K=16 GEMM_TENSOR_STAGES=3)
target_compile_definitions(gemm_tensor_tiled64x128_s3_cuda PRIVATE
  GEMM_TENSOR_CTA_M=64 GEMM_TENSOR_CTA_N=128 GEMM_TENSOR_CTA_K=16 GEMM_TENSOR_STAGES=3)
target_compile_definitions(gemm_optimized_k8_cuda PRIVATE GEMM_CTA_K=8)
target_compile_definitions(gemm_optimized_async_cuda PRIVATE GEMM_CTA_K=16 GEMM_USE_CP_ASYNC=1)

foreach(target
    gemm_cublas
    gemm_optimized_cuda
    gemm_optimized_k8_cuda
    gemm_optimized_async_cuda
    gemm_tensor_tiled_cuda
    gemm_tensor_tiled128_cuda
    gemm_tensor_tiled128_k32_cuda
    gemm_tensor_tiled128_s3_cuda
    gemm_tensor_tma128_cuda
    gemm_tensor_tma128_k32_cuda
    gemm_tensor_tma128_s3_cuda
    gemm_tensor_tma128x64_cuda
    gemm_tensor_tma128x64_k32_cuda
    gemm_tensor_tma128x64_w4x1_cuda
    gemm_tensor_tma128x64_w2x2_cuda
    gemm_tensor_tma128x64_w2x2_k32_cuda
    gemm_tensor_tiled128x64_s3_cuda
    gemm_tensor_tiled64x128_s3_cuda
    gemm_cublas_tensor
    gemm_cublaslt_tensor)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
  target_link_libraries(${target} PRIVATE gemm_common)
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;-Xcompiler=-Wall,-Wextra>
    )
  endif()
endforeach()

# Find cuBLAS next to the nvcc selected by FindCUDAToolkit. Systems with
# multiple CUDA installations commonly expose an older /usr/lib lib first.
get_filename_component(GEMM_CUDA_TOOLKIT_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
find_library(GEMM_CUBLAS_LIBRARY
  NAMES cublas
  PATHS
    "${GEMM_CUDA_TOOLKIT_ROOT}/lib64"
    "${GEMM_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(GEMM_CUBLASLT_LIBRARY
  NAMES cublasLt
  PATHS
    "${GEMM_CUDA_TOOLKIT_ROOT}/lib64"
    "${GEMM_CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
target_link_libraries(gemm_cublas PRIVATE "${GEMM_CUBLAS_LIBRARY}")
target_link_libraries(gemm_cublas_tensor PRIVATE "${GEMM_CUBLAS_LIBRARY}")
target_link_libraries(gemm_cublaslt_tensor PRIVATE "${GEMM_CUBLASLT_LIBRARY}")
target_link_libraries(gemm_tensor_tma128_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128_k32_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128_s3_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128x64_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128x64_k32_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128x64_w4x1_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128x64_w2x2_cuda PRIVATE CUDA::cuda_driver)
target_link_libraries(gemm_tensor_tma128x64_w2x2_k32_cuda PRIVATE CUDA::cuda_driver)
