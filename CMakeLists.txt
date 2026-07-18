cmake_minimum_required(VERSION 3.22)
project(Gemm_GPU_optim VERSION 0.1 LANGUAGES C CXX)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output all runtime binaries into the repository-level out/ folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/lib)
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/out)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/out)
endif()

option(ENABLE_CUDA "Enable CUDA support for GEMM experiments" ON)
if(ENABLE_CUDA)
  # This repository is tuned for the RTX 5080 (Blackwell, SM 120). Keep the
  # target explicit so a stale CMake cache cannot silently emit old SM 75 code.
  set(GEMM_CUDA_ARCHITECTURES "120" CACHE STRING
      "CUDA architectures used by GEMM targets (for example: 120, 90, 89)")
  set(CMAKE_CUDA_ARCHITECTURES "${GEMM_CUDA_ARCHITECTURES}")

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
  else()
    message(WARNING "CUDA toolkit not found. CUDA targets will be disabled.")
  endif()
endif()

add_subdirectory(code)

set(ALL_GEMM_TARGETS origin_gemm)
if(CUDAToolkit_FOUND)
  list(APPEND ALL_GEMM_TARGETS
    gemm_naive_cuda
    gemm_tiled_cuda
    gemm_tiled32_cuda
    gemm_tiled2x2_cuda
    gemm_tiled4x4_cuda
    gemm_warp_cuda
    gemm_tensor_core
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
    gemm_cublaslt_tensor
  )
endif()

add_custom_target(all_examples
  DEPENDS ${ALL_GEMM_TARGETS}
  COMMENT "Build all GEMM example executables"
)

enable_testing()
add_test(NAME origin_gemm_smoke
  COMMAND ${CMAKE_SOURCE_DIR}/out/origin_gemm 32 33 34 --verify --warmup 0 --repeat 1
)

if(CUDAToolkit_FOUND)
  add_test(NAME gemm_naive_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_naive_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tiled_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tiled32_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled32_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tiled2x2_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled2x2_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tiled4x4_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled4x4_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_warp_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_warp_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_core_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_core 64 64 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_cublas_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublas 63 65 67 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_cuda 128 128 32 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_cuda 127 131 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_k8_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_k8_cuda 128 128 32 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_k8_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_k8_cuda 127 131 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_async_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_async_cuda 128 128 32 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_optimized_async_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_async_cuda 127 131 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled_cuda 128 128 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled_cuda 80 96 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_cuda 256 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_k32_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_k32_cuda 256 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_k32_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_k32_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_s3_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_s3_cuda 256 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128_s3_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_s3_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_s3_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_s3_cuda 256 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_cuda 256 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_k32_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_k32_cuda 256 256 128 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_k32_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_k32_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128_s3_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_s3_cuda 144 160 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_cuda 256 128 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_k32_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_k32_cuda 256 128 128 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_k32_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_k32_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w4x1_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w4x1_cuda 256 128 128 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w4x1_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w4x1_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_cuda 256 128 128 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_k32_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_k32_cuda 256 128 128 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_k32_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_k32_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128x64_s3_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128x64_s3_cuda 256 128 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled128x64_s3_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128x64_s3_cuda 144 80 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled64x128_s3_verify_aligned
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled64x128_s3_cuda 128 256 64 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_tiled64x128_s3_verify_boundary
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled64x128_s3_cuda 80 144 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_cublas_tensor_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublas_tensor 80 96 33 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_cublaslt_tensor_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublaslt_tensor 80 96 33 --verify --warmup 1 --repeat 1
  )

  add_test(NAME gemm_naive_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_naive_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tiled_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tiled32_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled32_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tiled2x2_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled2x2_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tiled4x4_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tiled4x4_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_warp_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_warp_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_core_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_core 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_cublas_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublas 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_optimized_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_optimized_k8_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_k8_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_optimized_async_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_optimized_async_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled128_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled128_k32_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_k32_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled128_s3_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128_s3_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128_s3_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_s3_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128_k32_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128_k32_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128x64_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128x64_k32_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_k32_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128x64_w4x1_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w4x1_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tma128x64_w2x2_k32_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tma128x64_w2x2_k32_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled128x64_s3_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled128x64_s3_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_tiled64x128_s3_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_tiled64x128_s3_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_cublas_tensor_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublas_tensor 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_cublaslt_tensor_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_cublaslt_tensor 512 512 512 --no-verify --warmup 3 --repeat 10
  )
endif()

add_custom_target(test_gemm_verbose
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  DEPENDS all_examples
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Run GEMM tests with benchmark output visible"
)
