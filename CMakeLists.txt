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
  list(APPEND ALL_GEMM_TARGETS gemm_naive_cuda gemm_tiled_cuda gemm_tiled32_cuda gemm_tiled2x2_cuda gemm_warp_cuda gemm_tensor_core)
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
  add_test(NAME gemm_warp_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_warp_cuda 32 33 34 --verify --warmup 1 --repeat 1
  )
  add_test(NAME gemm_tensor_core_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_core 64 64 64 --verify --warmup 1 --repeat 1
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
  add_test(NAME gemm_warp_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_warp_cuda 512 512 512 --no-verify --warmup 3 --repeat 10
  )
  add_test(NAME gemm_tensor_core_perf_512
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_core 512 512 512 --no-verify --warmup 3 --repeat 10
  )
endif()

add_custom_target(test_gemm_verbose
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  DEPENDS all_examples
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Run GEMM tests with benchmark output visible"
)