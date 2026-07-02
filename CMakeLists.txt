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
  list(APPEND ALL_GEMM_TARGETS gemm_naive_cuda gemm_tiled_cuda gemm_warp_cuda gemm_tensor_core)
endif()

add_custom_target(all_examples
  DEPENDS ${ALL_GEMM_TARGETS}
  COMMENT "Build all GEMM example executables"
)

if(CUDAToolkit_FOUND)
  enable_testing()
  add_test(NAME gemm_tensor_core_verify
    COMMAND ${CMAKE_SOURCE_DIR}/out/gemm_tensor_core 256 256 256 --verify
  )
endif()
