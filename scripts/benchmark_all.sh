#!/usr/bin/env bash
set -euo pipefail

WARMUP="${GEMM_WARMUP:-10}"
REPEAT="${GEMM_REPEAT:-50}"

if [[ "$#" -eq 0 ]]; then
  SIZES=(512 1024 2048 4096)
else
  SIZES=("$@")
fi

KERNELS=(
  gemm_naive_cuda
  gemm_tiled_cuda
  gemm_tiled4x4_cuda
  gemm_warp_cuda
  gemm_optimized_k8_cuda
  gemm_optimized_cuda
  gemm_optimized_async_cuda
  gemm_cublas
  gemm_tensor_core
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

for size in "${SIZES[@]}"; do
  echo
  echo "===== M=N=K=${size} ====="
  for kernel in "${KERNELS[@]}"; do
    executable="./out/${kernel}"
    if [[ ! -x "${executable}" ]]; then
      echo "SKIP ${kernel}: executable not found"
      continue
    fi

    echo "--- ${kernel} ---"
    "${executable}" "${size}" "${size}" "${size}" \
      --no-verify --warmup "${WARMUP}" --repeat "${REPEAT}"
  done
done
