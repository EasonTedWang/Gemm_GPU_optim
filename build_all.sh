#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${GEMM_BUILD_DIR:-build}"
CUDA_ARCHITECTURES="${GEMM_CUDA_ARCHITECTURES:-120}"

cmake -S . -B "${BUILD_DIR}" \
  -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMM_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
cmake --build "${BUILD_DIR}" --parallel --target all_examples

echo "Build complete. Executables are in $(pwd)/out"
