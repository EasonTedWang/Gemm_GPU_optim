#!/usr/bin/env bash
set -euo pipefail

# Build directory path
BUILD_DIR="build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. -DENABLE_CUDA=ON
cmake --build . --target all_examples

echo "Build complete. Executables are in ${PWD}/../out"
