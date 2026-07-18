#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-2048}"
OUTPUT="${2:-gemm_optimized_${SIZE}}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "Nsight Compute (ncu) was not found in PATH" >&2
  exit 1
fi

ncu --set full \
  --kernel-name regex:optimized_gemm_kernel \
  --launch-skip 1 \
  --launch-count 1 \
  --export "${OUTPUT}" \
  --force-overwrite \
  ./out/gemm_optimized_cuda "${SIZE}" "${SIZE}" "${SIZE}" \
    --no-verify --warmup 1 --repeat 1
