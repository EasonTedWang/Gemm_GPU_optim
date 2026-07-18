#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 M N K [options]" >&2
  exit 1
fi

M="$1"
N="$2"
K="$3"
shift 3

TILES_M=$(((M + 127) / 128))
TILES_N=$(((N + 127) / 128))
OUTPUT_TILES=$((TILES_M * TILES_N))
RECT_TILES_M=$(((M + 127) / 128))
RECT_TILES_N=$(((N + 63) / 64))
RECT_OUTPUT_TILES=$((RECT_TILES_M * RECT_TILES_N))
RTX5080_SMS=84
IS_SKINNY=0
if ((M >= 2 * N || N >= 2 * M)); then
  IS_SKINNY=1
fi

TMA_128_ALIGNED=0
TMA_128X64_ALIGNED=0
TMA_128X64_K32_ALIGNED=0
if ((M % 128 == 0 && N % 128 == 0 && K % 16 == 0)); then
  TMA_128_ALIGNED=1
fi
if ((M % 128 == 0 && N % 64 == 0 && K % 16 == 0)); then
  TMA_128X64_ALIGNED=1
fi
if ((M % 128 == 0 && N % 64 == 0 && K % 32 == 0)); then
  TMA_128X64_K32_ALIGNED=1
fi

if ((TMA_128X64_K32_ALIGNED == 1 && RECT_OUTPUT_TILES <= RTX5080_SMS)); then
  KERNEL="./out/gemm_tensor_tma128x64_w2x2_k32_cuda"
elif ((TMA_128X64_ALIGNED == 1 && M >= 32 * N)); then
  KERNEL="./out/gemm_tensor_tma128x64_cuda"
elif ((TMA_128X64_ALIGNED == 1)); then
  KERNEL="./out/gemm_tensor_tma128x64_w2x2_cuda"
elif ((TMA_128_ALIGNED == 1)); then
  KERNEL="./out/gemm_tensor_tma128_cuda"
elif ((OUTPUT_TILES >= 512)); then
  KERNEL="./out/gemm_tensor_tiled128_cuda"
elif ((IS_SKINNY == 1 && OUTPUT_TILES >= 64)); then
  KERNEL="./out/gemm_tensor_tiled128x64_s3_cuda"
elif ((OUTPUT_TILES >= 128)); then
  KERNEL="./out/gemm_tensor_tiled128_s3_cuda"
elif ((OUTPUT_TILES >= 64)); then
  KERNEL="./out/gemm_tensor_tiled128x64_s3_cuda"
else
  KERNEL="./out/gemm_tensor_tiled_cuda"
fi

echo "Dispatch: ${KERNEL} | tiles128=${OUTPUT_TILES} | tiles128x64=${RECT_OUTPUT_TILES} | TMA128=${TMA_128_ALIGNED} | TMA128x64=${TMA_128X64_ALIGNED}"
exec "${KERNEL}" "${M}" "${N}" "${K}" "$@"
