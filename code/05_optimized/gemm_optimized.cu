#include <cuda_runtime.h>

#ifndef GEMM_USE_CP_ASYNC
#define GEMM_USE_CP_ASYNC 0
#endif

#if GEMM_USE_CP_ASYNC
#include <cuda_pipeline_primitives.h>
#endif

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix_utils.h"

namespace {

constexpr int CTA_M = 128;
constexpr int CTA_N = 128;
#ifndef GEMM_CTA_K
#define GEMM_CTA_K 16
#endif
constexpr int CTA_K = GEMM_CTA_K;
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int THREAD_M = 8;
constexpr int THREAD_N = 8;
constexpr int THREADS = 256;
#if GEMM_USE_CP_ASYNC
constexpr int B_STRIDE = CTA_N;
#else
constexpr int B_SWIZZLE_GROUP = 8;
constexpr int B_STRIDE = CTA_N + CTA_N / B_SWIZZLE_GROUP;
#endif

static_assert((CTA_M / WARP_M) * (CTA_N / WARP_N) * 32 == THREADS);
static_assert(WARP_M % THREAD_M == 0 && WARP_N % THREAD_N == 0);
static_assert(CTA_K % 8 == 0, "CTA_K must be a multiple of eight");

__device__ __forceinline__ int swizzled_b_column(int column)
{
#if GEMM_USE_CP_ASYNC
    const int vector = column / 4;
    return (vector ^ (vector >> 1)) * 4 + column % 4;
#else
    return column + column / B_SWIZZLE_GROUP;
#endif
}

__device__ __forceinline__ int shared_a_index(int row, int column)
{
#if GEMM_USE_CP_ASYNC
    constexpr int vectorsPerRow = CTA_K / 4;
    const int vector = column / 4;
    const int rowSwizzle = (row / THREAD_M) & (vectorsPerRow - 1);
    return row * CTA_K + (vector ^ rowSwizzle) * 4 + column % 4;
#else
    return column * CTA_M + row;
#endif
}

template <bool GuardBounds>
__device__ __forceinline__ void load_tile(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* sharedA,
                                          float* sharedB,
                                          int blockRow,
                                          int blockCol,
                                          int tileK,
                                          int M,
                                          int N,
                                          int K,
                                          int threadId)
{
    // One 128-bit A load and one 128-bit B load per thread cover the complete
    // 128x8 and 8x128 input tiles. A is transposed while entering shared memory.
    const int aRow = threadId / 2;
    const int aCol = (threadId % 2) * 4;
    const int globalARow = blockRow + aRow;
    const int globalACol = tileK + aCol;

#if GEMM_USE_CP_ASYNC
    if constexpr (!GuardBounds) {
#pragma unroll
        for (int load = 0; load < CTA_K / 8; ++load) {
            const int column = aCol + load * 8;
            __pipeline_memcpy_async(sharedA + shared_a_index(aRow, column),
                                    A + globalARow * K + tileK + column,
                                    sizeof(float4));
        }

        const int bRow = threadId / (CTA_N / 4);
        const int bCol = (threadId % (CTA_N / 4)) * 4;
#pragma unroll
        for (int load = 0; load < CTA_K / 8; ++load) {
            const int row = bRow + load * 8;
            __pipeline_memcpy_async(sharedB + row * B_STRIDE + swizzled_b_column(bCol),
                                    B + (tileK + row) * N + blockCol + bCol,
                                    sizeof(float4));
        }
    }
#endif

#if GEMM_USE_CP_ASYNC
    if constexpr (GuardBounds) {
#endif
#pragma unroll
    for (int load = 0; load < CTA_K / 8; ++load) {
        const int loadACol = globalACol + load * 8;
        float aValues[4];
        if constexpr (!GuardBounds) {
            const float4 value =
                *reinterpret_cast<const float4*>(A + globalARow * K + loadACol);
            aValues[0] = value.x;
            aValues[1] = value.y;
            aValues[2] = value.z;
            aValues[3] = value.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int column = loadACol + i;
                aValues[i] =
                    (globalARow < M && column < K) ? A[globalARow * K + column] : 0.0f;
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sharedA[shared_a_index(aRow, aCol + load * 8 + i)] = aValues[i];
        }
    }

    const int bRow = threadId / (CTA_N / 4);
    const int bCol = (threadId % (CTA_N / 4)) * 4;
    const int globalBRow = tileK + bRow;
    const int globalBCol = blockCol + bCol;

#pragma unroll
    for (int load = 0; load < CTA_K / 8; ++load) {
        const int loadBRow = globalBRow + load * 8;
        float bValues[4];
        if constexpr (!GuardBounds) {
            const float4 value =
                *reinterpret_cast<const float4*>(B + loadBRow * N + globalBCol);
            bValues[0] = value.x;
            bValues[1] = value.y;
            bValues[2] = value.z;
            bValues[3] = value.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int column = globalBCol + i;
                bValues[i] =
                    (loadBRow < K && column < N) ? B[loadBRow * N + column] : 0.0f;
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sharedB[(bRow + load * 8) * B_STRIDE + swizzled_b_column(bCol + i)] =
                bValues[i];
        }
    }
#if GEMM_USE_CP_ASYNC
    }
#endif
}

template <bool GuardBounds>
__global__ __launch_bounds__(THREADS, 2)
void optimized_gemm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M,
                           int N,
                           int K)
{
    __shared__ float sharedA[2][CTA_K][CTA_M];
    __shared__ float sharedB[2][CTA_K][B_STRIDE];

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int laneId = threadId % 32;
    const int warpRow = warpId / (CTA_N / WARP_N);
    const int warpCol = warpId % (CTA_N / WARP_N);
    const int laneRow = laneId / (WARP_N / THREAD_N);
    const int laneCol = laneId % (WARP_N / THREAD_N);

    const int blockRow = blockIdx.y * CTA_M;
    const int blockCol = blockIdx.x * CTA_N;
    const int rowBase = blockRow + warpRow * WARP_M + laneRow * THREAD_M;
    const int colBase = blockCol + warpCol * WARP_N + laneCol * THREAD_N;

    float accum[THREAD_M][THREAD_N] = {};
    const int tileCount = (K + CTA_K - 1) / CTA_K;

    load_tile<GuardBounds>(A,
                           B,
                           &sharedA[0][0][0],
                           &sharedB[0][0][0],
                           blockRow,
                           blockCol,
                           0,
                           M,
                           N,
                           K,
                           threadId);
#if GEMM_USE_CP_ASYNC
    if constexpr (!GuardBounds) {
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
#endif
    __syncthreads();

    int readStage = 0;
    for (int tile = 0; tile < tileCount; ++tile) {
        const int writeStage = readStage ^ 1;
        if (tile + 1 < tileCount) {
            load_tile<GuardBounds>(A,
                                   B,
                                   &sharedA[writeStage][0][0],
                                   &sharedB[writeStage][0][0],
                                   blockRow,
                                   blockCol,
                                   (tile + 1) * CTA_K,
                                   M,
                                   N,
                                   K,
                                   threadId);
#if GEMM_USE_CP_ASYNC
            if constexpr (!GuardBounds) {
                __pipeline_commit();
            }
#endif
        }

#pragma unroll
        for (int k = 0; k < CTA_K; ++k) {
            float aFragment[THREAD_M];
            float bFragment[THREAD_N];

#pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                const int row = warpRow * WARP_M + laneRow * THREAD_M + i;
                aFragment[i] = (&sharedA[readStage][0][0])[shared_a_index(row, k)];
            }
#pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                const int column = warpCol * WARP_N + laneCol * THREAD_N + j;
                bFragment[j] = sharedB[readStage][k][swizzled_b_column(column)];
            }

#pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
#pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    accum[i][j] = fmaf(aFragment[i], bFragment[j], accum[i][j]);
                }
            }
        }

#if GEMM_USE_CP_ASYNC
        if constexpr (!GuardBounds) {
            if (tile + 1 < tileCount) {
                __pipeline_wait_prior(0);
            }
        }
#endif
        __syncthreads();
        readStage = writeStage;
    }

#pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int row = rowBase + i;
        if constexpr (!GuardBounds) {
            float* output = C + row * N + colBase;
            *reinterpret_cast<float4*>(output) =
                make_float4(accum[i][0], accum[i][1], accum[i][2], accum[i][3]);
            *reinterpret_cast<float4*>(output + 4) =
                make_float4(accum[i][4], accum[i][5], accum[i][6], accum[i][7]);
        } else if (row < M) {
#pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                const int column = colBase + j;
                if (column < N) {
                    C[row * N + column] = accum[i][j];
                }
            }
        }
    }
}

bool use_aligned_fast_path(const GemmProblem& p)
{
    return p.M % CTA_M == 0 && p.N % CTA_N == 0 && p.K % CTA_K == 0;
}

void launch_optimized_gemm(const float* d_A,
                           const float* d_B,
                           float* d_C,
                           const GemmProblem& p,
                           bool fastPath)
{
    const dim3 block(THREADS);
    const dim3 grid((p.N + CTA_N - 1) / CTA_N, (p.M + CTA_M - 1) / CTA_M);
    if (fastPath) {
        optimized_gemm_kernel<false><<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
    } else {
        optimized_gemm_kernel<true><<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
    }
}

void print_kernel_resources(bool fastPath)
{
    cudaFuncAttributes attributes{};
    int blocksPerSm = 0;
    if (fastPath) {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, optimized_gemm_kernel<false>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, optimized_gemm_kernel<false>, THREADS, 0));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, optimized_gemm_kernel<true>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, optimized_gemm_kernel<true>, THREADS, 0));
    }

    std::cout << "Kernel path: " << (fastPath ? "aligned-vectorized" : "boundary-safe")
#if GEMM_USE_CP_ASYNC
              << " | pipeline=cp.async"
#else
              << " | pipeline=software-prefetch"
#endif
              << " | registers/thread=" << attributes.numRegs
              << " | static_smem=" << attributes.sharedSizeBytes
              << " bytes | active_blocks/SM=" << blocksPerSm << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    RunConfig config;
    try {
        config = parse_run_config(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    if (config.showHelp) {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    const GemmProblem& p = config.problem;
    const bool fastPath = use_aligned_fast_path(p);
    print_run_config("CUDA optimized multilevel GEMM (CTA_K=" + std::to_string(CTA_K) +
#if GEMM_USE_CP_ASYNC
                         ", cp.async)",
#else
                         ", software-prefetch)",
#endif
                     config);

    std::vector<float> A(static_cast<std::size_t>(p.M) * p.K);
    std::vector<float> B(static_cast<std::size_t>(p.K) * p.N);
    std::vector<float> C(static_cast<std::size_t>(p.M) * p.N, 0.0f);
    random_matrix_seeded(A.data(), p.M, p.K, config.seed);
    random_matrix_seeded(B.data(), p.K, p.N, config.seed + 1);

    const std::size_t bytesA = sizeof(float) * A.size();
    const std::size_t bytesB = sizeof(float) * B.size();
    const std::size_t bytesC = sizeof(float) * C.size();

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice));

    print_kernel_resources(fastPath);
    for (int i = 0; i < config.warmup; ++i) {
        launch_optimized_gemm(d_A, d_B, d_C, p, fastPath);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_optimized_gemm(d_A, d_B, d_C, p, fastPath);
        }
        CUDA_CHECK(cudaGetLastError());
        const float totalMs = timer.stop_ms();
        print_benchmark_result(
            "Kernel", p, totalMs / static_cast<double>(config.repeat), &default_gpu_profile());
    } else {
        launch_optimized_gemm(d_A, d_B, d_C, p, fastPath);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    bool passed = true;
    if (config.verify) {
        std::vector<float> reference(C.size());
        cpu_gemm_reference(A.data(), B.data(), reference.data(), p.M, p.N, p.K);
        const VerificationResult result =
            compare_matrices(C.data(), reference.data(), static_cast<int>(C.size()));
        print_verification_result(result);
        passed = result.passed;
    }

    std::cout << "Result sample: C[0]=" << C.front() << " C[last]=" << C.back() << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? 0 : 1;
}
