#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>

#ifndef GEMM_TENSOR_USE_TMA
#define GEMM_TENSOR_USE_TMA 0
#endif

#if GEMM_TENSOR_USE_TMA
#include <cuda.h>
#include <cuda/ptx>
#endif

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix_utils.h"
#include "tensor_utils.cuh"

using namespace nvcuda;

namespace {

#ifndef GEMM_TENSOR_CTA
#define GEMM_TENSOR_CTA 64
#endif
#ifndef GEMM_TENSOR_CTA_M
#define GEMM_TENSOR_CTA_M GEMM_TENSOR_CTA
#endif
#ifndef GEMM_TENSOR_CTA_N
#define GEMM_TENSOR_CTA_N GEMM_TENSOR_CTA
#endif
constexpr int CTA_M = GEMM_TENSOR_CTA_M;
constexpr int CTA_N = GEMM_TENSOR_CTA_N;
#ifndef GEMM_TENSOR_CTA_K
#define GEMM_TENSOR_CTA_K 16
#endif
constexpr int CTA_K = GEMM_TENSOR_CTA_K;
#ifndef GEMM_TENSOR_STAGES
#define GEMM_TENSOR_STAGES 2
#endif
constexpr int PIPELINE_STAGES = GEMM_TENSOR_STAGES;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
#ifndef GEMM_TENSOR_WARP_GRID_M
#define GEMM_TENSOR_WARP_GRID_M 4
#endif
#ifndef GEMM_TENSOR_WARP_GRID_N
#define GEMM_TENSOR_WARP_GRID_N 2
#endif
constexpr int WARP_GRID_M = GEMM_TENSOR_WARP_GRID_M;
constexpr int WARP_GRID_N = GEMM_TENSOR_WARP_GRID_N;
constexpr int WARPS = WARP_GRID_M * WARP_GRID_N;
constexpr int THREADS = WARPS * 32;
constexpr int WARP_TILE_M = CTA_M / WARP_GRID_M;
constexpr int WARP_TILE_N = CTA_N / WARP_GRID_N;
constexpr int WARP_OUTPUT_M = WARP_TILE_M / WMMA_M;
constexpr int WARP_OUTPUT_N = WARP_TILE_N / WMMA_N;

static_assert(CTA_M == 64 || CTA_M == 128);
static_assert(CTA_N == 64 || CTA_N == 128);
static_assert(CTA_K == 16 || CTA_K == 32);
static_assert(PIPELINE_STAGES == 2 || PIPELINE_STAGES == 3);
static_assert(WARPS == 4 || WARPS == 8);
static_assert(CTA_M % (WARP_GRID_M * WMMA_M) == 0);
static_assert(CTA_N % (WARP_GRID_N * WMMA_N) == 0);
static_assert(WARP_OUTPUT_M * WARP_OUTPUT_N <= 8);

#if GEMM_TENSOR_USE_TMA
void check_driver(CUresult result, const char* expression, const char* file, int line)
{
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::cerr << "CUDA driver error at " << file << ':' << line << " for " << expression
              << ": " << (name ? name : "unknown") << " ("
              << (message ? message : "no description") << ")\n";
    std::exit(EXIT_FAILURE);
}

#define CU_CHECK(call) check_driver((call), #call, __FILE__, __LINE__)

void encode_tma_2d(CUtensorMap* tensorMap,
                   void* globalAddress,
                   int innerDimension,
                   int outerDimension,
                   int boxInner,
                   int boxOuter)
{
    const cuuint64_t globalDimensions[2] = {
        static_cast<cuuint64_t>(innerDimension),
        static_cast<cuuint64_t>(outerDimension)};
    const cuuint64_t globalStrides[1] = {
        static_cast<cuuint64_t>(innerDimension) * sizeof(half)};
    const cuuint32_t boxDimensions[2] = {
        static_cast<cuuint32_t>(boxInner), static_cast<cuuint32_t>(boxOuter)};
    constexpr cuuint32_t elementStrides[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(tensorMap,
                                    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                    2,
                                    globalAddress,
                                    globalDimensions,
                                    globalStrides,
                                    boxDimensions,
                                    elementStrides,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_NONE,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

__device__ __forceinline__ void issue_tma_tile(const CUtensorMap* tensorMapA,
                                               const CUtensorMap* tensorMapB,
                                               half* sharedA,
                                               half* sharedB,
                                               cuda::std::uint64_t* barrier,
                                               int blockRow,
                                               int blockCol,
                                               int tileK)
{
    const cuda::std::int32_t coordinatesA[2] = {tileK, blockRow};
    const cuda::std::int32_t coordinatesB[2] = {blockCol, tileK};
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared,
                                    cuda::ptx::space_global,
                                    sharedA,
                                    tensorMapA,
                                    coordinatesA,
                                    barrier);
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared,
                                    cuda::ptx::space_global,
                                    sharedB,
                                    tensorMapB,
                                    coordinatesB,
                                    barrier);
    constexpr cuda::std::uint32_t transactionBytes =
        sizeof(half) * (CTA_M * CTA_K + CTA_K * CTA_N);
    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                         cuda::ptx::scope_cta,
                                         cuda::ptx::space_shared,
                                         barrier,
                                         transactionBytes);
}
#endif

template <bool GuardBounds>
__device__ __forceinline__ void load_tensor_tile(const half* __restrict__ A,
                                                 const half* __restrict__ B,
                                                 half* sharedA,
                                                 half* sharedB,
                                                 int blockRow,
                                                 int blockCol,
                                                 int tileK,
                                                 int M,
                                                 int N,
                                                 int K,
                                                 int threadId)
{
    if constexpr (!GuardBounds) {
        constexpr int vectorsA = CTA_M * CTA_K / 8;
        constexpr int vectorsB = CTA_K * CTA_N / 8;
        constexpr int totalVectors = vectorsA + vectorsB;
        for (int index = threadId; index < totalVectors; index += THREADS) {
            if (index < vectorsA) {
                const int row = index / (CTA_K / 8);
                const int column = (index % (CTA_K / 8)) * 8;
                __pipeline_memcpy_async(sharedA + row * CTA_K + column,
                                        A + (blockRow + row) * K + tileK + column,
                                        16);
            } else {
                const int vector = index - vectorsA;
                const int row = vector / (CTA_N / 8);
                const int column = (vector % (CTA_N / 8)) * 8;
                __pipeline_memcpy_async(sharedB + row * CTA_N + column,
                                        B + (tileK + row) * N + blockCol + column,
                                        16);
            }
        }
    } else {
        for (int index = threadId; index < CTA_M * CTA_K; index += THREADS) {
            const int row = index / CTA_K;
            const int column = index % CTA_K;
            const int globalRow = blockRow + row;
            const int globalColumn = tileK + column;
            sharedA[index] =
                (globalRow < M && globalColumn < K) ? A[globalRow * K + globalColumn]
                                                    : __float2half(0.0f);
        }

        for (int index = threadId; index < CTA_K * CTA_N; index += THREADS) {
            const int row = index / CTA_N;
            const int column = index % CTA_N;
            const int globalRow = tileK + row;
            const int globalColumn = blockCol + column;
            sharedB[index] =
                (globalRow < K && globalColumn < N) ? B[globalRow * N + globalColumn]
                                                    : __float2half(0.0f);
        }
    }
}

template <bool GuardBounds>
__global__ __launch_bounds__(THREADS, 2)
void tensor_tiled_kernel(
#if GEMM_TENSOR_USE_TMA
                         const __grid_constant__ CUtensorMap tensorMapA,
                         const __grid_constant__ CUtensorMap tensorMapB,
#endif
                         const half* __restrict__ A,
                         const half* __restrict__ B,
                         float* __restrict__ C,
                         int M,
                         int N,
                         int K)
{
    __shared__ __align__(128) half sharedA[PIPELINE_STAGES][CTA_M][CTA_K];
    __shared__ __align__(128) half sharedB[PIPELINE_STAGES][CTA_K][CTA_N];
#if GEMM_TENSOR_USE_TMA
    __shared__ cuda::std::uint64_t tmaBarriers[PIPELINE_STAGES];
#endif

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int warpRow = warpId / WARP_GRID_N;
    const int warpColumn = warpId % WARP_GRID_N;
    const int blockRow = blockIdx.y * CTA_M;
    const int blockCol = blockIdx.x * CTA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        accum[WARP_OUTPUT_M][WARP_OUTPUT_N];
#pragma unroll
    for (int outputRow = 0; outputRow < WARP_OUTPUT_M; ++outputRow) {
#pragma unroll
        for (int outputColumn = 0; outputColumn < WARP_OUTPUT_N; ++outputColumn) {
            wmma::fill_fragment(accum[outputRow][outputColumn], 0.0f);
        }
    }

    const int tileCount = (K + CTA_K - 1) / CTA_K;
#if GEMM_TENSOR_USE_TMA
    if constexpr (!GuardBounds) {
        if (threadId == 0) {
#pragma unroll
            for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
                cuda::ptx::mbarrier_init(&tmaBarriers[stage], 1);
            }
            issue_tma_tile(&tensorMapA,
                           &tensorMapB,
                           &sharedA[0][0][0],
                           &sharedB[0][0][0],
                           &tmaBarriers[0],
                           blockRow,
                           blockCol,
                           0);
            if constexpr (PIPELINE_STAGES == 3) {
                if (tileCount > 1) {
                    issue_tma_tile(&tensorMapA,
                                   &tensorMapB,
                                   &sharedA[1][0][0],
                                   &sharedB[1][0][0],
                                   &tmaBarriers[1],
                                   blockRow,
                                   blockCol,
                                   CTA_K);
                }
            }
        }
    } else
#endif
    {
        load_tensor_tile<GuardBounds>(A,
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
        if constexpr (!GuardBounds) {
            __pipeline_commit();
            if constexpr (PIPELINE_STAGES == 3) {
                if (tileCount > 1) {
                    load_tensor_tile<GuardBounds>(A,
                                                  B,
                                                  &sharedA[1][0][0],
                                                  &sharedB[1][0][0],
                                                  blockRow,
                                                  blockCol,
                                                  CTA_K,
                                                  M,
                                                  N,
                                                  K,
                                                  threadId);
                    __pipeline_commit();
                    __pipeline_wait_prior(1);
                } else {
                    __pipeline_wait_prior(0);
                }
            } else {
                __pipeline_wait_prior(0);
            }
        }
    }
    __syncthreads();

    int readStage = 0;
    for (int tile = 0; tile < tileCount; ++tile) {
        int writeTile = tile + 1;
        if constexpr (!GuardBounds && PIPELINE_STAGES == 3) {
            writeTile = tile + 2;
        }
        const int writeStage = writeTile % PIPELINE_STAGES;
        if (writeTile < tileCount) {
#if GEMM_TENSOR_USE_TMA
            if constexpr (!GuardBounds) {
                if (threadId == 0) {
                    issue_tma_tile(&tensorMapA,
                                   &tensorMapB,
                                   &sharedA[writeStage][0][0],
                                   &sharedB[writeStage][0][0],
                                   &tmaBarriers[writeStage],
                                   blockRow,
                                   blockCol,
                                   writeTile * CTA_K);
                }
            } else
#endif
            {
                load_tensor_tile<GuardBounds>(A,
                                              B,
                                              &sharedA[writeStage][0][0],
                                              &sharedB[writeStage][0][0],
                                              blockRow,
                                              blockCol,
                                              writeTile * CTA_K,
                                              M,
                                              N,
                                              K,
                                              threadId);
                if constexpr (!GuardBounds) {
                    __pipeline_commit();
                }
            }
        }

#if GEMM_TENSOR_USE_TMA
        if constexpr (!GuardBounds) {
            const cuda::std::uint32_t phase =
                static_cast<cuda::std::uint32_t>((tile / PIPELINE_STAGES) & 1);
            while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                                                        cuda::ptx::scope_cta,
                                                        &tmaBarriers[readStage],
                                                        phase)) {
            }
        }
#endif

#pragma unroll
        for (int kOffset = 0; kOffset < CTA_K; kOffset += WMMA_K) {
            wmma::fragment<wmma::matrix_a,
                           WMMA_M,
                           WMMA_N,
                           WMMA_K,
                           half,
                           wmma::row_major>
                aFragments[WARP_OUTPUT_M];
#pragma unroll
            for (int outputRow = 0; outputRow < WARP_OUTPUT_M; ++outputRow) {
                const int tileRow = warpRow * WARP_TILE_M + outputRow * WMMA_M;
                wmma::load_matrix_sync(aFragments[outputRow],
                                       &sharedA[readStage][tileRow][kOffset],
                                       CTA_K);
            }

            wmma::fragment<wmma::matrix_b,
                           WMMA_M,
                           WMMA_N,
                           WMMA_K,
                           half,
                           wmma::row_major>
                bFragments[WARP_OUTPUT_N];
#pragma unroll
            for (int outputColumn = 0; outputColumn < WARP_OUTPUT_N; ++outputColumn) {
                const int tileColumn = warpColumn * WARP_TILE_N + outputColumn * WMMA_N;
                wmma::load_matrix_sync(bFragments[outputColumn],
                                       &sharedB[readStage][kOffset][tileColumn],
                                       CTA_N);
            }

#pragma unroll
            for (int outputRow = 0; outputRow < WARP_OUTPUT_M; ++outputRow) {
#pragma unroll
                for (int outputColumn = 0; outputColumn < WARP_OUTPUT_N; ++outputColumn) {
                    wmma::mma_sync(accum[outputRow][outputColumn],
                                   aFragments[outputRow],
                                   bFragments[outputColumn],
                                   accum[outputRow][outputColumn]);
                }
            }
        }

#if !GEMM_TENSOR_USE_TMA
        if constexpr (!GuardBounds) {
            if constexpr (PIPELINE_STAGES == 3) {
                if (tile + 1 < tileCount) {
                    __pipeline_wait_prior(tile + 2 < tileCount ? 1 : 0);
                }
            } else {
                if (tile + 1 < tileCount) {
                    __pipeline_wait_prior(0);
                }
            }
        }
#endif
        __syncthreads();
        readStage = (readStage + 1) % PIPELINE_STAGES;
    }

#pragma unroll
    for (int fragmentRow = 0; fragmentRow < WARP_OUTPUT_M; ++fragmentRow) {
        const int outputRow =
            blockRow + warpRow * WARP_TILE_M + fragmentRow * WMMA_M;
#pragma unroll
        for (int fragmentColumn = 0; fragmentColumn < WARP_OUTPUT_N; ++fragmentColumn) {
            const int outputColumn =
                blockCol + warpColumn * WARP_TILE_N + fragmentColumn * WMMA_N;
            if (outputRow < M && outputColumn < N) {
                wmma::store_matrix_sync(C + outputRow * N + outputColumn,
                                        accum[fragmentRow][fragmentColumn],
                                        N,
                                        wmma::mem_row_major);
            }
        }
    }
}

bool is_supported_shape(const GemmProblem& problem)
{
    return problem.M % WMMA_M == 0 && problem.N % WMMA_N == 0;
}

bool use_fast_path(const GemmProblem& problem)
{
    return problem.M % CTA_M == 0 && problem.N % CTA_N == 0 && problem.K % CTA_K == 0;
}

void launch_tensor_tiled(const half* d_A,
                         const half* d_B,
                         float* d_C,
                         const GemmProblem& problem,
                         bool fastPath
#if GEMM_TENSOR_USE_TMA
                         , const CUtensorMap& tensorMapA,
                         const CUtensorMap& tensorMapB
#endif
                         )
{
    const dim3 block(THREADS);
    const dim3 grid((problem.N + CTA_N - 1) / CTA_N,
                    (problem.M + CTA_M - 1) / CTA_M);
    if (fastPath) {
        tensor_tiled_kernel<false><<<grid, block>>>(
#if GEMM_TENSOR_USE_TMA
            tensorMapA, tensorMapB,
#endif
            d_A, d_B, d_C, problem.M, problem.N, problem.K);
    } else {
        tensor_tiled_kernel<true><<<grid, block>>>(
#if GEMM_TENSOR_USE_TMA
            tensorMapA, tensorMapB,
#endif
            d_A, d_B, d_C, problem.M, problem.N, problem.K);
    }
}

void print_kernel_resources(bool fastPath)
{
    cudaFuncAttributes attributes{};
    int blocksPerSm = 0;
    if (fastPath) {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, tensor_tiled_kernel<false>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, tensor_tiled_kernel<false>, THREADS, 0));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, tensor_tiled_kernel<true>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, tensor_tiled_kernel<true>, THREADS, 0));
    }

    std::cout << "Kernel path: "
              << (fastPath
#if GEMM_TENSOR_USE_TMA
                      ? "aligned-TMA"
#else
                      ? "aligned-cp.async"
#endif
                      : "boundary-safe")
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
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    if (config.showHelp) {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    const GemmProblem& problem = config.problem;
    if (!is_supported_shape(problem)) {
        std::cerr << "Tiled Tensor Core GEMM requires M and N to be multiples of 16.\n";
        return 1;
    }

    const bool fastPath = use_fast_path(problem);
    print_run_config("CUDA tiled Tensor Core GEMM (CTA=" + std::to_string(CTA_M) +
                         "x" + std::to_string(CTA_N) + "x" + std::to_string(CTA_K) +
                         ", stages=" + std::to_string(PIPELINE_STAGES) +
                         ", warp_grid=" + std::to_string(WARP_GRID_M) + "x" +
                         std::to_string(WARP_GRID_N) + ")",
                     config);

    std::vector<float> A(static_cast<std::size_t>(problem.M) * problem.K);
    std::vector<float> B(static_cast<std::size_t>(problem.K) * problem.N);
    std::vector<float> C(static_cast<std::size_t>(problem.M) * problem.N, 0.0f);
    random_matrix_seeded(A.data(), problem.M, problem.K, config.seed);
    random_matrix_seeded(B.data(), problem.K, problem.N, config.seed + 1);
    const std::vector<half> Ah = convert_to_half_vector(A);
    const std::vector<half> Bh = convert_to_half_vector(B);

    const std::size_t bytesA = sizeof(half) * Ah.size();
    const std::size_t bytesB = sizeof(half) * Bh.size();
    const std::size_t bytesC = sizeof(float) * C.size();

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, Ah.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, Bh.data(), bytesB, cudaMemcpyHostToDevice));

#if GEMM_TENSOR_USE_TMA
    alignas(64) CUtensorMap tensorMapA{};
    alignas(64) CUtensorMap tensorMapB{};
    if (fastPath) {
        encode_tma_2d(&tensorMapA, d_A, problem.K, problem.M, CTA_K, CTA_M);
        encode_tma_2d(&tensorMapB, d_B, problem.N, problem.K, CTA_N, CTA_K);
    }
#endif

    print_kernel_resources(fastPath);
    for (int i = 0; i < config.warmup; ++i) {
        launch_tensor_tiled(d_A, d_B, d_C, problem, fastPath
#if GEMM_TENSOR_USE_TMA
                            , tensorMapA, tensorMapB
#endif
                            );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_tensor_tiled(d_A, d_B, d_C, problem, fastPath
#if GEMM_TENSOR_USE_TMA
                                , tensorMapA, tensorMapB
#endif
                                );
        }
        CUDA_CHECK(cudaGetLastError());
        const float totalMs = timer.stop_ms();
        print_benchmark_result(
            "Tensor Core kernel", problem, totalMs / static_cast<double>(config.repeat));
    } else {
        launch_tensor_tiled(d_A, d_B, d_C, problem, fastPath
#if GEMM_TENSOR_USE_TMA
                            , tensorMapA, tensorMapB
#endif
                            );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    bool passed = true;
    if (config.verify) {
        const std::vector<float> Aref = convert_to_float_vector(Ah);
        const std::vector<float> Bref = convert_to_float_vector(Bh);
        std::vector<float> reference(C.size());
        cpu_gemm_reference(
            Aref.data(), Bref.data(), reference.data(), problem.M, problem.N, problem.K);
        const VerificationResult result = compare_matrices(
            C.data(), reference.data(), static_cast<int>(C.size()), 1e-2f, 1e-2f);
        print_verification_result(result);
        passed = result.passed;
    }

    std::cout << "Result sample: C[0]=" << C.front() << " C[last]=" << C.back() << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? 0 : 1;
}
