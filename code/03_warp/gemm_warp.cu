#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix_utils.h"

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;

__global__ void gemm_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float sharedA[TILE_M][TILE_K];
    __shared__ float sharedB[TILE_K][TILE_N];

    int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;
    int threadCount = blockDim.x * blockDim.y;
    int rowBase = blockIdx.y * TILE_M + threadIdx.y * THREAD_TILE_M;
    int colBase = blockIdx.x * TILE_N + threadIdx.x * THREAD_TILE_N;

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        for (int index = threadLinear; index < TILE_M * TILE_K; index += threadCount) {
            int tileRow = index / TILE_K;
            int tileCol = index % TILE_K;
            int globalRow = blockIdx.y * TILE_M + tileRow;
            int globalCol = t * TILE_K + tileCol;
            sharedA[tileRow][tileCol] =
                (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
        }

        for (int index = threadLinear; index < TILE_K * TILE_N; index += threadCount) {
            int tileRow = index / TILE_N;
            int tileCol = index % TILE_N;
            int globalRow = t * TILE_K + tileRow;
            int globalCol = blockIdx.x * TILE_N + tileCol;
            sharedB[tileRow][tileCol] =
                (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                float aVal = sharedA[threadIdx.y * THREAD_TILE_M + i][k];
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += aVal * sharedB[k][threadIdx.x * THREAD_TILE_N + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int outRow = rowBase + i;
            int outCol = colBase + j;
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = accum[i][j];
            }
        }
    }
}

static void launch_gemm_warp(const float* d_A, const float* d_B, float* d_C, const GemmProblem& p)
{
    dim3 block(TILE_N / THREAD_TILE_N, TILE_M / THREAD_TILE_M);
    dim3 grid((p.N + TILE_N - 1) / TILE_N, (p.M + TILE_M - 1) / TILE_M);
    gemm_warp_kernel<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
}

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
    print_run_config("CUDA register-tiled GEMM", config);

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

    for (int i = 0; i < config.warmup; ++i) {
        launch_gemm_warp(d_A, d_B, d_C, p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_gemm_warp(d_A, d_B, d_C, p);
        }
        CUDA_CHECK(cudaGetLastError());
        float totalMs = timer.stop_ms();
        print_benchmark_result("Kernel", p, totalMs / static_cast<double>(config.repeat), &default_gpu_profile());
    } else {
        launch_gemm_warp(d_A, d_B, d_C, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    bool passed = true;
    if (config.verify) {
        std::vector<float> reference(C.size());
        cpu_gemm_reference(A.data(), B.data(), reference.data(), p.M, p.N, p.K);
        VerificationResult result = compare_matrices(C.data(), reference.data(), static_cast<int>(C.size()));
        print_verification_result(result);
        passed = result.passed;
    }

    std::cout << "Result sample: C[0]=" << C.front()
              << " C[last]=" << C.back() << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? 0 : 1;
}