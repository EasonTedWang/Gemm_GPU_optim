#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix_utils.h"

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

static void launch_gemm_naive(const float* d_A, const float* d_B, float* d_C, const GemmProblem& p)
{
    dim3 block(16, 16);
    dim3 grid((p.N + block.x - 1) / block.x, (p.M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
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
    print_run_config("CUDA naive GEMM", config);

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
        launch_gemm_naive(d_A, d_B, d_C, p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_gemm_naive(d_A, d_B, d_C, p);
        }
        CUDA_CHECK(cudaGetLastError());
        float totalMs = timer.stop_ms();
        print_benchmark_result("Kernel", p, totalMs / static_cast<double>(config.repeat));
    } else {
        launch_gemm_naive(d_A, d_B, d_C, p);
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