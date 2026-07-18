#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.cuh"
#include "matrix_utils.h"
#include "tensor_utils.cuh"

namespace {

void launch_cublas_tensor(cublasHandle_t handle,
                          const half* d_A,
                          const half* d_B,
                          float* d_C,
                          const GemmProblem& problem)
{
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              problem.N,
                              problem.M,
                              problem.K,
                              &alpha,
                              d_B,
                              CUDA_R_16F,
                              problem.N,
                              d_A,
                              CUDA_R_16F,
                              problem.K,
                              &beta,
                              d_C,
                              CUDA_R_32F,
                              problem.N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    print_run_config("cuBLAS FP16 Tensor Core GEMM", config);

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

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    for (int i = 0; i < config.warmup; ++i) {
        launch_cublas_tensor(handle, d_A, d_B, d_C, problem);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_cublas_tensor(handle, d_A, d_B, d_C, problem);
        }
        const float totalMs = timer.stop_ms();
        print_benchmark_result(
            "cuBLAS Tensor", problem, totalMs / static_cast<double>(config.repeat));
    } else {
        launch_cublas_tensor(handle, d_A, d_B, d_C, problem);
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

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? 0 : 1;
}
