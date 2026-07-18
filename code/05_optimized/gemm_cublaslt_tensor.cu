#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.cuh"
#include "matrix_utils.h"
#include "tensor_utils.cuh"

namespace {

constexpr std::size_t MAX_WORKSPACE_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr int HEURISTIC_CANDIDATES = 8;

struct LtMatmulPlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr;
    cublasLtMatrixLayout_t layoutB = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    void* workspace = nullptr;
    std::size_t workspaceBytes = 0;
    float wavesCount = 0.0f;
};

void create_plan(LtMatmulPlan& plan, const GemmProblem& problem)
{
    CUBLAS_CHECK(cublasLtCreate(&plan.handle));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.layoutA, CUDA_R_16F, problem.M, problem.K, problem.K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.layoutB, CUDA_R_16F, problem.K, problem.N, problem.N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &plan.layoutC, CUDA_R_32F, problem.M, problem.N, problem.N));

    const cublasLtOrder_t rowMajor = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&plan.preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(plan.preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &MAX_WORKSPACE_BYTES,
                                                      sizeof(MAX_WORKSPACE_BYTES)));

    std::array<cublasLtMatmulHeuristicResult_t, HEURISTIC_CANDIDATES> results{};
    int resultCount = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(plan.handle,
                                                plan.operation,
                                                plan.layoutA,
                                                plan.layoutB,
                                                plan.layoutC,
                                                plan.layoutC,
                                                plan.preference,
                                                HEURISTIC_CANDIDATES,
                                                results.data(),
                                                &resultCount));

    int selected = -1;
    for (int i = 0; i < resultCount; ++i) {
        if (results[i].state == CUBLAS_STATUS_SUCCESS &&
            results[i].workspaceSize <= MAX_WORKSPACE_BYTES) {
            selected = i;
            break;
        }
    }
    if (selected < 0) {
        std::cerr << "cuBLASLt did not return a usable Tensor Core algorithm\n";
        std::exit(EXIT_FAILURE);
    }

    plan.algorithm = results[selected].algo;
    plan.workspaceBytes = results[selected].workspaceSize;
    plan.wavesCount = results[selected].wavesCount;
    if (plan.workspaceBytes > 0) {
        CUDA_CHECK(cudaMalloc(&plan.workspace, plan.workspaceBytes));
    }
}

void destroy_plan(LtMatmulPlan& plan)
{
    if (plan.workspace != nullptr) {
        CUDA_CHECK(cudaFree(plan.workspace));
    }
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(plan.preference));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.layoutC));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.layoutB));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan.layoutA));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(plan.operation));
    CUBLAS_CHECK(cublasLtDestroy(plan.handle));
}

void launch_cublaslt(const LtMatmulPlan& plan,
                     const half* d_A,
                     const half* d_B,
                     float* d_C)
{
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(plan.handle,
                                plan.operation,
                                &alpha,
                                d_A,
                                plan.layoutA,
                                d_B,
                                plan.layoutB,
                                &beta,
                                d_C,
                                plan.layoutC,
                                d_C,
                                plan.layoutC,
                                &plan.algorithm,
                                plan.workspace,
                                plan.workspaceBytes,
                                nullptr));
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
    print_run_config("cuBLASLt FP16 Tensor Core GEMM", config);

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

    LtMatmulPlan plan;
    create_plan(plan, problem);
    std::cout << "cuBLASLt plan: workspace=" << plan.workspaceBytes
              << " bytes | waves=" << plan.wavesCount << "\n";

    for (int i = 0; i < config.warmup; ++i) {
        launch_cublaslt(plan, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_cublaslt(plan, d_A, d_B, d_C);
        }
        const float totalMs = timer.stop_ms();
        print_benchmark_result(
            "cuBLASLt Tensor", problem, totalMs / static_cast<double>(config.repeat));
    } else {
        launch_cublaslt(plan, d_A, d_B, d_C);
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

    destroy_plan(plan);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? 0 : 1;
}
