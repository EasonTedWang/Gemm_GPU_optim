#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "matrix_utils.h"

#if defined(__CUDACC__)
#define USE_WMMA 1
#include <mma.h>
using namespace nvcuda;
#else
#define USE_WMMA 0
#endif

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_PER_BLOCK = 4;

#if USE_WMMA
__global__ void tensorcore_gemm_kernel(const half* A, const half* B, float* C, int M, int N, int K)
{
    int warpRow = blockIdx.y * WARPS_PER_BLOCK + threadIdx.y;
    int warpCol = blockIdx.x;
    int row = warpRow * WMMA_M;
    int col = warpCol * WMMA_N;

    if (row >= M || col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int tileK = 0; tileK < K; tileK += WMMA_K) {
        const half* aTile = A + row * K + tileK;
        const half* bTile = B + tileK * N + col;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

        wmma::load_matrix_sync(aFrag, aTile, K);
        wmma::load_matrix_sync(bFrag, bTile, N);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    float* cTile = C + row * N + col;
    wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
}
#endif

static std::vector<half> convert_to_half(const std::vector<float>& src)
{
    std::vector<half> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

static std::vector<float> convert_to_float(const std::vector<half>& src)
{
    std::vector<float> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = __half2float(src[i]);
    }
    return dst;
}

static bool is_tensor_core_shape(const GemmProblem& p)
{
    return p.M % WMMA_M == 0 && p.N % WMMA_N == 0 && p.K % WMMA_K == 0;
}

static void launch_tensorcore_gemm(const half* d_A, const half* d_B, float* d_C, const GemmProblem& p)
{
#if USE_WMMA
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((p.N + WMMA_N - 1) / WMMA_N,
              (p.M + WMMA_M * WARPS_PER_BLOCK - 1) / (WMMA_M * WARPS_PER_BLOCK));
    tensorcore_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
#else
    (void)d_A;
    (void)d_B;
    (void)d_C;
    (void)p;
#endif
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
    if (!is_tensor_core_shape(p)) {
        std::cerr << "Tensor Core GEMM requires M, N, and K to be multiples of 16. "
                  << "Given: M=" << p.M << " N=" << p.N << " K=" << p.K << "\n";
        return 1;
    }

#if !USE_WMMA
    std::cerr << "CUDA Tensor Core support is unavailable in this build.\n";
    return 1;
#endif

    print_run_config("CUDA Tensor Core GEMM", config);

    std::vector<float> A(static_cast<std::size_t>(p.M) * p.K);
    std::vector<float> B(static_cast<std::size_t>(p.K) * p.N);
    std::vector<float> C(static_cast<std::size_t>(p.M) * p.N, 0.0f);

    random_matrix_seeded(A.data(), p.M, p.K, config.seed);
    random_matrix_seeded(B.data(), p.K, p.N, config.seed + 1);

    std::vector<half> Ah = convert_to_half(A);
    std::vector<half> Bh = convert_to_half(B);

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

    for (int i = 0; i < config.warmup; ++i) {
        launch_tensorcore_gemm(d_A, d_B, d_C, p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (config.benchmark) {
        GpuTimer timer;
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            launch_tensorcore_gemm(d_A, d_B, d_C, p);
        }
        CUDA_CHECK(cudaGetLastError());
        float totalMs = timer.stop_ms();
        print_benchmark_result("Kernel", p, totalMs / static_cast<double>(config.repeat));
    } else {
        launch_tensorcore_gemm(d_A, d_B, d_C, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    bool passed = true;
    if (config.verify) {
        std::vector<float> Aref = convert_to_float(Ah);
        std::vector<float> Bref = convert_to_float(Bh);
        std::vector<float> reference(C.size());
        cpu_gemm_reference(Aref.data(), Bref.data(), reference.data(), p.M, p.N, p.K);
        VerificationResult result = compare_matrices(C.data(), reference.data(), static_cast<int>(C.size()), 1e-2f, 1e-2f);
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