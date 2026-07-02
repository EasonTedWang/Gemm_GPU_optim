#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "../common/matrix_utils.h"

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

static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#if USE_WMMA
// Tensor Core kernel for row-major A, B, C matrices.
// Each CUDA warp computes one 16x16 output tile (WMMA_M x WMMA_N).
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

    // Loop over K dimension in 16-element tiles.
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

static bool nearly_equal(float a, float b, float tol = 1e-2f)
{
    float diff = a - b;
    return fabs(diff) <= tol * fmaxf(1.0f, fabs(b));
}

static std::vector<half> convert_to_half(const std::vector<float>& src)
{
    std::vector<half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

int main(int argc, char** argv)
{
    int M = 256;
    int N = 256;
    int K = 256;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verify") {
            verify = true;
            continue;
        }
        if (i + 2 < argc && argv[i][0] != '-') {
            M = std::atoi(argv[i]);
            N = std::atoi(argv[i + 1]);
            K = std::atoi(argv[i + 2]);
            break;
        }
    }

    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        std::cerr << "Tensor Core GEMM requires M,N,K to be multiples of "
                  << WMMA_M << ". Given: " << M << "x" << N << "x" << K << std::endl;
        return 1;
    }

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    random_matrix(A.data(), M, K);
    random_matrix(B.data(), K, N);

    std::vector<half> Ah = convert_to_half(A);
    std::vector<half> Bh = convert_to_half(B);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, sizeof(half) * M * K);
    cudaMalloc(&d_B, sizeof(half) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, Ah.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, Bh.data(), sizeof(half) * K * N, cudaMemcpyHostToDevice);

#if USE_WMMA
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + WMMA_N - 1) / WMMA_N,
              (M + WMMA_M * WARPS_PER_BLOCK - 1) / (WMMA_M * WARPS_PER_BLOCK));

    tensorcore_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
#else
    std::cerr << "CUDA Tensor Core support is unavailable on this GPU." << std::endl;
    return 1;
#endif

    cudaMemcpy(C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    std::cout << "CUDA Tensor Core GEMM done: M=" << M << " N=" << N << " K=" << K << std::endl;
    std::cout << "Result sample: C[0]=" << C[0] << " C[last]=" << C[M * N - 1] << std::endl;

    bool passed = true;
    if (verify) {
        std::vector<float> reference(M * N);
        cpu_gemm(A.data(), B.data(), reference.data(), M, N, K);
        for (int i = 0; i < M * N; ++i) {
            if (!nearly_equal(C[i], reference[i])) {
                passed = false;
                std::cerr << "Mismatch at " << i << ": got " << C[i]
                          << " expected " << reference[i] << std::endl;
                break;
            }
        }
        std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return verify ? (passed ? 0 : 1) : 0;
}
