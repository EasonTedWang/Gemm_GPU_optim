#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../common/matrix_utils.h"

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;

__global__ void gemm_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float sharedA[TILE_M][TILE_K];
    __shared__ float sharedB[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y * 8;
    int col = blockIdx.x * TILE_N + threadIdx.x * 8;

    float accum[8][8] = {};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int aRow = blockIdx.y * TILE_M + threadIdx.y * 8;
        int aCol = t * TILE_K + threadIdx.x;
        int bRow = t * TILE_K + threadIdx.y;
        int bCol = blockIdx.x * TILE_N + threadIdx.x * 8;

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < TILE_K; ++j) {
                int rowIndex = aRow + i;
                int colIndex = aCol + j;
                sharedA[i][j] = (rowIndex < M && colIndex < K) ? A[rowIndex * K + colIndex] : 0.0f;
            }
        }

        for (int i = 0; i < TILE_K; ++i) {
            for (int j = 0; j < 8; ++j) {
                int rowIndex = bRow + i;
                int colIndex = bCol + j;
                sharedB[i][j] = (rowIndex < K && colIndex < N) ? B[rowIndex * N + colIndex] : 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            for (int i = 0; i < 8; ++i) {
                float aVal = sharedA[i][k];
                for (int j = 0; j < 8; ++j) {
                    accum[i][j] += aVal * sharedB[k][j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int outRow = row + i;
            int outCol = col + j;
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = accum[i][j];
            }
        }
    }
}

int main(int argc, char** argv)
{
    int M = 256;
    int N = 256;
    int K = 256;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    random_matrix(A.data(), M, K);
    random_matrix(B.data(), K, N);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(8, 8);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_warp_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    std::cout << "CUDA warp GEMM done: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Result sample: C[0]=" << C[0] << " C[last]=" << C[M * N - 1] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
