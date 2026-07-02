#include <iostream>
#include <cstdlib>
#include <vector>
#include "common/matrix_utils.h"

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

    std::cout << "Running CPU GEMM with M=" << M << " N=" << N << " K=" << K << "\n";
    cpu_gemm(A.data(), B.data(), C.data(), M, N, K);

    std::cout << "Result sample: C[0]=" << C[0] << " C[last]=" << C[M * N - 1] << "\n";
    return 0;
}



