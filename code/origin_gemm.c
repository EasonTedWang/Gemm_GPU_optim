#include <chrono>
#include <exception>
#include <iostream>
#include <vector>

#include "matrix_utils.h"

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
    print_run_config("CPU reference GEMM", config);

    std::vector<float> A(static_cast<std::size_t>(p.M) * p.K);
    std::vector<float> B(static_cast<std::size_t>(p.K) * p.N);
    std::vector<float> C(static_cast<std::size_t>(p.M) * p.N, 0.0f);

    random_matrix_seeded(A.data(), p.M, p.K, config.seed);
    random_matrix_seeded(B.data(), p.K, p.N, config.seed + 1);

    if (config.benchmark) {
        for (int i = 0; i < config.warmup; ++i) {
            cpu_gemm_reference(A.data(), B.data(), C.data(), p.M, p.N, p.K);
        }

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < config.repeat; ++i) {
            cpu_gemm_reference(A.data(), B.data(), C.data(), p.M, p.N, p.K);
        }
        auto stop = std::chrono::steady_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(stop - start).count();
        print_benchmark_result("CPU", p, totalMs / static_cast<double>(config.repeat));
    } else {
        cpu_gemm_reference(A.data(), B.data(), C.data(), p.M, p.N, p.K);
    }

    if (config.verify) {
        std::cout << "Verification: PASSED | CPU reference implementation is the oracle\n";
    }

    std::cout << "Result sample: C[0]=" << C.front()
              << " C[last]=" << C.back() << "\n";
    return 0;
}