#include "matrix_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parse_int_value(const std::string& value, const std::string& name, int minValue)
{
    std::size_t parsedChars = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &parsedChars);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid " + name + ": " + value);
    }

    if (parsedChars != value.size() || parsed < minValue) {
        throw std::invalid_argument("Invalid " + name + ": " + value);
    }
    return parsed;
}

unsigned int parse_seed_value(const std::string& value)
{
    std::size_t parsedChars = 0;
    unsigned long parsed = 0;
    try {
        parsed = std::stoul(value, &parsedChars);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid seed: " + value);
    }

    if (parsedChars != value.size()) {
        throw std::invalid_argument("Invalid seed: " + value);
    }
    return static_cast<unsigned int>(parsed);
}

} // namespace

void random_matrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

void random_matrix_seeded(float* matrix, int rows, int cols, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(rng);
    }
}

void fill_matrix(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value;
    }
}

void print_matrix(const float* matrix, int rows, int cols, int maxPrint)
{
    int r = std::min(rows, maxPrint);
    int c = std::min(cols, maxPrint);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    if (rows > maxPrint || cols > maxPrint) {
        std::cout << "..." << std::endl;
    }
}

void cpu_gemm_reference(const float* A, const float* B, float* C, int M, int N, int K)
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

RunConfig parse_run_config(int argc, char** argv)
{
    RunConfig config;
    std::vector<int> dims;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            config.showHelp = true;
        } else if (arg == "--verify") {
            config.verify = true;
        } else if (arg == "--no-verify") {
            config.verify = false;
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--no-benchmark") {
            config.benchmark = false;
        } else if (arg == "--warmup") {
            if (++i >= argc) {
                throw std::invalid_argument("Missing value after --warmup");
            }
            config.warmup = parse_int_value(argv[i], "warmup", 0);
        } else if (arg == "--repeat") {
            if (++i >= argc) {
                throw std::invalid_argument("Missing value after --repeat");
            }
            config.repeat = parse_int_value(argv[i], "repeat", 1);
        } else if (arg == "--seed") {
            if (++i >= argc) {
                throw std::invalid_argument("Missing value after --seed");
            }
            config.seed = parse_seed_value(argv[i]);
        } else if (!arg.empty() && arg.rfind("-", 0) == 0) {
            throw std::invalid_argument("Unknown option: " + arg);
        } else {
            dims.push_back(parse_int_value(arg, "dimension", 1));
        }
    }

    if (!dims.empty() && dims.size() != 3) {
        throw std::invalid_argument("Expected dimensions as exactly: M N K");
    }
    if (dims.size() == 3) {
        config.problem.M = dims[0];
        config.problem.N = dims[1];
        config.problem.K = dims[2];
    }

    return config;
}

void print_usage(std::ostream& os, const char* programName)
{
    os << "Usage: " << programName << " [M N K] [options]\n"
       << "\nOptions:\n"
       << "  --verify          Compare kernel output against the CPU reference\n"
       << "  --no-benchmark    Run once without timing output\n"
       << "  --warmup N        Warmup iterations before timing (default: 3)\n"
       << "  --repeat N        Timed iterations for average kernel time (default: 10)\n"
       << "  --seed N          Deterministic random seed (default: 2026)\n"
       << "  -h, --help        Show this help\n";
}

void print_run_config(const std::string& name, const RunConfig& config)
{
    const GemmProblem& p = config.problem;
    std::cout << "== " << name << " ==\n"
              << "Problem: M=" << p.M << " N=" << p.N << " K=" << p.K << "\n"
              << "Verify: " << (config.verify ? "on" : "off")
              << " | Benchmark: " << (config.benchmark ? "on" : "off")
              << " | Warmup: " << config.warmup
              << " | Repeat: " << config.repeat
              << " | Seed: " << config.seed << "\n";
}

VerificationResult compare_matrices(const float* actual,
                                    const float* expected,
                                    int count,
                                    float atol,
                                    float rtol)
{
    VerificationResult result;
    for (int i = 0; i < count; ++i) {
        float absError = std::fabs(actual[i] - expected[i]);
        float relError = absError / std::max(1.0f, std::fabs(expected[i]));
        float allowedError = atol + rtol * std::fabs(expected[i]);

        if (absError > result.maxAbsError) {
            result.maxAbsError = absError;
            result.maxRelError = relError;
            result.maxErrorIndex = i;
            result.actualAtMaxError = actual[i];
            result.expectedAtMaxError = expected[i];
        }

        if (result.passed && absError > allowedError) {
            result.passed = false;
            result.firstMismatch = i;
        }
    }
    return result;
}

void print_verification_result(const VerificationResult& result)
{
    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    std::cout << std::scientific << std::setprecision(3)
              << "Verification: " << (result.passed ? "PASSED" : "FAILED")
              << " | max_abs_error=" << result.maxAbsError
              << " | max_rel_error=" << result.maxRelError;
    if (!result.passed) {
        std::cout << " | first_mismatch=" << result.firstMismatch
                  << " | max_error_index=" << result.maxErrorIndex
                  << " | actual=" << result.actualAtMaxError
                  << " | expected=" << result.expectedAtMaxError;
    }

    std::cout.copyfmt(oldState);
    std::cout << "\n";
}

double gemm_gflops(const GemmProblem& problem, double avgMs)
{
    if (avgMs <= 0.0) {
        return 0.0;
    }
    double flop = 2.0 * static_cast<double>(problem.M) *
                  static_cast<double>(problem.N) *
                  static_cast<double>(problem.K);
    return flop / (avgMs * 1.0e6);
}

void print_benchmark_result(const std::string& label,
                            const GemmProblem& problem,
                            double avgMs)
{
    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    std::cout << std::fixed << std::setprecision(4)
              << label << ": avg_ms=" << avgMs
              << " | gflops=" << std::setprecision(2)
              << gemm_gflops(problem, avgMs);

    std::cout.copyfmt(oldState);
    std::cout << "\n";
}