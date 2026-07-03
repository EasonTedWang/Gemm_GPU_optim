#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

struct GemmProblem {
    int M = 256;
    int N = 256;
    int K = 256;
};

struct RunConfig {
    GemmProblem problem;
    bool verify = false;
    bool benchmark = true;
    bool showHelp = false;
    int warmup = 3;
    int repeat = 10;
    unsigned int seed = 2026;
};

struct VerificationResult {
    bool passed = true;
    int firstMismatch = -1;
    int maxErrorIndex = -1;
    float maxAbsError = 0.0f;
    float maxRelError = 0.0f;
    float actualAtMaxError = 0.0f;
    float expectedAtMaxError = 0.0f;
};

void random_matrix(float* matrix, int rows, int cols);
void random_matrix_seeded(float* matrix, int rows, int cols, unsigned int seed);
void fill_matrix(float* matrix, int rows, int cols, float value);
void print_matrix(const float* matrix, int rows, int cols, int maxPrint = 8);
void cpu_gemm_reference(const float* A, const float* B, float* C, int M, int N, int K);

RunConfig parse_run_config(int argc, char** argv);
void print_usage(std::ostream& os, const char* programName);
void print_run_config(const std::string& name, const RunConfig& config);

VerificationResult compare_matrices(const float* actual,
                                    const float* expected,
                                    int count,
                                    float atol = 1e-3f,
                                    float rtol = 1e-3f);
void print_verification_result(const VerificationResult& result);

double gemm_gflops(const GemmProblem& problem, double avgMs);
void print_benchmark_result(const std::string& label,
                            const GemmProblem& problem,
                            double avgMs);