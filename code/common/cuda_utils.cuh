#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t cudaCheckStatus = (call);                                       \
        if (cudaCheckStatus != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(cudaCheckStatus) << std::endl;          \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

class GpuTimer {
public:
    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start()
    {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    float stop_ms()
    {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};