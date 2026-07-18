#pragma once

#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

inline const char* cublas_status_string(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "license error";
    default:
        return "unknown error";
    }
}

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t cublasCheckStatus = (call);                                       \
        if (cublasCheckStatus != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << cublas_status_string(cublasCheckStatus) << std::endl;            \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)
