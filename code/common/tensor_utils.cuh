#pragma once

#include <cuda_fp16.h>

#include <vector>

inline std::vector<half> convert_to_half_vector(const std::vector<float>& source)
{
    std::vector<half> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        result[i] = __float2half(source[i]);
    }
    return result;
}

inline std::vector<float> convert_to_float_vector(const std::vector<half>& source)
{
    std::vector<float> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        result[i] = __half2float(source[i]);
    }
    return result;
}
